"""Shared Irodori generation jobs: long-text chunking and resumable batches.

Both the legacy (v2/v3) and the v4 screens go through these helpers so that
every Irodori batch gets the same guarantees:

* long lines are split at sentence boundaries instead of being silently cut at
  the model's 30 s limit;
* the text list and ``batch_manifest.json`` are rewritten after every line, so
  a crash or Stop keeps everything generated so far usable;
* a failed line is retried and, by default, skipped instead of aborting the
  whole batch;
* a re-run resumes where the previous run stopped (same texts + settings), or
  regenerates only the lines the QC pass rejected;
* the seed actually used for every line is recorded for reproduction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import soundfile as sf

from modules.dataset_io import atomic_write_json, read_json, sha256_file, write_text_list

logger = logging.getLogger(__name__)

MANIFEST_NAME = "batch_manifest.json"
# ~6-8 Japanese characters per second → 120 chars stays well under 30 s.
DEFAULT_MAX_CHARS = 120
DEFAULT_GAP_SEC = 0.25
MODEL_MAX_SECONDS = 30.0

RELEASE_IMMEDIATE = "immediate"
RELEASE_IDLE = "idle"
RELEASE_KEEP = "keep"

_SENTENCE_END_RE = re.compile(r"(?<=[。．！？!?♪…\n])")
_CLAUSE_END_RE = re.compile(r"(?<=[、，,])")


# ---------------------------------------------------------------- chunking

def _split_hard(piece: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    for clause in [c for c in _CLAUSE_END_RE.split(piece) if c]:
        while len(clause) > max_chars:
            parts.append(clause[:max_chars])
            clause = clause[max_chars:]
        if clause:
            parts.append(clause)
    return parts


def split_for_tts(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` at natural breaks.

    Text that already fits is returned unchanged as a single chunk so short
    lines keep exactly the prosody they had before chunking existed.
    """
    text = (text or "").strip()
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for sentence in _SENTENCE_END_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        pieces.extend(_split_hard(sentence, max_chars) if len(sentence) > max_chars else [sentence])
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current += piece
    if current:
        chunks.append(current)
    return chunks


def _concat_wavs(paths: Sequence[Path], out_path: Path, gap_sec: float) -> float:
    pieces = []
    sample_rate = None
    for path in paths:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(f"chunk sample rates differ: {sr} != {sample_rate}")
        if pieces:
            pieces.append(np.zeros(int(sample_rate * gap_sec), dtype=np.float32))
        pieces.append(audio)
    merged = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), merged, int(sample_rate or 48000), subtype="PCM_16")
    return merged.size / float(sample_rate or 48000)


def synthesize_to_file(
    bridge,
    *,
    text: str,
    out_path: str | Path,
    refs: Sequence[str | Path] = (),
    seed: int | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    gap_sec: float = DEFAULT_GAP_SEC,
    **request: Any,
) -> dict[str, Any]:
    """Synthesize ``text`` into ``out_path``, chunking long text.

    ``request`` carries the remaining :meth:`IrodoriBridge.synthesize` fields
    (``profile``, ``mode``, ``caption``, ``lora_path``...).  With several
    chunks every chunk reuses the first chunk's seed; for reference-free v4
    Voice Design the first chunk also becomes the speaker reference of the
    following chunks so the voice does not drift between them.
    """
    out_path = Path(out_path)
    if seed is None:
        # Pick the "random" seed here instead of letting the worker do it: its
        # 64-bit seeds lose precision in the browser (JS numbers are 53-bit),
        # which would make "pin this seed" silently reproduce another voice.
        seed = secrets.randbelow(2**31)
    profile = request.get("profile", "legacy")
    mode = request.get("mode", "clone")
    ref_list = [str(path) for path in refs if path]
    chunks = split_for_tts(text, max_chars if not (profile == "legacy" and mode == "design") else 0)
    if not chunks:
        raise ValueError("読み上げテキストが空です")

    def _call(chunk_text: str, chunk_out: Path, chunk_seed, chunk_refs: list[str], overrides=None):
        kwargs = dict(request)
        kwargs.update(overrides or {})
        if chunk_refs:
            if profile == "v4":
                kwargs["ref_wavs"] = chunk_refs
            else:
                kwargs["ref_wav"] = chunk_refs[0]
        return bridge.synthesize(text=chunk_text, out_path=chunk_out, seed=chunk_seed, **kwargs)

    if len(chunks) == 1:
        response = _call(chunks[0], out_path, seed, ref_list)
        duration = response.get("duration_sec")
        if duration is None:
            duration = sf.info(str(out_path)).duration
        max_seconds = response.get("max_seconds") or MODEL_MAX_SECONDS
        return {
            "used_seed": response.get("used_seed", seed),
            "duration_sec": round(float(duration), 3),
            "chunks": 1,
            "truncated_suspect": float(duration) >= float(max_seconds) - 0.25,
            "sample_rate": response.get("sample_rate"),
        }

    temp_dir = Path(tempfile.mkdtemp(prefix="vdc_chunks_"))
    try:
        chunk_paths: list[Path] = []
        chunk_seed = seed
        chunk_refs = ref_list
        overrides = None
        truncated = False
        for index, chunk_text in enumerate(chunks):
            chunk_path = temp_dir / f"{index:03d}.wav"
            response = _call(chunk_text, chunk_path, chunk_seed, chunk_refs, overrides)
            if index == 0:
                chunk_seed = response.get("used_seed", seed)
                if profile == "v4" and not ref_list:
                    chunk_refs = [str(chunk_path)]
                    overrides = {"mode": "clone", "no_ref": False}
            duration = float(response.get("duration_sec") or sf.info(str(chunk_path)).duration)
            max_seconds = float(response.get("max_seconds") or MODEL_MAX_SECONDS)
            truncated = truncated or duration >= max_seconds - 0.25
            chunk_paths.append(chunk_path)
        total = _concat_wavs(chunk_paths, out_path, gap_sec)
        return {
            "used_seed": chunk_seed,
            "duration_sec": round(total, 3),
            "chunks": len(chunks),
            "truncated_suspect": truncated,
            "sample_rate": sf.info(str(out_path)).samplerate,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def finish_release(bridge, release_mode: str) -> None:
    """Apply the user's post-generation VRAM policy."""
    if release_mode == RELEASE_IMMEDIATE:
        bridge.shutdown()
    elif release_mode == RELEASE_IDLE:
        bridge.schedule_idle_release()


# ---------------------------------------------------------------- batch

def settings_key(settings: dict[str, Any]) -> str:
    blob = json.dumps(settings, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def load_manifest(base_dir: str | Path) -> dict | None:
    return read_json(Path(base_dir) / MANIFEST_NAME)


def _file_fingerprint(path: str | Path) -> str:
    path = Path(path)
    try:
        return f"{path.name}:{path.stat().st_size}:{sha256_file(path)[:12]}"
    except OSError:
        return str(path)


def _line_seed(base_seed: int | None, index: int, redo_round: int) -> int | None:
    if base_seed is None:
        return None
    # A different, but still reproducible, seed for each QC redo round.
    return int(base_seed) + index - 1 + 7919 * redo_round


def run_irodori_batch(
    *,
    bridge,
    texts: Sequence[str],
    base_dir: str | Path,
    wavs_folder: str = "raw",
    esd_filename: str = "Neutral.txt",
    request: dict[str, Any],
    refs: Sequence[str | Path] = (),
    settings: dict[str, Any] | None = None,
    seed: int | None = None,
    resume: bool = True,
    redo_ids: set[str] | None = None,
    max_retries: int = 2,
    continue_on_error: bool = True,
    max_chars: int = DEFAULT_MAX_CHARS,
    release_mode: str = RELEASE_IMMEDIATE,
    run_signal_qc: bool = True,
    gpu_label: str = "Irodori一括生成",
) -> Iterator[tuple[float, object]]:
    """Generate ``texts`` into ``base_dir/wavs_folder``.

    Yields ``(pct, message)`` per line and finally ``(1.0, stats_dict)``.
    """
    from modules.gpu_gate import gpu_session, wait_messages

    base_dir = Path(base_dir)
    wav_dir = base_dir / wavs_folder
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base_dir / MANIFEST_NAME
    text_list_path = base_dir / esd_filename

    lines = [text.strip() for text in texts if text and text.strip()]
    if not lines:
        raise ValueError("No target texts were provided")

    settings = dict(settings or {})
    settings.setdefault("request", {k: v for k, v in request.items()})
    # Gradio re-uploads land in new temp paths, so identify references by
    # content rather than by path to keep resume working.
    settings["refs"] = [_file_fingerprint(path) for path in refs]
    key = settings_key(settings)
    line_request = {**request, "release_after_synthesis": False}

    previous = load_manifest(base_dir) if resume or redo_ids else None
    previous_records: dict[str, dict] = {}
    notice = ""
    if previous:
        if previous.get("settings_key") == key:
            previous_records = {r["id"]: r for r in previous.get("records", [])}
        else:
            notice = "前回と生成設定が異なるため、全行を生成し直します。"

    records: list[dict] = []
    todo: list[dict] = []
    redo_ids = set(redo_ids or ())
    for index, text in enumerate(lines, start=1):
        file_id = f"{index:04d}"
        old = previous_records.get(file_id)
        reusable = (
            old is not None
            and old.get("text") == text
            and old.get("status") == "done"
            and (wav_dir / f"{file_id}.wav").is_file()
        )
        if reusable and file_id not in redo_ids:
            records.append(dict(old))
            continue
        redo_round = int(old.get("redo_round", 0)) + 1 if (old and file_id in redo_ids) else 0
        record = {"id": file_id, "index": index, "text": text, "status": "pending",
                  "redo_round": redo_round}
        records.append(record)
        todo.append(record)

    def _save_progress(started_at: str, finished: bool = False) -> None:
        done = [r for r in records if r["status"] == "done"]
        write_text_list(text_list_path, [(r["id"], r["text"]) for r in done])
        atomic_write_json(manifest_path, {
            "version": 1,
            "settings": settings,
            "settings_key": key,
            "started_at": started_at,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "finished": finished,
            "records": records,
        })

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    skipped = len(records) - len(todo)
    if notice:
        yield 0.0, notice
    if skipped:
        yield skipped / len(records), f"再開: 生成済み{skipped}行をスキップし、残り{len(todo)}行を生成します"

    # The text list now reflects the manifest again, so any QC filter backup
    # describes an older state of this folder.
    stale_backup = base_dir / "Neutral.all.txt"
    if stale_backup.is_file() and esd_filename == "Neutral.txt":
        stale_backup.unlink()

    failures: list[dict] = []
    total_lines = len(records)
    if todo:
        for message in wait_messages(gpu_label):
            yield skipped / total_lines, message
        loop_started = time.time()
        with gpu_session(gpu_label):
            try:
                for position, record in enumerate(todo, start=1):
                    wav_path = wav_dir / f"{record['id']}.wav"
                    line_seed = _line_seed(seed, record["index"], record["redo_round"])
                    last_error = None
                    for attempt in range(1, max_retries + 2):
                        try:
                            info = synthesize_to_file(
                                bridge,
                                text=record["text"],
                                out_path=wav_path,
                                refs=refs,
                                seed=line_seed,
                                max_chars=max_chars,
                                **line_request,
                            )
                        except Exception as exc:  # retried below
                            last_error = exc
                            logger.warning("Line %s attempt %d failed: %s", record["id"], attempt, exc)
                            continue
                        record.update({
                            "status": "done",
                            "attempts": attempt,
                            "used_seed": info.get("used_seed"),
                            "duration_sec": info.get("duration_sec"),
                            "chunks": info.get("chunks"),
                            "truncated_suspect": info.get("truncated_suspect"),
                        })
                        record.pop("error", None)
                        break
                    else:
                        record.update({"status": "failed", "attempts": max_retries + 1,
                                       "error": str(last_error)})
                        failures.append(record)
                        _save_progress(started_at)
                        if not continue_on_error:
                            raise RuntimeError(
                                f"Voice clone failed at line {record['index']}: {last_error}"
                            ) from last_error
                        yield (skipped + position) / total_lines, (
                            f"{skipped + position}/{total_lines} | 行{record['index']}を{max_retries + 1}回失敗したためスキップ"
                        )
                        continue
                    _save_progress(started_at)
                    elapsed = time.time() - loop_started
                    remaining = elapsed / position * (len(todo) - position)
                    yield (skipped + position) / total_lines, (
                        f"{skipped + position}/{total_lines} | ~{remaining:.0f}s left"
                    )
            finally:
                _save_progress(started_at, finished=False)
                finish_release(bridge, release_mode)

    _save_progress(started_at, finished=True)

    qc_summary = None
    if run_signal_qc:
        try:
            from modules.audio_qc import run_qc

            for pct, payload in run_qc(base_dir, wavs_folder=wavs_folder):
                if isinstance(payload, dict):
                    qc_summary = payload
        except Exception:
            logger.exception("Signal QC after batch failed")

    done = [r for r in records if r["status"] == "done"]
    total_duration = sum(float(r.get("duration_sec") or 0.0) for r in done)
    yield 1.0, {
        "total_files": len(done),
        "total_duration_sec": total_duration,
        "output_dir": str(wav_dir),
        "esd_path": str(text_list_path),
        "manifest_path": str(manifest_path),
        "generated": len(todo) - len(failures),
        "skipped_existing": skipped,
        "failed": [{"index": r["index"], "text": r["text"], "error": r.get("error")} for r in failures],
        "chunked": sum(1 for r in done if (r.get("chunks") or 1) > 1),
        "qc": qc_summary,
    }


def format_batch_result(payload: dict, format_duration) -> str:
    """Human-readable summary shared by the batch tabs."""
    lines = [
        f"完了: {payload['total_files']}ファイル / {format_duration(payload['total_duration_sec'])}"
        f"（新規{payload.get('generated', 0)}・再利用{payload.get('skipped_existing', 0)}）",
        f"出力: {payload['output_dir']}",
        f"テキスト: {payload['esd_path']}",
    ]
    if payload.get("chunked"):
        lines.append(f"長文分割して生成: {payload['chunked']}行")
    failed = payload.get("failed") or []
    if failed:
        lines.append(f"⚠ 失敗してスキップ: {len(failed)}行（再実行すると失敗行だけ生成し直します）")
        for item in failed[:5]:
            lines.append(f"  行{item['index']}: {item['error']}")
    qc = payload.get("qc")
    if qc:
        lines.append(
            f"品質チェック（信号）: NG {qc['rejected']}件 / 注意 {qc['warned']}件"
            + ("　→ 品質チェック欄でASR確認・除外・再生成できます" if qc["rejected"] or qc["warned"] else "")
        )
    return "\n".join(lines)
