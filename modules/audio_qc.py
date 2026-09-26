"""Quality checks for generated training clips.

Synthetic clone output occasionally contains misreadings, silence, clipped
peaks or clips that were cut at the model's maximum duration.  Feeding those
into LoRA training degrades the adapter, so this module scores every clip in a
dataset folder and can rewrite the text list to exclude the bad ones.

Two layers:

* signal checks (always, CPU only): duration, near-max-length truncation,
  clipping, silence, edge silence and speaking-rate outliers within the batch;
* optional ASR check: transcribe with Whisper and compare to the script with a
  character error rate (CER).

Outputs (inside the dataset folder):

* ``qc_report.json`` / ``qc_report.csv`` — per-clip metrics and flags
* ``Neutral.all.txt`` — untouched backup of the text list, created the first
  time a filter is applied; the filter always starts from this backup.
"""

from __future__ import annotations

import csv
import logging
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from modules.dataset_io import atomic_write_json, find_text_list, read_json, read_text_list, write_text_list

logger = logging.getLogger(__name__)

QC_REPORT_JSON = "qc_report.json"
QC_REPORT_CSV = "qc_report.csv"
DEFAULT_ASR_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_CER_THRESHOLD = 0.35
DEFAULT_MAX_SECONDS = 30.0

# Flags that exclude a clip from training when the filter is applied.
REJECT_FLAGS = {
    "too_short": "短すぎる",
    "truncated": "最大長で切れた可能性",
    "clipping": "音割れ",
    "silent": "ほぼ無音",
    "speed_outlier": "話速が他の文と大きく異なる",
    "asr_mismatch": "音声認識の結果が原文と不一致",
    "missing": "wavがない/読めない",
}
# Flags shown for review but not excluded automatically.
WARN_FLAGS = {
    "edge_silence": "前後の無音が長い",
}

_FRAME_SEC = 0.02


# ---------------------------------------------------------------- text / CER

def _to_hiragana(text: str) -> str:
    return "".join(
        chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch
        for ch in text
    )


def normalize_for_cer(text: str) -> str:
    """Keep only letters/digits so punctuation, emoji and spacing don't count."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = _to_hiragana(text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in {"L", "N"})


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (r != h),
            )
        previous = current
    return previous[-1] / float(len(ref))


def countable_chars(text: str) -> int:
    return len(normalize_for_cer(text))


# ---------------------------------------------------------------- signal checks

def analyze_clip(path: str | Path, text: str, *, max_seconds: float = DEFAULT_MAX_SECONDS) -> dict:
    """Return signal metrics and per-clip flags for one wav."""
    path = Path(path)
    record: dict = {"file": path.name, "text": text, "flags": []}
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        record["flags"].append("missing")
        record["error"] = str(exc)
        return record
    mono = audio.mean(axis=1) if audio.size else np.zeros(0, dtype=np.float32)
    duration = mono.size / float(sr) if sr else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    rms_db = 20.0 * np.log10(max(rms, 1e-9))
    clip_ratio = float(np.mean(np.abs(mono) >= 0.999)) if mono.size else 0.0

    frame = max(1, int(sr * _FRAME_SEC))
    n_frames = mono.size // frame
    if n_frames:
        frames = mono[: n_frames * frame].reshape(n_frames, frame)
        frame_db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(np.square(frames), axis=1)), 1e-9))
        threshold = max(-55.0, float(frame_db.max()) - 35.0)
        voiced = frame_db > threshold
        silence_ratio = float(1.0 - voiced.mean())
        voiced_idx = np.flatnonzero(voiced)
        if voiced_idx.size:
            lead = voiced_idx[0] * _FRAME_SEC
            trail = (n_frames - 1 - voiced_idx[-1]) * _FRAME_SEC
        else:
            lead = trail = duration
    else:
        silence_ratio, lead, trail = 1.0, duration, duration

    chars = countable_chars(text)
    sec_per_char = duration / chars if chars else None
    record.update({
        "duration_sec": round(duration, 3),
        "sample_rate": int(sr),
        "peak": round(peak, 4),
        "rms_db": round(float(rms_db), 1),
        "clip_ratio": round(clip_ratio, 5),
        "silence_ratio": round(silence_ratio, 3),
        "lead_silence_sec": round(float(lead), 2),
        "trail_silence_sec": round(float(trail), 2),
        "chars": chars,
        "sec_per_char": None if sec_per_char is None else round(sec_per_char, 4),
    })

    flags = record["flags"]
    if duration < 0.3:
        flags.append("too_short")
    if max_seconds and duration >= max_seconds - 0.25:
        flags.append("truncated")
    if clip_ratio > 0.001:
        flags.append("clipping")
    if rms_db < -45.0 or silence_ratio > 0.7:
        flags.append("silent")
    if lead > 1.5 or trail > 1.5:
        flags.append("edge_silence")
    return record


def flag_speed_outliers(records: list[dict], *, low: float = 0.5, high: float = 2.0) -> None:
    """Flag clips whose seconds-per-character is far from the batch median.

    The median is robust against the few broken clips we are hunting for;
    at least five measurable clips are needed for a meaningful baseline.
    """
    rates = [r["sec_per_char"] for r in records if r.get("sec_per_char")]
    if len(rates) < 5:
        return
    median = statistics.median(rates)
    if median <= 0:
        return
    for record in records:
        rate = record.get("sec_per_char")
        if not rate:
            continue
        ratio = rate / median
        record["speed_ratio"] = round(ratio, 3)
        if (ratio < low or ratio > high) and "speed_outlier" not in record["flags"]:
            record["flags"].append("speed_outlier")


def is_rejected(record: dict) -> bool:
    return any(flag in REJECT_FLAGS for flag in record.get("flags", []))


def describe_flags(flags: list[str]) -> str:
    labels = {**REJECT_FLAGS, **WARN_FLAGS}
    return "、".join(labels.get(flag, flag) for flag in flags)


# ---------------------------------------------------------------- ASR

class WhisperTranscriber:
    """Lazy transformers Whisper pipeline (runs in the vdc venv)."""

    def __init__(self, model_name: str = DEFAULT_ASR_MODEL) -> None:
        self.model_name = model_name or DEFAULT_ASR_MODEL
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from transformers import pipeline

        use_cuda = torch.cuda.is_available()
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device="cuda:0" if use_cuda else "cpu",
        )
        return self._pipe

    def transcribe(self, path: str | Path) -> str:
        import librosa

        pipe = self._load()
        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        result = pipe(
            {"raw": audio, "sampling_rate": 16000},
            generate_kwargs={"language": "japanese", "task": "transcribe"},
        )
        return str(result.get("text", "")).strip()

    def close(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------- folder QC

def dataset_entries(folder: str | Path, wavs_folder: str = "raw") -> tuple[Path, list[tuple[str, str]]]:
    """Return (wav_dir, entries) using the pristine backup when one exists."""
    folder = Path(folder)
    backup = folder / "Neutral.all.txt"
    text_list = backup if backup.is_file() else find_text_list(folder)
    if text_list is None:
        raise FileNotFoundError(f"テキストリスト（Neutral.txt）が見つかりません: {folder}")
    return folder / wavs_folder, read_text_list(text_list)


def run_qc(
    folder: str | Path,
    *,
    wavs_folder: str = "raw",
    use_asr: bool = False,
    asr_model: str = DEFAULT_ASR_MODEL,
    cer_threshold: float = DEFAULT_CER_THRESHOLD,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> Iterator[tuple[float, object]]:
    """Score every clip. Yields ``(pct, message)`` then ``(1.0, summary)``."""
    folder = Path(folder)
    wav_dir, entries = dataset_entries(folder, wavs_folder)
    if not entries:
        raise ValueError(f"テキストリストが空です: {folder}")

    records: list[dict] = []
    total = len(entries)
    for index, (file_id, text) in enumerate(entries, start=1):
        record = analyze_clip(wav_dir / f"{file_id}.wav", text, max_seconds=max_seconds)
        record["id"] = file_id
        records.append(record)
        if index % 20 == 0 or index == total:
            yield (index / total) * (0.3 if use_asr else 1.0), f"信号チェック {index}/{total}"
    flag_speed_outliers(records)

    if use_asr:
        yield from _run_asr(records, wav_dir, asr_model=asr_model, cer_threshold=cer_threshold)

    summary = _write_report(folder, records, use_asr=use_asr, asr_model=asr_model,
                            cer_threshold=cer_threshold)
    yield 1.0, summary


def _run_asr(records: list[dict], wav_dir: Path, *, asr_model: str, cer_threshold: float):
    from modules.gpu_gate import gpu_session, wait_messages
    from modules.irodori_bridge import get_bridge

    for message in wait_messages("QC音声認識"):
        yield 0.3, message
    targets = [r for r in records if "missing" not in r["flags"]]
    transcriber = WhisperTranscriber(asr_model)
    started = time.time()
    with gpu_session("QC音声認識"):
        # Whisper and a warm Irodori model together can exceed small GPUs.
        get_bridge().shutdown()
        try:
            yield 0.3, f"音声認識モデルを読み込み中: {transcriber.model_name}"
            for index, record in enumerate(targets, start=1):
                try:
                    hypothesis = transcriber.transcribe(wav_dir / f"{record['id']}.wav")
                except Exception as exc:
                    logger.exception("ASR failed for %s", record["id"])
                    record["asr_error"] = str(exc)
                    continue
                cer = character_error_rate(record["text"], hypothesis)
                record["asr_text"] = hypothesis
                record["cer"] = round(cer, 3)
                if cer > cer_threshold:
                    record["flags"].append("asr_mismatch")
                elapsed = time.time() - started
                remaining = elapsed / index * (len(targets) - index)
                yield 0.3 + 0.7 * index / len(targets), f"音声認識 {index}/{len(targets)} | 残り約{remaining:.0f}秒"
        finally:
            transcriber.close()


def _write_report(folder: Path, records: list[dict], *, use_asr: bool, asr_model: str,
                  cer_threshold: float) -> dict:
    rejected = [r for r in records if is_rejected(r)]
    warned = [r for r in records if not is_rejected(r) and r["flags"]]
    flag_counts: dict[str, int] = {}
    for record in records:
        for flag in record["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    summary = {
        "folder": str(folder),
        "total": len(records),
        "rejected": len(rejected),
        "warned": len(warned),
        "flag_counts": flag_counts,
        "use_asr": use_asr,
        "asr_model": asr_model if use_asr else None,
        "cer_threshold": cer_threshold if use_asr else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(folder / QC_REPORT_JSON, {"summary": summary, "records": records})

    columns = ["id", "verdict", "flags", "duration_sec", "sec_per_char", "speed_ratio",
               "peak", "rms_db", "silence_ratio", "cer", "text", "asr_text"]
    temp = folder / (QC_REPORT_CSV + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["verdict"] = "NG" if is_rejected(record) else ("注意" if record["flags"] else "OK")
            row["flags"] = describe_flags(record["flags"])
            writer.writerow(row)
    temp.replace(folder / QC_REPORT_CSV)
    return summary


def load_report(folder: str | Path) -> dict | None:
    return read_json(Path(folder) / QC_REPORT_JSON)


def rejected_ids(folder: str | Path) -> set[str]:
    report = load_report(folder) or {}
    return {r["id"] for r in report.get("records", []) if is_rejected(r)}


def apply_filter(folder: str | Path, *, esd_filename: str = "Neutral.txt") -> dict:
    """Rewrite the text list without QC-rejected clips (keeps a backup)."""
    folder = Path(folder)
    report = load_report(folder)
    if not report:
        raise FileNotFoundError("先に品質チェックを実行してください（qc_report.json がありません）")
    text_list = folder / esd_filename
    backup = folder / "Neutral.all.txt"
    if not backup.is_file():
        if not text_list.is_file():
            raise FileNotFoundError(f"{text_list} がありません")
        backup.write_bytes(text_list.read_bytes())
    entries = read_text_list(backup)
    bad = {r["id"] for r in report.get("records", []) if is_rejected(r)}
    kept = [(file_id, text) for file_id, text in entries if file_id not in bad]
    write_text_list(text_list, kept)
    return {"kept": len(kept), "excluded": len(entries) - len(kept), "backup": str(backup),
            "text_list": str(text_list)}


def restore_unfiltered(folder: str | Path, *, esd_filename: str = "Neutral.txt") -> bool:
    folder = Path(folder)
    backup = folder / "Neutral.all.txt"
    if not backup.is_file():
        return False
    (folder / esd_filename).write_bytes(backup.read_bytes())
    return True


def status_text(folder: str | Path) -> str:
    """One-line QC state for a dataset folder (used by the LoRA tab)."""
    folder = Path(folder)
    report = load_report(folder)
    if not report:
        return "品質チェック: 未実施（学習前の実行を推奨）"
    summary = report.get("summary", {})
    filtered = (folder / "Neutral.all.txt").is_file()
    asr = "ASRあり" if summary.get("use_asr") else "信号のみ"
    state = "フィルタ適用済み" if filtered else "フィルタ未適用"
    return (
        f"品質チェック: {summary.get('created_at', '?')}／{asr}／"
        f"NG {summary.get('rejected', 0)}件・注意 {summary.get('warned', 0)}件／{state}"
    )


def report_rows(folder: str | Path, *, only_flagged: bool = True) -> list[list]:
    report = load_report(folder) or {}
    rows = []
    for record in report.get("records", []):
        if only_flagged and not record.get("flags"):
            continue
        rows.append([
            record.get("id"),
            "NG" if is_rejected(record) else ("注意" if record.get("flags") else "OK"),
            describe_flags(record.get("flags", [])),
            record.get("duration_sec"),
            record.get("cer"),
            record.get("text"),
            record.get("asr_text", ""),
        ])
    return rows


REPORT_HEADERS = ["ID", "判定", "理由", "秒", "CER", "原文", "認識結果"]
