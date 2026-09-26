"""LoRA fine-tuning pipeline for Irodori-TTS.

End-to-end: takes vdc's Voice Clone output (``output/{folder}/raw/*.wav`` +
``Neutral.txt``) and runs the three Irodori training stages — convert,
encode latents, train — using the Irodori venv as a subprocess.

UI calls :func:`run_lora_pipeline` and consumes the yielded progress events.
Trained runs keep evenly spaced checkpoints so :func:`compare_checkpoints` can
render the same lines with each of them for listening comparison, and an
interrupted run can be continued with ``resume=True``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from config import BASE_DIR, OUTPUT_DIR
from modules.dataset_io import find_text_list, read_text_list
# Re-exported: callers historically imported the GPU lock helpers from here.
from modules.gpu_gate import GPUBusyError, get_gate, gpu_session, wait_messages  # noqa: F401
from modules.irodori_bridge import _default_irodori_root

logger = logging.getLogger(__name__)


class SubprocessFailed(RuntimeError):
    """Raised when an external Irodori command exits non-zero."""

    def __init__(self, cmd: list[str], returncode: int, stderr_tail: Iterable[str]):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr_tail = tuple(stderr_tail)
        message = f"subprocess exited with code {returncode}: {' '.join(cmd)}"
        if self.stderr_tail:
            message += "\nlast stderr:\n" + "\n".join(self.stderr_tail)
        super().__init__(message)

LORA_OUTPUT_DIR = OUTPUT_DIR / "lora"
LORA_DATA_DIR = OUTPUT_DIR / "lora_data"
INIT_CHECKPOINT_REPO = "Aratako/Irodori-TTS-500M-v3"
TRAIN_CONFIG_RELATIVE = Path("configs") / "train_500m_v3_lora.yaml"
V4_LORA_OUTPUT_DIR = OUTPUT_DIR / "lora_v4"
V4_LORA_DATA_DIR = OUTPUT_DIR / "lora_data_v4"
V4_INIT_CHECKPOINT_REPO = "Aratako/Irodori-TTS-v4.1-Small"
V4_TRAIN_CONFIG_RELATIVE = Path("configs") / "train_v4_small_lora.yaml"
LORA_COMPARE_DIR = OUTPUT_DIR / "lora_compare"

PREFERRED_CHECKPOINT_FILE = "preferred_checkpoint.txt"
ARCHIVE_DIR_NAME = "_archive"
# Number of evenly spaced checkpoints kept per run for listening comparison.
COMPARE_CHECKPOINTS = 5
# Below this many utterances a held-out validation split is only a handful of
# clips, so its loss is noise; train on everything and compare by ear instead.
MIN_UTTERANCES_FOR_VALIDATION = 2000

STEP_PRESETS: dict[str, int] = {
    "quick": 3000,
    "normal": 10000,
    "full": 30000,
}

# The shipped train_500m_v3_lora.yaml targets large-scale training (batch=80,
# 16 dataloader workers). For vdc's small clone datasets that combo wastes
# I/O and gives ~50 s/step on a 4060 Ti. Override per preset so quick runs
# actually feel quick.
PRESET_TRAIN_OVERRIDES: dict[str, dict[str, str]] = {
    "quick":  {"batch_size": "8",  "num_workers": "2"},
    "normal": {"batch_size": "16", "num_workers": "4"},
    "full":   {"batch_size": "32", "num_workers": "4"},
}

# v4-Small is ~766M and includes a fine-tuned ModernBERT backbone.  Keep its
# defaults conservative enough for the local 12 GB RTX 3060; users can raise
# them from the advanced controls after observing actual VRAM headroom.
V4_PRESET_TRAIN_OVERRIDES: dict[str, dict[str, str]] = {
    "quick":  {"batch_size": "2", "num_workers": "2"},
    "normal": {"batch_size": "4", "num_workers": "2"},
    "full":   {"batch_size": "4", "num_workers": "4"},
}

_PERIODIC_CKPT_RE = re.compile(r"^checkpoint_(\d{7})$")
_BEST_CKPT_RE = re.compile(r"^checkpoint_best_val_loss_(\d{7})_")


@dataclass
class LoraResult:
    speaker: str
    output_dir: Path
    adapter_path: Path | None


def list_clone_sources() -> list[str]:
    """Return ``output/`` subfolders that look like Voice Clone outputs."""
    if not OUTPUT_DIR.is_dir():
        return []
    sources: list[str] = []
    for entry in sorted(OUTPUT_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in {"voice_design", "lora", "lora_data", "lora_v4", "lora_data_v4", "lora_compare"}:
            continue
        raw = entry / "raw"
        if raw.is_dir() and any(raw.glob("*.wav")):
            sources.append(entry.name)
    return sources


def _lora_paths(profile: str) -> tuple[Path, Path]:
    if profile == "v4":
        return V4_LORA_OUTPUT_DIR, V4_LORA_DATA_DIR
    if profile != "legacy":
        raise ValueError(f"unknown Irodori LoRA profile: {profile}")
    return LORA_OUTPUT_DIR, LORA_DATA_DIR


def list_loras(profile: str = "legacy") -> list[str]:
    """Return available LoRA adapter speakers (folder names under output/lora/)."""
    output_dir, _ = _lora_paths(profile)
    if not output_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in output_dir.iterdir()
        if p.is_dir() and _resolve_adapter_path(p) is not None
    )


def get_lora_training_wavs(speaker: str, profile: str = "legacy", limit: int = 1) -> list[str]:
    """Return up to ``limit`` wavs from the LoRA's training data (lab/{speaker}/...)."""
    if not speaker:
        return []
    _, data_dir = _lora_paths(profile)
    lab_speaker = data_dir / "lab" / speaker
    if not lab_speaker.is_dir():
        return []
    for wavs_dir in sorted(lab_speaker.rglob("wavs")):
        if not wavs_dir.is_dir():
            continue
        wavs = sorted(wavs_dir.glob("*.wav"))
        if wavs:
            return [str(path) for path in wavs[:max(1, limit)]]
    return []


def get_lora_training_wav(speaker: str, profile: str = "legacy") -> str | None:
    """Return a wav file from the LoRA's training data (lab/{speaker}/...).

    Used to auto-fill the reference audio when a LoRA is selected, since
    Irodori's clone mode requires ref_wav even with a trained adapter.
    """
    wavs = get_lora_training_wavs(speaker, profile, limit=1)
    return wavs[0] if wavs else None


def get_lora_adapter_path(speaker: str, profile: str = "legacy") -> str | None:
    """Return the adapter *directory* for a saved LoRA, if any.

    Irodori's ``SamplingRequest.lora_adapter`` expects a PEFT adapter directory
    (containing ``adapter_model.safetensors`` + ``adapter_config.json``), not
    the safetensors file itself.  A checkpoint chosen in the comparison panel
    (``preferred_checkpoint.txt``) wins over the automatic choice.
    """
    if not speaker:
        return None
    output_dir, _ = _lora_paths(profile)
    folder = output_dir / speaker
    if not folder.is_dir():
        return None
    preferred = get_preferred_checkpoint(speaker, profile)
    if preferred:
        candidate = folder / preferred
        if (candidate / "adapter_model.safetensors").is_file():
            return str(candidate)
    found = _resolve_adapter_path(folder)
    return str(found.parent) if found is not None else None


def _adapter_candidates(folder: Path) -> list[Path]:
    return [
        p for p in folder.rglob("adapter_model.safetensors")
        if ARCHIVE_DIR_NAME not in p.relative_to(folder).parts
    ]


def _resolve_adapter_path(folder: Path) -> Path | None:
    """train.py with --lora writes a PEFT adapter under output-dir.

    A single run can produce several intermediate ``checkpoint_*`` folders plus
    a ``checkpoint_final`` (or similar) at the end. We prefer:
      1. anything named like ``*final*`` or ``*last*``
      2. otherwise the newest ``adapter_model.safetensors`` by mtime

    Archived runs (``_archive/``) are ignored.  Returns the path of the
    safetensors file (caller derives the directory).
    """
    candidates = _adapter_candidates(folder)
    if not candidates:
        candidates = sorted(
            p for p in folder.rglob("*.safetensors")
            if ARCHIVE_DIR_NAME not in p.relative_to(folder).parts
        )
    if not candidates:
        return None

    def _is_final(p: Path) -> bool:
        parts = "/".join(p.parts).lower()
        return "final" in parts or "last" in parts

    final_first = sorted(
        candidates,
        key=lambda p: (0 if _is_final(p) else 1, -p.stat().st_mtime),
    )
    return final_first[0]


# ---------------------------------------------------------------- checkpoints

def list_lora_checkpoints(speaker: str, profile: str = "legacy") -> list[dict]:
    """All adapter checkpoints of the current run, ordered by training step."""
    output_dir, _ = _lora_paths(profile)
    folder = output_dir / speaker
    if not speaker or not folder.is_dir():
        return []
    checkpoints: list[dict] = []
    for child in folder.iterdir():
        if not child.is_dir() or not (child / "adapter_model.safetensors").is_file():
            continue
        periodic = _PERIODIC_CKPT_RE.match(child.name)
        best = _BEST_CKPT_RE.match(child.name)
        if periodic:
            step, kind = int(periodic.group(1)), "periodic"
        elif best:
            step, kind = int(best.group(1)), "best_val"
        elif "final" in child.name:
            step, kind = _checkpoint_step(child), "final"
        else:
            step, kind = _checkpoint_step(child), "other"
        checkpoints.append({"name": child.name, "path": str(child), "step": step, "kind": kind})
    order = {"periodic": 0, "best_val": 1, "other": 2, "final": 3}
    checkpoints.sort(key=lambda c: (c["step"] if c["step"] is not None else 10**9, order[c["kind"]]))
    return checkpoints


def _checkpoint_step(path: Path) -> int | None:
    """Step recorded in a LoRA checkpoint's trainer state (final/other names)."""
    state = path / "trainer_state.pt"
    if not state.is_file():
        return None
    try:
        import torch

        payload = torch.load(state, map_location="cpu", weights_only=False)
        return int(payload.get("step"))
    except Exception:
        return None


def get_preferred_checkpoint(speaker: str, profile: str = "legacy") -> str | None:
    output_dir, _ = _lora_paths(profile)
    marker = output_dir / speaker / PREFERRED_CHECKPOINT_FILE
    try:
        name = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return name or None


def set_preferred_checkpoint(speaker: str, checkpoint: str | None, profile: str = "legacy") -> None:
    """Pin the checkpoint used whenever this LoRA is selected (None = automatic)."""
    output_dir, _ = _lora_paths(profile)
    folder = output_dir / speaker
    marker = folder / PREFERRED_CHECKPOINT_FILE
    if not checkpoint:
        marker.unlink(missing_ok=True)
        return
    if not (folder / checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"checkpoint not found: {folder / checkpoint}")
    marker.write_text(checkpoint, encoding="utf-8")


def latest_resumable_checkpoint(folder: Path) -> tuple[Path, int] | None:
    """Newest periodic checkpoint that still has trainer (optimizer) state."""
    best: tuple[Path, int] | None = None
    if not folder.is_dir():
        return None
    for child in folder.iterdir():
        match = _PERIODIC_CKPT_RE.match(child.name)
        if not match or not (child / "trainer_state.pt").is_file():
            continue
        step = int(match.group(1))
        if best is None or step > best[1]:
            best = (child, step)
    return best


def _archive_previous_run(folder: Path) -> Path | None:
    """Move an earlier run's checkpoints aside so they can't be mistaken for new ones."""
    old = [p for p in folder.iterdir() if p.name.startswith("checkpoint_")] if folder.is_dir() else []
    if not old:
        return None
    target = folder / ARCHIVE_DIR_NAME / time.strftime("%Y%m%d_%H%M%S")
    target.mkdir(parents=True, exist_ok=True)
    for path in old:
        shutil.move(str(path), str(target / path.name))
    for name in (PREFERRED_CHECKPOINT_FILE, "config.json"):
        stale = folder / name
        if stale.is_file():
            shutil.move(str(stale), str(target / name))
    return target


# ---------------------------------------------------------------- data prep

def _load_esd_lines(path: Path) -> list[tuple[str, str]]:
    return read_text_list(path)


def _find_esd_file(folder: Path) -> Path | None:
    return find_text_list(folder)


def _stage(status: str, **extra) -> dict:
    return {"event": "stage", "status": status, **extra}


def _write_training_jsonl(
    *,
    lab_root: Path,
    speaker: str,
    emotion: str,
    out_jsonl: Path,
    include_speaker: bool = False,
    caption: str | None = None,
) -> int:
    """Walk lab/{speaker}/{emotion}/ and emit a HF-datasets-style JSONL.

    Output line shape::

        {"audio": "<absolute wav path>", "text": "<utterance text>"}
    """
    emotion_dir = lab_root / speaker / emotion
    txt_file = emotion_dir / f"{emotion}.txt"
    wavs_dir = emotion_dir / "wavs"
    if not txt_file.is_file():
        raise FileNotFoundError(f"{txt_file} not found")
    if not wavs_dir.is_dir():
        raise FileNotFoundError(f"{wavs_dir} not found")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for line in txt_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            file_id, text = line.split(":", 1)
            file_id = file_id.strip()
            text = text.strip()
            wav_path = wavs_dir / f"{file_id}.wav"
            if not wav_path.is_file():
                continue
            payload = {"audio": str(wav_path.resolve()), "text": text}
            if include_speaker:
                payload["speaker_id"] = speaker
            if caption and caption.strip():
                payload["caption"] = caption.strip()
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        raise ValueError(f"no entries produced from {txt_file}")
    return written


def _convert_clone_to_lab(
    *, source: Path, lab_root: Path, speaker: str, emotion: str
) -> int:
    """Translate vdc clone output into lab-format used by Irodori training.

    Returns the number of utterances exported.
    """
    raw_dir = source / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"missing raw/ under {source}")
    esd = _find_esd_file(source)
    if esd is None:
        raise FileNotFoundError(
            f"no text list (Neutral.txt / esd.list) found under {source}"
        )

    entries = _load_esd_lines(esd)
    if not entries:
        raise ValueError(f"text list {esd} has no usable lines")

    dest_dir = lab_root / speaker / emotion
    dest_wavs = dest_dir / "wavs"
    if dest_wavs.exists():
        shutil.rmtree(dest_wavs)
    dest_wavs.mkdir(parents=True, exist_ok=True)

    txt_lines: list[str] = []
    written = 0
    for file_id, text in entries:
        stem = file_id
        src_wav = raw_dir / f"{stem}.wav"
        if not src_wav.is_file():
            logger.warning("missing wav for id=%s under %s", stem, raw_dir)
            continue
        shutil.copy2(src_wav, dest_wavs / f"{stem}.wav")
        txt_lines.append(f"{stem}: {text}")
        written += 1

    if not written:
        raise ValueError(f"no wav/text pairs matched between {raw_dir} and {esd}")

    (dest_dir / f"{emotion}.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    return written


def _stream_subprocess(
    cmd: list[str], *, cwd: Path, env: dict | None = None
) -> Iterator[str]:
    """Yield stdout lines from a subprocess, mirroring stderr to logger.warning.

    Cancellation: when the generator is closed (Gradio cancel / GeneratorExit),
    the child is terminated, then killed if it doesn't exit promptly. Without
    this, training would keep holding the GPU after the user pressed Stop.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    stderr_tail: deque[str] = deque(maxlen=50)

    def _drain_stderr(stream):
        for line in stream:
            text = line.rstrip()
            stderr_tail.append(text)
            logger.warning("[subprocess] %s", text)

    stderr_thread = threading.Thread(target=_drain_stderr, args=(proc.stderr,), daemon=True)
    stderr_thread.start()

    cancelled = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line.rstrip("\n")
    except GeneratorExit:
        cancelled = True
        raise
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            except Exception:
                logger.exception("Failed to clean up subprocess")
        else:
            proc.wait()
        stderr_thread.join(timeout=2)
    if cancelled:
        return
    if proc.returncode != 0:
        raise SubprocessFailed(cmd, proc.returncode, stderr_tail)


def _irodori_python() -> Path:
    root = _default_irodori_root()
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _ensure_init_checkpoint(profile: str = "legacy") -> Path:
    """Download (or reuse) the selected base checkpoint for LoRA training.

    We do this from the vdc venv so the user sees progress in the same console.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed in the vdc venv. "
            "Re-run setup.bat to install dependencies."
        ) from exc
    if profile == "v4":
        snapshot = snapshot_download(
            repo_id=V4_INIT_CHECKPOINT_REPO,
            allow_patterns=["model.safetensors", "tokenizer/*"],
        )
        return Path(snapshot) / "model.safetensors"
    path = hf_hub_download(repo_id=INIT_CHECKPOINT_REPO, filename="model.safetensors")
    return Path(path)


def _sanitize_speaker(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip()).strip("._")
    return cleaned or "speaker"


def _resolve_steps(steps: int | str) -> int:
    if isinstance(steps, str):
        return STEP_PRESETS.get(steps.lower(), STEP_PRESETS["quick"])
    return int(steps)


def checkpoint_schedule(max_steps: int, n_utterances: int) -> dict[str, str]:
    """train.py overrides that keep comparable checkpoints and useful logs."""
    save_every = max(100, int(round(max_steps / COMPARE_CHECKPOINTS / 100.0)) * 100)
    log_every = max(10, min(100, max_steps // 300))
    overrides = {
        "--save-every": str(save_every),
        "--log-every": str(log_every),
        # train.py keeps checkpoint_best_n + 1 periodic checkpoints when there
        # is no validation split.
        "--checkpoint-best-n": str(COMPARE_CHECKPOINTS),
    }
    if n_utterances < MIN_UTTERANCES_FOR_VALIDATION:
        overrides["--valid-ratio"] = "0"
    return overrides


def run_lora_pipeline(
    *,
    source: str | Path,
    speaker: str,
    steps: int | str = "quick",
    emotion: str = "neutral",
    batch_size: int | None = None,
    num_workers: int | None = None,
    learning_rate: float | None = None,
    profile: str = "legacy",
    caption: str | None = None,
    resume: bool = False,
) -> Iterator[dict]:
    """Yield progress events while running the full LoRA pipeline.

    Yield shape::

        {"event": "stage", "status": "...", ...}
        {"event": "progress", "step": int, "max_steps": int, "loss": float | None}
        {"event": "done", "result": LoraResult, ...}

    ``resume=True`` continues the newest periodic checkpoint of the same
    speaker (optimizer + dataloader state included) and reuses the already
    encoded dataset.
    """
    safe_speaker = _sanitize_speaker(speaker)
    lora_output_dir, lora_data_dir = _lora_paths(profile)
    init_checkpoint_repo = (
        V4_INIT_CHECKPOINT_REPO if profile == "v4" else INIT_CHECKPOINT_REPO
    )
    train_config_relative = (
        V4_TRAIN_CONFIG_RELATIVE if profile == "v4" else TRAIN_CONFIG_RELATIVE
    )
    max_steps = _resolve_steps(steps)
    source_dir = OUTPUT_DIR / source if isinstance(source, str) else Path(source)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source folder not found: {source_dir}")

    irodori_root = _default_irodori_root()
    py = _irodori_python()
    if not py.is_file():
        raise RuntimeError(
            f"Irodori venv python not found at {py}. Re-run setup.bat to install."
        )

    train_output_dir = lora_output_dir / safe_speaker
    manifest_path = lora_data_dir / "manifests" / f"{safe_speaker}_manifest.jsonl"
    resume_from: tuple[Path, int] | None = None
    if resume:
        resume_from = latest_resumable_checkpoint(train_output_dir)
        if resume_from is None:
            raise RuntimeError(
                f"再開できるチェックポイントがありません: {train_output_dir}（checkpoint_XXXXXXX が必要です）"
            )
        if resume_from[1] >= max_steps:
            raise RuntimeError(
                f"最新チェックポイントは既に step {resume_from[1]} です。"
                f"再開するにはステップ数を {resume_from[1]} より大きくしてください。"
            )
        if not manifest_path.is_file():
            raise RuntimeError(f"学習データのmanifestがありません: {manifest_path}")

    # Train and inference must not share the GPU; wait for any running job.
    gate = get_gate()
    for message in wait_messages("LoRA学習"):
        yield _stage("gpu_wait", message=message)
    if not gate.try_acquire("LoRA学習", timeout=None):
        raise GPUBusyError("GPU gate could not be acquired")
    try:
        from modules.irodori_bridge import get_bridge
        try:
            get_bridge().shutdown()
        except Exception:
            logger.exception("Failed to shut down Irodori worker before training")

        lora_data_dir.mkdir(parents=True, exist_ok=True)
        lora_output_dir.mkdir(parents=True, exist_ok=True)
        lab_root = lora_data_dir / "lab"
        jsonl_dir = lora_data_dir / "jsonl"
        latents_root = lora_data_dir / "latents"
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        latents_root.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if resume_from is None:
            archived = _archive_previous_run(train_output_dir)
            if archived is not None:
                yield _stage("archive", message=f"前回の学習結果を退避しました: {archived}")

            # ── Stage 1: convert ──
            yield _stage("convert", message=f"Converting {source_dir.name} -> lab/{safe_speaker}/{emotion}")
            n_utts = _convert_clone_to_lab(
                source=source_dir, lab_root=lab_root, speaker=safe_speaker, emotion=emotion,
            )
            yield _stage("convert_done", utterances=n_utts)

            # ── Stage 2: create jsonl (pure file processing, runs in vdc venv) ──
            jsonl_path = jsonl_dir / f"{safe_speaker}.jsonl"
            yield _stage("create_jsonl", message=f"Writing {jsonl_path.name}")
            _write_training_jsonl(
                lab_root=lab_root,
                speaker=safe_speaker,
                emotion=emotion,
                out_jsonl=jsonl_path,
                include_speaker=(profile == "v4"),
                caption=caption if profile == "v4" else None,
            )
            yield _stage("create_jsonl_done", jsonl=str(jsonl_path))

            # ── Stage 3: encode latents (runs in Irodori venv via subprocess) ──
            latent_dir = latents_root / safe_speaker
            encode_script = BASE_DIR / "modules" / "irodori_encode_latents.py"
            yield _stage("encode_latents", message="Encoding waveforms to latents")
            if manifest_path.exists():
                manifest_path.unlink()
            encode_error: SubprocessFailed | None = None
            encode_cmd = [
                str(py), str(encode_script),
                "--input-jsonl", str(jsonl_path),
                "--latent-dir", str(latent_dir),
                "--manifest", str(manifest_path),
            ]
            try:
                list(_stream_subprocess(encode_cmd, cwd=irodori_root))
            except SubprocessFailed as exc:
                encode_error = exc
            if not manifest_path.is_file():
                if encode_error is not None:
                    raise RuntimeError(
                        f"encode_latents failed before producing manifest: {manifest_path}"
                    ) from encode_error
                raise RuntimeError(f"encode_latents did not produce manifest: {manifest_path}")
        else:
            yield _stage(
                "resume",
                message=f"{resume_from[0].name}（step {resume_from[1]}）から再開します。データ変換はスキップします",
            )
        manifest_count = _count_nonempty_lines(manifest_path)
        if manifest_count == 0:
            message = (
                f"encode_latents wrote 0 entries to {manifest_path}. "
                "All audio files were rejected; check the [subprocess] logs above for skip reasons."
            )
            if resume_from is None and encode_error is not None:
                raise RuntimeError(message) from encode_error
            raise RuntimeError(message)
        if resume_from is None and encode_error is not None:
            if manifest_count < n_utts:
                raise RuntimeError(
                    f"encode_latents exited non-zero after writing only "
                    f"{manifest_count}/{n_utts} entries to {manifest_path}"
                ) from encode_error
            logger.warning(
                "encode_latents exited non-zero after writing %s/%s manifest entries; continuing",
                manifest_count, n_utts,
            )
            yield _stage(
                "encode_latents_warning",
                message=(
                    "Encoder exited non-zero after writing a complete manifest; "
                    "continuing to training."
                ),
                entries=manifest_count,
            )
        n_utts = manifest_count
        yield _stage(
            "encode_latents_done", manifest=str(manifest_path), entries=manifest_count,
        )

        # ── Stage 4: train ──
        yield _stage("init_checkpoint", message=f"Resolving init checkpoint ({init_checkpoint_repo})")
        init_ckpt = _ensure_init_checkpoint(profile)
        yield _stage("init_checkpoint_done", checkpoint=str(init_ckpt))

        train_output_dir.mkdir(parents=True, exist_ok=True)
        train_config = irodori_root / train_config_relative
        if not train_config.is_file():
            raise RuntimeError(f"train config not found: {train_config}")

        preset_key = steps if isinstance(steps, str) else "quick"
        profile_presets = (
            V4_PRESET_TRAIN_OVERRIDES if profile == "v4" else PRESET_TRAIN_OVERRIDES
        )
        preset = profile_presets.get(preset_key, profile_presets["quick"])
        eff_batch = int(batch_size) if batch_size is not None else int(preset["batch_size"])
        eff_workers = int(num_workers) if num_workers is not None else int(preset["num_workers"])
        schedule = checkpoint_schedule(max_steps, n_utts)
        yield _stage(
            "train",
            message=(
                f"Training LoRA ({max_steps} steps, batch={eff_batch}, "
                f"checkpoint every {schedule['--save-every']} steps"
                + (", validation off (small dataset)" if "--valid-ratio" in schedule else "")
                + ")"
            ),
            max_steps=max_steps,
        )
        cmd = [
            str(py), str(irodori_root / "train.py"),
            "--config", str(train_config),
            "--manifest", str(manifest_path),
            "--init-checkpoint", str(init_ckpt),
            "--output-dir", str(train_output_dir),
            "--lora",
            "--max-steps", str(max_steps),
            "--batch-size", str(eff_batch),
            "--num-workers", str(eff_workers),
        ]
        for flag, value in schedule.items():
            cmd.extend([flag, value])
        if resume_from is not None:
            cmd.extend(["--resume", str(resume_from[0])])
        if learning_rate is not None:
            cmd.extend(["--lr", str(float(learning_rate))])
        for line in _stream_subprocess(cmd, cwd=irodori_root):
            ev = _parse_train_line(line, max_steps)
            if ev is not None:
                yield ev

        adapter_path = _resolve_adapter_path(train_output_dir)
        yield {
            "event": "done",
            "speaker": safe_speaker,
            "output_dir": str(train_output_dir),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "utterances": n_utts,
            "steps": max_steps,
            "profile": profile,
            "checkpoints": [c["name"] for c in list_lora_checkpoints(safe_speaker, profile)],
        }
    finally:
        gate.release()


def _count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# train.py prints ``step=1200 loss=0.512300 rf=0.51 lr=1.000e-04`` (and
# ``valid step=1000 loss=...`` for validation). Older logs used
# ``[step 1234/30000] loss=...``; accept both.  Anything unrecognised becomes a
# generic log line so the UI can still surface it.
_STEP_RE = re.compile(
    r"(?<![A-Za-z_])step\s*[:=]?\s*(?P<step>\d+)(?:\s*(?:/|of)\s*(?P<max>\d+))?",
    re.IGNORECASE,
)
_LOSS_RE = re.compile(r"(?<![A-Za-z_])loss\s*[:=]?\s*(?P<loss>[0-9.]+(?:[eE][+-]?\d+)?)")
_RESUMED_RE = re.compile(r"Resumed from step=(?P<step>\d+)")


def _parse_train_line(line: str, max_steps: int) -> dict | None:
    if not line.strip():
        return None
    stripped = line.strip()
    resumed = _RESUMED_RE.search(stripped)
    if resumed:
        return {"event": "progress", "step": int(resumed.group("step")), "max_steps": max_steps,
                "loss": None, "raw": line}
    step_m = _STEP_RE.search(stripped)
    if step_m and not stripped.lower().startswith("valid"):
        loss_m = _LOSS_RE.search(stripped)
        return {
            "event": "progress",
            "step": int(step_m.group("step")),
            "max_steps": int(step_m.group("max")) if step_m.group("max") else max_steps,
            "loss": float(loss_m.group("loss")) if loss_m else None,
            "raw": line,
        }
    return {"event": "log", "raw": line}


# ---------------------------------------------------------------- comparison

def compare_checkpoints(
    *,
    speaker: str,
    profile: str = "legacy",
    texts: list[str],
    seed: int = 1234,
    checkpoints: list[str] | None = None,
    refs: list[str] | None = None,
    model_variant: str = "full",
    model_precision: str = "bf16",
    release_mode: str = "idle",
) -> Iterator[tuple[float, object]]:
    """Render the same ``texts`` with every checkpoint using a fixed seed.

    Yields ``(pct, message)`` then ``(1.0, rows)`` where each row is
    ``{"checkpoint", "step", "line", "text", "path"}``.  Output lives in
    ``output/lora_compare/{profile}/{speaker}/``.
    """
    from modules.irodori_bridge import get_bridge
    from modules.irodori_jobs import finish_release, synthesize_to_file

    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        raise ValueError("比較用のテキストを1行以上入力してください")
    available = list_lora_checkpoints(speaker, profile)
    if checkpoints:
        wanted = set(checkpoints)
        available = [c for c in available if c["name"] in wanted]
    if not available:
        raise FileNotFoundError(f"{speaker} のチェックポイントが見つかりません")
    ref_list = refs or get_lora_training_wavs(speaker, profile, limit=3 if profile == "v4" else 1)
    if not ref_list:
        raise FileNotFoundError("参照音声（学習データのwav）が見つかりません")

    out_root = LORA_COMPARE_DIR / profile / speaker
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    request: dict = {"profile": profile, "mode": "clone"}
    if profile == "v4":
        request.update({"model_variant": model_variant, "model_precision": model_precision, "no_ref": False})

    bridge = get_bridge()
    rows: list[dict] = []
    total = len(available) * len(texts)
    for message in wait_messages("LoRA比較"):
        yield 0.0, message
    with gpu_session("LoRA比較"):
        try:
            done = 0
            for ckpt in available:
                for line_no, text in enumerate(texts, start=1):
                    out_path = out_root / f"{ckpt['name']}__{line_no:02d}.wav"
                    synthesize_to_file(
                        bridge, text=text, out_path=out_path, refs=ref_list, seed=seed,
                        lora_path=ckpt["path"], release_after_synthesis=False, **request,
                    )
                    rows.append({"checkpoint": ckpt["name"], "step": ckpt["step"],
                                 "line": line_no, "text": text, "path": str(out_path)})
                    done += 1
                    yield done / total, f"{done}/{total} | {ckpt['name']} 文{line_no}"
        finally:
            finish_release(bridge, release_mode)
    yield 1.0, rows
