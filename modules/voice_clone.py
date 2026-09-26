"""VoiceClone: batch synthesis using a reference voice."""

import os
import time
import logging
import librosa
import soundfile as sf
from config import OUTPUT_DIR
from modules.dataset_io import sanitize_segment, sanitize_text_list_name

_TTS_LANG_MAP = {
    "ja": "japanese", "en": "english", "zh": "chinese", "ko": "korean",
    "de": "german", "fr": "french", "es": "spanish", "it": "italian",
    "pt": "portuguese", "ru": "russian",
}

logger = logging.getLogger(__name__)

# Kept for callers that imported the old private name.
_sanitize_segment = sanitize_segment


def batch_clone(
    manager,
    ref_audio: str,
    ref_text: str,
    texts: list[str],
    output_folder: str = "clone",
    wavs_folder: str = "raw",
    esd_filename: str = "Neutral.txt",
    model_key: str = "1.7B-Base",
    target_sr: int = 44100,
    corpus_lang: str = "ja",
    tts_language: str | None = None,
    lora_path: str | None = None,
    resume: bool = True,
):
    """Clone a voice across all texts. Yields (progress_pct, status_msg) per file,
    then yields (1.0, stats_dict) as the final item."""
    if not ref_audio or not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
    if manager.backend != "irodori" and (not ref_text or not ref_text.strip()):
        raise ValueError("Reference transcript is empty")
    if not texts:
        raise ValueError("No target texts were provided")

    if manager.backend == "irodori":
        yield from _batch_clone_irodori(
            ref_audio=ref_audio, texts=texts,
            output_folder=output_folder, wavs_folder=wavs_folder,
            esd_filename=esd_filename, target_sr=target_sr,
            lora_path=lora_path, resume=resume,
        )
        return

    logger.info(
        "Starting batch clone: model=%s texts=%d output_folder=%s",
        model_key, len(texts), output_folder,
    )
    manager.load_model(model_key)

    # Pre-compute reference voice features (critical optimization)
    prompt_items = manager.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )

    safe_output_folder = sanitize_segment(output_folder, "clone")
    safe_wavs_folder = sanitize_segment(wavs_folder, "raw")
    safe_esd_filename = sanitize_text_list_name(esd_filename)

    base_dir = OUTPUT_DIR / safe_output_folder
    wav_dir = str(base_dir / safe_wavs_folder)
    os.makedirs(wav_dir, exist_ok=True)

    esd_lines = []
    total = len(texts)
    total_duration = 0.0
    start_time = time.time()

    for i, text in enumerate(texts):
        if not text or not text.strip():
            logger.warning("Skipping empty text at index=%d", i)
            continue
        filename = f"{i + 1:04d}.wav"

        try:
            lang_str = tts_language or _TTS_LANG_MAP.get(corpus_lang, "auto")
            wavs, sr = manager.current_model.generate_voice_clone(
                text=text,
                language=lang_str,
                voice_clone_prompt=prompt_items,
            )
        except Exception as e:
            raise RuntimeError(f"Voice clone failed at line {i + 1}: {e}") from e

        audio = wavs[0]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        wav_path = os.path.join(wav_dir, filename)
        sf.write(wav_path, audio, target_sr, subtype="PCM_16")

        stem = os.path.splitext(filename)[0]  # "0001"
        esd_lines.append(f"{stem}|{text}")

        duration = len(wavs[0]) / sr
        total_duration += duration

        elapsed = time.time() - start_time
        avg = elapsed / (i + 1)
        remaining = avg * (total - i - 1)
        yield (i + 1) / total, f"{i + 1}/{total} | ~{remaining:.0f}s left"

    # Write text list (esd.list / Neutral.txt)
    esd_path = str(base_dir / safe_esd_filename)
    with open(esd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(esd_lines))

    logger.info("Batch clone completed: generated=%d output=%s", len(esd_lines), wav_dir)
    yield 1.0, {
        "total_files": len(esd_lines),
        "total_duration_sec": total_duration,
        "output_dir": wav_dir,
        "esd_path": esd_path,
    }


def _batch_clone_irodori(
    *,
    ref_audio: str,
    texts: list[str],
    output_folder: str,
    wavs_folder: str,
    esd_filename: str,
    target_sr: int,
    lora_path: str | None = None,
    resume: bool = True,
):
    """Irodori (v3) variant of batch_clone using the shared resumable runner."""
    from modules.irodori_bridge import get_bridge
    from modules.irodori_jobs import RELEASE_KEEP, run_irodori_batch

    logger.info("Starting Irodori batch clone: texts=%d output_folder=%s", len(texts), output_folder)
    yield from run_irodori_batch(
        bridge=get_bridge(),
        texts=texts,
        base_dir=OUTPUT_DIR / sanitize_segment(output_folder, "clone"),
        wavs_folder=sanitize_segment(wavs_folder, "raw"),
        esd_filename=sanitize_text_list_name(esd_filename),
        request={
            "profile": "legacy",
            "mode": "clone",
            "target_sr": int(target_sr),
            "lora_path": lora_path,
        },
        refs=[ref_audio],
        resume=resume,
        # The legacy screens always kept the worker warm between jobs.
        release_mode=RELEASE_KEEP,
        gpu_label="Irodori一括クローン",
    )


def batch_clone_irodori_v4(
    *,
    ref_audios: list[str],
    texts: list[str],
    caption: str | None = None,
    output_folder: str = "irodori_v4_clone",
    wavs_folder: str = "raw",
    esd_filename: str = "Neutral.txt",
    target_sr: int = 44100,
    model_variant: str = "full",
    model_precision: str = "bf16",
    lora_path: str | None = None,
    seed: int | None = None,
    num_steps: int = 40,
    duration_scale: float = 1.0,
    max_ref_seconds: float = 120.0,
    release_mode: str = "immediate",
    resume: bool = True,
    redo_ids: set[str] | None = None,
    continue_on_error: bool = True,
):
    """Batch clone with Irodori v4 using one or more reference clips.

    This is intentionally separate from the legacy ``batch_clone`` path so
    existing v2/v3 model and LoRA behavior cannot be changed by v4 options.
    ``redo_ids`` regenerates just those line ids (e.g. QC rejects) with a new
    seed while reusing everything else.
    """
    refs = [str(path) for path in ref_audios if path and os.path.isfile(path)]
    if not refs:
        raise FileNotFoundError("V4一括生成には参照音声が1本以上必要です")
    if not texts:
        raise ValueError("No target texts were provided")

    from modules.irodori_bridge import get_bridge
    from modules.irodori_jobs import run_irodori_batch

    request = {
        "profile": "v4",
        "mode": "clone",
        "model_variant": model_variant,
        "model_precision": model_precision,
        "caption": (caption or "").strip() or None,
        "no_ref": False,
        "target_sr": int(target_sr),
        "lora_path": lora_path,
        "num_steps": int(num_steps),
        "duration_scale": float(duration_scale),
        "max_ref_seconds": float(max_ref_seconds),
    }
    for pct, payload in run_irodori_batch(
        bridge=get_bridge(),
        texts=texts,
        base_dir=OUTPUT_DIR / sanitize_segment(output_folder, "irodori_v4_clone"),
        wavs_folder=sanitize_segment(wavs_folder, "raw"),
        esd_filename=sanitize_text_list_name(esd_filename),
        request=request,
        refs=refs,
        settings={"request": request, "seed": seed},
        seed=seed,
        resume=resume,
        redo_ids=redo_ids,
        continue_on_error=continue_on_error,
        release_mode=release_mode,
        gpu_label="Irodori V4一括クローン",
    ):
        if isinstance(payload, dict):
            payload = {**payload, "profile": "v4", "model_variant": model_variant}
        yield pct, payload
