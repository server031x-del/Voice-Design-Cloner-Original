"""VoiceDesign: generate a new voice from a text prompt."""

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
import soundfile as sf
from config import VOICE_DESIGN_DIR, TTS_LANG

TTS_LANGUAGES = [
    "japanese", "english", "chinese", "korean",
    "german", "french", "spanish", "italian",
    "portuguese", "russian",
]


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_filename(name: str, default: str = "voice_design") -> str:
    """Convert arbitrary user input into a safe filename stem."""
    candidate = (name or "").strip()
    candidate = candidate.replace("\\", "_").replace("/", "_")
    candidate = _SAFE_NAME_RE.sub("_", candidate).strip("._")
    return candidate or default


def generate_voice_design(manager, text: str, instruct: str, language: str | None = None, **kwargs):
    """Generate a voice with VoiceDesign model. Returns (sample_rate, audio_array)."""
    if manager.backend == "irodori":
        return _generate_voice_design_irodori(text, instruct)
    manager.load_model("1.7B-VoiceDesign")
    wavs, sr = manager.current_model.generate_voice_design(
        text=text,
        language=language or TTS_LANG,
        instruct=instruct,
        **kwargs,
    )
    return sr, wavs[0]


def _generate_voice_design_irodori(text: str, caption: str):
    """Run the Irodori worker once for VoiceDesign and return (sr, audio)."""
    from modules.gpu_gate import gpu_session
    from modules.irodori_bridge import get_bridge
    bridge = get_bridge()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        with gpu_session("Irodori VoiceDesign"):
            bridge.synthesize(mode="design", text=text, caption=caption, out_path=out_path)
        audio, sr = sf.read(out_path, dtype="float32")
        return sr, audio
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def generate_clone_oneshot_irodori(
    *,
    text: str,
    ref_wav: str,
    caption: str | None = None,
    lora_path: str | None = None,
    seed: int | None = None,
):
    """One-shot Irodori clone (mode=clone, single utterance) for the Inference tab.

    Returns (sample_rate, audio_array). The worker writes to a temp wav which
    we then read back as numpy so it plugs into gradio's preview directly.
    Long text is split into sentence chunks instead of being cut at 30 s.
    """
    from modules.gpu_gate import gpu_session
    from modules.irodori_bridge import get_bridge
    from modules.irodori_jobs import synthesize_to_file
    bridge = get_bridge()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        with gpu_session("Irodori推論"):
            synthesize_to_file(
                bridge,
                text=text,
                out_path=out_path,
                refs=[ref_wav],
                seed=seed,
                mode="clone",
                caption=caption,
                lora_path=lora_path,
            )
        audio, sr = sf.read(out_path, dtype="float32")
        return sr, audio
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def generate_irodori_v4(
    *,
    text: str,
    caption: str | None = None,
    ref_wavs: list[str] | None = None,
    model_variant: str = "full",
    model_precision: str = "bf16",
    lora_path: str | None = None,
    seed: int | None = None,
    num_steps: int = 40,
    duration_scale: float = 1.0,
    max_ref_seconds: float = 120.0,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    release_mode: str = "idle",
    progress_callback=None,
):
    """Generate with the unified Irodori v4 text/ref/caption checkpoint.

    ``ref_wavs`` may contain multiple clips from one speaker.  With an empty
    list the request becomes pure Voice Design (text + optional caption).
    ``release_mode`` is ``"immediate"`` (stop the worker now), ``"idle"``
    (stop it after a few idle minutes) or ``"keep"``.

    Returns ``(sample_rate, audio_array, info)``; ``info`` holds the used seed
    and every setting needed to reproduce the clip.
    """
    if not text or not text.strip():
        raise ValueError("読み上げテキストを入力してください")

    refs = [str(path) for path in (ref_wavs or []) if path]
    from modules.gpu_gate import gpu_session
    from modules.irodori_bridge import get_bridge
    from modules.irodori_jobs import finish_release, synthesize_to_file

    bridge = get_bridge()
    settings = {
        "profile": "v4",
        "mode": "clone" if refs else "design",
        "model_variant": model_variant,
        "model_precision": model_precision,
        "caption": (caption or "").strip() or None,
        "lora_path": lora_path,
        "num_steps": int(num_steps),
        "duration_scale": float(duration_scale),
        "max_ref_seconds": float(max_ref_seconds),
        "cfg_scale_text": float(cfg_scale_text),
        "cfg_scale_caption": float(cfg_scale_caption),
        "cfg_scale_speaker": float(cfg_scale_speaker),
    }
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        with gpu_session("Irodori V4生成"):
            try:
                info = synthesize_to_file(
                    bridge,
                    text=text.strip(),
                    out_path=out_path,
                    refs=refs,
                    seed=seed,
                    no_ref=not refs,
                    release_after_synthesis=(release_mode == "immediate"),
                    progress_callback=progress_callback,
                    **settings,
                )
            finally:
                # Stopping the worker also destroys the CUDA context, which
                # releases the driver residue left after runtime.unload().
                finish_release(bridge, release_mode)
        audio, sr = sf.read(out_path, dtype="float32")
        info = {
            **info,
            "seed_requested": seed,
            "text": text.strip(),
            "ref_wavs": refs,
            "settings": settings,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        return sr, audio, info
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def save_voice(audio_tuple, name: str, sample_text: str = "", metadata: dict | None = None) -> str:
    """Save a Voice Design generation under ``output/voice_design/``.

    ``metadata`` (seed, caption, model, CFG...) is written next to the wav as
    ``{name}.json`` so the voice can be regenerated later.
    """
    return _save_named_clip(audio_tuple, name, sample_text, VOICE_DESIGN_DIR, "voice_design", metadata)


def save_irodori_infer(audio_tuple, name: str, sample_text: str = "") -> str:
    """Save an Irodori Inference one-shot generation under ``output/irodori_infer/``."""
    from config import OUTPUT_DIR
    target_dir = OUTPUT_DIR / "irodori_infer"
    return _save_named_clip(audio_tuple, name, sample_text, target_dir, "irodori_infer")


def save_irodori_v4(audio_tuple, name: str, sample_text: str = "", metadata: dict | None = None) -> str:
    """Save a unified v4 generation without mixing it into legacy outputs."""
    from config import OUTPUT_DIR
    target_dir = OUTPUT_DIR / "irodori_v4"
    return _save_named_clip(audio_tuple, name, sample_text, target_dir, "irodori_v4", metadata)


def _save_named_clip(
    audio_tuple,
    name: str,
    sample_text: str,
    target_dir: Path,
    default: str,
    metadata: dict | None = None,
) -> str:
    sr, audio = audio_tuple
    os.makedirs(target_dir, exist_ok=True)
    safe_name = _sanitize_filename(name, default=default)
    dest = str(target_dir / f"{safe_name}.wav")
    if os.path.exists(dest):
        base, ext = os.path.splitext(dest)
        i = 1
        while os.path.exists(f"{base}_{i}{ext}"):
            i += 1
        dest = f"{base}_{i}{ext}"
    sf.write(dest, audio, sr, subtype="PCM_16")
    txt_path = os.path.splitext(dest)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    if metadata:
        meta_path = os.path.splitext(dest)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(Path(dest).resolve())


def get_kept_voice_metadata_by_label(label: str) -> dict | None:
    """Return the generation settings saved with a Voice Design, if any."""
    if not label:
        return None
    p = VOICE_DESIGN_DIR / f"{label}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_kept_voice_labels_with_metadata() -> list[str]:
    """Saved Voice Design names that carry reproducible Irodori generation settings."""
    os.makedirs(VOICE_DESIGN_DIR, exist_ok=True)
    labels = []
    for path in VOICE_DESIGN_DIR.glob("*.json"):
        if not (VOICE_DESIGN_DIR / f"{path.stem}.wav").is_file():
            continue
        metadata = get_kept_voice_metadata_by_label(path.stem) or {}
        # Gemini-made references have no Irodori settings to load back.
        if "settings" in metadata:
            labels.append(path.stem)
    return sorted(labels)


def list_kept_voices() -> list[str]:
    """List all kept voice files (full paths)."""
    os.makedirs(VOICE_DESIGN_DIR, exist_ok=True)
    return sorted(str(p) for p in VOICE_DESIGN_DIR.glob("*.wav"))


def list_kept_voice_labels() -> list[str]:
    """List saved Voice Design names."""
    os.makedirs(VOICE_DESIGN_DIR, exist_ok=True)
    return sorted(p.stem for p in VOICE_DESIGN_DIR.glob("*.wav"))


def get_kept_voice_path_by_label(label: str) -> str | None:
    """Get wav path for a saved Voice Design name."""
    if not label:
        return None
    p = VOICE_DESIGN_DIR / f"{label}.wav"
    return str(p) if p.exists() else None


def get_kept_voice_text_by_label(label: str) -> str:
    """Get transcript text for a saved Voice Design name."""
    if not label:
        return ""
    p = VOICE_DESIGN_DIR / f"{label}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return ""


def list_kept_voice_numbers() -> list[int]:
    """List voice_design numbers from voice_design_*.wav files."""
    os.makedirs(VOICE_DESIGN_DIR, exist_ok=True)
    nums = []
    for p in VOICE_DESIGN_DIR.glob("voice_design*.wav"):
        m = re.search(r"voice_design_?(\d+)?\.wav$", p.name)
        if m:
            nums.append(int(m.group(1)) if m.group(1) else 0)
        elif p.name == "voice_design.wav":
            nums.append(0)
    return sorted(nums)


def get_kept_voice_path(num: int) -> str:
    """Get wav path for a voice_design number."""
    if num == 0:
        p = VOICE_DESIGN_DIR / "voice_design.wav"
    else:
        p = VOICE_DESIGN_DIR / f"voice_design_{num}.wav"
    return str(p) if p.exists() else None


def get_kept_voice_text(num: int) -> str:
    """Get transcript text for a voice_design number."""
    if num == 0:
        p = VOICE_DESIGN_DIR / "voice_design.txt"
    else:
        p = VOICE_DESIGN_DIR / f"voice_design_{num}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return ""
