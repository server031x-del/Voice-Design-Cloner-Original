"""Irodori-TTS persistent worker.

Runs inside the Irodori venv (located at %USERPROFILE%\\.vdc-engines\\Irodori-TTS\\.venv\\).
The vdc main process launches this via subprocess with CWD set to the Irodori
project root so the ``irodori_tts`` package imports cleanly.

Protocol: newline-delimited JSON on stdin/stdout.

Requests::

    {"op": "synthesize", "mode": "design", "text": "...", "caption": "...",
     "out_path": "...", "seed": null}
    {"op": "synthesize", "mode": "clone",  "text": "...", "ref_wav": "...",
     "caption": null, "out_path": "...", "seed": null,
     "target_sr": 44100}
    {"op": "release_runtime"}
    {"op": "shutdown"}

Responses::

    {"ok": true, "out_path": "...", "sample_rate": 48000, "used_seed": 12345}
    {"ok": false, "error": "..."}
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import fields
from pathlib import Path
from typing import Any

# vdc launches this worker with CWD set to the Irodori-TTS project root so the
# bundled ``irodori_tts`` package is importable. Python doesn't add CWD to
# sys.path automatically when running a script outside of it, so do it here.
sys.path.insert(0, os.getcwd())

# Force stdin/stdout/stderr to UTF-8. The bridge writes UTF-8 JSON on stdin,
# but on Japanese Windows the default Python text-mode encoding is cp932,
# which would mangle non-ASCII characters into surrogates and break the
# HF tokenizer downstream. reconfigure() is available on Python >= 3.7.
sys.stdin.reconfigure(encoding="utf-8", errors="strict")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Irodori (and friends like huggingface_hub / torch / DAC codec) print
# diagnostics, timings, and download progress to stdout. The bridge uses
# stdout as a JSON line protocol, so any rogue print would corrupt the
# protocol. Capture the real stdout for our own use and redirect Python's
# sys.stdout to stderr so every imported library logs there instead.
_PROTOCOL_STDOUT = sys.stdout
_PROTOCOL_LOCK = threading.Lock()
sys.stdout = sys.stderr

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from irodori_precision import load_bf16_runtime  # noqa: E402

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    save_wav,
)

try:  # Added by the upstream v4 runtime.
    from irodori_tts.inference_runtime import download_hf_checkpoint  # noqa: E402
except ImportError:  # Keep legacy v2/v3 usable before the engine is upgraded.
    download_hf_checkpoint = None

CHECKPOINTS = {
    "design": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
    "clone": "Aratako/Irodori-TTS-500M-v3",
}

V4_CHECKPOINTS = {
    "full": "Aratako/Irodori-TTS-v4.1-Small",
    "int8-weight-only": "Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only",
    "int8-dynamic": "Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-dynamic",
    "int4-weight-only": "Aratako/Irodori-TTS-v4.1-Small-Quantized/int4-weight-only",
    "float8-weight-only": "Aratako/Irodori-TTS-v4.1-Small-Quantized/float8-weight-only",
    "float8-dynamic": "Aratako/Irodori-TTS-v4.1-Small-Quantized/float8-dynamic",
    "small-meanflow": "Aratako/Irodori-TTS-v4.1-Small-MF",
    "large-full": "Aratako/Irodori-TTS-v4-Large",
    "large-int8-weight-only": "Aratako/Irodori-TTS-v4-Large-Quantized/int8-weight-only",
    "large-int8-dynamic": "Aratako/Irodori-TTS-v4-Large-Quantized/int8-dynamic",
    "large-int4-weight-only": "Aratako/Irodori-TTS-v4-Large-Quantized/int4-weight-only",
    "large-float8-weight-only": "Aratako/Irodori-TTS-v4-Large-Quantized/float8-weight-only",
    "large-float8-dynamic": "Aratako/Irodori-TTS-v4-Large-Quantized/float8-dynamic",
}
FULL_PRECISION_VARIANTS = {"full", "small-meanflow", "large-full"}

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"


class DownloadProgress(tqdm):
    """Send byte progress over the worker protocol even without a terminal."""

    def display(self, msg=None, pos=None):
        if getattr(self, "unit", None) != "B":
            return
        now = time.monotonic()
        if now - getattr(self, "_last_protocol_update", 0.0) < 1.0:
            return
        self._last_protocol_update = now
        total = getattr(self, "total", 0) or 0
        downloaded = getattr(self, "n", 0)
        if total <= 0 and downloaded <= 0:
            return
        phase = "モデル取得" if "Downloading" in (self.desc or "") else "モデルファイル構築"
        message = f"{phase}: {downloaded / 1024**3:.2f} GiB"
        if total > 0:
            message += f" / {total / 1024**3:.2f} GiB（{min(downloaded / total, 1.0):.0%}）"
        _emit({"event": "progress", "message": message,
               "fraction": min(downloaded / total, 1.0) if total > 0 else None})


def _download_v4_checkpoint(source: str) -> str:
    owner, repo, *subfolder = source.split("/")
    checkpoint = f"{subfolder[0]}/model.safetensors" if subfolder else "model.safetensors"
    patterns = [checkpoint, "tokenizer/*"]
    if subfolder:
        patterns.append(f"{subfolder[0]}/tokenizer/*")
    snapshot = Path(snapshot_download(repo_id=f"{owner}/{repo}", allow_patterns=patterns,
                                      tqdm_class=DownloadProgress))
    path = snapshot / checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path}")
    return str(path)


def _log(message: str) -> None:
    print(f"[Irodori] {message}", file=sys.stderr, flush=True)


def _cleanup_cuda_memory(label: str) -> dict[str, float | bool]:
    """Return unused CUDA allocator blocks to the driver.

    This deliberately keeps live model tensors allocated.  Call
    ``WorkerState.release_runtime`` first when the model itself must also leave
    the GPU.
    """
    import torch

    if not torch.cuda.is_available():
        gc.collect()
        return {"cuda_available": False, "allocated_mib": 0.0, "reserved_mib": 0.0}

    try:
        torch.cuda.synchronize()
    except Exception as exc:
        _log(f"CUDA synchronize during cleanup failed: {exc}")
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        _log(f"CUDA empty_cache failed: {exc}")
    snapshot: dict[str, float | bool] = {
        "cuda_available": True,
        "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / 1024**2, 1),
    }
    _log(
        f"CUDA cleanup ({label}): allocated={snapshot['allocated_mib']:.1f} MiB, "
        f"reserved={snapshot['reserved_mib']:.1f} MiB."
    )
    return snapshot


class WorkerState:
    def __init__(self) -> None:
        self.runtime: InferenceRuntime | None = None
        self.runtime_profile: tuple[str, str, str, str] | None = None

    def ensure_runtime(
        self,
        *,
        mode: str,
        profile: str,
        model_variant: str,
        model_precision: str | None,
    ) -> InferenceRuntime:
        if profile == "v4":
            if model_variant not in V4_CHECKPOINTS:
                raise ValueError(
                    f"unknown Irodori v4 model variant: {model_variant}; "
                    f"choose from {', '.join(V4_CHECKPOINTS)}"
                )
            if download_hf_checkpoint is None:
                raise RuntimeError(
                    "The installed Irodori-TTS engine predates v4. Update "
                    "%USERPROFILE%\\.vdc-engines\\Irodori-TTS and run uv sync --extra cu128."
                )
            repo = V4_CHECKPOINTS[model_variant]
            precision = (
                (model_precision or "bf16")
                if model_variant in FULL_PRECISION_VARIANTS
                else "bf16"
            )
            if "int4" in model_variant or "float8" in model_variant:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError(f"{model_variant} requires an NVIDIA CUDA GPU")
                major, minor = torch.cuda.get_device_capability()
                capability = major + minor / 10.0
                minimum = 8.9 if "float8" in model_variant else 8.0
                if capability < minimum:
                    raise RuntimeError(
                        f"{model_variant} requires compute capability {minimum:.1f} or newer; "
                        f"the selected GPU reports {major}.{minor}"
                    )
            cache_profile = (profile, model_variant, precision, repo)
        else:
            if mode not in CHECKPOINTS:
                raise ValueError(f"unknown legacy mode: {mode}")
            repo = CHECKPOINTS[mode]
            precision = model_precision or "fp32"
            cache_profile = ("legacy", mode, precision, repo)

        if self.runtime is not None and self.runtime_profile == cache_profile:
            _log(
                f"Runtime already loaded for profile={profile}, "
                f"variant={model_variant}, mode={mode}."
            )
            return self.runtime
        if self.runtime is not None:
            self.release_runtime(reason="model switch")
        _log(
            f"Preparing runtime for profile={profile}, variant={model_variant}, "
            f"mode={mode}, precision={precision}."
        )
        _log(f"Checking/downloading checkpoint: {repo}")
        _emit({"event": "progress", "message": f"モデル取得を確認しています: {repo}"})
        if profile == "v4":
            ckpt = _download_v4_checkpoint(repo)
        else:
            ckpt = hf_hub_download(repo_id=repo, filename="model.safetensors")
        device = default_runtime_device()
        _log(f"Loading Irodori runtime on device={device}.")
        _emit({"event": "progress", "message": f"モデルを{device}へ読み込んでいます（{precision}）"})
        key = RuntimeKey(
                checkpoint=ckpt,
                model_device=device,
                codec_repo=CODEC_REPO,
                model_precision=precision,
                codec_device=device,
                codec_precision="fp32",
                codec_deterministic_encode=True,
                codec_deterministic_decode=True,
                compile_model=False,
                compile_dynamic=False,
            )
        if profile == "v4" and model_variant in FULL_PRECISION_VARIANTS and precision == "bf16":
            from irodori_tts.model import TextToLatentRFDiT
            self.runtime = load_bf16_runtime(InferenceRuntime.from_key, key, TextToLatentRFDiT)
        else:
            self.runtime = InferenceRuntime.from_key(key)
        self.runtime_profile = cache_profile
        _log(f"Runtime ready for profile={cache_profile}.")
        return self.runtime

    def release_runtime(self, *, reason: str) -> tuple[bool, dict[str, float | bool]]:
        """Drop the active runtime and release its model/codec CUDA tensors."""
        runtime = self.runtime
        runtime_profile = self.runtime_profile
        self.runtime = None
        self.runtime_profile = None

        released = runtime is not None
        if runtime is not None:
            _log(f"Unloading runtime for profile={runtime_profile} ({reason}).")
            unload = getattr(runtime, "unload", None)
            try:
                if callable(unload):
                    unload()
            except Exception as exc:
                # Cleanup must not hide the original synthesis/model-switch
                # outcome. Dropping the final runtime reference below still
                # lets Python reclaim its tensors.
                _log(f"Runtime unload hook failed: {type(exc).__name__}: {exc}")
            del runtime

        memory = _cleanup_cuda_memory(reason)
        return released, memory


def _save_wav_resampled(path: Path, audio, src_sr: int, target_sr: int | None) -> None:
    if target_sr is None or target_sr == src_sr:
        save_wav(path, audio, src_sr)
        return
    # Resample via torchaudio (already a direct dep of Irodori-TTS) rather
    # than librosa, which isn't pinned in Irodori's pyproject.
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    wav = audio.squeeze().detach().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    resampled = torchaudio.functional.resample(
        wav, orig_freq=int(src_sr), new_freq=int(target_sr),
    )
    sf.write(str(path), resampled.squeeze(0).numpy().astype(np.float32), int(target_sr))


def handle_synthesize(state: WorkerState, req: dict[str, Any]) -> dict[str, Any]:
    mode = req["mode"]
    profile = str(req.get("profile") or "legacy").lower()
    if profile not in {"legacy", "v4"}:
        return {"ok": False, "error": f"unknown profile: {profile}"}
    if profile == "legacy" and mode not in CHECKPOINTS:
        return {"ok": False, "error": f"unknown mode: {mode}"}

    model_variant = str(req.get("model_variant") or "full")
    _log(f"Synthesis requested: profile={profile}, mode={mode}, variant={model_variant}.")
    runtime = state.ensure_runtime(
        mode=mode,
        profile=profile,
        model_variant=model_variant,
        model_precision=req.get("model_precision"),
    )

    text = req["text"]
    out_path = Path(req["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_sr = req.get("target_sr")

    is_v4 = profile == "v4"
    ref_wavs = [str(path) for path in (req.get("ref_wavs") or []) if path]
    sampling_kwargs = dict(
        text=text,
        caption=req.get("caption"),
        ref_wav=req.get("ref_wav") if (mode == "clone" or is_v4) else None,
        ref_wavs=ref_wavs or None,
        ref_latent=None,
        no_ref=bool(req.get("no_ref", mode == "design")),
        ref_normalize_db=-16.0,
        ref_ensure_max=True,
        num_candidates=1,
        decode_mode="sequential",
        # v3 (clone) has an auto Duration Predictor and prefers seconds=None.
        # v2-VoiceDesign (design) does not, so fix it at 30.0 like
        # master/design/voice_design.py.
        seconds=(30.0 if profile == "legacy" and mode == "design" else None),
        duration_scale=float(req.get("duration_scale", 1.0)),
        max_seconds=float(req.get("max_seconds", 30.0)),
        max_ref_seconds=float(req.get("max_ref_seconds", 120.0 if is_v4 else 30.0)),
        lora_adapter=req.get("lora_path"),
        max_text_len=None,
        max_caption_len=None,
        num_steps=int(
            req.get("num_steps")
            if req.get("num_steps") is not None
            else (4 if model_variant == "small-meanflow" else 40)
        ),
        cfg_scale_text=float(req.get("cfg_scale_text", 3.0)),
        cfg_scale_caption=float(req.get("cfg_scale_caption", 3.0)),
        cfg_scale_speaker=float(req.get("cfg_scale_speaker", 5.0)),
        cfg_guidance_mode="independent",
        cfg_scale=None,
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        truncation_factor=None,
        rescale_k=None,
        rescale_sigma=None,
        context_kv_cache=True,
        speaker_kv_scale=None,
        speaker_kv_min_t=None,
        speaker_kv_max_layers=None,
        seed=req.get("seed"),
        trim_tail=True,
        tail_window_size=20,
        tail_std_threshold=0.05,
        tail_mean_threshold=0.1,
    )

    supported_fields = {field.name for field in fields(SamplingRequest)}
    if is_v4 and "ref_wavs" not in supported_fields:
        raise RuntimeError(
            "The installed Irodori-TTS runtime does not expose v4 multi-reference inference. "
            "Update the engine before using the Irodori V4 screen."
        )
    sampling = SamplingRequest(**{
        key: value for key, value in sampling_kwargs.items() if key in supported_fields
    })

    release_after = bool(req.get("release_after_synthesis", False))
    result = None
    response: dict[str, Any] | None = None
    try:
        _log("Generating audio...")
        _emit({"event": "progress", "message": "音声を生成しています"})
        result = runtime.synthesize(sampling, log_fn=lambda msg: _log(str(msg)))
        _log(f"Saving wav: {out_path}")
        _save_wav_resampled(out_path, result.audio, result.sample_rate, target_sr)
        response = {
            "ok": True,
            "out_path": str(out_path),
            "sample_rate": target_sr or result.sample_rate,
            "used_seed": result.used_seed,
            "duration_sec": round(float(result.audio.shape[-1]) / float(result.sample_rate), 3),
            "max_seconds": getattr(sampling, "max_seconds", None),
            "profile": profile,
            "model_variant": model_variant,
        }
        _log("Synthesis complete.")
    finally:
        # Sampling tensors are request-scoped.  Drop the returned object before
        # empty_cache() so its storage cannot keep allocator blocks alive.
        result = None
        if release_after:
            # The local reference would otherwise keep the runtime alive until
            # this function returns, defeating the explicit unload.
            runtime = None
            released, memory = state.release_runtime(reason="post-synthesis release")
        else:
            released = False
            memory = _cleanup_cuda_memory("post-synthesis cache trim")
        if response is not None:
            response["runtime_released"] = released
            response["gpu_memory"] = memory

    assert response is not None
    return response


def _emit(payload: dict[str, Any]) -> None:
    with _PROTOCOL_LOCK:
        _PROTOCOL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _PROTOCOL_STDOUT.flush()


def main() -> int:
    state = WorkerState()
    _emit({"ok": True, "event": "ready"})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"invalid json: {exc}"})
            continue

        op = req.get("op")
        if op == "shutdown":
            released, memory = state.release_runtime(reason="worker shutdown")
            _emit({"ok": True, "event": "shutdown", "runtime_released": released,
                   "gpu_memory": memory})
            return 0
        if op == "release_runtime":
            released, memory = state.release_runtime(reason="explicit release")
            _emit({"ok": True, "event": "runtime_released", "runtime_released": released,
                   "gpu_memory": memory})
            continue
        if op == "synthesize":
            try:
                _emit(handle_synthesize(state, req))
            except Exception as exc:
                _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                       "trace": traceback.format_exc()})
            continue
        _emit({"ok": False, "error": f"unknown op: {op}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
