"""Bridge between vdc and the Irodori-TTS worker subprocess.

Irodori cannot share a Python process with the rest of vdc (it needs a different
torch / sentencepiece). We launch ``modules/irodori_worker.py`` inside Irodori's
own venv as a persistent subprocess and exchange newline-delimited JSON on
stdin/stdout.

Public surface:

    IrodoriBridge.is_available()   -> bool
    IrodoriBridge.ensure_started() -> None
    IrodoriBridge.synthesize(...)  -> dict      (synchronous, one-shot)
    IrodoriBridge.schedule_idle_release(sec) -> None
    IrodoriBridge.shutdown()       -> None

Every request is guarded by an *inactivity* timeout: as long as the worker keeps
printing progress to stderr (model downloads, sampling logs) a request may run
as long as it needs, but a worker that goes silent for ``VDC_IRODORI_TIMEOUT``
seconds (default 900) is killed so the UI and the GPU gate never hang forever.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def _default_irodori_root() -> Path:
    """Return the directory where setup.bat / setup.sh installed Irodori-TTS."""
    if sys.platform == "win32":
        base = Path(os.environ.get("USERPROFILE", str(Path.home())))
    else:
        base = Path.home()
    return base / ".vdc-engines" / "Irodori-TTS"


def _worker_script_path() -> Path:
    return Path(__file__).resolve().parent / "irodori_worker.py"


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(1.0, float(raw))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, raw)
        return default


DEFAULT_INACTIVITY_TIMEOUT = _env_seconds("VDC_IRODORI_TIMEOUT", 900.0)
DEFAULT_IDLE_RELEASE_SECONDS = _env_seconds("VDC_IRODORI_IDLE_SECONDS", 300.0)

_EOF = object()


class IrodoriUnavailable(RuntimeError):
    pass


class IrodoriTimeout(IrodoriUnavailable):
    """The worker stopped responding and was terminated."""


class IrodoriBridge:
    """Manages a persistent Irodori worker subprocess."""

    def __init__(
        self,
        irodori_root: Path | None = None,
        *,
        inactivity_timeout: float | None = None,
    ) -> None:
        self.irodori_root = irodori_root or _default_irodori_root()
        self.inactivity_timeout = inactivity_timeout or DEFAULT_INACTIVITY_TIMEOUT
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        # Bounded so a long-running worker that prints a lot of progress lines
        # doesn't leak memory.
        self._stderr_buf: deque[str] = deque(maxlen=200)
        self._stderr_thread: threading.Thread | None = None
        self._stdout_queue: queue.Queue = queue.Queue()
        self._last_activity = time.monotonic()
        self._idle_deadline: float | None = None
        self._idle_thread: threading.Thread | None = None
        self._idle_guard = threading.Lock()

    # ------------------------------------------------------------------ availability

    def is_available(self) -> bool:
        """True iff Irodori was installed by setup and looks runnable."""
        py = self._worker_python()
        return self.irodori_root.is_dir() and py.is_file() and _worker_script_path().is_file()

    def _worker_python(self) -> Path:
        if sys.platform == "win32":
            return self.irodori_root / ".venv" / "Scripts" / "python.exe"
        return self.irodori_root / ".venv" / "bin" / "python"

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status_text(self) -> str:
        if self.is_running():
            return "running"
        if self.is_available():
            return "ready (not started)"
        return "not installed"

    # ------------------------------------------------------------------ lifecycle

    def ensure_started(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            if not self.is_available():
                raise IrodoriUnavailable(
                    f"Irodori-TTS is not installed at {self.irodori_root}. "
                    "Re-run setup.bat / setup.sh."
                )
            py = self._worker_python()
            script = _worker_script_path()
            logger.info("Starting Irodori worker: %s %s (cwd=%s)", py, script, self.irodori_root)
            self._proc = subprocess.Popen(
                [str(py), str(script)],
                cwd=str(self.irodori_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self._stderr_buf.clear()
            self._stdout_queue = queue.Queue()
            self._last_activity = time.monotonic()
            self._stderr_thread = threading.Thread(
                target=self._stderr_pump, args=(self._proc.stderr,), daemon=True,
            )
            self._stderr_thread.start()
            threading.Thread(
                target=self._stdout_pump,
                args=(self._proc.stdout, self._stdout_queue),
                daemon=True,
            ).start()
            ready = self._wait_response()
            if not ready.get("ok") or ready.get("event") != "ready":
                err = self._collected_stderr()
                self._kill_locked()
                raise IrodoriUnavailable(f"Irodori worker did not start cleanly: {ready!r} stderr={err!r}")

    def shutdown(self) -> None:
        self.cancel_idle_release()
        with self._lock:
            if self._proc is None:
                return
            try:
                if self._proc.poll() is None:
                    try:
                        self._proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                        self._proc.stdin.flush()
                    except Exception:
                        pass
                    try:
                        self._proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
            finally:
                self._proc = None

    # ------------------------------------------------------------------ I/O

    @staticmethod
    def _stdout_pump(stream, out_queue: queue.Queue) -> None:
        """Parse protocol lines on a thread so that reads can time out."""
        try:
            for raw in stream:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out_queue.put(json.loads(raw))
                except json.JSONDecodeError as exc:
                    logger.warning("invalid worker output: %r (%s)", raw, exc)
        except Exception:
            pass
        finally:
            out_queue.put(_EOF)

    def _stderr_pump(self, stream) -> None:
        try:
            for line in stream:
                self._last_activity = time.monotonic()
                self._stderr_buf.append(line)
                logger.warning("[irodori_worker] %s", line.rstrip())
        except Exception:
            pass

    def _collected_stderr(self) -> str:
        return "".join(self._stderr_buf)

    def _kill_locked(self) -> None:
        """Terminate the worker. Caller must hold ``self._lock``."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        except Exception:
            logger.exception("Failed to kill Irodori worker")

    def _wait_response(self) -> dict[str, Any]:
        """Wait for the next protocol message, enforcing the inactivity timeout.

        Caller must hold ``self._lock``.
        """
        self._last_activity = time.monotonic()
        while True:
            try:
                item = self._stdout_queue.get(timeout=0.5)
            except queue.Empty:
                idle = time.monotonic() - self._last_activity
                if idle > self.inactivity_timeout:
                    err = self._collected_stderr()[-2000:]
                    self._kill_locked()
                    raise IrodoriTimeout(
                        f"Irodoriワーカーが{int(idle)}秒間応答しなかったため停止しました"
                        f"（VDC_IRODORI_TIMEOUT で調整できます）。stderr末尾={err!r}"
                    )
                continue
            if item is _EOF:
                err = self._collected_stderr()
                self._kill_locked()
                raise IrodoriUnavailable(f"Irodori worker exited without a response. stderr={err!r}")
            return item

    def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise IrodoriUnavailable("Irodori worker is not running")
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            return self._wait_response()

    # ------------------------------------------------------------------ idle release

    def schedule_idle_release(self, seconds: float | None = None) -> None:
        """Shut the worker down after ``seconds`` without new requests.

        Keeps the model warm for quick re-rolls while still freeing VRAM (and
        the CUDA context) once the user stops generating.
        """
        delay = DEFAULT_IDLE_RELEASE_SECONDS if seconds is None else float(seconds)
        with self._idle_guard:
            self._idle_deadline = time.monotonic() + delay
            if self._idle_thread is None or not self._idle_thread.is_alive():
                self._idle_thread = threading.Thread(target=self._idle_monitor, daemon=True)
                self._idle_thread.start()

    def cancel_idle_release(self) -> None:
        with self._idle_guard:
            self._idle_deadline = None

    def idle_release_pending(self) -> bool:
        return self._idle_deadline is not None

    def _idle_monitor(self) -> None:
        from modules.gpu_gate import get_gate

        gate = get_gate()
        while True:
            time.sleep(2.0)
            with self._idle_guard:
                deadline = self._idle_deadline
                if deadline is None:
                    self._idle_thread = None
                    return
            if time.monotonic() < deadline:
                continue
            if not self.is_running():
                self.cancel_idle_release()
                continue
            # Never pull the worker out from under an active GPU job.
            if not gate.try_acquire("アイドル解放", timeout=0):
                continue
            try:
                with self._idle_guard:
                    due = self._idle_deadline is not None and time.monotonic() >= self._idle_deadline
                if due:
                    logger.info("Irodori worker idle; releasing GPU memory.")
                    self.shutdown()
            except Exception:
                logger.exception("Idle release of Irodori worker failed")
            finally:
                gate.release()

    # ------------------------------------------------------------------ ops

    def synthesize(
        self,
        *,
        mode: str,
        text: str,
        out_path: str | Path,
        profile: str = "legacy",
        model_variant: str = "full",
        model_precision: str | None = None,
        caption: str | None = None,
        ref_wav: str | Path | None = None,
        ref_wavs: Sequence[str | Path] | None = None,
        no_ref: bool | None = None,
        seed: int | None = None,
        target_sr: int | None = None,
        lora_path: str | Path | None = None,
        num_steps: int | None = None,
        duration_scale: float | None = None,
        max_ref_seconds: float | None = None,
        max_seconds: float | None = None,
        cfg_scale_text: float | None = None,
        cfg_scale_caption: float | None = None,
        cfg_scale_speaker: float | None = None,
        release_after_synthesis: bool | None = None,
    ) -> dict[str, Any]:
        """Run one synthesis. Starts worker if needed. Synchronous."""
        self.cancel_idle_release()
        self.ensure_started()
        req = {
            "op": "synthesize",
            "mode": mode,
            "profile": profile,
            "model_variant": model_variant,
            "text": text,
            "out_path": str(out_path),
            "caption": caption,
            "seed": seed,
            "target_sr": target_sr,
        }
        optional_values = {
            "model_precision": model_precision,
            "no_ref": no_ref,
            "num_steps": num_steps,
            "duration_scale": duration_scale,
            "max_ref_seconds": max_ref_seconds,
            "max_seconds": max_seconds,
            "cfg_scale_text": cfg_scale_text,
            "cfg_scale_caption": cfg_scale_caption,
            "cfg_scale_speaker": cfg_scale_speaker,
            "release_after_synthesis": release_after_synthesis,
        }
        req.update({key: value for key, value in optional_values.items() if value is not None})
        if ref_wav is not None:
            req["ref_wav"] = str(ref_wav)
        if ref_wavs:
            req["ref_wavs"] = [str(path) for path in ref_wavs]
        if lora_path is not None:
            req["lora_path"] = str(lora_path)
        resp = self._send(req)
        if not resp.get("ok"):
            msg = resp.get("error") or "Irodori worker reported failure"
            trace = resp.get("trace")
            if trace:
                logger.error("Irodori worker traceback:\n%s", trace)
            raise RuntimeError(msg)
        return resp

    def release_runtime(self) -> dict[str, Any]:
        """Unload the active Irodori model while keeping the worker available."""
        if not self.is_running():
            return {
                "ok": True,
                "event": "runtime_released",
                "runtime_released": False,
                "gpu_memory": None,
            }
        resp = self._send({"op": "release_runtime"})
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error") or "Irodori runtime release failed")
        return resp


# A module-level singleton so the same worker survives across UI calls.
_bridge: IrodoriBridge | None = None


def get_bridge() -> IrodoriBridge:
    global _bridge
    if _bridge is None:
        _bridge = IrodoriBridge()
        atexit.register(_bridge.shutdown)
    return _bridge
