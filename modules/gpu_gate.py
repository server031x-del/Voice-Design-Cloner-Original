"""Process-wide GPU gate shared by Irodori inference, QC ASR and LoRA training.

Only one GPU job runs at a time.  Callers wait for the gate instead of failing
immediately, and generator-based UI handlers can use :func:`wait_messages` to
show "waiting for <job>" while another job is still running.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator


class GPUBusyError(RuntimeError):
    """Raised when the GPU gate could not be acquired in time."""


class GPUGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holder: str | None = None
        self._since: float | None = None

    def try_acquire(self, label: str, timeout: float | None = 0.0) -> bool:
        if timeout is None:
            acquired = self._lock.acquire()
        elif timeout <= 0:
            acquired = self._lock.acquire(blocking=False)
        else:
            acquired = self._lock.acquire(timeout=timeout)
        if acquired:
            self._holder = label
            self._since = time.time()
        return acquired

    def release(self) -> None:
        self._holder = None
        self._since = None
        self._lock.release()

    def busy_label(self) -> str | None:
        """Label of the running job, or None when the GPU is free."""
        if not self._lock.locked():
            return None
        return self._holder or "GPUジョブ"

    def busy_seconds(self) -> float:
        since = self._since
        return 0.0 if since is None else max(0.0, time.time() - since)


_GATE = GPUGate()


def get_gate() -> GPUGate:
    return _GATE


@contextmanager
def gpu_session(label: str = "Irodori推論", *, wait: bool = True, timeout: float | None = None):
    """Hold the GPU for the duration of the ``with`` block.

    ``wait=True`` (default) blocks until the running job finishes; ``timeout``
    bounds that wait.  ``wait=False`` keeps the old fail-fast behavior.
    """
    gate = _GATE
    acquired = gate.try_acquire(label, timeout=(timeout if wait else 0.0))
    if not acquired:
        holder = gate.busy_label() or "別のGPUジョブ"
        raise GPUBusyError(
            f"GPUは「{holder}」が使用中です。終了してから再実行してください。"
        )
    try:
        yield
    finally:
        gate.release()


def wait_messages(label: str, poll_seconds: float = 1.0) -> Iterator[str]:
    """Yield status strings while another job holds the GPU.

    Does not acquire the gate; follow it with ``gpu_session(label)`` (which
    will then usually acquire immediately).
    """
    while True:
        holder = _GATE.busy_label()
        if holder is None:
            return
        elapsed = int(_GATE.busy_seconds())
        yield f"GPU待機中: 「{holder}」の終了を待っています（{elapsed}秒経過）… → {label}"
        time.sleep(poll_seconds)
