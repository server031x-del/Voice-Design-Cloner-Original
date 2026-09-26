"""Small shared helpers for dataset folders (wavs + text list) and manifests.

Used by the Irodori batch runner, the QC pass, the LoRA pipeline and the local
audition scripts so that text-list parsing, atomic writes and audio stats are
implemented once.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
# vdc text lists look like ``0001|テキスト``.
_ESD_LINE_RE = re.compile(r"^(?P<id>[^|]+)\|(?P<text>.+)$")


def sanitize_segment(value: str, default: str) -> str:
    """Sanitize an untrusted folder/file segment for safe relative-path use."""
    segment = (value or "").strip().replace("\\", "/")
    segment = segment.split("/")[-1]  # basename only
    segment = _SAFE_SEGMENT_RE.sub("_", segment).strip("._")
    return segment or default


def sanitize_text_list_name(value: str, default: str = "Neutral.txt") -> str:
    name = sanitize_segment(value, default)
    return name if name.endswith(".txt") else name + ".txt"


def atomic_write_text(path: str | Path, content: str) -> None:
    """Write via a temp file + rename so readers never see a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_stats(path: str | Path) -> dict[str, object]:
    """Basic, reproducible facts about a rendered wav."""
    path = Path(path)
    info = sf.info(str(path))
    audio, _ = sf.read(str(path), dtype="float32")
    samples = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.duration), 3),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def read_text_list(path: str | Path) -> list[tuple[str, str]]:
    """Parse a ``id|text`` list, skipping blank or malformed lines."""
    entries: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _ESD_LINE_RE.match(line)
        if not match:
            continue
        entries.append((match.group("id").strip(), match.group("text").strip()))
    return entries


def write_text_list(path: str | Path, entries: Iterable[tuple[str, str]]) -> None:
    atomic_write_text(path, "\n".join(f"{file_id}|{text}" for file_id, text in entries))


def find_text_list(folder: str | Path) -> Path | None:
    """The clone tab saves a text list (default Neutral.txt) at the folder root."""
    folder = Path(folder)
    for candidate in (folder / "Neutral.txt", folder / "esd.list"):
        if candidate.is_file():
            return candidate
    # QC backups and reports are not training text lists.
    txts = [
        p for p in folder.glob("*.txt")
        if p.is_file() and not p.name.endswith(".all.txt") and not p.name.startswith("qc_")
    ]
    if len(txts) == 1:
        return txts[0]
    return None
