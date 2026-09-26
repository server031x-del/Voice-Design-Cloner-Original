"""Build a small, data-free source bundle for the Colab notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output" / "colab" / "voice_design_cloner_colab_source.zip"

TOP_LEVEL_FILES = (
    "app.py",
    "app.sh",
    "config.py",
    "config.example.json",
    "lang.py",
    "requirements.txt",
    "setup.sh",
    "README.md",
    "README.en.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_LICENSES.md",
)


def collect_files() -> list[Path]:
    files = [ROOT / name for name in TOP_LEVEL_FILES if (ROOT / name).is_file()]
    sources = {
        "modules": {".py"},
        "ui": {".py"},
        "presets": {".json"},
        "corpus": {".txt"},
        "assets": {".png", ".jpg", ".jpeg", ".svg"},
    }
    for folder, suffixes in sources.items():
        base = ROOT / folder
        if not base.is_dir():
            continue
        files.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in suffixes
            and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(ROOT).parts)
        )
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output .zip path")
    parser.add_argument("--force", action="store_true", help="Replace an existing output archive")
    args = parser.parse_args()

    destination = args.out if args.out.is_absolute() else ROOT / args.out
    if destination.exists() and not args.force:
        parser.error(f"output already exists; pass --force to replace it: {destination}")

    files = collect_files()
    manifest = {
        "format": "voice-design-cloner-colab-source",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
        archive.writestr("colab_source_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"Created: {destination}")
    print(f"Files: {len(files)} | Size: {destination.stat().st_size / 1024 / 1024:.2f} MiB")
    print("Excluded: Git metadata, virtualenvs, logs, output audio, LoRAs, references, and config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
