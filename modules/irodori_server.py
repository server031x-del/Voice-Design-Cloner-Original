"""Client for an OpenAI-compatible Irodori-TTS server (optional, not used by the UI).

The server address must be given explicitly or via ``IRODORI_SERVER_URL``;
there is intentionally no built-in LAN address.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class IrodoriServerError(RuntimeError):
    pass


class IrodoriServerClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        resolved = base_url or os.environ.get("IRODORI_SERVER_URL")
        if not resolved:
            raise IrodoriServerError(
                "Irodori server URL is not configured. Pass base_url or set IRODORI_SERVER_URL."
            )
        self.base_url = resolved.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("IRODORI_SERVER_API_KEY", "")
        self._known_voices: set[str] = set()

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, path: str, *, data: bytes | None = None, content_type: str | None = None, timeout: int = 600) -> bytes:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=self._headers(content_type),
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise IrodoriServerError(f"Irodori server HTTP {exc.code}: {body}") from exc
        except OSError as exc:
            raise IrodoriServerError(f"Could not connect to Irodori server {self.base_url}: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return json.loads(self._request("/health", timeout=10).decode("utf-8"))

    def list_voices(self) -> set[str]:
        payload = json.loads(self._request("/v1/audio/voices", timeout=15).decode("utf-8"))
        voices = {
            str(item.get("id"))
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        }
        self._known_voices.update(voices)
        return voices

    @staticmethod
    def voice_id_for_file(ref_wav: str | Path) -> str:
        path = Path(ref_wav)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        stem = re.sub(r"[^0-9A-Za-z_-]+", "-", path.stem).strip("-_").lower() or "voice"
        return f"vdc-{stem[:40]}-{digest}"

    def ensure_voice(self, ref_wav: str | Path, voice_id: str | None = None) -> str:
        path = Path(ref_wav)
        if not path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {path}")
        resolved_id = voice_id or self.voice_id_for_file(path)
        if resolved_id in self._known_voices or resolved_id in self.list_voices():
            return resolved_id

        boundary = f"----IrodoriVDC{uuid.uuid4().hex}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = bytearray()

        def add(value: bytes) -> None:
            body.extend(value)
            body.extend(b"\r\n")

        add(f"--{boundary}".encode())
        add(b'Content-Disposition: form-data; name="voice_id"')
        add(b"")
        add(resolved_id.encode("utf-8"))
        add(f"--{boundary}".encode())
        add(f'Content-Disposition: form-data; name="file"; filename="{path.name}"'.encode("utf-8"))
        add(f"Content-Type: {content_type}".encode())
        add(b"")
        body.extend(path.read_bytes())
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())

        payload = json.loads(
            self._request(
                "/v1/audio/voices",
                data=bytes(body),
                content_type=f"multipart/form-data; boundary={boundary}",
                timeout=120,
            ).decode("utf-8")
        )
        registered_id = str(payload.get("id") or resolved_id)
        self._known_voices.add(registered_id)
        logger.info("Registered Irodori server voice: %s (%s)", registered_id, path)
        return registered_id

    def synthesize(
        self,
        *,
        text: str,
        out_path: str | Path,
        ref_wav: str | Path | None = None,
        voice_id: str | None = None,
        seed: int | None = None,
        num_steps: int = 40,
        duration_scale: float = 1.0,
        cfg_scale_text: float = 3.0,
        cfg_scale_speaker: float = 5.0,
    ) -> dict[str, Any]:
        if ref_wav is not None:
            voice_id = self.ensure_voice(ref_wav, voice_id)
        voice = voice_id or "none"
        request_payload = {
            "model": "irodori-tts",
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "speed": 1.0,
            "irodori": {
                "seed": seed,
                "num_steps": num_steps,
                "duration_scale": duration_scale,
                # The shared server currently runs fp32 on a 6 GB GPU.
                # Disabling CFG keeps peak VRAM below the available capacity.
                "cfg_scale_text": 0.0,
                "cfg_scale_speaker": 0.0,
                "cfg_guidance_mode": "alternating",
                "chunking_enabled": False,
            },
        }
        wav_bytes = self._request(
            "/v1/audio/speech",
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
            timeout=600,
        )
        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(wav_bytes)
        return {"ok": True, "out_path": str(destination), "voice": voice, "server": self.base_url}


_client: IrodoriServerClient | None = None


def get_server_client() -> IrodoriServerClient:
    global _client
    if _client is None:
        _client = IrodoriServerClient()
    return _client
