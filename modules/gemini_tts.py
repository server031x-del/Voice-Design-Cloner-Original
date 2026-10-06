"""Gemini TTS voice design client (REST, no extra dependencies).

Workflow used by vdc:

1. ``create_voice`` designs a persistent voice from a natural-language
   description (``POST /v1beta/voices`` with ``type="prompted"``) and returns
   its ``voice_...`` id plus a preview WAV.
2. ``synthesize`` renders text with that voice (``POST /v1beta/interactions``).
3. :func:`build_irodori_reference` renders a set of varied Japanese lines with
   the designed voice and saves them as a ~30 s reference clip under
   ``output/voice_design/`` so Irodori-TTS (V4 or V3) can clone the voice
   locally.

The API key comes from ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` or, if the user
chose to save it, ``config.json`` (git-ignored).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MODELS = ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"]
DEFAULT_MODEL = MODELS[0]
DEFAULT_LANGUAGE = "ja-JP"
GENDERS = ["female", "male"]
# Unary responses are 24 kHz mono 16-bit WAV unless configured otherwise.
PCM_SAMPLE_RATE = 24000
MAX_STORED_VOICES = 200

# Varied delivery (statement / question / exclamation / calm / longer
# sentence) gives Irodori's speaker encoder a fuller picture of the voice.
DEFAULT_REFERENCE_LINES = [
    "こんにちは、はじめまして。今日はよろしくお願いします。",
    "朝の空気が澄んでいて、遠くの山までくっきり見えました。",
    "えっ、それ本当なの？ちょっと信じられないな。",
    "大丈夫、ゆっくりでいいから、一緒に考えていこうね。",
    "駅前の小さな喫茶店で、温かいコーヒーを飲みながら本を読むのが好きです。",
    "やった、ついに完成した！みんな本当にありがとう！",
    "少し疲れちゃったけど、明日もまた頑張ろうと思います。",
    "それでは、今日のお話はここまで。また次回お会いしましょう。",
]


class GeminiTTSError(RuntimeError):
    pass


# ---------------------------------------------------------------- API key

def _config_path() -> Path:
    from config import BASE_DIR
    return BASE_DIR / "config.json"


def get_api_key() -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        data = json.loads(_config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = str(data.get("gemini_api_key") or "").strip()
    return value or None


def api_key_source() -> str:
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return "環境変数 GEMINI_API_KEY"
    if os.environ.get("GOOGLE_API_KEY", "").strip():
        return "環境変数 GOOGLE_API_KEY"
    if get_api_key():
        return "config.json（保存済み）"
    return ""


def save_api_key(value: str | None) -> None:
    """Persist (or clear) the key in the git-ignored config.json."""
    path = _config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if value:
        data["gemini_api_key"] = value.strip()
    else:
        data.pop("gemini_api_key", None)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- audio helpers

def pcm16_to_wav(pcm: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def ensure_wav(data: bytes) -> bytes:
    """Wrap headerless linear PCM in a RIFF header if needed."""
    return data if data[:4] == b"RIFF" else pcm16_to_wav(data)


def _find_audio_data(node: Any) -> str | None:
    """Return the last base64 audio payload in an Interactions response."""
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "audio" and isinstance(value.get("data"), str):
                found.append(value["data"])
            output_audio = value.get("output_audio")
            if isinstance(output_audio, dict) and isinstance(output_audio.get("data"), str):
                found.append(output_audio["data"])
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return found[-1] if found else None


# ---------------------------------------------------------------- client

class GeminiTTSClient:
    def __init__(self, api_key: str | None = None, *, timeout: float = 180.0, max_retries: int = 4) -> None:
        self.api_key = (api_key or get_api_key() or "").strip()
        if not self.api_key:
            raise GeminiTTSError(
                "Gemini APIキーが設定されていません。環境変数 GEMINI_API_KEY を設定するか、画面で入力してください。"
            )
        self.timeout = timeout
        self.max_retries = max_retries

    def _request(self, method: str, path: str, payload: dict | None = None,
                 query: list[tuple[str, str]] | None = None) -> dict:
        url = f"{API_BASE}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"x-goog-api-key": self.api_key, "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in (429, 500, 502, 503, 504)
                if retryable and attempt < self.max_retries:
                    delay = _retry_delay(exc.headers.get("Retry-After"), attempt)
                    logger.warning("Gemini HTTP %s, retrying in %.1fs", exc.code, delay)
                    time.sleep(delay)
                    continue
                raise GeminiTTSError(f"Gemini API HTTP {exc.code}: {_error_message(detail)}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    time.sleep(_retry_delay(None, attempt))
                    continue
                raise GeminiTTSError(f"Gemini APIに接続できません: {exc}") from exc
        raise GeminiTTSError("Gemini API request failed")  # pragma: no cover

    # -- voices ---------------------------------------------------------

    def create_voice(
        self,
        description: str,
        *,
        display_name: str,
        gender: str = "female",
        language_code: str = DEFAULT_LANGUAGE,
        model: str = DEFAULT_MODEL,
        store: bool = True,
    ) -> dict:
        """Design a voice. Returns ``{"id", "display_name", "sample_wav", "raw"}``."""
        if not (description or "").strip():
            raise GeminiTTSError("声の説明を入力してください")
        voice: dict[str, Any] = {
            "model": model,
            "type": "prompted",
            "display_name": (display_name or "vdc voice").strip()[:100],
            "language_code": language_code or DEFAULT_LANGUAGE,
            "prompted": {"input": description.strip()},
        }
        if gender in GENDERS:
            voice["gender"] = gender
        response = self._request("POST", "/voices", {"store": bool(store), "voice": voice})
        voice_id = response.get("id") or response.get("name", "").split("/")[-1]
        if not voice_id:
            raise GeminiTTSError(f"voice id がレスポンスにありません: {list(response)}")
        return {
            "id": voice_id,
            "display_name": response.get("display_name", voice["display_name"]),
            "sample_wav": _decode_sample(response),
            "raw": response,
        }

    def get_voice(self, voice_id: str) -> dict:
        response = self._request("GET", f"/voices/{urllib.parse.quote(voice_id)}")
        response["sample_wav"] = _decode_sample(response)
        return response

    def list_voices(self, language_codes: list[str] | None = None) -> list[dict]:
        query: list[tuple[str, str]] = [("type", "prompted")]
        for code in language_codes or []:
            query.append(("language_code", code))
        voices: list[dict] = []
        page_token = None
        for _ in range(20):
            page_query = list(query) + ([("page_token", page_token)] if page_token else [])
            response = self._request("GET", "/voices", query=page_query)
            voices.extend(response.get("voices") or [])
            page_token = response.get("next_page_token") or response.get("nextPageToken")
            if not page_token:
                break
        return voices

    def delete_voice(self, voice_id: str) -> None:
        self._request("DELETE", f"/voices/{urllib.parse.quote(voice_id)}")

    # -- synthesis ------------------------------------------------------

    def synthesize(self, text: str, *, voice: str, model: str = DEFAULT_MODEL,
                   style: str | None = None) -> bytes:
        """Render ``text`` with ``voice`` and return WAV bytes (24 kHz mono)."""
        content: dict[str, Any] = {"type": "text", "text": text}
        if style and style.strip():
            content["annotations"] = [{"type": "speech_metadata", "style": style.strip()}]
        payload = {
            "model": model,
            "input": [{"type": "user_input", "content": [content]}],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": voice}]},
        }
        response = self._request("POST", "/interactions", payload)
        data = _find_audio_data(response)
        if not data:
            raise GeminiTTSError("音声データがレスポンスにありません（安全フィルタで拒否された可能性があります）")
        return ensure_wav(base64.b64decode(data))


def _decode_sample(response: dict) -> bytes | None:
    sample = response.get("sample_audio") or {}
    data = sample.get("data") if isinstance(sample, dict) else None
    return ensure_wav(base64.b64decode(data)) if data else None


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    try:
        if retry_after:
            return min(60.0, float(retry_after))
    except ValueError:
        pass
    return min(60.0, (2 ** attempt) * 2.0 + random.uniform(0, 1.0))


def _error_message(detail: str) -> str:
    try:
        payload = json.loads(detail)
        return payload.get("error", {}).get("message") or detail[:500]
    except ValueError:
        return detail[:500]


# ---------------------------------------------------------------- Irodori hand-off

def wav_bytes_to_array(data: bytes) -> tuple[int, np.ndarray]:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return int(sr), audio


def build_irodori_reference(
    client: GeminiTTSClient,
    *,
    voice_id: str,
    name: str,
    lines: list[str],
    model: str = DEFAULT_MODEL,
    style: str | None = None,
    description: str | None = None,
    gap_sec: float = 0.4,
    max_total_sec: float = 60.0,
) -> Iterator[tuple[float, object]]:
    """Render ``lines`` with a Gemini voice and save an Irodori reference.

    Writes the individual clips to ``output/gemini_voices/{name}/`` and a
    joined clip (≤ ``max_total_sec``) to ``output/voice_design/{name}.wav`` with
    ``.txt``/``.json`` sidecars, so it shows up as a saved Voice Design in the
    Irodori screens.  Yields ``(pct, message)`` then ``(1.0, result)``.
    """
    from config import OUTPUT_DIR
    from modules.dataset_io import atomic_write_json, sanitize_segment, write_text_list
    from modules.voice_design import save_voice

    lines = [line.strip() for line in lines if line and line.strip()]
    if not lines:
        raise GeminiTTSError("リファレンス用のテキストを1行以上入力してください")
    safe_name = sanitize_segment(name, "gemini_voice")
    clip_dir = OUTPUT_DIR / "gemini_voices" / safe_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    clips: list[tuple[str, np.ndarray]] = []
    sample_rate = PCM_SAMPLE_RATE
    for index, line in enumerate(lines, start=1):
        yield (index - 1) / len(lines), f"Geminiで生成中 {index}/{len(lines)}: {line[:24]}"
        wav = client.synthesize(line, voice=voice_id, model=model, style=style)
        (clip_dir / f"{index:02d}.wav").write_bytes(wav)
        sample_rate, audio = wav_bytes_to_array(wav)
        clips.append((line, audio))
    write_text_list(clip_dir / "Neutral.txt", [(f"{i:02d}", text) for i, (text, _) in enumerate(clips, start=1)])

    joined: list[np.ndarray] = []
    used_lines: list[str] = []
    total = 0.0
    gap = np.zeros(int(sample_rate * gap_sec), dtype=np.float32)
    for text, audio in clips:
        seconds = audio.size / float(sample_rate)
        if joined and total + seconds > max_total_sec:
            break
        if joined:
            joined.append(gap)
            total += gap_sec
        joined.append(audio.astype(np.float32))
        used_lines.append(text)
        total += seconds
    reference = np.concatenate(joined)
    peak = float(np.max(np.abs(reference))) if reference.size else 0.0
    if peak > 0.99:
        reference = reference * (0.99 / peak)

    metadata = {
        "source": "gemini",
        "voice_id": voice_id,
        "model": model,
        "description": description,
        "style": style,
        "lines": used_lines,
        "clip_dir": str(clip_dir),
        "sample_rate": sample_rate,
        "duration_sec": round(total, 2),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(clip_dir / "voice.json", metadata)
    destination = save_voice((sample_rate, reference), safe_name, sample_text="".join(used_lines),
                             metadata=metadata)
    yield 1.0, {**metadata, "reference_path": destination, "clips": len(clips)}
