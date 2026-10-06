"""Gemini TTS client tests with the HTTP layer mocked (no API key needed)."""

from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from modules import gemini_tts
from modules.gemini_tts import GeminiTTSClient, GeminiTTSError


def _wav_bytes(seconds: float = 1.0, sr: int = 24000) -> bytes:
    buffer = io.BytesIO()
    t = np.arange(int(seconds * sr)) / sr
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class _Response:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class GeminiClientTests(unittest.TestCase):
    def setUp(self):
        self.requests: list = []
        self.client = GeminiTTSClient(api_key="test-key", max_retries=2)

    def _fake(self, *responses):
        queue = list(responses)

        def urlopen(request, timeout=None):
            self.requests.append(request)
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return _Response(item)

        return mock.patch("urllib.request.urlopen", side_effect=urlopen)

    def test_create_voice_payload_and_sample(self):
        sample = base64.b64encode(_wav_bytes()).decode()
        with self._fake({"id": "voice_abc", "display_name": "noa",
                         "sample_audio": {"data": sample, "mime_type": "audio/wav"}}):
            created = self.client.create_voice("落ち着いた若い女性の声。", display_name="noa", gender="female")
        body = json.loads(self.requests[0].data.decode("utf-8"))
        self.assertTrue(self.requests[0].full_url.endswith("/v1beta/voices"))
        self.assertEqual(self.requests[0].get_header("X-goog-api-key"), "test-key")
        self.assertEqual(body["voice"]["type"], "prompted")
        self.assertEqual(body["voice"]["language_code"], "ja-JP")
        self.assertEqual(body["voice"]["prompted"]["input"], "落ち着いた若い女性の声。")
        self.assertTrue(body["store"])
        self.assertEqual(created["id"], "voice_abc")
        self.assertEqual(created["sample_wav"][:4], b"RIFF")

    def test_synthesize_reads_last_audio_step_and_style(self):
        audio = base64.b64encode(_wav_bytes(0.5)).decode()
        response = {"steps": [
            {"type": "user_input", "content": [{"type": "text", "text": "x"}]},
            {"type": "model_output", "content": [{"type": "audio", "data": audio}]},
        ]}
        with self._fake(response):
            wav = self.client.synthesize("こんにちは", voice="voice_abc", style="明るく")
        body = json.loads(self.requests[0].data.decode("utf-8"))
        self.assertEqual(body["generation_config"]["speech_config"], [{"voice": "voice_abc"}])
        self.assertEqual(body["input"][0]["content"][0]["annotations"][0]["style"], "明るく")
        self.assertEqual(wav[:4], b"RIFF")

    def test_headerless_pcm_is_wrapped(self):
        pcm = (np.zeros(2400, dtype=np.int16)).tobytes()
        wav = gemini_tts.ensure_wav(pcm)
        info = sf.info(io.BytesIO(wav))
        self.assertEqual((info.samplerate, info.channels, info.frames), (24000, 1, 2400))

    def test_rate_limit_is_retried(self):
        error = urllib.error.HTTPError("u", 429, "Too Many", {"Retry-After": "0"}, io.BytesIO(b"{}"))
        with self._fake(error, {"voices": [{"id": "voice_1"}]}), mock.patch("time.sleep"):
            voices = self.client.list_voices()
        self.assertEqual([v["id"] for v in voices], ["voice_1"])
        self.assertEqual(len(self.requests), 2)

    def test_api_error_message_is_surfaced(self):
        body = io.BytesIO(json.dumps({"error": {"message": "API key not valid"}}).encode())
        error = urllib.error.HTTPError("u", 400, "Bad", {}, body)
        with self._fake(error), self.assertRaisesRegex(GeminiTTSError, "API key not valid"):
            self.client.list_voices()

    def test_missing_key_is_reported(self):
        with mock.patch.object(gemini_tts, "get_api_key", return_value=None):
            with self.assertRaises(GeminiTTSError):
                GeminiTTSClient(api_key=None)


class IrodoriReferenceTests(unittest.TestCase):
    def test_reference_is_saved_as_voice_design(self):
        from modules import voice_design

        client = mock.Mock()
        client.synthesize.side_effect = lambda text, **_: _wav_bytes(0.1 * len(text))
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch("config.OUTPUT_DIR", Path(temp)), \
                mock.patch.object(voice_design, "VOICE_DESIGN_DIR", Path(temp) / "voice_design"):
            events = list(gemini_tts.build_irodori_reference(
                client, voice_id="voice_abc", name="noa gemini", lines=["一文目です。", "", "二文目です。"],
                description="落ち着いた声",
            ))
            result = events[-1][1]
            self.assertEqual(result["clips"], 2)
            self.assertTrue(Path(result["reference_path"]).is_file())
            self.assertEqual(Path(result["reference_path"]).name, "noa_gemini.wav")
            meta = voice_design.get_kept_voice_metadata_by_label("noa_gemini")
            self.assertEqual(meta["source"], "gemini")
            self.assertEqual(meta["voice_id"], "voice_abc")
            # Gemini references have no Irodori settings to "load back".
            self.assertEqual(voice_design.list_kept_voice_labels_with_metadata(), [])
            self.assertTrue((Path(temp) / "gemini_voices" / "noa_gemini" / "02.wav").is_file())


if __name__ == "__main__":
    unittest.main()
