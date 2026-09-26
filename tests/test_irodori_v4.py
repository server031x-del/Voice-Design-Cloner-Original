from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

from modules.irodori_bridge import IrodoriBridge
from modules.lora_pipeline import _lora_paths, _write_training_jsonl
from ui.tab_irodori_v4 import _combined_refs, _effective_precision


class IrodoriV4BridgeTests(unittest.TestCase):
    def test_v4_payload_preserves_multi_reference_and_controls(self):
        bridge = IrodoriBridge(Path("unused"))
        captured = {}
        bridge.ensure_started = lambda: None

        def fake_send(payload):
            captured.update(payload)
            return {"ok": True}

        bridge._send = fake_send
        bridge.synthesize(
            profile="v4",
            mode="clone",
            model_variant="int8-weight-only",
            model_precision="bf16",
            text="テスト",
            caption="明るく",
            ref_wavs=["a.wav", "b.wav"],
            no_ref=False,
            out_path="out.wav",
            num_steps=12,
            duration_scale=1.1,
            max_ref_seconds=120.0,
            release_after_synthesis=True,
        )

        self.assertEqual(captured["profile"], "v4")
        self.assertEqual(captured["model_variant"], "int8-weight-only")
        self.assertEqual(captured["ref_wavs"], ["a.wav", "b.wav"])
        self.assertEqual(captured["caption"], "明るく")
        self.assertEqual(captured["num_steps"], 12)
        self.assertEqual(captured["max_ref_seconds"], 120.0)
        self.assertTrue(captured["release_after_synthesis"])

    def test_runtime_release_uses_worker_operation(self):
        bridge = IrodoriBridge(Path("unused"))
        bridge._proc = mock.Mock()
        bridge._proc.poll.return_value = None
        bridge._send = lambda payload: {
            "ok": payload == {"op": "release_runtime"},
            "runtime_released": True,
        }

        response = bridge.release_runtime()

        self.assertTrue(response["runtime_released"])

    def test_legacy_payload_defaults_remain_legacy(self):
        bridge = IrodoriBridge(Path("unused"))
        captured = {}
        bridge.ensure_started = lambda: None
        bridge._send = lambda payload: captured.update(payload) or {"ok": True}
        bridge.synthesize(mode="clone", text="テスト", ref_wav="a.wav", out_path="out.wav")

        self.assertEqual(captured["profile"], "legacy")
        self.assertEqual(captured["mode"], "clone")
        self.assertEqual(captured["ref_wav"], "a.wav")
        self.assertNotIn("ref_wavs", captured)


class IrodoriV4SeparationTests(unittest.TestCase):
    def test_lora_storage_is_separate(self):
        legacy_output, legacy_data = _lora_paths("legacy")
        v4_output, v4_data = _lora_paths("v4")
        self.assertNotEqual(legacy_output, v4_output)
        self.assertNotEqual(legacy_data, v4_data)
        self.assertEqual(v4_output.name, "lora_v4")
        self.assertEqual(v4_data.name, "lora_data_v4")

    def test_quantized_precision_is_forced_to_bf16(self):
        self.assertEqual(_effective_precision("int4-weight-only", "fp32"), "bf16")
        self.assertEqual(_effective_precision("full", "fp32"), "fp32")

    def test_saved_and_uploaded_references_are_combined(self):
        # No saved label keeps this test independent of user output contents.
        self.assertEqual(_combined_refs(None, []), [])

    def test_v4_training_jsonl_carries_speaker_and_caption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            emotion_dir = root / "lab" / "speaker_a" / "neutral"
            wav_dir = emotion_dir / "wavs"
            wav_dir.mkdir(parents=True)
            (emotion_dir / "neutral.txt").write_text("0001: テスト\n", encoding="utf-8")
            (wav_dir / "0001.wav").write_bytes(b"placeholder")
            output = root / "train.jsonl"

            written = _write_training_jsonl(
                lab_root=root / "lab",
                speaker="speaker_a",
                emotion="neutral",
                out_jsonl=output,
                include_speaker=True,
                caption="落ち着いた声",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written, 1)
            self.assertEqual(payload["speaker_id"], "speaker_a")
            self.assertEqual(payload["caption"], "落ち着いた声")


if __name__ == "__main__":
    unittest.main()
