"""Tests for the Irodori job helpers, QC, GPU gate, bridge timeouts and LoRA utilities."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from modules import audio_qc, irodori_jobs, lora_pipeline
from modules.dataset_io import read_text_list, write_text_list
from modules.gpu_gate import GPUBusyError, GPUGate, get_gate, gpu_session
from modules.irodori_bridge import IrodoriBridge, IrodoriTimeout, IrodoriUnavailable, _EOF


def _tone(seconds: float, sr: int = 48000, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeBridge:
    """Writes a tone whose length follows the text, like a well-behaved worker."""

    def __init__(self, fail_texts: dict[str, int] | None = None, sr: int = 48000):
        self.calls: list[dict] = []
        self.fail_texts = dict(fail_texts or {})
        self.sr = sr
        self.shutdowns = 0
        self.idle_scheduled = 0

    def synthesize(self, *, text, out_path, seed=None, **kwargs):
        self.calls.append({"text": text, "seed": seed, **kwargs})
        remaining = self.fail_texts.get(text, 0)
        if remaining:
            self.fail_texts[text] = remaining - 1
            raise RuntimeError(f"boom:{text}")
        seconds = 0.15 * len(text)
        sr = kwargs.get("target_sr") or self.sr
        sf.write(str(out_path), _tone(seconds, sr), sr)
        return {"ok": True, "used_seed": seed if seed is not None else 4242,
                "duration_sec": seconds, "max_seconds": 30.0, "sample_rate": sr}

    def shutdown(self):
        self.shutdowns += 1

    def schedule_idle_release(self, seconds=None):
        self.idle_scheduled += 1


class TrainLogParserTests(unittest.TestCase):
    def test_current_train_py_format_is_progress(self):
        event = lora_pipeline._parse_train_line("step=1200 loss=0.512300 rf=0.5 lr=1.000e-04", 3000)
        self.assertEqual(event["event"], "progress")
        self.assertEqual(event["step"], 1200)
        self.assertEqual(event["max_steps"], 3000)
        self.assertAlmostEqual(event["loss"], 0.5123)

    def test_legacy_format_still_parses(self):
        event = lora_pipeline._parse_train_line("[step 1234/30000] loss=0.5", 3000)
        self.assertEqual((event["step"], event["max_steps"]), (1234, 30000))

    def test_validation_and_other_lines_are_logs(self):
        for line in ("valid step=1000 loss=0.61 rf=0.6", "Using stratified logit-normal timestep sampling.",
                     "Checkpoint retention: latest=1 + best_val_loss=5."):
            self.assertEqual(lora_pipeline._parse_train_line(line, 3000)["event"], "log", line)

    def test_resume_line_sets_progress(self):
        event = lora_pipeline._parse_train_line("Resumed from step=600", 3000)
        self.assertEqual((event["event"], event["step"]), ("progress", 600))

    def test_checkpoint_schedule(self):
        small = lora_pipeline.checkpoint_schedule(3000, 100)
        self.assertEqual(small["--save-every"], "600")
        self.assertEqual(small["--valid-ratio"], "0")
        large = lora_pipeline.checkpoint_schedule(30000, 5000)
        self.assertNotIn("--valid-ratio", large)


class ChunkingTests(unittest.TestCase):
    def test_short_text_is_untouched(self):
        self.assertEqual(irodori_jobs.split_for_tts("こんにちは。元気？", 120), ["こんにちは。元気？"])

    def test_long_text_is_split_without_losing_content(self):
        text = "今日はとても良い天気ですね。" * 20
        chunks = irodori_jobs.split_for_tts(text, 60)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 60 for c in chunks))
        self.assertEqual("".join(chunks), text)

    def test_sentence_without_breaks_is_hard_split(self):
        chunks = irodori_jobs.split_for_tts("あ" * 250, 100)
        self.assertEqual([len(c) for c in chunks], [100, 100, 50])

    def test_chunked_design_reuses_seed_and_first_chunk_as_reference(self):
        bridge = FakeBridge()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out.wav"
            info = irodori_jobs.synthesize_to_file(
                bridge, text="一文目です。" * 5 + "二文目です。" * 5, out_path=out, max_chars=30,
                profile="v4", mode="design", no_ref=True,
            )
            self.assertTrue(out.is_file())
            self.assertEqual(info["chunks"], len(bridge.calls))
            self.assertGreater(info["chunks"], 1)
        first_seed = bridge.calls[0]["seed"]
        # A random seed is chosen up front and stays browser-safe (< 2**31).
        self.assertIsNotNone(first_seed)
        self.assertLess(first_seed, 2**31)
        self.assertEqual(info["used_seed"], first_seed)
        self.assertTrue(all(call["seed"] == first_seed for call in bridge.calls[1:]))
        self.assertEqual(bridge.calls[0]["mode"], "design")
        self.assertEqual(bridge.calls[1]["mode"], "clone")
        self.assertFalse(bridge.calls[1]["no_ref"])
        self.assertEqual(len(bridge.calls[1]["ref_wavs"]), 1)

    def test_legacy_design_is_never_chunked(self):
        bridge = FakeBridge()
        with tempfile.TemporaryDirectory() as temp:
            irodori_jobs.synthesize_to_file(bridge, text="あ" * 300, out_path=Path(temp) / "o.wav",
                                            profile="legacy", mode="design")
        self.assertEqual(len(bridge.calls), 1)


class BatchRunnerTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.base = Path(self._temp.name) / "clone"
        self.ref = Path(self._temp.name) / "ref.wav"
        sf.write(str(self.ref), _tone(1.0), 48000)

    def tearDown(self):
        self._temp.cleanup()

    def _run(self, bridge, texts, **kwargs):
        kwargs.setdefault("request", {"profile": "v4", "mode": "clone", "target_sr": 44100})
        events = list(irodori_jobs.run_irodori_batch(
            bridge=bridge, texts=texts, base_dir=self.base, refs=[str(self.ref)],
            run_signal_qc=False, **kwargs,
        ))
        return events[-1][1]

    def test_failed_line_is_skipped_and_text_list_is_written(self):
        bridge = FakeBridge(fail_texts={"にばんめ": 99})
        result = self._run(bridge, ["いちばんめ", "にばんめ", "さんばんめ"], seed=10)
        self.assertEqual(result["total_files"], 2)
        self.assertEqual(len(result["failed"]), 1)
        entries = read_text_list(self.base / "Neutral.txt")
        self.assertEqual([e[0] for e in entries], ["0001", "0003"])
        manifest = json.loads((self.base / "batch_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([r["used_seed"] for r in manifest["records"] if r["status"] == "done"], [10, 12])

    def test_transient_failure_is_retried(self):
        bridge = FakeBridge(fail_texts={"いちばんめ": 1})
        result = self._run(bridge, ["いちばんめ"])
        self.assertEqual(result["total_files"], 1)
        self.assertEqual(result["failed"], [])

    def test_stop_on_error_raises(self):
        bridge = FakeBridge(fail_texts={"いちばんめ": 99})
        with self.assertRaises(RuntimeError):
            self._run(bridge, ["いちばんめ"], continue_on_error=False, max_retries=0)

    def test_resume_skips_done_lines_and_retries_failed_ones(self):
        texts = ["いちばんめ", "にばんめ", "さんばんめ"]
        self._run(FakeBridge(fail_texts={"にばんめ": 99}), texts)
        second = FakeBridge()
        result = self._run(second, texts)
        self.assertEqual([c["text"] for c in second.calls], ["にばんめ"])
        self.assertEqual(result["total_files"], 3)
        self.assertEqual(result["skipped_existing"], 2)

    def test_changed_settings_regenerate_everything(self):
        texts = ["いちばんめ", "にばんめ"]
        self._run(FakeBridge(), texts)
        second = FakeBridge()
        self._run(second, texts, request={"profile": "v4", "mode": "clone", "target_sr": 44100,
                                          "caption": "明るく"})
        self.assertEqual(len(second.calls), 2)

    def test_redo_ids_regenerate_with_a_new_seed(self):
        texts = ["いちばんめ", "にばんめ"]
        self._run(FakeBridge(), texts, seed=100)
        second = FakeBridge()
        self._run(second, texts, seed=100, redo_ids={"0002"})
        self.assertEqual([c["text"] for c in second.calls], ["にばんめ"])
        self.assertNotEqual(second.calls[0]["seed"], 101)

    def test_release_policy_is_applied(self):
        bridge = FakeBridge()
        self._run(bridge, ["いちばんめ"], release_mode=irodori_jobs.RELEASE_IMMEDIATE)
        self.assertEqual(bridge.shutdowns, 1)
        idle = FakeBridge()
        self._run(idle, ["いちばんめ"], release_mode=irodori_jobs.RELEASE_IDLE, resume=False)
        self.assertEqual((idle.shutdowns, idle.idle_scheduled), (0, 1))


class AudioQCTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()

    def _wav(self, name, audio, sr=48000):
        path = self.root / name
        sf.write(str(path), audio, sr)
        return path

    def test_signal_flags(self):
        good = audio_qc.analyze_clip(self._wav("good.wav", _tone(2.0)), "こんにちは、元気ですか")
        self.assertEqual(good["flags"], [])
        silent = audio_qc.analyze_clip(self._wav("silent.wav", np.zeros(48000, np.float32)), "あいう")
        self.assertIn("silent", silent["flags"])
        clipped = audio_qc.analyze_clip(self._wav("clip.wav", np.clip(_tone(2.0, amp=3.0), -1, 1)), "あいう")
        self.assertIn("clipping", clipped["flags"])
        long = audio_qc.analyze_clip(self._wav("long.wav", _tone(30.0)), "あ" * 100)
        self.assertIn("truncated", long["flags"])

    def test_cer_ignores_punctuation_and_kana_script(self):
        self.assertEqual(audio_qc.character_error_rate("こんにちは、世界！", "コンニチハ世界"), 0.0)
        self.assertGreater(audio_qc.character_error_rate("こんにちは", "さようなら"), 0.5)

    def test_speed_outliers(self):
        records = [{"sec_per_char": 0.15, "flags": []} for _ in range(6)]
        records.append({"sec_per_char": 0.6, "flags": []})
        audio_qc.flag_speed_outliers(records)
        self.assertIn("speed_outlier", records[-1]["flags"])
        self.assertEqual(records[0]["flags"], [])

    def test_run_filter_and_restore(self):
        raw = self.root / "raw"
        raw.mkdir()
        sf.write(str(raw / "0001.wav"), _tone(1.5), 48000)
        sf.write(str(raw / "0002.wav"), np.zeros(48000, np.float32), 48000)
        write_text_list(self.root / "Neutral.txt", [("0001", "こんにちは"), ("0002", "さようなら")])
        summary = [p for _, p in audio_qc.run_qc(self.root) if isinstance(p, dict)][0]
        self.assertEqual(summary["rejected"], 1)
        self.assertEqual(audio_qc.rejected_ids(self.root), {"0002"})
        result = audio_qc.apply_filter(self.root)
        self.assertEqual((result["kept"], result["excluded"]), (1, 1))
        self.assertEqual([e[0] for e in read_text_list(self.root / "Neutral.txt")], ["0001"])
        self.assertTrue((self.root / "qc_report.csv").is_file())
        self.assertTrue(audio_qc.restore_unfiltered(self.root))
        self.assertEqual(len(read_text_list(self.root / "Neutral.txt")), 2)


class GPUGateTests(unittest.TestCase):
    def test_second_job_waits_instead_of_failing(self):
        gate = get_gate()
        order: list[str] = []
        self.assertTrue(gate.try_acquire("first", timeout=0))

        def second():
            with gpu_session("second"):
                order.append("second")

        worker = threading.Thread(target=second)
        worker.start()
        time.sleep(0.3)
        self.assertEqual(gate.busy_label(), "first")
        order.append("first-done")
        gate.release()
        worker.join(timeout=5)
        self.assertEqual(order, ["first-done", "second"])
        self.assertIsNone(gate.busy_label())

    def test_fail_fast_mode_still_available(self):
        gate = get_gate()
        self.assertTrue(gate.try_acquire("holder", timeout=0))
        try:
            with self.assertRaises(GPUBusyError):
                with gpu_session("other", wait=False):
                    pass
        finally:
            gate.release()

    def test_timeout(self):
        gate = GPUGate()
        self.assertTrue(gate.try_acquire("a", timeout=0))
        self.assertFalse(gate.try_acquire("b", timeout=0.1))
        gate.release()


class BridgeTimeoutTests(unittest.TestCase):
    def _bridge(self, timeout):
        bridge = IrodoriBridge(Path("unused"), inactivity_timeout=timeout)
        bridge._proc = mock.Mock()
        bridge._proc.poll.return_value = None
        return bridge

    def test_silent_worker_is_killed(self):
        bridge = self._bridge(0.6)
        proc = bridge._proc
        started = time.monotonic()
        with self.assertRaises(IrodoriTimeout):
            bridge._wait_response()
        self.assertLess(time.monotonic() - started, 5)
        proc.kill.assert_called_once()
        self.assertIsNone(bridge._proc)

    def test_stderr_activity_keeps_request_alive(self):
        bridge = self._bridge(0.8)

        def chatter():
            for _ in range(4):
                time.sleep(0.4)
                bridge._last_activity = time.monotonic()
            bridge._stdout_queue.put({"ok": True})

        threading.Thread(target=chatter, daemon=True).start()
        self.assertEqual(bridge._wait_response(), {"ok": True})

    def test_worker_exit_is_reported(self):
        bridge = self._bridge(10)
        bridge._stdout_queue.put(_EOF)
        with self.assertRaises(IrodoriUnavailable):
            bridge._wait_response()

    def test_idle_release_shuts_down_when_gpu_is_free(self):
        bridge = self._bridge(10)
        bridge.shutdown = mock.Mock(side_effect=lambda: bridge.cancel_idle_release())
        bridge.schedule_idle_release(0)
        deadline = time.time() + 6
        while time.time() < deadline and not bridge.shutdown.called:
            time.sleep(0.1)
        bridge.shutdown.assert_called()

    def test_idle_release_can_be_cancelled(self):
        bridge = self._bridge(10)
        bridge.shutdown = mock.Mock()
        bridge.schedule_idle_release(0.5)
        bridge.cancel_idle_release()
        time.sleep(2.6)
        bridge.shutdown.assert_not_called()


class LoraCheckpointTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        patcher = mock.patch.object(lora_pipeline, "V4_LORA_OUTPUT_DIR", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.speaker_dir = self.root / "spk"
        for name in ("checkpoint_0000600", "checkpoint_0001200", "checkpoint_final"):
            ckpt = self.speaker_dir / name
            ckpt.mkdir(parents=True)
            (ckpt / "adapter_model.safetensors").write_bytes(b"x")
        (self.speaker_dir / "checkpoint_0001200" / "trainer_state.pt").write_bytes(b"x")

    def tearDown(self):
        self._temp.cleanup()

    def test_listing_orders_by_step(self):
        names = [c["name"] for c in lora_pipeline.list_lora_checkpoints("spk", "v4")]
        self.assertEqual(names, ["checkpoint_0000600", "checkpoint_0001200", "checkpoint_final"])

    def test_preferred_checkpoint_wins(self):
        self.assertTrue(lora_pipeline.get_lora_adapter_path("spk", "v4").endswith("checkpoint_final"))
        lora_pipeline.set_preferred_checkpoint("spk", "checkpoint_0000600", "v4")
        self.assertTrue(lora_pipeline.get_lora_adapter_path("spk", "v4").endswith("checkpoint_0000600"))
        lora_pipeline.set_preferred_checkpoint("spk", None, "v4")
        self.assertTrue(lora_pipeline.get_lora_adapter_path("spk", "v4").endswith("checkpoint_final"))

    def test_latest_resumable_needs_trainer_state(self):
        path, step = lora_pipeline.latest_resumable_checkpoint(self.speaker_dir)
        self.assertEqual((path.name, step), ("checkpoint_0001200", 1200))

    def test_archive_moves_previous_run_aside(self):
        archived = lora_pipeline._archive_previous_run(self.speaker_dir)
        self.assertIsNotNone(archived)
        self.assertEqual(lora_pipeline.list_lora_checkpoints("spk", "v4"), [])
        self.assertIsNone(lora_pipeline._resolve_adapter_path(self.speaker_dir))
        self.assertEqual(len(list(archived.iterdir())), 3)


class VoiceMetadataTests(unittest.TestCase):
    def test_saved_voice_keeps_seed_and_settings(self):
        from modules import voice_design

        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(voice_design, "VOICE_DESIGN_DIR", Path(temp)):
            info = {"used_seed": 77, "settings": {"caption": "低めの声", "model_variant": "full"}}
            voice_design.save_voice((48000, _tone(0.5)), "noa", "テスト", metadata=info)
            self.assertEqual(voice_design.get_kept_voice_metadata_by_label("noa")["used_seed"], 77)
            self.assertEqual(voice_design.list_kept_voice_labels_with_metadata(), ["noa"])


if __name__ == "__main__":
    unittest.main()
