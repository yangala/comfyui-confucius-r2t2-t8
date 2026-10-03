"""The transcribe node must send 16 kHz mono, not the source audio format."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def load_plugin():
    root = Path(__file__).resolve().parents[1]
    name = "r2t2_node_input_test"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)])
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[name] = plugin
    spec.loader.exec_module(plugin)
    return plugin


class NodeInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_output_directory = lambda: str(Path(__file__).resolve().parents[1] / ".runtime")
        previous = sys.modules.get("folder_paths")
        sys.modules["folder_paths"] = folder_paths
        try:
            cls.plugin = load_plugin()
        finally:
            if previous is None:
                sys.modules.pop("folder_paths", None)
            else:
                sys.modules["folder_paths"] = previous
        cls.node = cls.plugin.NODE_CLASS_MAPPINGS["R2T2Transcribe"]()

    def run_node(self, audio, channel="mean"):
        captured = {}

        def fake_transcribe(payload, options, config):
            captured["payload"] = payload
            captured["options"] = options
            return {"text": "ok", "language": "Chinese"}

        model = {"config": {"n_ctx": 8192}}
        with patch.object(self.plugin.nodes.manager, "transcribe", fake_transcribe):
            self.node.transcribe(model, audio, "offline", "Chinese", "", "", channel)
        return captured

    def stereo_audio(self, seconds=3, rate=44100):
        time = np.arange(seconds * rate, dtype=np.float32) / rate
        left = np.sin(2 * np.pi * 220 * time)
        right = np.sin(2 * np.pi * 440 * time)
        return {"waveform": np.ascontiguousarray(np.stack((left, right))[None]),
                "sample_rate": rate}

    def test_stereo_source_is_downmixed_and_resampled_before_the_request(self):
        audio = self.stereo_audio()
        captured = self.run_node(audio)
        self.assertEqual(captured["options"]["sample_rate"], 16000)
        self.assertEqual(captured["options"]["channels"], 1)
        self.assertNotIn("channel", captured["options"])
        self.assertEqual(len(captured["payload"]), 3 * 16000 * 4)

    def test_channel_selects_the_downmix(self):
        left = self.run_node(self.stereo_audio(), "left")["payload"]
        right = self.run_node(self.stereo_audio(), "right")["payload"]
        mean = self.run_node(self.stereo_audio(), "mean")["payload"]
        self.assertNotEqual(left, right)
        self.assertNotEqual(mean, left)
        self.assertNotEqual(mean, right)

    def test_source_format_no_longer_bounds_the_request_size(self):
        # 30 minutes of 48 kHz stereo is 691 MB raw; the 16 kHz mono request is 115 MB.
        audio = self.stereo_audio(seconds=1800, rate=48000)
        captured = self.run_node(audio)
        self.assertEqual(len(captured["payload"]), 1800 * 16000 * 4)
        self.assertLess(len(captured["payload"]), 512 * 1024 * 1024)

    def test_invalid_audio_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_node({"waveform": np.zeros((2, 2, 100), np.float32), "sample_rate": 44100})


if __name__ == "__main__":
    unittest.main()
