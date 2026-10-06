import unittest

import torch

from modules.irodori_precision import load_bf16_runtime


class PrecisionBeforeTransferTests(unittest.TestCase):
    def test_float_weights_are_converted_before_transfer_and_complex_buffers_survive(self):
        seen = []

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
                self.register_buffer("rotary", torch.tensor([1 + 2j], dtype=torch.complex64))

            def to(self, *args, **kwargs):
                seen.append(self.weight.dtype)
                return super().to(*args, **kwargs)

        original = Model.to
        model = load_bf16_runtime(lambda key: Model().to("cpu"), None, Model)
        self.assertEqual(seen, [torch.bfloat16])
        self.assertEqual(model.rotary.dtype, torch.complex64)
        self.assertEqual(model.rotary.item(), 1 + 2j)
        self.assertIs(Model.to, original)

    def test_override_is_restored_on_load_failure(self):
        class Model(torch.nn.Module):
            pass
        original = Model.to
        def fail(key):
            raise RuntimeError("load failed")
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            load_bf16_runtime(fail, None, Model)
        self.assertIs(Model.to, original)
