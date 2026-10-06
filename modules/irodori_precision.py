"""Keep full BF16 checkpoint loading from first allocating FP32 CUDA weights."""

from __future__ import annotations


def load_bf16_runtime(factory, key, model_class):
    """Convert CPU parameters before upstream's first model.to(device).

    Irodori main 89f9d8f moves the full model to CUDA before applying BF16.
    The worker handles one request at a time, so scope this compatibility
    override to its runtime construction and restore it even on failure.
    Convert floating buffers individually to preserve complex rotary caches.
    """
    import torch

    original_to = model_class.to

    def precision_first(model, *args, **kwargs):
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                    parameter.data = parameter.data.to(dtype=torch.bfloat16)
            for child in model.modules():
                for name, buffer in child._buffers.items():
                    if buffer is not None and buffer.is_floating_point() and buffer.dtype != torch.bfloat16:
                        child._buffers[name] = buffer.to(dtype=torch.bfloat16)
        return original_to(model, *args, **kwargs)

    model_class.to = precision_first
    try:
        return factory(key)
    finally:
        model_class.to = original_to
