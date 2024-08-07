import torch

__all__ = ["get_amp_supported_dtype"]


def get_amp_supported_dtype():
    if torch.npu.is_bf16_supported():  # type: ignore[attr-defined]
        return [torch.float16, torch.bfloat16]
    return [torch.float16]
