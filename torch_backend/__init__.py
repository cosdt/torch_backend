import torch
import os

import torch_backend._C
import torch_backend.backend
import torch_backend.meta

# TODO(FFFrog):
# an API must been provided by torch_backend to show device name
# used in torch.

torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module("npu", torch_backend.backend)

unsupported_dtype = [
    torch.quint8,
    torch.quint4x2,
    torch.quint2x4,
    torch.qint32,
    torch.qint8,
]
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
    unsupported_dtype=unsupported_dtype,
)

supported_dtypes = [
    torch.uint8,
    torch.int8,
    torch.float64,
    torch.float32,
    torch.int32,
    torch.int64,
    torch.int16,
    torch.float16,
    torch.bool,
    torch.bfloat16,
]
torch_backend._C.generate_tensor_types(supported_dtypes)


# This function is an entrypoint called by PyTorch
# when running 'import torch'. There is no need to do anything.
# See this RFC: https://github.com/pytorch/pytorch/pull/127074
def _autoload():
    pass
