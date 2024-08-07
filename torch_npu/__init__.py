import os
import atexit

# Disable autoloading before running 'import torch'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import torch

import torch_npu
import torch_npu._C
import torch_npu.npu
from .meta import _meta_registrations

torch.utils.rename_privateuse1_backend("npu")
# rename device name to 'npu' and register funcs
torch._register_device_module("npu", torch_npu.npu)
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

# this must be placed at the end
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
torch_npu._C.generate_tensor_types(supported_dtypes)


# This function is an entrypoint called by PyTorch
# when running 'import torch'. There is no need to do anything.
# See this RFC: https://github.com/pytorch/pytorch/pull/127074
def _autoload():
    pass
