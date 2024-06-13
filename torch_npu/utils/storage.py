import copy
import torch
from torch.storage import _warn_typed_storage_removal

import torch_npu


def _cpu(self):
    """Returns a CPU copy of this storage if it's not already on the CPU"""
    if self.device.type != 'cpu':
        fake_tensor = torch_npu._C._tensor_construct_from_storage(self)
        return fake_tensor.cpu().untyped_storage()
    else:
        return self


def _deepcopy(self, memo):
    if self.device.type != 'cpu':
        src_tensor = torch_npu._C._tensor_construct_from_storage(self)
        dst_tensor = src_tensor.clone()
        dst_tensor = torch_npu.npu_format_cast(dst_tensor, torch_npu.get_npu_format(src_tensor))
        return dst_tensor._typed_storage()
    else:
        return self._new_wrapped_storage(copy.deepcopy(self._untyped_storage, memo))


def _add_storage_methods():
    torch.storage.UntypedStorage.cpu = _cpu
    torch.storage.TypedStorage._deepcopy = _deepcopy
