__all__ = [  # noqa 405
    "is_initialized",
    "init",
    "synchronize",
    "device_count",
    "can_device_access_peer",
    "set_device",
    "current_device",
    "get_device_name",
    "get_device_properties",
    "get_device_capability",
    "is_available",
    "device",
    "device_of",
    "stream",
    "set_stream",
    "current_stream",
    "default_stream",
    "set_sync_debug_mode",
    "get_sync_debug_mode",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "set_per_process_memory_fraction",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_snapshot",
    "memory_summary",
    "get_allocator_backend",
    "Stream",
    "Event",
    "get_rng_state",
    "set_rng_state",
    "get_rng_state_all",
    "set_rng_state_all",
    "is_bf16_supported",
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "ShortTensor",
    "ShortStorage",
    "LongTensor",
    "LongStorage",
    "IntTensor",
    "IntStorage",
    "HalfTensor",
    "HalfStorage",
    "CharTensor",
    "CharStorage",
    "DoubleTensor",
    "DoubleStorage",
    "FloatTensor",
    "FloatStorage",
    "BFloat16Tensor",
    "BFloat16Storage",
]

from typing import Tuple, Union
import traceback
import threading
import os
import torch
from torch.storage import _LegacyStorage, _warn_typed_storage_removal
from torch._utils import classproperty

import torch_backend
import torch_backend.backend
from .utils import (  # noqa 401
    synchronize,
    device_count,
    can_device_access_peer,
    set_device,
    current_device,
    get_device_name,
    get_device_properties,
    get_device_capability,
    _get_device_index,
    device,
    device_of,
    stream,
    set_stream,
    current_stream,
    default_stream,
    set_sync_debug_mode,
    get_sync_debug_mode,
    is_bf16_supported,
)
from .streams import Stream, Event
from .autocast_mode import *  # noqa 403

default_generators: Tuple[torch._C.Generator] = ()  # type: ignore[assignment]

_is_internal_in_bad_fork = False
_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_original_pid = False

def _is_in_bad_fork():
    return _is_internal_in_bad_fork


def is_initialized():
    r"""Returns whether PyTorch's NPU state has been initialized."""
    return _initialized and not _is_internal_in_bad_fork


def _lazy_call(cb):
    if _initialized:
        cb()
    else:
        # Don't store the actual traceback to avoid memory cycle
        _queued_calls.append((cb, traceback.format_stack()))


class _DeferredNpuCallError(Exception):
    pass


def init():
    r"""Initialize PyTorch's NPU state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for NPU functionality will not
    be until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's NPU methods
    automatically initialize NPU state on-demand.

    Does nothing if the NPU state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    def _queue_call(queued_calls):
        for queued_call, orig_traceback in queued_calls:
            try:
                queued_call()
            except Exception as e:
                msg = (
                    f"NPU call failed lazily at initialization with error: {str(e)}\n\n"
                    f"NPU call was originally invoked at:\n\n{orig_traceback}"
                )
                raise _DeferredNpuCallError(msg) from e

    global _initialized, _original_pid, _queued_calls
    if _initialized or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if _initialized:
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_internal_in_bad_fork:
            raise RuntimeError(
                "Cannot re-initialize NPU in forked subprocess. To use NPU with "
                "multiprocessing, you must use the 'spawn' start method"
            )

        torch_backend._C._init()

        _original_pid = os.getpid()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True
        try:
            _queue_call(_queued_calls)
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


def _get_device(device: Union[int, str, torch.device]) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, int):
        return torch.device("npu", device)
    return device


def _get_generator(device: torch.device) -> torch._C.Generator:
    r"""Return the NPU Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """

    idx = device.index
    if idx is None:
        idx = current_device()
    return torch.npu.default_generators[idx]  # type: ignore[attr-defined]


def _set_rng_state_offset(
        offset: int, device: Union[int, str, torch.device] = "npu"
) -> None:
    r"""Sets the random number generator state offset of the specified NPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'npu'`` (i.e., ``torch.device('npu')``, the current NPU device).
    """
    final_device = _get_device(device)

    def cb():
        default_generator = _get_generator(final_device)
        default_generator.set_offset(offset)

    _lazy_call(cb)


def _get_rng_state_offset(device: Union[int, str, torch.device] = "npu") -> int:
    r"""Returns the random number generator state offset of the specified NPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'npu'`` (i.e., ``torch.device('npu')``, the current NPU device).

    .. warning::
        This function eagerly initializes NPU.
    """
    _lazy_init()
    final_device = _get_device(device)
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()


def is_available():
    if not hasattr(torch_backend._C, "setDevice"):
        return False
    return device_count() > 0


from .random import *  # noqa: F403
from .memory import *  # noqa: F403


class _NPULegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        raise RuntimeError(
            "from_buffer: Not available for NPU storage"
        )

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError(
            "_new_with_weak_ptr: Not available for NPU storage"
        )

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError(
            "_new_shared_filename: Not available for NPU storage"
        )

# TODO(FFFrog):
# Option 1: add a decorator to modify the __module__ of storage class below to device name
# Option 2: add __init__ to _LegacyStorageMeta in PyTorch, in which modify __module__

class ByteStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


del _LegacyStorage
del _NPULegacyStorage

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
