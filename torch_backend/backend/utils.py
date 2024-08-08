import os
import warnings
import contextlib

import torch

import torch_backend
import torch_backend._C
from .device import (
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
)

__all__ = ["synchronize", "device_count", "can_device_access_peer", "set_device", "current_device", "get_device_name",
           "get_device_properties", "get_device_capability", "device", "device_of",
           "stream", "set_stream", "current_stream", "default_stream", "set_sync_debug_mode", "get_sync_debug_mode",
           "is_support_inf_nan", "is_bf16_supported"]


@contextlib.contextmanager
def stream(stream):
    r"""Context-manager that selects a given stream.

    All NPU kernels queued within its context will be enqueued on a selected
    stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    .. note:: Streams are per-device. If the selected stream is not on the
        current device, this function will also change the current device to
        match the stream.
    """
    if stream is None:
        yield
        return
    src_prev_stream = current_stream()

    if src_prev_stream.device != stream.device:
        # The given stream is on a different device; have to restore the
        # current_stream on that device on exit as well
        with device(stream.device):
            dst_prev_stream = current_stream()

    set_stream(stream)
    try:
        yield
    finally:
        if src_prev_stream.device != stream.device:
            set_stream(dst_prev_stream)
        set_stream(src_prev_stream)


def set_stream(stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.
    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch_backend._C._setStream(stream_id=stream.stream_id,
                                device_index=stream.device_index,
                                device_type=stream.device_type)


def current_stream(device=None):
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch_backend.backend.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_backend.backend._lazy_init()
    streamdata = torch_backend._C._getCurrentStream(
        _get_device_index(device, optional=True))
    return torch_backend.backend.Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def default_stream(device=None):
    r"""Returns the default :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch_backend.backend.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_backend.backend._lazy_init()
    streamdata = torch_backend._C._getDefaultStream(
        _get_device_index(device, optional=True))
    return torch_backend.backend.Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def set_sync_debug_mode(debug_mode):
    r"""Sets the debug mode for npu synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error.
    """

    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`")

    torch_backend._C._npu_set_sync_debug_mode(debug_mode)


def get_sync_debug_mode():
    r"""Returns current value of debug mode for npu synchronizing operations."""

    return torch_backend._C._npu_get_sync_debug_mode()


def _dummy_type(name):
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(
            "Tried to instantiate dummy base class {}".format(class_name))

    return type(name, (object,), {"__init__": init_err})


if not hasattr(torch_backend._C, '_NPUStreamBase'):
    # Define dummy base classes
    torch_backend._C.__dict__['_NPUStreamBase'] = _dummy_type('NPUStreamBase')
    torch_backend._C.__dict__['_EventBase'] = _dummy_type('NPUEventBase')


def is_support_inf_nan():
    torch_backend.backend._lazy_init()
    return torch_backend._C._npu_is_support_inf_nan()


def is_bf16_supported():
    torch_backend.backend._lazy_init()
    return torch_backend._C._npu_is_bf16_supported()
