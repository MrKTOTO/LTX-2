from collections.abc import Iterator
from contextlib import contextmanager
import gc
import logging
from typing import TypeVar

import torch

_M = TypeVar("_M", bound=torch.nn.Module)
logger = logging.getLogger(__name__)


def _model_cuda_devices(model: torch.nn.Module) -> list[torch.device]:
    devices: list[torch.device] = []
    for tensor in (*tuple(model.parameters()), *tuple(model.buffers())):
        device = tensor.device
        if device.type == "cuda" and device not in devices:
            devices.append(device)
    return devices


def _cleanup_devices(devices: list[torch.device]) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    try:
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        logger.warning("Host empty cache cleanup failed; ignoring.", exc_info=True)


@contextmanager
def gpu_model(model: _M) -> Iterator[_M]:
    """Context manager that yields a model and releases its memory on exit.
    Moves all parameters and buffers to ``meta`` device on exit, which
    immediately releases the underlying storage on **both** GPU and CPU,
    then runs ``cleanup_memory()`` to reclaim fragmented CUDA memory.
    Usage::
        with gpu_model(build_encoder()) as encoder:
            ...  # use encoder — typed as the concrete class
        # GPU + CPU memory freed automatically
    """
    try:
        yield model
    finally:
        devices = _model_cuda_devices(model)
        _cleanup_devices(devices)
        # .to("meta") releases storage for all parameters/buffers regardless
        # of their original device (CUDA or CPU).
        model.to("meta")
        _cleanup_devices(devices)
