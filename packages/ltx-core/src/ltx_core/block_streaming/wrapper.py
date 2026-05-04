"""Block streaming wrapper: streams transformer blocks through a WeightsProvider."""

from __future__ import annotations

import itertools
from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch
from torch import nn

from ltx_core.block_streaming.disk import LoraSource
from ltx_core.block_streaming.provider import WeightsProvider
from ltx_core.block_streaming.source import WeightSource
from ltx_core.block_streaming.utils import assign_tensor_to_module


class BlockStreamingWrapper(nn.Module):
    """Streams sequential model blocks through GPU buffer caches.
    The wrapper delegates all weight management to a :class:`WeightsProvider`
    which handles CPU-to-GPU copies, caching, LoRA fusion, and stream
    synchronization internally.
    Use :class:`StreamingModelBuilder` to construct this wrapper -- it
    handles checkpoint parsing, source selection, and provider creation.
    Args:
        model: The wrapped model (non-block params already on GPU).
        blocks: Sequential blocks to stream (``nn.ModuleList``).
        provider: Provides GPU-ready weights on demand.
        target_device: GPU device for compute.
    """

    def __init__(
        self,
        model: nn.Module,
        blocks: nn.ModuleList,
        provider: WeightsProvider | dict[torch.device, WeightsProvider],
        target_device: torch.device,
        block_devices: list[torch.device] | None = None,
        shared_source: WeightSource | None = None,
        shared_lora_sources: list[LoraSource] | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._blocks = blocks
        self._target_device = target_device
        self._providers = provider if isinstance(provider, dict) else {target_device: provider}
        self._block_devices = block_devices or [target_device] * len(blocks)
        self._shared_source = shared_source
        self._shared_lora_sources = shared_lora_sources or []
        if len(self._block_devices) != len(blocks):
            raise ValueError(f"Expected {len(blocks)} block devices, got {len(self._block_devices)}")

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()
        for provider in self._providers.values():
            provider.prime()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _provider_for_block(self, block_idx: int) -> WeightsProvider:
        return self._providers[self._block_devices[block_idx]]

    @classmethod
    def _move_value(cls, value: Any, device: torch.device) -> Any:  # noqa: ANN401
        if isinstance(value, torch.Tensor):
            return value if value.device == device else value.to(device)
        if isinstance(value, tuple):
            return tuple(cls._move_value(item, device) for item in value)
        if isinstance(value, list):
            return [cls._move_value(item, device) for item in value]
        if isinstance(value, dict):
            return {key: cls._move_value(item, device) for key, item in value.items()}
        if is_dataclass(value) and not isinstance(value, type):
            updates = {
                field.name: cls._move_value(getattr(value, field.name), device)
                for field in fields(value)
            }
            return replace(value, **updates)
        return value

    def _pre_hook(
        self,
        block_idx: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Load GPU weights for a block and inject them into its parameters."""
        device = self._block_devices[block_idx]
        gpu_weights = self._provider_for_block(block_idx).get(block_idx)

        block = self._blocks[block_idx]
        for name, _param in itertools.chain(block.named_parameters(), block.named_buffers()):
            assign_tensor_to_module(block, name, gpu_weights[name])

        moved_args = tuple(self._move_value(arg, device) for arg in args)
        moved_kwargs = {key: self._move_value(value, device) for key, value in kwargs.items()}
        return moved_args, moved_kwargs

    def _post_hook(self, block_idx: int) -> None:
        """Record a compute-done event and release the block weights."""
        device = self._block_devices[block_idx]
        compute_done = torch.cuda.Event()
        compute_done.record(torch.cuda.current_stream(device))
        self._provider_for_block(block_idx).release(block_idx, event=compute_done)

    def _register_hooks(self) -> None:
        for idx, block in enumerate(self._blocks):
            pre = block.register_forward_pre_hook(
                lambda _mod, _args, _kwargs, *, idx=idx: self._pre_hook(idx, _args, _kwargs),
                with_kwargs=True,
            )
            post = block.register_forward_hook(
                lambda _mod, _args, _out, *, idx=idx: self._post_hook(idx),
            )
            self._hooks.extend([pre, post])

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Remove hooks and release all resources."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for provider in self._providers.values():
            provider.cleanup()
        if self._shared_source is not None:
            self._shared_source.cleanup()
        for lora in self._shared_lora_sources:
            lora.cleanup()

    # ------------------------------------------------------------------
    # Forward and attribute delegation
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Proxy attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
