"""Builder that constructs a BlockStreamingWrapper from safetensors checkpoints."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Generic

import torch
from torch import nn

from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader, LoraSource, block_cache_namespace
from ltx_core.block_streaming.pool import BlockLayout, WeightPool
from ltx_core.block_streaming.provider import WeightsProvider
from ltx_core.block_streaming.source import DiskWeightSource, PinnedWeightSource, RamWeightSource, WeightSource
from ltx_core.block_streaming.utils import build_pool_layout, resolve_attr
from ltx_core.block_streaming.wrapper import BlockStreamingWrapper
from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.helpers import create_meta_model, load_state_dict, read_model_config
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDictLoader,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

logger = logging.getLogger(__name__)

DISK_CPU_SLOTS = 2
_DEFAULT_GPU_SLOTS = 2


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except ValueError:
        return default


def _log_cuda_memory(device: torch.device, label: str) -> None:
    if device.type != "cuda":
        return
    try:
        with torch.cuda.device(device):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            allocated_bytes = torch.cuda.memory_allocated(device)
            reserved_bytes = torch.cuda.memory_reserved(device)
        logger.info(
            "%s | device=%s free=%.2fGB total=%.2fGB allocated=%.2fGB reserved=%.2fGB",
            label,
            device,
            free_bytes / 1024**3,
            total_bytes / 1024**3,
            allocated_bytes / 1024**3,
            reserved_bytes / 1024**3,
        )
    except Exception:
        logger.debug("Failed to query CUDA memory for %s", device, exc_info=True)


def _unique_devices(devices: list[torch.device]) -> list[torch.device]:
    unique: list[torch.device] = []
    for device in devices:
        if device not in unique:
            unique.append(device)
    return unique


def _block_device_map(devices: list[torch.device], block_count: int) -> list[torch.device]:
    if not devices:
        raise ValueError("At least one block streaming device is required")
    strategy = os.environ.get("LTX_STREAM_BLOCK_DEVICE_STRATEGY", "contiguous").strip().lower()
    if strategy == "round_robin":
        return [devices[idx % len(devices)] for idx in range(block_count)]

    placement: list[torch.device] = []
    for idx in range(block_count):
        device_idx = min(len(devices) - 1, idx * len(devices) // block_count)
        placement.append(devices[device_idx])
    return placement


def _resolve_block_devices(
    target_device: torch.device,
    block_count: int,
    block_devices: list[torch.device] | None = None,
) -> list[torch.device]:
    if not block_devices:
        return [target_device] * block_count
    devices = block_devices
    return _block_device_map(devices, block_count)


@dataclass(frozen=True)
class StreamingModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType]):
    """Immutable builder for :class:`BlockStreamingWrapper`.
    Reads block weights from safetensors on demand.  ``cpu_slots`` and
    ``gpu_slots`` control the memory/speed trade-off (see :meth:`build`).
    Args:
        model_class_configurator: Creates the model from a config dict.
        model_path: One or more ``.safetensors`` checkpoint paths.
        model_sd_ops: Key remapping applied to safetensors keys.
        module_ops: Module-level mutations for the meta model.
        loras: LoRA adapters fused into weights at load time.
        model_loader: Strategy for reading checkpoint metadata.
        registry: Shared cache for loaded state dicts.
        blocks_attr: Dotted path to the ``nn.ModuleList`` (e.g.
            ``"velocity_model.transformer_blocks"``).
        blocks_prefix: State-dict key prefix for block weights
            (e.g. ``"transformer_blocks"``).
        state_dict_prefix: Key prefix for non-block weights
            (e.g. ``"velocity_model."``).
        model_wrapper: Optional callable wrapping the model
            (e.g. ``X0Model``).
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)

    # Streaming-specific
    blocks_attr: str = ""
    blocks_prefix: str = ""
    state_dict_prefix: str = ""
    model_wrapper: Callable[[ModelType], nn.Module] | None = None

    def with_sd_ops(self, sd_ops: SDOps | None) -> StreamingModelBuilder:
        return replace(self, model_sd_ops=sd_ops)

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> StreamingModelBuilder:
        return replace(self, module_ops=module_ops)

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> StreamingModelBuilder:
        return replace(self, loras=loras)

    def model_config(self) -> dict:
        """Read model configuration from the checkpoint metadata."""
        return read_model_config(self.model_path, self.model_loader)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        """Create a model on the meta device and apply module operations."""
        return create_meta_model(self.model_class_configurator, config, module_ops)

    def build(
        self,
        target_device: torch.device,
        dtype: torch.dtype,
        cpu_slots_count: int | None = None,
        gpu_slots_count: int | None = None,
        staging_device: torch.device | None = None,
        block_devices: list[torch.device] | None = None,
        **_kwargs: object,
    ) -> BlockStreamingWrapper:
        """Build and return a ready-to-use :class:`BlockStreamingWrapper`.
        Args:
            target_device: GPU device for compute.
            dtype: Weight dtype (e.g. ``torch.bfloat16``).
            cpu_slots_count: Number of pinned CPU buffer slots.
                ``None`` = RAM streaming (all blocks pre-loaded with LoRA fusion).
            gpu_slots_count: Number of GPU buffer slots.
                ``None`` = ``_DEFAULT_GPU_SLOTS`` (2).
        """
        if not self.blocks_prefix:
            raise ValueError("blocks_prefix must be non-empty for streaming")

        # 1. Create meta model (no weights allocated).
        config = read_model_config(self.model_path, self.model_loader)
        meta_model: nn.Module = create_meta_model(self.model_class_configurator, config, self.module_ops)
        if self.model_wrapper is not None:
            meta_model = self.model_wrapper(meta_model)
        meta_model.eval()

        blocks = resolve_attr(meta_model, self.blocks_attr)
        layout = build_pool_layout(blocks[0], dtype)

        # 2. Determine slot counts.
        cpu_slots_count = cpu_slots_count if cpu_slots_count is not None else len(blocks)
        gpu_slots_count = gpu_slots_count if gpu_slots_count is not None else _env_int(
            "LTX_STREAM_GPU_SLOTS",
            _DEFAULT_GPU_SLOTS,
        )

        # 3. Build source and load non-block weights.
        if cpu_slots_count >= len(blocks):
            source, lora_sources = self._build_pinned_source(meta_model, target_device, dtype, cpu_slots_count)
        else:
            source, lora_sources = self._build_disk_source(
                meta_model,
                layout,
                target_device,
                dtype,
                cpu_slots_count,
                staging_device=staging_device,
            )

        # 4. Create provider(s) and wrapper.
        resolved_block_devices = _resolve_block_devices(target_device, len(blocks), block_devices)
        provider_devices = _unique_devices(resolved_block_devices)
        if provider_devices != [target_device]:
            logger.info(
                "Using model-parallel block devices: %s",
                ",".join(str(device) for device in provider_devices),
            )
        providers: dict[torch.device, WeightsProvider] = {}
        prefetch_blocks = _env_int("LTX_STREAM_PREFETCH_BLOCKS", 0, minimum=0)
        for device in provider_devices:
            copy_stream = torch.cuda.Stream(device=device)
            gpu_pool = WeightPool(
                layout, gpu_slots_count, device, reuse_barrier=lambda event, stream=copy_stream: stream.wait_event(event)
            )
            providers[device] = WeightsProvider(
                gpu_pool,
                copy_stream,
                device,
                source,
                lora_sources,
                self.blocks_prefix,
                prefetch_blocks=prefetch_blocks,
                block_count=len(blocks),
                owns_source=len(provider_devices) == 1,
                owns_lora_sources=len(provider_devices) == 1,
            )
        provider: WeightsProvider | dict[torch.device, WeightsProvider]
        provider = providers[target_device] if provider_devices == [target_device] else providers
        return BlockStreamingWrapper(
            model=meta_model,
            blocks=blocks,
            provider=provider,
            target_device=target_device,
            block_devices=resolved_block_devices,
            shared_source=source if len(provider_devices) > 1 else None,
            shared_lora_sources=lora_sources if len(provider_devices) > 1 else None,
        )

    def _build_pinned_source(
        self,
        meta_model: nn.Module,
        target_device: torch.device,
        dtype: torch.dtype,
        cpu_slots_count: int,
        staging_device: torch.device | None = None,
    ) -> tuple[WeightSource, list[LoraSource]]:
        """Pre-load all blocks into pinned CPU buffers with LoRA fusion."""
        model_sd = load_state_dict(
            self.model_path, self.model_loader, self.registry, torch.device("cpu"), self.model_sd_ops
        )

        if self.loras:
            lora_sds = [
                load_state_dict([lora.path], self.model_loader, self.registry, torch.device("cpu"), lora.sd_ops)
                for lora in self.loras
            ]
            lora_sd_and_strengths = [
                LoraStateDictWithStrength(sd, lora.strength) for sd, lora in zip(lora_sds, self.loras, strict=True)
            ]
            model_sd = apply_loras(
                model_sd=model_sd,
                lora_sd_and_strengths=lora_sd_and_strengths,
                dtype=dtype,
                destination_sd=model_sd if isinstance(self.registry, DummyRegistry) else None,
            )

        # Partition: non-block weights go to GPU, block weights go directly
        # to pinned buffers.  This avoids holding the full state dict and
        # pinned copies simultaneously.
        non_block_sd: dict[str, torch.Tensor] = {}
        block_tensors: dict[int, dict[str, torch.Tensor]] = {}
        prefix_dot = self.blocks_prefix + "."
        non_block_load_batch = _env_int("LTX_NON_BLOCK_LOAD_BATCH", 1 if target_device.type == "cuda" else 64)

        cpu_stream_mode = os.environ.get("LTX_CPU_STREAM_MODE", "ram").strip().lower()
        use_pinned_blocks = cpu_stream_mode == "pinned"

        logger.info(
            "Building %s CPU streaming source on %s with %s checkpoint tensors",
            "pinned" if use_pinned_blocks else "pageable",
            target_device,
            len(model_sd.sd),
        )
        _log_cuda_memory(target_device, "Before partitioning pinned streaming source")

        for key, tensor in model_sd.sd.items():
            if key.startswith(prefix_dot):
                rest = key[len(prefix_dot) :]
                idx_str, _, param_name = rest.partition(".")
                try:
                    block_idx = int(idx_str)
                except ValueError:
                    non_block_sd[self.state_dict_prefix + key] = tensor
                    continue
                block_tensors.setdefault(block_idx, {})[param_name] = tensor
            else:
                non_block_sd[self.state_dict_prefix + key] = tensor

        logger.info(
            "Pinned CPU source partitioned: non_block_tensors=%s block_count=%s non_block_load_batch=%s",
            len(non_block_sd),
            len(block_tensors),
            non_block_load_batch,
        )
        _log_cuda_memory(target_device, "After partitioning pinned streaming source")

        gpu_batch: dict[str, torch.Tensor] = {}
        total_non_block = len(non_block_sd)
        for index, (state_key, tensor) in enumerate(non_block_sd.items(), start=1):
            gpu_batch[state_key] = tensor.to(device=target_device, dtype=dtype)
            if len(gpu_batch) >= non_block_load_batch:
                meta_model.load_state_dict(gpu_batch, strict=False, assign=True)
                gpu_batch.clear()
                if target_device.type == "cuda":
                    torch.cuda.synchronize(target_device)
                if index == non_block_load_batch or index % max(non_block_load_batch * 16, non_block_load_batch) == 0:
                    _log_cuda_memory(target_device, f"Pinned source loaded {index}/{total_non_block} non-block tensors")

        if gpu_batch:
            meta_model.load_state_dict(gpu_batch, strict=False, assign=True)
            gpu_batch.clear()
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)

        _log_cuda_memory(target_device, "After loading non-block weights into model")
        del model_sd, non_block_sd

        # Build block weights in CPU RAM. Pinned host memory is faster for H2D, but
        # 46GB-class checkpoints can exhaust the CUDA host registration budget on
        # Windows/8GB cards. Default to pageable RAM and allow opting into pinned
        # blocks through LTX_CPU_STREAM_MODE=pinned.
        cpu_blocks: dict[int, dict[str, torch.Tensor]] = {}
        for idx in range(cpu_slots_count):
            src = block_tensors.pop(idx)
            if use_pinned_blocks:
                cpu_blocks[idx] = {name: tensor.to(dtype=dtype).pin_memory() for name, tensor in src.items()}
            else:
                cpu_blocks[idx] = {name: tensor.to(dtype=dtype).contiguous() for name, tensor in src.items()}

        _log_cuda_memory(target_device, "After materializing CPU block weights")

        return (PinnedWeightSource(cpu_blocks) if use_pinned_blocks else RamWeightSource(cpu_blocks)), []

    def _build_disk_source(
        self,
        meta_model: nn.Module,
        layout: BlockLayout,
        target_device: torch.device,
        dtype: torch.dtype,
        cpu_slots_count: int,
        staging_device: torch.device | None = None,
    ) -> tuple[WeightSource, list[LoraSource]]:
        """Create a DiskWeightSource backed by a DiskBlockReader for lazy loading."""
        lora_sources = [LoraSource(lora.path, lora.sd_ops, lora.strength) for lora in self.loras]
        checkpoint_paths = list(self.model_path) if isinstance(self.model_path, tuple) else [self.model_path]
        reader = DiskTensorReader(checkpoint_paths)

        block_key_map: dict[int, list[tuple[str, str]]] = {}
        non_block_keys: list[tuple[str, str]] = []

        for sft_key in reader.keys():  # noqa: SIM118
            model_key = self.model_sd_ops.apply_to_key(sft_key) if self.model_sd_ops else sft_key
            if model_key is None:
                continue
            if model_key.startswith(self.blocks_prefix + "."):
                rest = model_key[len(self.blocks_prefix) + 1 :]
                idx_str, _, param_name = rest.partition(".")
                try:
                    block_idx = int(idx_str)
                except ValueError:
                    non_block_keys.append((sft_key, model_key))
                    continue
                block_key_map.setdefault(block_idx, []).append((sft_key, param_name))
            else:
                non_block_keys.append((sft_key, model_key))

        self._load_non_block_weights(
            reader,
            non_block_keys,
            meta_model,
            target_device,
            dtype,
            sd_ops=self.model_sd_ops,
            key_prefix=self.state_dict_prefix,
            lora_sources=lora_sources,
            matmul_device=target_device,
        )

        staging_device_name = os.environ.get("LTX_STREAM_STAGING_DEVICE")
        staging_device = staging_device or (torch.device(staging_device_name) if staging_device_name else torch.device("cpu"))
        if staging_device == target_device:
            staging_device = torch.device("cpu")
        if staging_device.type == "cuda" and target_device.type == "cuda":
            staging_idx = staging_device.index if staging_device.index is not None else torch.cuda.current_device()
            target_idx = target_device.index if target_device.index is not None else torch.cuda.current_device()
            if not torch.cuda.can_device_access_peer(target_idx, staging_idx):
                logger.warning(
                    "CUDA staging device %s cannot peer-copy to compute device %s; "
                    "falling back to pinned CPU staging",
                    staging_device,
                    target_device,
                )
                staging_device = torch.device("cpu")
        pin_memory = staging_device.type == "cpu"
        if staging_device.type == "cuda":
            logger.info(
                "Using CUDA staging device %s for disk-streamed block weights before compute device %s",
                staging_device,
                target_device,
            )

        cpu_pool = WeightPool(
            layout,
            cpu_slots_count,
            staging_device,
            reuse_barrier=lambda event: event.synchronize(),
            pin_memory=pin_memory,
        )
        cache_dir = os.environ.get("LTX_STREAM_BLOCK_CACHE_DIR")
        cache_namespace = block_cache_namespace(checkpoint_paths, self.blocks_prefix, dtype) if cache_dir else None
        if cache_dir:
            logger.info("Using disk block cache at %s/%s", cache_dir, cache_namespace)
        block_reader = DiskBlockReader(
            reader=reader,
            block_key_map=block_key_map,
            dtype=dtype,
            cache_dir=cache_dir,
            cache_namespace=cache_namespace,
        )
        source = DiskWeightSource(cpu_pool, block_reader)
        return source, lora_sources

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse_lora_delta(
        model_key: str,
        tensor: torch.Tensor,
        lora_sources: list[LoraSource],
        matmul_device: torch.device | None = None,
    ) -> torch.Tensor:
        """Add all matching LoRA deltas to *tensor* in-place."""
        if not lora_sources or not model_key.endswith(".weight"):
            return tensor
        prefix = model_key[: -len(".weight")]
        device = tensor.device if tensor.device.type == "cuda" else matmul_device
        for source in lora_sources:
            delta = source.get_delta(prefix, device=device)
            if delta is not None:
                tensor = tensor.add_(delta.to(device=tensor.device, dtype=tensor.dtype))
        return tensor

    @staticmethod
    @torch.inference_mode()
    def _load_non_block_weights(
        reader: DiskTensorReader,
        non_block_keys: list[tuple[str, str]],
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        sd_ops: SDOps | None = None,
        key_prefix: str = "",
        lora_sources: list[LoraSource] | None = None,
        matmul_device: torch.device | None = None,
    ) -> None:
        """Load non-block weights into *model* on *device*."""
        batch_size = _env_int("LTX_NON_BLOCK_LOAD_BATCH", 1 if device.type == "cuda" else 64)
        state_dict: dict[str, torch.Tensor] = {}
        sources = lora_sources or []

        logger.info(
            "Loading %s non-block tensors onto %s in batches of %s",
            len(non_block_keys),
            device,
            batch_size,
        )
        _log_cuda_memory(device, "Before loading non-block tensors")

        def flush_state_dict() -> None:
            if not state_dict:
                return
            model.load_state_dict(state_dict, strict=False, assign=True)
            state_dict.clear()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        for index, (sft_key, model_key) in enumerate(non_block_keys, start=1):
            tensor = reader.get_tensor(sft_key).to(device=device, dtype=dtype)
            tensor = StreamingModelBuilder._fuse_lora_delta(model_key, tensor, sources, matmul_device)
            if sd_ops is not None:
                for kv in sd_ops.apply_to_key_value(model_key, tensor):
                    state_dict[key_prefix + kv.new_key] = kv.new_value
            else:
                state_dict[key_prefix + model_key] = tensor

            if len(state_dict) >= batch_size:
                flush_state_dict()
                if index == batch_size or index % max(batch_size * 16, batch_size) == 0:
                    _log_cuda_memory(device, f"Loaded {index}/{len(non_block_keys)} non-block tensors")

        flush_state_dict()
        _log_cuda_memory(device, "Finished loading non-block tensors")
