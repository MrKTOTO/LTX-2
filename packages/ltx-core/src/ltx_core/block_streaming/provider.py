"""GPU weights provider for block streaming."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import os
from threading import Lock
import time

import torch

from ltx_core.block_streaming.disk import LoraSource
from ltx_core.block_streaming.pool import WeightPool
from ltx_core.block_streaming.source import WeightSource


class WeightsProvider:
    """Provides GPU-ready block weights via H2D copy from a pinned CPU weight source.
    Args:
        pool: Pre-allocated GPU weight buffer pool.
        copy_stream: Dedicated CUDA stream for async H2D copies.
        target_device: GPU device for compute.
        source: Pinned CPU weight source.
        lora_sources: LoRA adapters fused on H2D copy.
        blocks_prefix: State-dict prefix for LoRA key matching.
    """

    def __init__(
        self,
        pool: WeightPool,
        copy_stream: torch.cuda.Stream,
        target_device: torch.device,
        source: WeightSource,
        lora_sources: list[LoraSource] | None = None,
        blocks_prefix: str = "",
        prefetch_blocks: int = 1,
        block_count: int | None = None,
        owns_source: bool = True,
        owns_lora_sources: bool = True,
    ) -> None:
        self._copy_stream = copy_stream
        self._pool = pool
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._events: dict[int, torch.cuda.Event] = {}
        self._ready_events: dict[int, torch.cuda.Event] = {}
        self._in_use: set[int] = set()
        self._target_device = target_device
        self._source = source
        self._lora_sources = lora_sources or []
        self._owns_source = owns_source
        self._owns_lora_sources = owns_lora_sources
        self._blocks_prefix = blocks_prefix
        self._prefetch_blocks = prefetch_blocks
        self._block_count = block_count
        self._lock = Lock()
        gpu_prefetch_workers = max(0, int(os.environ.get("LTX_STREAM_GPU_PREFETCH_WORKERS", "1")))
        self._gpu_prefetch_blocks = (
            max(0, min(prefetch_blocks, self._pool.capacity - 1)) if gpu_prefetch_workers > 0 else 0
        )
        self._executor = (
            ThreadPoolExecutor(
                max_workers=gpu_prefetch_workers,
                thread_name_prefix="ltx-gpu-prefetch",
            )
            if gpu_prefetch_workers > 0
            else None
        )
        self._prefetches: dict[int, Future[None]] = {}

    def prime(self) -> None:
        """Start loading the first few blocks before the first forward hook."""
        warm_blocks = min(self._prefetch_blocks + 1, self._block_count or (self._prefetch_blocks + 1))
        for idx in range(warm_blocks):
            self._source.prefetch(idx)
            if idx < self._pool.capacity and self._executor is not None:
                self._ensure_gpu_prefetch(idx)

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        """Return GPU weights for block *idx*, waiting for any async prefetch."""
        with self._lock:
            future = None if idx in self._cache else self._prefetches.get(idx)

        if future is not None:
            future.result()
        else:
            with self._lock:
                cache_hit = idx in self._cache
            if not cache_hit:
                self._load_to_gpu(idx)

        with self._lock:
            gpu_weights = self._cache[idx]
            ready_event = self._ready_events.get(idx)
            self._in_use.add(idx)

        if ready_event is not None:
            torch.cuda.current_stream(self._target_device).wait_event(ready_event)
        self._prefetch(idx)
        return gpu_weights

    def _prefetch(self, idx: int) -> None:
        for offset in range(1, self._prefetch_blocks + 1):
            prefetch_idx = idx + offset
            if self._block_count is not None and prefetch_idx >= self._block_count:
                break
            self._source.prefetch(prefetch_idx, min_keep_idx=idx)
            if offset <= self._gpu_prefetch_blocks and self._executor is not None:
                self._ensure_gpu_prefetch(prefetch_idx)

    def _ensure_gpu_prefetch(self, idx: int) -> Future[None] | None:
        if self._block_count is not None and (idx < 0 or idx >= self._block_count):
            return None
        if self._executor is None:
            return None
        with self._lock:
            if idx in self._cache:
                return None
            future = self._prefetches.get(idx)
            if future is not None:
                return future
            future = self._executor.submit(self._load_to_gpu, idx)
            self._prefetches[idx] = future
            return future

    def _load_to_gpu(self, idx: int) -> None:
        try:
            cpu_weights = self._source.get(idx)
            gpu_weights = self._acquire_gpu_slot()
            h2d_event = self._copy_to_gpu(idx, gpu_weights, cpu_weights)
            self._source.release(idx, event=h2d_event)
            with self._lock:
                self._cache[idx] = gpu_weights
                self._ready_events[idx] = h2d_event
                self._cache.move_to_end(idx)
                self._prefetches.pop(idx, None)
        except Exception:
            with self._lock:
                self._prefetches.pop(idx, None)
            raise

    def _acquire_gpu_slot(self) -> dict[str, torch.Tensor]:
        while True:
            with self._lock:
                while len(self._cache) >= self._pool.capacity:
                    evicted_idx = None
                    for candidate_idx in self._cache:
                        if candidate_idx not in self._in_use:
                            evicted_idx = candidate_idx
                            break
                    if evicted_idx is None:
                        break
                    evicted_weights = self._cache.pop(evicted_idx)
                    ready_event = self._ready_events.pop(evicted_idx, None)
                    compute_event = self._events.pop(evicted_idx, None)
                    self._pool.release(evicted_weights, event=compute_event or ready_event)
                if self._pool.free_count > 0:
                    return self._pool.acquire()
            time.sleep(0.001)

    def _copy_to_gpu(
        self,
        idx: int,
        gpu_weights: dict[str, torch.Tensor],
        cpu_weights: dict[str, torch.Tensor],
    ) -> torch.cuda.Event:
        """Enqueue H2D copy + LoRA fusion on the copy stream and wait on compute.
        The wait is intentionally inside this method so callers -- and
        instrumentation regions wrapping it -- observe the full transfer time.
        """
        with torch.inference_mode(), torch.cuda.stream(self._copy_stream):
            for name, gpu_tensor in gpu_weights.items():
                gpu_tensor.copy_(cpu_weights[name], non_blocking=True)
            if self._lora_sources:
                self._fuse_block_loras(idx, gpu_weights)
            h2d_event = torch.cuda.Event()
            h2d_event.record(self._copy_stream)

        return h2d_event

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        """Attach a compute-done event -- waited before this buffer is recycled."""
        with self._lock:
            self._in_use.discard(idx)
            self._events[idx] = event

    def cleanup(self) -> None:
        """Synchronize streams and release all resources."""
        for future in self._prefetches.values():
            future.cancel()
        self._prefetches.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._copy_stream.synchronize()
        torch.cuda.current_stream(self._target_device).synchronize()
        self._cache.clear()
        self._events.clear()
        self._ready_events.clear()
        self._in_use.clear()
        self._pool.cleanup()
        if self._owns_source:
            self._source.cleanup()
        if self._owns_lora_sources:
            for lora in self._lora_sources:
                lora.cleanup()

    def __len__(self) -> int:
        return len(self._cache)

    def _fuse_block_loras(self, idx: int, weights: dict[str, torch.Tensor]) -> None:
        """Fuse LoRA deltas directly into GPU block weights."""
        for name, tensor in weights.items():
            if not name.endswith(".weight"):
                continue
            full_key = f"{self._blocks_prefix}.{idx}.{name}"
            prefix = full_key[: -len(".weight")]
            for source in self._lora_sources:
                delta = source.get_delta(prefix, device=self._target_device)
                if delta is not None:
                    tensor.add_(delta.to(dtype=tensor.dtype))
