"""Weight sources for block streaming: protocol and implementations."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import os
from threading import Lock
from typing import Protocol

import torch

from ltx_core.block_streaming.disk import DiskBlockReader
from ltx_core.block_streaming.pool import WeightPool


class WeightSource(Protocol):
    """Provides pinned CPU weights for a given block index."""

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        """Return CPU weights for block *idx*."""
        ...

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        """Signal that an async operation using these weights is guarded by *event*."""
        ...

    def cleanup(self) -> None:
        """Release all resources (buffers, readers, events)."""
        ...

    def prefetch(self, idx: int, min_keep_idx: int | None = None) -> None:
        """Start loading weights for block *idx* if supported."""
        ...


class DiskWeightSource(WeightSource):
    """Reads block weights from disk into pinned CPU buffers on demand."""

    def __init__(self, pool: WeightPool, reader: DiskBlockReader) -> None:
        self._pool = pool
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._events: dict[int, torch.cuda.Event] = {}
        self._reader = reader
        prefetch_workers = max(1, int(os.environ.get("LTX_STREAM_PREFETCH_WORKERS", "1")))
        self._executor = ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="ltx-disk-prefetch")
        self._prefetches: dict[int, Future[dict[str, torch.Tensor]]] = {}
        self._lock = Lock()

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        """Return CPU weights for block *idx*. Reads from disk on miss."""
        with self._lock:
            if idx in self._cache:
                return self._cache[idx]
            future = self._prefetches.pop(idx, None)
        if future is not None:
            weights = future.result()
            with self._lock:
                self._cache[idx] = weights
            return weights

        weights = self._read_block(idx)
        with self._lock:
            self._cache[idx] = weights
        return weights

    def _acquire_slot_locked(
        self,
        allow_evict: bool = True,
        evict_before_idx: int | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if allow_evict and self._pool.free_count == 0 and self._cache:
            evicted_idx = None
            if evict_before_idx is not None:
                for candidate_idx in self._cache:
                    if candidate_idx < evict_before_idx:
                        evicted_idx = candidate_idx
                        break
            else:
                evicted_idx = next(iter(self._cache))
            if evicted_idx is not None:
                evicted_weights = self._cache.pop(evicted_idx)
                self._pool.release(evicted_weights, event=self._events.pop(evicted_idx, None))
        if self._pool.free_count == 0:
            return None
        return self._pool.acquire()

    def _fill_block(self, idx: int, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            self._reader.read_into(weights, idx)
        return weights

    def _read_block(self, idx: int) -> dict[str, torch.Tensor]:
        with self._lock:
            weights = self._acquire_slot_locked()
            if weights is None:
                raise RuntimeError(f"No free streaming weight buffers available for block {idx}")
        return self._fill_block(idx, weights)

    def prefetch(self, idx: int, min_keep_idx: int | None = None) -> None:
        if not self._reader.has_block(idx):
            return
        with self._lock:
            if idx in self._cache or idx in self._prefetches:
                return
            weights = self._acquire_slot_locked(
                allow_evict=min_keep_idx is not None,
                evict_before_idx=min_keep_idx,
            )
            if weights is None:
                return
            self._prefetches[idx] = self._executor.submit(self._fill_block, idx, weights)

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        """Attach an H2D event -- waited before this buffer is recycled."""
        with self._lock:
            self._events[idx] = event

    def cleanup(self) -> None:
        """Clear cache and close the disk reader."""
        for future in self._prefetches.values():
            future.cancel()
        self._prefetches.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._cache.clear()
        self._events.clear()
        self._pool.cleanup()
        self._reader.cleanup()

    def __len__(self) -> int:
        return len(self._cache)


class PinnedWeightSource(WeightSource):
    """Pre-loaded pinned CPU weights."""

    def __init__(self, weights: dict[int, dict[str, torch.Tensor]]) -> None:
        self._weights = weights

    def get(self, idx: int) -> dict[str, torch.Tensor]:
        return self._weights[idx]

    def release(self, idx: int, event: torch.cuda.Event) -> None:
        pass

    def cleanup(self) -> None:
        self._weights.clear()

    def prefetch(self, idx: int, min_keep_idx: int | None = None) -> None:
        pass

    def __len__(self) -> int:
        return len(self._weights)
