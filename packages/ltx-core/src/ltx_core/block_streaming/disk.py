"""Safetensors I/O and LoRA fusion for block streaming."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import safetensors
import torch
from collections import OrderedDict
from contextlib import nullcontext
from safetensors.torch import save_file
from threading import Lock

from ltx_core.loader.sd_ops import SDOps


class DiskTensorReader:
    """Key-based tensor accessor over one or more safetensors files."""

    def __init__(self, paths: list[str]) -> None:
        self._handles: list[safetensors.safe_open] = []
        self._key_to_handle_idx: dict[str, int] = {}
        for path in paths:
            handle = safetensors.safe_open(path, framework="pt", device="cpu")
            handle_idx = len(self._handles)
            self._handles.append(handle)
            for sft_key in handle.keys():  # noqa: SIM118
                self._key_to_handle_idx[sft_key] = handle_idx

    def keys(self) -> list[str]:
        return list(self._key_to_handle_idx.keys())

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._handles[self._key_to_handle_idx[key]].get_tensor(key)

    def close(self) -> None:
        self._handles.clear()
        self._key_to_handle_idx.clear()


class DiskBlockReader:
    """Reads one block at a time from safetensors into provided buffers.
    Maps block indices to safetensors keys via a pre-computed key map.
    """

    def __init__(
        self,
        reader: DiskTensorReader,
        block_key_map: dict[int, list[tuple[str, str]]],
        dtype: torch.dtype,
        cache_dir: str | None = None,
        cache_namespace: str | None = None,
    ) -> None:
        self._reader = reader
        self._block_key_map = block_key_map
        self._dtype = dtype
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._cache_namespace = cache_namespace or "default"
        self._max_cache_handles = max(0, int(os.environ.get("LTX_STREAM_CACHE_HANDLES", "4")))
        self._cache_handles: OrderedDict[int, safetensors.safe_open] = OrderedDict()
        self._cache_handles_lock = Lock()
        if self._cache_dir:
            (self._cache_dir / self._cache_namespace).mkdir(parents=True, exist_ok=True)

    def read_into(self, target: dict[str, torch.Tensor], block_idx: int) -> None:
        if self._read_cached_into(target, block_idx):
            return

        target_device = next(iter(target.values())).device
        context = torch.cuda.device(target_device) if target_device.type == "cuda" else nullcontext()
        with context:
            for sft_key, param_name in self._block_key_map[block_idx]:
                tensor = self._reader.get_tensor(sft_key)
                if tensor.dtype != self._dtype:
                    tensor = tensor.to(self._dtype)
                target[param_name].copy_(tensor, non_blocking=target_device.type == "cuda")

    def has_block(self, block_idx: int) -> bool:
        return block_idx in self._block_key_map

    def cleanup(self) -> None:
        self._cache_handles.clear()
        self._reader.close()

    def _cache_path(self, block_idx: int) -> Path:
        assert self._cache_dir is not None
        return self._cache_dir / self._cache_namespace / f"block_{block_idx:03d}.safetensors"

    def _manifest_path(self) -> Path:
        assert self._cache_dir is not None
        return self._cache_dir / self._cache_namespace / "manifest.json"

    def _read_cached_into(self, target: dict[str, torch.Tensor], block_idx: int) -> bool:
        if self._cache_dir is None:
            return False

        path = self._cache_path(block_idx)
        if not path.exists():
            self._write_cache_file(path, block_idx)

        with self._cache_handles_lock:
            handle = self._cache_handles.pop(block_idx, None)
        if handle is None or self._max_cache_handles == 0:
            handle = safetensors.safe_open(str(path), framework="pt", device="cpu")

        target_device = next(iter(target.values())).device
        context = torch.cuda.device(target_device) if target_device.type == "cuda" else nullcontext()
        with context:
            for param_name in handle.keys():  # noqa: SIM118
                tensor = handle.get_tensor(param_name)
                target[param_name].copy_(tensor, non_blocking=target_device.type == "cuda")
        if self._max_cache_handles > 0:
            with self._cache_handles_lock:
                self._cache_handles[block_idx] = handle
                while len(self._cache_handles) > self._max_cache_handles:
                    self._cache_handles.popitem(last=False)
        return True

    def _write_cache_file(self, path: Path, block_idx: int) -> None:
        tmp_path = path.with_suffix(".tmp")
        tensors = {}
        for sft_key, param_name in self._block_key_map[block_idx]:
            tensor = self._reader.get_tensor(sft_key)
            if tensor.dtype != self._dtype:
                tensor = tensor.to(self._dtype)
            tensors[param_name] = tensor.contiguous()

        metadata = {"block_idx": str(block_idx), "dtype": str(self._dtype)}
        save_file(tensors, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, path)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "namespace": self._cache_namespace,
            "dtype": str(self._dtype),
            "blocks": sorted(self._block_key_map),
        }
        manifest_path = self._manifest_path()
        tmp_path = manifest_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp_path, manifest_path)


def block_cache_namespace(checkpoint_paths: list[str], blocks_prefix: str, dtype: torch.dtype) -> str:
    hasher = hashlib.sha1()
    hasher.update(blocks_prefix.encode("utf-8"))
    hasher.update(str(dtype).encode("utf-8"))
    for checkpoint_path in checkpoint_paths:
        path = Path(checkpoint_path)
        stat = path.stat()
        hasher.update(str(path.resolve()).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("ascii"))
        hasher.update(str(int(stat.st_mtime)).encode("ascii"))
    return hasher.hexdigest()[:16]


class LoraSource:
    """Pinned-memory cache of LoRA A/B matrices for on-the-fly fusion.
    At init, loads all matched A/B pairs into pinned CPU memory.
    :meth:`get_delta` computes ``(B * strength) @ A`` on the given device.
    """

    def __init__(self, path: str, sd_ops: SDOps | None, strength: float) -> None:
        self.strength = strength

        # param_prefix -> (pinned_a, pinned_b)
        self._pinned_ab: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        a_keys: dict[str, str] = {}
        b_keys: dict[str, str] = {}
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            # First pass: build key map.
            for sft_key in handle.keys():  # noqa: SIM118
                model_key = sd_ops.apply_to_key(sft_key) if sd_ops is not None else sft_key
                if model_key is None:
                    continue
                if model_key.endswith(".lora_A.weight"):
                    a_keys[model_key[: -len(".lora_A.weight")]] = sft_key
                elif model_key.endswith(".lora_B.weight"):
                    b_keys[model_key[: -len(".lora_B.weight")]] = sft_key

            # Second pass: load and pin matched A+B pairs (orphans silently skipped).
            for prefix in a_keys.keys() & b_keys.keys():
                self._pinned_ab[prefix] = (
                    handle.get_tensor(a_keys[prefix]).pin_memory(),
                    handle.get_tensor(b_keys[prefix]).pin_memory(),
                )

    def get_delta(self, param_prefix: str, device: torch.device | None = None) -> torch.Tensor | None:
        """Return ``(B * strength) @ A`` for *param_prefix*, or ``None``."""
        pair = self._pinned_ab.get(param_prefix)
        if pair is None:
            return None
        a, b = pair
        if device is not None and device.type == "cuda":
            a = a.to(device=device)
            b = b.to(device=device)
        delta = torch.matmul(b * self.strength, a)
        return delta

    def cleanup(self) -> None:
        self._pinned_ab.clear()
