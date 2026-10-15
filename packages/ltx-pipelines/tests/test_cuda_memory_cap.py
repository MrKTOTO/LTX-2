"""The CUDA allocator cap follows the VRAM free at startup minus the configured headroom."""

import pytest
import torch

from ltx_pipelines.utils.helpers import cap_cuda_memory_to_free_vram

GIB = 1024**3


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (7 * GIB, 8 * GIB))
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: calls.append((fraction, device))
    )
    return calls


def test_cap_is_free_vram_minus_headroom(fake_cuda: list, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LTX_CUDA_MEMORY_HEADROOM_MB", "512")
    cap_cuda_memory_to_free_vram()
    ((fraction, device),) = fake_cuda
    assert fraction == pytest.approx(6.5 / 8)
    assert device == torch.device("cuda", 0)


def test_no_cap_without_headroom_setting(fake_cuda: list, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LTX_CUDA_MEMORY_HEADROOM_MB", raising=False)
    cap_cuda_memory_to_free_vram()
    assert fake_cuda == []
