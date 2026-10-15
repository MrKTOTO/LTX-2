"""Every CPU weight source of the streaming builder must serve the checkpoint's weights."""

import pytest
import torch
from safetensors.torch import save_file

from ltx_core.block_streaming import StreamingModelBuilder
from ltx_core.block_streaming.source import MappedWeightSource, PinnedWeightSource


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 8)
        self.blocks = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU()) for _ in range(3))
        self.scale = torch.nn.Parameter(torch.ones(8))

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return x * self.scale


class _Configurator:
    @classmethod
    def from_metadata(cls, _metadata: dict) -> _Net:
        return _Net()


@pytest.fixture
def checkpoint(tmp_path: object) -> tuple[str, _Net]:
    torch.manual_seed(0)
    reference = _Net()
    with torch.no_grad():
        for param in reference.parameters():
            param.normal_()
    state = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    # Store one block weight as float32 and the rest as bfloat16, like LTX checkpoints that
    # keep small tables in float32.
    state = {key: value if key == "blocks.1.0.bias" else value.to(torch.bfloat16) for key, value in state.items()}
    path = tmp_path / "net.safetensors"
    save_file(state, str(path), metadata={"config": "{}"})
    expected = _Net()
    expected.load_state_dict({key: value.to(torch.float32) for key, value in state.items()})
    return str(path), expected


@pytest.mark.parametrize(("mode", "source_type"), [("ram", PinnedWeightSource), ("mmap", MappedWeightSource)])
def test_cpu_sources_serve_checkpoint_weights(
    checkpoint: tuple[str, _Net], mode: str, source_type: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LTX_CPU_STREAM_MODE", mode)
    path, expected = checkpoint
    builder = StreamingModelBuilder(
        model_class_configurator=_Configurator, model_path=path, blocks_attr="blocks", blocks_prefix="blocks"
    )
    wrapper = builder.build(device=torch.device("cpu"), dtype=torch.bfloat16)
    try:
        assert isinstance(wrapper._provider._source, source_type)
        x = torch.randn(2, 4)
        with torch.inference_mode():
            got = wrapper(x.to(torch.bfloat16))
            want = expected.to(torch.bfloat16)(x.to(torch.bfloat16))
            # Reuse after every block was evicted once (3 blocks, 2 GPU slots).
            again = wrapper(x.to(torch.bfloat16))
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        torch.testing.assert_close(again, want, rtol=0, atol=0)
    finally:
        wrapper.teardown()
