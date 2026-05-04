import os

import torch

from ltx_core.model.transformer.gelu_approx import GELUApprox


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def _chunk_tokens() -> int:
        try:
            return max(1, int(os.environ.get("LTX_FEED_FORWARD_CHUNK_TOKENS", "1024")))
        except ValueError:
            return 1024

    def add_to_residual(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run FFN in token chunks and add results into residual in-place."""
        chunk_tokens = self._chunk_tokens()
        tokens = x.shape[1]
        for start in range(0, tokens, chunk_tokens):
            stop = min(start + chunk_tokens, tokens)
            out = self.net(x[:, start:stop, :])
            if gate is not None:
                out.mul_(gate[:, start:stop, :])
            residual[:, start:stop, :].add_(out)
            del out
        return residual
