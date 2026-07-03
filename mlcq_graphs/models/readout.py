from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation, Set2Set

READOUTS = ("mean", "max", "add", "mean_max", "attention", "mean_attention", "set2set")


class GraphReadout(nn.Module):
    """Graph-level readout turning node embeddings into one vector per graph.

    Modes: mean, max, add, mean_max, attention, mean_attention, set2set.
    """

    def __init__(
        self,
        mode: str,
        hidden_dim: int,
        set2set_steps: int = 3,
        subset_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.subset_pooling = subset_pooling
        if mode not in READOUTS:
            raise ValueError(f"Unknown readout '{mode}'. Expected one of {READOUTS}")
        self.mode = mode
        self.attention: nn.Module | None = None
        self.set2set: nn.Module | None = None

        if mode in {"attention", "mean_attention"}:
            gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
                nn.ReLU(),
                nn.Linear(max(hidden_dim // 2, 1), 1),
            )
            self.attention = AttentionalAggregation(gate_nn=gate_nn)
        if mode == "set2set":
            self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps)

        doubled = {"mean_max", "mean_attention", "set2set"}
        self.out_dim = 2 * hidden_dim if mode in doubled else hidden_dim
        if subset_pooling:
            # mean and max over the subset nodes, appended to the main readout
            self.out_dim += 2 * hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        subset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        g = self._pool(x, batch)
        if not self.subset_pooling:
            return g
        if subset_mask is None:
            raise ValueError("subset pooling needs data.pool_mask")
        num_graphs = g.size(0)
        mask = subset_mask.bool()
        sub_x, sub_batch = x[mask], batch[mask]
        sub_mean = global_mean_pool(sub_x, sub_batch, size=num_graphs)
        sub_max = global_max_pool(sub_x, sub_batch, size=num_graphs)
        # graphs without subset nodes get zeros instead of -inf from max pooling
        has_subset = torch.zeros(num_graphs, dtype=torch.bool, device=x.device)
        has_subset[sub_batch] = True
        sub_max = torch.where(has_subset.unsqueeze(-1), sub_max, torch.zeros_like(sub_max))
        return torch.cat([g, sub_mean, sub_max], dim=-1)

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return global_mean_pool(x, batch)
        if self.mode == "max":
            return global_max_pool(x, batch)
        if self.mode == "add":
            return global_add_pool(x, batch)
        if self.mode == "mean_max":
            return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        if self.mode == "attention":
            return self.attention(x, batch)
        if self.mode == "mean_attention":
            return torch.cat([global_mean_pool(x, batch), self.attention(x, batch)], dim=-1)
        return self.set2set(x, batch)
