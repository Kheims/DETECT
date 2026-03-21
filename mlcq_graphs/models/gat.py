from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from mlcq_graphs.models.base import BaseGraphClassifier
from mlcq_graphs.models.registry import register_model


@register_model("gat")
class GATGraphClassifier(BaseGraphClassifier):
    """Graph Attention Network (GAT) architecture using GATv2Conv.

    Uses multi-head attention with concatenation for all layers except the last,
    which averages heads to maintain hidden_dim output.
    """

    def __init__(
        self,
        num_node_types: int,
        type_emb_dim: int,
        num_numeric_feats: int,
        num_token_feats: int,
        hidden_dim: int,
        num_labels: int,
        dropout: float,
        num_layers: int = 2,
        num_graph_features: int = 0,
        num_heads: int = 4,
        use_type_features: bool = True,
        use_numeric_features: bool = True,
        use_token_features: bool = False,
    ) -> None:
        super().__init__(
            num_node_types=num_node_types,
            type_emb_dim=type_emb_dim,
            num_numeric_feats=num_numeric_feats,
            num_token_feats=num_token_feats,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout,
            num_layers=num_layers,
            num_graph_features=num_graph_features,
            use_type_features=use_type_features,
            use_numeric_features=use_numeric_features,
            use_token_features=use_token_features,
        )

        self.num_heads = num_heads

        # Create GAT layers dynamically
        self.convs = nn.ModuleList()

        if num_layers == 1:
            # Single layer: average heads to output hidden_dim
            self.convs.append(
                GATv2Conv(
                    self.in_dim,
                    hidden_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                )
            )
        else:
            # First layer: concat heads, output = hidden_dim
            self.convs.append(
                GATv2Conv(
                    self.in_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                )
            )

            # Middle layers (if num_layers > 2): concat heads
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        concat=True,
                        dropout=dropout,
                    )
                )

            # Last layer: average heads, output = hidden_dim
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                )
            )

        pool_dim = 2 * num_layers * hidden_dim + num_graph_features
        self.lin1 = nn.Linear(pool_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.compose_features(data)
        layer_outputs: list[torch.Tensor] = []

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = self.residual_proj(x) if i == 0 and self.residual_proj is not None else x
            x = conv(x, data.edge_index)
            x = bn(x)
            x = x + residual
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        x_jk = torch.cat(layer_outputs, dim=-1)
        g = torch.cat([global_mean_pool(x_jk, data.batch), global_max_pool(x_jk, data.batch)], dim=-1)

        graph_x = getattr(data, "graph_x", None)
        if graph_x is not None:
            g = torch.cat([g, graph_x], dim=-1)

        g = self.lin1(g)
        g = F.relu(g)
        g = F.dropout(g, p=self.dropout, training=self.training)
        return self.lin2(g)
