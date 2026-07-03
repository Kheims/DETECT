from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from mlcq_graphs.models.base import BaseGraphClassifier
from mlcq_graphs.models.registry import register_model


@register_model("node_mlp")
class NodeMLPClassifier(BaseGraphClassifier):
    """GCN without message passing: each layer is a per-node linear map, edges are ignored."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = nn.ModuleList([nn.Linear(self.in_dim, self.hidden_dim)])
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.build_head()

    def encode_nodes(self, data: Data) -> torch.Tensor:
        x = self.compose_features(data)
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=self.dropout, training=self.training)
        return x
