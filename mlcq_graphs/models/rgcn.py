from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

from mlcq_graphs.models.base import BaseGraphClassifier
from mlcq_graphs.models.registry import register_model


@register_model("rgcn")
class RGCNGraphClassifier(BaseGraphClassifier):
    """Relational GCN with one weight matrix per edge type.

    Expects data.edge_type, see training.edge_type_mode.
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
        use_type_features: bool = True,
        use_numeric_features: bool = True,
        use_token_features: bool = False,
        pooling: str = "mean",
        set2set_steps: int = 3,
        subset_pooling: bool = False,
        num_graph_feats: int = 0,
        num_relations: int = 1,
        num_bases: int | None = None,
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
            use_type_features=use_type_features,
            use_numeric_features=use_numeric_features,
            use_token_features=use_token_features,
            pooling=pooling,
            set2set_steps=set2set_steps,
            subset_pooling=subset_pooling,
            num_graph_feats=num_graph_feats,
        )
        self.num_relations = num_relations

        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(self.in_dim, hidden_dim, num_relations, num_bases=num_bases))
        for _ in range(self.num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases))

        # Readout and classifier head
        self.build_head()

    def encode_nodes(self, data: Data) -> torch.Tensor:
        """Message passing through R-GCN.

        Args:
            data: PyG Data object with node features, edges and edge_type

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        edge_type = getattr(data, "edge_type", None)
        if edge_type is None:
            raise ValueError("rgcn needs data.edge_type (dataset.store_edge_types: true)")

        x = self.compose_features(data)
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
