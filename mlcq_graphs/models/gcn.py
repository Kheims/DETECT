from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from mlcq_graphs.models.base import BaseGraphClassifier
from mlcq_graphs.models.registry import register_model


@register_model("gcn")
class GCNGraphClassifier(BaseGraphClassifier):
    """Graph Convolutional Network (GCN) architecture.

    Refactored from the original implementation with dynamic layer creation
    via nn.ModuleList. With default num_layers=2, produces identical behavior
    to the original hardcoded GCNGraphClassifier.
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
    ) -> None:
        """Initialize GCN graph classifier.

        Args:
            num_node_types: Number of unique node types for embedding
            type_emb_dim: Dimension of type embeddings
            num_numeric_feats: Number of numeric node features
            num_token_feats: Number of token features
            hidden_dim: Hidden dimension for GCN layers
            num_labels: Number of output labels
            dropout: Dropout probability
            num_layers: Number of GCN layers (default: 2)
            use_type_features: Whether to use type embeddings
            use_numeric_features: Whether to use numeric features
            use_token_features: Whether to use token features
        """
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

        # Create GCN layers dynamically
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.in_dim, hidden_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Readout and classifier head
        self.build_head()

    def encode_nodes(self, data: Data) -> torch.Tensor:
        """Message passing through GCN.

        Args:
            data: PyG Data object with node features and edges

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Compose features from base class
        x = self.compose_features(data)

        # Apply GCN layers: conv -> relu -> dropout
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x
