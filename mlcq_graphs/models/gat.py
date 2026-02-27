from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

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
        num_heads: int = 4,
        use_type_features: bool = True,
        use_numeric_features: bool = True,
        use_token_features: bool = False,
    ) -> None:
        """Initialize GAT graph classifier.

        Args:
            num_node_types: Number of unique node types for embedding
            type_emb_dim: Dimension of type embeddings
            num_numeric_feats: Number of numeric node features
            num_token_feats: Number of token features
            hidden_dim: Hidden dimension for GAT layers
            num_labels: Number of output labels
            dropout: Dropout probability
            num_layers: Number of GAT layers (default: 2)
            num_heads: Number of attention heads (default: 4)
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

        # Classifier head
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through GAT.

        Args:
            data: PyG Data object with node features and edges

        Returns:
            Logits tensor [batch_size, num_labels]
        """
        # Compose features from base class
        x = self.compose_features(data)

        # Apply GAT layers: conv -> relu -> dropout
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling and classification
        g = global_mean_pool(x, data.batch)
        g = self.lin1(g)
        g = F.relu(g)
        g = F.dropout(g, p=self.dropout, training=self.training)
        return self.lin2(g)
