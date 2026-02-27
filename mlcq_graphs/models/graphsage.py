from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool

from mlcq_graphs.models.base import BaseGraphClassifier
from mlcq_graphs.models.registry import register_model


@register_model("graphsage")
class GraphSAGEClassifier(BaseGraphClassifier):
    """GraphSAGE architecture using SAGEConv.

    Supports configurable aggregation functions (mean, max, lstm).
    Default aggregation is mean.
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
        aggregation: str = "mean",
        use_type_features: bool = True,
        use_numeric_features: bool = True,
        use_token_features: bool = False,
    ) -> None:
        """Initialize GraphSAGE graph classifier.

        Args:
            num_node_types: Number of unique node types for embedding
            type_emb_dim: Dimension of type embeddings
            num_numeric_feats: Number of numeric node features
            num_token_feats: Number of token features
            hidden_dim: Hidden dimension for SAGE layers
            num_labels: Number of output labels
            dropout: Dropout probability
            num_layers: Number of SAGE layers (default: 2)
            aggregation: Aggregation function ('mean', 'max', 'lstm') (default: 'mean')
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

        # Create GraphSAGE layers dynamically
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(self.in_dim, hidden_dim, aggr=aggregation))
        for _ in range(self.num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregation))

        # Classifier head
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through GraphSAGE.

        Args:
            data: PyG Data object with node features and edges

        Returns:
            Logits tensor [batch_size, num_labels]
        """
        # Compose features from base class
        x = self.compose_features(data)

        # Apply SAGE layers: conv -> relu -> dropout
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
