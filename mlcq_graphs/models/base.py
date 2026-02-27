from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch_geometric.data import Data


class BaseGraphClassifier(ABC, nn.Module):
    """Abstract base class for graph classification models.

    All GNN architectures share identical feature composition logic
    (type embeddings, numeric features, token embeddings) via compose_features().
    Subclasses implement architecture-specific message passing in forward().
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
    ) -> None:
        """Initialize base graph classifier.

        Args:
            num_node_types: Number of unique node types for embedding
            type_emb_dim: Dimension of type embeddings
            num_numeric_feats: Number of numeric node features
            num_token_feats: Number of token features
            hidden_dim: Hidden dimension for GNN layers
            num_labels: Number of output labels
            dropout: Dropout probability
            num_layers: Number of GNN layers (default: 2)
            use_type_features: Whether to use type embeddings
            use_numeric_features: Whether to use numeric features
            use_token_features: Whether to use token features

        Raises:
            ValueError: If no feature sources are enabled or num_layers < 1
        """
        super().__init__()

        # Validate at least one feature source is enabled
        if not (use_type_features or use_numeric_features or use_token_features):
            raise ValueError(
                "At least one feature source must be enabled "
                "(use_type_features, use_numeric_features, or use_token_features)"
            )

        # Validate num_layers
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        # Store configuration
        self.use_type_features = use_type_features
        self.use_numeric_features = use_numeric_features
        self.use_token_features = use_token_features
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout = dropout
        self.num_layers = num_layers

        # Set up type embeddings if enabled
        self.type_emb = nn.Embedding(num_node_types, type_emb_dim) if use_type_features else None

        # Compute input dimension from enabled features
        self.in_dim = 0
        if use_type_features:
            self.in_dim += type_emb_dim
        if use_numeric_features:
            self.in_dim += num_numeric_feats
        if use_token_features:
            self.in_dim += num_token_feats

    def compose_features(self, data: Data) -> torch.Tensor:
        """Extract and concatenate enabled features from PyG Data object.

        This method implements the exact feature composition logic from the
        original GCNGraphClassifier to preserve behavioral compatibility.

        Args:
            data: PyG Data object containing node features

        Returns:
            Composed feature tensor [num_nodes, in_dim]

        Raises:
            ValueError: If required feature tensor is missing
        """
        feature_parts: list[torch.Tensor] = []

        if self.use_type_features and self.type_emb is not None:
            if data.type_id is None:
                raise ValueError("Missing type_id tensor for type features.")
            feature_parts.append(self.type_emb(data.type_id))

        if self.use_numeric_features:
            if data.x is None:
                raise ValueError("Missing x tensor for numeric features.")
            feature_parts.append(data.x)

        if self.use_token_features:
            token_x = getattr(data, "token_x", None)
            if token_x is None:
                raise ValueError("Missing token_x tensor for token features.")
            feature_parts.append(token_x)

        x = feature_parts[0] if len(feature_parts) == 1 else torch.cat(feature_parts, dim=-1)
        return x

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the model.

        Subclasses implement architecture-specific message passing.

        Args:
            data: PyG Data object

        Returns:
            Logits tensor [batch_size, num_labels]
        """
        pass
