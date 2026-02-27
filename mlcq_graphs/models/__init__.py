"""Multi-architecture GNN model library.

This package provides a flexible model abstraction layer with:
- BaseGraphClassifier: Abstract base class with shared feature composition
- register_model: Decorator for registering architectures
- get_model: Factory function for instantiating models by name
- MODEL_REGISTRY: Global registry of available architectures

Supported architectures will be registered when their modules are imported.
"""

from __future__ import annotations

from mlcq_graphs.models.base import BaseGraphClassifier
from mlcq_graphs.models.registry import MODEL_REGISTRY, get_model, register_model

# Import architecture modules to trigger @register_model decorators
from mlcq_graphs.models.gat import GATGraphClassifier
from mlcq_graphs.models.gcn import GCNGraphClassifier
from mlcq_graphs.models.graphsage import GraphSAGEClassifier

__all__ = [
    "BaseGraphClassifier",
    "get_model",
    "MODEL_REGISTRY",
    "register_model",
    "GCNGraphClassifier",
    "GATGraphClassifier",
    "GraphSAGEClassifier",
]
