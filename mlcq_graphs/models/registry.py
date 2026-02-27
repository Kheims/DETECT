from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlcq_graphs.models.base import BaseGraphClassifier

# Module-level registry mapping architecture names to model classes
MODEL_REGISTRY: dict[str, type[BaseGraphClassifier]] = {}


def register_model(name: str):
    """Decorator to register a model architecture in the global registry.

    Args:
        name: Architecture name ('gcn', 'gat', 'graphsage')

    Returns:
        Decorator function

    Raises:
        ValueError: If name is already registered

    Example:
        @register_model("gcn")
        class GCNGraphClassifier(BaseGraphClassifier):
            ...
    """

    def decorator(cls: type[BaseGraphClassifier]) -> type[BaseGraphClassifier]:
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Registered models: {list(MODEL_REGISTRY.keys())}"
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs) -> BaseGraphClassifier:
    """Factory function to instantiate a model by name.

    Args:
        name: Architecture name ( 'gcn', 'gat', 'graphsage')
        **kwargs: Model constructor arguments

    Returns:
        Instantiated model

    Raises:
        ValueError: If name is not found in registry

    Example:
        model = get_model('gcn', num_node_types=10, type_emb_dim=16, ...)
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available architectures: {available}"
        )
    model_class = MODEL_REGISTRY[name]
    return model_class(**kwargs)
