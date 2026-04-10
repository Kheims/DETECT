"""Classical ML models for multi-label code smell detection using OO metrics."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


CLASSICAL_MODELS = {
    "random_forest": lambda cfg: MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", None),
            min_samples_split=cfg.get("min_samples_split", 2),
            class_weight="balanced",
            random_state=cfg.get("seed", 42),
            n_jobs=-1,
        )
    ),
    "svm": lambda cfg: MultiOutputClassifier(
        SVC(
            kernel=cfg.get("kernel", "rbf"),
            C=cfg.get("C", 1.0),
            class_weight="balanced",
            random_state=cfg.get("seed", 42),
            probability=True,
        )
    ),
    "xgboost": lambda cfg: MultiOutputClassifier(
        XGBClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", 5),
            learning_rate=cfg.get("lr", 0.1),
            random_state=cfg.get("seed", 42),
            n_jobs=-1,
            eval_metric="logloss",
        )
    ),
    "decision_tree": lambda cfg: MultiOutputClassifier(
        DecisionTreeClassifier(
            max_depth=cfg.get("max_depth", None),
            class_weight="balanced",
            random_state=cfg.get("seed", 42),
        )
    ),
    "knn": lambda cfg: MultiOutputClassifier(
        KNeighborsClassifier(
            n_neighbors=cfg.get("n_neighbors", 5),
            weights=cfg.get("weights", "distance"),
            n_jobs=-1,
        )
    ),
}


def build_classical_model(model_name, cfg):
    if model_name not in CLASSICAL_MODELS:
        raise ValueError(f"Unknown classical model: {model_name}. Available: {list(CLASSICAL_MODELS.keys())}")
    return CLASSICAL_MODELS[model_name](cfg)
