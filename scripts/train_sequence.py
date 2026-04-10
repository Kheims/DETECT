"""Train sequence-based DL models on tokenized code for multi-label code smell detection.

Supports multi-seed runs with post-training threshold tuning per label.

Usage:
    python scripts/train_sequence.py --config config/experiments/bilstm_tokens.yml
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score as sk_f1

from mlcq_graphs.config import load_config, parse_cli_overrides
from mlcq_graphs.models.sequence import build_sequence_model
from mlcq_graphs.training.losses import WeightedBCELoss, FocalLoss
from mlcq_graphs.evaluation.metrics import evaluate_full

LABEL_NAMES = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]


def tune_thresholds(logits, labels, steps=91):
    """Tune per-label thresholds on validation set to maximize per-label F1."""
    probs = torch.sigmoid(logits)
    num_labels = labels.shape[1]
    thresholds = torch.full((num_labels,), 0.5)
    for li in range(num_labels):
        best_f1, best_t = -1, 0.5
        for t in torch.linspace(0.05, 0.95, steps):
            preds = (probs[:, li] >= t).long()
            f1 = sk_f1(labels[:, li].numpy(), preds.numpy(), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t.item()
        thresholds[li] = best_t
    return thresholds


def run_single_seed(seed, dataset, meta, training_cfg, device):
    """Train and evaluate a single seed. Returns test results dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ratio = training_cfg.get("train_ratio", 0.8)
    val_ratio = training_cfg.get("val_ratio", 0.1)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    batch_size = training_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Build model
    model_name = training_cfg.get("model", "bilstm_attention")
    model_cfg = {
        "vocab_size": meta["vocab_size"],
        "embed_dim": training_cfg.get("embed_dim", 128),
        "hidden_dim": training_cfg.get("hidden_dim", 256),
        "num_layers": training_cfg.get("num_layers", 2),
        "dropout": training_cfg.get("dropout", 0.3),
        "num_labels": training_cfg.get("num_labels", 4),
        "num_filters": training_cfg.get("num_filters", 128),
        "filter_sizes": training_cfg.get("filter_sizes", [3, 4, 5]),
    }
    model = build_sequence_model(model_name, model_cfg).to(device)

    # Loss with class weights
    all_labels = torch.stack([dataset[i][1] for i in train_set.indices])
    label_counts = all_labels.sum(dim=0)
    n_samples = len(train_set)
    k = all_labels.shape[1]
    pos_weight = n_samples / (k * label_counts.clamp(min=1))

    loss_name = training_cfg.get("loss", "focal")
    if loss_name == "focal":
        criterion = FocalLoss(
            gamma=training_cfg.get("focal_gamma", 2.0),
            class_weight=pos_weight,
        ).to(device)
    else:
        criterion = WeightedBCELoss(pos_weight=pos_weight).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.get("lr", 0.001),
                                  weight_decay=training_cfg.get("weight_decay", 1e-5))

    # Training loop
    epochs = training_cfg.get("epochs", 50)
    patience = training_cfg.get("patience", 10)
    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation with fixed thresholds for stable early stopping
        model.eval()
        all_logits, all_labels_v = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                all_logits.append(model(X_batch).cpu())
                all_labels_v.append(y_batch)

        val_logits = torch.cat(all_logits)
        val_labels = torch.cat(all_labels_v)
        val_results = evaluate_full(val_logits, val_labels,
                                     torch.full((4,), 0.5), LABEL_NAMES)
        val_f1 = val_results["f1_macro"]

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Post-training threshold tuning on val set
    all_logits_v, all_labels_v = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            all_logits_v.append(model(X_batch).cpu())
            all_labels_v.append(y_batch)
    val_logits = torch.cat(all_logits_v)
    val_labels = torch.cat(all_labels_v)
    tuned_thresholds = tune_thresholds(val_logits, val_labels)

    # Test evaluation with tuned thresholds
    all_logits_t, all_labels_t = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            all_logits_t.append(model(X_batch).cpu())
            all_labels_t.append(y_batch)
    test_logits = torch.cat(all_logits_t)
    test_labels = torch.cat(all_labels_t)

    test_results = evaluate_full(test_logits, test_labels, tuned_thresholds, LABEL_NAMES)
    test_results["thresholds"] = tuned_thresholds.tolist()
    return test_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, remaining = parser.parse_known_args()

    overrides = parse_cli_overrides(remaining)
    cfg = load_config(Path(args.config), overrides)
    training_cfg = cfg.get("training", {})

    device = training_cfg.get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}")

    seeds = training_cfg.get("seeds", [42, 43, 44, 45, 46])
    model_name = training_cfg.get("model", "bilstm_attention")

    # Load dataset once
    token_dir = cfg.get("dataset", {}).get("token_dir", "artifacts/token_dataset")
    print(f"Loading token dataset from {token_dir}...")
    dataset = torch.load(os.path.join(token_dir, "token_dataset.pt"), weights_only=False)
    with open(os.path.join(token_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    print(f"Dataset: {len(dataset)} samples, vocab: {meta['vocab_size']}")

    # Run each seed
    all_results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        result = run_single_seed(seed, dataset, meta, training_cfg, device)
        all_results.append(result)
        print(f"  Thresholds: {[f'{t:.2f}' for t in result['thresholds']]}")
        print(f"  F1-macro: {result['f1_macro']:.4f}, MCC-macro: {result.get('mcc_macro', 0):.4f}")
        for name in LABEL_NAMES:
            pl = result["per_label"].get(name, {})
            print(f"  {name}: F1={pl.get('f1', 0):.4f}, MCC={pl.get('mcc', 0):.4f}")

    # Aggregate
    print(f"\n=== Aggregated ({len(seeds)} seeds) ===")
    for metric in ["f1_macro", "mcc_macro"]:
        values = [r.get(metric, 0) for r in all_results]
        print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    for name in LABEL_NAMES:
        f1s = [r["per_label"][name].get("f1", 0) for r in all_results]
        mccs = [r["per_label"][name].get("mcc", 0) for r in all_results]
        print(f"  {name}: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}, MCC={np.mean(mccs):.4f}±{np.std(mccs):.4f}")

    # Save
    output_dir = cfg.get("run", {}).get("artifacts_root", "artifacts")
    os.makedirs(output_dir, exist_ok=True)
    save_data = {
        "model": model_name,
        "seeds": seeds,
        "device": device,
        "per_seed": all_results,
        "aggregated": {
            "f1_macro": {"mean": float(np.mean([r["f1_macro"] for r in all_results])),
                         "std": float(np.std([r["f1_macro"] for r in all_results]))},
            "mcc_macro": {"mean": float(np.mean([r.get("mcc_macro", 0) for r in all_results])),
                          "std": float(np.std([r.get("mcc_macro", 0) for r in all_results]))},
        }
    }
    results_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
