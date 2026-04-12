"""Fine-tune CodeBERT (microsoft/codebert-base) for multi-label code smell detection.

Tokenizes raw code snippets with RobertaTokenizer, fine-tunes the pretrained
encoder with a classification head, applies post-training threshold tuning.

Usage:
    python scripts/train_codebert.py --config config/experiments/sequence_codebert_pipeline.yml
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

from mlcq_graphs.models.sequence import CodeBERTClassifier
from mlcq_graphs.training.losses import FocalLoss, WeightedBCELoss
from mlcq_graphs.evaluation.metrics import evaluate_full

LABEL_NAMES = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]
Y_ORDER = ["feature envy", "long method", "blob", "data class"]


def build_codebert_dataset(normalized_json_path, max_length=256):
    """Tokenize code snippets with RobertaTokenizer and build a TensorDataset."""
    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    with open(normalized_json_path, "r") as f:
        samples = json.load(f)

    codes = [s["code_snippet"] for s in samples]
    labels = [s["y"] for s in samples]

    print(f"Tokenizing {len(codes)} samples with RobertaTokenizer (max_length={max_length})...")
    encoded = tokenizer(
        codes,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    y = torch.tensor(labels, dtype=torch.float32)

    print(f"Dataset built: {input_ids.shape[0]} samples, {input_ids.shape[1]} tokens")
    return TensorDataset(input_ids, attention_mask, y)


def tune_thresholds(logits, labels, steps=91):
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


def run_single_seed(seed, dataset, training_cfg, device):
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
        generator=torch.Generator().manual_seed(seed),
    )

    batch_size = training_cfg.get("batch_size", 16)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = CodeBERTClassifier(
        num_labels=training_cfg.get("num_labels", 4),
        dropout=training_cfg.get("dropout", 0.1),
    ).to(device)

    # Class weights
    all_labels = torch.stack([dataset[i][2] for i in train_set.indices])
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("lr", 2e-5),
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )

    epochs = training_cfg.get("epochs", 10)
    patience = training_cfg.get("patience", 3)
    best_val_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids, mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_logits, all_labels_v = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, labels = batch[0].to(device), batch[1].to(device), batch[2]
                all_logits.append(model(input_ids=ids, attention_mask=mask).cpu())
                all_labels_v.append(labels)

        val_logits = torch.cat(all_logits)
        val_labels = torch.cat(all_labels_v)
        val_results = evaluate_full(val_logits, val_labels,
                                     torch.full((4,), 0.5), LABEL_NAMES)
        val_f1 = val_results["f1_macro"]

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

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Threshold tuning on val
    all_logits_v, all_labels_v = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids, mask = batch[0].to(device), batch[1].to(device)
            all_logits_v.append(model(input_ids=ids, attention_mask=mask).cpu())
            all_labels_v.append(batch[2])
    tuned_thresholds = tune_thresholds(torch.cat(all_logits_v), torch.cat(all_labels_v))

    # Test
    all_logits_t, all_labels_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids, mask = batch[0].to(device), batch[1].to(device)
            all_logits_t.append(model(input_ids=ids, attention_mask=mask).cpu())
            all_labels_t.append(batch[2])

    test_results = evaluate_full(
        torch.cat(all_logits_t), torch.cat(all_labels_t),
        tuned_thresholds, LABEL_NAMES,
    )
    test_results["thresholds"] = tuned_thresholds.tolist()
    return test_results


def run_codebert(
    training_cfg: dict,
    normalized_json_path: str,
    output_dir: str,
    run_name: str | None = None,
) -> dict:
    """Run CodeBERT fine-tuning. Entry point for pipeline orchestration."""
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
    max_length = training_cfg.get("max_length", 256)

    dataset = build_codebert_dataset(normalized_json_path, max_length=max_length)

    all_results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        result = run_single_seed(seed, dataset, training_cfg, device)
        all_results.append(result)
        print(f"  F1-macro: {result['f1_macro']:.4f}, MCC: {result.get('mcc_macro', 0):.4f}")

    # Aggregate
    aggregated = {
        "f1_macro": {
            "mean": float(np.mean([r["f1_macro"] for r in all_results])),
            "std": float(np.std([r["f1_macro"] for r in all_results])),
        },
        "mcc_macro": {
            "mean": float(np.mean([r.get("mcc_macro", 0) for r in all_results])),
            "std": float(np.std([r.get("mcc_macro", 0) for r in all_results])),
        },
        "per_label": {},
    }
    for name in LABEL_NAMES:
        aggregated["per_label"][name] = {
            "f1": {
                "mean": float(np.mean([r["per_label"][name].get("f1", 0) for r in all_results])),
                "std": float(np.std([r["per_label"][name].get("f1", 0) for r in all_results])),
            },
            "mcc": {
                "mean": float(np.mean([r["per_label"][name].get("mcc", 0) for r in all_results])),
                "std": float(np.std([r["per_label"][name].get("mcc", 0) for r in all_results])),
            },
        }

    os.makedirs(output_dir, exist_ok=True)
    save_data = {
        "model": "codebert",
        "seeds": seeds,
        "device": device,
        "per_seed": all_results,
        "aggregated": aggregated,
    }
    filename = f"{run_name}.json" if run_name else "codebert_results.json"
    results_path = os.path.join(output_dir, filename)
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    save_data["results_path"] = results_path
    return save_data


if __name__ == "__main__":
    import argparse
    from mlcq_graphs.config import load_config, parse_cli_overrides

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, remaining = parser.parse_known_args()

    overrides = parse_cli_overrides(remaining)
    cfg = load_config(Path(args.config), overrides)
    training_cfg = cfg.get("training", {})
    norm_json = cfg.get("normalization", {}).get("output_json",
        "artifacts/cache/normalization/*/normalized.default.json")
    output_dir = cfg.get("run", {}).get("artifacts_root", "artifacts")

    run_codebert(
        training_cfg=training_cfg,
        normalized_json_path=str(norm_json),
        output_dir=output_dir,
    )
