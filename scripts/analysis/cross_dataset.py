"""Construct overlap between the method-level smells of two datasets.

Each cell is the ROC-AUC of a detector trained for a source smell, scored
against the labels of a target smell.

Usage:
    python scripts/analysis/cross_dataset.py --config config/experiments/cross_dataset.yml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def load_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    entries = json.loads(resolve(spec["input_json"]).read_text())
    if spec.get("level_csv"):
        level = {int(r[spec["level_key"]]): r[spec["level_column"]] for r in csv.DictReader(resolve(spec["level_csv"]).open())}
    else:
        level = {i: str(e.get("level", "method")) for i, e in enumerate(entries)}
    idx = sorted(i for i, lv in level.items() if lv == "method")
    labels = {name: np.array([int(bool(entries[i]["y"][col])) for i in idx]) for name, col in spec["labels"].items()}
    return {"idx": idx, "labels": labels}


def load_features(spec: dict[str, Any], idx: list[int]) -> np.ndarray:
    if spec["source"] == "csv":
        rows = {int(r[spec["key"]]): r for r in csv.DictReader(resolve(spec["path"]).open())}
        x = np.array([[float(rows[i][c]) for c in spec["columns"]] for i in idx])
    elif spec["source"] == "npy":
        arr = np.load(resolve(spec["path"]))
        order = json.loads(resolve(spec["index"]).read_text())
        pos = {int(j): k for k, j in enumerate(order)}
        x = arr[[pos[i] for i in idx]].astype(float)
    else:
        raise ValueError(spec["source"])
    return signed_log1p(x) if spec.get("transform") == "signed_log1p" else x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))["cross_dataset"]
    data = {name: load_dataset(spec) for name, spec in cfg["datasets"].items()}
    results: dict[str, Any] = {}
    for det_name, det in cfg["detectors"].items():
        feats = {ds: load_features(det["features"][ds], data[ds]["idx"]) for ds in det["features"]}
        cells = {}
        for src in feats:
            for src_label, y_src in data[src]["labels"].items():
                scaler = StandardScaler().fit(feats[src])
                model = LogisticRegression(C=float(det.get("C", 1.0)), class_weight="balanced", max_iter=2000)
                model.fit(scaler.transform(feats[src]), y_src)
                for tgt in feats:
                    if tgt == src:
                        continue
                    s = model.predict_proba(scaler.transform(feats[tgt]))[:, 1]
                    for tgt_label, y_tgt in data[tgt]["labels"].items():
                        cells[f"{src}:{src_label} -> {tgt}:{tgt_label}"] = roc_auc_score(y_tgt, s)
                    # source detector on its own dataset, other smells (fit on all data, in-sample)
                for other_label, y_other in data[src]["labels"].items():
                    if other_label != src_label:
                        s = model.predict_proba(scaler.transform(feats[src]))[:, 1]
                        cells[f"{src}:{src_label} -> {src}:{other_label} (in-sample)"] = roc_auc_score(y_other, s)
        results[det_name] = cells
    for score_name, spec in (cfg.get("score_files") or {}).items():
        rows = [json.loads(line) for line in resolve(spec["path"]).open()]
        by = {(int(r["json_index"]), r["smell"]): float(r["score"]) for r in rows if r.get("score") is not None}
        ds = spec["dataset"]
        cells = {}
        for question in spec["questions"]:
            idx = [i for i in data[ds]["idx"] if (i, question) in by]
            if len(idx) < len(data[ds]["idx"]) * 0.95:
                continue
            s = np.array([by[(i, question)] for i in idx])
            pos = {i: k for k, i in enumerate(data[ds]["idx"])}
            for tgt_label, y in data[ds]["labels"].items():
                cells[f"question {question} -> {ds}:{tgt_label}"] = roc_auc_score(y[[pos[i] for i in idx]], s)
        results[score_name] = cells
    out = resolve(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    for name, cells in results.items():
        print(name)
        for k, v in cells.items():
            print(f"  {k:<70} ROC-AUC {v:.3f}")
    print(f"[cross_dataset] wrote {out}")


if __name__ == "__main__":
    main()
