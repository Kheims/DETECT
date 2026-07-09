"""Detection results per smell: MCC, F1, precision and recall of every model family.

Reads CV records.json files and the rule-based tool results.
Writes results_<table>.csv, results.md and results_<table>.png in out_dir.

Usage:
    python scripts/cv/results_tables.py --config config/experiments/results_tables.yml
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402

NAMES = {"is_feature_envy": "Feature Envy", "is_long_method": "Long Method", "is_blob": "Blob",
         "is_data_class": "Data Class", "is_complex_method": "Complex Method",
         "is_long_parameter_list": "Long Parameter List"}
METRICS = ("mcc", "f1", "precision", "recall")
SEED_RE = re.compile(r"_s(\d+)_f(\d+)")


def records(pattern: str) -> list[dict[str, Any]]:
    out = []
    for f in sorted(glob.glob(str(PROJECT_ROOT / pattern))):
        data = json.loads(Path(f).read_text())
        rows = data if isinstance(data, list) else [data]
        for r in rows:
            if r.get("seed") is None:
                m = SEED_RE.search(f)
                r = {**r, "seed": int(m.group(1)) if m else None}
            out.append(r)
    if not out:
        raise SystemExit(f"no records for {pattern}")
    return out


def strategy(r: dict[str, Any]) -> str:
    combo = r.get("combo") or {}
    return combo.get("split_strategy") or combo.get("training.split_strategy") or r.get("split_strategy") or "kfold"


def summarize(rows: list[dict[str, Any]], label: str, strat: str) -> dict[str, Any] | None:
    by_seed: dict[Any, list[dict[str, float]]] = defaultdict(list)
    for r in rows:
        if strategy(r) != strat:
            continue
        m = (r.get("per_label_tuned") or {}).get(label)
        if m is None or m.get("mcc") is None:
            continue
        by_seed[r["seed"]].append({k: float(m[k]) for k in METRICS})
    if not by_seed:
        return None
    seed_means = {k: [np.mean([f[k] for f in folds]) for folds in by_seed.values()] for k in METRICS}
    out = {k: float(np.mean(v)) for k, v in seed_means.items()}
    out.update({f"{k}_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in seed_means.items()})
    out["seeds"] = len(by_seed)
    out["folds"] = sum(len(v) for v in by_seed.values())
    return out


def rule_rows(source: Path, keys: dict[str, str], labels: list[str]) -> list[dict[str, Any]]:
    data = json.loads(source.read_text())
    out = []
    for tool, key in keys.items():
        res = data[key].get("results") or {}
        for label in labels:
            dets = (res.get(label) or {}).get("detectors") or {}
            for setting in ("default", "tuned"):
                best = None
                for rule, d in dets.items():
                    m = ((d.get(setting) or {}).get("kfold_test")) if d.get(setting) else None
                    if m and (best is None or m["mcc"] > best[1]["mcc"]):
                        best = (rule, m)
                if best is None:
                    continue
                rule, m = best
                p, r = m["precision"], m["recall"]
                out.append({"model": f"{tool}: {rule} ({'shipped settings' if setting == 'default' else 'threshold tuned'})",
                            "kind": "rule", "label": label, "strategy": "kfold", "mcc": m["mcc"],
                            "mcc_sd": m.get("mcc_sd", 0.0), "f1": 2 * p * r / (p + r) if p + r else 0.0,
                            "precision": p, "recall": r, "seeds": m.get("seeds"), "folds": m.get("folds")})
    return out


def figure(rows: list[dict[str, Any]], labels: list[str], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"metrics": "#2b6cb0", "graph": "#805ad5", "embedding": "#38a169", "fine-tuned": "#d69e2e",
              "LLM": "#dd6b20", "rule": "#718096"}
    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels) + 3.5, 0.42 * len(models) + 1.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels):
        for i, model in enumerate(models):
            r = next((x for x in rows if x["model"] == model and x["label"] == label and x["strategy"] == "kfold"),
                     None)
            if r is None:
                continue
            c = colors.get(r["kind"], "black")
            ax.barh(i, r["mcc"], color=c, height=0.62, xerr=r["mcc_sd"], error_kw={"lw": 0.8})
            ax.text(max(r["mcc"], 0) + 0.02, i, f"MCC {r['mcc']:.2f}  F1 {r['f1']:.2f}", va="center", fontsize=7)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlim(min(-0.05, min((x["mcc"] for x in rows if x["label"] == label), default=0) - 0.05), 1.15)
        ax.set_title(NAMES[label], fontsize=10)
        ax.set_xlabel("MCC (bar) and F1, threshold tuned on validation")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(models)), models, fontsize=8)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, ["metrics + classical model", "graph neural network", "embedding + linear probe",
                         "fine-tuned encoder", "LLM, zero-shot", "rule-based tool"],
               loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(f"{title}: mean over 5 seeds x 5 folds (random 5-fold split), error bar = SD over seeds",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args, extra = ap.parse_known_args()
    cfg = load_config(Path(args.config), parse_cli_overrides(extra))["results_tables"]
    out = PROJECT_ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    md = []
    for tname, t in cfg["tables"].items():
        labels = list(t["labels"])
        rows = []
        for m in t["models"]:
            recs = records(m["records"])
            for strat in cfg["strategies"]:
                for label in labels:
                    s = summarize(recs, label, strat)
                    if s is not None:
                        rows.append({"model": m["name"], "kind": m["kind"], "label": label, "strategy": strat, **s})
        if t.get("rule_tools"):
            rt = t["rule_tools"]
            rows += rule_rows(PROJECT_ROOT / rt["source"], dict(rt["keys"]), labels)
        fields = ["model", "kind", "label", "strategy", "mcc", "mcc_sd", "f1", "f1_sd", "precision", "precision_sd",
                  "recall", "recall_sd", "seeds", "folds"]
        with (out / f"results_{tname}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        figure(rows, labels, t["title"], out / f"results_{tname}.png")
        md.append(f"## {t['title']}\n")
        for label in labels:
            md.append(f"### {NAMES[label]}\n")
            md.append("| Model | MCC | F1 | Precision | Recall | MCC, unseen projects |")
            md.append("|---|---|---|---|---|---|")
            kf = sorted((r for r in rows if r["label"] == label and r["strategy"] == "kfold"), key=lambda r: -r["mcc"])
            for r in kf:
                g = next((x for x in rows if x["model"] == r["model"] and x["label"] == label
                          and x["strategy"] == "group_kfold"), None)
                md.append(f"| {r['model']} | {r['mcc']:.3f} ± {r['mcc_sd']:.3f} | {r['f1']:.3f} | "
                          f"{r['precision']:.3f} | {r['recall']:.3f} | {g['mcc']:.3f} |" if g else
                          f"| {r['model']} | {r['mcc']:.3f} ± {r['mcc_sd']:.3f} | {r['f1']:.3f} | "
                          f"{r['precision']:.3f} | {r['recall']:.3f} | |")
            md.append("")
        print(f"[tables] {tname}: {len(rows)} rows")
    (out / "results.md").write_text("\n".join(md))
    print(f"[tables] wrote {out}")


if __name__ == "__main__":
    main()
