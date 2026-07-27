"""Model randomization test of the GCN Integrated Gradients.

Compares IG line scores of the trained GCN with a reinitialized head and fully
reinitialized layers. Writes artifacts/xai/gcn/sanity.csv and sanity_rows.csv.

Usage:
    python scripts/xai/xai_gcn_sanity.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.xai import xai_gcn as xg  # noqa: E402
from mlcq_graphs.xai import common  # noqa: E402

N_PER_SMELL = 60


def line_scores(model, d, j: int, k: int) -> np.ndarray:
    attrs, _, _, _ = xg.integrated_gradients(model, d, [k])
    nodes = common.ast_nodes(j)
    lines = common.snippet_lines(j)
    ls = np.zeros(len(lines))
    for nd, v in zip(nodes, attrs[k]):
        if 0 <= nd.start_line < len(lines):
            ls[nd.start_line] += v
    return ls


def randomized(model, scope: str, seed: int = 0):
    import copy

    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    mods = [m.lin2, m.lin1] if scope == "head" else [m.lin2, m.lin1, *reversed(list(m.convs)), m.type_emb]
    for mod in mods:
        mod.reset_parameters()
    return m.eval()


def main() -> None:
    import argparse

    from gensim.models import KeyedVectors

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs_dir", default="artifacts/runs_ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds of the random initialization")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap draws over snippets for the interval")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    xg.DEVICE = torch.device(args.device)
    runs = xg.find_runs(42, ROOT / args.runs_dir)
    model = xg.load_model(runs[0])
    ds = torch.load(common.CACHE / "dataset" / common.DATASET_FP / "dataset.pt", weights_only=False)
    test = set(common.folds(42, "kfold")[0]["test"])
    vectors = KeyedVectors.load(str(xg.TOKEN_ENCODER / "word2vec.kv"), mmap="r")
    graphs = xg.prepare([d for d in ds if int(d.json_index) in test], vectors)
    rng = np.random.default_rng(0)
    scopes = {"head randomized": "head", "all layers randomized": "all"}
    variants = {(v, s): randomized(model, sc, seed=s) for v, sc in scopes.items() for s in seeds}
    rows, per_snippet = [], []
    for k, smell in enumerate(common.LABELS):
        elig = [j for j in sorted(graphs) if common.eligible(j, smell) and len(common.snippet_lines(j)) >= 4]
        pos = [j for j in elig if common.label(j, smell) == 1]
        neg = [j for j in elig if common.label(j, smell) == 0]
        pick = list(rng.choice(pos, size=min(len(pos), N_PER_SMELL // 2), replace=False)) + \
            list(rng.choice(neg, size=N_PER_SMELL - min(len(pos), N_PER_SMELL // 2), replace=False))
        res = {(v, s): {} for (v, s) in variants}
        for j in pick:
            g = xg.on_device(graphs[int(j)])
            base = line_scores(model, g, int(j), k)
            for key, m in variants.items():
                other = line_scores(m, g, int(j), k)
                if base.std() > 0 and other.std() > 0 and np.abs(base).std() > 0 and np.abs(other).std() > 0:
                    # magnitudes, since a reset head can flip every sign
                    res[key][int(j)] = float(stats.spearmanr(np.abs(base), np.abs(other)).statistic)
                    per_snippet.append({"smell": smell, "randomization": key[0], "seed": key[1], "json_index": int(j),
                                        "spearman": float(stats.spearmanr(base, other).statistic),
                                        "spearman_abs": res[key][int(j)]})
        for v in scopes:
            # mean over the initialization seeds
            snippets = sorted(set.intersection(*(set(res[(v, s)]) for s in seeds)))
            vals = np.array([np.mean([res[(v, s)][j] for s in seeds]) for j in snippets])
            boot = np.random.default_rng(0).choice(vals, size=(args.boot, len(vals))).mean(1)
            seed_means = [float(np.mean([res[(v, s)][j] for j in snippets])) for s in seeds]
            rows.append({"smell": smell, "randomization": v, "samples": len(vals), "seeds": len(seeds),
                         "measure": "spearman of |line score|",
                         "mean_spearman": float(vals.mean()), "median_spearman": float(np.median(vals)),
                         "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                         "seed_mean_min": min(seed_means), "seed_mean_max": max(seed_means),
                         "run": runs[0].name})
            print(f"[sanity] {common.NAMES[smell]:13s} {v:22s} Spearman with trained {vals.mean():.2f} "
                  f"[{rows[-1]['ci_low']:.2f}, {rows[-1]['ci_high']:.2f}] ({len(vals)} snippets, {len(seeds)} seeds)",
                  flush=True)
    with (ROOT / "artifacts/xai/gcn/sanity_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_snippet[0]))
        w.writeheader()
        w.writerows(per_snippet)
    out = ROOT / "artifacts/xai/gcn/sanity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
