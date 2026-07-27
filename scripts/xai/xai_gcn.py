"""Interpretability study of the reference GCN (AST graphs, node type and word2vec token features).

Each seed-42 kfold checkpoint explains the test samples of its fold: Integrated Gradients
per line, deletion, top tokens and nearest training graphs. Outputs go to artifacts/xai/gcn/.

Usage:
    python scripts/xai/xai_gcn.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.models.registry import get_model  # noqa: E402
from mlcq_graphs.xai import common  # noqa: E402
from scripts.train_gcn_baseline import _load_token_rows_from_dot as _train_token_rows  # noqa: E402

OUT = ROOT / "artifacts/xai/gcn"
DEVICE = torch.device("cpu")
STEPS = 32
# fewer IG steps for graphs above this batch size
MAX_BATCH_NODES = 160_000
FRACTIONS = (0.05, 0.10, 0.20)
TOKEN_ENCODER = common.CACHE / "token_encoder" / "0997b8485c35ecca728e3f629a766863449a1d1b586c38982703938125a4f745"


def find_runs(seed: int = 42, runs_dir: Path = ROOT / "artifacts/runs_ckpt") -> dict[int, Path]:
    """Reference GCN run of each kfold fold, trained on the full training part."""
    runs: dict[int, list[Path]] = defaultdict(list)
    for cfg_path in runs_dir.glob("*/config.json"):
        c = json.loads(cfg_path.read_text())
        sp = str(c.get("split_path", ""))
        if (c.get("architecture") == "gcn" and c.get("feature_mode") == "type_only" and c.get("use_token_features")
                and c.get("train_fraction") in (None, 1, 1.0) and not c.get("hybrid")
                and f"split_kfold_repo_url_k5_" in sp and sp.endswith(f"_seed_{seed}.json")
                and (cfg_path.parent / f"best_model_gcn_focal_s{seed}.pt").exists()):
            fold = int(sp.split("_f")[-1].split("_")[0])
            runs[fold].append(cfg_path.parent)
    dup = {f: [p.name[:10] for p in ps] for f, ps in runs.items() if len(ps) > 1}
    if dup:
        raise SystemExit(f"several full-training GCN runs for one fold: {dup}")
    return {f: ps[0] for f, ps in runs.items()}


def load_model(run: Path, seed: int = 42) -> torch.nn.Module:
    c = json.loads((run / "config.json").read_text())
    ck = torch.load(run / f"best_model_gcn_focal_s{seed}.pt", map_location="cpu", weights_only=False)
    tcfg = ck["config"]
    model = get_model("gcn", num_node_types=int(c["num_node_types"]), type_emb_dim=int(tcfg.get("type_emb_dim", 128)),
                      num_numeric_feats=0, num_token_feats=int(c["num_token_feats"]),
                      hidden_dim=int(tcfg.get("hidden_dim", 512)), num_labels=4, dropout=float(tcfg.get("dropout", 0.2)),
                      num_layers=int(c["num_layers"]), use_type_features=True, use_numeric_features=False,
                      use_token_features=True, pooling=c["pooling"], set2set_steps=3, subset_pooling=False,
                      num_graph_feats=0)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model.to(DEVICE)


def prepare(ds: list, vectors) -> dict[int, object]:
    """Graph of each json_index with token_x and undirected edges, as in training."""
    return {int(d.json_index): prepare_one(d, vectors) for d in ds}


def prepare_one(d, vectors):
    from torch_geometric.utils import to_undirected

    dim = int(vectors.vector_size)
    if True:
        j = int(d.json_index)
        nodes = common.ast_nodes(j)
        if len(nodes) != int(d.num_nodes):
            raise SystemExit(f"node count mismatch for {j}: dot {len(nodes)} dataset {int(d.num_nodes)}")
        # token text decoded exactly as in training, else the lookup misses
        tx = torch.zeros((int(d.num_nodes), dim))
        for index, kind, text in _train_token_rows(common._dot_paths()[j]):
            if kind == "SyntaxToken" and text in vectors:
                tx[index] = torch.tensor(vectors[text], dtype=torch.float32)
        d = d.clone()
        d.token_x = tx
        d.edge_index = to_undirected(d.edge_index, num_nodes=d.num_nodes)
        return d


def forward_from_x(model, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, n_graphs: int):
    """Logits and pooled embedding from the composed node features (eval mode)."""
    h = x
    for conv in model.convs:
        h = F.relu(conv(h, edge_index))
    g = model.readout(h, batch)
    logits = model.lin2(F.relu(model.lin1(g)))
    return logits, g


def node_features(model, d) -> torch.Tensor:
    with torch.no_grad():
        return model.compose_features(d)


def on_device(d):
    return d.to(DEVICE) if DEVICE.type != "cpu" else d


def integrated_gradients(model, d, smells_k: list[int]) -> tuple[dict[int, np.ndarray], dict[int, float], torch.Tensor]:
    """IG node attributions per label index, completeness error, base logits."""
    x = node_features(model, d)
    n = x.size(0)
    steps = STEPS if n * STEPS <= MAX_BATCH_NODES else max(8, MAX_BATCH_NODES // n)
    alphas = (torch.arange(1, steps + 1, dtype=torch.float32, device=x.device) - 0.5) / steps  # midpoint rule
    xs = (alphas.view(-1, 1, 1) * x.unsqueeze(0)).reshape(steps * n, -1).requires_grad_(True)
    ei = torch.cat([d.edge_index + s * n for s in range(steps)], dim=1)
    batch = torch.arange(steps, device=x.device).repeat_interleave(n)
    logits, _ = forward_from_x(model, xs, ei, batch, steps)
    with torch.no_grad():
        zb = torch.zeros(n, dtype=torch.long, device=x.device)
        base_logits, emb = forward_from_x(model, x, d.edge_index, zb, 1)
        zero_logits, _ = forward_from_x(model, torch.zeros_like(x), d.edge_index, zb, 1)
    attrs, err = {}, {}
    for i, k in enumerate(smells_k):
        grad = torch.autograd.grad(logits[:, k].sum(), xs, retain_graph=i < len(smells_k) - 1)[0]
        avg = grad.view(steps, n, -1).mean(0)
        a = (x * avg).sum(-1).cpu().numpy()
        delta = float(base_logits[0, k] - zero_logits[0, k])
        attrs[k] = a
        err[k] = abs(a.sum() - delta) / (abs(delta) + 1e-6)
    return attrs, err, base_logits[0].cpu(), emb[0].cpu()


def deletion(model, d, attr: np.ndarray, k: int, rng: np.random.Generator) -> dict[str, list[float]]:
    x = node_features(model, d)
    n = x.size(0)
    batch = torch.zeros(n, dtype=torch.long, device=x.device)
    with torch.no_grad():
        base = float(forward_from_x(model, x, d.edge_index, batch, 1)[0][0, k])
    out = {}
    orders = {"ig": np.argsort(-attr), "abs_ig": np.argsort(-np.abs(attr))}
    for r in range(2):
        orders[f"random{r}"] = rng.permutation(n)
    for name, order in orders.items():
        drops = []
        for frac in FRACTIONS:
            m = max(1, int(math.ceil(frac * n)))
            xm = x.clone()
            xm[torch.as_tensor(order[:m], device=x.device)] = 0
            with torch.no_grad():
                drops.append(base - float(forward_from_x(model, xm, d.edge_index, batch, 1)[0][0, k]))
        out[name] = drops
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="explain only the first N test samples per fold (smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs_dir", default="artifacts/runs_ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fold", type=int, action="append", help="folds to explain (default: all); outputs in fold<f>/")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="merge the fold<f>/ outputs")
    args = ap.parse_args()
    if args.merge:
        return merge()
    global DEVICE
    DEVICE = torch.device(args.device)
    torch.set_num_threads(args.threads or max(1, torch.get_num_threads()))
    from gensim.models import KeyedVectors

    OUT.mkdir(parents=True, exist_ok=True)
    runs = find_runs(args.seed, ROOT / args.runs_dir)
    if sorted(runs) != [0, 1, 2, 3, 4]:
        raise SystemExit(f"expected 5 kfold runs, found {sorted(runs)}")
    print(f"[gcn] runs {[(f, r.name[:10]) for f, r in sorted(runs.items())]}")
    ds = torch.load(common.CACHE / "dataset" / common.DATASET_FP / "dataset.pt", weights_only=False)
    vectors = KeyedVectors.load(str(TOKEN_ENCODER / "word2vec.kv"), mmap="r")
    raw = {int(d.json_index): d for d in ds}

    class Lazy(dict):
        def __missing__(self, j):
            return prepare_one(raw[j], vectors)

    graphs = Lazy()
    folds = common.folds(args.seed, "kfold")
    rng = np.random.default_rng(0)

    rows, dels, comp_err, emb_rows = [], [], [], []
    tok_stats = defaultdict(lambda: defaultdict(list))
    type_stats = defaultdict(lambda: defaultdict(list))
    check = []
    out_dir = OUT if args.fold is None else OUT / ("fold" + "_".join(str(f) for f in sorted(args.fold)))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs.json").write_text(json.dumps(
        {str(f): r.name for f, r in sorted(runs.items()) if args.fold is None or f in args.fold}, indent=1))
    for f, run in sorted(runs.items()):
        if args.fold is not None and f not in args.fold:
            continue
        model = load_model(run, args.seed)
        stored = {}
        pred_path = ROOT / "artifacts/runs" / run.name / "predictions.jsonl"
        if pred_path.exists():
            for line in pred_path.open():
                r = json.loads(line)
                if r["part"] == "test" and r.get("seed") == args.seed:
                    stored[r["json_index"]] = r["prob"]
        test = folds[f]["test"][: args.limit or None]
        train = folds[f]["train"]
        # embeddings of the training graphs for the k-NN explanations
        train_emb, train_y = [], []
        with torch.no_grad():
            for j in train:
                d = on_device(graphs[j])
                x = node_features(model, d)
                _, g = forward_from_x(model, x, d.edge_index, torch.zeros(x.size(0), dtype=torch.long, device=x.device), 1)
                train_emb.append(g[0].cpu().numpy())
                train_y.append(d.y.cpu().numpy())
        train_emb = np.array(train_emb)
        train_y = np.array(train_y)
        tn = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-9)
        for t, j in enumerate(test):
            d = on_device(graphs[j])
            lv = common.levels()[j]
            ks = [k for k, s in enumerate(common.LABELS) if common.LEVEL_OF[s] == lv]
            attrs, err, logits, emb = integrated_gradients(model, d, ks)
            probs = torch.sigmoid(logits).numpy()
            if j not in stored:
                raise SystemExit(f"json_index {j}: no stored test probability to check against")
            check.append(float(np.abs(np.array(stored[j]) - probs).max()))
            if check[-1] > 1e-4:
                raise SystemExit(f"json_index {j}: recomputed probabilities differ from the stored ones by {check[-1]:.6f}")
            nodes = common.ast_nodes(j)
            lines = common.snippet_lines(j)
            e = emb.numpy()
            sims = tn @ (e / (np.linalg.norm(e) + 1e-9))
            nn_idx = np.argsort(-sims)[:5]
            emb_rows.append({"json_index": j, "fold": f, "emb": e.astype(np.float32),
                             "neighbours": [int(train[i]) for i in nn_idx], "sims": [float(s) for s in sims[nn_idx]],
                             "nn_y": train_y[nn_idx].tolist(), "prob": probs.tolist()})
            for k in ks:
                smell = common.LABELS[k]
                a = attrs[k]
                comp_err.append(err[k])
                ls = np.zeros(len(lines))
                syn = tok = 0.0
                for nd, v in zip(nodes, a):
                    ln = nd.start_line
                    if 0 <= ln < len(lines):
                        ls[ln] += v
                    if nd.kind == "SyntaxToken":
                        tok += abs(v)
                    else:
                        syn += abs(v)
                y = common.label(j, smell)
                rows.append({"model": "gcn", "method": "integrated_gradients", "json_index": j, "smell": smell,
                             "label": y, "score": float(probs[k]), "line_scores": ls.round(5).tolist(),
                             "fold": f, "logit": float(logits[k]), "token_share": float(tok / (tok + syn + 1e-12)),
                             "n_nodes": int(d.num_nodes)})
                if y == 1:
                    for nd, v in zip(nodes, a):
                        if nd.kind == "SyntaxToken" and nd.text is not None:
                            tok_stats[smell][nd.text].append(float(v))
                        type_stats[smell][nd.type].append(float(v))
                dl = deletion(model, d, a, k, rng)
                dels.append({"json_index": j, "smell": smell, "label": y, **{f"{m}_{int(fr * 100)}": float(dl[m][i])
                                                                           for m in dl for i, fr in enumerate(FRACTIONS)}})
            if (t + 1) % 200 == 0:
                print(f"[gcn] fold {f}: {t + 1}/{len(test)} explained, median completeness error "
                      f"{np.median(comp_err):.4f}", flush=True)
        print(f"[gcn] fold {f} done", flush=True)

    OUTD = out_dir
    common.write_rows(OUTD / "line_attributions.jsonl", rows)
    with (OUTD / "deletion.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dels[0]))
        w.writeheader()
        w.writerows(dels)
    np.savez_compressed(OUTD / "embeddings.npz", json_index=np.array([r["json_index"] for r in emb_rows]),
                        fold=np.array([r["fold"] for r in emb_rows]), emb=np.stack([r["emb"] for r in emb_rows]),
                        prob=np.array([r["prob"] for r in emb_rows]))
    with (OUTD / "neighbours.jsonl").open("w") as fh:
        for r in emb_rows:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "emb"}) + "\n")
    sums = {name: {smell: {tk: [float(np.sum(v)), len(v), int(np.sum(np.array(v) > 0))] for tk, v in st.items()}
                   for smell, st in stats.items()} for name, stats in (("tokens", tok_stats), ("node_types", type_stats))}
    (OUTD / "token_sums.json").write_text(json.dumps(sums))
    (OUTD / "completeness.json").write_text(json.dumps({"errors": [float(e) for e in comp_err], "check": check}))
    if args.fold is None:
        merge()


def merge() -> None:
    """Combine the fold<f>/ outputs (or the single output) into OUT."""
    dirs = sorted(p for p in OUT.glob("fold*") if p.is_dir()) or [OUT]
    rows, dels, nbrs, comp, check = [], [], [], [], []
    embs = {"json_index": [], "fold": [], "emb": [], "prob": []}
    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0])))
    for d in dirs:
        rows += common.read_rows(d / "line_attributions.jsonl")
        with (d / "deletion.csv").open() as fh:
            dels += list(csv.DictReader(fh))
        nbrs += [json.loads(line) for line in (d / "neighbours.jsonl").open()]
        z = np.load(d / "embeddings.npz")
        for k in embs:
            embs[k].append(z[k])
        c = json.loads((d / "completeness.json").read_text())
        comp += c["errors"]
        check += c["check"]
        for name, st in json.loads((d / "token_sums.json").read_text()).items():
            for smell, toks in st.items():
                for tk, (s, n, p) in toks.items():
                    acc = sums[name][smell][tk]
                    acc[0] += s
                    acc[1] += n
                    acc[2] += p
    if dirs != [OUT]:
        common.write_rows(OUT / "line_attributions.jsonl", rows)
        with (OUT / "deletion.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dels[0]))
            w.writeheader()
            w.writerows(dels)
        with (OUT / "neighbours.jsonl").open("w") as fh:
            for r in nbrs:
                fh.write(json.dumps(r) + "\n")
        np.savez_compressed(OUT / "embeddings.npz", **{k: np.concatenate(v) for k, v in embs.items()})
    for name, st in sums.items():
        with (OUT / f"top_{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["smell", name[:-1], "count", "mean_ig", "share_positive"])
            for smell, toks in st.items():
                items = [(tk, v) for tk, v in toks.items() if v[1] >= 20]
                items.sort(key=lambda x: -x[1][0] / x[1][1])
                for tk, (s, n, p) in items[:40] + items[-15:]:
                    w.writerow([smell, tk, n, round(s / n, 5), round(p / n, 3)])
    summary = {"samples": len(nbrs), "rows": len(rows), "folds_dirs": [d.name for d in dirs],
               "completeness_rel_error_median": float(np.median(comp)),
               "completeness_rel_error_p90": float(np.percentile(comp, 90)),
               "stored_prob_max_abs_diff_median": float(np.median(check)) if check else None,
               "stored_prob_max_abs_diff_max": float(np.max(check)) if check else None, "steps_max": STEPS,
               "runs": {k: v for d in dirs if (d / "runs.json").exists()
                        for k, v in json.loads((d / "runs.json").read_text()).items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[gcn] {summary}")


if __name__ == "__main__":
    main()
