"""Interpretability study of a BiLSTM with attention on MLCQ, snippet by snippet.

One model per seed-42 kfold fold. On the test samples: attention, Integrated Gradients,
deletion and attention/IG agreement. Outputs go to artifacts/xai/bilstm/.

Usage:
    python scripts/xai/xai_bilstm.py prepare            # write inputs.json.gz
    python scripts/xai/xai_bilstm.py run --fold 0       # train and attribute one fold
    python scripts/xai/xai_bilstm.py merge              # merge the fold files
    python scripts/xai/xai_bilstm.py report             # REPORT.md (after scripts/cv/eval_eligible.py)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.xai import common  # noqa: E402

MODEL = "bilstm_attention"
OUT = PROJECT_ROOT / "artifacts" / "xai" / "bilstm"
LABELS = common.LABELS
SMELLS_OF = {"method": ["is_feature_envy", "is_long_method"], "class": ["is_blob", "is_data_class"]}
PAD, UNK = 0, 1
MAX_LEN = 512
MIN_FREQ = 2
MAX_VOCAB = 10000
SEED = 42
IG_STEPS = 32
FRACS = (0.05, 0.10, 0.20)
N_RANDOM = 3
TOP_FRAC = 0.10
MIN_TOKEN_COUNT = 20
N_TOP_TOKENS = 30
HP = {"embed_dim": 128, "hidden_dim": 256, "num_layers": 2, "dropout": 0.3, "num_labels": 4,
      "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 32, "max_epochs": 50, "patience": 10,
      "focal_gamma": 2.0, "threshold_steps": 101}


# ---------------------------------------------------------------- prepare

def prepare(out: Path) -> None:
    """Token texts and lines, labels, levels and folds of every sample in one file."""
    folds = common.folds(seed=SEED, strategy="kfold")
    ids = sorted({i for f in folds.values() for part in f.values() for i in part})
    data = {}
    for i in ids:
        toks = common.tokens(i)
        data[str(i)] = {"tokens": [t.text or "" for t in toks], "lines": [t.start_line for t in toks],
                        "y": [int(v) for v in common.samples()[i]["y"]], "level": common.levels()[i],
                        "n_lines": len(common.snippet_lines(i))}
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "inputs.json.gz", "wt") as fh:
        json.dump({"samples": data, "folds": {str(k): v for k, v in folds.items()}}, fh)
    lengths = np.array([len(d["tokens"]) for d in data.values()])
    print(f"[prepare] {len(data)} samples, tokens median {np.median(lengths):.0f}, "
          f"{int((lengths > MAX_LEN).sum())} longer than {MAX_LEN}; wrote {out / 'inputs.json.gz'}")


def load_inputs(out: Path) -> dict[str, Any]:
    with gzip.open(out / "inputs.json.gz", "rt") as fh:
        raw = json.load(fh)
    return {"samples": {int(k): v for k, v in raw["samples"].items()},
            "folds": {int(k): v for k, v in raw["folds"].items()}}


# ---------------------------------------------------------------- model helpers

def build_vocab(token_lists: list[list[str]]) -> dict[str, int]:
    counts = Counter(t for toks in token_lists for t in toks)
    kept = [t for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= MIN_FREQ]
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    for t in kept[:MAX_VOCAB - 2]:
        vocab[t] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int]) -> tuple[list[int], int]:
    """Ids padded to MAX_LEN, and the number of real (kept) tokens."""
    ids = [vocab.get(t, UNK) for t in tokens[:MAX_LEN]]
    n = len(ids)
    return ids + [PAD] * (MAX_LEN - n), n


def pick_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def new_model(vocab_size: int):
    from mlcq_graphs.models.sequence import BiLSTMAttentionClassifier

    return BiLSTMAttentionClassifier(vocab_size=vocab_size, embed_dim=HP["embed_dim"], hidden_dim=HP["hidden_dim"],
                                     num_labels=HP["num_labels"], num_layers=HP["num_layers"],
                                     dropout=HP["dropout"])


def forward_emb(model, emb):
    """Forward pass from embeddings, returns (logits, attention weights)."""
    import torch

    output, _ = model.lstm(model.dropout(emb))
    attn = torch.softmax(model.attention(output).squeeze(-1), dim=1)
    context = torch.bmm(attn.unsqueeze(1), output).squeeze(1)
    return model.fc(model.dropout(context)), attn


def predict(model, x, device, batch: int = 128) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            out.append(torch.sigmoid(model(x[s:s + batch].to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


# ---------------------------------------------------------------- training

def train_fold(fold: int, inputs: dict[str, Any], out: Path, device) -> tuple[Any, dict[str, int]]:
    import torch
    from sklearn.metrics import f1_score

    from mlcq_graphs.evaluation.cv import prediction_rows, tune_thresholds, write_jsonl
    from mlcq_graphs.training.losses import FocalLoss

    samples, split = inputs["samples"], inputs["folds"][fold]
    vocab = build_vocab([samples[i]["tokens"] for i in split["train"]])

    def tensors(part: str):
        idx = split[part]
        enc = [encode(samples[i]["tokens"], vocab) for i in idx]
        x = torch.tensor([e[0] for e in enc], dtype=torch.long)
        y = torch.tensor([samples[i]["y"] for i in idx], dtype=torch.float32)
        return np.array(idx), x, y

    data = {p: tensors(p) for p in ("train", "val", "test")}
    truncated = {p: int(sum(len(samples[i]["tokens"]) > MAX_LEN for i in split[p])) for p in split}
    unk_rate = {p: float((data[p][1] == UNK).sum() / max(1, (data[p][1] != PAD).sum())) for p in data}

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = new_model(len(vocab)).to(device)
    y_train = data["train"][2]
    # inverse-frequency class weights
    class_weight = (len(y_train) / (len(LABELS) * y_train.sum(0).clamp(min=1.0))).to(device)
    loss_fn = FocalLoss(gamma=HP["focal_gamma"], class_weight=class_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    gen = torch.Generator().manual_seed(SEED)

    best_f1, best_state, best_epoch, bad, history = -1.0, None, -1, 0, []
    t0 = time.time()
    for epoch in range(HP["max_epochs"]):
        model.train()
        perm = torch.randperm(len(y_train), generator=gen)
        total = 0.0
        for s in range(0, len(perm), HP["batch_size"]):
            b = perm[s:s + HP["batch_size"]]
            xb, yb = data["train"][1][b].to(device), y_train[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        p_val = predict(model, data["val"][1], device)
        f1 = float(f1_score(data["val"][2].numpy().astype(int), (p_val >= 0.5).astype(int), average="macro",
                            zero_division=0))
        history.append({"epoch": epoch, "loss": total / len(y_train), "val_f1_macro": f1})
        print(f"[fold {fold}] epoch {epoch:2d} loss {total / len(y_train):.4f} val macro F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(best_state)

    probs = {p: predict(model, data[p][1], device) for p in ("val", "test")}
    thresholds = tune_thresholds(data["val"][2].numpy(), probs["val"], HP["threshold_steps"])
    rows = []
    for p in ("val", "test"):
        rows += prediction_rows(model=MODEL, strategy="kfold", seed=SEED, fold=fold, part=p, indices=data[p][0],
                                y=data[p][2].numpy().astype(int), probs=probs[p])
    fdir = out / f"fold{fold}"
    fdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(fdir / "predictions.jsonl", rows)
    torch.save(best_state, fdir / "model.pt")
    (fdir / "vocab.json").write_text(json.dumps(vocab))
    (fdir / "train.json").write_text(json.dumps({
        "fold": fold, "hyperparameters": HP, "max_len": MAX_LEN, "min_freq": MIN_FREQ, "max_vocab": MAX_VOCAB,
        "vocab_size": len(vocab), "sizes": {p: len(split[p]) for p in split}, "truncated": truncated,
        "unk_rate": unk_rate, "best_epoch": best_epoch, "best_val_f1_macro": best_f1,
        "epochs_run": len(history), "train_seconds": time.time() - t0, "device": str(device),
        "thresholds_val": [float(t) for t in thresholds], "history": history}, indent=1))
    print(f"[fold {fold}] best epoch {best_epoch} val macro F1 {best_f1:.4f}, "
          f"{time.time() - t0:.0f}s, truncated {truncated}", flush=True)
    return model, vocab


# ---------------------------------------------------------------- attributions

def rank_desc(v: np.ndarray) -> np.ndarray:
    return np.argsort(-v, kind="stable")


def attribute_fold(fold: int, inputs: dict[str, Any], out: Path, device, model=None, vocab=None) -> None:
    import torch
    from scipy.stats import spearmanr

    fdir = out / f"fold{fold}"
    if model is None:
        vocab = json.loads((fdir / "vocab.json").read_text())
        model = new_model(len(vocab)).to(device)
        model.load_state_dict(torch.load(fdir / "model.pt", map_location=device))
    samples = inputs["samples"]
    # cuDNN LSTM gradients need train mode, so zero the dropout
    model.train()
    model.dropout.p = 0.0
    model.lstm.dropout = 0.0
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        base_emb = model.embedding(torch.zeros(1, MAX_LEN, dtype=torch.long, device=device))
        base_logits, _ = forward_emb(model, base_emb)
    alphas = ((torch.arange(IG_STEPS, dtype=torch.float32, device=device) + 0.5) / IG_STEPS).view(-1, 1, 1)

    check = model_eval_logits = None
    line_rows, del_rows, agree_rows, comp_rows = [], [], [], []
    tok_sum: dict[str, dict[str, float]] = {s: defaultdict(float) for s in LABELS}
    tok_cnt: dict[str, dict[str, int]] = {s: defaultdict(int) for s in LABELS}
    t0 = time.time()
    test = inputs["folds"][fold]["test"]
    for k_sample, j in enumerate(test):
        s = samples[j]
        smells = SMELLS_OF.get(s["level"], [])
        if not smells:
            continue
        ids, n = encode(s["tokens"], vocab)
        if n == 0:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        lines = np.array(s["lines"][:n])
        n_lines = s["n_lines"]
        with torch.no_grad():
            emb = model.embedding(x)
            logits, attn = forward_emb(model, emb)
        if check is None:
            # train mode with zero dropout must match eval mode
            model.eval()
            with torch.no_grad():
                model_eval_logits = model(x)
            model.train()
            check = float((model_eval_logits - logits).abs().max())
        attn_tok = attn[0, :n].cpu().numpy().astype(np.float64)
        pad_mass = float(1.0 - attn[0, :n].sum())
        probs = torch.sigmoid(logits[0]).cpu().numpy()

        # Integrated Gradients for each smell of the level
        path = (base_emb + alphas * (emb - base_emb)).detach().requires_grad_(True)
        path_logits, _ = forward_emb(model, path)
        ig = {}
        for c, smell in enumerate(smells):
            li = LABELS.index(smell)
            (grad,) = torch.autograd.grad(path_logits[:, li].sum(), path, retain_graph=c < len(smells) - 1)
            attr = ((emb - base_emb)[0] * grad.mean(0)).sum(-1)
            ig[smell] = attr[:n].detach().cpu().numpy().astype(np.float64)
            full = float(attr.sum())
            delta = float(logits[0, li] - base_logits[0, li])
            comp_rows.append({"fold": fold, "json_index": j, "smell": smell, "sum_attr": full,
                              "sum_attr_real_tokens": float(ig[smell].sum()), "delta_logit": delta,
                              "rel_error": abs(full - delta) / max(abs(delta), 1e-8)})

        # deletion: rankings by attention, IG and random orders
        rng = np.random.default_rng(SEED + j)
        orders = {"attention": rank_desc(attn_tok)}
        for smell in smells:
            orders[f"integrated_gradients:{smell}"] = rank_desc(ig[smell])
        for r in range(N_RANDOM):
            orders[f"random{r}"] = rng.permutation(n)
        variants, keys = [], []
        for frac in FRACS:
            kk = max(1, int(round(frac * n)))
            for name, order in orders.items():
                v = list(ids)
                for pos in order[:kk]:
                    v[pos] = PAD
                variants.append(v)
                keys.append((frac, kk, name))
        with torch.no_grad():
            dl = forward_emb(model, model.embedding(torch.tensor(variants, dtype=torch.long, device=device)))[0]
        dl = dl.cpu().numpy()
        for (frac, kk, name), row_logits in zip(keys, dl):
            for smell in smells:
                if name.startswith("integrated_gradients:") and not name.endswith(smell):
                    continue
                method = "integrated_gradients" if name.startswith("integrated_gradients") else \
                    ("random" if name.startswith("random") else name)
                li = LABELS.index(smell)
                drop = float(logits[0, li]) - float(row_logits[li])
                pdrop = float(probs[li]) - float(1 / (1 + math.exp(-row_logits[li])))
                del_rows.append({"fold": fold, "json_index": j, "smell": smell, "label": s["y"][li],
                                 "method": method, "draw": name if method == "random" else "", "frac": frac,
                                 "k": kk, "n_tokens": n, "logit_drop": drop, "prob_drop": pdrop})

        for smell in smells:
            li = LABELS.index(smell)
            # agreement attention vs IG over the real tokens
            kk = max(1, int(math.ceil(TOP_FRAC * n)))
            top_a, top_i = set(rank_desc(attn_tok)[:kk]), set(rank_desc(ig[smell])[:kk])
            rho = spearmanr(attn_tok, ig[smell])[0] if n >= 3 else float("nan")
            rho_abs = spearmanr(attn_tok, np.abs(ig[smell]))[0] if n >= 3 else float("nan")
            agree_rows.append({"fold": fold, "json_index": j, "smell": smell, "label": s["y"][li], "n_tokens": n,
                               "spearman_signed_ig": rho, "spearman_abs_ig": rho_abs, "top_k": kk,
                               "top10_overlap": len(top_a & top_i) / kk, "chance_overlap": kk / n})
            for method, tok_scores in (("attention", attn_tok), ("integrated_gradients", ig[smell])):
                ls = np.zeros(n_lines)
                np.add.at(ls, np.clip(lines, 0, n_lines - 1), tok_scores)
                line_rows.append({"model": MODEL, "method": method, "json_index": j, "smell": smell,
                                  "label": s["y"][li], "score": float(probs[li]),
                                  "line_scores": [round(float(v), 6) for v in ls], "fold": fold, "n_tokens": n,
                                  "truncated": len(s["tokens"]) > MAX_LEN,
                                  **({"pad_attention": pad_mass} if method == "attention" else {})})
            if s["y"][li] == 1:
                for t, v in zip(s["tokens"][:n], ig[smell]):
                    tok_sum[smell][t] += float(v)
                    tok_cnt[smell][t] += 1
        if (k_sample + 1) % 200 == 0:
            print(f"[fold {fold}] attributed {k_sample + 1}/{len(test)} test samples, {time.time() - t0:.0f}s",
                  flush=True)

    common.write_rows(fdir / "line_attributions.jsonl", line_rows)
    write_csv(fdir / "deletion_rows.csv", del_rows)
    write_csv(fdir / "agreement_rows.csv", agree_rows)
    write_csv(fdir / "completeness_rows.csv", comp_rows)
    (fdir / "token_ig.json").write_text(json.dumps({s: {t: [tok_sum[s][t], tok_cnt[s][t]] for t in tok_cnt[s]}
                                                    for s in LABELS}))
    (fdir / "attribute.json").write_text(json.dumps({"train_vs_eval_max_abs_logit_diff": check,
                                                     "seconds": time.time() - t0, "device": str(device)}))
    print(f"[fold {fold}] attributions done in {time.time() - t0:.0f}s (train/eval logit diff {check:.2e})",
          flush=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- merge

def _mean(v: list[float]) -> float:
    v = [x for x in v if not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def _median(v: list[float]) -> float:
    v = [x for x in v if not math.isnan(x)]
    return float(np.median(v)) if v else float("nan")


def _r(x: float) -> float:
    return round(float(x), 4)


def merge(out: Path, k: int = 5) -> None:
    fdirs = [out / f"fold{f}" for f in range(k)]
    with (out / "predictions.jsonl").open("w") as fh:
        for d in fdirs:
            fh.write((d / "predictions.jsonl").read_text())
    rows = [r for d in fdirs for r in common.read_rows(d / "line_attributions.jsonl")]
    common.write_rows(out / "line_attributions.jsonl", rows)

    trunc = []
    for d in fdirs:
        t = json.loads((d / "train.json").read_text())
        trunc.append({"fold": t["fold"], "vocab_size": t["vocab_size"], "best_epoch": t["best_epoch"],
                      "epochs_run": t["epochs_run"], "best_val_f1_macro": _r(t["best_val_f1_macro"]),
                      "train_seconds": round(t["train_seconds"], 1),
                      **{f"truncated_{p}": t["truncated"][p] for p in ("train", "val", "test")},
                      **{f"n_{p}": t["sizes"][p] for p in ("train", "val", "test")},
                      "unk_rate_test": _r(t["unk_rate"]["test"])})
    write_csv(out / "truncation.csv", trunc)

    # deletion: one value per (sample, smell, method, k), the random draws averaged
    per: dict[tuple, list[list[float]]] = defaultdict(list)
    labels: dict[tuple, int] = {}
    for d in fdirs:
        for r in read_csv(d / "deletion_rows.csv"):
            key = (r["smell"].removeprefix("is_"), r["method"], float(r["frac"]), int(r["json_index"]))
            per[key].append([float(r["logit_drop"]), float(r["prob_drop"])])
            labels[key] = int(r["label"])
    cells: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, vals in per.items():
        ld, pdrop = np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals])
        c = cells[key[:3]]
        c["logit"].append(ld)
        c["prob"].append(pdrop)
        if labels[key] == 1:
            c["logit_pos"].append(ld)
            c["prob_pos"].append(pdrop)
    aopc = defaultdict(list)
    for (smell, method, frac), c in cells.items():
        aopc[(smell, method)].append(np.mean(c["logit"]))
    tab = []
    for (smell, method, frac), c in sorted(cells.items(), key=lambda kv: (LABELS.index("is_" + kv[0][0]),
                                                                          kv[0][1], kv[0][2])):
        tab.append({"smell": smell, "method": method, "frac": frac, "n": len(c["logit"]),
                    "logit_drop": _r(np.mean(c["logit"])), "prob_drop": _r(np.mean(c["prob"])),
                    "n_pos": len(c["logit_pos"]), "logit_drop_pos": _r(_mean(c["logit_pos"])),
                    "prob_drop_pos": _r(_mean(c["prob_pos"])),
                    # AOPC: mean drop over the values of k
                    "aopc_logit": _r(np.mean(aopc[(smell, method)]))})
    write_csv(out / "deletion.csv", tab)

    agree = [r for d in fdirs for r in read_csv(d / "agreement_rows.csv")]
    atab = []
    for subset in ("all", "positive"):
        for lab in LABELS:
            sub = [r for r in agree if r["smell"] == lab and (subset == "all" or r["label"] == "1")]
            if not sub:
                continue

            def col(name: str) -> list[float]:
                return [float(r[name]) for r in sub]

            atab.append({"smell": lab.removeprefix("is_"), "subset": subset, "n": len(sub),
                         "spearman_signed_ig_mean": _r(_mean(col("spearman_signed_ig"))),
                         "spearman_signed_ig_median": _r(_median(col("spearman_signed_ig"))),
                         "spearman_abs_ig_mean": _r(_mean(col("spearman_abs_ig"))),
                         "top10_overlap_mean": _r(_mean(col("top10_overlap"))),
                         "chance_overlap_mean": _r(_mean(col("chance_overlap")))})
    write_csv(out / "agreement.csv", atab)

    comp = [r for d in fdirs for r in read_csv(d / "completeness_rows.csv")]
    ctab = []
    for lab in LABELS:
        sub = [r for r in comp if r["smell"] == lab]
        if not sub:
            continue
        err = np.array([float(r["rel_error"]) for r in sub])
        delta = np.array([float(r["delta_logit"]) for r in sub])
        ctab.append({"smell": lab.removeprefix("is_"), "n": len(sub), "mean_rel_error": _r(err.mean()),
                     "median_rel_error": _r(np.median(err)), "p90_rel_error": _r(np.quantile(err, 0.9)),
                     "mean_abs_delta": _r(np.abs(delta).mean())})
    write_csv(out / "completeness.csv", ctab)

    tok = []
    sums: dict[str, dict[str, list[float]]] = {s: defaultdict(lambda: [0.0, 0]) for s in LABELS}
    for d in fdirs:
        for s, m in json.loads((d / "token_ig.json").read_text()).items():
            for t, (v, c) in m.items():
                sums[s][t][0] += v
                sums[s][t][1] += c
    for s in LABELS:
        cand = [(t, v / c, c) for t, (v, c) in sums[s].items() if c >= MIN_TOKEN_COUNT and v / c > 0]
        cand.sort(key=lambda r: -r[1])
        for rank, (t, m, c) in enumerate(cand[:N_TOP_TOKENS], 1):
            tok.append({"smell": s.removeprefix("is_"), "rank": rank, "token": t, "mean_ig": round(m, 5),
                        "count": c})
    write_csv(out / "top_tokens.csv", tok)
    print(f"[merge] {len(rows)} line rows; wrote predictions.jsonl, line_attributions.jsonl, deletion.csv, "
          f"agreement.csv, completeness.csv, top_tokens.csv, truncation.csv in {out}")


# ---------------------------------------------------------------- report

def md_table(rows: list[list[Any]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def report(out: Path, records_path: Path) -> None:
    parts = ["# BiLSTM with attention on MLCQ: interpretability study", ""]
    trunc = read_csv(out / "truncation.csv")
    inputs = load_inputs(out)
    n_long = sum(len(s["tokens"]) > MAX_LEN for s in inputs["samples"].values())
    parts += [f"Model: BiLSTMAttentionClassifier (embed 128, hidden 256, 2 layers, dropout 0.3), AST token texts, "
              f"vocabulary of each training part (min frequency {MIN_FREQ}, at most {MAX_VOCAB} entries), "
              f"input padded or truncated to {MAX_LEN} tokens. {n_long} of {len(inputs['samples'])} snippets "
              f"({100 * n_long / len(inputs['samples']):.1f}%) have more than {MAX_LEN} tokens and are truncated. "
              f"Focal loss (gamma 2, inverse-frequency class weights), Adam, early stopping on validation macro F1. "
              f"One model per fold of the 5-fold split of the reference GCN runs (seed 42).", ""]
    parts += [md_table([[r["fold"], r["vocab_size"], r["best_epoch"], r["epochs_run"], r["best_val_f1_macro"],
                         r["truncated_test"], r["unk_rate_test"]] for r in trunc],
                       ["fold", "vocab", "best epoch", "epochs run", "val macro F1 (0.5)", "truncated test",
                        "UNK rate test"]), ""]

    recs = json.loads(records_path.read_text())
    parts += ["## Detection on the samples of each smell's level", "",
              f"scripts/cv/eval_eligible.py, thresholds re-tuned on the eligible validation samples. Mean and standard "
              f"deviation over the {len(recs)} folds (one seed).", ""]
    rows = []
    for lab in LABELS:
        mcc = [r["per_label_tuned"][lab]["mcc"] for r in recs if r["per_label_tuned"][lab]["mcc"] is not None]
        f1 = [r["per_label_tuned"][lab]["f1"] for r in recs]
        rows.append([common.NAMES[lab], f"{np.mean(mcc):.3f} ({np.std(mcc):.3f})",
                     f"{np.mean(f1):.3f} ({np.std(f1):.3f})"])
    rows.append(["macro", f"{np.mean([r['mcc_tuned'] for r in recs]):.3f}",
                 f"{np.mean([r['f1_macro_tuned'] for r in recs]):.3f}"])
    parts += [md_table(rows, ["smell", "MCC", "F1"]), ""]

    comp = read_csv(out / "completeness.csv")
    att = [json.loads((out / f"fold{f}" / "attribute.json").read_text()) for f in range(5)]
    parts += ["## Integrated Gradients completeness", "",
              f"Relative error |sum of attributions - (logit(x) - logit(baseline))| / |logit(x) - logit(baseline)|, "
              f"{IG_STEPS} midpoint steps, all eligible test pairs. The largest gap between the train-mode "
              f"(dropout zeroed, used for the gradients) and eval-mode logits is "
              f"{max(a['train_vs_eval_max_abs_logit_diff'] for a in att):.1e}.", ""]
    parts += [md_table([[r["smell"], r["n"], r["mean_rel_error"], r["median_rel_error"], r["p90_rel_error"],
                         r["mean_abs_delta"]] for r in comp],
                       ["smell", "pairs", "mean", "median", "90th percentile", "mean abs logit gap"]), ""]

    dele = read_csv(out / "deletion.csv")
    cell = {(r["smell"], r["method"], float(r["frac"])): r for r in dele}
    parts += ["## Faithfulness by deletion", "",
              "Mean drop of the smell logit when the top k% real tokens (ranked by each method) are replaced by "
              "PAD, on all eligible test samples, then on the positive ones only (random: mean of 3 draws). AOPC "
              "is the mean drop over the three values of k. A larger drop means the ranking finds tokens the "
              "prediction depends on.", ""]
    rows = []
    for smell in dict.fromkeys(r["smell"] for r in dele):
        for method in ("attention", "integrated_gradients", "random"):
            rows.append([smell, method] + [f"{float(cell[(smell, method, f)]['logit_drop']):.3f}" for f in FRACS]
                        + [f"{float(cell[(smell, method, f)]['logit_drop_pos']):.3f}" for f in FRACS]
                        + [f"{float(cell[(smell, method, FRACS[0])]['aopc_logit']):.3f}"])
    parts += [md_table(rows, ["smell", "method", "5%", "10%", "20%", "5% pos", "10% pos", "20% pos", "AOPC"]), ""]

    agree = read_csv(out / "agreement.csv")
    parts += ["## Agreement of attention and Integrated Gradients", "",
              "Per sample, Spearman correlation over the real tokens between the attention weight and the IG "
              "attribution (signed, and absolute value), and the share of the top 10% tokens by attention that "
              "are also in the top 10% by signed IG. The chance column is the overlap expected from a random "
              "ranking (k/n).", ""]
    parts += [md_table([[r["smell"], r["subset"], r["n"], r["spearman_signed_ig_mean"],
                         r["spearman_signed_ig_median"], r["spearman_abs_ig_mean"], r["top10_overlap_mean"],
                         r["chance_overlap_mean"]] for r in agree],
                       ["smell", "samples", "n", "Spearman signed (mean)", "Spearman signed (median)",
                        "Spearman abs (mean)", "top 10% overlap", "chance"]), ""]

    tok = read_csv(out / "top_tokens.csv")
    parts += ["## Tokens with the highest mean IG on positive test samples", "",
              f"Mean signed IG per occurrence (count in parentheses), over the positive test samples of each smell, "
              f"tokens seen at least {MIN_TOKEN_COUNT} times. First 15 here, {N_TOP_TOKENS} in top_tokens.csv.", ""]
    for smell in dict.fromkeys(r["smell"] for r in tok):
        sub = [r for r in tok if r["smell"] == smell][:15]
        parts.append(f"- {smell}: " + ", ".join(f"`{r['token']}` ({float(r['mean_ig']):.3f}, {r['count']})"
                                                for r in sub))
    parts.append("")
    parts += ["## Example snippets", "",
              "IG line score (sum of the token attributions of the line) at the left, for the positive test "
              "sample of each smell with the highest predicted probability among snippets of at most 40 lines.", ""]
    rows = [r for r in common.read_rows(out / "line_attributions.jsonl") if r["method"] == "integrated_gradients"]
    for smell in ("is_feature_envy", "is_long_method", "is_data_class"):
        cand = [r for r in rows if r["smell"] == smell and r["label"] == 1
                and len(common.snippet_lines(r["json_index"])) <= 40]
        if not cand:
            continue
        r = max(cand, key=lambda r: r["score"])
        code = common.snippet_lines(r["json_index"])
        parts += [f"### {common.NAMES[smell]}: json_index {r['json_index']} (fold {r['fold']}, "
                  f"probability {r['score']:.3f})", "", "```"]
        for line, v in zip(code, r["line_scores"]):
            parts.append(f"{v:+8.3f} | {line}")
        parts += ["```", ""]
    (out / "REPORT.md").write_text("\n".join(parts))
    print(f"[report] wrote {out / 'REPORT.md'}")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["prepare", "run", "train", "attribute", "merge", "report"])
    ap.add_argument("--fold", type=int, action="append", help="fold(s) for run/train/attribute (default: all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--records", type=Path,
                    default=PROJECT_ROOT / "artifacts/analysis/mlcq/eligible_all" / MODEL / "records.json")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out

    if args.phase == "prepare":
        return prepare(out)
    if args.phase == "merge":
        return merge(out)
    if args.phase == "report":
        return report(out, args.records)
    inputs = load_inputs(out)
    device = pick_device(args.device)
    for fold in args.fold if args.fold is not None else sorted(inputs["folds"]):
        model = vocab = None
        if args.phase in ("run", "train"):
            model, vocab = train_fold(fold, inputs, out, device)
        if args.phase in ("run", "attribute"):
            attribute_fold(fold, inputs, out, device, model, vocab)


if __name__ == "__main__":
    main()
