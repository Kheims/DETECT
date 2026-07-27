"""Numbers of the interpretability appendix, computed from the stored results.

Intervals come from a repository bootstrap. Writes artifacts/xai/appendix/tables.json.

Usage:
    python scripts/xai/xai_appendix_tables.py [--draws 2000]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.xai import common  # noqa: E402

OUT = ROOT / "artifacts/xai/appendix"
SURVEY_LABELS = ROOT / "data/derived/MLCQCodeSmellSamples.normalized.rule_default.json"
NAMES = {"is_feature_envy": "Feature Envy", "is_long_method": "Long Method",
         "is_blob": "Blob", "is_data_class": "Data Class"}
SEV = {"is_feature_envy": "feature envy", "is_long_method": "long method",
       "is_blob": "blob", "is_data_class": "data class"}


class Labels:
    def __init__(self, path: Path):
        self.samples = json.loads(path.read_text())
        self.repo = [s["repo_url"] for s in self.samples]

    def y(self, j: int, smell: str) -> int:
        return int(self.samples[j]["y"][common.LABELS.index(smell)])


def boot_mean_diff(values: dict[str, list[float]], draws: int, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-sample values (grouped by repository) and its repository bootstrap interval."""
    repos = list(values)
    sums = np.array([np.sum(values[r]) for r in repos])
    counts = np.array([len(values[r]) for r in repos])
    point = sums.sum() / counts.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(repos), size=(draws, len(repos)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def by_repo(pairs: list[tuple[str, float]]) -> dict[str, list[float]]:
    d = defaultdict(list)
    for r, v in pairs:
        d[r].append(v)
    return d


# ---------------------------------------------------------------- deletion
def gcn_deletion(lab: Labels, draws: int) -> list[dict]:
    rows = list(csv.DictReader((ROOT / "artifacts/xai/gcn/deletion.csv").open()))
    draws_cols = sorted({c.split("_")[0] for c in rows[0] if c.startswith("random")})
    out = []
    for smell in common.LABELS:
        for subset in ("positive", "all"):
            top, rnd, diff = [], [], []
            for r in rows:
                if r["smell"] != smell:
                    continue
                j = int(r["json_index"])
                if subset == "positive" and not lab.y(j, smell):
                    continue
                t = np.mean([float(r[f"ig_{p}"]) for p in (5, 10, 20)])
                x = np.mean([np.mean([float(r[f"{d}_{p}"]) for p in (5, 10, 20)]) for d in draws_cols])
                top.append((lab.repo[j], t))
                rnd.append((lab.repo[j], x))
                diff.append((lab.repo[j], t - x))
            out.append({"model": "GCN", "smell": smell, "subset": subset, "n": len(top),
                        "top": boot_mean_diff(by_repo(top), draws)[0],
                        "random": boot_mean_diff(by_repo(rnd), draws)[0],
                        "diff": boot_mean_diff(by_repo(diff), draws), "random_draws": len(draws_cols)})
    return out


def bilstm_deletion(lab: Labels, draws: int) -> list[dict]:
    per = defaultdict(lambda: defaultdict(list))  # (j, smell) -> method -> logit drops
    for p in sorted((ROOT / "artifacts/xai/bilstm").glob("fold*/deletion_rows.csv")):
        for r in csv.DictReader(p.open()):
            per[(int(r["json_index"]), r["smell"])][r["method"]].append(float(r["logit_drop"]))
    out = []
    for smell in common.LABELS:
        for subset in ("positive", "all"):
            vals = defaultdict(list)
            n = 0
            for (j, s), m in per.items():
                if s != smell or (subset == "positive" and not lab.y(j, smell)):
                    continue
                if not {"integrated_gradients", "attention", "random"} <= set(m):
                    continue
                n += 1
                aopc = {k: float(np.mean(v)) for k, v in m.items()}
                r = lab.repo[j]
                for k in ("integrated_gradients", "attention", "random"):
                    vals[k].append((r, aopc[k]))
                vals["ig_minus_random"].append((r, aopc["integrated_gradients"] - aopc["random"]))
                vals["att_minus_random"].append((r, aopc["attention"] - aopc["random"]))
            out.append({"model": "BiLSTM", "smell": smell, "subset": subset, "n": n,
                        **{k: (boot_mean_diff(by_repo(v), draws) if "minus" in k else boot_mean_diff(by_repo(v), draws)[0])
                           for k, v in vals.items()}})
    return out


# ---------------------------------------------------------------- ERASER
def eraser(lab: Labels, draws: int) -> list[dict]:
    rows = common.read_rows(ROOT / "artifacts/xai/llm_occlusion/qwen7b_line_attributions.jsonl")
    out = []
    for smell in common.LABELS:
        rs = [r for r in rows if r["smell"] == smell and r["method"] == "occlusion"]
        g = lambda f: by_repo([(lab.repo[r["json_index"]], f(r)) for r in rs])  # noqa: E731
        out.append({"smell": smell, "pairs": len(rs),
                    "positives": sum(lab.y(r["json_index"], smell) for r in rs),
                    "chunked": sum(bool(r["chunked"]) for r in rs),
                    "truncated": sum(bool(r["truncated"]) for r in rs),
                    "comp": boot_mean_diff(g(lambda r: r["comprehensiveness"]), draws)[0],
                    "comp_random": boot_mean_diff(g(lambda r: r["random_comprehensiveness"]), draws)[0],
                    "suff": boot_mean_diff(g(lambda r: r["sufficiency"]), draws)[0],
                    "suff_random": boot_mean_diff(g(lambda r: r["random_sufficiency"]), draws)[0],
                    "comp_diff": boot_mean_diff(g(lambda r: r["comprehensiveness"] - r["random_comprehensiveness"]), draws),
                    "suff_diff": boot_mean_diff(g(lambda r: r["sufficiency"] - r["random_sufficiency"]), draws),
                    "forward_passes": int(sum(r["forward_passes"] for r in rs))})
    return out


# ---------------------------------------------------------------- Feature Envy lines
def load_line_sources() -> dict[str, dict[tuple[int, str], dict]]:
    src = {"GCN, IG": ("artifacts/xai/gcn/line_attributions.jsonl", "integrated_gradients"),
           "BiLSTM, IG": ("artifacts/xai/bilstm/line_attributions.jsonl", "integrated_gradients"),
           "BiLSTM, attention": ("artifacts/xai/bilstm/line_attributions.jsonl", "attention"),
           "Qwen 7B, occlusion": ("artifacts/xai/llm_occlusion/qwen7b_line_attributions.jsonl", "occlusion")}
    out = {}
    for name, (path, method) in src.items():
        out[name] = {(r["json_index"], r["smell"]): r for r in common.read_rows(ROOT / path) if r["method"] == method}
    return out


def fe_lines(lab: Labels, src: dict, draws: int) -> list[dict]:
    access = {}
    for line in (ROOT / "artifacts/xai/access_lines.jsonl").open():
        r = json.loads(line)
        access[r["json_index"]] = r
    out = []
    for name, data in src.items():
        for label in (1, 0):
            mass, share, node_share, mass_minus_nodes, hit, hit_minus, size_hit, hit_minus_size = [], [], [], [], [], [], [], []
            mass_node_share, no_positive = [], 0
            for (j, smell), r in data.items():
                if smell != "is_feature_envy" or lab.y(j, smell) != label or j not in access:
                    continue
                n = len(r["line_scores"])
                fl = [i for i in access[j]["foreign_lines"] if i < n]
                if not fl or n < 3:
                    continue
                cnt = np.zeros(n)
                for nd in common.ast_nodes(j):
                    if 0 <= nd.start_line < n:
                        cnt[nd.start_line] += 1
                s = np.array(r["line_scores"], float)
                pos = np.clip(s, 0, None)
                ns = cnt[fl].sum() / cnt.sum() if cnt.sum() > 0 else len(fl) / n
                h = float(int(np.argmax(s)) in fl)
                rep = lab.repo[j]
                share.append((rep, len(fl) / n))
                node_share.append((rep, ns))
                if pos.sum() > 0:
                    # undefined without a positive line score
                    m = pos[fl].sum() / pos.sum()
                    mass.append((rep, m))
                    mass_node_share.append((rep, ns))
                    mass_minus_nodes.append((rep, m - ns))
                else:
                    no_positive += 1
                hs = float(int(np.argmax(cnt)) in fl)
                hit.append((rep, h))
                hit_minus.append((rep, h - len(fl) / n))
                size_hit.append((rep, hs))
                hit_minus_size.append((rep, h - hs))
            if not share:
                continue
            out.append({"explainer": name, "label": label, "n": len(share), "n_no_positive": no_positive,
                        "mass": boot_mean_diff(by_repo(mass), draws)[0],
                        "mass_node_share": boot_mean_diff(by_repo(mass_node_share), draws)[0],
                        "share": boot_mean_diff(by_repo(share), draws)[0],
                        "node_share": boot_mean_diff(by_repo(node_share), draws)[0],
                        "mass_minus_node_share": boot_mean_diff(by_repo(mass_minus_nodes), draws),
                        "top_hit": boot_mean_diff(by_repo(hit), draws)[0],
                        "top_hit_minus_chance": boot_mean_diff(by_repo(hit_minus), draws),
                        "size_top_hit": boot_mean_diff(by_repo(size_hit), draws)[0],
                        "top_hit_minus_size": boot_mean_diff(by_repo(hit_minus_size), draws)})
    return out


# ---------------------------------------------------------------- randomization
def sanity_repo(lab: Labels, draws: int) -> list[dict]:
    """Randomization test: per-snippet mean over seeds, with a repository bootstrap."""
    per = defaultdict(lambda: defaultdict(list))
    by_seed = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader((ROOT / "artifacts/xai/gcn/sanity_rows.csv").open()):
        per[(r["smell"], r["randomization"])][int(r["json_index"])].append((float(r["spearman_abs"]), float(r["spearman"])))
        by_seed[(r["smell"], r["randomization"])][int(r["seed"])].append(float(r["spearman"]))
    out = []
    for (smell, rnd), snips in per.items():
        mag = [(lab.repo[j], float(np.mean([v[0] for v in vs]))) for j, vs in snips.items()]
        sgn = [(lab.repo[j], float(np.mean([v[1] for v in vs]))) for j, vs in snips.items()]
        seeds = sorted({len(vs) for vs in snips.values()})
        out.append({"smell": smell, "randomization": rnd, "snippets": len(snips), "seeds": seeds,
                    "repos": len({lab.repo[j] for j in snips}),
                    "magnitude": boot_mean_diff(by_repo(mag), draws), "signed": boot_mean_diff(by_repo(sgn), draws),
                    "signed_seed_means": {str(k): float(np.mean(v)) for k, v in sorted(by_seed[(smell, rnd)].items())}})
    return out


# ---------------------------------------------------------------- agreement
def agreement(src: dict) -> list[dict]:
    names = ["GCN, IG", "BiLSTM, IG", "BiLSTM, attention"]
    out = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = src[names[a]], src[names[b]]
            for smell in common.LABELS:
                rho, same, chance = [], [], []
                for key in set(A) & set(B):
                    if key[1] != smell:
                        continue
                    x, y = np.array(A[key]["line_scores"], float), np.array(B[key]["line_scores"], float)
                    if "attention" in names[a] or "attention" in names[b]:
                        # attention has no sign, compare magnitudes
                        x, y = np.abs(x), np.abs(y)
                    if len(x) != len(y) or len(x) < 4 or x.std() == 0 or y.std() == 0:
                        continue
                    rho.append(stats.spearmanr(x, y).statistic)
                    same.append(int(np.argmax(x)) == int(np.argmax(y)))
                    chance.append(1 / len(x))
                out.append({"a": names[a], "b": names[b], "smell": smell, "pairs": len(rho),
                            "spearman": float(np.nanmean(rho)), "same_top": float(np.mean(same)),
                            "same_top_chance": float(np.mean(chance))})
    return out


# ---------------------------------------------------------------- reviewer consensus
def consensus(lab: Labels) -> list[dict]:
    index = {(s["repo_url"], s["commit_hash"], s["file_path"], int(s["start_line"]), int(s["end_line"])): i
             for i, s in enumerate(lab.samples)}
    # one vote per reviewer: the latest review of a reviewer for a (snippet, smell)
    latest: dict[tuple, dict[str, tuple[str, str]]] = defaultdict(dict)
    with (ROOT / "data/mlcq_zenodo/MLCQCodeSmellSamples.csv").open(newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            k = (r["repository"], r["commit_hash"], r["path"], int(r["start_line"]), int(r["end_line"]))
            if k in index:
                cur = latest[(index[k], r["smell"])].get(r["reviewer_id"])
                if cur is None or r["review_timestamp"] > cur[0]:
                    latest[(index[k], r["smell"])][r["reviewer_id"]] = (r["review_timestamp"], r["severity"])
    votes = {key: [sev for _, sev in d.values()] for key, d in latest.items()}
    CONSENSUS_VOTES.clear()
    CONSENSUS_VOTES.update(votes)
    llm = {}
    for line in (ROOT / "artifacts/llm_judge/qwen36_27b_fp8/scores.jsonl").open():
        r = json.loads(line)
        if r.get("applicable"):
            llm[(r["json_index"], r["smell"])] = r["score"]
    bil = {}
    for line in (ROOT / "artifacts/xai/bilstm/predictions.jsonl").open():
        r = json.loads(line)
        if r["part"] == "test":
            for i, s in enumerate(common.LABELS):
                bil[(r["json_index"], s)] = r["prob"][i]
    out = []
    for smell in common.LABELS:
        groups = defaultdict(list)
        for (j, name), v in votes.items():
            if name != SEV[smell] or (j, smell) not in llm or len(v) < 2:
                continue
            present = sum(s != "none" for s in v)
            y = lab.y(j, smell)
            g = ("unanimous positive" if present == len(v) else
                 "unanimous negative" if present == 0 else
                 "split, labeled positive" if y else "split, labeled negative")
            groups[g].append((llm[(j, smell)], bil.get((j, smell))))
        for g in ("unanimous positive", "split, labeled positive", "split, labeled negative", "unanimous negative"):
            r = groups.get(g, [])
            b = [x[1] for x in r if x[1] is not None]
            out.append({"smell": smell, "group": g, "n": len(r),
                        "llm_yes": float(np.mean([x[0] >= 0.5 for x in r])) if r else None,
                        "bilstm_n": len(b), "bilstm_yes": float(np.mean([x >= 0.5 for x in b])) if b else None})
        pos = [j for j in range(len(lab.samples)) if lab.y(j, smell)]
        nrev = [len(votes.get((j, SEV[smell]), [])) for j in pos]
        out.append({"smell": smell, "group": "positives", "n": len(pos),
                    "positives_by_reviewers": {str(k): nrev.count(k) for k in sorted(set(nrev))},
                    "llm_yes": None, "bilstm_n": None, "bilstm_yes": None})
    return out


CONSENSUS_VOTES: dict = {}


def unanimous_example(lab: Labels, src: dict) -> dict:
    """Unanimous Feature Envy positive with the most reviewers, explained by all four explainers."""
    cands = []
    for (j, name), v in CONSENSUS_VOTES.items():
        if name != "feature envy" or len(v) < 2 or any(s == "none" for s in v) or not lab.y(j, "is_feature_envy"):
            continue
        if all((j, "is_feature_envy") in d for d in src.values()):
            cands.append((-len(v), j, v))
    cands.sort()
    return {"candidates": [(j, v) for _, j, v in cands], "chosen": cands[0][1] if cands else None}


# ---------------------------------------------------------------- reasoning and tools
def reasoning() -> list[dict]:
    rows = list(csv.DictReader((ROOT / "artifacts/context_study/report_rule_default/qwen36_27b_fp8/contrasts.csv").open()))
    out = []
    for r in rows:
        if (r["metric"] == "mcc" and r["scope"] == "subset" and r["strategy"] == "kfold"
                and r["threshold"] == "frozen" and r["score_temperature"] in ("1.0", "1")
                and r["contrast"] in ("REASON - C0", "C4 - REASON")):
            lo, hi = json.loads(r["pooled_diff_ci"])
            out.append({"smell": r["smell"], "contrast": r["contrast"], "pairs": int(r["n"]),
                        "positives": int(r["positives"]), "diff": float(r["pooled_diff"]), "lo": lo, "hi": hi,
                        "a": float(r["pooled_a"]), "b": float(r["pooled_b"])})
    return out


def score_extremes() -> list[dict]:
    root = ROOT / "artifacts/context_study/scores/qwen36_27b_fp8"
    sc = {}
    for c in ("C0", "REASON", "C4"):
        files = sorted((root / c).glob("scores.shard*.jsonl")) or [root / c / "scores.jsonl"]
        d = {}
        for f in files:
            for line in f.open():
                r = json.loads(line)
                if r.get("applicable") is not False and r.get("score") is not None:
                    d[(r["json_index"], r["smell"])] = r["score"]
        sc[c] = d
    common_keys = set.intersection(*(set(v) for v in sc.values()))
    out = []
    for c, d in sc.items():
        v = np.array([d[k] for k in common_keys])
        out.append({"condition": c, "pairs": len(v), "near_0_or_1": float(np.mean((v < 0.01) | (v > 0.99))),
                    "between_0.1_and_0.9": float(np.mean((v >= 0.1) & (v <= 0.9)))})
    return out


def line_size(src: dict, draws: int, lab: "Labels") -> list[dict]:
    """Spearman between |line score| and the number of AST nodes on the line."""
    nodes_per_line: dict[int, np.ndarray] = {}
    out = []
    for name in ("GCN, IG", "BiLSTM, IG", "BiLSTM, attention", "Qwen 7B, occlusion"):
        for smell in common.LABELS:
            vals = []
            for (j, s), r in src[name].items():
                if s != smell:
                    continue
                if j not in nodes_per_line:
                    n_lines = len(common.snippet_lines(j))
                    cnt = np.zeros(n_lines)
                    try:
                        for nd in common.ast_nodes(j):
                            if 0 <= nd.start_line < n_lines:
                                cnt[nd.start_line] += 1
                    except Exception:  # snippets without an AST
                        cnt[:] = np.nan
                    nodes_per_line[j] = cnt
                cnt = nodes_per_line[j]
                x = np.abs(np.array(r["line_scores"], float))
                if len(x) != len(cnt) or len(x) < 4 or np.isnan(cnt).any() or x.std() == 0 or cnt.std() == 0:
                    continue
                vals.append((lab.repo[j], float(stats.spearmanr(x, cnt).statistic)))
            if vals:
                out.append({"explainer": name, "smell": smell, "n": len(vals),
                            "spearman_abs_vs_nodes": boot_mean_diff(by_repo(vals), draws)})
    return out


def setup_mcc() -> list[dict]:
    """Mean seed-42 kfold test MCC of the explained detectors, per smell."""
    files = {"GCN": "artifacts/analysis/rescoring/train_stored/rule_default_eligible/gnn_gcn/records.json",
             "BiLSTM": "artifacts/analysis/rescoring_xai/train_stored/rule_default_eligible/bilstm_attention/records.json",
             "Qwen 7B": "artifacts/analysis/rescoring_xai/train_stored/rule_default_eligible/judge_qwen7b/records.json"}
    out = []
    for name, path in files.items():
        recs = json.loads((ROOT / path).read_text())
        recs = recs if isinstance(recs, list) else recs.get("records", [])
        cells = [r for r in recs if r["seed"] == 42 and r["combo"]["split_strategy"] == "kfold"]
        out.append({"model": name, "folds": len(cells),
                    **{s: float(np.mean([r["per_label_tuned"][s]["mcc"] for r in cells])) for s in common.LABELS}})
    return out


def agent_stops() -> dict:
    root = ROOT / "artifacts/context_study/scores/qwen36_27b_fp8/C4"
    c = defaultdict(int)
    for f in root.glob("transcripts.shard*.jsonl"):
        for line in f.open():
            c[json.loads(line).get("stop_reason")] += 1
    return dict(c)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=str(SURVEY_LABELS))
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    lab = Labels(Path(args.labels))
    src = load_line_sources()
    res = {"labels": str(Path(args.labels).relative_to(ROOT)), "draws": args.draws,
           "gcn_runs": json.loads((ROOT / "artifacts/xai/gcn/summary.json").read_text()).get("runs"),
           "gcn_summary": json.loads((ROOT / "artifacts/xai/gcn/summary.json").read_text()),
           "gcn_deletion": gcn_deletion(lab, args.draws), "bilstm_deletion": bilstm_deletion(lab, args.draws),
           "eraser": eraser(lab, args.draws), "fe_lines": fe_lines(lab, src, args.draws),
           "agreement": agreement(src), "line_size": line_size(src, args.draws, lab), "consensus": consensus(lab),
           "unanimous_example": unanimous_example(lab, src), "reasoning": reasoning(),
           "score_extremes": score_extremes(), "agent_stops": agent_stops(), "setup_mcc": setup_mcc(),
           "sanity": list(csv.DictReader((ROOT / "artifacts/xai/gcn/sanity.csv").open())),
           "sanity_repo": sanity_repo(lab, args.draws)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.json").write_text(json.dumps(res, indent=1))
    print(f"[appendix] wrote {OUT / 'tables.json'}")


if __name__ == "__main__":
    main()
