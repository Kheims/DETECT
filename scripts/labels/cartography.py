"""Dataset cartography of MLCQ from the training dynamics of fine-tuned code models.

Confidence, variability and correctness per training sample (Swayamdipta et al. 2020),
regions compared on review attributes. Writes summary.json, samples.csv and figures.

Usage:
    rsync -av --include '*/' --include 'dynamics.npz' --include 'record.json' --exclude '*' \\
      slurm-ext:code_smells_detection/fe_relational/artifacts/code_finetune/code_finetune/ \\
      artifacts/code_finetune/code_finetune/
    python scripts/labels/cartography.py --config config/experiments/cartography.yml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from scipy.stats.contingency import odds_ratio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402

# Okabe and Ito colour-blind safe palette
OKABE_ITO = {"orange": "#E69F00", "sky_blue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
             "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
ITEM_RE = re.compile(r"^(?P<strategy>.+)_s(?P<seed>\d+)_f(?P<fold>\d+)$")
GROUPS = ("positives", "negatives")
REGIONS = ("easy", "ambiguous", "hard", "other")
BINARY = ("minor_only", "any_present_vote", "non_unanimous", "rule_inconsistent", "single_review", "truncated")
REVIEW_CONTINUOUS = ("n_reviews", "share_present_votes", "severity_rank", "num_tokens")

INTERPRETATION = (
    "Cartography finds training samples that a model finds hard to learn (low confidence and low "
    "variability over epochs), not wrong labels. A hard positive can be a correct but atypical smell, "
    "a snippet truncated by the token budget, or a smell that the code alone does not show; an easy "
    "negative can still be a missed smell. Whether a label is wrong can only be settled by an "
    "independent audit of the listed samples.")
SELECTION_NOTES = {
    "groups": "Positives and negatives are the training labels of the fine-tuning items (train_y of "
              "dynamics.npz); label_check compares them with the stored and the rule labels. Negatives "
              "are restricted to the level reviewed for the smell (eligible level).",
    "quantile_regions": "With regions.mode quantile, the thresholds are quantiles of each group, model "
                        "and variant, so every region holds a fixed share of its group whatever the "
                        "absolute difficulty; compare regions within a group, not counts across groups.",
    "non_unanimous": "Unanimity is defined only for (sample, smell) pairs with at least 2 reviews, so "
                     "non_unanimous tests hold for that subset only. In MLCQ this subset is selected: "
                     "phase 4 (26 Jul to 13 Sep 2019) cross-checked only samples already rated above none "
                     "that had one review or two differing reviews, and samples with a single none review "
                     "were never re-checked. A reviewer almost always rates both smells of the level of a "
                     "sample (feature envy and long method for a method, blob and data class for a class; "
                     "zenodo_reviewer_sample_pairs), so a negative of one smell can have several reviews "
                     "because the other smell of its level was rated present. selection_counts gives, per "
                     "model and smell, how many multi-review training negatives have a present vote for the "
                     "smell, and for the smell or the other smell of its level.",
    "any_present_vote": "For a negative, a present vote implies at least 2 reviews (with a single review, a "
                        "present vote gives a positive label under the rule; "
                        "selection_counts.negatives.single_review_with_present_vote counts exceptions). "
                        "Among negatives, any_present_vote therefore holds only for cross-checked samples and "
                        "is linked to single_review by construction.",
    "single_review": "A single review is mostly a sample never selected for the phase 4 cross-check, "
                     "which for a positive pair is rare (phase 4 targeted positive ratings).",
    "rule_inconsistent": "Stored label differs from the label of the documented rule (upper median of "
                         "the severities of the original reviews); all such pairs are among json_index 0 "
                         "to 174, whose stored vote counts are inflated.",
    "truncated": "Truncated means more model tokens than the token budget of the item (token_budget of "
                 "record.json); it depends on the model.",
    "item_average": "Per-sample statistics are means over the items whose training part holds the sample "
                    "(n_items); confidence_sd_items is their sample standard deviation over items.",
    "item_composition": "Items differ in how far training went by best_epoch, so a sample covered by items "
                        "that learned its group slowly gets a lower confidence whatever the sample. "
                        "item_baseline_confidence is the mean, over the items of the sample, of the mean "
                        "confidence of the sample's group (positives, or eligible negatives) in that item. "
                        "item_composition reports, per group, its Spearman correlation with the confidence "
                        "and the share of hard samples in each third of it. Folds are random, so this adds "
                        "noise to the tests rather than bias, but a hard sample with a low baseline can be "
                        "hard because of its items; it weighs most when items are missing or unbalanced.",
}


# ------------------------------------------------------------ helpers

def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def tmp_path(path: Path) -> Path:
    """Per-process temporary name, so that concurrent writers never share one."""
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def clean(obj: Any) -> Any:
    """JSON-safe copy: NumPy scalars to Python, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    tmp.write_text(json.dumps(clean(obj), indent=1))
    tmp.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})
    tmp.replace(path)


def fmt(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else f"{float(value):.6g}"
    if isinstance(value, (np.integer, np.bool_)):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    return value


def read_csv(path: Path, **kwargs: Any) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, **kwargs))


def label_matrix(entries: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for e in entries:
        y = e["y"]
        if isinstance(y, str):
            y = json.loads(y.lower())
        rows.append([int(bool(v)) for v in y])
    return np.array(rows, dtype=np.int64)


# ------------------------------------------------------------ sample metadata

class Meta:
    """Review metadata, rule labels and size features, indexed by json_index."""

    def __init__(self, ccfg: dict[str, Any]) -> None:
        inputs, lcfg = ccfg["inputs"], ccfg["labels"]
        rows = read_csv(resolve(inputs["sample_meta_csv"]))
        self.n = len(rows)
        if [int(r["json_index"]) for r in rows] != list(range(self.n)):
            raise SystemExit("sample metadata must hold json_index 0..n-1, in order")
        self.rows = rows
        self.shorts = list(lcfg["short_names"])
        self.names = list(lcfg["display_names"])
        if len(self.shorts) != len(LABEL_ORDER) or len(self.names) != len(LABEL_ORDER):
            raise SystemExit("labels.short_names and labels.display_names must follow LABEL_ORDER")
        self.order = list(lcfg["severity_order"])
        if self.order[:2] != ["none", "minor"]:
            raise SystemExit("labels.severity_order must start with none, minor")
        self.level = np.array([r["level"] for r in rows])
        self.smell_level = [str(lcfg["levels"][s]) for s in self.shorts]
        self.eligible = np.stack([self.level == lv for lv in self.smell_level], axis=1)
        self.y_stored = np.array([[int(r[f"y_{s}"]) for s in self.shorts] for r in rows], dtype=np.int64)
        rule_entries = json.loads(resolve(inputs["rule_labels_json"]).read_text())
        self.y_rule = label_matrix(rule_entries)
        if self.y_rule.shape != self.y_stored.shape:
            raise SystemExit(f"rule labels have shape {self.y_rule.shape}, expected {self.y_stored.shape}")
        self.n_reviews = np.array([[int(r[f"n_reviews_{s}"]) for s in self.shorts] for r in rows])
        self.votes = np.array([[[int(r[f"votes_{s}_{v}"]) for v in self.order] for s in self.shorts] for r in rows])
        self.unanimous = np.array([[float(r[f"unanimous_{s}"]) if r[f"unanimous_{s}"] != "" else np.nan
                                    for s in self.shorts] for r in rows])
        self.severity_raw = np.array([[r[f"severity_raw_{s}"] for s in self.shorts] for r in rows], dtype=object)
        self.severity_stored = np.array([[r[f"severity_{s}"] for s in self.shorts] for r in rows], dtype=object)
        sizes = {int(r["sample_idx"]): r for r in read_csv(resolve(inputs["size_features_csv"]))}
        if set(sizes) != set(range(self.n)):
            raise SystemExit("size features must hold sample_idx 0..n-1")
        self.size_columns = [c for c in next(iter(sizes.values())) if c != "sample_idx"]
        self.size = {c: np.array([float(sizes[i][c]) for i in range(self.n)]) for c in self.size_columns}
        self.zenodo: dict[int, dict[str, str]] = {}
        self.reviewer_sample_pairs: dict[str, int] = {}
        if inputs.get("zenodo_csv"):
            self.zenodo, self.reviewer_sample_pairs = zenodo_index(resolve(inputs["zenodo_csv"]), rows)

    def present_votes(self, k: int) -> np.ndarray:
        return self.votes[:, k, 1:].sum(axis=1)

    def above_minor_votes(self, k: int) -> np.ndarray:
        return self.votes[:, k, 2:].sum(axis=1)

    def severity_rank(self, k: int) -> np.ndarray:
        rank = {v: i for i, v in enumerate(self.order)}
        return np.array([rank.get(v, np.nan) for v in self.severity_raw[:, k]], dtype=np.float64)


def zenodo_index(path: Path, rows: list[dict[str, str]]) -> tuple[dict[int, dict[str, str]], dict[str, int]]:
    """MLCQ sample_id, code_name and link of each json_index, and (reviewer, sample) pair counts."""
    by_key: dict[tuple, dict[str, str]] = {}
    smells_rated: dict[tuple[str, str], set[str]] = {}
    for z in read_csv(path, delimiter=";"):
        key = (z["repository"], z["commit_hash"], z["path"], int(z["start_line"]), int(z["end_line"]))
        by_key.setdefault(key, {"mlcq_sample_id": z["sample_id"], "code_name": z["code_name"], "link": z["link"]})
        smells_rated.setdefault((z["reviewer_id"], z["sample_id"]), set()).add(z["smell"])
    out = {}
    for r in rows:
        key = (r["repo_url"], r["commit_hash"], r["file_path"], int(r["start_line"]), int(r["end_line"]))
        if key in by_key:
            out[int(r["json_index"])] = by_key[key]
    pairs = Counter(len(v) for v in smells_rated.values())
    return out, {f"smells_rated_{n}": c for n, c in sorted(pairs.items())}


# ------------------------------------------------------------ training dynamics

def find_items(model_dir: Path, strategy: str, require_record: bool) -> list[Path]:
    items = []
    if not model_dir.is_dir():
        return items
    for d in sorted(model_dir.iterdir()):
        m = ITEM_RE.match(d.name)
        if not d.is_dir() or not m or m["strategy"] != strategy:
            continue
        if not (d / "dynamics.npz").exists() or (require_record and not (d / "record.json").exists()):
            continue
        items.append(d)
    return items


def epoch_mask(epochs: np.ndarray, best_epoch: int, mode: str) -> np.ndarray:
    if mode == "all":
        return np.ones(len(epochs), dtype=bool)
    if mode == "to_best":
        return epochs <= best_epoch
    raise SystemExit(f"unknown epochs mode: {mode} (all or to_best)")


class Dynamics:
    """Cartography statistics of one model, averaged over its items."""

    def __init__(self, items: list[Path], n: int, variants: dict[str, Any], threshold: float,
                 eligible: np.ndarray) -> None:
        n_labels = len(LABEL_ORDER)
        self.variants = list(variants)
        sums = {v: {key: np.zeros((n, n_labels)) for key in ("conf", "conf2", "var", "corr", "base")}
                for v in variants}
        self.count = np.zeros(n, dtype=np.int64)
        self.gold = np.full((n, n_labels), -1, dtype=np.int64)
        self.num_tokens = np.full(n, -1, dtype=np.int64)
        self.items: list[dict[str, Any]] = []
        budgets = set()
        for d in items:
            with np.load(d / "dynamics.npz", allow_pickle=False) as z:
                if "train_prob" not in z.files:
                    raise SystemExit(f"{rel(d)}: no train_prob in dynamics.npz (fine-tuning without dynamics: true)")
                if [str(x) for x in z["label_order"]] != list(LABEL_ORDER):
                    raise SystemExit(f"{rel(d)}: label_order {list(z['label_order'])} differs from LABEL_ORDER")
                idx = z["train_index"].astype(np.int64)
                y = z["train_y"].astype(np.int64)
                prob = z["train_prob"].astype(np.float64)
                epochs = z["epochs"].astype(np.int64)
                best_epoch = int(z["best_epoch"])
                tokens = z["train_num_tokens"].astype(np.int64)
            if prob.shape != (len(epochs), len(idx), n_labels):
                raise SystemExit(f"{rel(d)}: train_prob shape {prob.shape} does not match epochs and train_index")
            seen = self.gold[idx] >= 0
            if (self.gold[idx][seen] != y[seen]).any():
                raise SystemExit(f"{rel(d)}: training labels differ from those of an earlier item")
            seen_tok = self.num_tokens[idx] >= 0
            if (self.num_tokens[idx][seen_tok] != tokens[seen_tok]).any():
                raise SystemExit(f"{rel(d)}: token counts differ from those of an earlier item")
            self.gold[idx] = y
            self.num_tokens[idx] = tokens
            p_gold = np.where(y[None] == 1, prob, 1.0 - prob)
            correct = (prob >= threshold).astype(np.int64) == y[None]
            for v, spec in variants.items():
                keep = epoch_mask(epochs, best_epoch, str(spec["epochs"]))
                if not keep.any():
                    raise SystemExit(f"{rel(d)}: no epoch kept for variant {v} (best_epoch={best_epoch})")
                c = p_gold[keep].mean(axis=0)
                sums[v]["conf"][idx] += c
                sums[v]["conf2"][idx] += c ** 2
                sums[v]["var"][idx] += p_gold[keep].std(axis=0)
                sums[v]["corr"][idx] += correct[keep].mean(axis=0)
                # item baseline: mean confidence of the sample's group in this item
                for k in range(n_labels):
                    for sel in (y[:, k] == 1, (y[:, k] == 0) & eligible[idx, k]):
                        if sel.any():
                            sums[v]["base"][idx[sel], k] += c[sel, k].mean()
            self.count[idx] += 1
            record = json.loads((d / "record.json").read_text()) if (d / "record.json").exists() else {}
            budget = record.get("token_budget")
            budgets.add(budget)
            self.items.append({"item": d.name, "epochs": len(epochs), "best_epoch": best_epoch,
                               "n_train": len(idx), "token_budget": budget,
                               "mcc_tuned": record.get("mcc_tuned")})
        # one token budget per model, otherwise truncation is undefined
        self.token_budget = budgets.pop() if len(budgets) == 1 else None
        self.stats: dict[str, dict[str, np.ndarray]] = {}
        cnt = self.count[:, None].astype(np.float64)
        in_group = (self.gold == 1) | ((self.gold == 0) & eligible)
        with np.errstate(invalid="ignore", divide="ignore"):
            for v in variants:
                s = sums[v]
                mean = s["conf"] / cnt
                sd = np.sqrt(np.maximum(s["conf2"] - cnt * mean ** 2, 0.0) / (cnt - 1))
                self.stats[v] = {
                    "confidence": mean,
                    "variability": s["var"] / cnt,
                    "correctness": s["corr"] / cnt,
                    "confidence_sd_items": np.where(cnt >= 2, sd, np.nan),
                    "item_baseline": np.where(in_group, s["base"] / cnt, np.nan),
                }


# ------------------------------------------------------------ attributes and tests

def binary_attribute(meta: Meta, dyn: Dynamics, k: int, name: str) -> tuple[np.ndarray, np.ndarray, str] | None:
    """(values, defined mask, subset) of a binary attribute for smell k, or None when undefined."""
    nrev = meta.n_reviews[:, k]
    reviewed = nrev > 0
    if name == "minor_only":
        return (meta.present_votes(k) > 0) & (meta.above_minor_votes(k) == 0), reviewed, "reviewed pairs"
    if name == "any_present_vote":
        return meta.present_votes(k) > 0, reviewed, "reviewed pairs"
    if name == "non_unanimous":
        return (meta.unanimous[:, k] == 0, nrev >= 2,
                "pairs with at least 2 reviews, a selected subset (selection_notes.non_unanimous)")
    if name == "rule_inconsistent":
        return meta.y_stored[:, k] != meta.y_rule[:, k], np.ones(meta.n, dtype=bool), "all pairs"
    if name == "single_review":
        return nrev == 1, reviewed, "reviewed pairs (selection_notes.single_review)"
    if name == "truncated":
        if dyn.token_budget is None:
            return None
        return dyn.num_tokens > int(dyn.token_budget), dyn.num_tokens >= 0, "training samples of the model"
    raise SystemExit(f"unknown binary attribute: {name} (one of {', '.join(BINARY)})")


def continuous_attribute(meta: Meta, dyn: Dynamics, k: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    nrev = meta.n_reviews[:, k]
    reviewed = nrev > 0
    if name == "n_reviews":
        return nrev.astype(np.float64), reviewed
    if name == "share_present_votes":
        return meta.present_votes(k) / np.maximum(nrev, 1), reviewed
    if name == "severity_rank":
        rank = meta.severity_rank(k)
        return rank, reviewed & np.isfinite(rank)
    if name == "num_tokens":
        return dyn.num_tokens.astype(np.float64), dyn.num_tokens >= 0
    if name in meta.size:
        return meta.size[name], np.isfinite(meta.size[name])
    raise SystemExit(f"unknown continuous attribute: {name} (one of {', '.join(REVIEW_CONTINUOUS)} "
                     f"or a column of size_features_csv)")


def fisher_test(in_region: np.ndarray, attr: np.ndarray, level: float) -> dict[str, Any]:
    """Region vs rest of the group x attribute present vs absent."""
    a, b = int((in_region & attr).sum()), int((in_region & ~attr).sum())
    c, d = int((~in_region & attr).sum()), int((~in_region & ~attr).sum())
    out: dict[str, Any] = {
        "table": {"region_with": a, "region_without": b, "rest_with": c, "rest_without": d},
        "share_region": a / (a + b) if a + b else None,
        "share_rest": c / (c + d) if c + d else None,
    }
    if min(a + b, c + d, a + c, b + d) == 0:
        out["skipped"] = "a row or a column of the 2x2 table is empty"
        return out
    table = [[a, b], [c, d]]
    _, p = fisher_exact(table, alternative="two-sided")
    res = odds_ratio(table, kind="conditional")
    ci = res.confidence_interval(confidence_level=level, alternative="two-sided")
    # JSON has no infinity: infinite odds ratios are written as "inf"
    out.update({
        "odds_ratio": finite_or_inf(res.statistic),
        "odds_ratio_ci": [finite_or_inf(ci.low), finite_or_inf(ci.high)],
        "odds_ratio_sample": finite_or_inf((a * d) / (b * c)) if b * c else "inf" if a * d else None,
        "p": float(p),
    })
    return out


def finite_or_inf(x: float) -> float | str:
    return float(x) if np.isfinite(x) else "inf"


def fmt_or(x: float | str | None) -> str:
    return "n.d." if x is None else x if isinstance(x, str) else f"{x:.3g}"


def fmt_short(x: float | str | None) -> str:
    if x is None or isinstance(x, str):
        return "n.d." if x is None else x
    return f"{x:.0f}" if x >= 100 else f"{x:.2g}"


def fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.2g}"


def mann_whitney(x_region: np.ndarray, x_rest: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "median_region": float(np.median(x_region)) if len(x_region) else None,
        "median_rest": float(np.median(x_rest)) if len(x_rest) else None,
        "mean_region": float(np.mean(x_region)) if len(x_region) else None,
        "mean_rest": float(np.mean(x_rest)) if len(x_rest) else None,
    }
    if not len(x_region) or not len(x_rest):
        out["skipped"] = "empty region or rest"
        return out
    if np.ptp(np.concatenate([x_region, x_rest])) == 0:
        out["skipped"] = "no variation"
        return out
    res = mannwhitneyu(x_region, x_rest, alternative="two-sided")
    out.update({
        "u": float(res.statistic),
        # P(region value > rest value) + P(equal) / 2
        "prob_superiority": float(res.statistic) / (len(x_region) * len(x_rest)),
        "p": float(res.pvalue),
    })
    return out


def holm(tests: list[dict[str, Any]]) -> None:
    """Holm step-down adjustment in place (p_holm) over the tests with a p-value."""
    done = [t for t in tests if t.get("p") is not None]
    done.sort(key=lambda t: t["p"])
    m, running = len(done), 0.0
    for rank, t in enumerate(done):
        running = max(running, min(1.0, (m - rank) * t["p"]))
        t["p_holm"] = running
        t["holm_family_size"] = m


def thresholds(conf: np.ndarray, var: np.ndarray, rcfg: dict[str, Any]) -> dict[str, float]:
    keys = {"confidence_low": conf, "confidence_high": conf, "variability_high": var}
    mode = str(rcfg["mode"])
    if mode == "quantile":
        return {k: float(np.quantile(x, float(rcfg[k]))) for k, x in keys.items()}
    if mode == "fixed":
        return {k: float(rcfg[k]) for k in keys}
    raise SystemExit(f"unknown regions.mode: {mode} (quantile or fixed)")


def assign_regions(conf: np.ndarray, var: np.ndarray, th: dict[str, float]) -> np.ndarray:
    region = np.full(len(conf), "other", dtype=object)
    ambiguous = var > th["variability_high"]
    hard = ~ambiguous & (conf <= th["confidence_low"])
    easy = ~ambiguous & ~hard & (conf >= th["confidence_high"])
    region[ambiguous], region[hard], region[easy] = "ambiguous", "hard", "easy"
    return region


def item_composition(conf: np.ndarray, base: np.ndarray, region: np.ndarray) -> dict[str, Any]:
    """How much the confidence of a group follows the items that cover each
    sample (selection_notes.item_composition). Descriptive, not tested."""
    ok = np.isfinite(base)
    if ok.sum() < 3 or np.ptp(base[ok]) == 0:
        return {"skipped": "fewer than 3 samples or no variation of the item baseline"}
    conf, base, region = conf[ok], base[ok], region[ok]
    third = np.digitize(base, np.quantile(base, [1 / 3, 2 / 3]), right=True)
    names = ("low", "middle", "high")
    return {
        "spearman_confidence_item_baseline": float(spearmanr(conf, base).statistic),
        "item_baseline_range": [float(base.min()), float(base.max())],
        "n_by_baseline_third": {names[t]: int((third == t).sum()) for t in range(3)},
        "hard_share_by_baseline_third": {names[t]: float((region[third == t] == "hard").mean())
                                         if (third == t).any() else None for t in range(3)},
    }


# ------------------------------------------------------------ analysis

def group_members(meta: Meta, dyn: Dynamics, k: int, group: str, min_items: int) -> np.ndarray:
    in_train = dyn.count >= min_items
    if group == "positives":
        return np.flatnonzero(in_train & (dyn.gold[:, k] == 1))
    return np.flatnonzero(in_train & (dyn.gold[:, k] == 0) & meta.eligible[:, k])


def analyse_group(meta: Meta, dyn: Dynamics, k: int, members: np.ndarray, st: dict[str, np.ndarray],
                  ccfg: dict[str, Any], group: str) -> tuple[dict[str, Any], np.ndarray]:
    tcfg = ccfg["tests"][group]
    conf, var, corr = (st[key][members, k] for key in ("confidence", "variability", "correctness"))
    out: dict[str, Any] = {"n": len(members)}
    if not len(members):
        out["skipped"] = "empty group"
        return out, np.array([], dtype=object)
    th = thresholds(conf, var, ccfg["regions"])
    region = assign_regions(conf, var, th)
    n_items = dyn.count[members]
    out.update({
        "thresholds": th,
        "n_items": {"min": int(n_items.min()), "median": float(np.median(n_items)), "max": int(n_items.max())},
        "means": {"confidence": float(conf.mean()), "variability": float(var.mean()),
                  "correctness": float(corr.mean())},
        "regions": {},
    })
    for reg in REGIONS:
        sel = region == reg
        out["regions"][reg] = {
            "n": int(sel.sum()),
            "confidence_mean": float(conf[sel].mean()) if sel.any() else None,
            "variability_mean": float(var[sel].mean()) if sel.any() else None,
            "correctness_mean": float(corr[sel].mean()) if sel.any() else None,
        }
    out["item_composition"] = item_composition(conf, st["item_baseline"][members, k], region)
    # share of every binary attribute in every region (descriptive)
    shares: dict[str, Any] = {}
    for name in BINARY:
        got = binary_attribute(meta, dyn, k, name)
        if got is None:
            continue
        values, defined, _ = got
        vals, ok = values[members], defined[members]
        shares[name] = {reg: {"n_defined": int((ok & (region == reg)).sum()),
                              "n_with": int((ok & (region == reg) & vals).sum())} for reg in REGIONS}
    out["attribute_counts_by_region"] = shares

    level = float(ccfg["tests"]["ci_level"])
    tests = []
    for reg in tcfg["regions"]:
        in_reg = region == reg
        for name in tcfg.get("binary", []):
            base = {"attribute": name, "kind": "fisher", "region": reg}
            got = binary_attribute(meta, dyn, k, name)
            if got is None:
                tests.append({**base, "skipped": "token budget unknown (record.json missing or budgets differ)"})
                continue
            values, defined, subset = got
            ok = defined[members]
            tests.append({**base, "subset": subset, **fisher_test(in_reg[ok], values[members][ok], level),
                          "n_region": int(in_reg[ok].sum()), "n_rest": int((~in_reg[ok]).sum()),
                          "n_undefined_left_out": int((~ok).sum())})
        for name in tcfg.get("continuous", []):
            values, defined = continuous_attribute(meta, dyn, k, name)
            ok = defined[members]
            vals = values[members]
            t = {"attribute": name, "kind": "mann_whitney", "region": reg,
                 **mann_whitney(vals[ok & in_reg], vals[ok & ~in_reg]),
                 "n_region": int((ok & in_reg).sum()), "n_rest": int((ok & ~in_reg).sum()),
                 "n_undefined_left_out": int((~ok).sum())}
            tests.append(t)
    out["tests"] = tests
    return out, region


def selection_counts(meta: Meta, dyn: Dynamics, k: int, min_items: int) -> dict[str, Any]:
    """Review counts of the training groups of smell k (selection_notes.non_unanimous
    and selection_notes.any_present_vote)."""
    siblings = [i for i, lv in enumerate(meta.smell_level) if lv == meta.smell_level[k] and i != k]
    nrev = meta.n_reviews[:, k]
    present = meta.present_votes(k) > 0
    present_any = present.copy()
    for i in siblings:
        present_any |= meta.present_votes(i) > 0
    out: dict[str, Any] = {"other_smells_of_level": [meta.shorts[i] for i in siblings]}
    for group in GROUPS:
        m = group_members(meta, dyn, k, group, min_items)
        multi, single = nrev[m] >= 2, nrev[m] == 1
        out[group] = {
            "n": len(m), "unreviewed": int((nrev[m] == 0).sum()), "single_review": int(single.sum()),
            "single_review_with_present_vote": int((single & present[m]).sum()),
            "multi_review": int(multi.sum()),
            "multi_review_with_present_vote": int((multi & present[m]).sum()),
            "multi_review_with_present_vote_for_smell_or_other_smell_of_level": int((multi & present_any[m]).sum()),
        }
    return out


def label_check(meta: Meta, dyn: Dynamics) -> dict[str, Any]:
    known = dyn.count > 0
    out = {"samples_in_training": int(known.sum())}
    for name, ref in (("stored", meta.y_stored), ("rule", meta.y_rule)):
        diff = (dyn.gold != ref) & known[:, None]
        out[f"train_y_differs_from_{name}"] = {s: int(diff[:, k].sum()) for k, s in enumerate(meta.shorts)}
    return out


def sample_rows(model: str, variant: str, meta: Meta, dyn: Dynamics, k: int, group: str,
                members: np.ndarray, region: np.ndarray) -> list[dict[str, Any]]:
    st = dyn.stats[variant]
    minor_only = (meta.present_votes(k) > 0) & (meta.above_minor_votes(k) == 0)
    budget = dyn.token_budget
    rows = []
    for j, reg in zip(members, region):
        rows.append({
            "model": model, "variant": variant, "smell": meta.shorts[k], "json_index": int(j),
            "group": group, "gold": int(dyn.gold[j, k]), "n_items": int(dyn.count[j]),
            "confidence": st["confidence"][j, k], "variability": st["variability"][j, k],
            "correctness": st["correctness"][j, k], "confidence_sd_items": st["confidence_sd_items"][j, k],
            "item_baseline_confidence": st["item_baseline"][j, k], "region": reg, "level": meta.level[j], "severity_raw": meta.severity_raw[j, k],
            "severity_stored": meta.severity_stored[j, k], "y_stored": int(meta.y_stored[j, k]),
            "y_rule": int(meta.y_rule[j, k]), "n_reviews": int(meta.n_reviews[j, k]),
            "present_votes": int(meta.present_votes(k)[j]), "minor_only": int(minor_only[j]),
            "unanimous": "" if np.isnan(meta.unanimous[j, k]) else int(meta.unanimous[j, k]),
            "num_tokens": int(dyn.num_tokens[j]),
            "truncated": "" if budget is None else int(dyn.num_tokens[j] > int(budget)),
        })
    return rows


SAMPLE_COLUMNS = ["model", "variant", "smell", "json_index", "group", "gold", "n_items", "confidence",
                  "variability", "correctness", "confidence_sd_items", "item_baseline_confidence", "region",
                  "level", "severity_raw",
                  "severity_stored", "y_stored", "y_rule", "n_reviews", "present_votes", "minor_only",
                  "unanimous", "num_tokens", "truncated"]


# ------------------------------------------------------------ figures

class Figures:
    def __init__(self, directory: Path, prefix: str, dpi: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt, self.dir, self.prefix, self.dpi = plt, directory, prefix, dpi
        self.written: list[str] = []
        directory.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
            "legend.frameon": False, "figure.dpi": 100, "savefig.bbox": "tight",
        })

    def save(self, fig: Any, name: str) -> None:
        for ext in ("pdf", "png"):
            path = self.dir / f"{self.prefix}{name}.{ext}"
            tmp = tmp_path(path)
            fig.savefig(tmp, format=ext, dpi=self.dpi)
            tmp.replace(path)
            self.written.append(rel(path))
        self.plt.close(fig)


SEVERITY_STYLE = {
    "minor": ("positif, sévérité mineure", OKABE_ITO["sky_blue"], "o"),
    "major": ("positif, sévérité majeure", OKABE_ITO["orange"], "o"),
    "critical": ("positif, sévérité critique", OKABE_ITO["vermillion"], "o"),
    "inconsistent": ("positif non conforme à la règle", OKABE_ITO["purple"], "X"),
}


def plot_map(figs: Figures, model: str, variant: str, vlabel: str, meta: Meta, dyn: Dynamics,
             groups: dict[int, dict[str, tuple[np.ndarray, dict[str, Any]]]]) -> None:
    plt = figs.plt
    st = dyn.stats[variant]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4), sharey=True)
    var_max = 0.0
    for k, by_group in groups.items():
        for members, _ in by_group.values():
            if len(members):
                var_max = max(var_max, float(np.nanmax(st["variability"][members, k])))
    x_hi = max(0.05, var_max * 1.08)
    for k, ax in enumerate(axes.flat):
        neg, _ = groups[k]["negatives"]
        pos, pos_out = groups[k]["positives"]
        ax.scatter(st["variability"][neg, k], st["confidence"][neg, k], s=4, c=OKABE_ITO["black"], alpha=0.12,
                   linewidths=0, rasterized=True, label="négatif de niveau éligible")
        inconsistent = meta.y_stored[pos, k] != meta.y_rule[pos, k]
        category = np.where(inconsistent, "inconsistent", meta.severity_raw[pos, k])
        for cat, (label, color, marker) in SEVERITY_STYLE.items():
            sel = pos[category == cat]
            if len(sel):
                ax.scatter(st["variability"][sel, k], st["confidence"][sel, k], s=16 if marker == "o" else 30,
                           c=color, marker=marker, alpha=0.85, linewidths=0.3, edgecolors="white",
                           rasterized=True, label=label)
        th = pos_out.get("thresholds")
        if th:
            for yv in (th["confidence_low"], th["confidence_high"]):
                ax.axhline(yv, color=OKABE_ITO["black"], lw=0.7, ls="--", alpha=0.5)
            ax.axvline(th["variability_high"], color=OKABE_ITO["black"], lw=0.7, ls="--", alpha=0.5)
            ax.text(0.01, 0.99, "facile", transform=ax.transAxes, va="top", fontsize=7.5, alpha=0.8)
            ax.text(0.01, 0.01, "difficile", transform=ax.transAxes, va="bottom", fontsize=7.5, alpha=0.8)
            ax.text(0.99, 0.99, "ambigu", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, alpha=0.8)
        ax.set_title(f"{meta.names[k]} ({len(pos)} positifs, {len(neg)} négatifs éligibles)")
        ax.set_xlim(0, x_hi)
        ax.set_ylim(-0.02, 1.02)
        if k >= 2:
            ax.set_xlabel("Variabilité (écart type sur les époques)")
        if k % 2 == 0:
            ax.set_ylabel("Confiance (probabilité moyenne\nde l'étiquette d'entraînement)")
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = ["négatif de niveau éligible"] + [s[0] for s in SEVERITY_STYLE.values()]
    labels = [lab for lab in order if lab in handles]
    fig.legend([handles[lab] for lab in labels], labels, ncols=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), markerscale=1.4)
    fig.suptitle(f"Cartographie des données d'entraînement : {model}, {vlabel}\n"
                 "(seuils des régions calculés sur les positifs, lignes pointillées)", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    figs.save(fig, f"map_{model}_{variant}")


def plot_enrichment(figs: Figures, model: str, variant: str, vlabel: str, meta: Meta,
                    results: dict[str, dict[str, Any]], ccfg: dict[str, Any]) -> None:
    plt = figs.plt
    labels = dict(ccfg.get("attribute_labels", {}))
    tcfg = ccfg["tests"]
    fig, axes = plt.subplots(2, len(meta.shorts), figsize=(3.2 * len(meta.shorts), 6.6), sharey=True)
    group_names = {"positives": "positifs", "negatives": "négatifs éligibles"}
    for row, group in enumerate(GROUPS):
        attrs = list(tcfg[group].get("binary", []))
        region = next(iter(tcfg[group]["regions"]))
        for k, s in enumerate(meta.shorts):
            ax = axes[row, k]
            res = results[s][group]
            tests = {t["attribute"]: t for t in res.get("tests", [])
                     if t["kind"] == "fisher" and t["region"] == region}
            x = np.arange(len(attrs))
            share_reg = [100 * (tests.get(a, {}).get("share_region") or 0.0) for a in attrs]
            share_rest = [100 * (tests.get(a, {}).get("share_rest") or 0.0) for a in attrs]
            n_reg = res.get("regions", {}).get(region, {}).get("n", 0)
            ax.bar(x - 0.2, share_reg, width=0.38, color=OKABE_ITO["vermillion"],
                   label="région « difficile »" if region == "hard" else f"région {region}")
            ax.bar(x + 0.2, share_rest, width=0.38, color=OKABE_ITO["blue"], label="reste du groupe")
            for xi, a in zip(x, attrs):
                t = tests.get(a, {})
                if t.get("p") is None:
                    text = "n.d."
                else:
                    text = f"RC {fmt_short(t.get('odds_ratio'))}\n{fmt_p(t.get('p_holm', t['p']))}"
                top = max(share_reg[xi], share_rest[xi])
                ax.text(xi, top + 2, text, ha="center", va="bottom", fontsize=6)
            ax.set_xticks(x, [labels.get(a, a) for a in attrs], rotation=35, ha="right", fontsize=7.5)
            ax.set_title(f"{meta.names[k]}, {group_names[group]}\n({n_reg} difficiles sur {res.get('n', 0)})")
            ax.grid(axis="x", visible=False)
            if k == 0:
                ax.set_ylabel("Part des échantillons (%)")
    for ax in axes.flat:
        ax.set_ylim(0, 118)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, ncols=2, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Exemples difficiles et qualité des annotations : {model}, {vlabel}\n"
                 "RC : rapport de cotes conditionnel (test exact de Fisher), p ajusté par Holm", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    figs.save(fig, f"enrichment_{model}_{variant}")


# ------------------------------------------------------------ exports

def export_cases(out_dir: Path, meta: Meta, cases: dict[int, dict[int, dict[str, Any]]],
                 n_models: int, models: list[str]) -> list[str]:
    written = []
    columns = ["group", "json_index", "mlcq_sample_id", "repo", "file_path", "start_line", "end_line", "lines",
               "level", "code_name", "link", "severity_raw", "severity_stored", "y_stored", "y_rule",
               "n_reviews", "votes_none", "votes_minor", "votes_major", "votes_critical", "unanimous",
               "minor_only", "rule_inconsistent", "n_models_analysed", "n_models_flagged", "models_flagged",
               "confidence_mean", "variability_mean", "correctness_mean", "item_baseline_confidence_mean",
               "n_items_min"]
    for k, s in enumerate(meta.shorts):
        rows = []
        for j, entry in cases.get(k, {}).items():
            r = meta.rows[j]
            z = meta.zenodo.get(j, {})
            flagged = entry["models"]
            rows.append({
                "group": entry["group"], "json_index": j, "mlcq_sample_id": z.get("mlcq_sample_id", ""),
                "repo": r["repo"], "file_path": r["file_path"], "start_line": r["start_line"],
                "end_line": r["end_line"], "lines": f"{r['start_line']}-{r['end_line']}", "level": r["level"],
                "code_name": z.get("code_name", ""), "link": z.get("link", ""),
                "severity_raw": meta.severity_raw[j, k], "severity_stored": meta.severity_stored[j, k],
                "y_stored": int(meta.y_stored[j, k]), "y_rule": int(meta.y_rule[j, k]),
                "n_reviews": int(meta.n_reviews[j, k]),
                **{f"votes_{v}": int(meta.votes[j, k, i]) for i, v in enumerate(meta.order)},
                "unanimous": "" if np.isnan(meta.unanimous[j, k]) else int(meta.unanimous[j, k]),
                "minor_only": int(meta.present_votes(k)[j] > 0 and meta.above_minor_votes(k)[j] == 0),
                "rule_inconsistent": int(meta.y_stored[j, k] != meta.y_rule[j, k]),
                "n_models_analysed": n_models, "n_models_flagged": len(flagged),
                "models_flagged": "|".join(m for m in models if m in flagged),
                "confidence_mean": float(np.mean([v["confidence"] for v in flagged.values()])),
                "variability_mean": float(np.mean([v["variability"] for v in flagged.values()])),
                "correctness_mean": float(np.mean([v["correctness"] for v in flagged.values()])),
                "item_baseline_confidence_mean": float(np.mean([v["item_baseline"] for v in flagged.values()])),
                "n_items_min": int(min(v["n_items"] for v in flagged.values())),
            })
        # hard positives first (least confident first), then easy negatives (most confident first)
        rows.sort(key=lambda r: (r["group"] != "hard_positive", -r["n_models_flagged"],
                                 r["confidence_mean"] if r["group"] == "hard_positive" else -r["confidence_mean"]))
        path = out_dir / f"hard_cases_{s}.csv"
        atomic_csv(path, rows, columns)
        written.append(rel(path))
    return written


def cross_model(per_model: dict[str, dict[str, Any]], meta: Meta, variants: list[str]) -> dict[str, Any]:
    """Spearman correlation of the positives' confidence and Jaccard overlap of
    the hard positives between every pair of models."""
    models = list(per_model)
    out: dict[str, Any] = {}
    for v in variants:
        out[v] = {}
        for k, s in enumerate(meta.shorts):
            pairs = []
            for a in range(len(models)):
                for b in range(a + 1, len(models)):
                    ma, mb = per_model[models[a]], per_model[models[b]]
                    pa, pb = ma["members"][v][k]["positives"], mb["members"][v][k]["positives"]
                    common = np.intersect1d(pa, pb)
                    entry: dict[str, Any] = {"models": [models[a], models[b]], "n_common_positives": len(common)}
                    if len(common) >= 3:
                        rho = spearmanr(ma["dyn"].stats[v]["confidence"][common, k],
                                        mb["dyn"].stats[v]["confidence"][common, k])
                        entry["spearman_confidence"] = float(rho.statistic)
                    ha, hb = set(ma["hard"][v][k]), set(mb["hard"][v][k])
                    entry["hard_positives"] = [len(ha), len(hb)]
                    entry["jaccard_hard_positives"] = len(ha & hb) / len(ha | hb) if ha | hb else None
                    pairs.append(entry)
            out[v][s] = pairs
    return out


# ------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    ccfg = cfg["cartography"]
    variants = dict(ccfg["variants"])
    out_dir = resolve(ccfg["out_dir"])
    figs = Figures(resolve(ccfg["figures_dir"]), str(ccfg["figure_prefix"]), int(ccfg.get("figure_dpi", 200)))
    min_items = int(ccfg["min_items_per_sample"])
    export_variant = str(ccfg["export_variant"])
    if export_variant not in variants:
        raise SystemExit(f"export_variant {export_variant} is not a configured variant")

    meta = Meta(ccfg)
    print(f"[cartography] {meta.n} samples, {len(meta.zenodo)} linked to the Zenodo release", flush=True)

    root = resolve(ccfg["runs_root"])
    per_model: dict[str, dict[str, Any]] = {}
    left_out: dict[str, str] = {}
    for model in ccfg["models"]:
        items = find_items(root / model, str(ccfg["strategy"]), bool(ccfg["require_record"]))
        if len(items) < int(ccfg["min_items_per_model"]):
            left_out[model] = f"{len(items)} complete items (min_items_per_model {ccfg['min_items_per_model']})"
            print(f"[cartography] {model}: {left_out[model]}, left out", flush=True)
            continue
        dyn = Dynamics(items, meta.n, variants, float(ccfg["correct_threshold"]), meta.eligible)
        print(f"[cartography] {model}: {len(items)} items, {int((dyn.count > 0).sum())} training samples, "
              f"best epochs {dict(sorted(Counter(i['best_epoch'] for i in dyn.items).items()))}", flush=True)
        for k, s in enumerate(meta.shorts):
            neg = selection_counts(meta, dyn, k, int(ccfg["min_items_per_sample"]))["negatives"]
            print(f"[cartography] {model} {s} eligible negatives: {neg['multi_review']} of {neg['n']} with 2 or "
                  f"more reviews, {neg['multi_review_with_present_vote']} of them with a present vote for the "
                  f"smell, {neg['multi_review_with_present_vote_for_smell_or_other_smell_of_level']} for the "
                  f"smell or the other smell of its level", flush=True)
        per_model[model] = {"dyn": dyn, "items": items}

    summary: dict[str, Any] = {
        "config": rel(resolve(args.config)),
        "method": "Swayamdipta et al., EMNLP 2020 (dataset cartography)",
        "definitions": {
            "confidence": "mean over the kept epochs of the probability of the training label",
            "variability": "population standard deviation over the kept epochs of that probability",
            "correctness": f"share of the kept epochs where (p >= {ccfg['correct_threshold']}) equals the training label",
            "variants": variants,
            "regions": {**ccfg["regions"], "assignment": "ambiguous: variability > variability_high; hard: "
                        "confidence <= confidence_low and not ambiguous; easy: confidence >= confidence_high "
                        "and not ambiguous; other: the rest"},
            "groups": {"positives": "training positives of the smell",
                       "negatives": "training negatives of the smell at its eligible level"},
            "min_items_per_sample": min_items,
            "tests": ccfg["tests"],
            "multiple_testing": "Holm over all tests with a p-value of one model and variant",
        },
        "interpretation_note": INTERPRETATION,
        "selection_notes": SELECTION_NOTES,
        "zenodo_reviewer_sample_pairs": meta.reviewer_sample_pairs,
        "runs_root": rel(root),
        "models_left_out": left_out,
        "models": {},
    }
    if not per_model:
        atomic_json(out_dir / "summary.json", summary)
        print(f"[cartography] no model with enough items; wrote {rel(out_dir / 'summary.json')}", flush=True)
        return

    cases: dict[int, dict[int, dict[str, Any]]] = {}
    for model, entry in per_model.items():
        dyn = entry["dyn"]
        msum: dict[str, Any] = {
            "run_dir": rel(root / model),
            "n_items": len(entry["items"]),
            "items": dyn.items,
            "best_epoch_counts": dict(sorted(Counter(i["best_epoch"] for i in dyn.items).items())),
            "epochs_counts": dict(sorted(Counter(i["epochs"] for i in dyn.items).items())),
            "token_budget": dyn.token_budget,
            "n_items_per_sample": dict(sorted(Counter(int(c) for c in dyn.count if c > 0).items())),
            "label_check": label_check(meta, dyn),
            "selection_counts": {s: selection_counts(meta, dyn, k, min_items) for k, s in enumerate(meta.shorts)},
            "variants": {},
        }
        entry["members"], entry["hard"] = {}, {}
        rows: list[dict[str, Any]] = []
        for v, vspec in variants.items():
            st = dyn.stats[v]
            results: dict[str, Any] = {}
            groups: dict[int, dict[str, tuple[np.ndarray, dict[str, Any]]]] = {}
            entry["members"][v], entry["hard"][v] = {}, {}
            all_tests: list[dict[str, Any]] = []
            for k, s in enumerate(meta.shorts):
                results[s], groups[k], entry["members"][v][k] = {}, {}, {}
                for group in GROUPS:
                    members = group_members(meta, dyn, k, group, min_items)
                    res, region = analyse_group(meta, dyn, k, members, st, ccfg, group)
                    results[s][group] = res
                    groups[k][group] = (members, res)
                    entry["members"][v][k][group] = members
                    all_tests.extend(res.get("tests", []))
                    rows.extend(sample_rows(model, v, meta, dyn, k, group, members, region))
                    if group == "positives":
                        entry["hard"][v][k] = members[region == "hard"].tolist()
                    if v == export_variant:
                        wanted = {"positives": "hard_positive", "negatives": "easy_negative"}[group]
                        region_name = "hard" if group == "positives" else "easy"
                        if wanted in ccfg["export_groups"]:
                            for j in members[region == region_name]:
                                case = cases.setdefault(k, {}).setdefault(int(j), {"group": wanted, "models": {}})
                                case["models"][model] = {key: float(st[key][j, k]) for key in
                                                         ("confidence", "variability", "correctness",
                                                          "item_baseline")}
                                case["models"][model]["n_items"] = int(dyn.count[j])
                results[s]["positives_at_other_level"] = int(
                    ((dyn.gold[:, k] == 1) & (dyn.count >= min_items) & ~meta.eligible[:, k]).sum())
            holm(all_tests)
            msum["variants"][v] = results
            for k, s in enumerate(meta.shorts):
                for group in GROUPS:
                    res = results[s][group]
                    if "regions" not in res:
                        continue
                    reg = {r: res["regions"][r]["n"] for r in REGIONS}
                    print(f"[cartography] {model} {v} {s} {group}: n={res['n']} regions={reg}", flush=True)
                    ic = res["item_composition"]
                    if "skipped" not in ic:
                        shares = {t: round(x, 3) for t, x in ic["hard_share_by_baseline_third"].items()
                                  if x is not None}
                        print(f"    item composition: Spearman(confidence, item baseline)="
                              f"{ic['spearman_confidence_item_baseline']:.3f}, hard share by baseline third "
                              f"{shares}", flush=True)
                    for t in res["tests"]:
                        if t["kind"] == "fisher" and t.get("p") is not None:
                            ci = t["odds_ratio_ci"]
                            print(f"    {t['attribute']}: {t['region']} {t['share_region']:.3f} vs rest "
                                  f"{t['share_rest']:.3f}, OR={fmt_or(t['odds_ratio'])} "
                                  f"[{fmt_or(ci[0])}, {fmt_or(ci[1])}] p={t['p']:.3g} p_holm={t['p_holm']:.3g}",
                                  flush=True)
            plot_map(figs, model, v, str(vspec.get("label", v)), meta, dyn, groups)
            plot_enrichment(figs, model, v, str(vspec.get("label", v)), meta, results, ccfg)
        samples_path = out_dir / model / "samples.csv"
        atomic_csv(samples_path, rows, SAMPLE_COLUMNS)
        msum["samples_csv"] = rel(samples_path)
        summary["models"][model] = msum

    summary["audit_lists"] = {
        "variant": export_variant, "groups": list(ccfg["export_groups"]),
        "note": "Rows pooled over the analysed models (n_models_flagged of n_models_analysed). "
                + INTERPRETATION,
        "files": export_cases(out_dir, meta, cases, len(per_model), list(per_model)),
        "counts": {s: dict(Counter(c["group"] for c in cases.get(k, {}).values()))
                   for k, s in enumerate(meta.shorts)},
    }
    if len(per_model) >= 2:
        summary["cross_model"] = cross_model(per_model, meta, list(variants))
    summary["figures"] = figs.written
    atomic_json(out_dir / "summary.json", summary)
    print(f"[cartography] wrote {rel(out_dir / 'summary.json')}, {len(summary['audit_lists']['files'])} audit "
          f"lists and {len(figs.written)} figure files", flush=True)
    print(f"[cartography] {INTERPRETATION}", flush=True)


if __name__ == "__main__":
    main()
