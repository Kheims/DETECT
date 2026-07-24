"""Plot figures and tables for the context study and the rewrite tests.

Reads the outputs of eval_context_judge.py, llm_context_judge.py and
analyze_metamorphic.py. Writes context_conditions.png, agent_decisions.png,
comment_flips.png and agent_decisions.csv.

Usage: python scripts/llm/context_rewrite_figures.py [--out artifacts/analysis/round3]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import matthews_corrcoef, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]
NAMES = {"is_feature_envy": "Feature Envy", "is_long_method": "Long Method", "is_blob": "Blob",
         "is_data_class": "Data Class", "is_complex_method": "Complex Method",
         "is_long_parameter_list": "Long Parameter List"}
CONDITIONS = [
    ("C1", "+ skeleton of the enclosing class"),
    ("C2", "+ static facts (member accesses, counts)"),
    ("C3", "+ skeleton, facts and referenced types"),
    ("PAD", "+ filler text, as long as the line above"),
    ("SHUF", "+ skeleton of a class from another repository"),
    ("FACTS_ONLY", "static facts only, no code"),
    ("C4", "agent: calls repository tools, then answers"),
    ("REPLAY", "agent's tool outputs, without the agent's text"),
]
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#a0aec0"


def ci(text: str) -> tuple[float, float]:
    lo, hi = json.loads(text)
    return float(lo), float(hi)


def context_conditions(ranking: Path, out: Path) -> None:
    rows = [r for r in csv.DictReader(ranking.open()) if r["scope"] == "subset"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    for ax, smell in zip(axes, LABELS):
        by = {r["condition"]: r for r in rows if r["smell"] == smell}
        base = by["C0"]
        for i, (cond, _) in enumerate(CONDITIONS):
            r = by[cond]
            d = float(r["roc_auc_diff"])
            lo, hi = ci(r["roc_auc_diff_ci"])
            color = ORANGE if cond in ("C4", "REPLAY") else BLUE
            ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=color, ms=5, capsize=0, lw=2)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{NAMES[smell]}\nsnippet only: ROC-AUC {float(base['roc_auc']):.3f}\n"
                     f"n = {base['n']}, positives = {base['positives']}", fontsize=10)
        ax.grid(axis="x", color="#e2e8f0")
        ax.set_xlabel("ROC-AUC change against snippet only")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(CONDITIONS)), [c[1] for c in CONDITIONS], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Qwen3.6-27B on MLCQ: what does context in the prompt change? "
                 "(95% intervals, repositories resampled; same samples in every condition)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "context_conditions.png", dpi=160)
    fig.savefig(out / "context_conditions.pdf")
    plt.close(fig)


def load_scores(base: Path, cond: str) -> dict[tuple[int, str], dict]:
    rows = {}
    for f in glob.glob(str(base / cond / "scores*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            rows[(r["json_index"], r["smell"])] = r
    return rows


def agent_decisions(scores_dir: Path, samples: Path, out: Path, n_boot: int = 2000) -> None:
    data = json.loads(samples.read_text())
    conds = {c: load_scores(scores_dir, c) for c in ("C0", "REPLAY", "C4")}
    rng = np.random.default_rng(0)
    table = []
    for smell in LABELS:
        keys = sorted(k for k in conds["C4"] if k[1] == smell)
        y = np.array([int(data[i]["y"][LABELS.index(smell)]) for i, _ in keys])
        s = {c: np.array([conds[c][k]["score"] for k in keys]) for c in conds}
        tools = np.array([conds["C4"][k]["n_tool_calls"] for k in keys])
        boots = rng.integers(0, len(y), (n_boot, len(y)))
        for c in conds:
            pred = (s[c] > 0.5).astype(int)
            d_mcc = [matthews_corrcoef(y[b], pred[b]) - matthews_corrcoef(y[b], (s["C0"][b] > 0.5).astype(int))
                     for b in boots] if c != "C0" else [0.0]
            table.append({
                "smell": smell, "condition": c, "n": len(y), "positives": int(y.sum()),
                "roc_auc": roc_auc_score(y, s[c]), "mcc_at_0.5": matthews_corrcoef(y, pred),
                "mcc_diff_vs_C0": matthews_corrcoef(y, pred) - matthews_corrcoef(y, (s["C0"] > 0.5).astype(int)),
                "mcc_diff_lo": float(np.percentile(d_mcc, 2.5)), "mcc_diff_hi": float(np.percentile(d_mcc, 97.5)),
                "share_near_0_or_1": float(np.mean((s[c] < 0.01) | (s[c] > 0.99))),
                "same_decision_as_C0": float(np.mean(pred == (s["C0"] > 0.5))),
                "share_without_tool_call": float(np.mean(tools == 0)) if c == "C4" else float("nan"),
            })
    with (out / "agent_decisions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    names = {"C0": "snippet only", "REPLAY": "tool outputs, no agent text", "C4": "agent with tools"}
    colors = {"C0": BLUE, "REPLAY": GREY, "C4": ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(LABELS))
    for j, (key, title) in enumerate([("roc_auc", "Ranking: ROC-AUC"),
                                      ("mcc_at_0.5", "Decisions: MCC at the fixed threshold 0.5"),
                                      ("share_near_0_or_1", "Share of scores below 0.01 or above 0.99")]):
        ax = axes[j]
        for k, c in enumerate(("C0", "REPLAY", "C4")):
            vals = [next(r[key] for r in table if r["smell"] == s and r["condition"] == c) for s in LABELS]
            bars = ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=colors[c], label=names[c])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x, [NAMES[s] for s in LABELS])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.08)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    no_tool = [next(r["share_without_tool_call"] for r in table if r["smell"] == s and r["condition"] == "C4")
               for s in LABELS]
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Qwen3.6-27B agent on MLCQ: after the agent writes its reasoning, the yes/no probability is "
                 "almost 0 or 1, so the ranking is lost.\nThe decisions stay close to the snippet-only prompt. "
                 "Conversations without a tool call: " + ", ".join(
                     f"{NAMES[s]} {v:.0%}" for s, v in zip(LABELS, no_tool)), fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "agent_decisions.png", dpi=160)
    fig.savefig(out / "agent_decisions.pdf")
    plt.close(fig)


def comment_flips(summary: Path, out: Path) -> None:
    rows = list(csv.DictReader(summary.open()))
    panels = [
        ("MLCQ", [("qwen27b_zeroshot", "Qwen3.6-27B"), ("mm_unixcoder", "UniXcoder"),
                  ("mm_modernbert", "ModernBERT")], LABELS),
        ("DACOS", [("dacos_qwen27b_zeroshot", "Qwen3.6-27B"), ("dacos_unixcoder", "UniXcoder"),
                   ("dacos_size_logreg", "size model")],
         ["is_complex_method", "is_long_parameter_list"]),
    ]
    transforms = [("comment_claims_smell", "comment says the smell is present", ORANGE),
                  ("comment_denies_smell", "comment says the code is clean", BLUE),
                  ("rename_locals", "local variables renamed (noise reference)", GREY)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [4, 2]}, sharey=True)
    for ax, (dataset, fams, smells) in zip(axes, panels):
        groups = [(f, fl, s) for s in smells for f, fl in fams]
        x = np.arange(len(groups))
        for k, (t, tl, color) in enumerate(transforms):
            vals, lo, hi = [], [], []
            for f, _, s in groups:
                r = next((r for r in rows if r["family"] == f and r["smell"] == s.removeprefix("is_") and r["transform"] == t
                          and r["strategy"] == "kfold"), None)
                v = float(r["flip_rate"]) if r else np.nan
                vals.append(v)
                lo.append(v - float(r["flip_rate_lo"]) if r else 0)
                hi.append(float(r["flip_rate_hi"]) - v if r else 0)
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=color, label=tl, yerr=[lo, hi],
                   error_kw={"lw": 0.8})
            for xi, v in zip(x, vals):
                if np.isnan(v):
                    ax.text(xi + (k - 1) * 0.27, 0.004, "not run", rotation=90, ha="center", fontsize=6,
                            color="#4a5568")
        ax.set_xticks(x, [fl for _, fl, _ in groups], fontsize=8, rotation=35, ha="right")
        for i, s in enumerate(smells):
            ax.text(i * len(fams) + (len(fams) - 1) / 2, 0.31, NAMES[s], ha="center", fontsize=10, weight="bold")
            if i:
                ax.axvline(i * len(fams) - 0.5, color="#cbd5e0", lw=0.8)
        ax.set_title(dataset, fontsize=11)
        ax.set_ylim(0, 0.33)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("share of decisions that change\n(threshold tuned on the original code)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=8, frameon=False)
    fig.suptitle("One added comment line, same code: the LLM judge follows a comment that claims the smell, "
                 "the fine-tuned models do not.\nQwen3.6-27B: zero-shot LLM judge. UniXcoder, ModernBERT: fine-tuned encoders. Size model: logistic regression on size features. 95% intervals.", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "comment_flips.png", dpi=160)
    fig.savefig(out / "comment_flips.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/analysis/round3")
    ap.add_argument("--context_report", default="artifacts/context_study/report/qwen36_27b_fp8")
    ap.add_argument("--context_scores", default="artifacts/context_study/scores/qwen36_27b_fp8")
    ap.add_argument("--samples", default="data/MLCQCodeSmellSamples.normalized.json")
    ap.add_argument("--metamorphic_summary", default="artifacts/analysis/metamorphic/summary.csv")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    context_conditions(ROOT / args.context_report / "ranking.csv", out)
    agent_decisions(ROOT / args.context_scores, ROOT / args.samples, out)
    comment_flips(ROOT / args.metamorphic_summary, out)
    print(f"[round3] wrote figures and agent_decisions.csv in {out}")


if __name__ == "__main__":
    main()
