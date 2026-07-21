"""Analyse the blind re-annotation of MLCQ samples.

Inputs: the case key from select_audit_cases.py and the annotator judgments.
Outputs per smell and group: panel positive share, agreement with MLCQ and
Krippendorff alpha.

Usage:
    python scripts/labels/analyze_audit.py --config config/experiments/label_audit.yml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.labels.analyze_mlcq import krippendorff_alpha  # noqa: E402
from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402

SEVERITY = ["none", "minor", "major", "critical"]
GROUPS = ["easy_pos", "hard_pos", "hard_neg", "easy_neg"]


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))["audit_analysis"]

    key = {k["case_id"]: k for k in json.loads(resolve(cfg["key"]).read_text())}
    ratings: dict[str, dict[str, int]] = defaultdict(dict)
    for block in json.loads(resolve(cfg["judgments"]).read_text()):
        for j in block["judgments"] or []:
            if j["case_id"] in key:
                ratings[j["case_id"]][block["annotator"]] = SEVERITY.index(j["severity"])
    annotators = sorted({a for r in ratings.values() for a in r})
    with resolve(cfg["sample_meta"]).open() as fh:
        meta = {int(r["json_index"]): r for r in csv.DictReader(fh)}
    short = {"feature_envy": "fe", "long_method": "lm", "blob": "blob", "data_class": "dc"}
    judge = {}
    if cfg.get("llm_scores"):
        for line in resolve(cfg["llm_scores"]).read_text().splitlines():
            r = json.loads(line)
            judge[(int(r["json_index"]), str(r["smell"]).removeprefix("is_"))] = r.get("score")

    summary: dict[str, Any] = {"annotators": annotators, "missing": [], "smells": {}}
    rows = []
    for case_id, k in key.items():
        r = ratings.get(case_id, {})
        if len(r) < len(annotators):
            summary["missing"].append(case_id)
        present = sum(v > 0 for v in r.values())
        m = meta[k["json_index"]]
        s = short[k["smell"]]
        rows.append({**k, **{f"rating_{a}": r.get(a) for a in annotators},
                     "panel_present": int(present * 2 > len(r)) if r else None,
                     "panel_votes_present": present, "panel_n": len(r),
                     "mlcq_reviews": int(m[f"n_reviews_{s}"] or 0),
                     "mlcq_present_votes": sum(int(m[f"votes_{s}_{v}"] or 0) for v in SEVERITY[1:]),
                     "mlcq_severity": m[f"severity_{s}"],
                     "llm_judge_score": judge.get((k["json_index"], k["smell"]))})

    for smell in sorted({k["smell"] for k in key.values()}):
        out: dict[str, Any] = {"groups": {}}
        sr = [r for r in rows if r["smell"] == smell and r["panel_present"] is not None]
        for group in GROUPS:
            g = [r for r in sr if r["group"] == group]
            k_present = sum(r["panel_present"] for r in g)
            agree = sum(r["panel_present"] == r["stored_label"] for r in g)
            scores = [r["llm_judge_score"] for r in g if r["llm_judge_score"] is not None]
            out["groups"][group] = {
                "n": len(g), "panel_present": k_present, "panel_present_rate": k_present / max(len(g), 1),
                "panel_present_ci95": wilson(k_present, len(g)), "agree_with_mlcq": agree / max(len(g), 1),
                "mlcq_multi_review": sum(r["mlcq_reviews"] >= 2 for r in g),
                "mlcq_any_present_vote": sum(r["mlcq_present_votes"] > 0 for r in g),
                "llm_judge_median_score": float(np.median(scores)) if scores else None,
            }
        for a, b in (("hard_pos", "easy_pos"), ("hard_neg", "easy_neg")):
            ga, gb = out["groups"][a], out["groups"][b]
            table = [[ga["panel_present"], ga["n"] - ga["panel_present"]],
                     [gb["panel_present"], gb["n"] - gb["panel_present"]]]
            odds, p = fisher_exact(table)
            out[f"fisher_{a}_vs_{b}"] = {"table": table, "odds_ratio": float(odds), "p": float(p)}
        units = [[r[f"rating_{a}"] for a in annotators if r[f"rating_{a}"] is not None] for r in sr]
        out["panel_alpha_nominal"] = krippendorff_alpha([[int(v > 0) for v in u] for u in units], 2, "nominal")
        out["panel_alpha_ordinal"] = krippendorff_alpha(units, len(SEVERITY), "ordinal")
        out["panel_vs_mlcq_agreement_all"] = sum(r["panel_present"] == r["stored_label"] for r in sr) / max(len(sr), 1)
        summary["smells"][smell] = out
        print(f"[audit] {smell}: alpha panel {out['panel_alpha_nominal']:.3f} | " + " | ".join(
            f"{g} present {v['panel_present']}/{v['n']}" for g, v in out["groups"].items()))

    out_dir = resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=1))
    with (out_dir / "audit_cases.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    agreement_counts = Counter(tuple(sorted(r.values())) for r in ratings.values())
    print(f"[audit] {len(ratings)} rated cases, {len(summary['missing'])} incomplete, "
          f"{sum(1 for c in agreement_counts if len(set(c)) == 1)} distinct unanimous rating patterns -> {out_dir}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = cfg["figure"]["group_labels"]
    smells = list(summary["smells"])
    fig, axes = plt.subplots(1, len(smells), figsize=(3.2 * len(smells), 3.0), sharey=True)
    for ax, smell in zip(np.atleast_1d(axes), smells):
        groups = summary["smells"][smell]["groups"]
        rates = [groups[g]["panel_present_rate"] for g in GROUPS]
        lo = [rates[i] - groups[g]["panel_present_ci95"][0] for i, g in enumerate(GROUPS)]
        hi = [groups[g]["panel_present_ci95"][1] - rates[i] for i, g in enumerate(GROUPS)]
        ax.bar(range(len(GROUPS)), rates, yerr=[lo, hi], capsize=3,
               color=["#009E73", "#E69F00", "#D55E00", "#0072B2"])
        ax.set_xticks(range(len(GROUPS)), [labels[g] for g in GROUPS], rotation=30, ha="right")
        ax.set_title(cfg["figure"]["smell_titles"][smell])
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel(cfg["figure"]["ylabel"])
    fig.tight_layout()
    fig_dir = resolve(cfg["figure"]["dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{cfg['figure']['name']}.{ext}", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
