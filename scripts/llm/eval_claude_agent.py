"""Evaluate the Claude Code judge (scripts/llm/claude_agent_judge.py) on its subset.

Reports ROC-AUC, PR-AUC, precision, recall and MCC per smell and condition,
reference scorers on the same pairs and the run costs.

Usage:
    python scripts/llm/eval_claude_agent.py --config config/experiments/claude_agent.yml
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402

from scripts.llm.claude_agent_judge import load_meta  # noqa: E402


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def reference_scores(spec: dict[str, Any], labels: list[str]) -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    if spec["kind"] == "judge_scores":
        for line in resolve(spec["path"]).open():
            row = json.loads(line)
            if row.get("applicable"):
                out[(int(row["json_index"]), row["smell"])] = float(row["score"])
    elif spec["kind"] == "oof_predictions":
        for path in sorted(glob.glob(str(resolve(spec["glob"])))):
            for line in open(path):
                row = json.loads(line)
                if row["part"] == "test" and row["strategy"] == spec["strategy"] and int(row["seed"]) == int(spec["seed"]):
                    probs = row.get("prob", row.get("probs"))
                    for col, label in enumerate(labels):
                        out[(int(row["json_index"]), label)] = float(probs[col])
    else:
        raise ValueError(spec["kind"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    root = load_config(args.config, parse_cli_overrides(extra))
    cfg, ecfg = root["claude_agent"], root["claude_agent_eval"]
    out_dir = resolve(cfg["out_dir"])
    entries = json.loads(resolve(cfg["input_json"]).read_text())
    meta = {int(r["json_index"]): r for r in load_meta(cfg, entries)}
    labels = list(cfg["label_columns"])

    rows: dict[tuple[str, int, str], dict] = {}
    for line in (out_dir / "results.jsonl").open():
        r = json.loads(line)
        if r.get("score") is not None:
            rows[(r["condition"], r["json_index"], r["smell"])] = r
    pairs = [tuple(p) for p in json.loads((out_dir / "subset.json").read_text())]
    conditions = list(cfg["conditions"])
    refs = {name: reference_scores(spec, labels) for name, spec in ecfg["references"].items()}
    rng = np.random.default_rng(int(ecfg["bootstrap_seed"]))

    report: dict[str, Any] = {"by_smell": {}, "costs": {}}
    for label in labels:
        # MLCQ meta rows hold the label under its CSV column, DACOS rows under the label name
        col = label if label in next(iter(meta.values())) else cfg["label_columns"][label]
        sp = [(i, s) for i, s in pairs if s == label and all((c, i, s) in rows for c in conditions)]
        missing = sum(1 for i, s in pairs if s == label) - len(sp)
        # keep references covering 95% of the pairs, then compare on their common pairs
        kept = [name for name, ref in refs.items()
                if sum((i, label) in ref for i, _ in sp) >= 0.95 * max(1, len(sp))]
        common = [(i, s) for i, s in sp if all((i, label) in refs[name] for name in kept)]
        dropped = len(sp) - len(common)
        sp = common
        y = np.array([int(meta[i][col]) for i, _ in sp])
        if len(sp) < 10 or y.min() == y.max():
            report["by_smell"][label] = {"n": len(sp), "missing_pairs": missing}
            continue
        scorers = {c: np.array([rows[(c, i, label)]["score"] for i, _ in sp]) for c in conditions}
        for name in kept:
            scorers[name] = np.array([refs[name][(i, label)] for i, _ in sp])
        res: dict[str, Any] = {"n": len(sp), "n_pos": int(y.sum()), "missing_pairs": missing,
                               "dropped_for_common_support": dropped, "scorers": {}}
        for name, s in scorers.items():
            m = {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s)}
            if name in conditions:
                pred = np.array([rows[(name, i, label)]["answer"] == "yes" for i, _ in sp], dtype=int)
                tp = int(((pred == 1) & (y == 1)).sum())
                m.update(precision=tp / max(1, pred.sum()), recall=tp / max(1, y.sum()),
                         mcc=matthews_corrcoef(y, pred), yes_rate=float(pred.mean()))
            res["scorers"][name] = m
        # agent queries that opened at least one file (more than one turn) against those that did not
        if "agent" in conditions and ecfg["baseline"] in conditions:
            turns = np.array([rows[("agent", i, label)].get("num_turns") or 1 for i, _ in sp])
            split = {}
            for part, mask in (("used_tools", turns > 1), ("no_tools", turns <= 1)):
                if mask.sum() >= 10 and 0 < y[mask].sum() < mask.sum():
                    a, s = scorers["agent"][mask], scorers[ecfg["baseline"]][mask]
                    split[part] = {"n": int(mask.sum()), "n_pos": int(y[mask].sum()),
                                   "auc_agent": roc_auc_score(y[mask], a),
                                   "auc_" + ecfg["baseline"]: roc_auc_score(y[mask], s),
                                   "mean_score_change_pos": float((a - s)[y[mask] == 1].mean()),
                                   "mean_score_change_neg": float((a - s)[y[mask] == 0].mean())}
            res["by_tool_use"] = split
        # paired bootstrap of AUC differences against the snippet condition
        base = scorers[ecfg["baseline"]]
        diffs: dict[str, list[float]] = defaultdict(list)
        for _ in range(int(ecfg["bootstrap"])):
            idx = rng.integers(0, len(y), len(y))
            if y[idx].min() == y[idx].max():
                continue
            b = roc_auc_score(y[idx], base[idx])
            for name, s in scorers.items():
                if name != ecfg["baseline"]:
                    diffs[name].append(roc_auc_score(y[idx], s[idx]) - b)
        res["auc_diff_vs_" + ecfg["baseline"]] = {
            name: {"mean": res["scorers"][name]["roc_auc"] - res["scorers"][ecfg["baseline"]]["roc_auc"],
                   "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}
            for name, d in diffs.items()}
        report["by_smell"][label] = res
    for c in conditions:
        rs = [r for k, r in rows.items() if k[0] == c]
        def med(key: str) -> float | None:
            v = [float(r[key]) for r in rs if r.get(key) is not None]
            return float(np.median(v)) if v else None
        report["costs"][c] = {"queries": len(rs), "median_wall_seconds": med("wall_seconds"),
                              "median_turns": med("num_turns"), "median_output_tokens": med("output_tokens"),
                              "median_cache_read_tokens": med("cache_read_tokens"),
                              "median_cost_usd": med("cost_usd"),
                              "total_cost_usd": float(sum(r.get("cost_usd") or 0 for r in rs))}
    path = out_dir / "evaluation.json"
    path.write_text(json.dumps(report, indent=1))
    for label, res in report["by_smell"].items():
        if "scorers" not in res:
            print(f"{label}: only {res['n']} complete pairs, {res['missing_pairs']} missing")
            continue
        print(f"{label} (n {res['n']}, positives {res['n_pos']}, missing {res['missing_pairs']})")
        for name, m in res["scorers"].items():
            extra_txt = (f"  P {m['precision']:.2f} R {m['recall']:.2f} MCC {m['mcc']:+.3f}"
                         if "mcc" in m else "")
            print(f"  {name:<18} ROC-AUC {m['roc_auc']:.3f}  PR-AUC {m['pr_auc']:.3f}{extra_txt}")
        for name, d in res["auc_diff_vs_" + ecfg["baseline"]].items():
            print(f"    {name} - {ecfg['baseline']}: {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    for c, m in report["costs"].items():
        print(f"cost {c}: {m}")
    print(f"[claude_agent] wrote {path}")


if __name__ == "__main__":
    main()
