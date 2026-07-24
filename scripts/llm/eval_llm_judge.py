"""Evaluate the zero-shot LLM judge scores on the shared cross-validation folds.

Thresholds are tuned on each validation fold. Writes records.json,
predictions.jsonl and diagnostics.json per judge and score temperature.

Usage:
    python scripts/llm/eval_llm_judge.py --config config/experiments/llm_judge.yml --judge qwen25_coder_7b_instruct
    python scripts/llm/eval_llm_judge.py --config ... --judge qwen36_27b_fp8 --judge qwen25_coder_7b_instruct
    python scripts/llm/eval_llm_judge.py --config ... --judge qwen36_27b_fp8 '--llm_judge.eval.score_temperatures=[1, 8]'
    python scripts/llm/eval_llm_judge.py --config config/experiments/dacos_llm_judge.yml --judge qwen36_27b_fp8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.evaluation.cv import (  # noqa: E402
    build_record,
    folds_label_order,
    iter_folds,
    label_matrix_from_json,
    load_folds,
    prediction_rows,
    write_jsonl,
)


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def load_scores(score_dir: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], list[str]]:
    """{(json_index, smell): row} over every shard file, and the prompt hashes seen."""
    files = sorted(score_dir.glob("scores*.jsonl"))
    if not files:
        raise SystemExit(f"no scores*.jsonl under {score_dir}")
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    hashes: set[str] = set()
    for path in files:
        with path.open() as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (int(row["json_index"]), str(row["smell"]))
                if key in rows:
                    raise SystemExit(f"duplicate score for {key} in {path}")
                rows[key] = row
                hashes.add(str(row.get("prompt_hash")))
    return rows, sorted(hashes)


def score_matrix(
    rows: dict[tuple[int, str], dict[str, Any]],
    indices: list[int],
    label_order: list[str],
    nan_fill: float,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the score matrix, a mask of complete samples and a mask of filled NaN cells.

    Applicable scores are rescaled to sigmoid(logit(score) / temperature)."""
    probs = np.zeros((len(indices), len(label_order)), dtype=np.float64)
    complete = np.ones(len(indices), dtype=bool)
    filled = np.zeros_like(probs, dtype=bool)
    applicable = np.zeros_like(probs, dtype=bool)
    for r, idx in enumerate(indices):
        for c, smell in enumerate(label_order):
            row = rows.get((idx, smell))
            if row is None:
                complete[r] = False
                continue
            applicable[r, c] = bool(row.get("applicable"))
            value = row["score"]
            if value is None or math.isnan(float(value)):
                probs[r, c] = nan_fill
                filled[r, c] = True
            else:
                probs[r, c] = float(value)
    if temperature != 1.0:
        eps = 1e-15
        clipped = np.clip(probs, eps, 1 - eps)
        rescaled = 1.0 / (1.0 + np.exp(-np.log(clipped / (1 - clipped)) / temperature))
        probs = np.where(applicable, rescaled, probs)
    return probs, complete, filled


def diagnostics(
    rows: dict[tuple[int, str], dict[str, Any]],
    entries: list[dict[str, Any]],
    label_order: list[str],
) -> dict[str, Any]:
    """Fold-free ranking metrics per smell on applicable samples, plus NaN counts."""
    out: dict[str, Any] = {}
    for c, smell in enumerate(label_order):
        applicable = [r for (i, s), r in rows.items() if s == smell and r.get("applicable")]
        y, p = [], []
        nan = 0
        for r in applicable:
            v = r["score"]
            if v is None or math.isnan(float(v)):
                nan += 1
                continue
            yv = label_matrix_from_json(entries, np.array([r["json_index"]]), len(label_order))[0, c]
            y.append(int(yv))
            p.append(float(v))
        y_arr, p_arr = np.array(y), np.array(p)
        both = len(set(y)) == 2
        out[smell] = {
            "applicable": len(applicable),
            "positives": int(y_arr.sum()) if len(y) else 0,
            "nan": nan,
            "one_sided": sum(1 for r in applicable if (r["p_yes"] > 0) != (r["p_no"] > 0)),
            "mean_yes_no_mass": float(np.mean([r["p_yes"] + r["p_no"] for r in applicable])) if applicable else None,
            "roc_auc": float(roc_auc_score(y_arr, p_arr)) if both else None,
            "pr_auc": float(average_precision_score(y_arr, p_arr)) if both else None,
            "yes_rate_at_0.5": float(np.mean(p_arr >= 0.5)) if len(p) else None,
            "mean_score_pos": float(p_arr[y_arr == 1].mean()) if both else None,
            "mean_score_neg": float(p_arr[y_arr == 0].mean()) if both else None,
        }
    return out


def plot_score_distributions(
    rows: dict[tuple[int, str], dict[str, Any]],
    entries: list[dict[str, Any]],
    label_order: list[str],
    diag: dict[str, Any],
    fig_cfg: dict[str, Any],
    title: str,
    out_stem: Path,
) -> bool:
    """Plot log-odds histograms of positives and negatives per smell as PDF and PNG.

    Returns False when matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval] matplotlib not available: figure skipped", flush=True)
        return False
    colors = fig_cfg["colors"]
    clip = float(fig_cfg["log_odds_clip"])
    bins = np.linspace(-clip, clip, int(fig_cfg["bins"]) + 1)
    fig, axes = plt.subplots(1, len(label_order), figsize=tuple(fig_cfg["size"]), sharex=True)
    for c, (ax, smell) in enumerate(zip(np.atleast_1d(axes), label_order)):
        y, p = [], []
        for (i, s), r in rows.items():
            v = r["score"]
            if s != smell or not r.get("applicable") or v is None or math.isnan(float(v)):
                continue
            y.append(int(label_matrix_from_json(entries, np.array([i]), len(label_order))[0, c]))
            p.append(float(v))
        y_arr = np.array(y)
        p_arr = np.clip(np.array(p, dtype=np.float64), 1e-15, 1 - 1e-15)
        p_arr = np.clip(np.log(p_arr / (1 - p_arr)), -clip, clip)
        for value, key, label in ((0, "negative", "not smelly"), (1, "positive", "smelly")):
            part = p_arr[y_arr == value]
            if len(part) == 0:
                continue
            weights = np.full(len(part), 1.0 / len(part))
            ax.hist(part, bins=bins, weights=weights, histtype="step", linewidth=1.5, color=colors[key],
                    label=f"{label} (n={len(part)})")
            ax.hist(part, bins=bins, weights=weights, histtype="stepfilled", alpha=0.15, color=colors[key])
        ax.axvline(0.0, color="#8a8984", linewidth=0.8, linestyle="--")
        d = diag["per_smell"][smell]
        auc = "NA" if d["roc_auc"] is None else f"{d['roc_auc']:.2f}"
        ax.set_title(f"{smell.removeprefix('is_').replace('_', ' ')}\nROC AUC {auc}", fontsize=9)
        ax.set_xlabel("log P(yes) - log P(no)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    np.atleast_1d(axes)[0].set_ylabel("fraction of samples", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)
    return True


def run_name(name: str, suffix: str, temperature: float) -> str:
    """<judge><tag><name_suffix>, plus _T<T> when the scores are rescaled."""
    return f"{name}{suffix}" if temperature == 1.0 else f"{name}{suffix}_T{temperature:g}"


def evaluate_judge(cfg: dict[str, Any], judge: str) -> None:
    jcfg = cfg["llm_judge"]
    ecfg = jcfg["eval"]
    name = f"{judge}{jcfg.get('tag') or ''}"
    score_dir = resolve(jcfg["out_root"]) / name
    judge_cfg = jcfg["judges"][judge]
    model = str(judge_cfg.get("hf_id") or judge_cfg.get("model_path"))

    folds = load_folds(resolve(jcfg["folds"]))
    label_order = folds_label_order(folds, jcfg.get("label_order"))
    entries = json.loads(resolve(jcfg["input_json"]).read_text())
    rows, hashes = load_scores(score_dir)
    if len(hashes) != 1:
        raise SystemExit(f"{score_dir} mixes prompt hashes {hashes}")
    temperatures = [float(t) for t in ecfg["score_temperatures"]]
    if len(set(temperatures)) != len(temperatures) or any(t <= 0 for t in temperatures):
        raise SystemExit(f"eval.score_temperatures must be distinct positive values: {temperatures}")
    for temperature in temperatures:
        evaluate_run(ecfg, judge, model, run_name(name, ecfg.get("name_suffix") or "", temperature),
                     temperature, folds, label_order, entries, rows, hashes[0])


def evaluate_run(
    ecfg: dict[str, Any],
    judge: str,
    model: str,
    run: str,
    temperature: float,
    folds: dict[str, Any],
    label_order: list[str],
    entries: list[dict[str, Any]],
    rows: dict[tuple[int, str], dict[str, Any]],
    prompt_hash: str,
) -> None:
    """Records, predictions, diagnostics and figure of one score temperature."""
    out = resolve(ecfg["out_dir"]) / run
    nan_fill = float(ecfg["nan_fill"])
    missing = str(ecfg["missing"])
    if missing not in ("error", "drop"):
        raise SystemExit(f"eval.missing must be error or drop, not {missing!r}")
    steps = int(ecfg["threshold_steps"])

    all_indices = [int(i) for i in folds["meta"]["json_index"]]
    probs, complete, filled = score_matrix(rows, all_indices, label_order, nan_fill, temperature)
    position = {idx: k for k, idx in enumerate(all_indices)}
    n_missing = int((~complete).sum())
    if n_missing and missing == "error":
        raise SystemExit(f"{n_missing} fold samples have no complete scores (set eval.missing=drop "
                         "to evaluate on the scored samples only)")
    print(f"[eval] {run}: {int(complete.sum())} scored samples, {n_missing} missing, "
          f"{int(filled.sum())} NaN cells filled with {nan_fill}; score temperature {temperature}", flush=True)
    combo_extra = {} if temperature == 1.0 else {"score_temperature": temperature}

    records, preds = [], []
    for fold in iter_folds(folds, ecfg["strategies"], ecfg["seeds"], ecfg["folds_subset"]):
        parts = {}
        for part in ("val", "test"):
            idx = np.array([i for i in getattr(fold, part) if complete[position[int(i)]]], dtype=np.int64)
            rows_pos = np.array([position[int(i)] for i in idx], dtype=np.int64)
            parts[part] = (idx, label_matrix_from_json(entries, idx, len(label_order)), probs[rows_pos],
                           int(filled[rows_pos].sum()))
        (vi, yv, pv, _), (ti, yt, pt, nan_test) = parts["val"], parts["test"]
        record = build_record(
            combo={"run": run, "model": model, **combo_extra, "split_strategy": fold.strategy, "cv_fold": fold.fold},
            seed=fold.seed, y_val=yv, p_val=pv, y_test=yt, p_test=pt, threshold_steps=steps,
            label_names=label_order,
            extra={"n_val": int(len(vi)), "n_test": int(len(ti)), "nan_filled_test": nan_test,
                   "prompt_hash": prompt_hash, "judge": judge},
        )
        records.append(record)
        for part, (idx, y, p, _) in parts.items():
            preds.extend(prediction_rows(model=run, strategy=fold.strategy, seed=fold.seed, fold=fold.fold,
                                         part=part, indices=idx, y=y, probs=p))
        per = record["per_label_tuned"]
        shown = " ".join(f"{s.removeprefix('is_')}={fmt(per[s]['mcc'])}" for s in label_order)
        print(f"[eval] {run} {fold.strategy} seed={fold.seed} fold={fold.fold} n_test={len(ti)} "
              f"macroMCC={fmt(record['mcc_tuned'])} fixed={fmt(record['mcc_fixed'])} {shown}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "records.json.tmp"
    tmp.write_text(json.dumps(records, indent=1))
    tmp.replace(out / "records.json")
    write_jsonl(out / "predictions.jsonl", preds)
    diag = {"judge": judge, "model": model, "prompt_hash": prompt_hash, "nan_fill": nan_fill,
            "score_temperature": temperature,
            "scored_samples": int(complete.sum()), "missing_samples": n_missing,
            "per_smell": diagnostics(rows, entries, label_order)}
    tmp = out / "diagnostics.json.tmp"
    tmp.write_text(json.dumps(diag, indent=1))
    tmp.replace(out / "diagnostics.json")
    for smell, d in diag["per_smell"].items():
        print(f"[eval] {run} {smell}: applicable={d['applicable']} pos={d['positives']} nan={d['nan']} "
              f"roc_auc={fmt(d['roc_auc'])} pr_auc={fmt(d['pr_auc'])} yes_rate@0.5={fmt(d['yes_rate_at_0.5'])}",
              flush=True)
    fig_cfg = ecfg["figure"]
    if fig_cfg["enabled"]:
        stem = out / "score_distributions"
        if plot_score_distributions(rows, entries, label_order, diag, fig_cfg, f"{run} ({model})", stem):
            print(f"[eval] wrote {stem}.pdf and .png", flush=True)
    print(f"[eval] wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--judge", action="append", default=None, help="judge name (repeatable)")
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))
    for judge in args.judge or [str(cfg["llm_judge"]["default_judge"])]:
        evaluate_judge(cfg, judge)


if __name__ == "__main__":
    main()
