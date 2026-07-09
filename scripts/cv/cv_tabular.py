"""Train and evaluate tabular models on the shared cross-validation folds.

Feature sets, models and runs come from the `tabular_cv` section of the config.
Writes per-run records.json and prediction files, plus one file per metamorphic variant.

Usage:
    python scripts/cv/cv_tabular.py --config config/experiments/shortcut_baselines.yml
    python scripts/cv/cv_tabular.py --config ... --shard-index 0 --shard-count 4
    python scripts/cv/cv_tabular.py --config config/experiments/dacos_tabular.yml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
    shard,
    variant_predictions_path,
    write_jsonl,
)
from mlcq_graphs.models.classical import build_classical_model  # noqa: E402

LABEL_COLUMNS = {"y_fe", "y_lm", "y_blob", "y_dc"}
PARTS = ("train", "val", "test")


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_feature_set(spec: dict[str, Any]) -> tuple[dict[int, np.ndarray], list[str]]:
    """Return {json_index: feature vector} and the feature names."""
    source = str(spec.get("source", "csv"))
    if source == "csv":
        with resolve(spec["path"]).open() as fh:
            rows = list(csv.DictReader(fh))
        key = str(spec.get("key", "sample_idx"))
        columns = spec.get("columns")
        if columns is None:
            exclude = {key, *LABEL_COLUMNS, *spec.get("exclude_columns", [])}
            columns = [c for c in rows[0].keys() if c not in exclude]
            labels = [c for c in columns if c == "y" or c.startswith("y_")]
            if labels:
                raise ValueError(f"{spec['path']}: columns {labels} look like labels; list the feature columns "
                                 "in `columns` or add them to `exclude_columns`")
        table = {int(float(r[key])): np.array([float(r[c]) for c in columns], dtype=np.float64) for r in rows}
        transform = spec.get("transform")
        if transform == "signed_log1p":
            table = {k: np.sign(v) * np.log1p(np.abs(v)) for k, v in table.items()}
        return table, list(columns)
    if source == "npy":
        matrix = np.load(resolve(spec["path"]))
        index = json.loads(resolve(spec["index"]).read_text())
        if len(index) != matrix.shape[0]:
            raise ValueError(f"{spec['path']}: {matrix.shape[0]} rows but {len(index)} indices")
        names = [f"{spec.get('prefix', 'e')}{i}" for i in range(matrix.shape[1])]
        return {int(j): matrix[k].astype(np.float64) for k, j in enumerate(index)}, names
    raise ValueError(f"unknown feature source: {source}")


def feature_matrix(tables: list[dict[int, np.ndarray]], allow_missing: list[bool], indices: np.ndarray) -> np.ndarray:
    """Rows of the concatenated feature sets, with NaN for allowed missing samples."""
    widths = [len(next(iter(t.values()))) for t in tables]
    rows = []
    for i in indices:
        parts = []
        for t, ok, w in zip(tables, allow_missing, widths):
            if int(i) in t:
                parts.append(t[int(i)])
            elif ok:
                parts.append(np.full(w, np.nan))
            else:
                raise ValueError(f"sample {int(i)} has no features and missing values are not allowed")
        rows.append(np.concatenate(parts))
    return np.stack(rows)


def missing_files(spec: dict[str, Any]) -> list[str]:
    return [str(spec[k]) for k in ("path", "index") if k in spec and not resolve(spec[k]).exists()]


FeatureCache = dict[Any, tuple[dict[int, np.ndarray], list[str]]]


def allows_missing(spec: dict[str, Any]) -> bool:
    return str(spec.get("missing", "error")) == "train_median"


def fold_matrices(
    tcfg: dict[str, Any], cache: FeatureCache, features: list[str], fold: Any,
) -> tuple[dict[str, np.ndarray], int, tuple[np.ndarray, StandardScaler]]:
    """Imputed and standardised train, val and test matrices, with the medians and scaler."""
    tables = []
    for fs in features:
        if fs not in cache:
            cache[fs] = load_feature_set(tcfg["feature_sets"][fs])
        tables.append(cache[fs][0])
    allow = [allows_missing(tcfg["feature_sets"][fs]) for fs in features]
    X = {p: feature_matrix(tables, allow, getattr(fold, p)) for p in PARTS}
    medians = np.nan_to_num(np.nanmedian(X["train"], axis=0))
    n_imputed = int(sum(np.isnan(v).any(axis=1).sum() for v in X.values()))
    X = {p: np.where(np.isnan(v), medians, v) for p, v in X.items()}
    scaler = StandardScaler().fit(X["train"])
    return {p: scaler.transform(v) for p, v in X.items()}, n_imputed, (medians, scaler)


VARIANT_SPEC_KEYS = ("path", "index")


def variant_spec(base: dict[str, Any], override: dict[str, Any], variant: str, fs: str) -> dict[str, Any]:
    """Base feature-set spec with the variant files, refusing any other changed key."""
    differ = sorted(k for k, v in override.items() if k not in VARIANT_SPEC_KEYS and v != base.get(k))
    if differ:
        raise SystemExit(f"variant {variant}: feature set {fs} gives other values than its base spec for {differ}; "
                         f"a variant spec gives only {list(VARIANT_SPEC_KEYS)}, the rest comes from feature_sets")
    return {**base, **override}


def variant_matrix(
    tcfg: dict[str, Any], cache: FeatureCache, variant: str, features: list[str], indices: np.ndarray,
    prep: tuple[np.ndarray, StandardScaler],
) -> np.ndarray:
    """Test rows with the variant feature sets, imputed and scaled as in the base fold."""
    overrides = tcfg["variants"][variant]
    tables, allow = [], []
    for fs in features:
        if fs not in cache:
            cache[fs] = load_feature_set(tcfg["feature_sets"][fs])
        spec = tcfg["feature_sets"][fs]
        if fs in overrides:
            key = (variant, fs)
            if key not in cache:
                cache[key] = load_feature_set(overrides[fs])
                if cache[key][1] != cache[fs][1]:
                    raise SystemExit(f"variant {variant}: feature set {fs} has columns {cache[key][1]}, "
                                     f"the base has {cache[fs][1]}")
            tables.append(cache[key][0])
            spec = overrides[fs]
        else:
            tables.append(cache[fs][0])
        allow.append(allows_missing(spec))
    medians, scaler = prep
    X = feature_matrix(tables, allow, indices)
    return scaler.transform(np.where(np.isnan(X), medians, X))


def fit_soft(model: Any, X: np.ndarray, shares: np.ndarray) -> Any:
    """Fit one estimator per label on the review shares, using weighted duplicated rows."""
    from sklearn.base import clone

    n = len(X)
    X2 = np.vstack([X, X])
    y2 = np.r_[np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)]
    estimators = []
    for i in range(shares.shape[1]):
        w2 = np.r_[shares[:, i], 1.0 - shares[:, i]]
        keep = w2 > 0
        estimators.append(clone(model.estimator).fit(X2[keep], y2[keep], sample_weight=w2[keep]))
    model.estimators_ = estimators
    return model


def positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    return np.column_stack([est.predict_proba(X)[:, 1] for est in model.estimators_])


SELECTION_METRICS = {"pr_auc_macro": average_precision_score, "roc_auc_macro": roc_auc_score}


def selection_score(y: np.ndarray, probs: np.ndarray, metric: str) -> float:
    """Macro average of a ranking metric over the labels with both classes in y."""
    fn = SELECTION_METRICS[metric]
    scores = [fn(y[:, i], probs[:, i]) for i in range(y.shape[1]) if 0 < int(y[:, i].sum()) < len(y)]
    return float(np.mean(scores)) if scores else float("nan")


def run_options(run: dict[str, Any]) -> tuple[list[list[str]], list[str], bool]:
    """Feature-set lists and model keys a run chooses from, and whether it selects on val."""
    if run.get("feature_candidates") is not None and run.get("features") is not None:
        raise ValueError(f"run {run.get('name')}: give features or feature_candidates, not both")
    if run.get("model_candidates") is not None and run.get("model") is not None:
        raise ValueError(f"run {run.get('name')}: give model or model_candidates, not both")
    feature_options = ([list(f) for f in run["feature_candidates"]] if run.get("feature_candidates") is not None
                       else [list(run["features"])])
    model_options = list(run["model_candidates"]) if run.get("model_candidates") is not None else [run["model"]]
    selecting = run.get("feature_candidates") is not None or run.get("model_candidates") is not None
    return feature_options, model_options, selecting


def run_name(run: dict[str, Any], feature_options: list[list[str]], model_options: list[str]) -> str:
    features = "|".join("+".join(f) for f in feature_options)
    return str(run.get("name") or f"{features}__{'|'.join(model_options)}")


Matrices = tuple[dict[str, np.ndarray], int, tuple[np.ndarray, StandardScaler]]


def select_on_val(
    candidates: list[tuple[list[str], str, str]], models_cfg: dict[str, Any], seed: int, metric: str,
    matrices: Callable[[list[str]], Matrices], Y: dict[str, np.ndarray],
) -> tuple[tuple[Any, list[str], str, str, Matrices], dict[str, float]]:
    """Fit every candidate on train and keep the one with the best validation score."""
    best: tuple[float, Any, list[str], str, str, Matrices] | None = None
    scores: dict[str, float] = {}
    for features, key, label in candidates:
        prepared = matrices(features)
        X = prepared[0]
        spec = dict(models_cfg[key])
        model = build_classical_model(spec.pop("type"), {**spec, "seed": seed})
        model.fit(X["train"], Y["train"])
        score = selection_score(Y["val"], positive_proba(model, X["val"]), metric)
        scores[label] = score
        rank = score if np.isfinite(score) else -np.inf
        if best is None or rank > best[0]:
            best = (rank, model, features, key, label, prepared)
    assert best is not None
    return best[1:], scores


def write_run(out: Path, records: list[dict[str, Any]], preds: list[dict[str, Any]],
              variant_preds: dict[str, list[dict[str, Any]]]) -> None:
    """Write predictions first and records.json last."""
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "predictions.jsonl", preds)
    for variant, rows in variant_preds.items():
        write_jsonl(variant_predictions_path(out, variant), rows)
    tmp = out / "records.json.tmp"
    tmp.write_text(json.dumps(records, indent=1))
    tmp.replace(out / "records.json")
    # folds of this output whose chosen feature sets each variant overrides
    applied = ", ".join(f"{v} {sum(bool(r['variant_overrides'][v]) for r in records)}/{len(records)} folds"
                        for v in sorted(variant_preds))
    print(f"[tabular] wrote {out}" + (f" (variants: {applied})" if variant_preds else ""), flush=True)


def is_complete(out: Path, n_records: int, variants: list[str]) -> bool:
    try:
        complete = len(json.loads((out / "records.json").read_text())) == n_records
    except (OSError, json.JSONDecodeError):
        return False
    return complete and all(variant_predictions_path(out, v).exists() for v in variants)


def short_name(label: str) -> str:
    return label.removeprefix("is_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))
    tcfg = cfg["tabular_cv"]
    config_name = str(cfg.get("run", {}).get("name", args.config.stem))

    folds = load_folds(resolve(tcfg["folds"]))
    entries = json.loads(resolve(tcfg.get("input_json", folds["meta"]["input_json"])).read_text())
    label_order = folds_label_order(folds, tcfg.get("label_order"))
    num_labels = len(label_order)
    steps = int(tcfg.get("threshold_steps", 101))
    out_root = resolve(tcfg.get("out_dir", f"artifacts/tabular_cv/{config_name}"))
    fold_list = list(iter_folds(folds, tcfg.get("strategies"), tcfg.get("seeds"), tcfg.get("folds_subset")))

    # check feature files before any fit
    on_missing = str(tcfg.get("missing_features", "error"))
    absent = {fs: missing_files(spec) for fs, spec in tcfg["feature_sets"].items() if missing_files(spec)}
    variants: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, overrides in (tcfg.get("variants") or {}).items():
        unknown = sorted(set(overrides) - set(tcfg["feature_sets"]))
        if unknown:
            raise SystemExit(f"variant {variant} overrides feature sets that are not in feature_sets: {unknown}")
        variants[variant] = {fs: variant_spec(tcfg["feature_sets"][fs], dict(spec), variant, fs)
                             for fs, spec in overrides.items()}
    tcfg["variants"] = variants
    absent_variants: dict[str, list[str]] = {}
    for variant, overrides in variants.items():
        files = sorted({p for spec in overrides.values() for p in missing_files(spec)})
        if files:
            if on_missing != "skip":
                raise SystemExit(f"variant {variant}: feature files not found: {files} "
                                 "(missing_features: skip leaves such variants out)")
            print(f"[tabular] variant {variant} left out (missing feature files {files})", flush=True)
            absent_variants[variant] = files
    runs = []
    for run in tcfg["runs"]:
        feature_options, model_options, selecting = run_options(run)
        name = run_name(run, feature_options, model_options)
        kept = [f for f in feature_options if not any(fs in absent for fs in f)]
        if len(kept) < len(feature_options):
            dropped = [f for f in feature_options if f not in kept]
            if on_missing != "skip":
                files = sorted({p for f in dropped for fs in f for p in absent.get(fs, [])})
                raise SystemExit(f"run {name}: feature files not found: {files} "
                                 "(missing_features: skip leaves such runs and candidates out)")
            print(f"[tabular] {name}: left out {dropped} (missing feature files)", flush=True)
            if not kept:
                continue
        used = {fs for f in kept for fs in f}
        run_variants = [v for v, overrides in variants.items() if v not in absent_variants and used & set(overrides)]
        runs.append((run, name, kept, model_options, selecting, run_variants))

    jobs = [(r, f) for r in runs for f in fold_list]
    jobs = shard(jobs, args.shard_index, args.shard_count)
    suffix = f"_shard{args.shard_index}of{args.shard_count}" if args.shard_count > 1 else ""
    remaining = Counter(r[1] for r, _ in jobs)
    run_variant_names = {r[1]: r[5] for r, _ in jobs}
    shares = None
    if any(str(r[0].get("target", "consensus")) == "review_share" for r in runs):
        obj = json.loads(resolve(tcfg["review_shares"]).read_text())
        if list(obj["meta"]["label_order"]) != label_order:
            raise SystemExit(f"review_shares label order {obj['meta']['label_order']} differs from {label_order}")
        shares = np.asarray(obj["share"], dtype=np.float64)
        if len(shares) != len(entries):
            raise SystemExit(f"review_shares has {len(shares)} rows, input_json {len(entries)}")
    for run, name, _, _, selecting, _ in runs:
        target = str(run.get("target", "consensus"))
        if target not in ("consensus", "review_share"):
            raise SystemExit(f"run {name}: unknown target {target!r} (consensus | review_share)")
        if target == "review_share" and selecting:
            raise SystemExit(f"run {name}: target review_share cannot select on validation")
    finished = set()
    if bool(tcfg.get("skip_existing", False)):
        finished = {name for name, n in remaining.items()
                    if is_complete(out_root / f"{name}{suffix}", n, run_variant_names[name])}
        for name in sorted(finished):
            print(f"[tabular] {name}{suffix} already complete, skipping", flush=True)

    feature_cache: FeatureCache = {}
    by_run: dict[str, tuple[list, list, dict[str, list]]] = {}
    for (run, name, feature_options, model_options, selecting, run_variants), fold in jobs:
        if name in finished:
            continue
        model_key = "|".join(model_options)
        features_key = "|".join("+".join(f) for f in feature_options)
        Y = {p: label_matrix_from_json(entries, getattr(fold, p), num_labels) for p in PARTS}

        selection: dict[str, Any] = {}
        if not selecting:
            chosen_features = feature_options[0]
            X, n_imputed, prep = fold_matrices(tcfg, feature_cache, chosen_features, fold)
            model_spec = dict(tcfg["models"][model_options[0]])
            model_type = model_spec.pop("type")
            started = time.perf_counter()
            model = build_classical_model(model_type, {**model_spec, "seed": fold.seed})
            if str(run.get("target", "consensus")) == "review_share":
                assert shares is not None
                fit_soft(model, X["train"], shares[fold.train, :num_labels])
            else:
                model.fit(X["train"], Y["train"])
        else:
            # the duration covers the preparation and the fit of every candidate
            started = time.perf_counter()
            metric = str(run.get("selection_metric", tcfg.get("selection_metric", "pr_auc_macro")))
            by_features = run.get("feature_candidates") is not None
            candidates = [(f, key, f"{'+'.join(f)}__{key}" if by_features else key)
                          for f in feature_options for key in model_options]
            (model, chosen_features, chosen, label, (X, n_imputed, prep)), scores = select_on_val(
                candidates, tcfg["models"], fold.seed, metric,
                lambda f: fold_matrices(tcfg, feature_cache, f, fold), Y)
            selection = {"selected_model": chosen, "selection_metric": metric, "selection_val_scores": scores}
            if by_features:
                selection["selected_features"] = "+".join(chosen_features)
        p_val, p_test = positive_proba(model, X["val"]), positive_proba(model, X["test"])
        # an empty list: the variant leaves the chosen feature sets untouched
        overrides = ({"variant_overrides": {v: [fs for fs in chosen_features if fs in tcfg["variants"][v]]
                                            for v in run_variants}} if run_variants else {})
        record = build_record(
            combo={"run": name, "model": model_key, "features": features_key,
                   "split_strategy": fold.strategy, "cv_fold": fold.fold},
            seed=fold.seed,
            y_val=Y["val"], p_val=p_val, y_test=Y["test"], p_test=p_test,
            threshold_steps=steps,
            label_names=label_order,
            extra={"train_duration_sec": time.perf_counter() - started,
                   "train_target": str(run.get("target", "consensus")),
                   "num_features": int(X["train"].shape[1]), "imputed_samples": n_imputed, **selection,
                   **overrides},
        )
        records, preds, variant_preds = by_run.setdefault(name, ([], [], {}))
        records.append(record)
        for part, idx, y, p in (("val", fold.val, Y["val"], p_val), ("test", fold.test, Y["test"], p_test)):
            preds.extend(prediction_rows(model=name, strategy=fold.strategy, seed=fold.seed, fold=fold.fold,
                                         part=part, indices=idx, y=y, probs=p))
        # every variant is scored on every fold, even when it overrides nothing
        for variant in run_variants:
            Xv = variant_matrix(tcfg, feature_cache, variant, chosen_features, fold.test, prep)
            variant_preds.setdefault(variant, []).extend(prediction_rows(
                model=name, strategy=fold.strategy, seed=fold.seed, fold=fold.fold, part="test",
                indices=fold.test, y=Y["test"], probs=positive_proba(model, Xv)))
        per = record["per_label_tuned"]
        shown = " ".join(f"{short_name(s)}={'NA' if per[s]['mcc'] is None else round(per[s]['mcc'], 3)}"
                         for s in label_order)
        chosen_note = f" selected={label}" if selection else ""
        print(f"[tabular] {name} {fold.strategy} seed={fold.seed} fold={fold.fold} "
              f"macroMCC={record['mcc_tuned']:.3f} {shown}{chosen_note}", flush=True)
        remaining[name] -= 1
        if remaining[name] == 0:
            write_run(out_root / f"{name}{suffix}", *by_run.pop(name))


if __name__ == "__main__":
    main()
