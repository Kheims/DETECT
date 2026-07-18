"""Scan whole MLCQ project snapshots with the size-only detector and PMD.

Reports alerts per KLOC and per 100 units, latency per file and extraction
failures. Units are not annotated, so no precision is reported.

Usage:
    python scripts/analysis/scan_projects.py --config config/experiments/industrial.yml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cv.cv_tabular import load_feature_set, positive_proba  # noqa: E402
from scripts.data.extract_size_features import CLASS_TYPES, METHOD_TYPES, SIZE_COLUMNS, _parser, size_features  # noqa: E402
from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402
from mlcq_graphs.evaluation.cv import label_matrix_from_json, load_folds  # noqa: E402
from mlcq_graphs.models.classical import build_classical_model  # noqa: E402
from scripts.analysis.run_pmd import (CLASS_KINDS, CLASS_RULES, METHOD_KINDS, METHOD_RULES, ensure_pmd, parse_report,  # noqa: E402
                     rel_path, run_pmd, update_report, write_ruleset)

LEVEL_OF_NODE = {**{t: "method" for t in METHOD_TYPES}, **{t: "class" for t in CLASS_TYPES}}


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def scan_file(task: tuple[str, str, str, str, str]) -> dict[str, Any]:
    """Declarations of one file with their size counts, wall time and failure if any."""
    project, root, rel, prefix, suffix = task
    started = time.perf_counter()
    raw = (Path(root) / rel.lstrip("/")).read_bytes()
    try:
        text, decode_error = raw.decode("utf-8"), 0
    except UnicodeDecodeError:
        text, decode_error = raw.decode("utf-8", errors="replace"), 1
    # the MLCQ snippets have LF line ends, line numbers do not change
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    out = {"file": rel, "loc_nonblank": sum(1 for line in lines if line.strip()), "units": [],
           "parse_error": 0, "decode_error": decode_error, "failed": ""}
    try:
        tree = _parser().parse(text.encode("utf-8"))
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            level = LEVEL_OF_NODE.get(node.type)
            if level:
                s, e = node.start_point[0], node.end_point[0]
                name = node.child_by_field_name("name")
                feats = size_features((0, "\n".join(lines[s:e + 1]), prefix, suffix))
                feats.pop("sample_idx")
                out["units"].append({"project": project, "file": rel, "level": level, "start_line": s + 1,
                                     "end_line": e + 1,
                                     "name": name.text.decode("utf-8", errors="replace") if name is not None else "",
                                     **feats})
            stack.extend(reversed(node.children))
        out["parse_error"] = int(tree.root_node.has_error)
    except Exception as exc:  # noqa: BLE001 - a failure is counted, not fatal
        out["units"], out["failed"] = [], f"{type(exc).__name__}: {exc}"[:200]
    out["sec"] = time.perf_counter() - started
    return out


def java_files(root: Path) -> list[Path]:
    """Regular files ending in .java (some projects have directories named *.java)."""
    return [p for p in root.rglob("*.java") if p.is_file()]


def choose_snapshots(scfg: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    root = resolve(scfg["snapshots"])
    with resolve(scfg["fetch_log"]).open() as fh:
        log = [json.loads(line) for line in fh if line.strip()]
    candidates = []
    for row in log:
        d = root / f"{row['repo'].replace('/', '__')}@{row['commit']}"
        if (d / ".done").exists():
            candidates.append({"snapshot": d.name, "dir": d, "java_files": len(java_files(d))})
    candidates.sort(key=lambda c: (c["java_files"], c["snapshot"]))
    rng = np.random.default_rng(int(scfg["seed"]))
    chosen = []
    for k, stratum in enumerate(np.array_split(np.arange(len(candidates)), int(scfg["n_projects"]))):
        pick = candidates[int(stratum[rng.integers(len(stratum))])]
        lo, hi = candidates[int(stratum[0])]["java_files"], candidates[int(stratum[-1])]["java_files"]
        chosen.append({**pick, "stratum": k, "stratum_java_files": [lo, hi]})
    return chosen, len(candidates)


def refit_size_logreg(cfg: dict[str, Any]) -> tuple[Any, StandardScaler, list[str], dict[str, Any], np.ndarray, Any]:
    """Size-only logistic regression refit on every MLCQ sample of the folds."""
    scfg = cfg["scan"]
    base = load_config(resolve(scfg["shortcut_config"]), {})["tabular_cv"]
    run = scfg["shortcut_run"]
    if not any(r.get("features") == run["features"] and r.get("model") == run["model"] for r in base["runs"]):
        raise SystemExit(f"{scfg['shortcut_config']} has no run {run}")
    if len(run["features"]) != 1:
        raise SystemExit("the scan expects a single feature set")
    fspec = base["feature_sets"][run["features"][0]]
    table, columns = load_feature_set(fspec)
    folds = load_folds(resolve(base["folds"]))
    entries = json.loads(resolve(base["input_json"]).read_text())
    idx = np.asarray(folds["meta"]["json_index"], dtype=np.int64)
    X = np.stack([table[int(i)] for i in idx])
    Y = label_matrix_from_json(entries, idx, len(LABEL_ORDER))
    scaler = StandardScaler().fit(X)
    spec = dict(base["models"][run["model"]])
    model = build_classical_model(spec.pop("type"), {**spec, "seed": int(scfg["refit_seed"])})
    model.fit(scaler.transform(X), Y)
    return model, scaler, list(columns), fspec, idx, Y


def median_thresholds(scfg: dict[str, Any]) -> tuple[np.ndarray, list[list[float]]]:
    records = json.loads(resolve(scfg["shortcut_records"]).read_text())
    kept = [r for r in records if r["combo"]["split_strategy"] in scfg["strategies"]]
    per_label = [[float(r["thresholds"][j]) for r in kept] for j in range(len(LABEL_ORDER))]
    return np.asarray([float(np.median(v)) for v in per_label]), per_label


def feature_rows(units: list[dict[str, Any]], columns: list[str], fspec: dict[str, Any]) -> np.ndarray:
    X = np.asarray([[float(u[c]) for c in columns] for u in units], dtype=np.float64)
    if fspec.get("transform") == "signed_log1p":
        X = np.sign(X) * np.log1p(np.abs(X))
    return X


def pmd_units(violations: dict[str, list[dict[str, Any]]], rules: list[str], root: Path, level: str,
              spans: dict[str, list[tuple[int, int]]]) -> tuple[set[tuple], int]:
    """Distinct units of the level flagged by any of the rules, and LawOfDemeter violations outside them."""
    kinds, level_rules = (METHOD_KINDS, METHOD_RULES) if level == "method" else (CLASS_KINDS, CLASS_RULES)
    keys: set[tuple] = set()
    outside = 0
    for file, vs in violations.items():
        rel = "/" + str(Path(file).relative_to(root))
        for v in vs:
            if v["rule"] not in rules:
                continue
            if v["rule"] == "LawOfDemeter":
                hit = [s for s in spans.get(rel, ()) if s[0] <= v["beginline"] <= s[1]]
                keys.update(("span", rel, *s) for s in hit)
                outside += not hit
            elif (v["kind"] in kinds) if v["kind"] else (v["rule"] in level_rules):
                keys.add(("declaration", rel, v["beginline"], v["class"]))
    return keys, outside


def rates(count: float, kloc: float, units: int) -> dict[str, Any]:
    return {"alerts": int(count), "per_kloc": count / kloc if kloc else None,
            "per_10_kloc": 10.0 * count / kloc if kloc else None,
            "per_100_units": 100.0 * count / units if units else None}


def quantiles(values: list[float], qs: tuple[float, ...] = (0.5, 0.95)) -> dict[str, float | None]:
    if not values:
        return {f"p{int(q * 100)}": None for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(values, q)) for q in qs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    scfg, pcfg = cfg["scan"], cfg["pmd"]
    label_levels = cfg["label_levels"]
    out = resolve(scfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    marker = str(scfg["test_marker"])

    model, scaler, columns, fspec, mlcq_idx, mlcq_y = refit_size_logreg(cfg)
    thresholds, per_label_thr = median_thresholds(scfg)
    print(f"[scan] size logreg refit on {len(mlcq_idx)} MLCQ samples; median kfold thresholds "
          f"{dict(zip(LABEL_ORDER, thresholds.round(3).tolist()))}", flush=True)

    # MLCQ reference: prevalence and alert rate of the refit model at the level of each smell
    with resolve(cfg["effort_aware"]["sample_meta"]).open() as fh:
        mlcq_level = {int(r["json_index"]): r["level"] for r in csv.DictReader(fh)}
    table, _ = load_feature_set(fspec)
    mlcq_prob = positive_proba(model, scaler.transform(np.stack([table[int(i)] for i in mlcq_idx])))
    phase3 = int(scfg["phase3_lines"])
    with resolve(cfg["effort_aware"]["size_features"]).open() as fh:
        key = cfg["effort_aware"]["size_key"]
        mlcq_lines = {int(r[key]): float(r["loc_physical"]) for r in csv.DictReader(fh)}
    long_enough = np.array([mlcq_lines[int(i)] > phase3 for i in mlcq_idx])
    mlcq_ref = {}
    for j, label in enumerate(LABEL_ORDER):
        mask = np.array([mlcq_level[int(i)] == label_levels[label] for i in mlcq_idx])
        alert = mlcq_prob[:, j] >= thresholds[j]
        big = mask & long_enough
        mlcq_ref[label] = {"level": label_levels[label], "units": int(mask.sum()),
                           "positives_per_100_units": 100.0 * float(mlcq_y[mask, j].mean()),
                           "refit_alerts_per_100_units": 100.0 * float(alert[mask].mean()),
                           "phase3_units": int(big.sum()),
                           "positives_per_100_phase3_units": 100.0 * float(mlcq_y[big, j].mean()),
                           "refit_alerts_per_100_phase3_units": 100.0 * float(alert[big].mean())}

    chosen, n_candidates = choose_snapshots(scfg)
    print(f"[scan] {len(chosen)} snapshots drawn from {n_candidates} (strata by number of Java files)", flush=True)
    binary = ensure_pmd(pcfg)
    rules = list(pcfg["rules"])
    ruleset = write_ruleset(out / "default_ruleset.xml", str(pcfg["rules_category"]), rules, {})

    unit_rows: list[dict[str, Any]] = []
    projects = []
    totals: dict[str, Counter] = defaultdict(Counter)
    file_latency: list[float] = []
    failures: Counter = Counter()
    failed_examples: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=int(scfg["workers"])) as pool:
        for snap in chosen:
            root = snap["dir"]
            rels = sorted("/" + str(p.relative_to(root)) for p in java_files(root))
            tasks = [(snap["snapshot"], str(root), rel, str(scfg["wrapper_prefix"]), str(scfg["wrapper_suffix"]))
                     for rel in rels]
            started = time.perf_counter()
            results = list(pool.map(scan_file, tasks, chunksize=int(scfg["chunksize"])))
            units = [u for r in results for u in r["units"]]
            predicted = time.perf_counter()
            probs = (positive_proba(model, scaler.transform(feature_rows(units, columns, fspec))) if units
                     else np.zeros((0, len(LABEL_ORDER))))
            predict_sec = time.perf_counter() - predicted
            scan_sec = time.perf_counter() - started
            per_unit = predict_sec / len(units) if units else 0.0
            file_latency.extend(r["sec"] + per_unit * len(r["units"]) for r in results)
            for r in results:
                failures["files"] += 1
                failures["read_or_parse_exception"] += int(bool(r["failed"]))
                failures["not_utf8"] += r["decode_error"]
                failures["tree_sitter_error_in_file"] += r["parse_error"]
                if r["failed"] and len(failed_examples) < 5:
                    failed_examples.append({"file": f"{snap['snapshot']}{r['file']}", "error": r["failed"]})
            failures["units"] += len(units)
            failures["units_with_parse_error"] += sum(u["has_parse_error"] for u in units)

            pmd_out = out / "pmd" / snap["snapshot"]
            pmd_out.mkdir(parents=True, exist_ok=True)
            timing = run_pmd(binary, [root / rel.lstrip("/") for rel in rels], ruleset, pmd_out / "default.xml",
                             int(pcfg["threads"]), reuse=bool(pcfg.get("reuse_reports", False)))
            violations, errors = parse_report(pmd_out / "default.xml")
            failures["pmd_processing_errors"] += len(errors)
            root_resolved = root.resolve()
            spans: dict[str, dict[str, list[tuple[int, int]]]] = {"method": defaultdict(list),
                                                                   "class": defaultdict(list)}
            for u in units:
                spans[u["level"]][u["file"]].append((u["start_line"], u["end_line"]))

            proj: dict[str, Any] = {"snapshot": snap["snapshot"], "stratum": snap["stratum"],
                                    "stratum_java_files": snap["stratum_java_files"], "java_files": len(rels),
                                    "scan_sec": scan_sec, "predict_sec": predict_sec,
                                    "pmd": {k: timing.get(k) for k in ("wall_sec", "exit_code", "rule_errors",
                                                                      "reused")},
                                    "pmd_processing_errors": len(errors),
                                    "tree_sitter_files_with_errors": int(sum(r["parse_error"] for r in results)),
                                    "failed_files": int(sum(bool(r["failed"]) for r in results)),
                                    "parts": {}}
            loc_of = {r["file"]: r["loc_nonblank"] for r in results}
            for part, keep in (("production", lambda f: marker not in f), ("test", lambda f: marker in f),
                               ("all", lambda f: True)):
                files = [f for f in rels if keep(f)]
                kloc = sum(loc_of[f] for f in files) / 1000.0
                sel = [k for k, u in enumerate(units) if keep(u["file"])]
                counts: dict[str, Any] = {"files": len(files), "kloc": kloc,
                                          "methods": sum(units[k]["level"] == "method" for k in sel),
                                          "classes": sum(units[k]["level"] == "class" for k in sel),
                                          "size_logreg": {}, "size_logreg_phase3_units": {}, "pmd": {},
                                          "pmd_violations_per_kloc": {}}
                pmd_part = {f: vs for f, vs in violations.items()
                            if keep("/" + str(Path(f).relative_to(root_resolved)))}
                for j, label in enumerate(LABEL_ORDER):
                    lv = [k for k in sel if units[k]["level"] == label_levels[label]]
                    big = [k for k in lv if units[k]["loc_physical"] > int(scfg["phase3_lines"])]
                    flagged = sum(probs[k, j] >= thresholds[j] for k in lv) if lv else 0
                    flagged_big = sum(probs[k, j] >= thresholds[j] for k in big) if big else 0
                    counts["size_logreg"][label] = rates(flagged, kloc, len(lv))
                    counts["size_logreg_phase3_units"][label] = {**rates(flagged_big, kloc, len(big)),
                                                                 "units": len(big)}
                    pmd_flagged, outside = pmd_units(pmd_part, list(scfg["pmd_alerts"][label]), root_resolved,
                                                     label_levels[label], spans[label_levels[label]])
                    counts["pmd"][label] = {**rates(len(pmd_flagged), kloc, len(lv)),
                                            "rules": list(scfg["pmd_alerts"][label]),
                                            "violations_outside_units": outside}
                    totals[part][f"size_logreg|{label}"] += flagged
                    totals[part][f"size_logreg_phase3_units|{label}"] += flagged_big
                    totals[part][f"units_phase3|{label}"] += len(big)
                    totals[part][f"pmd|{label}"] += len(pmd_flagged)
                    totals[part][f"pmd_outside|{label}"] += outside
                    totals[part][f"units|{label}"] += len(lv)
                for rule in rules:
                    n = sum(1 for vs in pmd_part.values() for v in vs if v["rule"] == rule)
                    counts["pmd_violations_per_kloc"][rule] = n / kloc if kloc else None
                    totals[part][f"violations|{rule}"] += n
                totals[part]["kloc_x1000"] += int(round(kloc * 1000))
                totals[part]["files"] += len(files)
                proj["parts"][part] = counts
            kloc_all = proj["parts"]["all"]["kloc"]
            proj["scan_sec_per_kloc"] = scan_sec / kloc_all if kloc_all else None
            proj["pmd_sec_per_kloc"] = timing["wall_sec"] / kloc_all if kloc_all else None
            projects.append(proj)
            for k, u in enumerate(units):
                unit_rows.append({**{c: u[c] for c in ("project", "file", "level", "start_line", "end_line", "name",
                                                         "loc_physical", "loc_nonblank", "has_parse_error")},
                                  "is_test": int(marker in u["file"]),
                                  **{f"prob_{lab}": round(float(probs[k, j]), 6) for j, lab in enumerate(LABEL_ORDER)},
                                  **{f"alert_{lab}": int(probs[k, j] >= thresholds[j]
                                                         and u["level"] == label_levels[lab])
                                     for j, lab in enumerate(LABEL_ORDER)}})
            a = proj["parts"]["all"]
            print(f"[scan] {snap['snapshot']}: {len(rels)} files, {a['kloc']:.1f} KLOC, {a['methods']} methods, "
                  f"{a['classes']} classes; size logreg alerts per 10 KLOC "
                  f"{ {k[3:]: round(float(v['per_10_kloc']), 1) for k, v in a['size_logreg'].items()} }; PMD "
                  f"{ {k[3:]: round(v['per_10_kloc'], 1) for k, v in a['pmd'].items()} }; scan "
                  f"{proj['scan_sec_per_kloc']:.3f} s/KLOC, PMD {proj['pmd_sec_per_kloc']:.3f} s/KLOC"
                  f"{' (PMD report reused)' if timing.get('reused') else ''}", flush=True)

    summary: dict[str, Any] = {}
    for part, tot in totals.items():
        kloc = tot["kloc_x1000"] / 1000.0
        part_sum: dict[str, Any] = {"files": tot["files"], "kloc": kloc, "size_logreg": {},
                                    "size_logreg_phase3_units": {}, "pmd": {}, "pmd_violations_per_kloc": {}}
        for label in LABEL_ORDER:
            n_units = tot[f"units|{label}"]
            part_sum["size_logreg"][label] = rates(tot[f"size_logreg|{label}"], kloc, n_units)
            part_sum["size_logreg_phase3_units"][label] = {
                **rates(tot[f"size_logreg_phase3_units|{label}"], kloc, tot[f"units_phase3|{label}"]),
                "units": tot[f"units_phase3|{label}"]}
            part_sum["pmd"][label] = {**rates(tot[f"pmd|{label}"], kloc, n_units),
                                      "violations_outside_units": tot[f"pmd_outside|{label}"]}
            part_sum["units_" + label_levels[label]] = n_units
            for det in ("size_logreg", "pmd"):
                per_proj = [p["parts"][part][det][label]["per_10_kloc"] for p in projects
                            if p["parts"][part]["kloc"] > 0]
                part_sum[det][label]["per_10_kloc_over_projects"] = {
                    "median": float(np.median(per_proj)) if per_proj else None,
                    "min": float(np.min(per_proj)) if per_proj else None,
                    "max": float(np.max(per_proj)) if per_proj else None}
        for rule in rules:
            part_sum["pmd_violations_per_kloc"][rule] = tot[f"violations|{rule}"] / kloc if kloc else None
        summary[part] = part_sum
    kloc_all = summary["all"]["kloc"]
    scan_sec = sum(p["scan_sec"] for p in projects)
    pmd_sec = sum(p["pmd"]["wall_sec"] for p in projects)
    summary["timing"] = {
        "scan_sec": scan_sec, "scan_sec_per_kloc": scan_sec / kloc_all, "scan_workers": int(scfg["workers"]),
        "file_latency_sec": {**quantiles(file_latency), "files": len(file_latency)},
        "scan_sec_per_kloc_over_projects": quantiles([p["scan_sec_per_kloc"] for p in projects]),
        "pmd_sec": pmd_sec, "pmd_sec_per_kloc": pmd_sec / kloc_all, "pmd_threads": int(pcfg["threads"]),
        "pmd_sec_per_kloc_over_projects": quantiles([p["pmd_sec_per_kloc"] for p in projects]),
        "pmd_reports_reused": int(sum(bool(p["pmd"].get("reused")) for p in projects)),
        "note": "file latency = tree-sitter parse, unit extraction and 14 size counts of one file in a worker "
                "process plus its share of the batched logistic regression; scan = the same for a whole project "
                "with the worker pool; PMD = one JVM per project, default pass, start-up included; the machine "
                "was shared with other jobs, so the times are upper bounds",
    }
    summary["extraction"] = {**dict(failures), "failed_examples": failed_examples}

    for part in ("production", "test", "all"):
        s = summary[part]
        print(f"[scan] {part}: {s['files']} files, {s['kloc']:.1f} KLOC", flush=True)
        for label in LABEL_ORDER:
            a, b, p = s["size_logreg"][label], s["size_logreg_phase3_units"][label], s["pmd"][label]
            r = mlcq_ref[label]
            print(f"  {label:16s} size logreg {a['alerts']} alerts = {a['per_10_kloc']:.1f} per 10 KLOC = "
                  f"{a['per_100_units']:.1f} per 100 {label_levels[label]} units (over {phase3} lines "
                  f"{b['per_100_units']:.1f}); PMD {p['alerts']} = {p['per_10_kloc']:.1f} per 10 KLOC = "
                  f"{p['per_100_units']:.1f} per 100 ({p['violations_outside_units']} violations outside units); "
                  f"MLCQ positives {r['positives_per_100_units']:.1f} per 100 (over {phase3} lines "
                  f"{r['positives_per_100_phase3_units']:.1f}), refit alerts on MLCQ "
                  f"{r['refit_alerts_per_100_units']:.1f} per 100 (over {phase3} lines "
                  f"{r['refit_alerts_per_100_phase3_units']:.1f})", flush=True)
    print(f"[scan] timing {summary['timing']}", flush=True)
    print(f"[scan] extraction {summary['extraction']}", flush=True)

    with (out / "units.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(unit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(unit_rows)
    update_report(resolve(cfg["report"]), {"scan": {
        "nature": "workload simulation on unannotated projects: alert counts, latency and extraction failures; "
                  "an alert on an unannotated unit is not a false positive, so no precision is claimed",
        "selection": {"candidates": n_candidates, "n_projects": len(chosen), "seed": scfg["seed"],
                      "strata": "equal-size strata of the snapshots sorted by number of Java files"},
        "detector": {"run": scfg["shortcut_run"], "refit_samples": int(len(mlcq_idx)),
                     "thresholds": dict(zip(LABEL_ORDER, thresholds.tolist())),
                     "threshold_rule": f"median of the {len(per_label_thr[0])} {scfg['strategies']} "
                                       "val-tuned thresholds",
                     "size_columns": [c for c in SIZE_COLUMNS if c in columns]},
        "definitions": {
            "pmd_alert": "a declaration of the level of the smell reported by one of its rules in the default "
                         "pass (the unit kind of the message must be of the level: class-level NcssCount and "
                         "CyclomaticComplexity are not long method alerts); LawOfDemeter flags every method whose "
                         "line range holds a violation, as for the MLCQ samples; violations_outside_units counts "
                         "the LawOfDemeter violations outside any method (field initializers, static blocks)",
            "phase3_units": f"units longer than {phase3} physical lines (the filter of MLCQ phase 3); compare "
                            "the scan alert rate with the alert rate of the same model on the MLCQ units of the "
                            "same length, not with the MLCQ prevalence, which is not the prevalence of the "
                            "scanned code",
        },
        "mlcq_reference": mlcq_ref,
        "summary": summary,
        "projects": projects,
        "units_csv": rel_path(out / "units.csv"),
    }})
    print(f"[scan] wrote {resolve(cfg['report'])} and {out}", flush=True)


if __name__ == "__main__":
    main()
