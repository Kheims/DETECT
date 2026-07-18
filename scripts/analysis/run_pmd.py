"""Score PMD 7 design rules and DesigniteJava smells against the MLCQ and DACOS labels.

PMD runs a default pass (rule decisions) and a minimum pass (metric values).
Reports the default and the validation-tuned configuration of each detector.

Usage:
    python scripts/analysis/run_pmd.py --config config/experiments/industrial.yml
    python scripts/analysis/run_pmd.py --config config/experiments/industrial.yml --datasets mlcq --tools pmd
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.evaluation.cv import iter_folds, load_folds  # noqa: E402

REPORT_NS = "{http://pmd.sourceforge.net/report/2.0.0}"
METHOD_KINDS = {"method", "constructor"}
CLASS_KINDS = {"class", "interface", "enum", "record", "annotation"}
# value of the metric reported in the message of each rule
VALUE_PATTERNS = {
    "NcssCount": r"NCSS line count of ([\d,]+)",
    "CyclomaticComplexity": r"cyclomatic complexity of ([\d,]+)",
    "CognitiveComplexity": r"cognitive complexity of ([\d,]+)",
    "NPathComplexity": r"NPath complexity of ([\d,]+)",
    "ExcessiveParameterList": r"\(([\d,]+) parameters",
    "ExcessiveImports": r"number of imports \(([\d,]+)\)",
    "CouplingBetweenObjects": r"A value of ([\d,]+)",
}
GOD_CLASS = re.compile(r"WMC=([\d,]+), ATFD=([\d,]+), TCC=([\d.,]+)(%?)")
DATA_CLASS = re.compile(r"WOC=([\d.,]+)(%?), NOPA=([\d,]+), NOAM=([\d,]+), WMC=([\d,]+)")
UNIT_KIND = re.compile(r"^The (\w+) '")
# rules whose violation sits on a method declaration or on a class declaration
METHOD_RULES = {"NcssCount", "CyclomaticComplexity", "CognitiveComplexity", "NPathComplexity",
                "ExcessiveParameterList"}
CLASS_RULES = {"GodClass", "DataClass", "TooManyMethods", "TooManyFields", "CouplingBetweenObjects",
               "NcssCount", "CyclomaticComplexity"}


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel_path(path: Path) -> str:
    """Path relative to the repository when it is inside it, absolute otherwise."""
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def update_report(path: Path, sections: dict[str, Any]) -> None:
    """Replace the given top-level sections of the shared JSON report."""
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update(sections)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(path)


def ensure_pmd(pcfg: dict[str, Any]) -> Path:
    """bin/pmd of the configured release, downloaded and unzipped when missing."""
    home = resolve(pcfg["home"])
    binary = home / "bin" / "pmd"
    if binary.exists():
        return binary
    archive = home.parent / Path(str(pcfg["url"])).name
    home.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"[pmd] downloading {pcfg['url']}", flush=True)
        urllib.request.urlretrieve(str(pcfg["url"]), archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != str(pcfg["sha256"]):
        raise SystemExit(f"{archive}: sha256 {digest} differs from the config")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(home.parent)
    binary.chmod(0o755)
    return binary


def write_ruleset(path: Path, category: str, rules: list[str], properties: dict[str, dict[str, Any]]) -> Path:
    lines = ['<?xml version="1.0"?>',
             '<ruleset name="industrial" xmlns="http://pmd.sourceforge.net/ruleset/2.0.0"',
             '    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
             '    xsi:schemaLocation="http://pmd.sourceforge.net/ruleset/2.0.0 '
             'https://pmd.sourceforge.io/ruleset_2_0_0.xsd">',
             "  <description>scripts/analysis/run_pmd.py</description>"]
    for rule in rules:
        props = properties.get(rule) or {}
        if not props:
            lines.append(f'  <rule ref="{category}/{rule}"/>')
            continue
        lines.append(f'  <rule ref="{category}/{rule}"><properties>')
        lines.extend(f'    <property name="{k}" value="{v}"/>' for k, v in props.items())
        lines.append("  </properties></rule>")
    lines.append("</ruleset>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def report_complete(report: Path) -> bool:
    """True when the XML report ends with its closing tag (the run was not interrupted)."""
    if not report.exists() or report.stat().st_size < 10:
        return False
    with report.open("rb") as fh:
        fh.seek(-64, 2) if report.stat().st_size > 64 else fh.seek(0)
        return b"</pmd>" in fh.read()


def run_pmd(binary: Path, files: list[Path], ruleset: Path, report: Path, threads: int,
            reuse: bool = False) -> dict[str, Any]:
    """One PMD run on a file list. Returns its wall time and exit code, reusing a complete earlier report."""
    file_list = report.with_suffix(".files.txt")
    listing = "\n".join(str(f) for f in files) + "\n"
    timing_file = report.with_suffix(".timing.json")
    ruleset_copy = report.with_suffix(".ruleset_used.xml")
    if (reuse and report_complete(report) and timing_file.exists() and file_list.exists()
            and file_list.read_text() == listing and ruleset_copy.exists()
            and ruleset_copy.read_text() == ruleset.read_text()):
        timing = json.loads(timing_file.read_text())
        timing["reused"] = True
        return timing
    file_list.write_text(listing)
    cmd = [str(binary), "check", "--file-list", str(file_list), "-R", str(ruleset), "-f", "xml",
           "-r", str(report), "--no-cache", "--no-progress", "--no-fail-on-violation", "--no-fail-on-error",
           "-t", str(threads)]
    if report.exists():
        report.unlink()
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - started
    # exit code 1: a rule crashed on some file, the rest of the report is written
    if proc.returncode not in (0, 1) or not report.exists():
        raise SystemExit(f"pmd failed ({proc.returncode}): {proc.stderr[-2000:]}")
    rule_errors = [line for line in proc.stderr.splitlines() if line.startswith("[ERROR]")]
    timing = {"wall_sec": wall, "files": len(files), "ms_per_file": 1000.0 * wall / max(len(files), 1),
              "threads": threads, "exit_code": proc.returncode, "rule_errors": len(rule_errors),
              "rule_errors_sample": [line[:300] for line in rule_errors[:5]]}
    timing_file.write_text(json.dumps(timing, indent=1))
    ruleset_copy.write_text(ruleset.read_text())
    return timing


def parse_report(report: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    """Violations grouped by resolved file path and the processing errors."""
    root = ET.parse(report).getroot()
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in root.iter(f"{REPORT_NS}file"):
        name = str(Path(f.get("name", "")).resolve())
        for v in f.iter(f"{REPORT_NS}violation"):
            message = " ".join((v.text or "").split())
            kind = UNIT_KIND.match(message)
            by_file[name].append({
                "rule": v.get("rule"), "beginline": int(v.get("beginline")), "endline": int(v.get("endline")),
                "begincolumn": int(v.get("begincolumn")), "class": v.get("class", ""),
                "method": v.get("method", ""), "message": message, "kind": kind.group(1) if kind else "",
            })
    errors = [{"file": e.get("filename", ""), "msg": (e.get("msg") or "")[:300]}
              for e in root.iter(f"{REPORT_NS}error")]
    return by_file, errors


def number(text: str) -> float:
    return float(text.replace(",", ""))


def parse_values(v: dict[str, Any]) -> dict[str, float]:
    """Metric values carried by one violation message."""
    rule, msg = v["rule"], v["message"]
    if rule in VALUE_PATTERNS:
        m = re.search(VALUE_PATTERNS[rule], msg)
        return {"value": number(m.group(1))} if m else {}
    if rule == "GodClass":
        m = GOD_CLASS.search(msg)
        if m:
            tcc = number(m.group(3)) / (100.0 if m.group(4) else 1.0)
            return {"WMC": number(m.group(1)), "ATFD": number(m.group(2)), "TCC": tcc}
    if rule == "DataClass":
        m = DATA_CLASS.search(msg)
        if m:
            woc = number(m.group(1)) / (100.0 if m.group(2) else 1.0)
            nopa, noam = number(m.group(3)), number(m.group(4))
            return {"WOC": woc, "NOPA": nopa, "NOAM": noam, "WMC": number(m.group(5)), "NOPA+NOAM": nopa + noam}
    return {}


def unit_anchor(minimum: list[dict[str, Any]], start: int, end: int, level: str) -> dict[str, Any] | None:
    """Outermost method or class declaration reported by the minimum pass in [start, end]."""
    kinds = METHOD_KINDS if level == "method" else CLASS_KINDS
    found = [v for v in minimum if v["rule"] in ("NcssCount", "CyclomaticComplexity") and v["kind"] in kinds
             and start <= v["beginline"] <= end]
    return min(found, key=lambda v: (v["beginline"], v["begincolumn"])) if found else None


def method_violations(violations: list[dict[str, Any]], rule: str, lines: set[int],
                      parameter_offset: int) -> list[dict[str, Any]]:
    """Violations of a method rule reported on the name line of one of the given declarations."""
    if rule == "ExcessiveParameterList":
        return [v for v in violations if v["rule"] == rule
                and any(n <= v["beginline"] <= n + parameter_offset for n in lines)]
    return [v for v in violations if v["rule"] == rule and v["kind"] in METHOD_KINDS and v["beginline"] in lines]


def class_violation(violations: list[dict[str, Any]], rule: str, anchor: dict[str, Any]) -> dict[str, Any] | None:
    """Violation of a class rule that belongs to the anchor declaration (same line and type name)."""
    same = [v for v in violations if v["rule"] == rule and v["beginline"] == anchor["beginline"]
            and v["class"] == anchor["class"] and not v["method"]]
    return same[0] if same else None


def sample_features(default: list[dict[str, Any]], minimum: list[dict[str, Any]], start: int, end: int,
                    level: str, rules: list[str], parameter_offset: int,
                    method_lines: set[int] | None = None) -> dict[str, float]:
    """Binary decision of every rule (default pass) and the metric values of the sample."""
    feats: dict[str, float] = {"covered": 0.0}
    for rule in rules:
        feats[rule] = 0.0
    lod = [v for v in default if v["rule"] == "LawOfDemeter" and start <= v["beginline"] <= end]
    feats["LawOfDemeter"] = float(bool(lod))
    feats["LawOfDemeter.count"] = float(len(lod))
    if level == "method":
        if method_lines is None:
            anchor = unit_anchor(minimum, start, end, level)
            method_lines = {anchor["beginline"]} if anchor is not None else set()
        if not method_violations(minimum, "NcssCount", method_lines, parameter_offset):
            return feats
        feats["covered"] = 1.0
        for rule in rules:
            if rule not in METHOD_RULES:
                continue
            feats[rule] = float(bool(method_violations(default, rule, method_lines, parameter_offset)))
            values = [parse_values(v).get("value")
                      for v in method_violations(minimum, rule, method_lines, parameter_offset)]
            values = [v for v in values if v is not None]
            feats[f"{rule}.value"] = max(values) if values else 0.0
        return feats
    anchor = unit_anchor(minimum, start, end, level)
    if anchor is None:
        return feats
    feats["covered"] = 1.0
    for rule in rules:
        if rule == "ExcessiveImports":
            d = [v for v in default if v["rule"] == rule]
            m = [v for v in minimum if v["rule"] == rule]
            feats[rule] = float(bool(d))
            feats[f"{rule}.value"] = parse_values(m[0]).get("value", 0.0) if m else 0.0
            continue
        if rule not in CLASS_RULES:
            continue
        d = class_violation(default, rule, anchor)
        m = class_violation(minimum, rule, anchor)
        feats[rule] = float(d is not None)
        for k, val in (parse_values(d).items() if d is not None else ()):
            feats[f"{rule}.{k}"] = val
        if m is not None and "value" in parse_values(m):
            feats[f"{rule}.value"] = parse_values(m)["value"]
        elif f"{rule}.value" not in feats:
            feats[f"{rule}.value"] = 0.0
    return feats


def detector_columns(det: dict[str, Any], feats: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Binary decision and continuous score of a detector for every sample."""
    if "any" in det:
        flags = np.array([[f.get(r, 0.0) for r in det["any"]] for f in feats])
        return (flags.sum(axis=1) > 0).astype(np.int64), flags.sum(axis=1)
    binary = np.array([f.get(det["binary"], 0.0) for f in feats]).astype(np.int64)
    score = np.array([f.get(det["score"], 0.0) for f in feats], dtype=np.float64)
    return binary, score


def tune_on_scores(y: np.ndarray, score: np.ndarray, steps: int) -> float:
    """Threshold on an unbounded score: max F1 over quantiles of the validation scores."""
    best_f1, best_t = -1.0, float("inf")
    for t in np.unique(np.quantile(score, np.linspace(0.0, 1.0, steps), method="lower")):
        f1 = f1_score(y, (score >= t).astype(np.int64), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t


def binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "mcc": float(matthews_corrcoef(y, pred)),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "flag_rate": float(pred.mean()),
        "roc_auc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan"),
    }


def seed_mean(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    out: dict[str, Any] = {"folds": len(rows), "seeds": len(by_seed)}
    for k in keys:
        per_seed = [float(np.nanmean([r[k] for r in rs])) for rs in by_seed.values()
                    if any(np.isfinite(r[k]) for r in rs)]
        out[k] = float(np.mean(per_seed)) if per_seed else None
        out[f"{k}_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None
    return out


def evaluate(name: str, detectors: dict[str, list[dict[str, Any]]], label_order: list[str],
             labels: dict[int, np.ndarray], level_of: dict[int, str], label_levels: dict[str, str],
             feats: dict[int, dict[str, float]], folds: dict[str, Any], strategies: list[str],
             steps: int) -> dict[str, Any]:
    """Metrics of the default configuration and of the validation-tuned configuration of each detector."""
    keys = ["mcc", "precision", "recall", "flag_rate", "roc_auc"]
    out: dict[str, Any] = {}
    fold_list = list(iter_folds(folds, strategies))
    for label, dets in detectors.items():
        j = label_order.index(label)
        level = label_levels[label]
        pool = sorted(i for i in feats if level_of[i] == level)
        y_all = np.array([labels[i][j] for i in pool])
        out[label] = {"level": level, "samples": len(pool), "positives": int(y_all.sum()),
                      "covered": int(sum(feats[i]["covered"] for i in pool)), "detectors": {}}
        for det in dets:
            binary, score = detector_columns(det, [feats[i] for i in pool])
            col = dict(zip(pool, range(len(pool))))
            tunable = det.get("property") is not None
            comparison = str(det.get("comparison", ">="))
            if comparison not in (">=", ">"):
                raise SystemExit(f"{det['name']}: comparison must be >= or >")
            shift = 1.0 if comparison == ">" else 0.0
            res: dict[str, Any] = {"property": det.get("property"), "comparison": comparison,
                                   "default": {"all_samples": binary_metrics(y_all, binary, score)},
                                   "tuned": {} if tunable else None}
            if det.get("note"):
                res["note"] = det["note"]
            for strategy in strategies:
                rows_default, rows_tuned = [], []
                for fold in (f for f in fold_list if f.strategy == strategy):
                    val = [col[int(i)] for i in fold.val if int(i) in col]
                    test = [col[int(i)] for i in fold.test if int(i) in col]
                    rows_default.append({"seed": fold.seed,
                                         **binary_metrics(y_all[test], binary[test], score[test])})
                    if tunable:
                        t = tune_on_scores(y_all[val], score[val], steps)
                        pred = (score[test] >= t).astype(np.int64)
                        rows_tuned.append({"seed": fold.seed, "threshold": t,
                                           **binary_metrics(y_all[test], pred, score[test])})
                res["default"][f"{strategy}_test"] = seed_mean(rows_default, keys)
                if tunable:
                    thr = np.array([r["threshold"] for r in rows_tuned])
                    res["tuned"][f"{strategy}_test"] = {
                        **seed_mean(rows_tuned, keys), "threshold_median": float(np.median(thr)),
                        "threshold_min": float(np.min(thr)), "threshold_max": float(np.max(thr)),
                        "property_value_median": float(np.median(thr - shift)),
                        "property_value_min": float(np.min(thr - shift)),
                        "property_value_max": float(np.max(thr - shift))}
            out[label]["detectors"][det["name"]] = res
            a, k = res["default"]["all_samples"], res["default"].get("kfold_test", {})
            line = (f"[{name}] {label} {det['name']}: default, all samples MCC {fmt(a['mcc'])} "
                    f"P {fmt(a['precision'])} R {fmt(a['recall'])} flagged {fmt(a['flag_rate'])} "
                    f"AUC {fmt(a['roc_auc'])} | default, kfold test MCC {fmt(k.get('mcc'))} "
                    f"P {fmt(k.get('precision'))} R {fmt(k.get('recall'))} AUC {fmt(k.get('roc_auc'))}")
            if tunable and "kfold_test" in res["tuned"]:
                t = res["tuned"]["kfold_test"]
                line += (f" | tuned, kfold test MCC {fmt(t['mcc'])} P {fmt(t['precision'])} R {fmt(t['recall'])} "
                         f"flagged {fmt(t['flag_rate'])} at {det['property']} median {t['property_value_median']:g} "
                         f"[{t['property_value_min']:g}, {t['property_value_max']:g}]")
                if shift:
                    line += (f" (fires above the property: score >= {t['threshold_median']:g} "
                             f"[{t['threshold_min']:g}, {t['threshold_max']:g}])")
            print(line, flush=True)
    return out


def fmt(value: Any) -> str:
    return "nan" if value is None or not np.isfinite(float(value)) else f"{float(value):.3f}"


def consistency(feats: dict[int, dict[str, float]], rules: list[str], level_of: dict[int, str]) -> dict[str, Any]:
    """Per level, range of the minimum-pass values with and without a default-pass violation."""
    out: dict[str, Any] = {}
    for level in sorted(set(level_of.values())):
        units = [f for i, f in feats.items() if level_of[i] == level and f["covered"]]
        for rule in rules:
            flagged = [f[f"{rule}.value"] for f in units if f.get(rule) and f"{rule}.value" in f]
            clean = [f[f"{rule}.value"] for f in units if not f.get(rule) and f"{rule}.value" in f]
            if flagged or clean:
                out.setdefault(level, {})[rule] = {
                    "flagged": len(flagged), "min_flagged": min(flagged) if flagged else None,
                    "not_flagged": len(clean), "max_not_flagged": max(clean) if clean else None}
    return out


def snapshot_dir(root: Path, repo_url: str, commit: str) -> Path:
    tail = repo_url.rstrip("/").removesuffix(".git")
    tail = tail.split(":", 1)[1] if tail.startswith("git@") else tail.split("://", 1)[-1].split("/", 1)[-1]
    return root / f"{'__'.join(tail.split('/')[-2:])}@{commit}"


def two_passes(binary: Path, pcfg: dict[str, Any], files: list[Path], rules: list[str], out: Path,
               tag: str) -> tuple[dict, dict, dict[str, Any]]:
    """Default and minimum passes on the same files."""
    category = str(pcfg["rules_category"])
    minimum_props = {r: p for r, p in pcfg["minimum_properties"].items() if r in rules}
    timing, parsed = {}, {}
    for pass_name, props, pass_rules in (("default", {}, rules), ("minimum", minimum_props, list(minimum_props))):
        ruleset = write_ruleset(out / f"{tag}_{pass_name}_ruleset.xml", category, pass_rules, props)
        report = out / f"{tag}_{pass_name}.xml"
        timing[pass_name] = run_pmd(binary, files, ruleset, report, int(pcfg["threads"]),
                                    reuse=bool(pcfg.get("reuse_reports", False)))
        parsed[pass_name] = parse_report(report)
        n_viol = sum(len(v) for v in parsed[pass_name][0].values())
        timing[pass_name].update({"violations": n_viol, "processing_errors": len(parsed[pass_name][1])})
        print(f"[pmd] {tag} {pass_name} pass: {len(files)} files in {timing[pass_name]['wall_sec']:.1f} s "
              f"({timing[pass_name]['ms_per_file']:.1f} ms per file, {pcfg['threads']} threads"
              f"{', timing of the earlier run, report reused' if timing[pass_name].get('reused') else ''}), "
              f"{n_viol} violations, {len(parsed[pass_name][1])} processing errors", flush=True)
    timing["errors_sample"] = parsed["default"][1][:5]
    timing["error_files"] = sorted({str(Path(e["file"]).resolve()) for e in parsed["minimum"][1]})
    return parsed["default"][0], parsed["minimum"][0], timing


def write_features(path: Path, feats: dict[int, dict[str, float]], extra: dict[int, dict[str, Any]]) -> None:
    columns = sorted({k for f in feats.values() for k in f})
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        head = list(next(iter(extra.values())).keys())
        writer.writerow(["json_index", *head, *columns])
        for i in sorted(feats):
            writer.writerow([i, *extra[i].values(), *[feats[i].get(c, "") for c in columns]])


def run_mlcq(cfg: dict[str, Any], binary: Path, out: Path) -> dict[str, Any]:
    pcfg = cfg["pmd"]
    mcfg = pcfg["mlcq"]
    entries = json.loads(resolve(mcfg["input_json"]).read_text())
    folds = load_folds(resolve(mcfg["folds"]))
    label_order = list(folds["meta"]["label_order"])
    indexed = [int(i) for i in folds["meta"]["json_index"]]
    with resolve(mcfg["sample_meta"]).open() as fh:
        level_of = {int(r["json_index"]): r["level"] for r in csv.DictReader(fh)}
    root = resolve(mcfg["snapshots"])
    path_of: dict[int, Path] = {}
    for i in indexed:
        e = entries[i]
        p = (snapshot_dir(root, e["repo_url"], e["commit_hash"]) / e["file_path"].lstrip("/")).resolve()
        if p.exists():
            path_of[i] = p
    files = sorted(set(path_of.values()))
    print(f"[pmd] mlcq: {len(path_of)} of {len(indexed)} samples have their snapshot file, "
          f"{len(files)} distinct files", flush=True)
    rules = list(pcfg["rules"])
    default, minimum, timing = two_passes(binary, pcfg, files, rules, out, "mlcq")

    labels, feats, extra = {}, {}, {}
    for i in indexed:
        y = entries[i]["y"]
        y = json.loads(y.lower()) if isinstance(y, str) else y
        labels[i] = np.array([int(bool(v)) for v in y[:len(label_order)]])
        e = entries[i]
        key = str(path_of[i]) if i in path_of else ""
        feats[i] = sample_features(default.get(key, []), minimum.get(key, []), int(e["start_line"]),
                                   int(e["end_line"]), level_of[i], rules, int(pcfg["parameter_list_offset"]))
        extra[i] = {"level": level_of[i], "file": rel_path(path_of[i]) if i in path_of else "",
                    "start_line": e["start_line"], "end_line": e["end_line"]}
    write_features(out / "mlcq_pmd_features.csv", feats, extra)
    coverage = {lv: {"samples": sum(1 for i in indexed if level_of[i] == lv),
                     "with_file": sum(1 for i in indexed if level_of[i] == lv and i in path_of),
                     "declaration_found": int(sum(feats[i]["covered"] for i in indexed if level_of[i] == lv))}
                for lv in ("method", "class")}
    # missing declaration: PMD error on the file or suppressed violations
    suppress = re.compile(r'@SuppressWarnings\([^)]*"(all|PMD)[^"]*"')
    error_files = set(timing.pop("error_files"))
    missing = [i for i in indexed if i in path_of and not feats[i]["covered"]]
    coverage["missing_explained"] = {
        "samples": len(missing),
        "in_files_with_processing_error": sum(str(path_of[i]) in error_files for i in missing),
        "in_files_with_suppress_warnings_all_or_pmd": sum(
            str(path_of[i]) not in error_files and bool(suppress.search(path_of[i].read_text(errors="replace")))
            for i in missing)}
    coverage["evaluated_population"] = {
        "note": "every sample of the level is evaluated (the population of the learned models); a sample "
                "without snapshot file or without declaration is not flagged and scores 0",
        **{lv: {"samples": coverage[lv]["samples"],
                "scored_as_not_flagged": coverage[lv]["samples"] - coverage[lv]["declaration_found"]}
           for lv in ("method", "class")}}
    print(f"[pmd] mlcq coverage {coverage}", flush=True)
    results = evaluate("mlcq", mcfg["detectors"], label_order, labels, level_of, cfg["label_levels"], feats,
                       folds, list(mcfg["strategies"]), int(pcfg["threshold_steps"]))
    latency = single_file_latency(binary, pcfg, files, out)
    return {"timing": timing, "single_file_latency": latency, "coverage": coverage,
            "consistency": consistency(feats, rules, level_of), "results": results,
            "features_csv": rel_path(out / "mlcq_pmd_features.csv")}


def single_file_latency(binary: Path, pcfg: dict[str, Any], files: list[Path], out: Path) -> dict[str, Any]:
    """Wall time of PMD alone on single files (one JVM per file, default pass)."""
    lcfg = pcfg["mlcq"]["latency"]
    rng = np.random.default_rng(int(lcfg["seed"]))
    picked = [files[k] for k in sorted(rng.choice(len(files), size=min(int(lcfg["files"]), len(files)),
                                                  replace=False))]
    ruleset = write_ruleset(out / "latency_ruleset.xml", str(pcfg["rules_category"]), list(pcfg["rules"]), {})
    work = resolve(pcfg["work_dir"]) / "latency"
    work.mkdir(parents=True, exist_ok=True)
    walls, lines = [], []
    for k, f in enumerate(picked):
        t = run_pmd(binary, [f], ruleset, work / f"file{k}.xml", int(lcfg["threads"]))
        walls.append(t["wall_sec"])
        lines.append(sum(1 for line in f.read_text(errors="replace").split("\n") if line.strip()))
    out_lat = {"files": len(picked), "threads": int(lcfg["threads"]), "seed": int(lcfg["seed"]),
               "wall_sec_p50": float(np.quantile(walls, 0.5)), "wall_sec_p95": float(np.quantile(walls, 0.95)),
               "wall_sec_min": float(np.min(walls)), "wall_sec_max": float(np.max(walls)),
               "loc_nonblank_p50": float(np.quantile(lines, 0.5)),
               "note": "includes JVM start-up; the machine was shared with other jobs"}
    print(f"[pmd] single-file latency over {len(picked)} MLCQ files: p50 {out_lat['wall_sec_p50']:.2f} s, "
          f"p95 {out_lat['wall_sec_p95']:.2f} s (min {out_lat['wall_sec_min']:.2f}, max {out_lat['wall_sec_max']:.2f})",
          flush=True)
    return out_lat


def top_level_method_lines(source: str) -> set[int]:
    """Name lines (1-based) of the methods declared directly in the wrapper class."""
    from scripts.data.extract_size_features import METHOD_TYPES, _parser

    tree = _parser().parse(source.encode("utf-8"))
    lines = set()
    for decl in tree.root_node.children:
        body = decl.child_by_field_name("body")
        for node in (body.children if body is not None else []):
            name = node.child_by_field_name("name") if node.type in METHOD_TYPES else None
            if name is not None:
                lines.add(name.start_point[0] + 1)
    return lines


def run_dacos(cfg: dict[str, Any], binary: Path, out: Path) -> dict[str, Any]:
    pcfg = cfg["pmd"]
    dcfg = pcfg["dacos"]
    entries = json.loads(resolve(dcfg["input_json"]).read_text())
    folds = load_folds(resolve(dcfg["folds"]))
    label_order = list(folds["meta"]["label_order"])
    work = resolve(pcfg["work_dir"]) / "dacos"
    work.mkdir(parents=True, exist_ok=True)
    path_of, offset = {}, str(dcfg["wrapper_prefix"]).count("\n")
    for i, e in enumerate(entries):
        p = (work / f"S{i}.java").resolve()
        p.write_text(str(dcfg["wrapper_prefix"]) + e["code_snippet"].rstrip("\n") + str(dcfg["wrapper_suffix"]))
        path_of[i] = p
    rules = list(dcfg["rules"])
    default, minimum, timing = two_passes(binary, pcfg, sorted(path_of.values()), rules, out, "dacos")
    labels, feats, extra, level_of = {}, {}, {}, {}
    for i, e in enumerate(entries):
        labels[i] = np.array([int(bool(v)) for v in e["y"][:len(label_order)]])
        key = str(path_of[i])
        n_lines = e["code_snippet"].rstrip("\n").count("\n") + 1
        lines = top_level_method_lines(path_of[i].read_text())
        feats[i] = sample_features(default.get(key, []), minimum.get(key, []), offset + 1, offset + n_lines,
                                   "method", rules, int(pcfg["parameter_list_offset"]), method_lines=lines)
        level_of[i] = "method"
        extra[i] = {"project": e["project"], "file": rel_path(path_of[i]), "top_level_methods": len(lines)}
    write_features(out / "dacos_pmd_features.csv", feats, extra)
    covered = int(sum(f["covered"] for f in feats.values()))
    several = int(sum(x["top_level_methods"] > 1 for x in extra.values()))
    print(f"[pmd] dacos: method declaration found for {covered} of {len(entries)} snippets; {several} snippets "
          f"hold more than one top-level method (rule fires on any, value = max)", flush=True)
    levels = {label: "method" for label in label_order}
    results = evaluate("dacos", dcfg["detectors"], label_order, labels, level_of, levels, feats, folds,
                       list(dcfg["strategies"]), int(pcfg["threshold_steps"]))
    prevalence = {label: float(np.mean([labels[i][j] for i in labels])) for j, label in enumerate(label_order)}
    return {"timing": timing, "coverage": {"snippets": len(entries), "declaration_found": covered,
                                           "snippets_with_several_top_level_methods": several},
            "prevalence": prevalence, "consistency": consistency(feats, rules, level_of), "results": results,
            "features_csv": rel_path(out / "dacos_pmd_features.csv")}


def designite_batch(task: tuple[str, str, str, int]) -> dict[str, Any]:
    """DesigniteJava on one input folder."""
    jar, in_dir, out_dir, timeout = task
    started = time.perf_counter()
    try:
        code = subprocess.run(["java", "-jar", jar, "-i", in_dir, "-o", out_dir], capture_output=True,
                              text=True, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = -1
    return {"out": out_dir, "exit_code": code, "wall_sec": time.perf_counter() - started}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def designite_value(text: Any) -> float:
    """Metric value with DesigniteJava's sentinels (NaN, inf, -1) mapped to 0."""
    try:
        v = float(text)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not np.isfinite(v) or v == -1.0 else v


def designite_features(cfg: dict[str, Any], tag: str, sources: dict[int, str]) -> tuple[dict[int, dict[str, float]],
                                                                                        dict[str, Any]]:
    """Smells (1 when reported) and metrics of the type S<index> of every sample."""
    dcfg = cfg["designite"]
    work = resolve(dcfg["work_dir"]) / tag
    shutil.rmtree(work, ignore_errors=True)
    ids = sorted(sources)
    size = int(dcfg["batch_size"])
    tasks = []
    for b in range(0, len(ids), size):
        in_dir = work / f"batch_{b // size:04d}" / "input"
        in_dir.mkdir(parents=True)
        for i in ids[b:b + size]:
            (in_dir / f"S{i}.java").write_text(sources[i])
        tasks.append((str(resolve(dcfg["jar"])), str(in_dir), str(in_dir.parent / "output"), int(dcfg["timeout_sec"])))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(dcfg["workers"])) as pool:
        runs = list(pool.map(designite_batch, tasks))
    wall = time.perf_counter() - started

    def sample_of(type_name: str) -> int | None:
        name = type_name.strip()
        return int(name[1:]) if name[:1] == "S" and name[1:].isdigit() and int(name[1:]) in sources else None

    feats: dict[int, dict[str, float]] = {i: {"covered": 0.0} for i in ids}
    for run in runs:
        out_dir = Path(run["out"])
        for row in read_rows(out_dir / "typeMetrics.csv"):
            i = sample_of(row["Type Name"])
            if i is not None:
                feats[i]["covered"] = 1.0
                feats[i].update({f"type_{m}": designite_value(row.get(m)) for m in dcfg["type_metrics"]})
        for row in read_rows(out_dir / "methodMetrics.csv"):
            i = sample_of(row["Type Name"])
            if i is not None:
                for m in dcfg["method_metrics"]:
                    feats[i][f"method_{m}_max"] = max(feats[i].get(f"method_{m}_max", 0.0),
                                                      designite_value(row.get(m)))
        for name in ("implementationCodeSmells.csv", "designCodeSmells.csv"):
            for row in read_rows(out_dir / name):
                i = sample_of(row["Type Name"])
                if i is not None:
                    feats[i][row["Code Smell"].strip()] = 1.0
    timing = {"wall_sec": wall, "snippets": len(ids), "ms_per_snippet": 1000.0 * wall / max(len(ids), 1),
              "batches": len(runs), "workers": int(dcfg["workers"]),
              "failed_batches": int(sum(r["exit_code"] != 0 for r in runs))}
    print(f"[designite] {tag}: {len(ids)} snippets in {wall:.1f} s ({timing['ms_per_snippet']:.1f} ms per snippet, "
          f"{len(runs)} batches on {dcfg['workers']} workers, {timing['failed_batches']} failed), type found for "
          f"{int(sum(f['covered'] for f in feats.values()))}", flush=True)
    return feats, timing


def run_designite(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    """DesigniteJava on the snippets of a dataset, scored like PMD."""
    from scripts.extract_designite_metrics import prepare_snippet

    dcfg = cfg["designite"]
    scfg = dcfg[dataset]
    entries = json.loads(resolve(scfg["input_json"]).read_text())
    folds = load_folds(resolve(scfg["folds"]))
    label_order = list(folds["meta"]["label_order"])
    if dataset == "mlcq":
        indexed = [int(i) for i in folds["meta"]["json_index"]]
        with resolve(scfg["sample_meta"]).open() as fh:
            level_of = {int(r["json_index"]): r["level"] for r in csv.DictReader(fh)}
        label_levels = cfg["label_levels"]
    else:
        indexed = list(range(len(entries)))
        level_of = {i: "method" for i in indexed}
        label_levels = {label: "method" for label in label_order}
    # method snippets go into a class S<index>, class snippets are renamed to S<index>
    prefix, suffix = str(dcfg["method_wrapper_prefix"]), str(dcfg["method_wrapper_suffix"])
    sources = {i: (prefix.replace("{name}", f"S{i}") + entries[i]["code_snippet"].rstrip() + suffix
                   if level_of[i] == "method" else prepare_snippet(entries[i]["code_snippet"], f"S{i}") + "\n")
               for i in indexed}
    feats, timing = designite_features(cfg, dataset, sources)
    labels = {}
    for i in indexed:
        y = entries[i]["y"]
        y = json.loads(y.lower()) if isinstance(y, str) else y
        labels[i] = np.array([int(bool(v)) for v in y[:len(label_order)]])
    out = resolve(dcfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_features(out / f"{dataset}_designite_features.csv", feats, {i: {"level": level_of[i]} for i in indexed})
    smells = Counter(k for f in feats.values() for k, v in f.items()
                     if v == 1.0 and k != "covered" and not k.startswith(("type_", "method_")))
    results = evaluate(f"designite {dataset}", scfg["detectors"], label_order, labels, level_of, label_levels,
                       feats, folds, list(scfg["strategies"]), int(dcfg["threshold_steps"]))
    return {"timing": timing, "smells_reported": dict(smells.most_common()), "results": results,
            "coverage": {lv: {"samples": sum(1 for i in indexed if level_of[i] == lv),
                              "type_found": int(sum(feats[i]["covered"] for i in indexed if level_of[i] == lv))}
                         for lv in sorted(set(level_of.values()))},
            "features_csv": rel_path(out / f"{dataset}_designite_features.csv")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["mlcq", "dacos"], choices=["mlcq", "dacos"])
    parser.add_argument("--tools", nargs="+", default=["pmd", "designite"], choices=["pmd", "designite"])
    args, extra = parser.parse_known_args()
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    sections: dict[str, Any] = {}
    if "pmd" in args.tools:
        pcfg = cfg["pmd"]
        binary = ensure_pmd(pcfg)
        version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip().splitlines()
        version_line = next((line for line in version if line.startswith("PMD ")), "")
        print(f"[pmd] {version_line}", flush=True)
        out = resolve(pcfg["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        common = {"pmd_version": version_line, "rules_category": pcfg["rules_category"],
                  "minimum_properties": pcfg["minimum_properties"], "auxclasspath": "empty",
                  "configurations": {"default": "decision of each rule with its shipped properties",
                                     "tuned": "threshold t on the continuous score tuned on the validation part "
                                              "of each fold (max F1), flag when score >= t; the rule property "
                                              "named in property gives the same decision at property_value = t "
                                              "for a rule that fires at or above its property (comparison >=) "
                                              "and t - 1 for a rule that fires strictly above it (comparison >, "
                                              "integer scores)"},
                  "no_feature_envy_rule": "PMD has no Feature Envy rule; LawOfDemeter is reported in its place"}
        if "mlcq" in args.datasets:
            sections["pmd_mlcq"] = {**common, **run_mlcq(cfg, binary, out)}
        if "dacos" in args.datasets:
            sections["pmd_dacos"] = {**common, **run_dacos(cfg, binary, out)}
    if "designite" in args.tools:
        dcfg = cfg["designite"]
        common = {"tool": dcfg["source"], "input": dcfg["input"],
                  "no_feature_envy_rule": "DesigniteJava has no Feature Envy smell"}
        for dataset in args.datasets:
            sections[f"designite_{dataset}"] = {**common, **run_designite(cfg, dataset)}
    update_report(resolve(cfg["report"]), sections)
    print(f"[industrial] wrote {sorted(sections)} to {resolve(cfg['report'])}", flush=True)


if __name__ == "__main__":
    main()
