"""MLCQ prevalence before and after the selective cross-check.

Compares the first review of each (sample, smell), by sampling phase, with the
final label. Writes prevalence, transition and phase CSV files and summary.json
to out_dir.

Usage:
    python scripts/labels/mlcq_phase_prevalence.py --config config/experiments/mlcq_phase_prevalence.yml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402

# quotes from the EASE 2020 preprint used by the analysis
PAPER_QUOTES = [
    {"where": "EASE 2020, section 3.3",
     "text": "Developer is free to skip any sample that he or she is uncertain of."},
    {"where": "EASE 2020, section 3.3",
     "text": "Developer can choose whether she or he wants to assess the severity of both smells on a given, "
             "class or method, level or only one of them."},
    {"where": "EASE 2020, section 3.3",
     "text": "In the first two phases, we sampled from 678278 classes and 5101141 functions from 785 projects "
             "(7 projects could not be analysed), the third one - from 552750 classes and 2297722 functions "
             "(due to filtering out less than 4 line samples) from 785 projects while in the last phase we only "
             "performed crosscheck on already reviewed samples"},
    {"where": "EASE 2020, section 3.3, phase (1)",
     "text": "The first 2175 samples (date range: March 27, 2019 - April 2, 2019) were selected when there was a "
             "defect in randomisation part of selection query, and only 1 sample from each hundred could have been "
             "selected. Nevertheless, they were still selected randomly, but from a smaller population"},
    {"where": "EASE 2020, section 3.3, phase (2)",
     "text": "The next 2648 samples (date range: April 3, 2019 - April 12, 2019) were selected randomly from all "
             "available."},
    {"where": "EASE 2020, section 3.3, phase (3)",
     "text": "The next 4801 samples (date range: April 13, 2019 - July 25, 2019) were selected randomly from all "
             "samples longer than 4 lines, including opening and closing braces."},
    {"where": "EASE 2020, section 3.3, phase (4)",
     "text": "The last 5115 samples (date range: July 26, 2019 - September 13, 2019) were selected to perform a "
             "crosscheck - only from samples already tagged with severity higher than 'none'. Samples selected from "
             "crosscheck were those that had only 1 review or 2 reviews with different severities."},
    {"where": "EASE 2020, section 3.3",
     "text": "Then it turned out that the amount of \"none\" samples is higher that we have estimated, thus the need "
             "to focus the effort of professional developers on valuable samples that can reasonably constitute "
             "code smells to balance the data set."},
    {"where": "EASE 2020, section 3.3",
     "text": "the first selection step in the first three phases will be selecting whether we are looking for a "
             "class sample or method sample (and both were selected with equal probability)."},
    {"where": "EASE 2020, section 6",
     "text": "We did conduct an additional crosscheck in the end of research to verify the samples and assigned "
             "smell severities. However, we only did this with samples that were initially tagged with severity "
             "above none - therefore it is possible that some samples which should have severity above none do not "
             "have it."},
    {"where": "EASE 2020, section 6",
     "text": "Samples selected for review were chosen in four separate phases (described in detail in Section 3.3). "
             "This means that the population of source code entities used for selection was not uniform during "
             "the whole research."},
]


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


# p-value columns keep 6 significant digits instead of 6 decimals (which would print most of them as 0.0)
P_COLUMNS = {"mcnemar_exact_p", "binomial_p_vs_paper_pool"}


def fmt(v: Any, significant: bool = False) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return float(f"{v:.6g}") if significant else round(v, 6)
    return v


def jsonable(obj: Any) -> Any:
    """NaN and infinities become null."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(row.get(k), k in P_COLUMNS) for k in fields})
    tmp.replace(path)
    return rel(path)


def wilson(k: int, n: int, level: float) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(0.5 + level / 2))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return min(max(0.0, centre - half), p), max(min(1.0, centre + half), p)


def rate(k: int, n: int, level: float) -> dict[str, Any]:
    lo, hi = wilson(k, n, level)
    return {"n": n, "k": k, "rate": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi}


class Data:
    """Zenodo reviews grouped by (sample, smell) unit, normalized samples linked to them."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        icfg, lcfg = cfg["inputs"], cfg["labels"]
        self.order = list(lcfg["severity_order"])
        self.sev = {s: i for i, s in enumerate(self.order)}
        self.positive = {self.sev[s] for s in lcfg["positive"]}
        self.smells = list(lcfg["smell_names"])
        self.shorts = list(lcfg["short_names"])
        if len(self.smells) != len(LABEL_ORDER):
            raise SystemExit("labels.smell_names must have one entry per LABEL_ORDER label")
        self.smell_level = {s: str(lcfg["smell_level"][s]) for s in self.smells}
        self.type_level = {str(k): str(v) for k, v in lcfg["type_level"].items()}
        self.phases = [dict(p) for p in cfg["phases"]]
        self.phase_names = [str(p["name"]) for p in self.phases]
        self.crosscheck = str(cfg["crosscheck_phase"])
        self.min_long = int(cfg["length"]["min_lines_long"])
        by_time = str(cfg.get("first_review_order", "time")) == "time"

        with resolve(icfg["zenodo_reviews_csv"]).open(newline="") as fh:
            raw = list(csv.DictReader(fh, delimiter=str(icfg.get("zenodo_separator", ";"))))
        self.reviews: list[dict[str, Any]] = []
        for r in raw:
            sev, smell = r["severity"].strip(), r["smell"].strip()
            if sev not in self.sev or smell not in self.smells:
                raise SystemExit(f"review {r['id']}: unknown severity {sev!r} or smell {smell!r}")
            ts = r["review_timestamp"].strip()
            self.reviews.append({
                "id": int(r["id"]), "reviewer": int(r["reviewer_id"]), "sample": int(r["sample_id"]),
                "s": self.smells.index(smell), "sev": self.sev[sev], "pos": int(self.sev[sev] in self.positive),
                "ts": ts, "phase": self.phase_of(ts), "level": self.type_level[r["type"].strip()],
                "span": int(r["end_line"]) - int(r["start_line"]) + 1,
                "key": (r["repository"].strip(), r["commit_hash"].strip(), r["path"].strip(),
                        int(r["start_line"]), int(r["end_line"])),
                "raw_key": (r["repository"].strip(), r["commit_hash"].strip(), r["path"].strip(),
                            int(r["start_line"]), int(r["end_line"]), smell, sev),
            })
        sort_key = (lambda rv: (rv["ts"], rv["id"])) if by_time else (lambda rv: rv["id"])
        self.reviews.sort(key=sort_key)

        # samples, their level, span and sessions (the reviews one reviewer gave to one sample)
        self.sample_level: dict[int, str] = {}
        self.sample_span: dict[int, int] = {}
        self.sample_key: dict[int, tuple] = {}
        first_session_reviewer: dict[int, int] = {}
        for rv in self.reviews:
            for d, v in ((self.sample_level, rv["level"]), (self.sample_span, rv["span"]),
                         (self.sample_key, rv["key"])):
                if d.setdefault(rv["sample"], v) != v:
                    raise SystemExit(f"Zenodo sample {rv['sample']} has two levels, spans or locations")
            first_session_reviewer.setdefault(rv["sample"], rv["reviewer"])
        self.first_session_reviewer = first_session_reviewer
        self.first_session_phase = {}
        for rv in self.reviews:
            self.first_session_phase.setdefault(rv["sample"], rv["phase"])

        # units: (sample, smell index) -> reviews in the chosen order
        self.units: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for rv in self.reviews:
            self.units[(rv["sample"], rv["s"])].append(rv)
        # the same review chosen as first under time order and under id order?
        self.first_differs_time_vs_id = sum(
            1 for revs in self.units.values()
            if min(revs, key=lambda rv: (rv["ts"], rv["id"]))["id"] != min(revs, key=lambda rv: rv["id"])["id"])
        self.first_pos_differs_time_vs_id = sum(
            1 for revs in self.units.values()
            if min(revs, key=lambda rv: (rv["ts"], rv["id"]))["pos"] != min(revs, key=lambda rv: rv["id"])["pos"])

        # the review-level JSON used elsewhere in the repository: same rows as the release, in id order?
        raw_json = json.loads(resolve(icfg["raw_json"]).read_text())
        rj = [(e["repo_url"], e["commit_hash"], e["file_path"], int(e["start_line"]), int(e["end_line"]),
               e["smell"], e["severity"]) for e in raw_json]
        rj_keys = Counter(rj)
        by_id = sorted(self.reviews, key=lambda rv: rv["id"])
        kept = [rv["raw_key"] for rv in by_id if rj_keys[rv["raw_key"]] > 0]
        self.raw_json_check = {
            "raw_json_rows": len(rj), "zenodo_rows_with_a_raw_json_match": len(kept),
            "raw_json_in_zenodo_id_order": kept == rj,
            "note": "the raw JSON has no reviewer_id or review_timestamp, so the first review is taken from the "
                    "Zenodo release; the raw JSON is the release restricted to the samples kept in the normalized "
                    "data when raw_json_in_zenodo_id_order is true"}

        # normalized samples and their link
        entries = json.loads(resolve(icfg["normalized_json"]).read_text())
        sample_of_key = {k: s for s, k in self.sample_key.items()}
        with resolve(icfg["sample_meta_csv"]).open(newline="") as fh:
            meta = {int(r["json_index"]): r for r in csv.DictReader(fh)}
        self.norm: list[dict[str, Any]] = []
        level_mismatch = 0
        for i, e in enumerate(entries):
            key = (e["repo_url"], e["commit_hash"], e["file_path"], int(e["start_line"]), int(e["end_line"]))
            sample = sample_of_key.get(key, -1)
            y = e["y"]
            if isinstance(y, str):
                y = json.loads(y.lower())
            level = meta[i]["level"]
            if sample >= 0 and self.sample_level[sample] != level:
                level_mismatch += 1
            self.norm.append({"json_index": i, "sample": sample, "level": level,
                              "y": [int(bool(v)) for v in y[:len(self.smells)]]})
        self.norm_unlinked = sum(1 for x in self.norm if x["sample"] < 0)
        self.norm_level_mismatch = level_mismatch
        if self.norm_unlinked:
            print(f"[prevalence] warning: {self.norm_unlinked} normalized samples have no Zenodo sample", flush=True)

    def phase_of(self, ts: str) -> str:
        day = ts[:10]
        for p in self.phases:
            if str(p["start"]) <= day <= str(p["end"]):
                return str(p["name"])
        raise SystemExit(f"timestamp {ts} is outside every phase")

    def upper_median(self, revs: list[dict[str, Any]]) -> int:
        """Label rule of scripts/NormalizeFromJson.py (--method median --rule default), every record counted."""
        if not revs:
            return 0
        sevs = sorted(rv["sev"] for rv in revs)
        return int(sevs[len(sevs) // 2] in self.positive)


def unit_records(d: Data, population: str) -> dict[int, list[dict[str, Any]]]:
    """Per smell index, one record per sample of the smell's level in the population."""
    if population == "normalized":
        samples = [(x["sample"], x["level"], x["y"]) for x in d.norm if x["sample"] >= 0]
    else:
        samples = [(s, d.sample_level[s], None) for s in sorted(d.sample_level)]
    out: dict[int, list[dict[str, Any]]] = {}
    for s, smell in enumerate(d.smells):
        # the other smell reviewed on the same level (feature envy <-> long method, blob <-> data class)
        other = next(o for o, name in enumerate(d.smells) if o != s and d.smell_level[name] == d.smell_level[smell])
        rows = []
        for sample, level, y in samples:
            if level != d.smell_level[smell]:
                continue
            revs = d.units.get((sample, s), [])
            first = revs[0] if revs else None
            other_revs = d.units.get((sample, other), [])
            reviewers = []
            for rv in revs:
                if rv["reviewer"] not in reviewers:
                    reviewers.append(rv["reviewer"])
            later = [rv for rv in revs[1:] if rv["reviewer"] != revs[0]["reviewer"]]
            rows.append({
                "sample": sample, "reviewed": first is not None,
                "first_pos": first["pos"] if first else 0,
                "first_phase": first["phase"] if first else "",
                "first_in_first_session": bool(first) and first["reviewer"] == d.first_session_reviewer[sample],
                "long": d.sample_span[sample] >= d.min_long,
                "n_records": len(revs), "n_reviewers": len(reviewers),
                "crosschecked": any(rv["phase"] == d.crosscheck for rv in later),
                "later_reviews": len(later), "later_pos": sum(rv["pos"] for rv in later),
                "y_rule": d.upper_median(revs),
                "y_stored": None if y is None else y[s],
                "other_smell": d.shorts[other],
                "other_first": ("unreviewed" if not other_revs
                                else "above_none" if other_revs[0]["pos"] else "none"),
            })
        out[s] = rows
    return out


def section_prevalence(d: Data, pops: dict[str, dict[int, list[dict[str, Any]]]], lv: float
                       ) -> list[dict[str, Any]]:
    rows = []
    for pop, units in pops.items():
        for s, short in enumerate(d.shorts):
            us = units[s]
            level = d.smell_level[d.smells[s]]
            rev = [u for u in us if u["reviewed"]]

            def add(estimate: str, subset: str, sel: list[dict[str, Any]], field: str, source: str = "") -> None:
                rows.append({"population": pop, "smell": short, "level": level, "estimate": estimate,
                             "subset": subset, "label_source": source,
                             **rate(sum(int(u[field]) for u in sel), len(sel), lv)})

            # (a) first review
            add("first_review_above_none", "reviewed_units", rev, "first_pos")
            add("first_review_above_none", "first_review_in_first_session",
                [u for u in rev if u["first_in_first_session"]], "first_pos")
            # (b) by phase of the first review, and by line span
            for ph in d.phase_names:
                sel = [u for u in rev if u["first_phase"] == ph]
                add("first_review_above_none", f"phase_{ph}", sel, "first_pos")
                add("first_review_above_none", f"phase_{ph}_long", [u for u in sel if u["long"]], "first_pos")
                add("first_review_above_none", f"phase_{ph}_short", [u for u in sel if not u["long"]], "first_pos")
            add("first_review_above_none", "phases_1_to_3",
                [u for u in rev if u["first_phase"] != d.crosscheck], "first_pos")
            # every review record of the smell (the level of the paper's table 2)
            recs = [rv for u in rev for rv in d.units[(u["sample"], s)]]
            rows.append({"population": pop, "smell": short, "level": level, "estimate": "review_records_above_none",
                         "subset": "all_records", "label_source": "",
                         **rate(sum(rv["pos"] for rv in recs), len(recs), lv)})
            # (c) final labels
            sources = ["stored", "rule"] if pop == "normalized" else ["rule"]
            for src in sources:
                field = f"y_{src}"
                add("final_label_positive", "all_samples_of_level", us, field, src)
                add("final_label_positive", "reviewed_units", rev, field, src)
                for ph in d.phase_names:
                    add("final_label_positive", f"first_review_phase_{ph}",
                        [u for u in rev if u["first_phase"] == ph], field, src)
                add("final_label_positive", "first_review_phases_1_to_3",
                    [u for u in rev if u["first_phase"] != d.crosscheck], field, src)
    return rows


def section_transitions(d: Data, pops: dict[str, dict[int, list[dict[str, Any]]]], lv: float
                        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trans, effect = [], []
    for pop, units in pops.items():
        for s, short in enumerate(d.shorts):
            us = units[s]
            rev = [u for u in us if u["reviewed"]]
            sources = ["stored", "rule"] if pop == "normalized" else ["rule"]
            for src in sources:
                field = f"y_{src}"
                groups: Counter = Counter()
                pos: Counter = Counter()
                for u in us:
                    first = "unreviewed" if not u["reviewed"] else ("above_none" if u["first_pos"] else "none")
                    nb = str(min(u["n_reviewers"], 4)) + ("+" if u["n_reviewers"] >= 4 else "")
                    k = (first, nb, int(u["crosschecked"]))
                    groups[k] += 1
                    pos[k] += int(u[field])
                for (first, nb, cc), n in sorted(groups.items()):
                    trans.append({"population": pop, "smell": short, "label_source": src,
                                  "first_review": first, "reviewers_of_unit": nb,
                                  "has_crosscheck_phase_review": cc, "units": n,
                                  "final_positive": pos[(first, nb, cc)],
                                  "final_negative": n - pos[(first, nb, cc)]})
                fp = [u for u in rev if u["first_pos"]]
                fn = [u for u in rev if not u["first_pos"]]
                fp_again = [u for u in fp if u["later_reviews"] > 0]
                fn_again = [u for u in fn if u["later_reviews"] > 0]
                down = sum(1 for u in fp if not u[field])
                up = sum(1 for u in fn if u[field])
                k_first = len(fp)
                k_final = sum(int(u[field]) for u in rev)
                mcnemar = stats.binomtest(min(down, up), down + up, 0.5).pvalue if down + up else float("nan")
                later_fn = sum(u["later_reviews"] for u in fn_again)
                later_fn_pos = sum(u["later_pos"] for u in fn_again)
                later_fp = sum(u["later_reviews"] for u in fp_again)
                later_fp_pos = sum(u["later_pos"] for u in fp_again)
                effect.append({
                    "population": pop, "smell": short, "level": d.smell_level[d.smells[s]], "label_source": src,
                    "samples_of_level": len(us), "reviewed_units": len(rev),
                    "first_above_none": k_first, "first_rate": k_first / len(rev),
                    "final_positive_reviewed_units": k_final, "final_rate": k_final / len(rev),
                    "final_over_first": k_final / k_first if k_first else float("nan"),
                    "first_above_none_reviewed_again": len(fp_again),
                    "first_above_none_crosschecked": sum(u["crosschecked"] for u in fp),
                    "first_above_none_single_reviewer": sum(u["n_reviewers"] == 1 for u in fp),
                    "first_above_none_final_positive": k_first - down,
                    "first_above_none_final_negative": down,
                    "share_of_first_positives_kept": (k_first - down) / k_first if k_first else float("nan"),
                    "first_none": len(fn), "first_none_reviewed_again": len(fn_again),
                    "first_none_crosschecked": sum(u["crosschecked"] for u in fn),
                    "first_none_final_positive": up,
                    "later_reviews_of_first_none_units": later_fn,
                    "later_reviews_of_first_none_units_above_none": later_fn_pos,
                    "later_rate_on_first_none_units": later_fn_pos / later_fn if later_fn else float("nan"),
                    "later_reviews_of_first_above_none_units": later_fp,
                    "later_reviews_of_first_above_none_units_above_none": later_fp_pos,
                    "later_rate_on_first_above_none_units": later_fp_pos / later_fp if later_fp else float("nan"),
                    "mcnemar_exact_p": mcnemar,
                })
    return trans, effect


def section_later_by_other(d: Data, pops: dict[str, dict[int, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Later reviews of a smell by its first rating and by the first rating of the other smell of the sample."""
    rows = []
    for pop, units in pops.items():
        for s, short in enumerate(d.shorts):
            groups: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
            for u in units[s]:
                if not u["reviewed"] or not u["later_reviews"]:
                    continue
                g = groups[("above_none" if u["first_pos"] else "none", u["other_first"])]
                g[0] += 1
                g[1] += u["later_reviews"]
                g[2] += u["later_pos"]
            other = units[s][0]["other_smell"] if units[s] else ""
            for (first, other_first), (n, k, p) in sorted(groups.items()):
                rows.append({"population": pop, "smell": short, "other_smell": other, "first_review": first,
                             "other_first_review": other_first, "units_reviewed_again": n,
                             "later_reviews": k, "later_reviews_above_none": p,
                             "later_rate": p / k if k else float("nan")})
    return rows


def section_phases(d: Data, pops: dict[str, dict[int, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows = []
    norm_samples = {x["sample"] for x in d.norm if x["sample"] >= 0}
    for p in d.phases:
        name = str(p["name"])
        recs = [rv for rv in d.reviews if rv["phase"] == name]
        row = {"phase": name, "start": p["start"], "end": p["end"], "description": p.get("description", ""),
               "reviews": len(recs), "paper_reviews": p.get("paper_reviews"),
               "difference_to_paper": len(recs) - int(p["paper_reviews"]),
               "samples_first_reviewed": sum(1 for v in d.first_session_phase.values() if v == name),
               "normalized_samples_first_reviewed": sum(1 for s, v in d.first_session_phase.items()
                                                        if v == name and s in norm_samples)}
        for s, short in enumerate(d.shorts):
            row[f"normalized_units_first_reviewed_{short}"] = sum(
                1 for u in pops["normalized"][s] if u["first_phase"] == name)
        rows.append(row)
    return rows


def section_length(d: Data, cfg: dict[str, Any], lv: float) -> list[dict[str, Any]]:
    pool = cfg["length"]["paper_pool"]
    rows = []
    for ph in d.phase_names:
        if ph == d.crosscheck:
            continue
        for level in ("method", "class"):
            spans = [d.sample_span[s] for s, v in d.first_session_phase.items()
                     if v == ph and d.sample_level[s] == level]
            if not spans:
                continue
            k = sum(x >= d.min_long for x in spans)
            expected = int(pool[level]["long"]) / int(pool[level]["all"])
            rows.append({"phase": ph, "level": level, "samples_first_reviewed": len(spans),
                         "min_span": min(spans), "long_samples": k,
                         "long_share": k / len(spans), **{f"long_share_{a}": b for a, b in
                                                         zip(("ci_low", "ci_high"), wilson(k, len(spans), lv))},
                         "paper_pool_long_share": expected,
                         "binomial_p_vs_paper_pool": stats.binomtest(k, len(spans), expected).pvalue})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/mlcq_phase_prevalence.yml"))
    args, extra = parser.parse_known_args()
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    out_dir = resolve(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    lv = float(cfg["ci_level"])

    d = Data(cfg)
    print(f"[prevalence] {len(d.reviews)} Zenodo reviews, {len(d.sample_level)} samples, "
          f"{len(d.norm)} normalized samples", flush=True)
    pops = {"normalized": unit_records(d, "normalized"), "zenodo": unit_records(d, "zenodo")}

    prevalence = section_prevalence(d, pops, lv)
    trans, effect = section_transitions(d, pops, lv)
    later_by_other = section_later_by_other(d, pops)
    phases = section_phases(d, pops)
    length = section_length(d, cfg, lv)
    files = [write_csv(out_dir / "prevalence.csv", prevalence), write_csv(out_dir / "transitions.csv", trans),
             write_csv(out_dir / "crosscheck_effect.csv", effect),
             write_csv(out_dir / "later_reviews_by_other_smell.csv", later_by_other),
             write_csv(out_dir / "phases.csv", phases), write_csv(out_dir / "length_check.csv", length)]

    # stored labels outside the smell's level (should be none)
    off_level = {short: sum(x["y"][s] for x in d.norm if x["level"] != d.smell_level[d.smells[s]])
                 for s, short in enumerate(d.shorts)}
    uniform = str(cfg["uniform_phase"])

    def pick(pop: str, short: str, estimate: str, subset: str, source: str = "") -> dict[str, Any]:
        r = next(x for x in prevalence if x["population"] == pop and x["smell"] == short
                 and x["estimate"] == estimate and x["subset"] == subset and x["label_source"] == source)
        return {k: r[k] for k in ("n", "k", "rate", "ci_low", "ci_high")}

    headline = {}
    for pop in pops:
        headline[pop] = {}
        for short in d.shorts:
            src = "stored" if pop == "normalized" else "rule"
            headline[pop][short] = {
                "a_first_review_reviewed_units": pick(pop, short, "first_review_above_none", "reviewed_units"),
                f"b_first_review_phase_{uniform}": pick(pop, short, "first_review_above_none", f"phase_{uniform}"),
                "b_first_review_phase_1": pick(pop, short, "first_review_above_none", "phase_1"),
                "b_first_review_phase_3": pick(pop, short, "first_review_above_none", "phase_3"),
                f"c_final_{src}_all_samples_of_level": pick(pop, short, "final_label_positive",
                                                            "all_samples_of_level", src),
                f"c_final_{src}_reviewed_units": pick(pop, short, "final_label_positive", "reviewed_units", src),
                f"final_{src}_units_first_reviewed_in_phase_{uniform}": pick(
                    pop, short, "final_label_positive", f"first_review_phase_{uniform}", src),
            }
    summary = {
        "config": str(args.config), "label_order": LABEL_ORDER,
        "ci": f"Wilson score interval at level {lv}",
        "first_review_rule": ("earliest review_timestamp, ties broken by id" if cfg.get("first_review_order", "time")
                              == "time" else "smallest id"),
        "units_whose_first_review_differs_between_time_and_id_order": d.first_differs_time_vs_id,
        "units_whose_first_review_positivity_differs_between_time_and_id_order": d.first_pos_differs_time_vs_id,
        "raw_json_check": d.raw_json_check,
        "zenodo_reviews": len(d.reviews), "zenodo_samples": len(d.sample_level),
        "normalized_samples": len(d.norm), "normalized_unlinked": d.norm_unlinked,
        "normalized_level_differs_from_zenodo_type": d.norm_level_mismatch,
        "stored_positives_outside_level": off_level,
        "phase_rule": "date part of review_timestamp in the inclusive date ranges of the config",
        "phases": phases,
        "length_check": length,
        "headline": headline,
        "crosscheck_effect": effect,
        "later_reviews_by_other_smell": later_by_other,
        "paper_quotes": PAPER_QUOTES,
        "notes": [
            "A unit is one (sample, smell) pair at the smell's level; units of the level with no review of the "
            "smell count as negatives in the stored labels and are left out of the first-review shares.",
            "The cross-check re-reviewed samples, not single smells: a smell first rated none can get later reviews "
            "when the other smell of the same sample was rated above none, and the upper median of two reviews is "
            "positive as soon as one of them is above none.",
            "The later-review rates of crosscheck_effect.csv pool units re-reviewed for different reasons: units "
            "first rated none were mostly re-reviewed because the other smell was first rated above none. Compare "
            "first ratings within one other_first_review group of later_reviews_by_other_smell.csv.",
            "Every rate here is an annotation prevalence. No rate is a deployment prevalence: the units come from "
            "a fixed set of GitHub Java projects drawn in phases with different populations, reviewers could skip "
            "samples without leaving a record, and the label is a reviewer opinion with minor counted as positive.",
        ],
        "files": files,
    }
    path = out_dir / "summary.json"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(summary), indent=1, ensure_ascii=False))
    tmp.replace(path)
    files.append(rel(path))
    for short in d.shorts:
        h = headline["normalized"][short]
        a, b, c = (h["a_first_review_reviewed_units"], h[f"b_first_review_phase_{uniform}"],
                   h["c_final_stored_all_samples_of_level"])
        print(f"[prevalence] {short}: first review {a['k']}/{a['n']} = {a['rate']:.3f}, "
              f"phase {uniform} {b['k']}/{b['n']} = {b['rate']:.3f}, stored {c['k']}/{c['n']} = {c['rate']:.3f}",
              flush=True)
    print(f"[prevalence] wrote {', '.join(files)}", flush=True)


if __name__ == "__main__":
    main()
