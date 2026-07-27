"""Compare the line attributions of the detectors, snippet by snippet.

Reads the GCN, BiLSTM and Qwen line attributions and the access lines. Writes
localization.csv, agreement.csv, line_kinds.csv and cards.json to artifacts/xai/compare/.

Usage:
    python scripts/xai/xai_compare.py [--cards 289,20,1766,334,3047,4030,1049,2298,50]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.xai import common  # noqa: E402

OUT = ROOT / "artifacts/xai/compare"
SOURCES = {
    "GCN (Integrated Gradients)": ("artifacts/xai/gcn/line_attributions.jsonl", "integrated_gradients"),
    "BiLSTM (Integrated Gradients)": ("artifacts/xai/bilstm/line_attributions.jsonl", "integrated_gradients"),
    "BiLSTM (attention)": ("artifacts/xai/bilstm/line_attributions.jsonl", "attention"),
    "Qwen2.5-Coder 7B (occlusion)": ("artifacts/xai/llm_occlusion/qwen7b_line_attributions.jsonl", "occlusion"),
}


def load_sources() -> dict[str, dict[tuple[int, str], dict]]:
    out = {}
    for name, (path, method) in SOURCES.items():
        p = ROOT / path
        if not p.exists():
            print(f"[compare] missing {path}, {name} skipped")
            continue
        rows = {}
        for r in common.read_rows(p):
            if r["method"] == method:
                rows[(r["json_index"], r["smell"])] = r
        if rows:
            out[name] = rows
            print(f"[compare] {name}: {len(rows)} rows")
    return out


def line_kind(line: str, first: bool) -> str:
    s = line.strip()
    if first:
        return "signature"
    if not s or set(s) <= set("{}();"):
        return "braces only"
    if s.startswith(("//", "/*", "*", "@")):
        return "comment or annotation"
    if s.startswith("return"):
        return "return"
    if re.match(r"(if|else|for|while|switch|case|do|try|catch|finally)\b", s):
        return "condition, loop or try"
    if re.match(r"(final\s+)?[A-Z][\w<>\[\], ?]*\s+\w+\s*(=|;)", s) or re.match(r"(int|long|boolean|double|float|char|byte|short)\b", s):
        return "declaration"
    if "=" in s and "==" not in s:
        return "assignment"
    if "(" in s:
        return "call"
    return "other"


def positive_mass(scores: np.ndarray) -> np.ndarray:
    pos = np.clip(scores, 0, None)
    total = pos.sum()
    return pos / total if total > 0 else np.zeros_like(pos)


def localization(src: dict, access: dict, rows: list) -> None:
    for name, data in src.items():
        for label in (1, 0):
            mass_f, base_f, hit_f, chance_f = [], [], [], []
            mass_e, base_e, hit_e, chance_e = [], [], [], []
            for (j, smell), r in data.items():
                if smell != "is_feature_envy" or r["label"] != label or j not in access:
                    continue
                a = access[j]
                n = len(r["line_scores"])
                if not a["foreign_lines"] or n < 3:
                    continue
                s = np.array(r["line_scores"], dtype=float)
                m = positive_mass(s)
                fl = [i for i in a["foreign_lines"] if i < n]
                el = [i for i in a["envied_lines"] if i < n]
                top = int(np.argmax(s))
                mass_f.append(m[fl].sum())
                base_f.append(len(fl) / n)
                hit_f.append(top in fl)
                chance_f.append(len(fl) / n)
                if el:
                    mass_e.append(m[el].sum())
                    base_e.append(len(el) / n)
                    hit_e.append(top in el)
                    chance_e.append(len(el) / n)
            if not mass_f:
                continue
            rows.append({"explainer": name, "label": label, "n": len(mass_f),
                         "mass_on_foreign_lines": float(np.mean(mass_f)), "share_of_foreign_lines": float(np.mean(base_f)),
                         "lift_foreign": float(np.mean(mass_f) / np.mean(base_f)),
                         "pointing_foreign": float(np.mean(hit_f)), "pointing_foreign_chance": float(np.mean(chance_f)),
                         "n_envied": len(mass_e), "mass_on_envied_lines": float(np.mean(mass_e)) if mass_e else None,
                         "share_of_envied_lines": float(np.mean(base_e)) if base_e else None,
                         "pointing_envied": float(np.mean(hit_e)) if hit_e else None,
                         "pointing_envied_chance": float(np.mean(chance_e)) if chance_e else None})


def agreement(src: dict, rows: list) -> None:
    names = list(src)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = src[names[a]], src[names[b]]
            for smell in common.LABELS:
                rhos, top_same = [], []
                for key in set(A) & set(B):
                    if key[1] != smell:
                        continue
                    x, y = np.array(A[key]["line_scores"], float), np.array(B[key]["line_scores"], float)
                    if len(x) != len(y) or len(x) < 4 or x.std() == 0 or y.std() == 0:
                        continue
                    rhos.append(stats.spearmanr(x, y).statistic)
                    top_same.append(int(np.argmax(x)) == int(np.argmax(y)))
                if rhos:
                    rows.append({"a": names[a], "b": names[b], "smell": smell, "pairs": len(rhos),
                                 "mean_spearman": float(np.nanmean(rhos)), "same_top_line": float(np.mean(top_same))})


def kinds(src: dict, rows: list) -> None:
    for name, data in src.items():
        acc = defaultdict(lambda: defaultdict(list))
        for (j, smell), r in data.items():
            lines = common.snippet_lines(j)
            s = np.array(r["line_scores"], float)
            if len(s) != len(lines) or len(s) < 3:
                continue
            m = positive_mass(s)
            per = defaultdict(float)
            for i, line in enumerate(lines):
                per[line_kind(line, i == 0)] += m[i]
            for k, v in per.items():
                acc[(smell, r["label"])][k].append(v)
            for k in set(line_kind(l, i == 0) for i, l in enumerate(lines)) - set(per):
                acc[(smell, r["label"])][k].append(0.0)
        for (smell, label), d in acc.items():
            n = max(len(v) for v in d.values())
            for k, v in d.items():
                rows.append({"explainer": name, "smell": smell, "label": label, "kind": k,
                             "mean_share": float(np.sum(v) / n), "snippets": n})


def cards(src: dict, access: dict, ids: list[int]) -> list:
    size = {}
    p = ROOT / "artifacts/xai/size_contributions.jsonl"
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            size[(r["json_index"], r["smell"])] = r
    out = []
    for j in ids:
        level = common.levels().get(j)
        for smell in common.LABELS:
            if common.LEVEL_OF[smell] != level:
                continue
            card = {"json_index": j, "smell": smell, "label": common.label(j, smell), "level": level,
                    "severity": common.severity(j, smell), "lines": common.snippet_lines(j), "explainers": {}}
            for name, data in src.items():
                r = data.get((j, smell))
                if r is not None:
                    card["explainers"][name] = {"score": r["score"], "line_scores": r["line_scores"]}
            if j in access:
                card["foreign_lines"] = access[j]["foreign_lines"]
                card["own_lines"] = access[j]["own_lines"]
                card["envied_type"] = access[j]["envied_type"]
            if (j, smell) in size:
                card["size_model"] = {"prob": size[(j, smell)]["prob"], "contributions": size[(j, smell)]["contributions"]}
            out.append(card)
    return out


def write(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cards", default="289,20,1766,334,3047,4030,1049,2298,50")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    src = load_sources()
    access = {}
    p = ROOT / "artifacts/xai/access_lines.jsonl"
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            access[r["json_index"]] = r
    loc, agr, kin = [], [], []
    localization(src, access, loc)
    agreement(src, agr)
    kinds(src, kin)
    write(OUT / "localization.csv", loc)
    write(OUT / "agreement.csv", agr)
    write(OUT / "line_kinds.csv", kin)
    (OUT / "cards.json").write_text(json.dumps(cards(src, access, [int(x) for x in args.cards.split(",")])))
    for r in loc:
        print(f"[loc] {r['explainer']:32s} label {r['label']} n {r['n']:4d} mass on foreign {r['mass_on_foreign_lines']:.2f} "
              f"(share {r['share_of_foreign_lines']:.2f}, lift {r['lift_foreign']:.2f}) pointing {r['pointing_foreign']:.2f} "
              f"(chance {r['pointing_foreign_chance']:.2f})")
    for r in agr:
        print(f"[agree] {r['a']} vs {r['b']} {r['smell']}: rho {r['mean_spearman']:.2f} same top {r['same_top_line']:.2f} "
              f"({r['pairs']} pairs)")
    print(f"[compare] wrote {OUT}")


if __name__ == "__main__":
    main()
