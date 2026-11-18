from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
from collections import Counter
from pathlib import Path

from check_mlcq_availability import ROOT, SoftwareHeritage, request, sample_key

SWH = "https://archive.softwareheritage.org/api/1"


def slice_lines(text: str, start: int, end: int) -> str:
    """Same slicing as the original extractor: lines start..end, both included."""
    return "\n".join(text.splitlines()[start - 1:end])


def fetch_github(repo: str, sha: str, path: str) -> str | None:
    url = f"https://raw.githubusercontent.com/{repo}/{sha}/" + urllib.parse.quote(path.lstrip("/"))
    status, body, _ = request(url)
    return body.decode("utf-8", "replace") if status == 200 else None


def fetch_swh(swh: SoftwareHeritage, sha: str, path: str) -> str | None:
    status, body = swh._get(f"/revision/{sha}/directory/{urllib.parse.quote(path.lstrip('/'))}/")
    if status != 200:
        return None
    entry = json.loads(body)
    target = entry["content"]["checksums"]["sha1_git"] if "content" in entry else entry.get("target")
    status, body = swh._get(f"/content/sha1_git:{target}/raw/")
    return body.decode("utf-8", "replace") if status == 200 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the complete MLCQ dataset (4,770 snippets) with the code of every snippet.")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "MLCQCodeSmellSamples.csv")
    parser.add_argument("--detect-json", type=Path, default=ROOT / "data" / "MLCQCodeSmellSamples.json")
    parser.add_argument("--missing", type=Path, default=ROOT / "artifacts" / "mlcq_availability" / "missing_samples.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "mlcq_complete")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8"), delimiter=";"))
    used = json.loads(args.detect_json.read_text())
    code = {}
    for x in used:
        code.setdefault(sample_key(x["repo_url"], x["commit_hash"], x["file_path"], x["start_line"], x["end_line"]),
                        x["code_snippet"])
    missing = {m["sample_id"]: m for m in csv.DictReader(args.missing.open())}

    # Download the code of the missing samples once per sample
    swh = SoftwareHeritage(os.environ.get("SWH_TOKEN"))
    samples = {}
    for r in rows:
        samples.setdefault(r["sample_id"], r)
    provenance = {}
    todo = [s for s in samples.values() if s["sample_id"] in missing]
    for k, s in enumerate(todo, 1):
        m = missing[s["sample_id"]]
        key = sample_key(s["repository"], s["commit_hash"], s["path"], s["start_line"], s["end_line"])
        if m["recovered_on_github"] == "True":
            text, source = fetch_github(m["new_location"], s["commit_hash"], s["path"]), f"github:{m['new_location']}"
            time.sleep(0.15)
        else:
            text, source = fetch_swh(swh, s["commit_hash"], s["path"]), "software_heritage"
            if not swh.headers:
                time.sleep(60)  # anonymous limit: 120 requests/hour, 2 requests per file
        if text is None:
            print(f"  [{k}/{len(todo)}] FAILED {s['sample_id']} {source}", flush=True)
            continue
        code[key] = slice_lines(text, int(s["start_line"]), int(s["end_line"]))
        provenance[s["sample_id"]] = source
        if k % 25 == 0 or source == "software_heritage":
            print(f"  [{k}/{len(todo)}] {s['sample_id']} <- {source}", flush=True)

    out_rows, absent = [], []
    for r in rows:
        key = sample_key(r["repository"], r["commit_hash"], r["path"], r["start_line"], r["end_line"])
        if key not in code:
            absent.append(r["sample_id"])
            continue
        out_rows.append({"repo_url": r["repository"], "commit_hash": r["commit_hash"], "file_path": r["path"],
                         "start_line": int(r["start_line"]), "end_line": int(r["end_line"]),
                         "code_snippet": code[key], "smell": r["smell"], "severity": r["severity"]})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "MLCQCodeSmellSamples.json").write_text(json.dumps(out_rows, indent=4))
    with (args.out / "provenance.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "source"])
        for sid in sorted(samples, key=int):
            writer.writerow([sid, provenance.get(sid, "original_extraction" if sid not in missing else "absent")])

    # Checks: the old part is unchanged and every review is present
    old_rows = [x for x in out_rows if sample_key(x["repo_url"], x["commit_hash"], x["file_path"],
                                                   x["start_line"], x["end_line"]) not in
                {sample_key(samples[i]["repository"], samples[i]["commit_hash"], samples[i]["path"],
                            samples[i]["start_line"], samples[i]["end_line"]) for i in missing}]
    same_old = old_rows == [{k: x[k] for k in out_rows[0]} for x in used]
    print("== Checks")
    print(f"  reviews written: {len(out_rows)} of {len(rows)}; samples absent: {len(set(absent))}")
    print(f"  samples: {len({sample_key(x['repo_url'], x['commit_hash'], x['file_path'], x['start_line'], x['end_line']) for x in out_rows})}")
    print(f"  the 13,456 old reviews are identical to data/MLCQCodeSmellSamples.json, in the same order: {same_old}")
    print(f"  sources: {dict(Counter(v.split(':')[0] for v in provenance.values()))}")
    print(f"  empty snippets: {sum(not x['code_snippet'].strip() for x in out_rows)}")
    print(f"== Written to {args.out}")


if __name__ == "__main__":
    main()
