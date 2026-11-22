from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from check_mlcq_availability import CSV_NAME, ROOT, ensure_csv, owner_repo, request

RAW = "https://raw.githubusercontent.com"


def raw_urls(row: dict) -> list[str]:
    """The file at the commit of the review, then the file behind the link of the review."""
    first = f"{RAW}/{owner_repo(row['repository'])}/{row['commit_hash']}/{row['path'].lstrip('/')}"
    second = row["link"].replace("https://github.com/", RAW + "/").replace("/blob/", "/").split("#L")[0]
    return [first] + ([second] if second.startswith(RAW) and second != first else [])


def fetch_snippet(row: dict) -> str | None:
    for url in raw_urls(row):
        status, body, _ = request(urllib.parse.quote(url, safe=":/%"))
        if status == 200:
            lines = body.decode("utf-8", "replace").splitlines()
            return "\n".join(lines[int(row["start_line"]) - 1:int(row["end_line"])])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the code of each MLCQ snippet at the GitHub location of its reviews.")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / CSV_NAME)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "mlcq_extraction" / "MLCQCodeSmellSamples.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print(ensure_csv(args.csv))
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8"), delimiter=";"))
    samples = {}
    for r in rows:
        samples.setdefault(r["sample_id"], r)
    with ThreadPoolExecutor(args.workers) as pool:
        code = dict(zip(samples, pool.map(fetch_snippet, samples.values())))

    out = [{"repo_url": r["repository"], "commit_hash": r["commit_hash"], "file_path": r["path"],
            "start_line": int(r["start_line"]), "end_line": int(r["end_line"]),
            "code_snippet": code[r["sample_id"]], "smell": r["smell"], "severity": r["severity"]}
           for r in rows if code[r["sample_id"]]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=4))
    found = sum(bool(c) for c in code.values())
    print(f"{found} of {len(samples)} snippets read, {len(out)} of {len(rows)} reviews written to {args.out}")


if __name__ == "__main__":
    main()
