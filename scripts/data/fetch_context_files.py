"""Fetch the full source file that contains each MLCQ snippet, at the annotated commit.

Files come from raw.githubusercontent.com and are cached on disk.

Usage:
    uv run python scripts/data/fetch_context_files.py \
        --normalized data/MLCQCodeSmellSamples.normalized.json \
        --out data/context_files
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def repo_slug(repo_url: str) -> str:
    """Turn 'git@github.com:owner/repo.git' or an https URL into 'owner/repo'."""
    slug = repo_url.strip()
    for prefix in ("git@github.com:", "https://github.com/", "http://github.com/"):
        if slug.startswith(prefix):
            slug = slug[len(prefix):]
    return slug[:-4] if slug.endswith(".git") else slug


def file_key(repo_url: str, commit: str, path: str) -> str:
    return hashlib.sha1(f"{repo_slug(repo_url)}@{commit}:{path}".encode()).hexdigest()


def raw_url(repo_url: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo_slug(repo_url)}/{commit}/{path.lstrip('/')}"


def fetch(url: str, dest: Path, retries: int = 3) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                dest.write_bytes(resp.read())
            return "ok"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return "missing"
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return "error"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", default="data/MLCQCodeSmellSamples.normalized.json")
    parser.add_argument("--out", default="data/context_files")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    samples = json.loads(Path(args.normalized).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = {}
    for s in samples:
        key = file_key(s["repo_url"], s["commit_hash"], s["file_path"])
        jobs[key] = raw_url(s["repo_url"], s["commit_hash"], s["file_path"])

    status = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, url, out / f"{key}.java"): key for key, url in jobs.items()}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            status[futures[fut]] = fut.result()
            if i % 500 == 0:
                print(f"{i}/{len(jobs)} files processed")

    counts = {}
    for value in status.values():
        counts[value] = counts.get(value, 0) + 1
    manifest = {"n_files": len(jobs), "status_counts": counts, "status": status}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
