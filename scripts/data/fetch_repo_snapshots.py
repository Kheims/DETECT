"""Download the Java files of every MLCQ repository at the annotated commit.

Streams one GitHub tarball per (repository, commit) and keeps only .java files.
Finished snapshots have a .done marker and are skipped on restart.

Usage:
    python scripts/data/fetch_repo_snapshots.py --meta data/derived/mlcq_sample_meta.csv \
        --out data/repo_snapshots --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def snapshot_name(repo: str, commit: str) -> str:
    return f"{repo.replace('/', '__')}@{commit}"


def fetch_one(repo: str, commit: str, out: Path, token: str, retries: int = 3) -> dict:
    dest = out / snapshot_name(repo, commit)
    if (dest / ".done").exists():
        return {"repo": repo, "commit": commit, "status": "cached"}
    url = f"https://api.github.com/repos/{repo}/tarball/{commit}"
    req = urllib.request.Request(url, headers={"Authorization": f"token {token}",
                                               "Accept": "application/vnd.github+json"})
    for attempt in range(retries):
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        n_files = n_bytes = 0
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, \
                    tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".java"):
                        continue
                    # drop the "<owner>-<repo>-<sha>/" top directory of the tarball
                    rel = Path(*Path(member.name).parts[1:])
                    if rel.is_absolute() or ".." in rel.parts:
                        continue
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = tar.extractfile(member).read()
                    target.write_bytes(data)
                    n_files += 1
                    n_bytes += len(data)
            (dest / ".done").write_text(json.dumps({"files": n_files, "bytes": n_bytes}))
            return {"repo": repo, "commit": commit, "status": "ok", "files": n_files,
                    "bytes": n_bytes, "seconds": round(time.time() - start, 1)}
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 451):
                shutil.rmtree(dest, ignore_errors=True)
                return {"repo": repo, "commit": commit, "status": f"http_{exc.code}"}
            time.sleep(5 * 2 ** attempt)
        except Exception as exc:  # truncated stream, timeout, reset
            last = repr(exc)
            time.sleep(5 * 2 ** attempt)
    shutil.rmtree(dest, ignore_errors=True)
    return {"repo": repo, "commit": commit, "status": "error", "error": locals().get("last")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", type=Path, default=Path("data/derived/mlcq_sample_meta.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/repo_snapshots"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pairs = sorted({(r["repo"], r["commit_hash"]) for r in csv.DictReader(args.meta.open())})
    token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "fetch_log.jsonl").open("a")
    done = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(fetch_one, repo, commit, args.out, token) for repo, commit in pairs]
        for fut in cf.as_completed(futures):
            row = fut.result()
            log.write(json.dumps(row) + "\n")
            log.flush()
            done += 1
            if row["status"] not in ("ok", "cached") or done % 25 == 0:
                print(f"[{done}/{len(pairs)}] {row}", flush=True)
    print(f"[snapshots] {done} pairs processed, log in {args.out / 'fetch_log.jsonl'}")


if __name__ == "__main__":
    main()
