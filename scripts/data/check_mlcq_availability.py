from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZENODO_RECORD = "https://zenodo.org/api/records/3666840"
CSV_NAME = "MLCQCodeSmellSamples.csv"
USER_AGENT = "mlcq-availability-check"
SMELLS = ["feature envy", "long method", "blob", "data class"]
SEVERITY = {"none": 0, "minor": 1, "major": 2, "critical": 3}

# Current names of Apache repositories that were renamed after MLCQ was built.
# Only used when the commit search finds nothing; each one is checked, not assumed.
APACHE_RENAMED = {
    "apache/axis1-java": "apache/axis-axis1-java",
    "apache/axis2-java": "apache/axis-axis2-java-core",
    "apache/webservices-xmlschema": "apache/ws-xmlschema",
    "apache/wss4j": "apache/ws-wss4j",
}


# ---------------------------------------------------------------- HTTP helpers

def github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    if shutil.which("gh"):
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        return out.stdout.strip() or None
    return None


def request(url: str, *, method: str = "GET", headers: dict | None = None, timeout: int = 60):
    """Return (status, body, response headers). Network errors give status 'ERR <name>'."""
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read() if method == "GET" else b""
            return resp.status, body, dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, b"", dict(err.headers or {})
    except Exception as err:  # noqa: BLE001
        return f"ERR {type(err).__name__}", b"", {}


class GitHub:
    def __init__(self, token: str | None):
        self.headers = {"Accept": "application/vnd.github+json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        # Commit search allows 30 requests/min with a token, 10 without
        self.search_pause = 2.5 if token else 6.5

    def api(self, path: str):
        status, body, _ = request("https://api.github.com/" + path, headers=self.headers)
        data = json.loads(body) if status == 200 and body else None
        return status, data

    def search_commit(self, sha: str) -> list[str]:
        status, data = self.api(f"search/commits?q=hash:{sha}")
        time.sleep(self.search_pause)
        if status != 200 or not data:
            return []
        return sorted({item["repository"]["full_name"] for item in data.get("items", [])})

    def pick_location(self, candidates: list[str], original: str) -> tuple[str | None, bool]:
        """Choose the canonical copy among repositories holding the commit.

        Mirrors kept by other projects or users are not flagged as forks by GitHub,
        so the rule is: same repository name first, then organisation accounts,
        then the most starred. Returns (repository, is_personal_copy).
        """
        def short(name: str) -> str:
            return name.split("/")[-1].lower().removeprefix("org.eclipse.")

        ranked = []
        for full_name in candidates:
            status, data = self.api(f"repos/{full_name}")
            if status != 200 or not data:
                continue
            same_name = short(original) in short(full_name) or short(full_name) in short(original)
            is_org = data["owner"]["type"] == "Organization"
            ranked.append((not data["fork"], same_name, is_org, data["stargazers_count"], full_name))
        if not ranked:
            return None, False
        best = max(ranked)
        return best[4], not best[2]


def raw_url(repo: str, sha: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{sha}/" + urllib.parse.quote(path.lstrip("/"))


def link_url(link: str) -> str:
    """Browser link of the CSV without its #L anchor, percent-encoded."""
    base = link.split("#", 1)[0]
    return urllib.parse.quote(base, safe=":/")


def raw_line_count(repo: str, sha: str, path: str):
    status, body, _ = request(raw_url(repo, sha, path))
    lines = len(body.decode("utf-8", "replace").splitlines()) if status == 200 else 0
    return status, lines


class SoftwareHeritage:
    BASE = "https://archive.softwareheritage.org/api/1"

    def __init__(self, token: str | None):
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _get(self, path: str):
        status, body, headers = request(self.BASE + path, headers=self.headers)
        remaining = headers.get("X-RateLimit-Remaining")
        reset = headers.get("X-RateLimit-Reset")
        if remaining is not None and reset is not None and int(remaining) <= 1:
            wait = max(0, int(reset) - int(time.time())) + 5
            print(f"    Software Heritage rate limit reached, waiting {wait} s", flush=True)
            time.sleep(wait)
        return status, body

    def revision_exists(self, sha: str) -> bool:
        status, _ = self._get(f"/revision/{sha}/")
        return status == 200

    def file_line_count(self, sha: str, path: str):
        status, body = self._get(f"/revision/{sha}/directory/{urllib.parse.quote(path.lstrip('/'))}/")
        if status != 200:
            return status, 0
        entry = json.loads(body)
        target = entry["content"]["checksums"]["sha1_git"] if "content" in entry else entry.get("target")
        status, body = self._get(f"/content/sha1_git:{target}/raw/")
        lines = len(body.decode("utf-8", "replace").splitlines()) if status == 200 else 0
        return status, lines


# ------------------------------------------------------------------- data

def ensure_csv(path: Path) -> str:
    """Download the MLCQ CSV from Zenodo if needed and return its MD5 check result."""
    status, body, _ = request(f"{ZENODO_RECORD}/files", headers={"Accept": "application/json"})
    published = None
    if status == 200:
        for entry in json.loads(body).get("entries", []):
            if entry["key"] == CSV_NAME:
                published = entry.get("checksum", "").removeprefix("md5:")
    if not path.exists():
        print(f"Downloading {CSV_NAME} from Zenodo to {path}")
        status, body, _ = request(f"{ZENODO_RECORD}/files/{CSV_NAME}/content", timeout=300)
        if status != 200:
            raise SystemExit(f"Zenodo download failed (status {status}); pass --csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    local = hashlib.md5(path.read_bytes()).hexdigest()
    if published is None:
        return f"local md5 {local} (Zenodo checksum unavailable)"
    return f"md5 {local} {'matches' if local == published else 'DIFFERS FROM'} Zenodo ({published})"


def sample_key(repo, sha, path, start, end):
    return (repo, sha, path, str(start), str(end))


def owner_repo(git_url: str) -> str:
    return git_url.split(":", 1)[1].removesuffix(".git")


def prevalence(votes: dict, ids: set) -> dict:
    """Share of samples whose median severity is above none, per smell."""
    out = {}
    for smell in SMELLS:
        pos = sum(statistics.median(votes[(i, smell)]) > 0 for i in ids if (i, smell) in votes)
        out[smell] = {"positives": pos, "samples": len(ids), "share": round(pos / len(ids), 4)}
    return out


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Check which MLCQ samples can still be retrieved, and where.")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / CSV_NAME)
    parser.add_argument("--detect-json", type=Path, default=ROOT / "data" / "MLCQCodeSmellSamples.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "mlcq_availability")
    parser.add_argument("--control", type=int, default=40, help="retrieved samples checked as a control")
    parser.add_argument("--swh-files", action="store_true", help="fetch every remaining file from Software Heritage")
    args = parser.parse_args()

    print("== Data")
    print(" ", ensure_csv(args.csv))
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8"), delimiter=";"))
    used = json.loads(args.detect_json.read_text())
    used_keys = {sample_key(x["repo_url"], x["commit_hash"], x["file_path"], x["start_line"], x["end_line"]) for x in used}

    samples = {}
    for r in rows:
        samples.setdefault(r["sample_id"], r)
    missing = [s for s in samples.values()
               if sample_key(s["repository"], s["commit_hash"], s["path"], s["start_line"], s["end_line"]) not in used_keys]
    missing_ids = {s["sample_id"] for s in missing}
    kept_ids = set(samples) - missing_ids

    counts = {
        "original": {"reviews": len(rows), "samples": len(samples), "repositories": len({r["repository"] for r in rows})},
        "detect": {"reviews": len(used), "samples": len(kept_ids), "repositories": len({x["repo_url"] for x in used})},
        "missing": {"reviews": sum(r["sample_id"] in missing_ids for r in rows), "samples": len(missing),
                    "repositories": len({s["repository"] for s in missing})},
    }
    for name, c in counts.items():
        print(f"  {name:9s} {c['reviews']:6d} reviews  {c['samples']:5d} samples  {c['repositories']:4d} repositories")
    print(f"  share of reviews used: {counts['detect']['reviews'] / counts['original']['reviews']:.1%}")

    votes = defaultdict(list)
    for r in rows:
        votes[(r["sample_id"], r["smell"])].append(SEVERITY[r["severity"]])
    prev = {"original": prevalence(votes, set(samples)), "detect": prevalence(votes, kept_ids)}
    print("  prevalence (median rule), original -> used:")
    for smell in SMELLS:
        a, b = prev["original"][smell], prev["detect"][smell]
        print(f"    {smell:13s} {a['positives']:3d}/{a['samples']} = {a['share']:.2%}  ->  {b['positives']:3d}/{b['samples']} = {b['share']:.2%}")

    gh = GitHub(github_token())
    random.seed(0)
    control = random.sample([samples[i] for i in sorted(kept_ids)], min(args.control, len(kept_ids)))

    print("== Addresses of the CSV (raw file and browser link)")
    status = {}
    for label, group in (("control", control), ("missing", missing)):
        tally = Counter()
        for s in group:
            repo = owner_repo(s["repository"])
            raw_status, _, _ = request(raw_url(repo, s["commit_hash"], s["path"]), method="HEAD")
            link_status, _, _ = request(link_url(s["link"]), method="HEAD")
            status[s["sample_id"]] = {"raw": raw_status, "link": link_status}
            tally[(raw_status, link_status)] += 1
            time.sleep(0.3)
        print(f"  {label:8s} " + ", ".join(f"raw={r} link={l}: {n}" for (r, l), n in tally.items()))

    print("== Repositories and commits of the missing samples")
    commits = sorted({(owner_repo(s["repository"]), s["commit_hash"]) for s in missing})
    per_commit = {}
    for repo, sha in commits:
        repo_status, data = gh.api(f"repos/{repo}")
        commit_status, _ = gh.api(f"repos/{repo}/commits/{sha}")
        found = gh.search_commit(sha)
        location, personal = gh.pick_location(found, repo)
        if location is None and repo in APACHE_RENAMED:
            if gh.api(f"repos/{APACHE_RENAMED[repo]}/commits/{sha}")[0] == 200:
                location, personal = APACHE_RENAMED[repo], False
        per_commit[(repo, sha)] = {"repo_status": repo_status, "commit_status": commit_status,
                                   "found_in": found, "location": location,
                                   "location_is_personal_copy": personal}
        print(f"  {repo:40s} repo={repo_status} commit={commit_status} -> {location or 'not found on GitHub'}"
              + (" (personal copy)" if personal else ""), flush=True)

    print("== Files at the new GitHub locations")
    recovered = {}
    for s in missing:
        info = per_commit[(owner_repo(s["repository"]), s["commit_hash"])]
        ok = False
        if info["location"]:
            st, n = raw_line_count(info["location"], s["commit_hash"], s["path"])
            ok = st == 200 and n >= int(s["end_line"])
            time.sleep(0.15)
        recovered[s["sample_id"]] = ok
    print(f"  recovered on GitHub: {sum(recovered.values())} of {len(missing)}")

    print("== Software Heritage for the commits still missing")
    swh = SoftwareHeritage(os.environ.get("SWH_TOKEN"))
    still = sorted({(owner_repo(s["repository"]), s["commit_hash"]) for s in missing if not recovered[s["sample_id"]]})
    for repo, sha in still:
        per_commit[(repo, sha)]["in_software_heritage"] = swh.revision_exists(sha)
        print(f"  {repo:40s} {sha[:10]} archived={per_commit[(repo, sha)]['in_software_heritage']}")
    swh_files = {}
    if args.swh_files:
        todo = [s for s in missing if not recovered[s["sample_id"]]]
        for k, s in enumerate(todo, 1):
            st, n = swh.file_line_count(s["commit_hash"], s["path"])
            swh_files[s["sample_id"]] = st == 200 and n >= int(s["end_line"])
            print(f"  [{k}/{len(todo)}] {s['path'][-60:]} -> {'ok' if swh_files[s['sample_id']] else st}", flush=True)
            if not swh.headers:
                time.sleep(60)  # anonymous limit: 120 requests/hour, 2 requests per file
        print(f"  files retrieved from Software Heritage: {sum(swh_files.values())} of {len(todo)}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "missing_samples.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "repository", "commit_hash", "path", "start_line", "end_line",
                         "raw_status", "link_status", "new_location", "new_location_is_personal_copy", "recovered_on_github",
                         "commit_in_software_heritage", "file_from_software_heritage"])
        for s in missing:
            info = per_commit[(owner_repo(s["repository"]), s["commit_hash"])]
            writer.writerow([s["sample_id"], s["repository"], s["commit_hash"], s["path"], s["start_line"], s["end_line"],
                             status[s["sample_id"]]["raw"], status[s["sample_id"]]["link"], info["location"] or "",
                             info["location_is_personal_copy"], recovered[s["sample_id"]], info.get("in_software_heritage", ""),
                             swh_files.get(s["sample_id"], "")])
    report = {
        "counts": counts,
        "prevalence": prev,
        "control": Counter(f"raw={status[s['sample_id']]['raw']} link={status[s['sample_id']]['link']}" for s in control),
        "missing_addresses": Counter(f"raw={status[s['sample_id']]['raw']} link={status[s['sample_id']]['link']}" for s in missing),
        "commits": {f"{r}@{c}": v for (r, c), v in per_commit.items()},
        "recovered_on_github": sum(recovered.values()),
        "files_from_software_heritage": sum(swh_files.values()) if args.swh_files else None,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"== Written to {args.out}")


if __name__ == "__main__":
    main()
