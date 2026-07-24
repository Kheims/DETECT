"""Run Claude Code as a smell judge, with and without access to the repository.

- snippet: claude -p with all tools disabled, the model sees the snippet only.
- agent: claude -p in the repository snapshot with read-only tools.
- Results are appended to <out_dir>/results.jsonl and a restart skips done rows.

Usage:
    python scripts/llm/claude_agent_judge.py --config config/experiments/claude_agent.yml
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402

JSON_LINE = re.compile(r"\{[^{}]*\"smell_probability\"[^{}]*\}")


def resolve(path: Any) -> Path:
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def definitions(prompts_path: Path) -> dict[str, dict[str, str]]:
    """Smell name and definition paragraph of every question of the judge prompts."""
    prompts = yaml.safe_load(prompts_path.read_text())
    out = {}
    for label, spec in prompts["smells"].items():
        q = spec["question"]
        name = re.search(r"Code smell: (.+?)\.\n", q).group(1)
        definition = re.search(r"Definition: (.+?)\n\n", q, re.S).group(1)
        out[label] = {"name": name, "definition": " ".join(definition.split()), "level": spec.get("level", "")}
    return out


def snapshot_dir(root: Path, repo: str, commit: str) -> Path:
    return root / f"{repo.replace('/', '__')}@{commit}"


def select_subset(meta: list[dict[str, str]], scfg: dict[str, Any], label_col: dict[str, str],
                  label_level: dict[str, str], available: set[int]) -> list[tuple[int, str]]:
    """Sample n_pos positives and n_neg negatives per smell, optionally within listed strata."""
    rng = random.Random(int(scfg["seed"]))
    pairs = []
    strata = scfg.get("strata") or {}
    for label in scfg["smells"]:
        pool = [r for r in meta if r["level"] == label_level[label] and int(r["json_index"]) in available
                and (label not in strata or r.get("stratum") in {json.dumps(s) for s in strata[label]})]
        # MLCQ meta rows hold the label under its CSV column, DACOS rows under the label name
        key = label if label in pool[0] else label_col[label]
        pos = sorted(int(r["json_index"]) for r in pool if r[key] == "1")
        neg = sorted(int(r["json_index"]) for r in pool if r[key] == "0")
        pairs += [(i, label) for i in rng.sample(pos, min(len(pos), int(scfg["n_pos"])))]
        pairs += [(i, label) for i in rng.sample(neg, min(len(neg), int(scfg["n_neg"])))]
    return pairs


def load_meta(cfg: dict[str, Any], entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Load the sample meta CSV, or build one row per entry when the dataset has none (DACOS)."""
    if cfg.get("sample_meta"):
        return list(csv.DictReader(resolve(cfg["sample_meta"]).open()))
    rows = []
    for i, e in enumerate(entries):
        row = {"json_index": str(i), "level": str(e.get("level", "method")),
               "stratum": json.dumps(e.get(cfg.get("stratum_field", ""))) if cfg.get("stratum_field") else ""}
        for label, col in cfg["label_columns"].items():
            row[label] = "1" if e["y"][int(col)] else "0"
        rows.append(row)
    return rows


def build_prompt(template: str, smell: dict[str, str], entry: dict[str, Any], level: str) -> str:
    return template.format(
        smell=smell["name"], definition=smell["definition"], kind="method" if level == "method" else "class",
        file_path=str(entry.get("file_path", "")).lstrip("/"), start_line=entry.get("start_line"),
        end_line=entry.get("end_line"),
        code=entry["code_snippet"])


def run_claude(args: list[str], prompt: str, cwd: Path, timeout: float) -> tuple[str, str, int, float]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_")}
    start = time.time()
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=cwd, env=env, text=True, start_new_session=True)
    try:
        out, err = proc.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        return "", "timeout", -9, time.time() - start
    return out, err, proc.returncode, time.time() - start


def parse_answer(text: str) -> tuple[float | None, str | None]:
    found = JSON_LINE.findall(text or "")
    if not found:
        return None, None
    try:
        obj = json.loads(found[-1])
        return float(obj["smell_probability"]) / 100.0, str(obj.get("answer", "")).lower() or None
    except (ValueError, KeyError, TypeError):
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="print the first prompt of each condition and stop")
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))["claude_agent"]

    entries = json.loads(resolve(cfg["input_json"]).read_text())
    meta = load_meta(cfg, entries)
    defs = definitions(resolve(cfg["prompts"]))
    snapshots = resolve(cfg["snapshots"]) if cfg.get("snapshots") else None
    label_col = dict(cfg["label_columns"])
    label_level = {label: defs[label]["level"] for label in label_col}
    available = ({int(r["json_index"]) for r in meta
                  if (snapshot_dir(snapshots, r["repo"], r["commit_hash"]) / ".done").exists()}
                 if snapshots else {int(r["json_index"]) for r in meta})
    if cfg["subset"].get("from_file"):
        pairs = [(int(i), str(s)) for i, s in json.loads(resolve(cfg["subset"]["from_file"]).read_text())]
    else:
        pairs = select_subset(meta, cfg["subset"], label_col, label_level, available)
    level_of = {int(r["json_index"]): r["level"] for r in meta}
    repo_of = {int(r["json_index"]): (r.get("repo"), r.get("commit_hash")) for r in meta}

    out_dir = resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "subset.json").write_text(json.dumps(pairs))
    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("score") is not None:
                done.add((row["condition"], row["json_index"], row["smell"]))
    binary = shutil.which("claude") or str(Path("~/.npm-global/bin/claude").expanduser())
    neutral = Path(cfg["neutral_workdir"]).expanduser()
    neutral.mkdir(parents=True, exist_ok=True)

    jobs = []
    for cond_name, cond in cfg["conditions"].items():
        base = [binary, "-p", "--model", cfg["model"], "--output-format", "json", "--strict-mcp-config",
                "--setting-sources", "project", "--tools", cond["tools"]]
        if cond.get("restricted"):
            base.append("--restricted")
        for i, label in pairs:
            if (cond_name, i, label) in done:
                continue
            prompt = build_prompt(cond["template"], defs[label], entries[i], level_of[i])
            cwd = snapshot_dir(snapshots, *repo_of[i]) if cond.get("cwd") == "snapshot" else neutral
            jobs.append((cond_name, i, label, base, prompt, cwd))
    if args.dry_run:
        for cond_name in cfg["conditions"]:
            first = next(j for j in jobs if j[0] == cond_name)
            print(f"=== {cond_name} cwd={first[5]}\n{' '.join(first[3])}\n{first[4]}\n")
        return
    print(f"[claude_agent] {len(pairs)} (sample, smell) pairs, {len(jobs)} queries to run", flush=True)

    lock = threading.Lock()

    def work(job: tuple) -> dict[str, Any]:
        cond_name, i, label, base, prompt, cwd = job
        for attempt in range(int(cfg["retries"])):
            out, err, code, seconds = run_claude(base, prompt, cwd, float(cfg["timeout_seconds"]))
            row: dict[str, Any] = {"condition": cond_name, "json_index": i, "smell": label, "attempt": attempt,
                                   "returncode": code, "wall_seconds": round(seconds, 2)}
            try:
                payload = json.loads(out)
            except json.JSONDecodeError:
                payload = {}
            text = str(payload.get("result", ""))
            score, answer = parse_answer(text)
            usage = payload.get("usage") or {}
            row.update(score=score, answer=answer, text=text[-1500:], num_turns=payload.get("num_turns"),
                       input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                       cache_read_tokens=usage.get("cache_read_input_tokens"),
                       cache_creation_tokens=usage.get("cache_creation_input_tokens"),
                       cost_usd=payload.get("total_cost_usd"), duration_ms=payload.get("duration_ms"),
                       is_error=payload.get("is_error"), error=(err or "")[-300:] if code else None)
            with lock, results_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if score is not None:
                return row
            low = (text + (err or "")).lower()
            if code != 0 and ("usage limit" in low or "rate limit" in low or "overloaded" in low):
                time.sleep(float(cfg["quota_cooldown_seconds"]))
        return row

    finished = 0
    with cf.ThreadPoolExecutor(int(cfg["workers"])) as ex:
        for row in ex.map(work, jobs):
            finished += 1
            if finished % 10 == 0 or row.get("score") is None:
                print(f"[{finished}/{len(jobs)}] {row['condition']} {row['json_index']} {row['smell']} "
                      f"score={row.get('score')} turns={row.get('num_turns')} {row.get('wall_seconds')} s", flush=True)
    print(f"[claude_agent] wrote {results_path}")


if __name__ == "__main__":
    main()
