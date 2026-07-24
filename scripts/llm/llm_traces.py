"""Analyze the answers and tool use of the LLM judges on the MLCQ agent subset.

Reads the Claude agent judge and the Qwen context judge runs and writes
evidence, size, tools, agreement and severity tables and figures, plus
examples.md, to out_dir.

Usage:
    python scripts/llm/llm_traces.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]
NAMES = {"is_feature_envy": "Feature Envy", "is_long_method": "Long Method", "is_blob": "Blob",
         "is_data_class": "Data Class"}
# keyword rules for the kinds of evidence a written answer cites (case-insensitive)
EVIDENCE = {
    "size or length": r"(\b\d+\s*(non-blank |physical |source )?lines?\b|lines? of code|\bLOC\b|\blength|\blengthy\b|"
                      r"\bshort\b|too long|very long|fairly long|quite long|\bsize\b|\blarge\b|\bsmall\b|statements)",
    "control flow and complexity": r"(nest|branch|loop|conditional|if/else|if-else|switch|complex|cyclomatic)",
    "own versus other class data": r"(own (class|fields?|methods?|state|data)|other (class|object|type)|another (class|object)|"
                                   r"foreign|external (class|object|data)|its class|enclosing class|delegat)",
    "getters, setters, data holder": r"(getter|setter|accessor|data holder|pojo|dto|\bbean\b|value object|"
                                     r"no (real )?behaviou?r|plain data)",
    "responsibilities, cohesion": r"(responsibilit|does too much|cohesi|god class|many (fields|methods)|concerns)",
    "missing context": r"(can(not|'t) (be )?(judge|determin|tell|see|know)|without (seeing|more|the rest|access)|"
                       r"not visible|snippet alone|from this snippet|unclear|hard to (say|tell)|not shown|no context)",
}
OUTCOMES = ("TP", "FN", "FP", "TN")
# smell names are removed so that they do not count as evidence
SMELL_NAMES = re.compile(r"(long method|feature envy|data class|god class|large class|\bblob\b|"
                         r"long parameter list|complex method)", re.I)


def cited(text: str, rx: str) -> bool:
    return bool(re.search(rx, SMELL_NAMES.sub(" ", text), re.I))


def outcome(y: int, yes: bool) -> str:
    return {(1, True): "TP", (1, False): "FN", (0, True): "FP", (0, False): "TN"}[(y, yes)]


def load_labels() -> list[list[int]]:
    data = json.loads((ROOT / "data/MLCQCodeSmellSamples.normalized.json").read_text())
    return [d["y"] for d in data]


def load_loc() -> dict[int, int]:
    with (ROOT / "data/derived/size_features.csv").open() as f:
        return {int(r["sample_idx"]): int(r["loc_nonblank"]) for r in csv.DictReader(f)}


def load_claude(path: Path) -> dict[str, dict[tuple[int, str], dict]]:
    out: dict[str, dict] = defaultdict(dict)
    for line in path.open():
        r = json.loads(line)
        if r.get("is_error") or r.get("score") is None:
            continue
        key = (r["json_index"], r["smell"])
        old = out[r["condition"]].get(key)
        if old is None or r["attempt"] >= old["attempt"]:
            out[r["condition"]][key] = r
    return out


def load_qwen(base: Path, cond: str) -> tuple[dict, dict]:
    scores, texts = {}, {}
    for f in glob.glob(str(base / cond / "scores*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("applicable", True):
                scores[(r["json_index"], r["smell"])] = r
    for f in glob.glob(str(base / cond / "transcripts*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            # the assistant turns before the final question, without tool call blocks
            parts = [re.sub(r"<tool_call>.*?</tool_call>", " ", m["content"], flags=re.S)
                     for m in r["messages"] if m["role"] == "assistant"]
            texts[(r["json_index"], r["smell"])] = "\n".join(parts)
    return scores, texts


def judges(args) -> dict[str, dict[tuple[int, str], dict]]:
    """{judge: {(json_index, smell): {score, text, ...}}} on the agent subset."""
    out = {}
    claude = load_claude(ROOT / args.claude)
    for cond, name in (("snippet", "Claude Sonnet 5, snippet"), ("agent", "Claude Sonnet 5, agent")):
        out[name] = {k: {"score": float(r["score"]), "text": r["text"], "turns": r.get("num_turns", 1)}
                     for k, r in claude.get(cond, {}).items()}
    base = ROOT / args.qwen
    for cond, name in (("C0", "Qwen3.6-27B, one-word answer"), ("REASON", "Qwen3.6-27B, reasoning"),
                       ("C4", "Qwen3.6-27B, agent")):
        scores, texts = load_qwen(base, cond)
        if not scores:
            print(f"[traces] no {cond} scores, skipped")
            continue
        out[name] = {k: {"score": float(r["score"]), "text": texts.get(k), "tools": r.get("tool_calls", []),
                         "turns": r.get("turns", 1)} for k, r in scores.items()}
    # Qwen C0 covers every sample: keep the agent subset only
    subset = set(out.get("Qwen3.6-27B, agent", {})) or set(out["Claude Sonnet 5, agent"])
    for name in out:
        if name.startswith("Qwen"):
            out[name] = {k: v for k, v in out[name].items() if k in subset}
    return out


def evidence(js: dict, y: list[list[int]], out: Path) -> list[dict]:
    rows = []
    for name, rows_j in js.items():
        for smell in LABELS:
            for oc in OUTCOMES + ("all",):
                items = [(k, v) for k, v in rows_j.items() if k[1] == smell and v.get("text")
                         and (oc == "all" or outcome(y[k[0]][LABELS.index(smell)], v["score"] > 0.5) == oc)]
                if not items:
                    continue
                row = {"judge": name, "smell": smell, "outcome": oc, "n": len(items),
                       "mean_words": float(np.mean([len(v["text"].split()) for _, v in items]))}
                for ev, rx in EVIDENCE.items():
                    row[ev] = float(np.mean([cited(v["text"], rx) for _, v in items]))
                rows.append(row)
    with (out / "evidence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def evidence_figure(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    judges_t = list(dict.fromkeys(r["judge"] for r in rows))
    evs = list(EVIDENCE)
    fig, axes = plt.subplots(len(judges_t), 4, figsize=(17, 2.9 * len(judges_t) + 1), squeeze=False)
    for i, j in enumerate(judges_t):
        for k, smell in enumerate(LABELS):
            ax = axes[i, k]
            mat = np.full((len(evs), 4), np.nan)
            ns = []
            for c, oc in enumerate(OUTCOMES):
                r = next((x for x in rows if x["judge"] == j and x["smell"] == smell and x["outcome"] == oc), None)
                ns.append(r["n"] if r else 0)
                if r:
                    mat[:, c] = [r[e] for e in evs]
            ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            for a in range(len(evs)):
                for b in range(4):
                    if not np.isnan(mat[a, b]):
                        ax.text(b, a, f"{mat[a, b]:.0%}", ha="center", va="center", fontsize=7,
                                color="white" if mat[a, b] > 0.6 else "black")
            ax.set_xticks(range(4), [f"{oc}\nn={n}" for oc, n in zip(OUTCOMES, ns)], fontsize=7)
            ax.set_yticks(range(len(evs)), evs if k == 0 else [""] * len(evs), fontsize=8)
            if i == 0:
                ax.set_title(NAMES[smell], fontsize=10)
            if k == 0:
                ax.set_ylabel(j, fontsize=9)
    fig.suptitle("What the written answers cite: share of answers that mention each kind of evidence (keyword rules), "
                 "per outcome.\nTP: smelly and judged yes. FN: smelly, judged no. FP: not smelly, judged yes. "
                 "TN: not smelly, judged no.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "evidence.png", dpi=150)
    fig.savefig(out / "evidence.pdf")
    plt.close(fig)


def size_figure(js: dict, y: list[list[int]], loc: dict[int, int], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Claude Sonnet 5, snippet": "#805ad5", "Claude Sonnet 5, agent": "#b794f4",
              "Qwen3.6-27B, one-word answer": "#dd6b20", "Qwen3.6-27B, reasoning": "#f6ad55",
              "Qwen3.6-27B, agent": "#9c4221"}
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, smell in zip(axes, LABELS):
        ref = js.get("Qwen3.6-27B, agent") or next(iter(js.values()))
        keys = sorted(k for k in ref if k[1] == smell)
        sizes = np.array([loc[k[0]] for k in keys])
        edges = np.unique(np.quantile(sizes, np.linspace(0, 1, 7)))
        bins = np.clip(np.searchsorted(edges, sizes, side="right") - 1, 0, len(edges) - 2)
        centers = [f"{int(edges[b])}-{int(edges[b + 1])}" for b in range(len(edges) - 1)]
        lab = np.array([y[k[0]][LABELS.index(smell)] for k in keys])
        x = np.arange(len(centers))
        human = [lab[bins == b].mean() for b in range(len(centers))]
        ax.plot(x, human, color="black", lw=2.5, marker="o", label="human label: smelly")
        for name, j in js.items():
            ok = np.array([k in j for k in keys])
            yes = np.array([j[k]["score"] > 0.5 if k in j else False for k in keys])
            vals = [yes[(bins == b) & ok].mean() if ((bins == b) & ok).any() else np.nan for b in range(len(centers))]
            ax.plot(x, vals, color=colors.get(name, "grey"), marker=".", label=f"{name}: yes")
            for b, v in enumerate(vals):
                rows.append({"smell": smell, "bin": centers[b], "judge": name, "yes_rate": v, "human_rate": human[b]})
        ax.set_xticks(x, centers, fontsize=7, rotation=30)
        ax.set_xlabel("non-blank lines (6 bins with equal counts)")
        ax.set_title(NAMES[smell], fontsize=10)
        ax.set_ylim(0, 1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("share of samples")
    axes[-1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Does the LLM say yes because the code is long? Share of yes answers and of human positives per size bin "
                 "(agent subset: all positives plus three negatives per positive)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "size.png", dpi=150)
    fig.savefig(out / "size.pdf")
    plt.close(fig)
    with (out / "size.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def tools(js: dict, y: list[list[int]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import matthews_corrcoef

    rows = []
    agent = js.get("Qwen3.6-27B, agent", {})
    base = js.get("Qwen3.6-27B, reasoning") or js.get("Qwen3.6-27B, one-word answer", {})
    base_name = "reasoning" if js.get("Qwen3.6-27B, reasoning") else "one-word answer"
    per_tool = {}
    for smell in LABELS:
        keys = [k for k in agent if k[1] == smell]
        if not keys:
            continue
        lab = np.array([y[k[0]][LABELS.index(smell)] for k in keys])
        n_calls = np.array([len(agent[k]["tools"]) for k in keys])
        yes_a = np.array([agent[k]["score"] > 0.5 for k in keys])
        yes_b = np.array([base[k]["score"] > 0.5 if k in base else False for k in keys])
        names = Counter(t["name"] or "malformed call" for k in keys for t in agent[k]["tools"])
        per_tool[smell] = names
        for used in (False, True):
            m = (n_calls > 0) == used
            if m.sum() < 5:
                continue
            rows.append({"judge": "Qwen3.6-27B agent", "smell": smell, "tool_called": used, "n": int(m.sum()),
                         "positives": int(lab[m].sum()), "yes_rate": float(yes_a[m].mean()),
                         "mcc_agent": float(matthews_corrcoef(lab[m], yes_a[m])),
                         f"mcc_{base_name.replace(' ', '_')}": float(matthews_corrcoef(lab[m], yes_b[m])),
                         "changed_vs_" + base_name.replace(" ", "_"): float((yes_a[m] != yes_b[m]).mean()),
                         "mean_calls": float(n_calls[m].mean())})
        c_agent, c_snip = js.get("Claude Sonnet 5, agent", {}), js.get("Claude Sonnet 5, snippet", {})
        ck = [k for k in c_agent if k[1] == smell and k in c_snip]
        if ck:
            lab_c = np.array([y[k[0]][LABELS.index(smell)] for k in ck])
            multi = np.array([c_agent[k]["turns"] > 1 for k in ck])
            ya = np.array([c_agent[k]["score"] > 0.5 for k in ck])
            ys = np.array([c_snip[k]["score"] > 0.5 for k in ck])
            for used in (False, True):
                m = multi == used
                if m.sum() < 5:
                    continue
                rows.append({"judge": "Claude agent", "smell": smell, "tool_called": used, "n": int(m.sum()),
                             "positives": int(lab_c[m].sum()), "yes_rate": float(ya[m].mean()),
                             "mcc_agent": float(matthews_corrcoef(lab_c[m], ya[m])),
                             "mcc_snippet": float(matthews_corrcoef(lab_c[m], ys[m])),
                             "changed_vs_snippet": float((ya[m] != ys[m]).mean()), "mean_calls": float("nan")})
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (out / "tools.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if per_tool:
        tool_names = sorted({t for c in per_tool.values() for t in c})
        fig, ax = plt.subplots(figsize=(9, 3.6))
        x = np.arange(len(LABELS))
        bottom = np.zeros(len(LABELS))
        n_conv = np.array([sum(1 for k in agent if k[1] == s) for s in LABELS])
        for t, c in zip(tool_names, ["#2b6cb0", "#38a169", "#d69e2e", "#dd6b20", "#805ad5", "#a0aec0", "#e53e3e"]):
            vals = np.array([per_tool.get(s, {}).get(t, 0) for s in LABELS]) / n_conv
            ax.bar(x, vals, bottom=bottom, color=c, label=t)
            bottom += vals
        ax.set_xticks(x, [NAMES[s] for s in LABELS])
        ax.set_ylabel("tool calls per conversation")
        ax.legend(fontsize=8, frameon=False, ncol=3)
        ax.set_title("Which repository tools the Qwen3.6-27B agent calls, per smell (budget: 6 calls)", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "tools.png", dpi=150)
        fig.savefig(out / "tools.pdf")
        plt.close(fig)


def agreement(js: dict, y: list[list[int]], out: Path) -> None:
    """Agreement of each judge with the labels and with the other judges."""
    from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

    names = list(js)
    rows = []
    for smell in LABELS:
        for a in names:
            keys = sorted(k for k in js[a] if k[1] == smell)
            lab = np.array([y[k[0]][LABELS.index(smell)] for k in keys])
            dec = np.array([js[a][k]["score"] > 0.5 for k in keys])
            rows.append({"smell": smell, "a": a, "b": "human label", "n": len(keys), "positives": int(lab.sum()),
                         "agreement": float((dec == lab).mean()), "kappa": float(cohen_kappa_score(dec, lab)),
                         "mcc": float(matthews_corrcoef(lab, dec)), "recall": float(dec[lab == 1].mean()),
                         "yes_rate": float(dec.mean())})
            for b in names:
                if a < b:
                    common = sorted(set(js[a]) & set(js[b]))
                    common = [k for k in common if k[1] == smell]
                    da = np.array([js[a][k]["score"] > 0.5 for k in common])
                    db = np.array([js[b][k]["score"] > 0.5 for k in common])
                    rows.append({"smell": smell, "a": a, "b": b, "n": len(common), "positives": None,
                                 "agreement": float((da == db).mean()), "kappa": float(cohen_kappa_score(da, db)),
                                 "mcc": None, "recall": None, "yes_rate": None})
        pos = [k for k in js[names[0]] if k[1] == smell and y[k[0]][LABELS.index(smell)] == 1]
        pos = [k for k in pos if all(k in js[n] for n in names)]
        missed = np.mean([all(js[n][k]["score"] <= 0.5 for n in names) for k in pos]) if pos else float("nan")
        rows.append({"smell": smell, "a": "every judge says no", "b": "human positives", "n": len(pos),
                     "positives": len(pos), "agreement": float(missed), "kappa": None, "mcc": None, "recall": None,
                     "yes_rate": None})
    with (out / "agreement.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def severity(js: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((ROOT / "data/MLCQCodeSmellSamples.normalized.json").read_text())
    key = {"is_feature_envy": "feature envy", "is_long_method": "long method", "is_blob": "blob",
           "is_data_class": "data class"}
    levels = ("minor", "major", "critical")
    rows = []
    for name, j in js.items():
        for smell in LABELS:
            for lv in levels:
                ks = [k for k in j if k[1] == smell and data[k[0]]["y"][LABELS.index(smell)] == 1
                      and data[k[0]]["labels"][key[smell]]["severity"] == lv]
                if ks:
                    rows.append({"judge": name, "smell": smell, "severity": lv, "positives": len(ks),
                                 "recall": float(np.mean([j[k]["score"] > 0.5 for k in ks])),
                                 "mean_score": float(np.mean([j[k]["score"] for k in ks]))})
    with (out / "severity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    names = list(js)
    colors = ["#805ad5", "#b794f4", "#dd6b20", "#f6ad55", "#9c4221"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ax, smell in zip(axes, LABELS):
        x = np.arange(len(levels))
        for i, n in enumerate(names):
            vals = [next((r["recall"] for r in rows if r["judge"] == n and r["smell"] == smell and r["severity"] == lv),
                         np.nan) for lv in levels]
            ax.bar(x + (i - (len(names) - 1) / 2) * 0.16, vals, 0.16, color=colors[i % len(colors)], label=n)
        counts = [next((r["positives"] for r in rows if r["judge"] == names[-1] and r["smell"] == smell
                        and r["severity"] == lv), 0) for lv in levels]
        ax.set_xticks(x, [f"{lv}\n(n={c})" for lv, c in zip(levels, counts)])
        ax.set_title(NAMES[smell], fontsize=10)
        ax.set_ylim(0, 1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("share of human positives judged yes")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle("Which positives do the LLMs find? Share judged yes, per severity given by the MLCQ reviewers "
                 "(n: positives of the Qwen agent subset)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "severity.png", dpi=150)
    fig.savefig(out / "severity.pdf")
    plt.close(fig)


def examples(js: dict, y: list[list[int]], out: Path, per_cell: int = 2) -> None:
    rng = np.random.default_rng(0)
    lines = ["# Excerpts of written answers (first 60 words), drawn at random with seed 0\n"]
    for smell in LABELS:
        lines.append(f"## {NAMES[smell]}\n")
        for name, j in js.items():
            for oc in OUTCOMES:
                keys = [k for k, v in j.items() if k[1] == smell and v.get("text")
                        and outcome(y[k[0]][LABELS.index(smell)], v["score"] > 0.5) == oc]
                if not keys:
                    continue
                for i in rng.choice(len(keys), size=min(per_cell, len(keys)), replace=False):
                    k = keys[int(i)]
                    words = " ".join(j[k]["text"].split()[:60])
                    lines.append(f"- **{name}, {oc}** (json_index {k[0]}, score {j[k]['score']:.2f}): {words}")
        lines.append("")
    (out / "examples.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/analysis/llm_traces")
    ap.add_argument("--claude", default="artifacts/claude_agent/sonnet5/results.jsonl")
    ap.add_argument("--qwen", default="artifacts/context_study/scores/qwen36_27b_fp8")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    y, loc = load_labels(), load_loc()
    js = judges(args)
    for name, j in js.items():
        print(f"[traces] {name}: {len(j)} answers, {sum(1 for v in j.values() if v.get('text'))} with text")
    rows = evidence({k: v for k, v in js.items() if any(x.get("text") for x in v.values())}, y, out)
    evidence_figure(rows, out)
    size_figure(js, y, loc, out)
    tools(js, y, out)
    agreement(js, y, out)
    severity(js, out)
    examples({k: v for k, v in js.items() if any(x.get("text") for x in v.values())}, y, out)
    print(f"[traces] wrote {out}")


if __name__ == "__main__":
    main()
