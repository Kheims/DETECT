"""Shared inputs of the interpretability studies (scripts/xai_*.py).

- MLCQ samples with labels, level and severity
- cross-validation folds of the reference GCN runs
- the AST of each snippet, with nodes mapped to 0-based snippet lines
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]
NAMES = {"is_feature_envy": "Feature Envy", "is_long_method": "Long Method", "is_blob": "Blob",
         "is_data_class": "Data Class"}
LEVEL_OF = {"is_feature_envy": "method", "is_long_method": "method", "is_blob": "class", "is_data_class": "class"}
SEVERITY_KEY = {"is_feature_envy": "feature envy", "is_long_method": "long method", "is_blob": "blob",
                "is_data_class": "data class"}
DATASET_FP = "794469cebb2be239824bb32b1baec2b0505ea2f4a32954bef6a8913fb86623f2"
CONSTRUCTION_FP = "647bd56930c5c4669e1e19451fd76f8fcf0bbaafd0595a5831b60b39b1202022"
CACHE = ROOT / "artifacts" / "cache"
# line_scores[i] is the attribution of snippet line i (0-based)
LINE_ROW_FIELDS = ("model", "method", "json_index", "smell", "label", "score", "line_scores")

_NODE_RE = re.compile(r'^"n(\d+)" \[(.*)\]\s*$')
_ATTR_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


@lru_cache(maxsize=1)
def samples() -> list[dict[str, Any]]:
    return json.loads((ROOT / "data/MLCQCodeSmellSamples.normalized.json").read_text())


@lru_cache(maxsize=1)
def levels() -> dict[int, str]:
    with (ROOT / "data/derived/mlcq_sample_meta.csv").open() as f:
        return {int(r["json_index"]): r["level"] for r in csv.DictReader(f)}


def label(json_index: int, smell: str) -> int:
    return int(samples()[json_index]["y"][LABELS.index(smell)])


def severity(json_index: int, smell: str) -> str:
    return samples()[json_index]["labels"][SEVERITY_KEY[smell]]["severity"]


def eligible(json_index: int, smell: str) -> bool:
    return levels().get(json_index) == LEVEL_OF[smell]


def snippet_lines(json_index: int) -> list[str]:
    return samples()[json_index]["code_snippet"].split("\n")


@lru_cache(maxsize=1)
def dataset_order() -> list[int]:
    """json_index of each position of the graph dataset (split indices are positions)."""
    import torch

    ds = torch.load(CACHE / "dataset" / DATASET_FP / "dataset.pt", weights_only=False)
    return [int(d.json_index) for d in ds]


def folds(seed: int = 42, strategy: str = "kfold", k: int = 5) -> dict[int, dict[str, list[int]]]:
    """{fold: {"train"|"val"|"test": [json_index, ...]}} of the reference GCN runs."""
    order = dataset_order()
    out = {}
    for f in range(k):
        path = CACHE / "splits" / DATASET_FP / f"split_{strategy}_repo_url_k{k}_f{f}_seed_{seed}.json"
        split = json.loads(path.read_text())["split_indices"]
        out[f] = {part: [order[i] for i in idx] for part, idx in split.items()}
    return out


@lru_cache(maxsize=1)
def _dot_paths() -> dict[int, Path]:
    cdir = CACHE / "construction" / CONSTRUCTION_FP
    out = {}
    for line in (cdir / "manifest.jsonl").open():
        r = json.loads(line)
        out[int(r["json_index"])] = cdir / "dot" / Path(r["output_dot"]).name
    return out


@dataclass
class AstNode:
    index: int
    kind: str  # SyntaxNode or SyntaxToken
    type: str
    text: str | None
    start_line: int  # snippet line, 0-based (may be -1 for the wrapper)
    end_line: int


def _unquote(v: str) -> str:
    if v.startswith('"') and v.endswith('"'):
        return bytes(v[1:-1], "utf-8").decode("unicode_escape", errors="replace") if "\\" in v else v[1:-1]
    return v


def ast_nodes(json_index: int) -> list[AstNode]:
    """Nodes of the snippet AST in node_index order, lines mapped to the snippet."""
    rows = []
    for line in _dot_paths()[json_index].read_text(errors="replace").splitlines():
        m = _NODE_RE.match(line.strip())
        if not m:
            continue
        attrs = {k: _unquote(v) for k, v in _ATTR_RE.findall(m.group(2))}
        rows.append(attrs)
    # method snippets are wrapped in "public class SnippetWrapper {" on DOT line 1
    wrapped = any(r.get("token_text") == "SnippetWrapper" for r in rows[:12])
    shift = 2 if wrapped else 1
    nodes = []
    for r in rows:
        nodes.append(AstNode(index=int(r["node_index"]), kind=r.get("node_kind", ""), type=r.get("node_type", ""),
                             text=r.get("token_text"), start_line=int(r.get("start_line", 0)) - shift,
                             end_line=int(r.get("end_line", 0)) - shift))
    nodes.sort(key=lambda n: n.index)
    return nodes


def tokens(json_index: int) -> list[AstNode]:
    """The SyntaxToken nodes of the snippet (wrapper tokens removed), in source order."""
    n_lines = len(snippet_lines(json_index))
    toks = [n for n in ast_nodes(json_index) if n.kind == "SyntaxToken" and 0 <= n.start_line < n_lines]
    return toks


def node_line(node: AstNode, n_lines: int) -> int | None:
    """Line of a token node, None for syntax nodes and wrapper tokens."""
    if node.kind != "SyntaxToken" or not (0 <= node.start_line < n_lines):
        return None
    return node.start_line


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in LINE_ROW_FIELDS} | {k: v for k, v in r.items()
                                                                       if k not in LINE_ROW_FIELDS}) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()]
