from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from antlr4 import CommonTokenStream, InputStream, ParserRuleContext, Token
from antlr4.error.ErrorListener import ErrorListener
from antlr4.tree.Tree import TerminalNode


ROOT = Path(__file__).resolve().parents[1]
ANTLR_GENERATED_DIR = ROOT / "tools" / "antlr" / "generated"

if str(ANTLR_GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(ANTLR_GENERATED_DIR))

from Java8Lexer import Java8Lexer  # type: ignore  # noqa: E402
from Java8Parser import Java8Parser  # type: ignore  # noqa: E402


@dataclass
class GraphStats:
    node_count: int
    syntax_node_count: int
    syntax_token_count: int
    edge_count: int
    child_edge_count: int
    next_token_edge_count: int


@dataclass
class ParseResult:
    success: bool
    tree: ParserRuleContext | None
    parser: Java8Parser | None
    error: str | None


class CollectingErrorListener(ErrorListener):
    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(
        self,
        recognizer: Any,
        offendingSymbol: Any,
        line: int,
        column: int,
        msg: str,
        e: Exception | None,
    ) -> None:
        self.errors.append(f"line {line}:{column} {msg}")


def _escape_dot(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', r'\"')
        .replace("\n", r"\n")
        .replace("\r", r"\r")
        .replace("\t", r"\t")
    )


def _dot_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f'"{_escape_dot(str(value))}"'


def _dot_attrs(attrs: dict[str, Any]) -> str:
    return " ".join(f"{key}={_dot_value(value)}" for key, value in attrs.items())


def _token_type_name(token_type: int) -> str:
    symbolic_names = Java8Lexer.symbolicNames
    if 0 <= token_type < len(symbolic_names):
        name = symbolic_names[token_type]
        if name and name != "<INVALID>":
            return name

    literal_names = Java8Lexer.literalNames
    if 0 <= token_type < len(literal_names):
        literal = literal_names[token_type]
        if literal and literal != "<INVALID>":
            return literal

    return str(token_type)


def _token_end_column(token: Token) -> int:
    text = token.text or ""
    if not text:
        return token.column
    return token.column + len(text) - 1


def _parse_compilation_unit(source_text: str) -> ParseResult:
    lexer = Java8Lexer(InputStream(source_text))
    token_stream = CommonTokenStream(lexer)
    parser = Java8Parser(token_stream)

    lexer_listener = CollectingErrorListener()
    parser_listener = CollectingErrorListener()

    lexer.removeErrorListeners()
    parser.removeErrorListeners()
    lexer.addErrorListener(lexer_listener)
    parser.addErrorListener(parser_listener)

    try:
        tree = parser.compilationUnit()
    except Exception as exc:  # noqa: BLE001
        return ParseResult(success=False, tree=None, parser=None, error=f"exception: {exc}")

    all_errors = lexer_listener.errors + parser_listener.errors
    if all_errors:
        return ParseResult(success=False, tree=None, parser=None, error="; ".join(all_errors[:3]))

    return ParseResult(success=True, tree=tree, parser=parser, error=None)


def _build_dot(
    tree: ParserRuleContext,
    parser: Java8Parser,
    graph_comment: str,
) -> tuple[str, GraphStats]:
    lines: list[str] = ["digraph G {", f'graph [comment="{_escape_dot(graph_comment)}"]']
    node_lines: list[str] = []
    edge_lines: list[str] = []

    next_node_index = 0
    token_node_ids: list[str] = []

    syntax_node_count = 0
    syntax_token_count = 0
    child_edge_count = 0

    stack: list[tuple[Any, str | None]] = [(tree, None)]

    while stack:
        node, parent_id = stack.pop()

        node_id = f"n{next_node_index}"
        node_index = next_node_index
        next_node_index += 1

        if isinstance(node, ParserRuleContext):
            syntax_node_count += 1

            rule_index = node.getRuleIndex()
            node_type = (
                parser.ruleNames[rule_index]
                if 0 <= rule_index < len(parser.ruleNames)
                else "unknown"
            )

            start_token = node.start
            end_token = node.stop

            start_line = start_token.line if start_token is not None else -1
            start_col = start_token.column if start_token is not None else -1
            end_line = end_token.line if end_token is not None else start_line
            end_col = _token_end_column(end_token) if end_token is not None else start_col

            attrs = {
                "node_index": node_index,
                "node_kind": "SyntaxNode",
                "node_type": node_type,
                "start_line": start_line,
                "start_col": start_col,
                "end_line": end_line,
                "end_col": end_col,
            }
            node_lines.append(f'"{node_id}" [{_dot_attrs(attrs)}]')

            if parent_id is not None:
                edge_lines.append(f'"{parent_id}" -> "{node_id}" [edge_type="Child"]')
                child_edge_count += 1

            children = [node.getChild(i) for i in range(node.getChildCount())]
            for child in reversed(children):
                stack.append((child, node_id))

            continue

        if isinstance(node, TerminalNode):
            token = getattr(node, "symbol", None)
            if token is None:
                token = node.getSymbol()  # type: ignore[attr-defined]
            if token.type == Token.EOF:
                continue

            syntax_token_count += 1
            token_type = _token_type_name(token.type)
            token_text = token.text or ""
            token_index = len(token_node_ids)

            attrs = {
                "node_index": node_index,
                "node_kind": "SyntaxToken",
                "node_type": token_type,
                "token_text": token_text,
                "token_index": token_index,
                "start_line": token.line,
                "start_col": token.column,
                "end_line": token.line,
                "end_col": _token_end_column(token),
            }
            node_lines.append(f'"{node_id}" [{_dot_attrs(attrs)}]')

            if parent_id is not None:
                edge_lines.append(f'"{parent_id}" -> "{node_id}" [edge_type="Child"]')
                child_edge_count += 1

            token_node_ids.append(node_id)
            continue

        raise TypeError(f"Unsupported parse-tree node type: {type(node)}")

    next_token_edge_count = 0
    for src, dst in zip(token_node_ids, token_node_ids[1:]):
        edge_lines.append(f'"{src}" -> "{dst}" [edge_type="NextToken"]')
        next_token_edge_count += 1

    lines.extend(node_lines)
    lines.extend(edge_lines)
    lines.append("}")

    stats = GraphStats(
        node_count=len(node_lines),
        syntax_node_count=syntax_node_count,
        syntax_token_count=syntax_token_count,
        edge_count=len(edge_lines),
        child_edge_count=child_edge_count,
        next_token_edge_count=next_token_edge_count,
    )

    return "\n".join(lines) + "\n", stats


def _sanitize_stem(file_path: str) -> str:
    stem = Path(file_path).stem
    cleaned = "".join(ch for ch in stem if ch.isalnum() or ch == "_")
    return cleaned or "Snippet"


def _compact_y(y: list[bool]) -> str:
    return "[" + ",".join("true" if value else "false" for value in y) + "]"


def _normalized_output_filename(json_index: int, entry: dict[str, Any]) -> str:
    commit_hash = str(entry.get("commit_hash", "unknown"))
    start_line = int(entry.get("start_line", -1))
    end_line = int(entry.get("end_line", -1))
    y = _compact_y(entry.get("y", []))
    stem = _sanitize_stem(str(entry.get("file_path", "Snippet.java")))
    return f"{json_index}_{commit_hash}_{start_line}_{end_line}_{y}_{stem}.dot"


def _normalized_comment(json_index: int, entry: dict[str, Any]) -> str:
    return (
        f"source=normalized_json json_index={json_index} "
        f"commit_hash={entry.get('commit_hash', '')} "
        f"file_path={entry.get('file_path', '')} "
        f"start_line={entry.get('start_line', -1)} "
        f"end_line={entry.get('end_line', -1)} "
        f"y={entry.get('y', [])}"
    )


def _wrap_snippet_in_class(code_snippet: str) -> str:
    body = code_snippet.rstrip("\n")
    return f"public class SnippetWrapper {{\n{body}\n}}\n"


def _load_normalized_entries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Expected normalized JSON top-level list.")
    return data


def _write_manifest_line(manifest_fh: Any, record: dict[str, Any]) -> None:
    manifest_fh.write(json.dumps(record) + "\n")


def _process_json_task(task: dict[str, Any]) -> dict[str, Any]:
    json_index = int(task["json_index"])
    entry = task["entry"]
    output_path = Path(task["output_path"])
    overwrite = bool(task["overwrite"])

    started = time.perf_counter()
    status = "success"
    error: str | None = None
    stats: GraphStats | None = None

    if output_path.exists() and not overwrite:
        status = "skipped"
    else:
        source_text = _wrap_snippet_in_class(str(entry.get("code_snippet", "")))
        parse_result = _parse_compilation_unit(source_text)

        if parse_result.success and parse_result.tree is not None and parse_result.parser is not None:
            dot_text, stats = _build_dot(
                tree=parse_result.tree,
                parser=parse_result.parser,
                graph_comment=_normalized_comment(json_index, entry),
            )
            output_path.write_text(dot_text)
        else:
            status = "failed"
            error = parse_result.error

    duration_ms = int((time.perf_counter() - started) * 1000)
    return {
        "source_kind": "normalized_json",
        "json_index": json_index,
        "repo_url": entry.get("repo_url", ""),
        "commit_hash": entry.get("commit_hash", ""),
        "file_path": entry.get("file_path", ""),
        "start_line": entry.get("start_line", -1),
        "end_line": entry.get("end_line", -1),
        "y": entry.get("y", []),
        "output_dot": str(output_path),
        "status": status,
        "error": error,
        "graph_stats": asdict(stats) if stats is not None else None,
        "duration_ms": duration_ms,
    }


def run_from_json(args: argparse.Namespace) -> dict[str, Any]:
    entries = _load_normalized_entries(args.input_json)

    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")

    end_index = len(entries) if args.limit is None else min(len(entries), args.start_index + args.limit)
    subset = entries[args.start_index:end_index]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for offset, entry in enumerate(subset):
        json_index = args.start_index + offset
        output_name = _normalized_output_filename(json_index, entry)
        output_path = args.output_dir / output_name
        tasks.append(
            {
                "json_index": json_index,
                "entry": entry,
                "output_path": str(output_path),
                "overwrite": args.overwrite,
            }
        )

    success = 0
    failed = 0
    skipped = 0

    executor: ProcessPoolExecutor | None = None

    if args.workers <= 1:
        records_iter: Any = (_process_json_task(task) for task in tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        records_iter = executor.map(_process_json_task, tasks, chunksize=args.chunksize)

    with args.manifest_path.open("w") as manifest_fh:
        try:
            for processed, record in enumerate(records_iter, start=1):
                _write_manifest_line(manifest_fh, record)

                status = record.get("status")
                if status == "success":
                    success += 1
                elif status == "failed":
                    failed += 1
                elif status == "skipped":
                    skipped += 1

                if processed % args.progress_every == 0:
                    print(
                        f"[from-json] processed={processed}/{len(subset)} "
                        f"success={success} failed={failed} skipped={skipped}"
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    summary = {
        "mode": "from-json",
        "input_json": str(args.input_json),
        "output_dir": str(args.output_dir),
        "manifest_path": str(args.manifest_path),
        "start_index": args.start_index,
        "requested_limit": args.limit,
        "processed": len(subset),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "edge_types": ["Child", "NextToken"],
        "node_types": ["SyntaxNode", "SyntaxToken"],
    }
    return summary


def _dir_comment(input_rel_path: Path) -> str:
    return f"source=directory_file relative_path={input_rel_path.as_posix()}"


def _list_java_files(input_dir: Path) -> list[Path]:
    paths = [path for path in input_dir.rglob("*.java") if path.is_file()]
    paths.sort(key=lambda path: path.as_posix())
    return paths


def _process_dir_task(task: dict[str, Any]) -> dict[str, Any]:
    input_rel_path = Path(task["input_rel_path"])
    source_path = Path(task["source_path"])
    output_path = Path(task["output_path"])
    overwrite = bool(task["overwrite"])

    started = time.perf_counter()
    status = "success"
    error: str | None = None
    stats: GraphStats | None = None

    if output_path.exists() and not overwrite:
        status = "skipped"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_text = source_path.read_text(errors="replace")
        parse_result = _parse_compilation_unit(source_text)

        if parse_result.success and parse_result.tree is not None and parse_result.parser is not None:
            dot_text, stats = _build_dot(
                tree=parse_result.tree,
                parser=parse_result.parser,
                graph_comment=_dir_comment(input_rel_path),
            )
            output_path.write_text(dot_text)
        else:
            status = "failed"
            error = parse_result.error

    duration_ms = int((time.perf_counter() - started) * 1000)
    return {
        "source_kind": "directory_file",
        "input_rel_path": str(input_rel_path),
        "output_dot": str(output_path),
        "status": status,
        "error": error,
        "graph_stats": asdict(stats) if stats is not None else None,
        "duration_ms": duration_ms,
    }


def run_from_dir(args: argparse.Namespace) -> dict[str, Any]:
    print(f"[from-dir] scanning {args.input_dir} for .java files...")
    java_files = _list_java_files(args.input_dir)
    print(f"[from-dir] found {len(java_files)} Java files")
    subset = java_files if args.limit is None else java_files[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for source_path in subset:
        input_rel_path = source_path.relative_to(args.input_dir)
        output_rel_path = input_rel_path.with_suffix(".dot")
        output_path = args.output_dir / output_rel_path
        tasks.append(
            {
                "input_rel_path": str(input_rel_path),
                "source_path": str(source_path),
                "output_path": str(output_path),
                "overwrite": args.overwrite,
            }
        )

    success = 0
    failed = 0
    skipped = 0

    executor: ProcessPoolExecutor | None = None

    print(f"[from-dir] processing {len(tasks)} files with {args.workers} workers...")

    if args.workers <= 1:
        records_iter: Any = (_process_dir_task(task) for task in tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        records_iter = executor.map(_process_dir_task, tasks, chunksize=args.chunksize)

    with args.manifest_path.open("w") as manifest_fh:
        try:
            for processed, record in enumerate(records_iter, start=1):
                _write_manifest_line(manifest_fh, record)

                status = record.get("status")
                if status == "success":
                    success += 1
                elif status == "failed":
                    failed += 1
                elif status == "skipped":
                    skipped += 1

                if processed % args.progress_every == 0:
                    print(
                        f"[from-dir] processed={processed}/{len(subset)} "
                        f"success={success} failed={failed} skipped={skipped}"
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    summary = {
        "mode": "from-dir",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "manifest_path": str(args.manifest_path),
        "requested_limit": args.limit,
        "processed": len(subset),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "edge_types": ["Child", "NextToken"],
        "node_types": ["SyntaxNode", "SyntaxToken"],
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build AST DOT graphs with Child and NextToken edges.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    json_parser = subparsers.add_parser("from-json", help="Generate DOT graphs from normalized JSON snippets.")
    json_parser.add_argument(
        "--input-json",
        type=Path,
        default=ROOT / "data" / "MLCQCodeSmellSamples.normalized.json",
        help="Path to normalized JSON dataset.",
    )
    json_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "preprocessed" / "mlcq" / "dot",
        help="Output directory for DOT files.",
    )
    json_parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT / "data" / "preprocessed" / "mlcq" / "manifest.jsonl",
        help="Path for JSONL manifest.",
    )
    json_parser.add_argument("--start-index", type=int, default=0, help="Starting index in normalized JSON list.")
    json_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of entries to process (default: all from start-index).",
    )
    json_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    json_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default: 1).",
    )
    json_parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Task chunk size for multiprocessing map.",
    )
    json_parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N processed inputs.",
    )

    dir_parser = subparsers.add_parser("from-dir", help="Generate DOT graphs from a directory of Java files.")
    dir_parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing .java files.",
    )
    dir_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root directory (source tree is mirrored).",
    )
    dir_parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path for JSONL manifest.",
    )
    dir_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of Java files to process (default: all).",
    )
    dir_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    dir_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default: 1).",
    )
    dir_parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Task chunk size for multiprocessing map.",
    )
    dir_parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N processed inputs.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "from-json":
        summary = run_from_json(args)
    elif args.mode == "from-dir":
        summary = run_from_dir(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
