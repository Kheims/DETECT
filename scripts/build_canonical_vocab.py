"""Build the canonical node type vocabulary from the ANTLR Java8 grammar.

Extracts all parser rule names and lexer token names from the generated
ANTLR Python classes. The resulting vocab is a static JSON file used by
both the MLCQ dataset builder and the pre-training dataset builder to
ensure type_id indices match across datasets.

Usage:
    python scripts/build_canonical_vocab.py \
        --output config/canonical_node_type_vocab.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.antlr.generated.Java8Lexer import Java8Lexer
from tools.antlr.generated.Java8Parser import Java8Parser


def build_canonical_vocab() -> dict[str, int]:
    """Build a deterministic vocab from the ANTLR grammar.

    Order: parser rule names first (alphabetically), then lexer token names
    (alphabetically), then special entries. This ensures the mapping is
    stable across machines and Python versions.
    """
    # 1. Parser rule names (lowercase, e.g. "compilationUnit", "methodDeclaration")
    # Some ANTLR grammars append '_' to avoid Python keyword conflicts
    # (e.g. emptyStatement_). Include both variants for compatibility.
    raw_rules: set[str] = set(Java8Parser.ruleNames)
    for rule in list(raw_rules):
        if rule.endswith("_"):
            raw_rules.add(rule[:-1])  # also add without trailing underscore
    parser_rules = sorted(raw_rules)

    # 2. Lexer token names: symbolic names + literal names
    symbolic = []
    for name in Java8Lexer.symbolicNames:
        if name and name != "<INVALID>":
            symbolic.append(name)

    literal = []
    for name in Java8Lexer.literalNames:
        if name and name != "<INVALID>":
            literal.append(name)

    # Deduplicate (some tokens appear in both lists)
    lexer_names = sorted(set(symbolic) | set(literal))

    # 3. Special entries
    special = ["unknown"]

    # Combine: parser rules, then lexer tokens, then special
    all_names = parser_rules + lexer_names + special

    # Ensure no duplicates (parser rules are lowercase, lexer tokens uppercase/quoted)
    seen: set[str] = set()
    vocab: dict[str, int] = {}
    for name in all_names:
        if name not in seen:
            vocab[name] = len(vocab)
            seen.add(name)

    return vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical node type vocab.")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "config" / "canonical_node_type_vocab.json",
    )
    args = parser.parse_args()

    vocab = build_canonical_vocab()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(vocab, indent=2))

    # Summary
    parser_count = len([k for k in vocab if k[0].islower() and k != "unknown"])
    lexer_count = len(vocab) - parser_count - 1  # -1 for "unknown"
    print(f"Canonical vocab: {len(vocab)} entries")
    print(f"  Parser rules: {parser_count}")
    print(f"  Lexer tokens: {lexer_count}")
    print(f"  Special: 1 (unknown)")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
