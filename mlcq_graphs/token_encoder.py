from __future__ import annotations

import json
import shlex
from pathlib import Path
import time


def _parse_value(raw: str):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_attrs(attr_str: str) -> dict[str, object]:
    attrs: dict[str, object] = {}
    for token in shlex.split(attr_str, posix=True):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        attrs[key] = _parse_value(value)
    return attrs


def _safe_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _safe_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def sentence_from_dot(dot_path: Path) -> list[str]:
    token_rows: list[tuple[int, str]] = []
    for raw_line in dot_path.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith('"n') or '" -> "' in line:
            continue
        if "[" not in line or "]" not in line:
            continue

        left = line.index("[")
        right = line.rindex("]")
        attrs = _parse_attrs(line[left + 1 : right])
        if str(attrs.get("node_kind", "")) != "SyntaxToken":
            continue

        token_text = str(attrs.get("token_text", "")).strip()
        if token_text == "":
            continue

        token_index = _safe_int(attrs.get("token_index", len(token_rows)), len(token_rows))
        token_rows.append((token_index, token_text))

    token_rows.sort(key=lambda item: item[0])
    return [token for _, token in token_rows]


def build_corpus(
    dot_dir: Path,
    max_graphs: int | None,
    progress_every: int,
) -> tuple[list[list[str]], dict[str, object]]:
    sentences: list[list[str]] = []
    dot_paths = sorted(dot_dir.glob("*.dot"))
    if max_graphs is not None:
        dot_paths = dot_paths[:max_graphs]

    total = len(dot_paths)
    token_count = 0
    started = time.perf_counter()
    progress_every = max(1, int(progress_every))
    print(f"[token_encoder] corpus_scan start total_dot={total}")

    for processed, dot_path in enumerate(dot_paths, start=1):
        sentence = sentence_from_dot(dot_path)
        if sentence:
            sentences.append(sentence)
            token_count += len(sentence)

        should_log = (
            processed % progress_every == 0
            or processed == total
            or processed == 1
        )
        if should_log:
            elapsed = time.perf_counter() - started
            rate = processed / max(1e-9, elapsed)
            print(
                f"[token_encoder] corpus_scan processed={processed}/{total} "
                f"sentences={len(sentences)} tokens={token_count} rate={rate:.2f} dot/s"
            )

    elapsed_sec = time.perf_counter() - started
    corpus_meta = {
        "total_dot_files": total,
        "processed_dot_files": total,
        "num_sentences": len(sentences),
        "num_tokens": token_count,
        "corpus_scan_elapsed_sec": round(elapsed_sec, 3),
    }
    return sentences, corpus_meta


def train_word2vec_from_dot(
    dot_dir: Path,
    output_dir: Path,
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    epochs: int,
    progress_every: int,
    max_graphs: int | None,
) -> dict[str, object]:
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise ImportError(
            "gensim is required for token encoder training. Install dependencies with `uv sync`."
        ) from exc

    total_started = time.perf_counter()
    sentences, corpus_meta = build_corpus(
        dot_dir=dot_dir,
        max_graphs=max_graphs,
        progress_every=progress_every,
    )
    if not sentences:
        raise ValueError(f"No token sequences available in {dot_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[token_encoder] train_word2vec start "
        f"sentences={len(sentences)} vector_size={vector_size} window={window} "
        f"min_count={min_count} workers={workers} epochs={epochs}"
    )
    train_started = time.perf_counter()
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,
    )
    train_elapsed = time.perf_counter() - train_started
    print(
        f"[token_encoder] train_word2vec done elapsed_sec={train_elapsed:.2f} "
        f"vocab_size={len(model.wv.index_to_key)}"
    )

    model_path = output_dir / "word2vec.model"
    vectors_path = output_dir / "word2vec.kv"
    vocab_path = output_dir / "vocab.json"
    meta_path = output_dir / "meta.json"

    model.save(str(model_path))
    model.wv.save(str(vectors_path))
    vocab = list(model.wv.index_to_key)
    vocab_path.write_text(json.dumps(vocab, indent=2))

    total_elapsed = time.perf_counter() - total_started
    num_tokens = _safe_int(corpus_meta.get("num_tokens"), 0)
    corpus_scan_elapsed_sec = _safe_float(corpus_meta.get("corpus_scan_elapsed_sec"), 0.0)
    meta = {
        "method": "word2vec",
        "dot_dir": str(dot_dir),
        "num_graphs": len(sentences),
        "num_tokens": num_tokens,
        "vocab_size": len(vocab),
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "workers": workers,
        "epochs": epochs,
        "progress_every": progress_every,
        "corpus_scan_elapsed_sec": corpus_scan_elapsed_sec,
        "train_elapsed_sec": round(train_elapsed, 3),
        "total_elapsed_sec": round(total_elapsed, 3),
        "model_path": str(model_path),
        "vectors_path": str(vectors_path),
        "vocab_path": str(vocab_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
