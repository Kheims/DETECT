"""Build tokenized dataset from MLCQ normalized JSON for sequence models.

Pipeline:
1. Load normalized JSON with code snippets and multi-label y vectors
2. Tokenize each snippet (whitespace + punctuation split)
3. Build vocabulary
4. Convert to padded tensor sequences
5. Save as PyTorch dataset
"""

import json
import re
import sys
import os
from collections import Counter

import torch
from torch.utils.data import TensorDataset


def tokenize_code(code, max_tokens=512):
    """Simple tokenizer: split on whitespace and punctuation, keep meaningful tokens."""
    tokens = re.findall(r'[a-zA-Z_]\w*|[{}()\[\];,.<>=!&|+\-*/:%^~?@#]|"[^"]*"|\'[^\']*\'|\d+', code)
    return tokens[:max_tokens]


def build_vocab(all_tokens, min_freq=2, max_vocab=10000):
    """Build vocabulary from token list. Reserve 0=PAD, 1=UNK."""
    counter = Counter(all_tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counter.most_common(max_vocab - 2):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def tokens_to_ids(tokens, vocab, max_len=512):
    """Convert token list to padded ID tensor."""
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids[:max_len]


def build_token_dataset(input_json, output_dir, max_tokens=512, min_freq=2, max_vocab=10000):
    """Build tokenized dataset from normalized JSON."""
    print(f"Loading {input_json}...")
    with open(input_json, "r") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")

    # Tokenize all snippets
    print("Tokenizing...")
    all_tokens_flat = []
    sample_tokens = []
    for sample in samples:
        tokens = tokenize_code(sample["code_snippet"], max_tokens=max_tokens)
        sample_tokens.append(tokens)
        all_tokens_flat.extend(tokens)

    # Build vocabulary
    vocab = build_vocab(all_tokens_flat, min_freq=min_freq, max_vocab=max_vocab)
    print(f"Vocabulary size: {len(vocab)}")

    # Convert to tensors
    print("Converting to tensors...")
    input_ids = []
    labels = []
    for tokens, sample in zip(sample_tokens, samples):
        ids = tokens_to_ids(tokens, vocab, max_len=max_tokens)
        input_ids.append(ids)
        labels.append(sample["y"])

    X = torch.tensor(input_ids, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    dataset = TensorDataset(X, y)
    torch.save(dataset, os.path.join(output_dir, "token_dataset.pt"))

    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    meta = {
        "num_samples": len(samples),
        "vocab_size": len(vocab),
        "max_tokens": max_tokens,
        "label_order": ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"],
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved token dataset ({len(samples)} samples, vocab {len(vocab)}) to {output_dir}")
    return dataset, vocab, meta


if __name__ == "__main__":
    input_json = sys.argv[1] if len(sys.argv) > 1 else "data/MLCQCodeSmellSamples.normalized.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "artifacts/token_dataset"
    build_token_dataset(input_json, output_dir)
