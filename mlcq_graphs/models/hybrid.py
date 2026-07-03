"""Late-fusion model: GNN over the AST and BiLSTM with attention over the tokens.

branch="hybrid" concatenates the two vectors, branch="hybrid_weighted" mixes
the logits of two heads with one learned weight per label.
"""

from __future__ import annotations

import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from mlcq_graphs.models.base import BaseGraphClassifier

BRANCHES = ("graph", "seq", "hybrid", "hybrid_weighted")

# Same tokenizer as scripts/build_token_dataset.py.
_TOKEN_RE = re.compile(r'[a-zA-Z_]\w*|[{}()\[\];,.<>=!&|+\-*/:%^~?@#]|"[^"]*"|\'[^\']*\'|\d+')
PAD, UNK = 0, 1


def tokenize_code(code: str, max_tokens: int) -> list[str]:
    return _TOKEN_RE.findall(code)[:max_tokens]


def build_vocab(token_lists: list[list[str]], min_freq: int, max_vocab: int) -> dict[str, int]:
    """Vocabulary of the training snippets only. 0 is padding, 1 is unknown."""
    counter = Counter(tok for tokens in token_lists for tok in tokens)
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    for token, count in counter.most_common(max_vocab - 2):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens: list[str], vocab: dict[str, int], max_tokens: int) -> torch.Tensor:
    ids = [vocab.get(tok, UNK) for tok in tokens][:max_tokens]
    ids += [PAD] * (max_tokens - len(ids))
    return torch.tensor(ids, dtype=torch.long).view(1, -1)


class SeqAttentionEncoder(nn.Module):
    """BiLSTM with additive attention, without the output layer."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        mask_padding: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.mask_padding = mask_padding
        self.out_dim = hidden_dim * 2

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(ids))
        output, _ = self.lstm(emb)
        scores = self.attention(output).squeeze(-1)
        if self.mask_padding:
            scores = scores.masked_fill(ids == PAD, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), output).squeeze(1)
        return self.dropout(context)


class HybridClassifier(nn.Module):
    """Graph vector (GNN encoder + readout) and/or sequence vector, then a head."""

    def __init__(
        self,
        branch: str,
        gnn: BaseGraphClassifier | None,
        seq: SeqAttentionEncoder | None,
        hidden_dim: int,
        num_labels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if branch not in BRANCHES:
            raise ValueError(f"branch must be one of {BRANCHES}, got {branch}")
        uses_graph = branch != "seq"
        uses_seq = branch != "graph"
        if uses_graph and gnn is None:
            raise ValueError(f"branch={branch} needs a GNN encoder")
        if uses_seq and seq is None:
            raise ValueError(f"branch={branch} needs a sequence encoder")
        self.branch = branch
        self.gnn = gnn if uses_graph else None
        self.seq = seq if uses_seq else None
        self.dropout = dropout
        graph_dim = self.gnn.readout.out_dim + self.gnn.num_graph_feats if uses_graph else 0
        seq_dim = self.seq.out_dim if uses_seq else 0
        if branch == "hybrid_weighted":
            self.graph_head = nn.Sequential(nn.Linear(graph_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_labels))
            self.seq_head = nn.Sequential(nn.Linear(seq_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_labels))
            # sigmoid(0) = 0.5: both branches start with the same weight
            self.mix_logit = nn.Parameter(torch.zeros(num_labels))
        else:
            self.lin1 = nn.Linear(graph_dim + seq_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, num_labels)

    def graph_vector(self, data: Data) -> torch.Tensor:
        x = self.gnn.encode_nodes(data)
        g = self.gnn.readout(x, data.batch, getattr(data, "pool_mask", None))
        if self.gnn.num_graph_feats:
            g = torch.cat([g, data.graph_feat.view(g.size(0), -1)], dim=-1)
        return g

    def mix_weights(self) -> torch.Tensor:
        """Weight of the graph branch per label (hybrid_weighted only)."""
        return torch.sigmoid(self.mix_logit)

    def forward(self, data: Data) -> torch.Tensor:
        if self.branch == "hybrid_weighted":
            w = self.mix_weights()
            return w * self.graph_head(self.graph_vector(data)) + (1 - w) * self.seq_head(self.seq(data.seq_ids))
        parts = []
        if self.gnn is not None:
            parts.append(self.graph_vector(data))
        if self.seq is not None:
            parts.append(self.seq(data.seq_ids))
        h = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)
