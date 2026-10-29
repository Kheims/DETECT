from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_max(reps: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
    """Max over the chunk axis of [B, C, H], ignoring padding chunks (chunk_mask [B, C], bool)."""
    fill = torch.finfo(reps.dtype).min
    return reps.masked_fill(~chunk_mask.unsqueeze(-1), fill).max(dim=1).values


class _ChunkPooling(nn.Module):
    """Shared forward: encode the real chunks, max-pool them, dropout, linear head (self.fc)."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, ids: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        """ids [B, C, L], chunk_mask [B, C] -> logits [B, num_labels]."""
        b, c, length = ids.shape
        flat = chunk_mask.reshape(-1)
        reps = ids.new_zeros((b * c, self.fc.in_features), dtype=self.fc.weight.dtype)
        reps[flat] = self.encode(ids.reshape(b * c, length)[flat]).to(reps.dtype)
        pooled = masked_max(reps.view(b, c, -1), chunk_mask)
        return self.fc(self.dropout(pooled))


class ChunkedRNN(_ChunkPooling):
    """LSTM, BiLSTM, BiLSTM with attention or GRU applied to every chunk, then chunk max-pooling.

    The chunk encoder follows models/sequence.py: final hidden state (concatenated over the two
    directions) for the LSTM, the BiLSTM and the GRU (unidirectional), attention-weighted sum of
    the outputs for the attention model. Padding positions inside a chunk are not packed, as in
    models/sequence.py.
    """

    def __init__(self, kind: str, vocab_size: int, embed_dim: int, hidden_dim: int, num_labels: int,
                 num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        if kind not in ("lstm", "bilstm", "bilstm_attention", "gru"):
            raise ValueError(f"unknown recurrent model {kind!r}")
        self.kind = kind
        bidirectional = kind in ("bilstm", "bilstm_attention")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                       dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = nn.Linear(out_dim, 1) if kind == "bilstm_attention" else None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_labels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[N, L] token ids of N chunks -> [N, out_dim]."""
        emb = self.dropout(self.embedding(x))
        output, hidden = self.rnn(emb)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if self.attention is not None:
            weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)
            return torch.bmm(weights.unsqueeze(1), output).squeeze(1)
        if self.rnn.bidirectional:
            return torch.cat([hidden[-2], hidden[-1]], dim=1)
        return hidden[-1]


class ChunkedCNN(_ChunkPooling):
    """TextCNN of models/sequence.py on every chunk (ReLU, max over positions per filter size),
    then chunk max-pooling. Padding positions take part in the convolution, as in models/sequence.py."""

    def __init__(self, vocab_size: int, embed_dim: int, num_labels: int, num_filters: int = 128,
                 filter_sizes: tuple[int, ...] = (3, 4, 5), dropout: float = 0.3) -> None:
        super().__init__()
        self.kind = "cnn"
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[N, L] token ids of N chunks -> [N, num_filters * len(filter_sizes)]."""
        emb = self.dropout(self.embedding(x)).transpose(1, 2)
        return torch.cat([F.relu(conv(emb)).max(dim=2).values for conv in self.convs], dim=1)


class ChunkedEncoderClassifier(nn.Module):
    """Pretrained encoder on every chunk, first-token ([CLS]) vector per chunk, chunk max-pooling, linear head."""

    def __init__(self, encoder: nn.Module, num_labels: int, dropout: float, init_std: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(int(encoder.config.hidden_size), num_labels)
        nn.init.normal_(self.head.weight, std=init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids: torch.Tensor, attention_mask: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        """ids and attention_mask [B, C, L], chunk_mask [B, C] -> logits [B, num_labels]."""
        b, c, length = ids.shape
        flat = chunk_mask.reshape(-1)
        hidden = self.encoder(input_ids=ids.reshape(b * c, length)[flat],
                              attention_mask=attention_mask.reshape(b * c, length)[flat]).last_hidden_state
        cls = hidden[:, 0]
        reps = cls.new_zeros((b * c, cls.shape[-1]))
        reps[flat] = cls
        pooled = masked_max(reps.view(b, c, -1), chunk_mask)
        return self.head(self.dropout(pooled))
