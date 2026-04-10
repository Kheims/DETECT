"""Sequence-based DL models for multi-label code smell detection using token representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, num_layers=1, dropout=0.3, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * direction_factor, num_labels)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        output, (hidden, _) = self.lstm(emb)
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(self.dropout(hidden))


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        output, _ = self.lstm(emb)
        # attention weights
        attn_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)
        return self.fc(self.dropout(context))


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels, num_filters=128, filter_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, x):
        emb = self.dropout(self.embedding(x)).transpose(1, 2)
        conv_outs = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        cat = torch.cat(conv_outs, dim=1)
        return self.fc(self.dropout(cat))


SEQUENCE_MODELS = {
    "lstm": LSTMClassifier,
    "bilstm": lambda **kw: LSTMClassifier(bidirectional=True, **kw),
    "bilstm_attention": BiLSTMAttentionClassifier,
    "cnn": CNNClassifier,
}


def build_sequence_model(model_name, cfg):
    if model_name not in SEQUENCE_MODELS:
        raise ValueError(f"Unknown sequence model: {model_name}. Available: {list(SEQUENCE_MODELS.keys())}")

    common = {
        "vocab_size": cfg["vocab_size"],
        "embed_dim": cfg.get("embed_dim", 128),
        "num_labels": cfg.get("num_labels", 4),
    }

    if model_name == "cnn":
        return CNNClassifier(
            num_filters=cfg.get("num_filters", 128),
            filter_sizes=tuple(cfg.get("filter_sizes", [3, 4, 5])),
            dropout=cfg.get("dropout", 0.3),
            **common,
        )
    else:
        kw = {
            "hidden_dim": cfg.get("hidden_dim", 256),
            "num_layers": cfg.get("num_layers", 2),
            "dropout": cfg.get("dropout", 0.3),
            **common,
        }
        if model_name == "bilstm":
            return LSTMClassifier(bidirectional=True, **kw)
        return SEQUENCE_MODELS[model_name](**kw)
