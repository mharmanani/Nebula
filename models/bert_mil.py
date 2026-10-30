"""BERT-based multiple-instance aggregator."""

import torch
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder


class BertMILAggregator(torch.nn.Module):
    """A [CLS] token + one token per instance through a BERT encoder; returns sequence logits (from [CLS]) and per-token
    logits."""

    def __init__(
        self,
        embedding_dim,
        num_classes=None,
        num_token_classes=None,
        pooling_strategy="cls",
        attn_implementation="sdpa",
        detach_token_states_for_classification=False,
        **bert_kwargs,
    ):
        super().__init__()
        self.detach_token_states_for_classification = detach_token_states_for_classification
        self.pad_token = torch.nn.Parameter(torch.randn(1, embedding_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, embedding_dim))
        self.bert = BertEncoder(BertConfig(**bert_kwargs, hidden_size=embedding_dim, attn_implementation=attn_implementation))
        self.pooling_strategy = pooling_strategy
        self.classifier = torch.nn.Linear(embedding_dim, num_classes) if num_classes else None
        self.token_classifier = torch.nn.Linear(embedding_dim, num_token_classes) if num_token_classes else None

    def forward(self, hidden_states):
        """hidden_states: list of (N_i, D) tensors, one per case."""
        lengths = [len(h) for h in hidden_states]
        hidden_states = [torch.cat([self.cls_token, h], dim=0) for h in hidden_states]
        pad_length = max(len(h) for h in hidden_states)
        padded, masks = [], []
        for h in hidden_states:
            padding = pad_length - len(h)
            padded.append(torch.cat([h, self.pad_token.expand(padding, -1)], dim=0))
            masks.append(torch.cat([torch.ones(len(h)), torch.zeros(padding)], dim=0))
        padded = torch.stack(padded)
        mask = torch.stack(masks).bool().to(padded.device)
        bsz, seq_len = mask.shape
        mask = mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len)

        out = self.bert(padded, attention_mask=mask, return_dict=True).last_hidden_state
        cls_state = out[:, 0] if self.pooling_strategy == "cls" else out.mean(1)
        logits = self.classifier(cls_state) if self.classifier is not None else None
        token_states, token_logits = [], []
        for i, n in enumerate(lengths):
            t = out[i, 1 : 1 + n]
            token_states.append(t)
            if self.token_classifier is None:
                token_logits.append(None)
            else:
                token_logits.append(self.token_classifier(t.detach() if self.detach_token_states_for_classification else t))
        return {
            "logits": logits,
            "cls_hidden_state": cls_state,
            "token_hidden_states": token_states,
            "token_logits": token_logits,
        }
