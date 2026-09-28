"""Tokenise sessions and build the loss mask: learn only from Monkey's own turns."""

from __future__ import annotations

from .render import render
from .tools import TOOLS

IGNORE = -100


def encode_session(session: dict, tokenizer, max_len: int) -> dict | None:
    """input_ids/attention_mask/labels, or None if too long or nothing to learn."""
    text, spans = render(session["messages"], session.get("tools") or TOOLS)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    labels = [IGNORE] * len(ids)
    j = 0
    for i, (start, _end) in enumerate(enc["offset_mapping"]):
        while j < len(spans) and start >= spans[j][1]:
            j += 1
        if j < len(spans) and spans[j][0] <= start:
            labels[i] = ids[i]
    if all(l == IGNORE for l in labels):
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
