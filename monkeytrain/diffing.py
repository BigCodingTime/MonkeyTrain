"""
Turn a before/after file pair into a list of edit_file calls.

Each edit's old_string is expanded with context lines until it is unique in the
file at the moment it is applied, the same rule the model has to follow.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

MERGE_GAP = 2        # hunks separated by <= this many unchanged lines become one edit
MAX_CONTEXT = 6      # give up on a hunk that isn't unique with this much context


@dataclass
class Edit:
    old: str
    new: str


def _hunks(a: list[str], b: list[str]) -> list[tuple[int, int, int, int]]:
    ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    merged: list[list[int]] = []
    for _, i1, i2, j1, j2 in ops:
        if merged and i1 - merged[-1][1] <= MERGE_GAP:
            merged[-1][1], merged[-1][3] = i2, j2
        else:
            merged.append([i1, i2, j1, j2])
    return [tuple(h) for h in merged]


def _trim_newline(old: str, new: str) -> tuple[str, str]:
    if old.endswith("\n") and new.endswith("\n"):
        return old[:-1], new[:-1]
    return old, new


def compute_edits(old_text: str, new_text: str) -> list[Edit] | None:
    """Edits that turn old_text into new_text, or None if that can't be done cleanly."""
    a = old_text.splitlines(keepends=True)
    b = new_text.splitlines(keepends=True)
    current = old_text
    edits: list[Edit] = []
    # Every hunk is expressed with lines from the original file, but b's lines are
    # interleaved: take new lines from b[j1:j2] and context from a.
    for i1, i2, j1, j2 in _hunks(a, b):
        found = None
        for ctx in range(0, MAX_CONTEXT + 1):
            for before, after in ((ctx, ctx), (ctx, ctx + 1), (ctx + 1, ctx)):
                lo, hi = max(0, i1 - before), min(len(a), i2 + after)
                old = "".join(a[lo:hi])
                new = "".join(a[lo:i1] + b[j1:j2] + a[i2:hi])
                old, new = _trim_newline(old, new)
                if old.strip() and old != new and current.count(old) == 1:
                    found = Edit(old, new)
                    break
            if found:
                break
        if found is None:
            return None
        current = current.replace(found.old, found.new, 1)
        edits.append(found)
    if current != new_text or not edits:
        return None
    return edits
