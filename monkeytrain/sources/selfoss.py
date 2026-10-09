"""
StarCoder2 Self-OSS-Instruct (bigcode/self-oss-instruct-sc2-exec-filter-50k,
ODC-BY): coding questions with answers written by an open model and checked by
running them. Teaches Monkey to answer directly when no tool is needed, and
keeps its general coding ability from drifting during fine-tuning.
"""

from __future__ import annotations

import random
from typing import Iterator

from ..tools import random_system_prompt

REPO = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
MAX_RESPONSE_CHARS = 6000

# Monkey writes a file when asked for code; it answers in chat only when the
# user says so. A bare "Write a function..." belongs to create.py, so here the
# request always carries one of these cues.
ASK_CHAT = [
    "Just show me the code here, no need to create a file: {q}",
    "Don't create any files, just answer in chat. {q}",
    "Explain how you'd do this, with code: {q}",
    "Quick question (no files): {q}",
    "How would I do this in Python? Show me here.\n\n{q}",
    "Show me in chat how to do this: {q}",
]
CHAT_CUES = ("no need to create a file", "don't create any files", "explain how", "no files", "show me here", "show me in chat")


def iter_rows(token: str | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi(token=token).list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
    for f in sorted(files):
        table = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset", token=token),
                              columns=["instruction", "response"])
        yield from table.to_pylist()


def make_session(row: dict, rng: random.Random) -> dict | None:
    q, a = (row.get("instruction") or "").strip(), (row.get("response") or "").strip()
    if not q or not a or len(a) > MAX_RESPONSE_CHARS:
        return None
    return {"source": "selfoss", "messages": [
        {"role": "system", "content": random_system_prompt(rng)},
        {"role": "user", "content": rng.choice(ASK_CHAT).format(q=q)},
        {"role": "assistant", "content": a},
    ]}
