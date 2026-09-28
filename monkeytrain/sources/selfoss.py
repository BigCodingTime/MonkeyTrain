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
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}
