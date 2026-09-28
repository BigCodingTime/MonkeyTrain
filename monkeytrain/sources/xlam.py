"""
xLAM function calling 60k (Salesforce/xlam-function-calling-60k, CC-BY-4.0):
user requests with arbitrary API tools and the correct calls. Teaches picking
the right tool and filling arguments correctly for tools the model hasn't seen.

The dataset is gated: accept its terms on its Hugging Face page with the same
account as your token, or the builder skips it.
"""

from __future__ import annotations

import json
import random
from typing import Iterator

from ..tools import random_system_prompt

REPO = "Salesforce/xlam-function-calling-60k"

_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
          "list": "array", "dict": "object"}


def iter_rows(token: str | None = None) -> Iterator[dict]:
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi(token=token).list_repo_files(REPO, repo_type="dataset") if f.endswith(".json")]
    for f in files:
        data = json.loads(open(hf_hub_download(REPO, f, repo_type="dataset", token=token), encoding="utf-8").read())
        yield from data


def _convert_tool(t: dict) -> dict:
    props, required = {}, []
    for name, spec in (t.get("parameters") or {}).items():
        raw = str(spec.get("type", "string"))
        optional = "optional" in raw
        base = raw.split(",")[0].strip().split("[")[0].lower()
        props[name] = {"type": _TYPES.get(base, "string"), "description": str(spec.get("description", ""))}
        if not optional:
            required.append(name)
    return {"type": "function", "function": {
        "name": t["name"], "description": t.get("description", ""),
        "parameters": {"type": "object", "required": required, "properties": props}}}


def make_session(row: dict, rng: random.Random) -> dict | None:
    try:
        tools = [_convert_tool(t) for t in json.loads(row["tools"])]
        answers = json.loads(row["answers"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    if not tools or not answers:
        return None
    calls = [{"function": {"name": a["name"], "arguments": a.get("arguments", {})}} for a in answers]
    return {"source": "xlam", "tools": tools, "messages": [
        {"role": "system", "content": random_system_prompt(rng)},
        {"role": "user", "content": row["query"]},
        {"role": "assistant", "content": "", "tool_calls": calls},
    ]}
