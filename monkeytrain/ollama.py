"""Minimal Ollama client (stdlib only) and the Modelfile Monkey ships with."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .render import OLLAMA_TEMPLATE

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST


def _post(path: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise SystemExit(f"Can't reach Ollama at {HOST} ({e}). Is Ollama installed and running?") from e


def options(threads: int | None, ctx: int | None, **extra) -> dict:
    opts = dict(extra)
    if threads:
        opts["num_thread"] = threads
    if ctx:
        opts["num_ctx"] = ctx
    return opts


def chat(model: str, messages: list[dict], tools: list[dict], opts: dict) -> dict:
    msgs = [{k: m[k] for k in ("role", "content", "tool_calls") if k in m} for m in messages]
    return _post("/api/chat", {"model": model, "messages": msgs, "tools": tools,
                               "stream": False, "options": opts, "keep_alive": "10m"})


def generate(model: str, prompt: str, opts: dict) -> dict:
    # raw: the prompt is already in chat format, skip the Modelfile template
    return _post("/api/generate", {"model": model, "prompt": prompt, "raw": True, "stream": False,
                                   "options": opts, "keep_alive": "10m"})


def modelfile(gguf_path: str, ctx: int = 8192, threads: int | None = None) -> str:
    lines = [
        f"FROM {gguf_path}",
        f'TEMPLATE """{OLLAMA_TEMPLATE}"""',
        "PARAMETER stop <|im_end|>",
        "PARAMETER stop <|endoftext|>",
        "PARAMETER temperature 0.2",
        f"PARAMETER num_ctx {ctx}",
    ]
    if threads:
        lines.append(f"PARAMETER num_thread {threads}")
    return "\n".join(lines) + "\n"
