"""Minimal Ollama client (stdlib only) and the Modelfile Monkey ships with."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .contract import SAMPLING, STOP
from .render import OLLAMA_TEMPLATE

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST


class OllamaError(RuntimeError):
    """Ollama answered with an error for this request (e.g. it aborted a looping reply)."""


def _post(path: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:  # Ollama answered, but with an error
        detail = e.read().decode(errors="replace")
        if e.code == 404:
            raise SystemExit(f"Ollama doesn't have model '{body.get('model')}'. Check `ollama list`; for a "
                             f"Monkey model run `python -m monkeytrain.install --config configs/<model>.json` "
                             f"first. ({detail})") from e
        raise OllamaError(f"Ollama returned HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Can't reach Ollama at {HOST} ({e.reason}). Is Ollama installed and running?") from e


def find_ollama() -> str | None:
    """The ollama executable: PATH first, then the default Windows install folder
    (a terminal opened before installing Ollama doesn't see its PATH entry)."""
    import shutil
    found = shutil.which("ollama")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        exe = os.path.join(local, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(exe):
            return exe
    return None


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
    lines = [f"FROM {gguf_path}", f'TEMPLATE """{OLLAMA_TEMPLATE}"""']
    lines += [f"PARAMETER stop {s}" for s in STOP]
    # Sampling shared with MonkeyPaw via the contract; repeat_penalty 1.0 keeps edit_file copies exact.
    lines += [f"PARAMETER {k} {v}" for k, v in SAMPLING.items()]
    lines.append(f"PARAMETER num_ctx {ctx}")
    if threads:
        lines.append(f"PARAMETER num_thread {threads}")
    return "\n".join(lines) + "\n"
