"""
"Create mode": build something new from a description, with no file named.

Uses the same Self-OSS-Instruct rows as selfoss.py (ODC-BY), but turns each one
into an agent session instead of a chat answer:
  user describes a program -> (search finds nothing) -> write_file <name>.py -> summary
v3 had no sessions like this, so asked to "make a calculator" Monkey grepped,
found nothing and gave up. These teach it that a search with no results on a
new-thing request means "create the file", and to pick a sensible filename.
"""

from __future__ import annotations

import random
import warnings
import re

from ..sandbox import Workspace
from ..tools import random_system_prompt
from .commitpack import STOPWORDS, WORD

MIN_CODE_CHARS = 200
MAX_CODE_CHARS = 6000
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
TOP_NAME = re.compile(r"^(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
GENERIC_NAMES = ["main.py", "app.py", "script.py", "solution.py"]

ASK = [
    "{q}",
    "{q}\n\nPut it in a new file.",
    "Make a new Python file for this: {q}",
    "Build this for me: {q}",
    "Create a script that does this:\n\n{q}",
    "Can you write this? {q}",
]
CREATED = ["Created `{path}`.", "Done. I created `{path}`.", "I wrote it in `{path}`.", "`{path}` is ready."]


def _snake(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower().strip("_")
    return name or "main"


def extract_code(response: str) -> str | None:
    """The longest Python block in the answer: the program itself, not a usage snippet."""
    blocks = [b.strip("\n") + "\n" for b in CODE_BLOCK.findall(response)]
    blocks = [b for b in blocks if TOP_NAME.search(b)]
    if not blocks:
        return None
    code = max(blocks, key=len)
    if not (MIN_CODE_CHARS <= len(code) <= MAX_CODE_CHARS) or "\r" in code or "\x00" in code:
        return None
    try:
        with warnings.catch_warnings():  # odd escapes in sample code are fine
            warnings.simplefilter("ignore")
            compile(code, "<create>", "exec")
    except (SyntaxError, ValueError):
        return None
    return code


def choose_path(code: str, rng: random.Random, taken: set[str]) -> str:
    """Name the file after the main function or class, like a person would."""
    m = TOP_NAME.search(code)
    name = f"{_snake(m.group(1))}.py" if m and rng.random() < 0.8 else rng.choice(GENERIC_NAMES)
    return name if name not in taken else f"new_{name}"


def _explanation(response: str) -> str:
    """A short first paragraph of the answer's prose after the code, if there is one."""
    parts = CODE_BLOCK.split(response)
    for prose in parts[2::2] if len(parts) > 2 else []:
        para = prose.strip().split("\n\n")[0].strip()
        if 20 <= len(para) <= 300 and "```" not in para:
            return para
    return ""


def search_word(request: str) -> str | None:
    """What a person would search for: the name they asked for, else the most specific word."""
    named = re.findall(r"`([A-Za-z_][A-Za-z0-9_]{3,})(?:\(.*?\))?`", request)
    if named:
        return named[0]
    words = [w.strip(".").lower() for w in WORD.findall(request)]
    words = [w for w in words if w.isalpha() and len(w) >= 5 and w not in STOPWORDS]
    return max(words, key=len) if words else None


def _call(ws: Workspace, messages: list[dict], name: str, args: dict) -> str:
    messages.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]})
    result = ws.call(name, args)
    messages.append({"role": "tool", "content": result})
    return result


def make_session(row: dict, rng: random.Random, decoys: list[tuple[str, str]] | None = None) -> dict | None:
    q, a = (row.get("instruction") or "").strip(), (row.get("response") or "").strip()
    code = extract_code(a) if q else None
    if code is None or len(q) > 1500:
        return None
    files = dict(decoys or []) if rng.random() < 0.6 else {}
    path = choose_path(code, rng, set(files))
    ws = Workspace(files)
    messages: list[dict] = [{"role": "system", "content": random_system_prompt(rng, rng.choice(["project", "app", "scratch", "tools"]))},
                            {"role": "user", "content": rng.choice(ASK).format(q=q)}]

    # Sometimes look first, find nothing, and build it anyway: the step v3 never learned.
    word = search_word(q)
    roll = rng.random()
    if word and roll < 0.25:
        if _call(ws, messages, "grep", {"pattern": word, **({"glob": "*.py"} if rng.random() < 0.3 else {})}) != "No matches.":
            return None
    elif word and roll < 0.45:
        if _call(ws, messages, "glob", {"pattern": f"**/*{word}*"}) != "No files matched.":
            return None

    if not _call(ws, messages, "write_file", {"path": path, "content": code}).startswith("Created"):
        return None
    done = rng.choice(CREATED).format(path=path)
    why = _explanation(a)
    messages.append({"role": "assistant", "content": f"{done} {why}" if why else done})
    return {"source": "create", "messages": messages}


def make_eval_task(row: dict, rng: random.Random, decoys: list[tuple[str, str]] | None = None) -> dict | None:
    """A held-out build-from-scratch task: done when a new file exists and compiles."""
    q, a = (row.get("instruction") or "").strip(), (row.get("response") or "").strip()
    code = extract_code(a) if q else None
    if code is None or len(q) > 1500:
        return None
    m = TOP_NAME.search(code)
    return {"kind": "create", "system": random_system_prompt(rng, "project"),
            "request": rng.choice(ASK).format(q=q), "files": dict(decoys or []),
            "target_path": "(new file)", "expected": code, "expected_name": m.group(1) if m else "",
            "path_given": False, "lang": "Python"}
