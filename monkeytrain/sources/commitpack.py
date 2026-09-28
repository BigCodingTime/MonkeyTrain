"""
CommitPackFT (bigcode/commitpackft, MIT): real GitHub commits with the file
before and after, plus the commit message.

Each commit becomes an agent session in MonkeyPaw's format:
  user asks for the change -> (grep to find the file) -> read_file -> edit_file x N -> summary
Tool results come from running the calls in the sandbox, so they're exactly what
MonkeyPaw would return.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterator

from ..diffing import compute_edits
from ..sandbox import Workspace
from ..tools import random_system_prompt

REPO = "bigcode/commitpackft"

# commitpackft folder name -> share of commit sessions
LANGUAGES: dict[str, float] = {
    "python": 0.28, "javascript": 0.14, "typescript": 0.10, "java": 0.08, "go": 0.06,
    "rust": 0.05, "c++": 0.04, "c": 0.04, "c#": 0.05, "php": 0.03, "ruby": 0.02,
    "html": 0.03, "css": 0.02, "shell": 0.02, "markdown": 0.02, "json": 0.01, "yaml": 0.01,
}

MAX_LINES = 250
MAX_CHARS = 12_000
MAX_EDITS = 4
SKIP_SUBJECT = re.compile(r"^\s*(merge|bump|release|revert|wip|version|update changelog)\b", re.I)
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{3,}")
STOPWORDS = {
    "that", "this", "with", "from", "when", "into", "have", "make", "more", "less", "some", "only",
    "should", "instead", "also", "then", "than", "them", "they", "were", "will", "would", "could",
    "add", "adds", "added", "remove", "removed", "update", "updated", "change", "changed", "fix",
    "fixed", "fixes", "use", "used", "uses", "using", "allow", "allows", "support", "handle", "check",
    "move", "rename", "refactor", "improve", "clean", "cleanup", "code", "file", "files", "test",
    "tests", "none", "true", "false", "return", "self", "there", "about", "after", "before", "which",
}

ASK_WITH_PATH = [
    "{req} in {path}",
    "{req}\n\nFile: {path}",
    "In `{path}`: {req}",
    "Can you change {path}? {req}",
    "{path}: {req}",
]
ASK_NO_PATH = ["{req}", "{req}, please", "Please {req_lc}", "I need this change: {req}"]
DONE = [
    "Done. {subj} ({path}).",
    "Updated `{path}`: {subj_lc}.",
    "I made the change in `{path}`: {subj_lc}.",
    "Done — {subj_lc} in `{path}`.",
]
CREATE_DONE = ["Created `{path}`.", "Done. I created `{path}`.", "`{path}` is created."]


def download(lang: str, token: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, f"data/{lang}/data.jsonl", repo_type="dataset", token=token))


def iter_records(path: Path, rng: random.Random) -> Iterator[dict]:
    """Yield records in a random order (the file is grouped by repository)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rng.shuffle(lines)
    for line in lines:
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _clean(text: str) -> str | None:
    text = text.replace("\r\n", "\n")
    if "\r" in text or "\x00" in text:
        return None
    return text


def _request(rec: dict) -> str | None:
    subject = rec.get("subject", "").strip().rstrip(".")
    words = subject.split()
    if not (3 <= len(words) <= 25) or SKIP_SUBJECT.match(subject):
        return None
    body = rec.get("message", "").strip()
    if body.startswith(subject):
        body = body[len(subject):].strip()
    if body and len(body) < 400 and not body.lower().startswith(("signed-off", "co-authored")):
        return f"{subject}.\n\n{body}"
    return subject


def _lc(s: str) -> str:
    return s[:1].lower() + s[1:] if s[1:2].islower() or len(s) < 2 else s


def usable(rec: dict) -> dict | None:
    """Normalise a record or return None if it can't make a good session."""
    if rec.get("old_file") != rec.get("new_file"):
        return None
    old, new = _clean(rec.get("old_contents", "")), _clean(rec.get("new_contents", ""))
    req = _request(rec)
    if old is None or new is None or req is None or old == new:
        return None
    for t in (old, new):
        if t.count("\n") > MAX_LINES or len(t) > MAX_CHARS:
            return None
    path = rec["new_file"].replace("\\", "/")
    if old:
        edits = compute_edits(old, new)
        if edits is None or len(edits) > MAX_EDITS:
            return None
    else:
        edits = []
    return {"path": path, "old": old, "new": new, "request": req,
            "subject": rec["subject"].strip().rstrip("."), "lang": rec.get("lang", ""), "edits": edits}


def _grep_word(request: str, old: str) -> str | None:
    """
    A word from the request that also appears in the file, to grep for when the
    path isn't given. It must come from the request: the model can't know
    anything else before it has searched.
    """
    words = {w.strip(".") for w in WORD.findall(request)}
    for w in sorted(words, key=lambda w: (-len(w), w)):
        if len(w) >= 4 and w.lower() not in STOPWORDS and w in old:
            return re.escape(w)
    return None


def _call(ws: Workspace, messages: list[dict], name: str, args: dict) -> str:
    messages.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]})
    result = ws.call(name, args)
    messages.append({"role": "tool", "content": result})
    return result


def make_session(rec: dict, rng: random.Random, decoys: list[tuple[str, str]]) -> dict | None:
    """Build a training session from a record returned by usable()."""
    path, req = rec["path"], rec["request"]
    subj = rec["subject"]
    project = path.split("/")[0] if "/" in path else "project"
    files = {p: c for p, c in decoys if p != path}
    files[path] = rec["old"]
    ws = Workspace(files)
    messages: list[dict] = [{"role": "system", "content": random_system_prompt(rng, project)}]
    fmt = {"req": req, "req_lc": _lc(req), "path": path, "subj": subj, "subj_lc": _lc(subj)}

    if not rec["old"]:  # new file
        messages.append({"role": "user", "content": rng.choice(ASK_WITH_PATH).format(**fmt)})
        _call(ws, messages, "write_file", {"path": path, "content": rec["new"]})
        messages.append({"role": "assistant", "content": rng.choice(CREATE_DONE).format(**fmt)})
        return {"source": "commitpack", "messages": messages}

    edits = rec["edits"]
    word = _grep_word(req, rec["old"])
    if word and rng.random() < 0.35:
        messages.append({"role": "user", "content": rng.choice(ASK_NO_PATH).format(**fmt)})
        result = _call(ws, messages, "grep", {"pattern": word})
        if f"{path}:" not in result:
            return None
    else:
        messages.append({"role": "user", "content": rng.choice(ASK_WITH_PATH).format(**fmt)})

    _call(ws, messages, "read_file", {"path": path})
    for e in edits:
        result = _call(ws, messages, "edit_file", {"path": path, "old_string": e.old, "new_string": e.new})
        if not result.startswith("Edited"):
            return None
    if ws.files[path] != rec["new"]:
        return None
    messages.append({"role": "assistant", "content": rng.choice(DONE).format(**fmt)})
    return {"source": "commitpack", "messages": messages}


def make_eval_task(rec: dict, rng: random.Random, decoys: list[tuple[str, str]]) -> dict:
    """A held-out task for evaluate.py: workspace + request + the commit's result."""
    files = {p: c for p, c in decoys if p != rec["path"]}
    files[rec["path"]] = rec["old"]
    project = rec["path"].split("/")[0] if "/" in rec["path"] else "project"
    fmt = {"req": rec["request"], "req_lc": _lc(rec["request"]), "path": rec["path"]}
    with_path = rng.random() < 0.65
    ask = rng.choice(ASK_WITH_PATH if with_path else ASK_NO_PATH).format(**fmt)
    return {"system": random_system_prompt(rng, project), "request": ask, "files": files,
            "target_path": rec["path"], "expected": rec["new"], "path_given": with_path,
            "lang": rec["lang"]}
