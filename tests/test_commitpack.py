import random

from monkeytrain.diffing import compute_edits
from monkeytrain.sandbox import Workspace
from monkeytrain.sources import commitpack

OLD = """import os


def load(path):
    with open(path) as f:
        return f.read()


def save(path, data):
    with open(path, "w") as f:
        f.write(data)


def main():
    print(load("a.txt"))
"""

NEW = OLD.replace('with open(path) as f:', 'with open(path, encoding="utf-8") as f:') \
         .replace('    print(load("a.txt"))\n', '    text = load("a.txt")\n    print(text.upper())\n')


def test_edits_apply_cleanly_and_are_unique():
    edits = compute_edits(OLD, NEW)
    assert edits and len(edits) == 2
    text = OLD
    for e in edits:
        assert text.count(e.old) == 1
        text = text.replace(e.old, e.new)
    assert text == NEW


def test_repeated_lines_get_context():
    old = "x = 1\ny = 2\nx = 1\nz = 3\n"
    new = "x = 1\ny = 2\nx = 9\nz = 3\n"
    edits = compute_edits(old, new)
    assert edits and old.count(edits[0].old) == 1


def test_identical_or_unrepresentable_returns_none():
    assert compute_edits("a\n", "a\n") is None


RECORD = {"old_file": "pkg/io.py", "new_file": "pkg/io.py", "old_contents": OLD, "new_contents": NEW,
          "subject": "Read files as UTF-8 and upper-case output", "message": "Read files as UTF-8 and upper-case output\n",
          "lang": "Python"}


def test_session_replays_to_the_commit():
    rec = commitpack.usable(RECORD)
    assert rec is not None
    for seed in range(20):  # covers both the grep and the path-given variants
        s = commitpack.make_session(rec, random.Random(seed), [("other/x.py", "def load_other():\n    pass\n")])
        assert s is not None
        msgs = s["messages"]
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
        assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"]
        # replaying the assistant's calls reproduces the commit
        ws = Workspace({"pkg/io.py": OLD, "other/x.py": "def load_other():\n    pass\n"})
        for m in msgs:
            for c in m.get("tool_calls", []):
                ws.call(c["function"]["name"], c["function"]["arguments"])
        assert ws.files["pkg/io.py"] == NEW


def test_filters():
    assert commitpack.usable({**RECORD, "subject": "Merge branch 'dev'"}) is None
    assert commitpack.usable({**RECORD, "subject": "fix"}) is None
    assert commitpack.usable({**RECORD, "new_file": "pkg/other.py"}) is None
    assert commitpack.usable({**RECORD, "old_contents": "x\n" * 400}) is None


def test_new_file_session_uses_write_file():
    rec = commitpack.usable({**RECORD, "old_contents": ""})
    s = commitpack.make_session(rec, random.Random(0), [])
    call = s["messages"][2]["tool_calls"][0]["function"]
    assert call["name"] == "write_file" and call["arguments"]["content"] == NEW


def test_eval_task_shape():
    rec = commitpack.usable(RECORD)
    t = commitpack.make_eval_task(rec, random.Random(1), [])
    assert t["files"] == {"pkg/io.py": OLD} and t["expected"] == NEW and t["request"]
