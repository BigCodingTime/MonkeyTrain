import random

from monkeytrain.evaluate import score_create, summarize
from monkeytrain.render import render
from monkeytrain.sandbox import Workspace
from monkeytrain.sources import commitpack, create
from monkeytrain.tools import TOOLS

CODE = '''class Calculator:
    """Adds, subtracts, multiplies and divides two numbers."""

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def divide(self, a, b):
        if b == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return a / b
'''
ROW = {"instruction": "Write a Python class `Calculator` that can add, subtract and divide numbers safely.",
       "response": f"Here is the class:\n\n```python\n{CODE}```\n\nThe divide method refuses to divide by zero.\n\n"
                   "```python\nassert Calculator().add(1, 2) == 3\n```"}
DECOYS = [("weather.py", "def forecast(city):\n    return city\n")]


def sessions(n=40):
    return [create.make_session(ROW, random.Random(seed), DECOYS) for seed in range(n)]


def test_takes_the_program_not_the_usage_snippet():
    assert create.extract_code(ROW["response"]) == CODE


def test_rejects_answers_without_a_usable_program():
    assert create.make_session({**ROW, "response": "Just use `+`."}, random.Random(0)) is None
    assert create.extract_code("```python\ndef broken(:\n    pass\n" + "x = 1\n" * 60 + "```") is None


def test_writes_the_whole_program_to_a_new_file_without_being_told_a_path():
    for s in sessions():
        msgs = s["messages"]
        assert s["source"] == "create" and msgs[1]["role"] == "user"
        assert ".py" not in msgs[1]["content"]
        write = [c["function"] for m in msgs for c in m.get("tool_calls", []) if c["function"]["name"] == "write_file"]
        assert len(write) == 1 and write[0]["arguments"]["content"] == CODE
        assert write[0]["arguments"]["path"].endswith(".py")
        assert msgs[-1]["role"] == "assistant" and write[0]["arguments"]["path"] in msgs[-1]["content"]


def test_some_sessions_search_first_and_build_anyway_when_nothing_is_found():
    firsts = [s["messages"][2]["tool_calls"][0]["function"]["name"] for s in sessions(80)]
    assert {"grep", "glob", "write_file"} <= set(firsts)
    for s in sessions(80):
        results = [m["content"] for m in s["messages"] if m["role"] == "tool"]
        assert all(r in ("No matches.", "No files matched.") for r in results[:-1])
        assert results[-1].startswith("Created")


def test_names_the_file_after_the_main_class_most_of_the_time():
    paths = [m["tool_calls"][0]["function"]["arguments"]["path"]
             for s in sessions() for m in s["messages"] if m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "write_file"]
    assert paths.count("calculator.py") > len(paths) / 2


def test_sessions_render_within_the_training_length():
    for s in sessions(10):
        prompt, _ = render(s["messages"], TOOLS)
        assert "<tool_call>" in prompt and len(prompt) < 4096 * 3


def test_create_eval_task_scoring():
    task = create.make_eval_task(ROW, random.Random(0), DECOYS)
    assert task["kind"] == "create" and task["expected_name"] == "Calculator"
    built = score_create(task, {**task["files"], "calc.py": CODE})
    assert built["built_from_scratch"] and not built["touched_other_files"]
    broken = score_create(task, {**task["files"], "calc.py": "def x(:\n"})
    assert broken["created_file"] and not broken["built_from_scratch"]
    assert not score_create(task, dict(task["files"]))["created_file"]


def test_summary_keeps_create_tasks_out_of_the_edit_scores():
    edit = {"finished": True, "changed_target": True, "exact_match": True, "touched_other_files": False,
            "rewrote_existing_file": 0, "steps": 2, "tool_errors": 0, "format_errors": 0, "unknown_tools": 0,
            "seconds": 1.0, "gen_tokens": 10, "gen_seconds": 1.0}
    made = {**edit, "kind": "create", "built_from_scratch": True, "created_file": True}
    out = summarize([edit, made, {**made, "built_from_scratch": False}])
    assert out["tasks"] == 1 and out["exact_match_%"] == 100
    assert out["create_tasks"] == 2 and out["built_from_scratch_%"] == 50


OLD = "def load_config(path):\n    with open(path) as f:\n        return f.read()\n"
NEW = OLD.replace("open(path)", 'open(path, encoding="utf-8")')
RECORD = {"old_file": "pkg/io.py", "new_file": "pkg/io.py", "old_contents": OLD, "new_contents": NEW,
          "subject": "Read settings in load_config with utf-8 encoding", "message": "", "lang": "Python"}


def test_commitpack_grep_sometimes_misses_then_retries_with_another_word():
    rec = commitpack.usable(RECORD)
    retried = 0
    for seed in range(200):
        s = commitpack.make_session(rec, random.Random(seed), [])
        greps = [m for m in s["messages"] if m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "grep"]
        if len(greps) == 2:
            retried += 1
            i = s["messages"].index(greps[0])
            assert s["messages"][i + 1]["content"] == "No matches."
            assert "pkg/io.py:" in s["messages"][i + 3]["content"]
        ws = Workspace({"pkg/io.py": OLD})
        for m in s["messages"]:
            for c in m.get("tool_calls", []):
                ws.call(c["function"]["name"], c["function"]["arguments"])
        assert ws.files["pkg/io.py"] == NEW
    assert retried > 0


def test_commitpack_new_files_are_sometimes_asked_for_without_a_path():
    rec = commitpack.usable({**RECORD, "old_contents": ""})
    asks = [commitpack.make_session(rec, random.Random(seed), [])["messages"][1]["content"] for seed in range(40)]
    assert any("pkg/io.py" in a for a in asks) and any("pkg/io.py" not in a for a in asks)


def test_searches_for_the_name_the_user_asked_for():
    assert create.search_word("Write a function `get_closest_points(points, k)` that sorts points") == "get_closest_points"
    assert create.search_word("Make a simple calculator with a window") == "calculator"
