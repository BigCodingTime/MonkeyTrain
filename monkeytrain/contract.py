"""
Export the Monkey contract: everything MonkeyPaw has to reproduce exactly so
Monkey sees what it was trained on (MonkeePawLLM spec 2026-10-01, section 2).

  python -m monkeytrain.contract --out ../MonkeePawLLM/packages/core/src/monkey/contract.json

The fixtures come from running render(), parse_tool_calls() and the sandbox,
so they are correct by construction. MonkeyPaw replays them in its tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render import parse_tool_calls, render
from .sandbox import Workspace
from .tools import SYSTEM_TEMPLATE, TOOLS, system_prompt

CONTRACT_VERSION = 1
STOP = ["<|im_end|>", "<|endoftext|>"]
# Ollama sampling for Monkey. repeat_penalty must stay 1.0: old_string copies file text exactly.
SAMPLING = {"temperature": 0.2, "repeat_penalty": 1.0, "top_p": 0.95, "top_k": 40}

_ALL = [t["function"]["name"] for t in TOOLS]
_SYS = system_prompt("Windows", "Git Bash", "C:\\Gits\\demo")


def _call(name: str, **arguments) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


RENDER_CASES = [
    ("tool_call_then_answer", [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": "find foo"},
        {"role": "assistant", "content": "", "tool_calls": [_call("grep", pattern="foo")]},
        {"role": "tool", "content": "a.py:1:foo = 1"},
        {"role": "assistant", "content": "Found it in a.py."},
    ], ["grep"], False),
    ("generation_prompt", [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": "In src/config.py, change the timeout to 60"},
    ], ["read_file", "edit_file"], True),
    ("tricky_arguments", [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": "Use the café emoji 🐒 in greet()"},
        {"role": "assistant", "content": "", "tool_calls": [_call(
            "edit_file", path="src\\greet.py",
            old_string='def greet():\n\treturn "hi\\n"',
            new_string='def greet():\n\treturn "café 🐒 \u00fc\\t"\n# done\x01')]},
        {"role": "tool", "content": "Edited src/greet.py: replaced 1 occurrence."},
    ], ["edit_file"], True),
    ("no_system_no_tools", [{"role": "user", "content": "hello"}], [], True),
    ("all_tools", [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": "What does zip() do?"},
        {"role": "assistant", "content": "It pairs items from several iterables."},
    ], _ALL, False),
    ("several_calls_one_turn", [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": "read both"},
        {"role": "assistant", "content": "", "tool_calls": [
            _call("read_file", path="a.py"), _call("read_file", path="b.py", offset=3, limit=10)]},
    ], ["read_file"], False),
]

PARSE_CASES = [
    ("single_call", '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'),
    ("text_then_call", 'Let me look.\n<tool_call>\n{"name": "grep", "arguments": {"pattern": "load_user"}}\n</tool_call>'),
    ("two_calls_string_args", '<tool_call>\n{"name": "a", "arguments": {}}\n{"name": "b", "arguments": "{\\"x\\": 1}"}\n</tool_call>'),
    ("braces_inside_strings", '<tool_call>\n{"name": "edit_file", "arguments": {"path": "a.py", "old_string": "x = {\\"k\\": [1, 2]}", "new_string": "}{"}}\n</tool_call>'),
    ("broken_json", '<tool_call>\n{"name": "edit_file", "arguments": {"path": "a.py", "old_string": "x = 1"\n</tool_call>'),
    ("unterminated_block", '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}'),
    ("plain_text", "Just an answer, no tools.<|im_end|>"),
    ("missing_name", '<tool_call>\n{"arguments": {}}\n</tool_call>'),
    ("bad_string_arguments", '<tool_call>\n{"name": "x", "arguments": "not json"}\n</tool_call>'),
]

TOOL_FILES = {
    "src/app.py": "a = 1\nb = 2\na = 1\n",
    "src/util/x.py": "def foo():\n    pass\n",
    "README.md": "# hi\n",
    "empty.txt": "",
}

TOOL_CASES = [
    ("read_full", "read_file", {"path": "src/app.py"}),
    ("read_partial", "read_file", {"path": "src/app.py", "offset": 2, "limit": 1}),
    ("read_dot_slash", "read_file", {"path": "./README.md"}),
    ("read_empty", "read_file", {"path": "empty.txt"}),
    ("read_missing", "read_file", {"path": "nope.py"}),
    ("read_outside", "read_file", {"path": "../secret.txt"}),
    ("edit_ok", "edit_file", {"path": "src/app.py", "old_string": "b = 2", "new_string": "b = 5"}),
    ("edit_not_found", "edit_file", {"path": "src/app.py", "old_string": "zzz", "new_string": "y"}),
    ("edit_ambiguous", "edit_file", {"path": "src/app.py", "old_string": "a = 1", "new_string": "a = 3"}),
    ("edit_replace_all", "edit_file", {"path": "src/app.py", "old_string": "a = 1", "new_string": "a = 0", "replace_all": True}),
    ("edit_identical", "edit_file", {"path": "src/app.py", "old_string": "b = 2", "new_string": "b = 2"}),
    ("edit_empty_old", "edit_file", {"path": "src/app.py", "old_string": "", "new_string": "x"}),
    ("edit_missing_file", "edit_file", {"path": "nope.py", "old_string": "a", "new_string": "b"}),
    ("write_create", "write_file", {"path": "docs/new.txt", "content": "x\ny\n"}),
    ("write_overwrite", "write_file", {"path": "README.md", "content": "# new"}),
    ("glob_recursive", "glob", {"pattern": "**/*.py"}),
    ("glob_basename", "glob", {"pattern": "*.md"}),
    ("glob_none", "glob", {"pattern": "*.rs"}),
    ("grep_regex", "grep", {"pattern": "def \\w+"}),
    ("grep_in_path", "grep", {"pattern": "a = 1", "path": "src"}),
    ("grep_with_glob", "grep", {"pattern": "hi", "glob": "*.md"}),
    ("grep_invalid_regex", "grep", {"pattern": "("}),
    ("grep_none", "grep", {"pattern": "zzz"}),
]


def build_contract() -> dict:
    by_name = {t["function"]["name"]: t for t in TOOLS}
    render_fixtures = []
    for name, messages, tool_names, gen in RENDER_CASES:
        tools = [by_name[n] for n in _ALL if n in tool_names]
        text, _ = render(messages, tools, add_generation_prompt=gen)
        render_fixtures.append({"name": name, "messages": messages, "tools": tool_names,
                                "generationPrompt": gen, "expected": text})
    parse_fixtures = []
    for name, text in PARSE_CASES:
        content, calls, errors = parse_tool_calls(text)
        parse_fixtures.append({"name": name, "text": text, "expected": {
            "content": content,
            "calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in calls],
            "errorCount": len(errors)}})
    tool_fixtures = []
    for name, tool, args in TOOL_CASES:
        ws = Workspace(dict(TOOL_FILES))
        out = ws.call(tool, args)
        tool_fixtures.append({"name": name, "files": TOOL_FILES, "tool": tool, "args": args,
                              "expected": out, "filesAfter": ws.files})
    env = {"os": "Windows", "shell": "Git Bash", "workspace": "C:\\Gits\\demo"}
    return {
        "version": CONTRACT_VERSION,
        "tools": TOOLS,
        "systemTemplate": SYSTEM_TEMPLATE,
        "stop": STOP,
        "sampling": SAMPLING,
        "fixtures": {
            "systemPrompt": {**env, "expected": system_prompt(env["os"], env["shell"], env["workspace"])},
            "render": render_fixtures,
            "parse": parse_fixtures,
            "tools": tool_fixtures,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Write MonkeyPaw's Monkey contract.")
    ap.add_argument("--out", required=True, help="path of contract.json to write")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_contract(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
