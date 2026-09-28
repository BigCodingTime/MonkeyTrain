"""
MonkeyPaw tool definitions and system prompt.

This is the contract between training and MonkeyPaw: the names, arguments and
wording here must match what MonkeyPaw's core engine sends to the model
(MonkeePawLLM spec §6). If a tool changes there, change it here and retrain.
"""

from __future__ import annotations

import random


def _fn(name: str, description: str, required: list[str], properties: dict) -> dict:
    # Key order mirrors Ollama's Go structs so the rendered system prompt matches.
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "required": required, "properties": properties},
        },
    }


def _p(type_: str, description: str) -> dict:
    return {"type": type_, "description": description}


TOOLS: list[dict] = [
    _fn("read_file", "Read a file from the workspace. Returns numbered lines.",
        ["path"],
        {"path": _p("string", "File path relative to the workspace"),
         "offset": _p("integer", "Line number to start from (1-based)"),
         "limit": _p("integer", "Max lines to return (default 2000)")}),
    _fn("edit_file",
        "Replace old_string with new_string in a file. old_string must match exactly and be "
        "unique unless replace_all is true. Prefer this over write_file for existing files.",
        ["path", "old_string", "new_string"],
        {"path": _p("string", "File path relative to the workspace"),
         "old_string": _p("string", "Exact text to replace"),
         "new_string": _p("string", "Replacement text"),
         "replace_all": _p("boolean", "Replace every occurrence")}),
    _fn("write_file", "Create a new file or overwrite a file completely.",
        ["path", "content"],
        {"path": _p("string", "File path relative to the workspace"),
         "content": _p("string", "Full file content")}),
    _fn("glob", "Find files by glob pattern, e.g. **/*.py",
        ["pattern"],
        {"pattern": _p("string", "Glob pattern")}),
    _fn("grep", "Search file contents with a regular expression. Returns path:line:text.",
        ["pattern"],
        {"pattern": _p("string", "Regular expression"),
         "path": _p("string", "Directory or file to search (default: workspace)"),
         "glob": _p("string", "Only search files matching this glob")}),
    _fn("bash", "Run a shell command in the workspace. Returns output and exit code.",
        ["command"],
        {"command": _p("string", "Command to run"),
         "timeout": _p("integer", "Timeout in seconds")}),
    _fn("web_search", "Search the web. Returns the top results.",
        ["query"],
        {"query": _p("string", "Search query")}),
    _fn("web_fetch", "Fetch a web page as text.",
        ["url"],
        {"url": _p("string", "http(s) URL")}),
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

SYSTEM_TEMPLATE = """You are Monkey, the coding agent inside MonkeyPaw. You help with software tasks in the user's workspace by calling tools.

Rules:
- Read a file before editing it.
- For existing files, make small targeted changes with edit_file. Copy old_string exactly from the file, with enough surrounding lines to be unique. Only use write_file for new files.
- Search with grep or glob when you don't know which file to change.
- Call one tool at a time and wait for its result.
- When the task is done, reply with a short summary of what you changed. Don't call more tools.
- If no tools are needed, just answer.

Environment:
- OS: {os_name}
- Shell: {shell}
- Workspace: {workspace}"""

ENVIRONMENTS = [
    ("Windows", "PowerShell", "C:\\Users\\dev\\{proj}"),
    ("Windows", "Git Bash", "C:\\Gits\\{proj}"),
    ("Linux", "bash", "/home/dev/{proj}"),
    ("macOS", "zsh", "/Users/dev/{proj}"),
]


def system_prompt(os_name: str, shell: str, workspace: str) -> str:
    return SYSTEM_TEMPLATE.format(os_name=os_name, shell=shell, workspace=workspace)


def random_system_prompt(rng: random.Random, project: str = "project") -> str:
    os_name, shell, ws = rng.choice(ENVIRONMENTS)
    return system_prompt(os_name, shell, ws.format(proj=project))
