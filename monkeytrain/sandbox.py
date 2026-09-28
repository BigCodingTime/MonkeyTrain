"""
In-memory workspace that executes MonkeyPaw tools.

Used twice: to produce the exact tool results written into training data, and
to run the model in evaluation. Output formats follow the MonkeyPaw spec §6 --
if MonkeyPaw formats a result differently, change it here too.
"""

from __future__ import annotations

import posixpath
import re

READ_LIMIT = 2000
GLOB_MAX = 500
GREP_MAX = 200


class ToolError(Exception):
    pass


def glob_to_regex(pattern: str) -> re.Pattern:
    i, out = 0, []
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
            continue
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        out.append({"*": "[^/]*", "?": "[^/]"}.get(c, re.escape(c)))
        i += 1
    return re.compile("".join(out) + r"\Z")


def _matches_glob(path: str, pattern: str) -> bool:
    rx = glob_to_regex(pattern)
    # A pattern without a slash matches the file name anywhere, like ripgrep's --glob.
    return bool(rx.match(path) or ("/" not in pattern and rx.match(posixpath.basename(path))))


class Workspace:
    def __init__(self, files: dict[str, str]):
        self.files = dict(files)

    def _path(self, path: str) -> str:
        p = str(path).replace("\\", "/").strip()
        if p.startswith("./"):
            p = p[2:]
        norm = posixpath.normpath(p)
        if norm.startswith("/") or norm == ".." or norm.startswith("../") or re.match(r"^[A-Za-z]:", norm):
            raise ToolError(f"Error: path '{path}' is outside the workspace. Use a path relative to the workspace.")
        return norm

    def call(self, name: str, args: dict) -> str:
        """Run a tool. Returns the result text; errors are returned as text too."""
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return f"Error: unknown tool '{name}'."
        if not isinstance(args, dict):
            return f"Error: arguments for {name} must be a JSON object."
        try:
            return fn(**args)
        except ToolError as e:
            return str(e)
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"

    # ---- tools ----

    def tool_read_file(self, path: str, offset: int = 1, limit: int = READ_LIMIT) -> str:
        p = self._path(path)
        if p not in self.files:
            raise ToolError(f"Error: file not found: {p}")
        lines = self.files[p].split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        total = len(lines)
        start = max(1, int(offset))
        end = min(total, start + max(1, int(limit)) - 1)
        body = "\n".join(f"{n:>6}\t{lines[n - 1]}" for n in range(start, end + 1))
        if total == 0:
            return "(empty file)"
        if start > 1 or end < total:
            body += f"\n... (showing lines {start}-{end} of {total}; use offset to read more)"
        return body

    def tool_edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        p = self._path(path)
        if p not in self.files:
            raise ToolError(f"Error: file not found: {p}. Use write_file to create a new file.")
        if old_string == new_string:
            raise ToolError("Error: old_string and new_string are identical; nothing to change.")
        if not old_string:
            raise ToolError("Error: old_string is empty. Copy the exact text to replace from the file.")
        text = self.files[p]
        count = text.count(old_string)
        if count == 0:
            raise ToolError(f"Error: old_string not found in {p}. Read the file again and copy the text exactly, including indentation.")
        if count > 1 and not replace_all:
            raise ToolError(f"Error: old_string matches {count} places in {p}. Include more surrounding lines to make it unique, or set replace_all.")
        self.files[p] = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        n = count if replace_all else 1
        return f"Edited {p}: replaced {n} occurrence{'s' if n != 1 else ''}."

    def tool_write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        existed = p in self.files
        self.files[p] = content
        n = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return f"{'Overwrote' if existed else 'Created'} {p} ({n} lines)."

    def tool_glob(self, pattern: str) -> str:
        hits = sorted(p for p in self.files if _matches_glob(p, pattern))
        if not hits:
            return "No files matched."
        more = f"\n... ({len(hits) - GLOB_MAX} more)" if len(hits) > GLOB_MAX else ""
        return "\n".join(hits[:GLOB_MAX]) + more

    def tool_grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> str:
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        base = self._path(path) if path else "."
        out: list[str] = []
        for p in sorted(self.files):
            if base != "." and p != base and not p.startswith(base.rstrip("/") + "/"):
                continue
            if glob and not _matches_glob(p, glob):
                continue
            for n, line in enumerate(self.files[p].split("\n"), 1):
                if rx.search(line):
                    out.append(f"{p}:{n}:{line.strip()[:200]}")
                    if len(out) >= GREP_MAX:
                        return "\n".join(out) + f"\n... (stopped at {GREP_MAX} matches)"
        return "\n".join(out) if out else "No matches."

    def tool_bash(self, command: str, timeout: int | None = None) -> str:
        raise ToolError("Error: bash is not available in this sandbox.")

    def tool_web_search(self, query: str) -> str:
        raise ToolError("Error: web_search is not available in this sandbox.")

    def tool_web_fetch(self, url: str) -> str:
        raise ToolError("Error: web_fetch is not available in this sandbox.")
