from monkeytrain.sandbox import Workspace


def ws():
    return Workspace({"src/app.py": "a = 1\nb = 2\na = 1\n", "README.md": "# hi\n", "src/util/x.py": "def foo():\n    pass\n"})


def test_read_file_numbers_lines_and_pages():
    w = ws()
    assert w.call("read_file", {"path": "src/app.py"}) == "     1\ta = 1\n     2\tb = 2\n     3\ta = 1"
    out = w.call("read_file", {"path": "src/app.py", "offset": 2, "limit": 1})
    assert out.startswith("     2\tb = 2\n... (showing lines 2-2 of 3")


def test_edit_requires_unique_match():
    w = ws()
    assert "matches 2 places" in w.call("edit_file", {"path": "src/app.py", "old_string": "a = 1", "new_string": "a = 3"})
    assert w.call("edit_file", {"path": "src/app.py", "old_string": "b = 2", "new_string": "b = 5"}) == \
        "Edited src/app.py: replaced 1 occurrence."
    assert w.files["src/app.py"] == "a = 1\nb = 5\na = 1\n"
    assert "replaced 2 occurrences" in w.call("edit_file", {"path": "src/app.py", "old_string": "a = 1", "new_string": "a = 0", "replace_all": True})
    assert "not found" in w.call("edit_file", {"path": "src/app.py", "old_string": "zzz", "new_string": "y"})


def test_paths_cannot_escape_workspace():
    w = ws()
    for bad in ("../etc/passwd", "/etc/passwd", "C:\\Windows\\x", "src/../../x"):
        assert "outside the workspace" in w.call("read_file", {"path": bad})


def test_glob_and_grep():
    w = ws()
    assert w.call("glob", {"pattern": "**/*.py"}) == "src/app.py\nsrc/util/x.py"
    assert w.call("glob", {"pattern": "*.md"}) == "README.md"
    assert w.call("grep", {"pattern": r"def \w+"}) == "src/util/x.py:1:def foo():"
    assert w.call("grep", {"pattern": "a = ", "glob": "*.md"}) == "No matches."
    assert w.call("grep", {"pattern": "(", "path": "src"}) == "src/util/x.py:1:def foo():"  # bad regex -> literal
    assert w.call("grep", {"pattern": "a = 1", "path": "src/util"}) == "No matches."


def test_bad_calls_return_errors_not_exceptions():
    w = ws()
    assert w.call("nope", {}).startswith("Error: unknown tool")
    assert w.call("read_file", {"file": "x"}).startswith("Error: bad arguments")
    assert w.call("read_file", "src/app.py").startswith("Error")
    assert w.call("write_file", {"path": "new.txt", "content": "x\ny\n"}) == "Created new.txt (2 lines)."
