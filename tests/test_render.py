from monkeytrain.render import IM_END, parse_tool_calls, render, render_tool_def
from monkeytrain.tools import TOOLS

TOOL = {"type": "function", "function": {"name": "grep", "description": "Search",
        "parameters": {"type": "object", "required": ["pattern"],
                       "properties": {"pattern": {"type": "string", "description": "Regex"},
                                      "glob": {"type": "string", "description": "Filter"}}}}}

MESSAGES = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "find foo"},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "grep", "arguments": {"pattern": "foo"}}}]},
    {"role": "tool", "content": "a.py:1:foo = 1"},
    {"role": "assistant", "content": "Found it in a.py."},
]

# What Ollama produces from OLLAMA_TEMPLATE for MESSAGES (worked through by hand
# from the Go template). If this test changes, the Modelfile template must too.
EXPECTED = (
    "<|im_start|>system\nSYS\n\n# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
    '{"type": "function", "function": {"name":"grep","description":"Search","parameters":'
    '{"type":"object","required":["pattern"],"properties":{"glob":{"type":"string","description":"Filter"},'
    '"pattern":{"type":"string","description":"Regex"}}}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
    "<|im_start|>user\nfind foo<|im_end|>\n"
    "<|im_start|>assistant\n<tool_call>\n"
    '{"name": "grep", "arguments": {"pattern": "foo"}}\n</tool_call><|im_end|>\n'
    "<|im_start|>user\n<tool_response>\na.py:1:foo = 1\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\nFound it in a.py.<|im_end|>\n"
)


def test_render_matches_ollama_template():
    text, _ = render(MESSAGES, [TOOL])
    assert text == EXPECTED


def test_spans_cover_only_assistant_output():
    text, spans = render(MESSAGES, [TOOL])
    parts = [text[a:b] for a, b in spans]
    assert parts == [
        '<tool_call>\n{"name": "grep", "arguments": {"pattern": "foo"}}\n</tool_call>' + IM_END,
        "Found it in a.py." + IM_END,
    ]


def test_generation_prompt_ends_with_open_assistant_turn():
    text, _ = render(MESSAGES[:2], [TOOL], add_generation_prompt=True)
    assert text.endswith("<|im_start|>user\nfind foo<|im_end|>\n<|im_start|>assistant\n")


def test_properties_sorted_like_go():
    props = render_tool_def(TOOL).split('"properties":')[1]
    assert props.index('"glob"') < props.index('"pattern"')


def test_every_tool_renders():
    text, _ = render([{"role": "user", "content": "hi"}], TOOLS)
    for t in TOOLS:
        assert f'"name":"{t["function"]["name"]}"' in text


def test_parse_tool_calls_roundtrip():
    text, spans = render(MESSAGES, [TOOL])
    a, b = spans[0]
    content, calls, errors = parse_tool_calls(text[a:b])
    assert content == "" and errors == []
    assert calls == [{"function": {"name": "grep", "arguments": {"pattern": "foo"}}}]


def test_parse_multiple_and_broken_calls():
    _, calls, errors = parse_tool_calls('<tool_call>\n{"name": "a", "arguments": {}}\n{"name": "b", "arguments": "{\\"x\\": 1}"}\n</tool_call>')
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert calls[1]["function"]["arguments"] == {"x": 1}
    assert errors == []
    _, calls, errors = parse_tool_calls('<tool_call>\n{"name": "a", "arguments": {\n</tool_call>')
    assert calls == [] and errors
