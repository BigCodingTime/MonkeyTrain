"""
Prompt rendering that matches the Ollama template Monkey ships with.

Training text is produced by `render()` and inference text by Ollama using
OLLAMA_TEMPLATE. Both must produce the same bytes, or the model is trained on a
different format from the one it sees at runtime. tests/test_render.py pins the
format; change the two together.

Messages use the Ollama / OpenAI chat shape:
  {"role": "system" | "user" | "assistant" | "tool", "content": str,
   "tool_calls": [{"function": {"name": str, "arguments": dict}}]}   # assistant only
"""

from __future__ import annotations

import json
import re

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Qwen2.5-Coder has these as single tokens but was never trained to produce
# them, and LoRA can't change a token's embedding. Training updates their
# embedding rows directly (see train.py).
TRAINABLE_TOKENS = ["<tool_call>", "</tool_call>"]

TOOLS_PREAMBLE = (
    "\n\n# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_POSTAMBLE = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)

# Go template for the Ollama Modelfile. Same structure as Ollama's official
# qwen2.5-coder template, so Ollama's tool-call parser recognises <tool_call>.
OLLAMA_TEMPLATE = """{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}"""


def _sorted_properties(fn: dict) -> dict:
    # Go marshals map keys in sorted order.
    params = dict(fn.get("parameters", {}))
    if "properties" in params:
        params["properties"] = {k: params["properties"][k] for k in sorted(params["properties"])}
    return {**fn, "parameters": params}


def render_tool_def(tool: dict) -> str:
    fn = json.dumps(_sorted_properties(tool["function"]), ensure_ascii=False, separators=(",", ":"))
    return '{"type": "function", "function": ' + fn + "}"


def format_tool_call(call: dict) -> str:
    fn = call["function"]
    args = json.dumps(fn.get("arguments", {}), ensure_ascii=False)
    return '{"name": "' + fn["name"] + '", "arguments": ' + args + "}"


def _assistant_body(msg: dict) -> str:
    content = msg.get("content") or ""
    calls = msg.get("tool_calls") or []
    if content:
        # The template drops tool calls when content is present; keep data honest.
        if calls:
            raise ValueError("assistant message has both content and tool_calls")
        return content
    if calls:
        return "<tool_call>\n" + "".join(format_tool_call(c) + "\n" for c in calls) + "</tool_call>"
    return ""


def render(messages: list[dict], tools: list[dict] | None, add_generation_prompt: bool = False,
           ) -> tuple[str, list[tuple[int, int]]]:
    """
    Render a conversation. Returns (text, spans) where spans are the character
    ranges of assistant output (including <|im_end|>) -- the tokens to train on.

    Every assistant turn ends with <|im_end|>; with add_generation_prompt the
    text ends with an open assistant turn, exactly like Ollama at inference.
    """
    out: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0

    def emit(s: str) -> None:
        nonlocal pos
        out.append(s)
        pos += len(s)

    msgs = list(messages)
    system = ""
    if msgs and msgs[0]["role"] == "system":
        system = msgs.pop(0)["content"]

    if system or tools:
        head = IM_START + "system"
        if system:
            head += "\n" + system
        if tools:
            head += TOOLS_PREAMBLE + "".join("\n" + render_tool_def(t) for t in tools) + TOOLS_POSTAMBLE
        emit(head + IM_END + "\n")

    for msg in msgs:
        role = msg["role"]
        if role == "user":
            emit(f"{IM_START}user\n{msg['content']}{IM_END}\n")
        elif role == "tool":
            emit(f"{IM_START}user\n<tool_response>\n{msg['content']}\n</tool_response>{IM_END}\n")
        elif role == "assistant":
            emit(f"{IM_START}assistant\n")
            start = pos
            emit(_assistant_body(msg) + IM_END)
            if msg.get("train", True):
                spans.append((start, pos))
            emit("\n")
        else:
            raise ValueError(f"unknown role: {role}")

    if add_generation_prompt:
        emit(f"{IM_START}assistant\n")
    return "".join(out), spans


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)


def parse_tool_calls(text: str) -> tuple[str, list[dict], list[str]]:
    """
    Parse model output. Returns (content, calls, errors). Content is the text
    outside <tool_call> blocks. Each JSON object inside a block is one call.
    """
    text = text.replace(IM_END, "")
    calls: list[dict] = []
    errors: list[str] = []
    decoder = json.JSONDecoder()
    for block in _TOOL_CALL_RE.findall(text):
        i = 0
        block = block.strip()
        while i < len(block):
            while i < len(block) and block[i].isspace():
                i += 1
            if i >= len(block):
                break
            try:
                obj, end = decoder.raw_decode(block, i)
            except json.JSONDecodeError as e:
                errors.append(f"invalid JSON in tool call: {e.msg}")
                break
            i = end
            if not isinstance(obj, dict) or "name" not in obj:
                errors.append("tool call missing 'name'")
                continue
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    errors.append("tool call arguments are not a JSON object")
                    continue
            calls.append({"function": {"name": obj["name"], "arguments": args}})
    content = _TOOL_CALL_RE.sub("", text).strip()
    return content, calls, errors
