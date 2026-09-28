"""Loss-mask test against the real Qwen tokenizer (downloads ~7 MB once)."""

import pytest

from monkeytrain.encode import IGNORE, encode_session

transformers = pytest.importorskip("transformers")

SESSION = {"messages": [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "find foo"},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "grep", "arguments": {"pattern": "foo"}}}]},
    {"role": "tool", "content": "a.py:1:foo = 1"},
    {"role": "assistant", "content": "Found it in a.py."},
]}


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    except Exception as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")


def test_only_assistant_tokens_are_trained(tok):
    enc = encode_session(SESSION, tok, 8192)
    trained = tok.decode([t for t, l in zip(enc["input_ids"], enc["labels"]) if l != IGNORE])
    assert trained == ('<tool_call>\n{"name": "grep", "arguments": {"pattern": "foo"}}\n</tool_call><|im_end|>'
                       "Found it in a.py.<|im_end|>")


def test_chat_markers_are_single_special_tokens(tok):
    enc = encode_session(SESSION, tok, 8192)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert enc["input_ids"].count(im_end) == 5  # system, user, assistant, tool, assistant


def test_too_long_is_dropped(tok):
    assert encode_session(SESSION, tok, 50) is None
