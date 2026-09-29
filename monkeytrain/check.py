"""
Sanity check: can the model start a tool call?

Every held-out task needs a tool call first (grep or read_file), so a working
Monkey puts most of its probability on <tool_call> as the first token. The
v1 run scored ~0 here because the base model had never learned to produce
that token and LoRA alone couldn't teach it; export refuses to ship such a model.
"""

from __future__ import annotations

from .render import render
from .tools import TOOLS

MIN_TOOL_CALL_PROB = 0.5


def tool_call_probability(model, tokenizer, tasks: list[dict], n: int = 20) -> float:
    """Mean probability of <tool_call> as the first reply token over n tasks."""
    import torch

    target = tokenizer.convert_tokens_to_ids("<tool_call>")
    device = next(model.parameters()).device
    probs = []
    for t in tasks[:n]:
        prompt, _ = render([{"role": "system", "content": t["system"]}, {"role": "user", "content": t["request"]}],
                           TOOLS, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1].float()
        probs.append(logits.softmax(-1)[target].item())
    return sum(probs) / max(len(probs), 1)
