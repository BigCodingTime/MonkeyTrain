"""
Run a model as an agent on held-out tasks and score it. Uses Ollama, i.e. the
same quantized model and prompt format MonkeyPaw will use.

  python -m monkeytrain.evaluate --model monkey-1.5b
  python -m monkeytrain.evaluate --model qwen2.5-coder:1.5b      # the base model, to compare
  python -m monkeytrain.evaluate --model monkey-0.5b --n 20 --threads 4

Each task: a small workspace, a request taken from a real commit, and the file
as the commit left it. The model gets the MonkeyPaw tools and up to 8 steps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import ollama
from .render import parse_tool_calls, render
from .sandbox import Workspace
from .tools import TOOL_NAMES, TOOLS

MAX_STEPS = 8


def model_turn(model: str, messages: list[dict], opts: dict, mode: str = "raw",
               ) -> tuple[str, list[dict], list[str], dict]:
    """
    One model reply. "raw" builds the prompt with render() -- byte-identical to
    training -- and parses tool calls itself; Ollama only runs the model. "chat"
    lets Ollama build the prompt from the Modelfile template, which drifts from
    training as Ollama changes (0.35 renders tool JSON differently; Monkey then
    dropped old_string from edit_file calls).
    """
    if mode == "raw":
        prompt, _ = render(messages, TOOLS, add_generation_prompt=True)
        try:
            resp = ollama.generate(model, prompt, opts)
        except ollama.OllamaError as e:  # e.g. Ollama aborted a reply stuck in a loop
            return "", [], [str(e)], {}
        content, calls, errors = parse_tool_calls(resp.get("response", ""))
        return content, calls, errors, resp

    resp = ollama.chat(model, messages, TOOLS, opts)
    msg = resp.get("message", {})
    content = msg.get("content") or ""
    calls = [{"function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments", {})}}
             for c in msg.get("tool_calls") or []]
    errors: list[str] = []
    if not calls and "<tool_call>" in content:  # Ollama couldn't parse it; MonkeyPaw's fallback parser would try
        content, calls, errors = parse_tool_calls(content)
    return content, calls, errors, resp


def run_task(task: dict, model: str, opts: dict, mode: str = "raw") -> dict:
    ws = Workspace(task["files"])
    messages = [{"role": "system", "content": task["system"]}, {"role": "user", "content": task["request"]}]
    r = {"steps": 0, "calls": 0, "format_errors": 0, "unknown_tools": 0, "tool_errors": 0,
         "rewrote_existing_file": 0, "finished": False, "gen_tokens": 0, "gen_seconds": 0.0}
    t0 = time.time()
    for _ in range(MAX_STEPS):
        r["steps"] += 1
        content, calls, errors, resp = model_turn(model, messages, opts, mode)
        r["format_errors"] += len(errors)
        r["gen_tokens"] += resp.get("eval_count", 0)
        r["gen_seconds"] += resp.get("eval_duration", 0) / 1e9
        if not calls:
            messages.append({"role": "assistant", "content": content})
            r["finished"] = not errors
            break
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        for c in calls:
            name, args = c["function"]["name"], c["function"]["arguments"]
            r["calls"] += 1
            if name not in TOOL_NAMES:
                r["unknown_tools"] += 1
            if name == "write_file" and isinstance(args, dict) and str(args.get("path", "")).replace("\\", "/") in task["files"]:
                r["rewrote_existing_file"] += 1
            result = ws.call(name, args)
            if result.startswith("Error"):
                r["tool_errors"] += 1
            messages.append({"role": "tool", "content": result})
    target = task["target_path"]
    final = ws.files.get(target)
    r["changed_target"] = final is not None and final != task["files"][target]
    r["exact_match"] = final == task["expected"]
    r["touched_other_files"] = any(ws.files.get(p) != c for p, c in task["files"].items() if p != target) \
        or len(ws.files) > len(task["files"])
    r["seconds"] = time.time() - t0
    return r


def summarize(results: list[dict]) -> dict:
    n = max(len(results), 1)
    pct = lambda k: 100 * sum(bool(x[k]) for x in results) / n
    avg = lambda k: sum(x[k] for x in results) / n
    tok = sum(x["gen_tokens"] for x in results)
    sec = sum(x["gen_seconds"] for x in results)
    return {
        "tasks": len(results),
        "finished_%": pct("finished"),
        "changed_right_file_%": pct("changed_target"),
        "exact_match_%": pct("exact_match"),
        "touched_other_files_%": pct("touched_other_files"),
        "rewrote_whole_file_%": pct("rewrote_existing_file"),
        "avg_steps": avg("steps"),
        "tool_errors_per_task": avg("tool_errors"),
        "format_errors_per_task": avg("format_errors"),
        "unknown_tools_per_task": avg("unknown_tools"),
        "avg_seconds_per_task": avg("seconds"),
        "writing_tok_per_s": tok / sec if sec else 0.0,
    }


def load_tasks(path: str | None) -> list[dict]:
    if path:
        p = Path(path)
    else:
        from huggingface_hub import hf_hub_download
        from .hub import data_repo, require_token, username
        token = require_token()
        p = Path(hf_hub_download(data_repo(username(token)), "eval_tasks.jsonl",
                                 repo_type="dataset", token=token))
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Ollama model name")
    ap.add_argument("--tasks", default=None, help="eval_tasks.jsonl (default: download from your HF dataset)")
    ap.add_argument("--n", type=int, default=30, help="number of tasks (CPU runs take ~1-2 min each)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--mode", choices=["raw", "chat"], default="raw",
                    help="raw: our prompt format (as trained); chat: Ollama's template")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)[: args.n]
    # num_predict caps a reply, so a model that never stops can't hang the run
    opts = ollama.options(args.threads, args.ctx, temperature=0, seed=1, num_predict=1024)
    results = []
    for i, task in enumerate(tasks, 1):
        r = run_task(task, args.model, opts, args.mode)
        results.append(r)
        mark = "exact" if r["exact_match"] else ("changed" if r["changed_target"] else "no change")
        print(f"[{i}/{len(tasks)}] {task['target_path']}: {mark}, {r['steps']} steps, "
              f"{r['tool_errors']} tool errors, {r['seconds']:.0f}s", flush=True)

    summary = summarize(results)
    print("\n" + "\n".join(f"{k:>24}: {v:.1f}" if isinstance(v, float) else f"{k:>24}: {v}" for k, v in summary.items()))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    fname = out / f"{args.model.replace(':', '_').replace('/', '_')}-{args.mode}.json"
    fname.write_text(json.dumps({"model": args.model, "mode": args.mode, "summary": summary, "tasks": results}, indent=2), encoding="utf-8")
    print(f"\nsaved {fname}")


if __name__ == "__main__":
    main()
