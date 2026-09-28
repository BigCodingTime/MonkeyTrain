"""
Measure Monkey's speed on this machine at different CPU thread counts, so you
can pick the MonkeyPaw resource profile that balances speed against battery
and keeping the rest of the computer responsive.

  python -m monkeytrain.bench --model monkey-1.5b
  python -m monkeytrain.bench --model monkey-0.5b --threads 1,2,4 --ctx 4096
"""

from __future__ import annotations

import argparse
import os

from . import ollama

PROMPT = (
    "<|im_start|>system\nYou are Monkey, a coding assistant.<|im_end|>\n"
    "<|im_start|>user\nExplain what this function does and suggest one improvement:\n\n"
    "def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
    "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main() -> None:
    cores = os.cpu_count() or 4
    default = sorted({1, 2, max(2, cores // 4), max(2, cores // 2), cores})
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", default=",".join(map(str, default)), help="comma-separated thread counts")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=128, help="tokens to generate per run")
    args = ap.parse_args()

    print(f"{'threads':>7}  {'reading (tok/s)':>15}  {'writing (tok/s)':>15}  {'load (s)':>8}")
    for t in [int(x) for x in args.threads.split(",")]:
        opts = ollama.options(t, args.ctx, num_predict=args.tokens, temperature=0, seed=1)
        # The first call loads the model with these settings. The second is measured; its
        # different first line stops Ollama from reusing the cached prompt.
        first = ollama.generate(args.model, f"<!-- load {t} -->\n" + PROMPT, opts)
        r = ollama.generate(args.model, f"<!-- run {t} -->\n" + PROMPT, opts)
        read = r["prompt_eval_count"] / max(r["prompt_eval_duration"], 1) * 1e9
        write = r["eval_count"] / max(r["eval_duration"], 1) * 1e9
        print(f"{t:>7}  {read:>15.1f}  {write:>15.1f}  {first.get('load_duration', 0) / 1e9:>8.1f}")
    print("\nReading = processing the prompt/files, writing = generating the answer.")
    print("Pick the smallest thread count whose writing speed still feels fast enough; "
          "fewer threads = less battery and a more responsive computer.")


if __name__ == "__main__":
    main()
