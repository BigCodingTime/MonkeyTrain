"""
Download a finished Monkey from Hugging Face and register it with Ollama.

  python -m monkeytrain.install --config configs/monkey-1.5b.json
  python -m monkeytrain.install --config configs/monkey-0.5b.json --threads 2 --ctx 8192

--threads / --ctx set the model's defaults. MonkeyPaw's resource profiles
override them per request, so you normally leave --threads unset.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .hub import load_config, repos, require_token, username
from .ollama import modelfile

MODELS_DIR = Path.home() / ".monkeypaw" / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--threads", type=int, default=None, help="default CPU threads (omit = Ollama decides)")
    ap.add_argument("--ctx", type=int, default=8192, help="default context window in tokens")
    args = ap.parse_args()

    if shutil.which("ollama") is None:
        raise SystemExit("Ollama isn't installed. Get it from https://ollama.com/download and run this again.")

    cfg = load_config(args.config)
    from huggingface_hub import hf_hub_download
    token = require_token()
    r = repos(username(token), cfg["name"])
    gguf_name = f"{cfg['name']}-{cfg['quant'].upper()}.gguf"
    dest = MODELS_DIR / cfg["name"]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[install] downloading {gguf_name} from {r.gguf} ...")
    gguf = Path(hf_hub_download(r.gguf, gguf_name, token=token, local_dir=str(dest)))

    mf = dest / "Modelfile"
    mf.write_text(modelfile(str(gguf.resolve()), ctx=args.ctx, threads=args.threads), encoding="utf-8")
    subprocess.run(["ollama", "create", cfg["name"], "-f", str(mf)], check=True)
    print(f"\n[install] done. Try it:  ollama run {cfg['name']}")
    print(f"[install] measure speed: python -m monkeytrain.bench --model {cfg['name']}")


if __name__ == "__main__":
    main()
