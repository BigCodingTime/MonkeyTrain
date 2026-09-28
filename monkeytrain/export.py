"""
Merge the finished LoRA adapter into the base model, convert it to a quantized
GGUF file and upload it (with an Ollama Modelfile) to Hugging Face. Kaggle GPU.

  python -m monkeytrain.export --config configs/monkey-1.5b.json
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .hub import load_config, repos, require_token, username
from .ollama import modelfile
from .train import FINAL, read_status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="/kaggle/working/export" if Path("/kaggle").exists() else "export")
    ap.add_argument("--force", action="store_true", help="export again even if the GGUF exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from huggingface_hub import HfApi, snapshot_download
    token = require_token()
    api = HfApi(token=token)
    r = repos(username(token), cfg["name"])

    if read_status(api, r.checkpoints).get("state") != "done":
        raise SystemExit(f"{cfg['name']} hasn't finished training yet; run train first.")
    gguf_name = f"{cfg['name']}-{cfg['quant'].upper()}.gguf"
    if not args.force and api.repo_exists(r.gguf) and api.file_exists(r.gguf, gguf_name):
        print(f"[export] {gguf_name} is already on https://huggingface.co/{r.gguf} (use --force to redo).")
        return

    out = Path(args.output)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    local = snapshot_download(r.checkpoints, allow_patterns=[f"{FINAL}/*"], token=token, local_dir=str(out / "src"))
    adapter = Path(local) / FINAL

    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        str(adapter), max_seq_length=cfg["max_seq_len"], load_in_4bit=True, dtype=None, token=token)

    quant = cfg["quant"]
    gguf_dir = out / "gguf"
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)

    # Unsloth's output file name has changed between versions; find it.
    candidates = [p for p in list(out.rglob("*.gguf")) + list(Path.cwd().glob("*.gguf"))
                  if quant.replace("_", "").lower() in p.name.replace("_", "").lower()]
    if not candidates:
        raise SystemExit(f"No {quant} .gguf produced; files: {[p.name for p in out.rglob('*.gguf')]}")
    name = gguf_name
    final = out / name
    shutil.move(str(candidates[0]), final)
    (out / "Modelfile").write_text(modelfile(f"./{name}"), encoding="utf-8")

    api.create_repo(r.gguf, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(final), path_in_repo=name, repo_id=r.gguf,
                    commit_message=f"{cfg['name']} {quant}")
    api.upload_file(path_or_fileobj=str(out / "Modelfile"), path_in_repo="Modelfile", repo_id=r.gguf)
    size = os.path.getsize(final) / 1e9
    print(f"\n[export] uploaded {name} ({size:.2f} GB) to https://huggingface.co/{r.gguf}")
    print(f"[export] on your laptop: python -m monkeytrain.install --config {args.config}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
