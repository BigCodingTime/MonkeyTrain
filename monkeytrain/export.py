"""
Merge the finished adapter into the base model, check that the result can call
tools, convert it to a quantized GGUF file and upload it (with an Ollama
Modelfile) to Hugging Face. Kaggle GPU.

  python -m monkeytrain.export --config configs/monkey-1.5b.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .check import MIN_TOOL_CALL_PROB, tool_call_probability
from .hub import load_config, repos, require_token, username
from .ollama import modelfile
from .train import FINAL, read_status


def merge_adapter(base_model: str, adapter_dir: Path, token: str | None = None, device: str = "cpu"):
    """
    Base model (16-bit) + adapter -> plain merged model, with PEFT doing the
    merge so the trained token rows (render.TRAINABLE_TOKENS) are included.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, token=token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    return model.eval(), tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="/kaggle/working/export" if Path("/kaggle").exists() else "export")
    ap.add_argument("--force", action="store_true", help="export again even if the GGUF exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    token = require_token()
    api = HfApi(token=token)
    r = repos(username(token), cfg)

    if read_status(api, r.checkpoints).get("state") != "done":
        raise SystemExit(f"{r.checkpoints} hasn't finished training yet; run train first.")
    if not args.force and api.repo_exists(r.gguf) and api.file_exists(r.gguf, r.gguf_file):
        print(f"[export] {r.gguf_file} is already on https://huggingface.co/{r.gguf} (use --force to redo).")
        return

    out = Path(args.output)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    local = snapshot_download(r.checkpoints, allow_patterns=[f"{FINAL}/*"], token=token, local_dir=str(out / "src"))
    adapter = Path(local) / FINAL

    # ---- merge + check ----
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = merge_adapter(cfg["base_model"], adapter, token, device)
    tasks_path = hf_hub_download(r.data, "eval_tasks.jsonl", repo_type="dataset", token=token)
    tasks = [json.loads(line) for line in open(tasks_path, encoding="utf-8")]
    p = tool_call_probability(model, tokenizer, tasks)
    print(f"[check] P(<tool_call>) as first reply token on held-out tasks: {p:.2f} (want >= {MIN_TOOL_CALL_PROB})")
    if p < MIN_TOOL_CALL_PROB:
        raise SystemExit("[export] The merged model can't start tool calls, so it isn't exported. "
                         "Check the training log.")
    merged = out / "merged"
    model.save_pretrained(str(merged))
    tokenizer.save_pretrained(str(merged))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- GGUF (Unsloth drives llama.cpp's converter and quantizer) ----
    from unsloth import FastLanguageModel
    quant = cfg["quant"]
    gmodel, gtok = FastLanguageModel.from_pretrained(str(merged), max_seq_length=cfg["max_seq_len"],
                                                     load_in_4bit=False, dtype=None)
    gmodel.save_pretrained_gguf(str(out / "gguf"), gtok, quantization_method=quant)

    # Unsloth's output file name has changed between versions; find it.
    candidates = [p for p in list(out.rglob("*.gguf")) + list(Path.cwd().glob("*.gguf"))
                  if quant.replace("_", "").lower() in p.name.replace("_", "").lower()]
    if not candidates:
        raise SystemExit(f"No {quant} .gguf produced; files: {[p.name for p in out.rglob('*.gguf')]}")
    final = out / r.gguf_file
    shutil.move(str(candidates[0]), final)
    (out / "Modelfile").write_text(modelfile(f"./{r.gguf_file}"), encoding="utf-8")

    api.create_repo(r.gguf, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(final), path_in_repo=r.gguf_file, repo_id=r.gguf,
                    commit_message=f"{r.gguf_file}")
    api.upload_file(path_or_fileobj=str(out / "Modelfile"), path_in_repo="Modelfile", repo_id=r.gguf)
    size = os.path.getsize(final) / 1e9
    print(f"\n[export] uploaded {r.gguf_file} ({size:.2f} GB) to https://huggingface.co/{r.gguf}")
    print(f"[export] on your laptop: python -m monkeytrain.install --config {args.config}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
