"""
Fine-tune Qwen2.5-Coder into Monkey with LoRA plus two trained token rows.
Runs on a Kaggle T4 GPU with plain Transformers + PEFT.

Not Unsloth: its fast embedding/output code bypasses PEFT's trainable-token
wrapper, so the <tool_call> rows never trained (v2 run: bit-identical to the
base after 1,264 steps). TokenRowsWatch stops a run early if that ever recurs.

  python -m monkeytrain.train --config configs/monkey-1.5b.json

Safe to run again and again: every run resumes from the last checkpoint on
Hugging Face. Checkpoints are uploaded every `save_every_minutes`, and the run
saves and stops by itself before Kaggle's 12-hour limit. When training is
finished it uploads the adapter to `final/` and later runs do nothing.

Exit marker: writes TRAINING_DONE next to --output when training is finished,
so the notebook knows whether to export.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from .check import tool_call_probability
from .encode import encode_session
from .render import TRAINABLE_TOKENS
from .hub import load_config, repos, require_token, username

STATUS = "status.json"
LAST = "last-checkpoint"
FINAL = "final"


def read_status(api, repo: str) -> dict:
    from huggingface_hub import hf_hub_download
    try:
        return json.loads(Path(hf_hub_download(repo, STATUS, token=api.token)).read_text())
    except Exception:
        return {}


def write_status(api, repo: str, status: dict) -> None:
    api.upload_file(path_or_fileobj=json.dumps(status, indent=2).encode(), path_in_repo=STATUS,
                    repo_id=repo, commit_message=f"status: {status.get('state')} step {status.get('step')}")


def make_hub_callback(api, repo: str, save_every_s: float, budget_s: float, total_steps_ref: dict):
    from transformers import TrainerCallback

    class HubCheckpoints(TrainerCallback):
        """Time-based saving, upload to the Hub, and a clean stop before the time limit."""

        def __init__(self):
            self.started = time.time()
            self.last_save = time.time()
            self.out_of_time = False

        def on_step_end(self, args, state, control, **kw):
            now = time.time()
            if now - self.last_save >= save_every_s:
                control.should_save = True
            if now - self.started >= budget_s:
                self.out_of_time = True
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kw):
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not ckpt.exists():
                return
            t0 = time.time()
            api.upload_folder(repo_id=repo, folder_path=str(ckpt), path_in_repo=LAST,
                              delete_patterns="*", commit_message=f"checkpoint step {state.global_step}")
            write_status(api, repo, {"state": "training", "step": state.global_step,
                                     "total_steps": total_steps_ref.get("n"), "time": time.ctime()})
            self.last_save = time.time()
            print(f"[hub] uploaded checkpoint at step {state.global_step} ({time.time() - t0:.0f}s)", flush=True)

    return HubCheckpoints()


def trainable_token_ids(tokenizer) -> list[int]:
    """Embedding rows to train directly (see render.TRAINABLE_TOKENS)."""
    ids = tokenizer.convert_tokens_to_ids(TRAINABLE_TOKENS)
    if any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise SystemExit(f"Tokenizer is missing one of {TRAINABLE_TOKENS}")
    return ids


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_model(base_model: str, cfg: dict, token: str | None = None, dtype=None):
    """Base model + LoRA on every linear layer + trainable <tool_call> rows."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, token=token)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, bias="none",
        target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
        # LoRA can't teach a token the base model never produces; train those embedding rows too.
        # Qwen ties input and output embeddings, so this also changes the output rows.
        trainable_token_indices={"embed_tokens": trainable_token_ids(tokenizer)})
    model = get_peft_model(model, lora)
    # fp16 mixed precision needs fp32 trainable weights (the frozen base stays 16-bit).
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
    return model, tokenizer


def make_token_watch(model, after_steps: int = 50):
    """Stop the run if the <tool_call> rows haven't changed after `after_steps` steps."""
    from transformers import TrainerCallback

    deltas = [(n, p) for n, p in model.named_parameters() if "trainable_tokens_delta" in n]
    if not deltas:
        raise SystemExit("No trainable token rows in the model; the <tool_call> fix isn't active.")
    start = {n: p.detach().clone() for n, p in deltas}

    class TokenRowsWatch(TrainerCallback):
        checked = False

        def on_step_end(self, args, state, control, **kw):
            if self.checked or state.global_step < after_steps:
                return control
            self.checked = True
            moved = max((p.detach() - start[n].to(p.device)).abs().max().item() for n, p in deltas)
            print(f"[check] <tool_call> rows changed by {moved:.2e} after {state.global_step} steps", flush=True)
            if moved == 0:
                raise SystemExit("[check] The <tool_call> token rows aren't training. Stopping so no GPU time "
                                 "is wasted; send this log to get it fixed.")
            return control

    return TokenRowsWatch()


def training_kwargs(cfg: dict, out: Path, bf16: bool) -> dict:
    """TrainingArguments for this run, for both Transformers 4.x and 5.x."""
    import dataclasses
    from transformers import TrainingArguments
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}

    kw = dict(
        output_dir=str(out / "ckpt"),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        optim="adamw_torch",
        bf16=bf16, fp16=not bf16,
        logging_steps=10,
        save_strategy="steps", save_steps=10**9,  # saving is time-based, see HubCheckpoints
        save_total_limit=1,
        seed=cfg["seed"],
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=2,
    )
    # Batch sessions of similar length together (less padding). v5 renamed the option.
    if "train_sampling_strategy" in fields:
        kw["train_sampling_strategy"] = "group_by_length"
    else:
        kw["group_by_length"] = True
    # v5 dropped warmup_ratio; warmup_steps takes a fraction of training instead.
    if "warmup_ratio" in fields:
        kw["warmup_ratio"] = cfg["warmup_ratio"]
    else:
        kw["warmup_steps"] = cfg["warmup_ratio"]
    unknown = set(kw) - fields
    if unknown:
        raise SystemExit(f"This Transformers version doesn't accept: {sorted(unknown)}")
    return kw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="/kaggle/working/run" if Path("/kaggle").exists() else "run")
    ap.add_argument("--time-budget-hours", type=float, default=None, help="override config")
    ap.add_argument("--restart", action="store_true", help="ignore existing checkpoints and start over")
    args = ap.parse_args()

    started = time.time()
    cfg = load_config(args.config)
    budget_h = args.time_budget_hours or cfg["time_budget_hours"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    done_marker = out / "TRAINING_DONE"
    done_marker.unlink(missing_ok=True)

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    token = require_token()
    api = HfApi(token=token)
    r = repos(username(token), cfg)
    api.create_repo(r.checkpoints, private=True, exist_ok=True)

    status = {} if args.restart else read_status(api, r.checkpoints)
    if status.get("state") == "done":
        print(f"[train] {cfg['name']} already finished training (step {status.get('step')}). "
              f"Use --restart to train again from scratch.")
        done_marker.write_text("done")
        return

    resume = None
    if status.get("state") == "training":
        resume = snapshot_download(r.checkpoints, allow_patterns=[f"{LAST}/*"], token=token,
                                   local_dir=str(out / "resume"))
        resume = str(Path(resume) / LAST)
        print(f"[train] resuming from step {status.get('step')} of {status.get('total_steps')}")
    else:
        print("[train] no checkpoint found — starting fresh")

    # ---- model: 16-bit base + LoRA + trainable <tool_call> rows ----
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    # Real bf16 needs Ampere (sm_80) or newer; T4 is sm_75 and would emulate it slowly.
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    torch.manual_seed(cfg["seed"])
    model, tokenizer = build_model(cfg["base_model"], cfg, token, torch.bfloat16 if bf16 else torch.float16)
    model.to("cuda")
    model.print_trainable_parameters()

    # ---- data ----
    train_path = hf_hub_download(r.data, "train.jsonl", repo_type="dataset", token=token)
    rows, skipped = [], 0
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            enc = encode_session(json.loads(line), tokenizer, cfg["max_seq_len"])
            if enc is None:
                skipped += 1
            else:
                rows.append(enc)
    n_tokens = sum(len(x["input_ids"]) for x in rows)
    print(f"[data] {len(rows)} sessions ({skipped} skipped as too long), {n_tokens / 1e6:.1f}M tokens")
    dataset = Dataset.from_list(rows)

    targs = TrainingArguments(**training_kwargs(cfg, out, bf16))
    total_ref: dict = {}
    callback = make_hub_callback(api, r.checkpoints, cfg["save_every_minutes"] * 60,
                                 budget_h * 3600 - (time.time() - started), total_ref)
    trainer = Trainer(
        model=model, args=targs, train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100),
        callbacks=[callback, make_token_watch(model)],
    )
    total_ref["n"] = -(-len(dataset) // (cfg["batch_size"] * cfg["grad_accum"])) * cfg["epochs"]
    print(f"[train] {total_ref['n']} optimizer steps; saving every {cfg['save_every_minutes']} min; "
          f"stopping by {budget_h:.1f} h")

    trainer.train(resume_from_checkpoint=resume)

    if callback.out_of_time:
        print("\n[train] time budget reached — checkpoint uploaded. Run the notebook again to continue.")
        return

    tasks_path = hf_hub_download(r.data, "eval_tasks.jsonl", repo_type="dataset", token=token)
    tasks = [json.loads(line) for line in open(tasks_path, encoding="utf-8")]
    model.eval()
    p = tool_call_probability(model, tokenizer, tasks)
    print(f"[check] P(<tool_call>) as first reply token on held-out tasks: {p:.2f} (want >= 0.5)")

    final = out / FINAL
    shutil.rmtree(final, ignore_errors=True)
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    api.upload_folder(repo_id=r.checkpoints, folder_path=str(final), path_in_repo=FINAL,
                      delete_patterns="*", commit_message="final adapter")
    write_status(api, r.checkpoints, {"state": "done", "step": trainer.state.global_step,
                                      "total_steps": total_ref["n"], "time": time.ctime()})
    done_marker.write_text("done")
    print(f"\n[train] finished. Adapter uploaded to https://huggingface.co/{r.checkpoints}/tree/main/{FINAL}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Kaggle gives 2x T4; train on one
    main()
