"""
Runs train.py's Trainer setup on CPU with a tiny random Qwen2 + LoRA: time-based
checkpointing, stopping at the time budget, and resuming from the uploaded
checkpoint. Unsloth and the GPU are the only parts not covered.

Needs: pip install torch transformers datasets accelerate peft (skipped otherwise)
"""

import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

from monkeytrain.encode import encode_session  # noqa: E402
from monkeytrain.check import tool_call_probability  # noqa: E402
from monkeytrain.export import merge_adapter  # noqa: E402
from monkeytrain.train import LAST, make_hub_callback, trainable_token_ids, training_kwargs  # noqa: E402

CFG = {"batch_size": 2, "grad_accum": 2, "epochs": 1, "lr": 1e-3, "warmup_ratio": 0.03, "seed": 1}

SESSION = {"messages": [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "find foo"},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "grep", "arguments": {"pattern": "foo"}}}]},
    {"role": "tool", "content": "a.py:1:foo = 1"},
    {"role": "assistant", "content": "Found it."},
]}


class FakeHub:
    """Stands in for HfApi: 'uploads' by copying into a local folder."""

    def __init__(self, root: Path):
        self.root = root
        self.uploads = 0

    def upload_folder(self, repo_id, folder_path, path_in_repo, delete_patterns=None, commit_message=""):
        dest = self.root / path_in_repo
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(folder_path, dest)
        self.uploads += 1

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, commit_message=""):
        (self.root / path_in_repo).write_bytes(path_or_fileobj)


@pytest.fixture(scope="module")
def tok():
    try:
        t = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    t.padding_side = "right"
    return t


def tiny_base(tok):
    # Tied input/output embeddings, like Qwen2.5-Coder 0.5B/1.5B.
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                   tie_word_embeddings=True)
    return transformers.Qwen2ForCausalLM(cfg)


def tiny_model(tok):
    # Same LoRA + trainable token rows as train.py.
    lora = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
                           trainable_token_indices={"embed_tokens": trainable_token_ids(tok)})
    return peft.get_peft_model(tiny_base(tok), lora)


def make_trainer(tok, out: Path, hub: FakeHub, save_every_s: float, budget_s: float, cfg=CFG, n=40):
    from datasets import Dataset
    rows = [encode_session(SESSION, tok, 4096) for _ in range(n)]
    assert all(rows)
    kw = training_kwargs(cfg, out, bf16=False)
    kw.update(fp16=False, optim="adamw_torch", dataloader_num_workers=0, use_cpu=True)  # CPU stand-ins
    callback = make_hub_callback(hub, "user/repo", save_every_s, budget_s, {"n": n // 4})
    trainer = transformers.Trainer(
        model=tiny_model(tok), args=transformers.TrainingArguments(**kw), train_dataset=Dataset.from_list(rows),
        data_collator=transformers.DataCollatorForSeq2Seq(tok, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100),
        callbacks=[callback])
    return trainer, callback


def test_stop_at_budget_then_resume(tok, tmp_path):
    hub = FakeHub(tmp_path / "hub")
    hub.root.mkdir()

    # Run 1: save after every step, budget runs out right away -> stops early with a checkpoint.
    trainer, cb = make_trainer(tok, tmp_path / "run1", hub, save_every_s=0, budget_s=0)
    trainer.train()
    assert cb.out_of_time
    assert hub.uploads >= 1
    status = json.loads((hub.root / "status.json").read_text())
    assert status["state"] == "training"
    stopped_at = status["step"]
    assert 0 < stopped_at < 10
    assert (hub.root / LAST / "trainer_state.json").exists()
    assert (hub.root / LAST / "adapter_model.safetensors").exists()

    # Run 2: resume from the 'uploaded' checkpoint and finish all 10 steps.
    trainer, cb = make_trainer(tok, tmp_path / "run2", hub, save_every_s=10**6, budget_s=10**6)
    trainer.train(resume_from_checkpoint=str(hub.root / LAST))
    assert not cb.out_of_time
    assert trainer.state.global_step == 10


TASKS = [{"system": "SYS", "request": "find foo"}]


def test_tool_call_token_is_learned_and_survives_merge(tok, tmp_path):
    """The v1 bug: LoRA alone never learned <tool_call>. With trainable token rows,
    training teaches it, and saving + PEFT merge (export's path) keep it."""
    base_dir = tmp_path / "base"
    tiny_base(tok).save_pretrained(base_dir)
    before = tool_call_probability(tiny_base(tok), tok, TASKS)

    hub = FakeHub(tmp_path / "hub")
    hub.root.mkdir()
    fast = {**CFG, "lr": 5e-2, "batch_size": 4, "grad_accum": 1, "epochs": 1}
    trainer, _ = make_trainer(tok, tmp_path / "run", hub, 10**6, 10**6, cfg=fast, n=120)
    trainer.train()
    adapter = tmp_path / "adapter"
    trainer.model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    trained = tool_call_probability(trainer.model, tok, TASKS)

    merged, mtok = merge_adapter(str(base_dir), adapter)
    after_merge = tool_call_probability(merged, mtok, TASKS)
    assert before < 0.01
    assert trained > 0.3  # tiny random model, 30 steps: ~0.46 observed
    assert after_merge == pytest.approx(trained, abs=0.02)
