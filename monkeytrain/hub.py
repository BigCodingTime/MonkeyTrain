"""Hugging Face Hub helpers: token lookup and repo names."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def get_token() -> str | None:
    """HF_TOKEN env var, then Kaggle Secrets, then `hf auth login`'s cached token."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient  # only exists on Kaggle
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            return token
    except Exception:
        pass
    from huggingface_hub import get_token as cached
    return cached()


def require_token() -> str:
    token = get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token found.\n"
            "  Local:  run `hf auth login` (or set HF_TOKEN)\n"
            "  Kaggle: Add-ons -> Secrets -> add HF_TOKEN and tick it for this notebook"
        )
    return token


def username(token: str) -> str:
    from huggingface_hub import whoami
    return whoami(token=token)["name"]


@dataclass
class Repos:
    data: str        # dataset: train.jsonl, eval.jsonl
    checkpoints: str  # model: last-checkpoint/, final/, status.json
    gguf: str        # model: the file Ollama runs


def repos(user: str, model_name: str) -> Repos:
    return Repos(data=f"{user}/monkey-data",
                 checkpoints=f"{user}/{model_name}-train",
                 gguf=f"{user}/{model_name}-gguf")


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
