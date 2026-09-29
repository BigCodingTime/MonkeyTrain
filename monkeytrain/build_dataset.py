"""
Build Monkey's training data and upload it to a private Hugging Face dataset.

Runs on a normal laptop (CPU only, downloads ~1 GB once).

  python -m monkeytrain.build_dataset                   # build + upload
  python -m monkeytrain.build_dataset --no-push         # build locally only
  python -m monkeytrain.build_dataset --total 2000      # small test build

Output (data/): train.jsonl, eval_tasks.jsonl, stats.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, deque
from pathlib import Path

from .hub import data_repo, get_token, username
from .sources import commitpack, selfoss, xlam

DEFAULT_MIX = {"commitpack": 0.75, "selfoss": 0.15, "xlam": 0.10}


def parse_mix(s: str) -> dict[str, float]:
    mix = {}
    for part in s.split(","):
        k, v = part.split("=")
        mix[k.strip()] = float(v)
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def build_commitpack(n_train: int, n_eval: int, rng: random.Random, token: str | None):
    train, tasks = [], []
    shares = commitpack.LANGUAGES
    total_share = sum(shares.values())
    for lang, share in shares.items():
        want_train = round(n_train * share / total_share)
        want_eval = round(n_eval * share / total_share)
        if want_train + want_eval == 0:
            continue
        print(f"[commitpack] {lang}: downloading...", flush=True)
        path = commitpack.download(lang, token)
        decoy_pool: deque[tuple[str, str]] = deque(maxlen=64)
        got_train = got_eval = seen = 0
        for raw in commitpack.iter_records(path, rng):
            seen += 1
            rec = commitpack.usable(raw)
            if rec is None:
                if raw.get("old_contents") and len(raw["old_contents"]) < commitpack.MAX_CHARS:
                    decoy_pool.append((raw.get("old_file", "").replace("\\", "/"), raw["old_contents"].replace("\r\n", "\n")))
                continue
            decoys = rng.sample(list(decoy_pool), k=min(len(decoy_pool), rng.randint(2, 5)))
            if got_eval < want_eval and rec["old"]:
                tasks.append(commitpack.make_eval_task(rec, rng, decoys))
                got_eval += 1
            elif got_train < want_train:
                session = commitpack.make_session(rec, rng, decoys)
                if session:
                    train.append(session)
                    got_train += 1
            decoy_pool.append((rec["path"], rec["old"]))
            if got_train >= want_train and got_eval >= want_eval:
                break
        print(f"[commitpack] {lang}: {got_train} train, {got_eval} eval (scanned {seen})", flush=True)
    return train, tasks


def take(rows, make, n: int, rng: random.Random, name: str) -> list[dict]:
    """Shuffle all rows, keep the first n that make valid sessions."""
    out, buf = [], list(rows)
    rng.shuffle(buf)
    for row in buf:
        s = make(row, rng)
        if s:
            out.append(s)
            if len(out) >= n:
                break
    print(f"[{name}] {len(out)} sessions")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, default=24000, help="training sessions (default 24000)")
    ap.add_argument("--eval", type=int, default=300, help="held-out evaluation tasks")
    ap.add_argument("--mix", type=parse_mix, default=DEFAULT_MIX, help="e.g. commitpack=0.75,selfoss=0.15,xlam=0.1")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-push", action="store_true", help="don't upload to Hugging Face")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    token = get_token()
    mix = args.mix
    sessions: list[dict] = []

    n_cp = round(args.total * mix.get("commitpack", 0))
    cp_train, eval_tasks = build_commitpack(n_cp, args.eval, rng, token)
    sessions += cp_train

    if mix.get("selfoss"):
        sessions += take(selfoss.iter_rows(token), selfoss.make_session, round(args.total * mix["selfoss"]), rng, "selfoss")

    if mix.get("xlam"):
        try:
            sessions += take(xlam.iter_rows(token), xlam.make_session, round(args.total * mix["xlam"]), rng, "xlam")
        except Exception as e:  # gated: terms not accepted or no token
            print(f"[xlam] skipped ({type(e).__name__}). Accept the terms at "
                  f"https://huggingface.co/datasets/{xlam.REPO} to include it.")

    rng.shuffle(sessions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(out / "eval_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in eval_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    stats = {"train": len(sessions), "eval_tasks": len(eval_tasks),
             "by_source": Counter(s["source"] for s in sessions), "seed": args.seed}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/train.jsonl ({len(sessions)}) and eval_tasks.jsonl ({len(eval_tasks)})")
    print(json.dumps(stats["by_source"]))

    if args.no_push:
        return
    if not token:
        raise SystemExit("Not logged in to Hugging Face; run `hf auth login` or use --no-push.")
    from huggingface_hub import HfApi
    repo = data_repo(username(token))
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(out),
                      allow_patterns=["train.jsonl", "eval_tasks.jsonl", "stats.json"],
                      commit_message=f"dataset: {len(sessions)} sessions")
    print(f"uploaded to https://huggingface.co/datasets/{repo} (private)")


if __name__ == "__main__":
    main()
