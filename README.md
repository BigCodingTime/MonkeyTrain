# MonkeyTrain

Fine-tunes **Monkey**, the model behind MonkeyPaw, from Qwen2.5-Coder. Two sizes:

| Model | Base | Runs on laptop as | For MonkeyPaw profile |
|---|---|---|---|
| `monkey-0.5b` | Qwen2.5-Coder-0.5B-Instruct | ~0.5 GB (Q8_0) | `light` (battery saver) |
| `monkey-1.5b` | Qwen2.5-Coder-1.5B-Instruct | ~1 GB (Q4_K_M) | `balanced` / `max` |

Training runs free on Kaggle's GPUs overnight and resumes where it left off.
Running the finished model happens on your laptop through Ollama.

```
laptop: build_dataset ──► Hugging Face (private) ◄── Kaggle: train (repeat until done) ──► export
                                   │
laptop: install ◄──────────────────┘   then: evaluate, bench, use in MonkeyPaw
```

---

## One-time setup

### 1. Accounts
1. **Hugging Face** (free): https://huggingface.co/join
   - Create a token: *Settings → Access Tokens → Create new token → type **Write***. Copy it.
   - Optional but recommended: open https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k and accept its terms (adds general tool-calling practice; skipped otherwise).
2. **Kaggle** (free): https://www.kaggle.com
   - *Settings → Phone verification* (required for GPUs and internet).
3. **GitHub**: push this repo to GitHub so Kaggle can download the code. Public is simplest. If it's private, also create a GitHub token with read access and add it as a Kaggle secret named `GITHUB_TOKEN` (step 4).

### 2. Laptop
```powershell
cd C:\Gits\MonkeyTrain
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
hf auth login            # paste your Hugging Face write token
pytest                   # should pass
```
Install **Ollama**: https://ollama.com/download

### 3. Build the training data (laptop, ~20-40 min, downloads ~1.5 GB once)
```powershell
python -m monkeytrain.build_dataset
```
This turns ~18k real GitHub commits into MonkeyPaw agent sessions (find file → read → `edit_file` → summary), adds coding Q&A and tool-calling examples, and uploads everything to a **private** dataset `YOUR_HF_NAME/monkey-data`.
Try `--total 2000 --no-push` first if you want a quick look; the files land in `data/`.

### 4. Kaggle notebook
1. Kaggle → *Create → New Notebook* → *File → Import Notebook* → upload `notebooks/monkey_kaggle.ipynb`.
2. In the first code cell, set `REPO_URL` to your GitHub repo.
3. Right-hand panel: **Accelerator → GPU T4 x2**, **Internet → On**.
4. *Add-ons → Secrets → Add secret*: name `HF_TOKEN`, value = your Hugging Face write token. Tick the box so the notebook can use it. (Plus `GITHUB_TOKEN` if the repo is private.)

---

## Training

1. Set `CONFIG = "configs/monkey-0.5b.json"` for the first run. It's faster and tests the whole pipeline.
2. Click **Save Version → Save & Run All (Commit) → Save**.
3. Close the browser, shut down your laptop. It keeps running on Kaggle's servers.
4. Check progress any time: Kaggle → *Your Work* → the notebook → *Versions* → the running version's log. Checkpoints appear on `huggingface.co/YOUR_HF_NAME/monkey-0.5b-train` every 20 minutes.

**Resuming:** every run continues from the last checkpoint. If a run hits the time limit it prints *"Run the notebook again to continue"*. Just do *Save & Run All* again. If Kaggle cuts a run off or it crashes, you lose at most ~20 minutes. Kaggle gives ~30 GPU hours a week, and each run can last up to 12 hours.

**Finished:** the log prints `[check] P(<tool_call>) ...`, which should be ≥ 0.5, and the last run exports the model automatically (`monkey-0.5b-gguf` on Hugging Face). Then switch `CONFIG` to `configs/monkey-1.5b.json` and repeat.

To train a new version (e.g. after changing the data or training code), raise `"version"` in the config. The new run gets its own checkpoint repo and GGUF file, and the old ones are left alone.

To restart the current version from scratch: `python -m monkeytrain.train --config ... --restart` (or delete its `-train` repo on Hugging Face).

---

## Using Monkey on your laptop

```powershell
python -m monkeytrain.install --config configs/monkey-1.5b.json   # download + register with Ollama
ollama run monkey-1.5b                                              # chat with it
```

### Speed vs. battery
```powershell
python -m monkeytrain.bench --model monkey-1.5b
```
This prints the reading and writing speed at different CPU thread counts. Fewer threads use less battery and leave the rest of the computer more responsive. Use the numbers to decide MonkeyPaw's `light` / `balanced` / `max` thread counts. MonkeyPaw sends the thread count and context size with every request, so you can change them at any time without reinstalling.

Ollama settings that help on a low-spec laptop (set as Windows environment variables, then restart Ollama):
- `OLLAMA_MAX_LOADED_MODELS=1`: never keep two models in RAM.
- `OLLAMA_NUM_PARALLEL=1`: one request at a time, less memory.
- `OLLAMA_KEEP_ALIVE=5m`: free the RAM 5 minutes after the last use.

### Is it better than plain Qwen?
```powershell
ollama pull qwen2.5-coder:1.5b
python -m monkeytrain.evaluate --model qwen2.5-coder:1.5b
python -m monkeytrain.evaluate --model monkey-1.5b
```
Both run 30 held-out real-commit tasks as an agent with MonkeyPaw's tools and print a scorecard (finished, changed the right file, exact match with the real commit, whole-file rewrites, tool errors, speed). This takes roughly 30-60 minutes per model on CPU.

---

## Files

| Path | What it is |
|---|---|
| `monkeytrain/tools.py` | MonkeyPaw's tools and Monkey's system prompt, shared with MonkeyPaw |
| `monkeytrain/render.py` | Prompt format and the Ollama template, kept identical by tests |
| `monkeytrain/sandbox.py` | In-memory workspace running the tools, for data generation and evaluation |
| `monkeytrain/sources/` | Dataset converters (CommitPackFT, Self-OSS-Instruct, xLAM) |
| `monkeytrain/build_dataset.py` | Builds and uploads the training data |
| `monkeytrain/train.py` / `export.py` | Kaggle: resumable LoRA training, and GGUF export |
| `monkeytrain/install.py` / `bench.py` / `evaluate.py` | Laptop: install into Ollama, speed test, scorecard |
| `configs/*.json` | Per-model settings |
| `docs/design.md` | Why it's built this way |
