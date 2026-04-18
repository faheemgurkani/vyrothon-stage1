# Pocket-Agent

Fine-tune an open-weight base model (**≤ 2B parameters**) for **structured tool calls** in an on-device mobile-assistant setting: emits JSON inside `<tool_call>...</tool_call>` or plain text for refusals. This repository implements **synthetic data generation**, **QLoRA fine-tuning**, **merge + quantization**, **evaluation**, a **Gradio-style demo** (in the notebook), and **`inference.py`** for the grader.

---

## Hackathon alignment (problem summary)

| Requirement | Notes |
|-------------|--------|
| Base model ≤ 2B | `Qwen/Qwen2.5-0.5B-Instruct` (~494M) |
| Tool schema | Five tools: `weather`, `calendar`, `convert`, `currency`, `sql` — see [starter/tool_schemas.json](starter/tool_schemas.json) |
| Behaviors | Single-turn + multi-turn + refusals + argument fidelity — trained via templated synthetic data |
| Offline inference | `inference.py` uses only `torch` + `transformers` (no network imports) |
| Artifacts | LoRA adapter + merged/quantized model produced by the notebook (see below) |

**Compute:** Google Colab T4 (16 GB VRAM) as primary training path — [pocket_agent_colab.ipynb](pocket_agent_colab.ipynb).

---

## What is committed vs what is not (GitHub limits)

**GitHub blocks any single file larger than ~100 MB.** Typical merged weights are **one `model.safetensors` on the order of ~0.9–1.0 GB** (FP16 merge before/without aggressive shrinking), which **cannot** be pushed as a normal Git blob.

### Included in this repository (safe to push)

- [pocket_agent_colab.ipynb](pocket_agent_colab.ipynb) — full pipeline (data, train, quantize, eval, demo cell, optional `inference.py` writer).
- [inference.py](inference.py) — `run(prompt, history) -> str` for the grader.
- [Makefile](Makefile) — install / eval / demo helpers.
- [starter/](starter/) — `public_test.jsonl`, `teacher_examples.jsonl`, `tool_schemas.json`, `eval_harness_contract.py`.
- [.gitignore](.gitignore) — excludes `artifacts/`, `*.safetensors`, caches, secrets.

### Not included (by design) — large / regenerated artifacts

| Path (after training) | Role | Why not in Git |
|----------------------|------|----------------|
| `artifacts/quantized_model/model.safetensors` | Merged FP16 (or similar) full weights | Typically **~500 MB–1 GB** — **exceeds GitHub’s 100 MB per-file limit** |
| `artifacts/adapter/` | LoRA adapter | Can be tens of MB; optional to ship; often empty if merge-only path was used |
| `artifacts/tokenizer/` | Tokenizer copy | Redundant with files under `quantized_model/` for inference |

**How reviewers / you obtain weights**

1. **Run the notebook** on Colab (T4), then download `artifacts/` (zip, Google Drive copy, or extension “server mount” download) — see notebook Section 0b (Google Drive) and project tips.
2. **GitHub Releases** — attach `artifacts.zip` or split archives **outside** normal commits (optional workflow).
3. **Hugging Face Hub** — upload adapter or merged model to a **private/public** repo and `huggingface-cli download` in graded environments (if rules allow).
4. **Regenerate** — clone repo → run notebook end-to-end — satisfies “script/notebook produces artifacts.”

Document any **Release URL** or **Hub repo id** you use for submission in your hackathon form if the platform asks for artifact links.

---

## Submission checklist (repo contents)

Per hackathon instructions, this repo aims to provide:

- [x] **Training codebase** — notebook + `Makefile`; synthetic data in-notebook; eval harness in `starter/`.
- [x] **Trained artifacts** — produced by notebook (not stored in Git); LoRA under `artifacts/adapter/` when training runs; merged model under `artifacts/quantized_model/` — **obtain via Colab run or external hosting** (see above).
- [x] **Chatbot demo** — Gradio section in [pocket_agent_colab.ipynb](pocket_agent_colab.ipynb); CPU load after quantize cell.
- [x] **`inference.py`** — `def run(prompt: str, history: list[dict]) -> str`.
- [x] **README** — this file: setup, design, model choice, artifacts policy, limits.

Reproduce end-to-end:

```bash
make install          # or pip from Makefile
# Then: run pocket_agent_colab.ipynb on Colab T4, or locally with GPU
```

---

## Model

| Property | Value |
|----------|--------|
| **Base model** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Parameters** | 494 M |
| **Fine-tuning** | QLoRA (4-bit NF4 + LoRA on q/k/v/o projections) |
| **Inference (notebook path)** | Merged model + FP16 save; optional `torch.quantization.quantize_dynamic` int8 on CPU |
| **Size note** | Hackathon **gate** often cited as **≤ 500 MB** quantized; merged FP16 may be **larger** until further compression — tune Section 7 or use additional quant (GPTQ/AWQ/gguf) to meet gates. |

---

## Results and logs (embedded notebook outputs)

The committed [pocket_agent_colab.ipynb](pocket_agent_colab.ipynb) includes **saved cell outputs** from an end-to-end Colab run (paths dated **2026-04-18**). The run used **CPU-only for training** (`torch.cuda.is_available()` false in the captured `runtime_detect` cell), so **QLoRA training was skipped** and merge/quantize used a **base-model fallback** where no adapter was present. Treat the numbers below as **one concrete log trace**, not a guarantee for every rerun (seeds, GPU presence, and `adapter/` contents will change outcomes).

### Environment and setup

| Log line | Meaning |
|----------|---------|
| `Configuration loaded (run next cell...)` | Config cell executed. |
| `ARTIFACTS_DIR = './artifacts'` / `ADAPTER_DIR` / `QUANTIZED_DIR` / `TOKENIZER_DIR` | Local VM paths (Drive disabled in that run). |
| `No GPU detected — running on CPU. Training will be skipped...` | `HAS_GPU` false → Sections 4–5 training branches not taken. |
| `Device: cpu, dtype: torch.float32` | Dtype for non-training path. |
| `All packages installed.` | Section 1 `%pip` install completed. |

### Hugging Face Hub (tokenizer download)

| Log | Meaning |
|-----|---------|
| `HF_TOKEN` secret timeout / unauthenticated requests | Colab UI secret fetch failed in that environment; model still downloaded with public access warnings. |
| `Warning: You are sending unauthenticated requests...` | Rate-limit / auth advisory from `huggingface_hub`. |

### Synthetic data and splits

| Metric | Value (logged) |
|--------|----------------|
| Raw examples generated | **658** |
| Train split | **593** |
| Validation split | **65** |
| Sample slice / user (first train record) | Slice **A**, user asks for **m² → ft²** conversion |

Train/val `Dataset` rows matched **593 / 65** in the same run.

### Tokenizer (Section 4)

| Field | Value |
|-------|--------|
| `BASE_MODEL` load | `Qwen/Qwen2.5-0.5B-Instruct` |
| Vocab size | **151643** |
| Saved to | `./artifacts/tokenizer` |

### Training (Sections 4–5)

| Log | Meaning |
|-----|---------|
| `No GPU detected — skipping model load for training` | Base QLoRA model not instantiated for SFT. |
| `Skipping training (no GPU). Load pre-trained artifacts in Section 7.` | No `trainer.train()` loss curves in this trace. |

### Embedded eval harness (Section 6)

| Log | Meaning |
|-----|---------|
| `Evaluation utilities loaded.` | Scoring helpers initialized. |
| `All scoring sanity checks passed.` | Contract checks for +1.0 / +0.5 / −0.5 cases. |
| `run() function defined...` | Notebook `run()` ready after `set_eval_model()`. |
| `No model loaded — skipping validation eval.` | No GPU-trained model at that step. |
| `Model not loaded — skipping dev set eval.` | Per-example dev run skipped until CPU model registered later. |

### Merge, save, and on-disk sizes (Section 7)

| Step | Log |
|------|-----|
| Merge | `Step 1: Loading base model in fp32 for merging...` |
| Adapter | `WARNING: Adapter not found at ./artifacts/adapter. Using base model without LoRA.` |
| Post-merge | `Merge complete.` |
| Save | `Step 4: Saving merged model as fp16 safetensors...` |
| Reported directory total | `Saved model size: 999.5 MB` (notebook sums all files under `QUANTIZED_DIR`) |

`ls -la` on `/content/artifacts/quantized_model` in the same session showed:

- `model.safetensors` — **988097536** bytes (~**942 MiB** file; **GitHub-ineligible** as a normal blob).
- `tokenizer.json` — **11421892** bytes (~11 MB).
- `adapter/` directory listing **total 8** (empty dir aside from `.` / `..`) — consistent with “no LoRA checkpoint” merge.

Packaging:

- `artifacts.zip` — **765739956** bytes (~**730 MB** archive; still over single-file Git limits).

### CPU load + dynamic int8 (Section 7)

| Log | Meaning |
|-----|---------|
| `Loading saved model on CPU for inference...` | Reload FP16 weights from disk. |
| `Applying torch int8 dynamic quantization...` | `torch.quantization.quantize_dynamic` on `nn.Linear`. |
| `DeprecationWarning: torch.ao.quantization is deprecated...` | PyTorch 2.x forward-compat notice (suggests `torchao` migration paths). |
| `CPU model ready.` | `set_eval_model(cpu_model, cpu_tokenizer)` succeeded. |

### CPU latency benchmark (Section 8)

Eight fixed prompts (weather, convert, currency, calendar, SQL, chitchat, typo-adversarial, multi-turn “convert that”). **Reported timings (ms)**:

| Prompt (abbrev.) | Latency (ms) |
|------------------|-------------:|
| Weather Paris | 14365.6 |
| Miles → km | 14074.5 |
| USD → EUR | 3228.9 |
| Calendar list | 8891.5 |
| SQL | 3184.7 |
| Chitchat | 5187.1 |
| Typo weather Mumbai | 13732.0 |
| Multi-turn “convert to euros” | 3550.9 |

**Aggregate:** mean **8276.9 ms**, max **14365.6 ms**; printed gate: **`200 ms/turn gate: FAIL`** with hint to reduce `MAX_NEW_TOKENS` (and/or model size / quant / hardware). Sample decoded prefixes in the log show **off-format or weak tool JSON** for several prompts — expected when running **unadapted base** weights under this trace.

### `inference.py` writer (Section 9)

| Log | Meaning |
|-----|---------|
| `inference.py written to disk.` | String template flushed to repo root in Colab cwd. |
| `AST check passed: no network imports in inference.py.` | Local scan for grader compliance. |

### Google Drive / export (auxiliary cells)

| Log | Meaning |
|-----|---------|
| `Mounted at /content/drive` | `drive.mount` succeeded when that cell was run. |
| `files.download` JS payload | Zip download initiated in browser Colab (`artifacts.zip` size matches above). |

### Gradio demo (Section 10)

| Log | Meaning |
|-----|---------|
| Gradio `UserWarning` / `DeprecationWarning` | Defaults for `Chatbot` `type` / `allow_tags` — update Gradio API when upgrading. |
| `KeyboardInterrupt` during `gr.Blocks` / FastAPI route setup | Demo build interrupted in-session (e.g. stop button); **not** a failed install — rerun the cell to serve. |

### How to refresh this section

Re-run the notebook on **Colab T4** with `HAS_GPU=True`, then **Clear Outputs / Save** or export fresh logs. For grading, the important reproducible artifacts are **trained `adapter/`**, **merged weights**, and **`make eval`** on `starter/public_test.jsonl` after copying `artifacts/` locally.

---

## Tool schema (outputs)

The model emits either:

```text
<tool_call>{"tool": "<name>", "args": {...}}</tool_call>
```

or **plain natural language** for refusals.

```json
{"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}
{"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}
{"tool": "convert",  "args": {"value": "number", "from_unit": "string", "to_unit": "string"}}
{"tool": "currency", "args": {"amount": "number", "from": "ISO3", "to": "ISO3"}}
{"tool": "sql",      "args": {"query": "string"}}
```

---

## Quickstart (Google Colab)

1. Upload [pocket_agent_colab.ipynb](pocket_agent_colab.ipynb) to [Colab](https://colab.research.google.com) (or use the Colab VS Code extension with a T4 runtime).
2. **Config** → set paths; optional **`USE_GOOGLE_DRIVE = True`** so `artifacts/` persist under Drive.
3. Run **Section 0b** (artifact paths + Drive mount if enabled).
4. Run remaining cells through **training**, **merge/quantize**, then **`inference.py`** writer and **Gradio** if desired.
5. Copy `artifacts/` off the VM (Drive, zip + download in browser Colab, or extension server mount).

Teacher LLM augmentation: set `USE_TEACHER_LLM = True` and configure `TEACHER_BACKEND` (`openai` / `groq` / `anthropic` / `ollama`) and env vars as documented in the notebook.

---

## Local development

```bash
cd pocket-agent
make install
```

- **`make eval`** — requires `artifacts/quantized_model/` (see below).
- **`make demo`** — see [Demo usage](#demo-usage).

Full pipeline via notebook or:

```bash
make all   # uses jupyter nbconvert — needs GPU for training step
```

---

## Repository layout

```
pocket-agent/
├── .gitignore
├── pocket_agent_colab.ipynb
├── inference.py
├── Makefile
├── README.md
└── starter/
    ├── eval_harness_contract.py
    ├── public_test.jsonl
    ├── teacher_examples.jsonl
    └── tool_schemas.json
```

**Artifacts** (`artifacts/`) are gitignored — create them by running the notebook.

---

## Required artifacts and folder layout (inference + demos)

Everything that **loads the model** ([`inference.py`](inference.py), `make eval`, `make demo`, and the **Section 10** Gradio UI in the notebook) expects a single writable tree under the **Pocket-Agent project root** (the directory that contains `inference.py`). For example, if your tree is `competitions/pocket-agent/inference.py`, then `competitions/pocket-agent/artifacts/` is the right place — **not** a parent folder like `competitions/artifacts`. Paths are fixed in code: `Path(__file__).parent / "artifacts" / "quantized_model"`.

### Minimum layout to run the demo

You only need **`artifacts/quantized_model/`** filled by the notebook’s merge/save step (Section 7). **`artifacts/adapter/`** and **`artifacts/tokenizer/`** are **not** read by `inference.py` at runtime; they are for training / optional bookkeeping.

```
pocket-agent/                         # cwd for all CLI / Gradio / Python imports
├── inference.py
├── Makefile
├── starter/
└── artifacts/                        # you create this (notebook or unzip from Drive/Release)
    ├── adapter/                      # optional for running the demo — keep if you want to re-merge or inspect LoRA
    │   └── (adapter_config.json, adapter_model.safetensors, … after training)
    ├── tokenizer/                    # optional duplicate — Section 4 saves tokenizer here; inference does not use this path
    └── quantized_model/              # REQUIRED — must exist for demo / eval / grader
        ├── config.json
        ├── generation_config.json
        ├── model.safetensors         # merged FP16 weights (large file; not in Git)
        ├── tokenizer_config.json
        ├── tokenizer.json            # single-file tokenizer (Qwen-style)
        └── chat_template.jinja
```

| Path | Needed to run `inference.py` / demo? |
|------|-------------------------------------|
| `artifacts/quantized_model/*` (Hugging Face format above) | **Yes** — loader uses `AutoTokenizer.from_pretrained` + `AutoModelForCausalLM.from_pretrained` on this directory. |
| `artifacts/adapter/` | **No** for inference — only if you train or re-run merge with a new adapter. |
| `artifacts/tokenizer/` | **No** — redundant if `quantized_model/` contains tokenizer files (as the notebook saves). |

If `quantized_model/` is missing or incomplete, `inference.py` raises **`FileNotFoundError`** pointing at `artifacts/quantized_model`.

---

## Demo usage

All demos assume **dependencies installed** (`make install` or the notebook’s `%pip` cell) and **`artifacts/quantized_model/`** present as in the table above. Run commands from the **`pocket-agent/`** directory so `import inference` resolves and paths match.

### Google Colab (Gradio — notebook Section 10)

1. Run the notebook through **Section 7** (merge, save FP16, load on CPU, `set_eval_model`) so the model is in memory **or** ensure `artifacts/` is populated from Drive/zip on the VM.
2. Execute **Section 9** if you want a fresh `inference.py` on disk (optional if you cloned the repo with [`inference.py`](inference.py) already present).
3. Run **Section 10 — Gradio Chatbot Demo**. When the cell finishes building the UI, use the **public URL** or **Colab’s proxy link** Gradio prints (`Running on...` / `share=` if enabled). Chat in the UI; assistant turns should show raw `<tool_call>...</tool_call>` or plain text refusals depending on the prompt.
4. If the cell is slow to start, wait for FastAPI/Gradio startup; avoid interrupting the kernel mid-build (an interrupted run may show `KeyboardInterrupt` in stored outputs).

### Local / CLI — `make demo`

```bash
cd pocket-agent
make install
# Place artifacts/quantized_model/ (see layout above), then:
make demo
```

This launches a **Gradio** `ChatInterface` that calls `run(message, history)` from [`inference.py`](inference.py). Stop with **Ctrl+C** in the terminal.

### Quick smoke test (no Gradio)

```bash
cd pocket-agent
python inference.py
```

Runs the small `if __name__ == "__main__"` block at the bottom of [`inference.py`](inference.py) (a few fixed prompts).

### Public dev eval (optional)

```bash
cd pocket-agent
make eval
```

Loads [`inference.py`](inference.py) and scores against [`starter/public_test.jsonl`](starter/public_test.jsonl) via [`starter/eval_harness_contract.py`](starter/eval_harness_contract.py).

---

## Grading reference (private 20-example set)

| Slice | Count | Description |
|-------|-------|--------------|
| A | 8 | In-distribution |
| B | 5 | Paraphrased |
| C | 5 | Adversarial |
| D | 2 | Refusals & multi-turn |

Scoring: +1.0 / +0.5 / 0.0 / −0.5 as in the problem statement. **Hard gates** include: adapter compatible with declared base in `transformers` v5, quantized size limits, CPU latency, no network imports in `inference.py`, demo runs — verify against the official rubric before submit.

---

## Design decisions (summary)

- **Qwen2.5-0.5B-Instruct:** Under 2B, strong chat template, reasonable CPU inference after quantization.
- **QLoRA:** 4-bit training on T4; LoRA rank 16, alpha 32, attention projections.
- **Data:** Offline templates covering slices A–D; optional teacher backends (Groq/OpenAI/Anthropic/Ollama) — see notebook.
- **Inference:** `apply_chat_template` → `generate(**batch)` for Transformers v5 `BatchEncoding` compatibility.
- **`inference.py`:** Loads from `artifacts/quantized_model/` only; no `urllib`/`requests`/`socket`/`http`.

### Error analysis (bonus-oriented)

Issues observed during development: **ISO vs colloquial currency names** in code-switched prompts; **calendar `list` vs `create`** arg leakage; **default C vs F** for weather; **multi-turn** resolution when history is long; **SQL** string fidelity — mitigations described in earlier notebook iterations and training templates.

---

## License / attribution

Model: **Qwen2.5** via Hugging Face — follow the model card license. Starter schemas and harness mirror the hackathon contract.
