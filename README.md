# Local RAG (Retrieve-Then-Answer)

A fully local RAG pipeline to answer questions from your documents (20–30 files or more). It:
- Loads docs from a folder (txt, md, pdf, docx)
- Chunks and embeds with `sentence-transformers`
- Indexes in FAISS
- Answers with a local `llama.cpp`-compatible GGUF model

## Prereqs (Windows)
- Python 3.10 or newer
- Disk space for a small GGUF model (~1–2 GB is fine)

## Setup
```bash
# 1) Create and activate a venv (recommended)
python -m venv .venv
.\.venv\Scripts\activate

# 2) Install deps
pip install -r local_rag/requirements.txt

# 3) Put a local GGUF model at the configured path
# Default (edit in local_rag/config.yaml):
#   local_rag/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
# You can use any GGUF Instruct model. Examples:
# - Qwen2.5-1.5B-Instruct-Q4_K_M.gguf (small, decent)
# - Phi-3.5-mini-instruct-Q4_K_M.gguf (if available)
# - Mistral-7B-Instruct-Q4_K_M.gguf (larger, better)

# 4) Ensure docs are in your KB folder
# Default: c:/Users/karthik.nagaraj/Documents/GH-EDW Knowledge Base
# Update paths in local_rag/config.yaml as needed
```

## Build the index
```bash
python -m local_rag.src.ingest --config local_rag/config.yaml
```
This creates FAISS and metadata in `local-rag/index`.

## Ask questions (CLI)
```bash
python -m local_rag.src.query --config local_rag/config.yaml --q "How does X work?"
```
For an interactive REPL:
```bash
python -m local_rag.src.query --config local_rag/config.yaml --repl
```

## Tips
- If you change docs, re-run the ingest step.
- If `llama-cpp-python` can’t load your model, check `model_path` and that your CPU has AVX2 (most modern CPUs do). Optionally set `n_gpu_layers > 0` if you installed GPU support.
- To expand formats (HTML, PPTX), we can add lightweight readers later.
