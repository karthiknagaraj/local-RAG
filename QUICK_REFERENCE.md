# Local RAG Project - Quick Reference

## Project Overview

**Purpose:** Fully local RAG (Retrieval-Augmented Generation) application for private document Q&A without cloud dependencies.

**Tech Stack:** Python, SentenceTransformers, FAISS, llama-cpp-python, Gradio

---

## Project Structure

```
local_rag/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── rag.py                      # Core RAG engine (embeddings, FAISS, chunking)
│   ├── ingest.py                   # CLI index builder
│   ├── query.py                    # CLI Q&A interface
│   └── web_ui.py                   # Gradio web interface
├── models/                         # GGUF model storage
│   └── qwen2.5-3b-instruct-q4_k_m.gguf
├── index/                          # FAISS index files
│   ├── index.faiss                 # Binary FAISS index
│   ├── metadata.jsonl              # Chunk metadata
│   └── embedding_model.txt         # Model name
├── config.yaml                     # Central configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Setup and usage guide
├── DOCUMENTATION.md                # Comprehensive documentation
├── ARCHITECTURE.md                 # System architecture details
├── API_REFERENCE.md                # Complete API documentation
├── DEPLOYMENT.md                   # Production deployment guide
└── QUICK_REFERENCE.md              # This file
```

---

## Quick Commands

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Indexing
```bash
# Build index from documents
python -m local_rag.src.ingest --config local_rag/config.yaml
```

### Query (CLI)
```bash
# One-shot query
python -m local_rag.src.query --config local_rag/config.yaml --q "Your question?"

# Interactive REPL
python -m local_rag.src.query --config local_rag/config.yaml --repl
```

### Web UI
```bash
# Local only
python -m local_rag.src.web_ui --config local_rag/config.yaml

# Public sharing
python -m local_rag.src.web_ui --config local_rag/config.yaml --share

# Custom port
python -m local_rag.src.web_ui --config local_rag/config.yaml --port 8080
```

---

## Configuration Cheat Sheet

### config.yaml Key Settings

| Setting | Purpose | Common Values |
|---------|---------|---------------|
| `docs_dir` | Document collection path | `/path/to/your/docs` |
| `index_dir` | FAISS index storage | `./local_rag/index` |
| `model_path` | GGUF model file | `./models/qwen.gguf` |
| `embedding_model` | SentenceTransformer model | `all-MiniLM-L6-v2` |
| `top_k` | Chunks to retrieve | 3-7 |
| `chunk_size` | Chars per chunk | 800-1500 |
| `chunk_overlap` | Overlap chars | 100-300 |
| `llm.n_ctx` | Context window | 2048-4096 |
| `llm.temperature` | Response randomness | 0.1-0.3 (factual) |

---

## Troubleshooting Quick Fixes

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Index not found"
```bash
python -m local_rag.src.ingest --config config.yaml
```

### "Model not found"
```bash
# Download Qwen 2.5 3B
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='local_rag/models')"
```

### Slow responses
- Reduce `n_ctx` (4096 → 2048)
- Reduce `max_tokens` (512 → 256)
- Use smaller model (TinyLlama instead of Qwen)

### Poor answer quality
- Increase `top_k` (5 → 7)
- Decrease `temperature` (0.2 → 0.1)
- Use larger model (Qwen 3B instead of TinyLlama)

### Spelling errors in responses
- **Root cause:** Model too small (TinyLlama 1.1B)
- **Solution:** Upgrade to Qwen 2.5 3B or Phi-3.5-mini

---

## Code Architecture (Simplified)

### Data Flow: Indexing
```
Documents → read_file_auto() → chunk_text() → Embedder.encode() → FaissStore.add() → Save to disk
```

### Data Flow: Query
```
Question → Embedder.encode() → FaissStore.search() → make_context_from_hits() → LLM.generate() → Answer + Sources
```

---

## Key Classes & Functions

### rag.py
- `Embedder`: Wrapper for SentenceTransformer
- `FaissStore`: FAISS vector database wrapper
- `build_index()`: Build index from documents
- `load_index_and_embedder()`: Load existing index
- `chunk_text()`: Sentence-aware text chunking

### query.py
- `load_llm()`: Load GGUF model via llama-cpp-python
- `answer_question()`: Main Q&A pipeline
- `repl()`: Interactive command-line interface

### web_ui.py
- `create_ui()`: Create Gradio chat interface
- `answer()`: Gradio callback for questions

---

## Environment Variables

```bash
# Optional: Set offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Optional: Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/local_rag"
```

---

## Performance Tuning

### For Speed
```yaml
chunk_size: 800          # Smaller chunks
top_k: 3                 # Fewer chunks
llm:
  n_ctx: 2048            # Smaller context
  max_tokens: 256        # Shorter responses
  temperature: 0.1       # Deterministic
```

### For Quality
```yaml
chunk_size: 1500         # Larger chunks
top_k: 7                 # More context
llm:
  n_ctx: 4096            # Larger context
  max_tokens: 512        # Longer responses
  temperature: 0.3       # More creative
```

---

## Recommended Models

### Embedding Models
| Model | Dim | Size | Use Case |
|-------|-----|------|----------|
| all-MiniLM-L6-v2 | 384 | 80MB | Default, fast |
| all-mpnet-base-v2 | 768 | 420MB | Better quality |
| paraphrase-multilingual-* | 384 | 220MB | Multilingual |

### LLM Models (GGUF, Q4_K_M quantization)
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| TinyLlama 1.1B | 669MB | Fast | Low (spelling errors) |
| Qwen 2.5 3B | 2.1GB | Medium | High ✅ Recommended |
| Phi-3.5-mini 3.8B | 2.2GB | Medium | High |
| Llama 3.2 3B | 2.0GB | Medium | High |

---

## File Formats Supported

- `.txt` - Plain text
- `.md` - Markdown
- `.pdf` - PDF documents (via pypdf)
- `.docx` - Microsoft Word (via python-docx)

---

## Common Workflows

### 1. Initial Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Download models
python -m local_rag.src.ingest --config config.yaml
```

### 2. Daily Usage
```bash
# Web UI (recommended)
python -m local_rag.src.web_ui --config config.yaml
# → Open browser to http://127.0.0.1:7860
```

### 3. Adding New Documents
```bash
# 1. Copy docs to docs_dir folder
# 2. Rebuild index
python -m local_rag.src.ingest --config config.yaml
```

### 4. Model Upgrade
```bash
# 1. Download new model to models/
# 2. Update config.yaml model_path
# 3. Restart application
```

---

## Security Best Practices

- ✅ Use absolute paths in config.yaml
- ✅ Set file permissions: `chmod 600 config.yaml`
- ✅ Run on localhost only (no `--share` flag)
- ✅ Use firewall rules to restrict access
- ✅ Enable HTTPS via Nginx reverse proxy
- ✅ Implement authentication (Nginx BasicAuth or Gradio auth)
- ❌ Never commit models to git (use .gitignore)
- ❌ Never expose port 7860 to public internet

---

## Monitoring Metrics

| Metric | Target | Alert |
|--------|--------|-------|
| Query latency (p95) | <10s | >15s |
| Index size | <1M chunks | >1M |
| Memory usage | <8GB | >12GB |
| Error rate | <1% | >5% |

---

## Update Checklist

### Code Updates
- [ ] `git pull` or extract new version
- [ ] `pip install -r requirements.txt --upgrade`
- [ ] Review config.yaml for new settings
- [ ] Restart service

### Model Updates
- [ ] Download new model
- [ ] Update config.yaml `model_path`
- [ ] Test with sample query
- [ ] Restart service

### Index Updates
- [ ] Backup existing index: `tar -czf index_backup.tar.gz index/`
- [ ] Run ingest: `python -m local_rag.src.ingest --config config.yaml`
- [ ] Verify index size: `ls -lh index/`

---

## Useful Links

- **Documentation:** [DOCUMENTATION.md](DOCUMENTATION.md)
- **API Reference:** [API_REFERENCE.md](API_REFERENCE.md)
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Deployment:** [DEPLOYMENT.md](DEPLOYMENT.md)
- **SentenceTransformers:** https://www.sbert.net/
- **FAISS:** https://github.com/facebookresearch/faiss
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **Gradio:** https://www.gradio.app/

---

## Support

### Logs Location
- Application logs: `logs/local_rag.log`
- System logs: `sudo journalctl -u local-rag`

### Debug Mode
```python
# Add to top of script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling
```bash
python -m cProfile -o profile.stats -m local_rag.src.query --config config.yaml --q "test"
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime'); p.print_stats(20)"
```

---

**Last Updated:** December 16, 2025
**Version:** 1.0.0
