# Local RAG

<div align="center">

**Fully Local Retrieval-Augmented Generation for Private Document Q&A**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-local--RAG-lightgrey.svg)](https://github.com/karthiknagaraj/local-RAG)

[Quick Start](#quick-start) • [Documentation](#-documentation) • [Features](#-features) • [Architecture](#-architecture)

</div>

---

## 🚀 Quick Start

Get Local RAG running in 5 minutes. See [Getting Started Guide](docs/GETTING_STARTED.md) for detailed steps.

```bash
# 1. Clone repository
git clone https://github.com/karthiknagaraj/local-RAG.git
cd local-RAG

# 2. Setup environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Index your documents
python -m local_rag.src.ingest --config config.yaml

# 5. Launch web UI
python -m local_rag.src.web_ui --config config.yaml
```

Open browser to: **http://127.0.0.1:7860**

---

## ✨ Features

- 🔒 **100% Local** - No cloud dependencies, complete data privacy
- 📄 **Multi-Format** - PDF, DOCX, TXT, MD support
- 🎯 **Semantic Search** - FAISS vector database with cosine similarity
- 💬 **Web UI** - Gradio-based chat interface
- ⌨️ **CLI Tools** - Command-line indexing and querying
- 📝 **Source Attribution** - Every answer cites source documents
- 🚀 **Production Ready** - Deployment guides and configurations included

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [**Getting Started**](docs/GETTING_STARTED.md) | 5-minute setup guide (START HERE) |
| [**Full Documentation**](DOCUMENTATION.md) | Complete feature guide with diagrams |
| [**Architecture**](ARCHITECTURE.md) | System design and technical details |
| [**API Reference**](API_REFERENCE.md) | Code integration and API docs |
| [**Deployment**](DEPLOYMENT.md) | Production deployment strategies |
| [**Quick Reference**](QUICK_REFERENCE.md) | Command and config cheatsheet |
| [**Documentation Index**](INDEX.md) | Navigation guide for all docs |

---

## 🏗️ Architecture

```
Documents → Loader → Chunker → Embedder → FAISS Index
                                              ↓
Query → Embedding → Similarity Search → Context Building → LLM → Answer
```

**Key Components:**
- **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2, 384-dim)
- **Vector DB:** FAISS (IndexFlatIP for exact k-NN search)
- **LLM:** llama-cpp-python with Qwen 2.5 3B GGUF model
- **Web UI:** Gradio ChatInterface
- **Chunking:** Sentence-aware splitting with overlap

See [Architecture Documentation](ARCHITECTURE.md) for detailed diagrams.

---

## 🛠️ Usage

### Command Line

**One-shot query:**
```bash
python -m local_rag.src.query --config config.yaml --q "What is this about?"
```

**Interactive REPL:**
```bash
python -m local_rag.src.query --config config.yaml --repl
```

### Web Interface

```bash
# Local only
python -m local_rag.src.web_ui --config config.yaml

# With public sharing
python -m local_rag.src.web_ui --config config.yaml --share
```

### Python API

```python
from local_rag.src.rag import load_config, load_index_and_embedder
from local_rag.src.query import load_llm, answer_question

config = load_config("config.yaml")
retriever, embedder = load_index_and_embedder(config)
llm = load_llm(config)

answer, sources = answer_question(
    "Your question?", 
    retriever, embedder, llm, config
)
print(f"Answer: {answer}")
print(f"Sources: {sources}")
```

---

## 📋 Requirements

- **Python:** 3.9+
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 10GB free space (for models + index)
- **OS:** Windows, Linux, macOS

**Python Packages:**
- sentence-transformers (embeddings)
- faiss-cpu (vector search)
- llama-cpp-python (LLM inference)
- gradio (web UI)
- pypdf, python-docx (document parsing)
- pyyaml (configuration)

---

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
# Document collection
docs_dir: "/path/to/your/documents"
index_dir: "./index"

# Model and embedding
model_path: "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
embedding_model: "all-MiniLM-L6-v2"

# Retrieval parameters
top_k: 5              # Chunks to retrieve
chunk_size: 1200      # Characters per chunk
chunk_overlap: 200    # Overlap between chunks

# LLM parameters
llm:
  n_ctx: 4096         # Context window
  temperature: 0.2    # Response randomness
  max_tokens: 512     # Max response length
```

See [Configuration Guide](DOCUMENTATION.md#configuration) for all options.

---

## 🎯 Common Tasks

| Task | Command |
|------|---------|
| Index documents | `python -m local_rag.src.ingest --config config.yaml` |
| Ask question | `python -m local_rag.src.query --config config.yaml --q "..."` |
| Interactive CLI | `python -m local_rag.src.query --config config.yaml --repl` |
| Web UI | `python -m local_rag.src.web_ui --config config.yaml` |
| Public sharing | `python -m local_rag.src.web_ui --config config.yaml --share` |

For more commands, see [Quick Reference](QUICK_REFERENCE.md).

---

## 🚀 Deployment

**Development:** Single-user desktop setup  
**Team Server:** Multi-user with Nginx reverse proxy (5-20 users)  
**Production:** HA setup with load balancer and NFS (50+ users)

See [Deployment Guide](DEPLOYMENT.md) for detailed setup instructions.

---

## 🔧 Troubleshooting

**Can't find modules?**
```bash
pip install -r requirements.txt
```

**Index not found?**
```bash
python -m local_rag.src.ingest --config config.yaml
```

**Poor answer quality?**
- Increase `top_k` in config (5 → 7)
- Lower `temperature` (0.2 → 0.1)
- Check document relevance

For more help, see [Troubleshooting Guide](DOCUMENTATION.md#troubleshooting).

---

## 📊 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Embedding 1000 sentences | ~1 second | CPU-only |
| FAISS search (10k chunks) | ~30ms | Brute-force exact search |
| LLM generation (Qwen 3B) | ~8s | CPU-only per query |

**Recommended Specs:**
- **CPU:** 4+ cores (8+ for production)
- **RAM:** 16GB (32GB for HA)
- **Storage:** 50GB SSD

---

## 📖 Tips

- **Rebuild on changes:** Add new documents, then re-run: `python -m local_rag.src.ingest --config config.yaml`
- **Model compatibility:** Requires AVX2 CPU support (most modern CPUs have this)
- **GPU acceleration:** Add `n_gpu_layers: 35` in config for GPU inference (if CUDA installed)
- **Privacy first:** All processing happens locally—zero cloud dependencies
- **Expanding formats:** Can add HTML, PPTX, and other formats as needed

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Improved chunking strategies
- Alternative embedding models  
- Performance optimizations
- Additional deployment scenarios

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🔗 Resources

- [SentenceTransformers Documentation](https://www.sbert.net/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [Gradio Documentation](https://www.gradio.app/docs)
- [Qwen Models](https://huggingface.co/Qwen)

---

## 📧 Support

For issues, questions, or suggestions:
1. Check [Troubleshooting Guide](DOCUMENTATION.md#troubleshooting)
2. Review [Quick Reference](QUICK_REFERENCE.md)
3. Open an issue on [GitHub Issues](https://github.com/karthiknagaraj/local-RAG/issues)

---

**Made with ❤️ for local, private, offline AI**

