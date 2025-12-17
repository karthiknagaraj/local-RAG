# Getting Started with Local RAG

Quick start guide to get the Local RAG application running in 5 minutes.

## Prerequisites

- Python 3.9+ 
- 8GB RAM minimum
- 5GB free disk space

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/karthiknagaraj/local-RAG.git
cd local-RAG
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure
Edit `config.yaml`:
```yaml
docs_dir: "/path/to/your/documents"  # Your document folder
```

### 5. Build Index
```bash
python -m local_rag.src.ingest --config config.yaml
```

### 6. Launch Web UI
```bash
python -m local_rag.src.web_ui --config config.yaml
```

Open browser to: **http://127.0.0.1:7860**

## Quick Commands

| Task | Command |
|------|---------|
| Index documents | `python -m local_rag.src.ingest --config config.yaml` |
| Ask question (CLI) | `python -m local_rag.src.query --config config.yaml --q "Your question?"` |
| Interactive mode | `python -m local_rag.src.query --config config.yaml --repl` |
| Web UI | `python -m local_rag.src.web_ui --config config.yaml` |
| Web UI (public) | `python -m local_rag.src.web_ui --config config.yaml --share` |

## Documentation

- 📖 [Full Documentation](../DOCUMENTATION.md) - Complete feature guide
- 🏗️ [Architecture](../ARCHITECTURE.md) - System design and diagrams
- 🔧 [API Reference](../API_REFERENCE.md) - Code integration guide
- 🚢 [Deployment](../DEPLOYMENT.md) - Production setup
- ⚡ [Quick Reference](../QUICK_REFERENCE.md) - Command cheatsheet
- 📚 [Index](../INDEX.md) - Documentation navigation

## Troubleshooting

**"Module not found"**
```bash
pip install -r requirements.txt
```

**"Index not found"**
```bash
python -m local_rag.src.ingest --config config.yaml
```

**"Model not found"**
Models download automatically on first use. Or download manually:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

See [Troubleshooting Guide](../DOCUMENTATION.md#troubleshooting) for more help.

## Features

✅ **100% Local** - No cloud dependencies, all data stays on your machine  
✅ **Multi-format** - Support for PDF, DOCX, TXT, MD files  
✅ **Web UI** - Gradio-based chat interface  
✅ **CLI Tools** - Command-line indexing and querying  
✅ **Semantic Search** - FAISS-based vector similarity  
✅ **Source Attribution** - Every answer cites its sources  
✅ **Production Ready** - Deployment guides included  

## Models

**Embeddings:** all-MiniLM-L6-v2 (384-dim, 80MB)  
**LLM:** Qwen 2.5 3B Instruct GGUF (2.1GB, Q4_K_M quantization)

## Next Steps

1. Add your documents to the folder specified in `config.yaml`
2. Rebuild the index: `python -m local_rag.src.ingest --config config.yaml`
3. Start asking questions!

For detailed configuration options, see [Configuration Guide](../DOCUMENTATION.md#configuration).
