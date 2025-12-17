# Local RAG - API Reference

Complete API documentation for all modules, classes, and functions.

---

## Module: rag.py

Core RAG engine providing embeddings, vector storage, and document processing.

---

### Function: load_config

```python
def load_config(config_path: str) -> dict
```

Load configuration from YAML file.

**Parameters:**
- `config_path` (str): Absolute path to config.yaml file

**Returns:**
- dict: Configuration dictionary with keys:
  - `docs_dir`: Document collection directory
  - `index_dir`: FAISS index storage directory
  - `model_path`: Path to GGUF model file
  - `embedding_model`: SentenceTransformer model name
  - `top_k`: Number of chunks to retrieve
  - `chunk_size`: Max characters per chunk
  - `chunk_overlap`: Overlap between chunks
  - `llm`: LLM configuration dict

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is malformed

**Example:**
```python
config = load_config("local_rag/config.yaml")
print(config["top_k"])  # 5
```

---

### Function: read_file_auto

```python
def read_file_auto(filepath: str) -> str
```

Automatically detect file type and extract text content.

**Parameters:**
- `filepath` (str): Absolute path to document file

**Returns:**
- str: Extracted text content (empty string if parsing fails)

**Supported Formats:**
- `.txt`, `.md`: UTF-8 text files
- `.pdf`: PDF documents (via pypdf)
- `.docx`: Microsoft Word documents (via python-docx)

**Error Handling:**
- Returns empty string on parse failures (logged as warning)
- Tries utf-8 encoding first, falls back to latin-1

**Example:**
```python
text = read_file_auto("docs/manual.pdf")
print(f"Extracted {len(text)} characters")
```

---

### Function: split_sentences

```python
def split_sentences(text: str) -> List[str]
```

Split text into sentences using regex pattern.

**Parameters:**
- `text` (str): Input text to split

**Returns:**
- List[str]: List of sentences (preserves whitespace)

**Algorithm:**
- Splits on: `.`, `!`, `?` followed by whitespace
- Regex pattern: `r'(?<=[.!?])\s+'`

**Example:**
```python
text = "Hello world. How are you? I'm fine!"
sentences = split_sentences(text)
# ["Hello world.", "How are you?", "I'm fine!"]
```

---

### Function: chunk_text

```python
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]
```

Split text into overlapping chunks respecting sentence boundaries.

**Parameters:**
- `text` (str): Input text to chunk
- `chunk_size` (int, default=1200): Max characters per chunk
- `overlap` (int, default=200): Characters shared between consecutive chunks

**Returns:**
- List[str]: List of text chunks

**Algorithm:**
1. Split into sentences via `split_sentences()`
2. Accumulate sentences until chunk_size reached
3. Add overlap by including last `overlap` characters from previous chunk
4. Ensure sentence boundaries preserved (no mid-sentence splits)

**Edge Cases:**
- Very long sentences (>chunk_size): Split on word boundaries
- Very short text (<chunk_size): Returns single chunk
- Empty text: Returns empty list

**Example:**
```python
text = "Sentence one. Sentence two. Sentence three."
chunks = chunk_text(text, chunk_size=30, overlap=10)
# ["Sentence one. Sentence two.", "Sentence two. Sentence three."]
```

---

### Class: Embedder

Wrapper around SentenceTransformer for text embeddings.

#### Constructor

```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2")
```

**Parameters:**
- `model_name` (str): HuggingFace model identifier

**Common Models:**
- `all-MiniLM-L6-v2`: 384-dim, fast, good quality (default)
- `all-mpnet-base-v2`: 768-dim, slower, better quality
- `paraphrase-multilingual-MiniLM-L12-v2`: 384-dim, multilingual

**Example:**
```python
embedder = Embedder("all-MiniLM-L6-v2")
```

---

#### Method: encode

```python
def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray
```

Convert text(s) to dense vector embeddings.

**Parameters:**
- `texts` (str | List[str]): Single text or list of texts
- `batch_size` (int, default=32): Batch size for encoding

**Returns:**
- np.ndarray: Embeddings array of shape:
  - (embedding_dim,) if input is single string
  - (num_texts, embedding_dim) if input is list

**Performance:**
- Batching improves throughput (~1000 sentences/sec on CPU)
- Automatically uses GPU if available

**Example:**
```python
# Single text
vec = embedder.encode("Hello world")  # shape: (384,)

# Multiple texts
vecs = embedder.encode(["text1", "text2", "text3"])  # shape: (3, 384)
```

---

#### Property: dim

```python
@property
def dim(self) -> int
```

Get embedding dimension.

**Returns:**
- int: Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)

**Example:**
```python
print(embedder.dim)  # 384
```

---

### Class: FaissStore

FAISS vector database wrapper with persistence.

#### Constructor

```python
def __init__(self, dimension: int)
```

Initialize FAISS index with IndexFlatIP (inner product for cosine similarity).

**Parameters:**
- `dimension` (int): Embedding dimension (must match embedder.dim)

**Example:**
```python
store = FaissStore(dimension=384)
```

---

#### Method: add

```python
def add(self, embeddings: np.ndarray, metadatas: List[dict]) -> None
```

Add embeddings and metadata to index.

**Parameters:**
- `embeddings` (np.ndarray): Shape (num_vectors, dimension)
- `metadatas` (List[dict]): One metadata dict per embedding

**Metadata Schema:**
```python
{
    "text": "chunk text content",
    "source": "document.pdf",
    "chunk_id": 0  # 0-indexed chunk number within source
}
```

**Example:**
```python
embeddings = np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]])  # (2, 384)
metadatas = [
    {"text": "chunk 1", "source": "doc.pdf", "chunk_id": 0},
    {"text": "chunk 2", "source": "doc.pdf", "chunk_id": 1}
]
store.add(embeddings, metadatas)
```

---

#### Method: search

```python
def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[dict]]
```

Search for k-nearest neighbors.

**Parameters:**
- `query_embedding` (np.ndarray): 1D array of shape (dimension,)
- `k` (int, default=5): Number of results to return

**Returns:**
- Tuple[List[float], List[dict]]:
  - scores: Similarity scores (higher = more similar, range: [-1, 1])
  - metadatas: List of metadata dicts corresponding to top-k results

**Algorithm:**
- Computes cosine similarity via dot product (assumes L2-normalized vectors)
- Returns exact k-NN (brute-force search, O(n))

**Example:**
```python
query_vec = embedder.encode("What is Grand Maison?")
scores, metadatas = store.search(query_vec, k=5)

for score, meta in zip(scores, metadatas):
    print(f"Score: {score:.3f}, Source: {meta['source']}")
```

---

#### Method: save

```python
def save(self, index_path: str, metadata_path: str) -> None
```

Persist index and metadata to disk.

**Parameters:**
- `index_path` (str): Path to save FAISS index (e.g., "index/index.faiss")
- `metadata_path` (str): Path to save metadata JSONL (e.g., "index/metadata.jsonl")

**File Formats:**
- `index.faiss`: Binary FAISS index
- `metadata.jsonl`: One JSON object per line (newline-delimited JSON)

**Example:**
```python
store.save("index/index.faiss", "index/metadata.jsonl")
```

---

#### Method: load

```python
def load(self, index_path: str, metadata_path: str) -> None
```

Load index and metadata from disk.

**Parameters:**
- `index_path` (str): Path to FAISS index file
- `metadata_path` (str): Path to metadata JSONL file

**Raises:**
- `FileNotFoundError`: If index or metadata file doesn't exist

**Example:**
```python
store = FaissStore(dimension=384)
store.load("index/index.faiss", "index/metadata.jsonl")
```

---

### Function: build_index

```python
def build_index(config: dict) -> Tuple[FaissStore, Embedder]
```

Build FAISS index from document collection.

**Parameters:**
- `config` (dict): Configuration dictionary from `load_config()`

**Returns:**
- Tuple[FaissStore, Embedder]: Initialized and populated store + embedder

**Process:**
1. Initialize Embedder with `config["embedding_model"]`
2. Recursively scan `config["docs_dir"]` for supported files
3. Load and chunk each file via `read_file_auto()` and `chunk_text()`
4. Batch-encode chunks via `embedder.encode()`
5. Add embeddings to FaissStore
6. Save index to `config["index_dir"]`

**Output Files:**
- `{index_dir}/index.faiss`: FAISS index
- `{index_dir}/metadata.jsonl`: Chunk metadata
- `{index_dir}/embedding_model.txt`: Model name (for consistency check)

**Example:**
```python
config = load_config("config.yaml")
store, embedder = build_index(config)
print(f"Indexed {store.ntotal} chunks")
```

---

### Function: load_index_and_embedder

```python
def load_index_and_embedder(config: dict) -> Tuple[FaissStore, Embedder]
```

Load existing FAISS index and embedder.

**Parameters:**
- `config` (dict): Configuration dictionary

**Returns:**
- Tuple[FaissStore, Embedder]: Loaded store + embedder

**Validation:**
- Checks if `embedding_model.txt` matches `config["embedding_model"]`
- Raises `ValueError` if mismatch detected (prevents dimension mismatch)

**Example:**
```python
config = load_config("config.yaml")
store, embedder = load_index_and_embedder(config)
```

---

### Function: make_context_from_hits

```python
def make_context_from_hits(hits: List[dict], max_chars: int = 6000) -> str
```

Build context string from retrieved chunks.

**Parameters:**
- `hits` (List[dict]): Metadata dicts from FAISS search
- `max_chars` (int, default=6000): Max context length

**Returns:**
- str: Formatted context string

**Format:**
```
[Source: document1.pdf, Chunk 0]
chunk text content here...

[Source: document2.txt, Chunk 3]
more chunk text...
```

**Truncation:**
- If total length exceeds `max_chars`, truncates to fit
- Preserves full chunks (doesn't split mid-chunk)

**Example:**
```python
scores, hits = store.search(query_vec, k=5)
context = make_context_from_hits(hits, max_chars=4000)
print(context)
```

---

## Module: query.py

Command-line Q&A interface and LLM inference.

---

### Function: load_llm

```python
def load_llm(config: dict) -> Llama
```

Load GGUF model via llama-cpp-python.

**Parameters:**
- `config` (dict): Configuration dictionary

**Returns:**
- Llama: Initialized llama_cpp.Llama instance

**Configuration Keys:**
- `config["model_path"]`: Path to .gguf file
- `config["llm"]["n_ctx"]`: Context window size
- `config["llm"]["n_threads"]`: CPU threads (null = auto-detect)
- `config["llm"]["n_gpu_layers"]`: GPU offload layers (0 = CPU-only)

**Example:**
```python
config = load_config("config.yaml")
llm = load_llm(config)
```

---

### Function: answer_question

```python
def answer_question(
    question: str,
    retriever: FaissStore,
    embedder: Embedder,
    llm: Llama,
    config: dict
) -> Tuple[str, str]
```

Answer question using RAG pipeline.

**Parameters:**
- `question` (str): User query
- `retriever` (FaissStore): FAISS index
- `embedder` (Embedder): Text embedder
- `llm` (Llama): LLM instance
- `config` (dict): Configuration

**Returns:**
- Tuple[str, str]:
  - answer: Generated response text
  - sources: Source attribution string (e.g., "doc.pdf (3 chunks), other.txt (2 chunks)")

**Pipeline:**
1. Embed query via `embedder.encode()`
2. Search top-k chunks via `retriever.search()`
3. Build context via `make_context_from_hits()`
4. Format prompt with SYSTEM_PROMPT and PROMPT_TEMPLATE
5. Generate answer via `llm.create_chat_completion()`
6. Extract source filenames and chunk counts

**Example:**
```python
answer, sources = answer_question(
    "What is Grand Maison?",
    retriever, embedder, llm, config
)
print(f"Answer: {answer}")
print(f"Sources: {sources}")
```

---

### Function: repl

```python
def repl(config: dict) -> None
```

Interactive REPL for Q&A.

**Parameters:**
- `config` (dict): Configuration dictionary

**Usage:**
```
> What is Grand Maison?
[Answer with sources]

> How does it work?
[Answer with sources]

> exit
```

**Commands:**
- `exit`, `quit`: Exit REPL
- Any other text: Treated as question

**Example:**
```python
config = load_config("config.yaml")
repl(config)
```

---

## Module: ingest.py

Command-line index builder.

---

### Function: main

```python
def main() -> None
```

CLI entry point for indexing.

**Command-line Arguments:**
```bash
python -m local_rag.src.ingest --config <path>
```

**Arguments:**
- `--config`: Path to config.yaml (required)

**Output:**
```
Building index from: /path/to/docs
Found 25 files
Processed 342 chunks from 25 files
Index saved to: /path/to/index
```

**Example:**
```bash
python -m local_rag.src.ingest --config local_rag/config.yaml
```

---

## Module: web_ui.py

Gradio web interface.

---

### Function: create_ui

```python
def create_ui(config: dict) -> gr.ChatInterface
```

Create Gradio chat interface.

**Parameters:**
- `config` (dict): Configuration dictionary

**Returns:**
- gr.ChatInterface: Gradio interface object

**Interface Features:**
- Chat history persistence
- Example questions
- Markdown rendering for sources
- Mobile-responsive layout

**Example:**
```python
config = load_config("config.yaml")
ui = create_ui(config)
ui.launch(share=False, server_port=7860)
```

---

### Function: answer (internal)

```python
def answer(message: str, history: List) -> str
```

Gradio callback for answering questions.

**Parameters:**
- `message` (str): User question
- `history` (List): Chat history (unused in current implementation)

**Returns:**
- str: Response with markdown-formatted sources

**Format:**
```markdown
{answer text}

---
**Sources:** document.pdf (3 chunks), other.txt (2 chunks)
```

---

### Function: main

```python
def main() -> None
```

CLI entry point for web UI.

**Command-line Arguments:**
```bash
python -m local_rag.src.web_ui --config <path> [--share] [--port <port>]
```

**Arguments:**
- `--config`: Path to config.yaml (required)
- `--share`: Create public Gradio link (optional)
- `--port`: Server port (default: 7860)

**Example:**
```bash
# Local only
python -m local_rag.src.web_ui --config config.yaml

# Public sharing
python -m local_rag.src.web_ui --config config.yaml --share

# Custom port
python -m local_rag.src.web_ui --config config.yaml --port 8080
```

---

## Configuration Schema

### config.yaml

```yaml
# Required fields
docs_dir: str                    # Document collection directory
index_dir: str                   # FAISS index storage directory
model_path: str                  # Path to GGUF model file
embedding_model: str             # SentenceTransformer model name

# Retrieval parameters
top_k: int                       # Number of chunks to retrieve (default: 5)
chunk_size: int                  # Max characters per chunk (default: 1200)
chunk_overlap: int               # Overlap between chunks (default: 200)

# LLM parameters
llm:
  n_ctx: int                     # Context window size (default: 4096)
  n_threads: int | null          # CPU threads (null = auto-detect)
  n_gpu_layers: int              # GPU offload layers (0 = CPU-only)
  temperature: float             # Sampling temperature (default: 0.2)
  max_tokens: int                # Max response length (default: 512)
```

---

## Error Codes

### Common Exceptions

| Exception | Cause | Solution |
|-----------|-------|----------|
| `FileNotFoundError` | config.yaml or model file missing | Check file paths |
| `ValueError` | Embedding model mismatch | Rebuild index or update config |
| `RuntimeError` | FAISS dimension mismatch | Ensure embedder.dim matches store dimension |
| `IndexError` | Empty FAISS index | Run ingest.py first |
| `llama_cpp.LlamaError` | Model loading failure | Check model_path and disk space |

---

## Type Hints

### Common Types

```python
from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
from llama_cpp import Llama

# Configuration
Config = Dict[str, Any]

# Metadata
Metadata = Dict[str, Union[str, int]]
# Example: {"text": "...", "source": "doc.pdf", "chunk_id": 0}

# Search results
SearchResult = Tuple[List[float], List[Metadata]]
# Example: ([0.92, 0.87, ...], [{"text": "...", ...}, ...])

# Answer result
AnswerResult = Tuple[str, str]
# Example: ("Grand Maison is a CRM system...", "doc.pdf (3 chunks)")
```

---

## Constants

### Module: query.py

```python
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly using the provided context.
If the answer is not in the context, say "I don't have enough information to answer this question."
Always cite the source documents in your answer."""

PROMPT_TEMPLATE = """[INST] <<SYS>>
{system_prompt}
<</SYS>>

Context:
{context}

Question: {question} [/INST]"""
```

### Module: rag.py

```python
SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".docx"]
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARS = 6000
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `embedder.encode(texts)` | O(n·m) | n=num_texts, m=avg_length |
| `store.add(embeddings)` | O(n·d) | n=num_vectors, d=dimension |
| `store.search(query, k)` | O(n·d) | Brute-force search |
| `chunk_text(text)` | O(n) | n=text_length |
| `read_file_auto(pdf)` | O(p) | p=num_pages |

### Space Complexity

| Structure | Memory | Notes |
|-----------|--------|-------|
| FAISS IndexFlatIP | O(n·d·4 bytes) | n=num_vectors, d=dimension |
| Metadata | O(n·m) | m=avg_metadata_size |
| Embedder model | ~80MB | all-MiniLM-L6-v2 |
| LLM model (Q4_K_M) | ~2GB | Qwen 2.5 3B |

---

**Last Updated:** December 16, 2025
**Version:** 1.0.0
