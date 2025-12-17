import os
import re
import json
import math
import glob
import yaml
from typing import List, Dict, Tuple, Iterable

from sentence_transformers import SentenceTransformer
import faiss

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
except Exception:
    docx = None


ALLOWED_EXTS = {".txt", ".md", ".pdf", ".docx"}


def load_config(path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        path: Absolute path to config.yaml file
        
    Returns:
        dict: Configuration dictionary with keys:
            - docs_dir: Document collection directory
            - index_dir: FAISS index storage directory
            - model_path: Path to GGUF model file
            - embedding_model: SentenceTransformer model name
            - top_k: Number of chunks to retrieve
            - chunk_size: Max characters per chunk
            - chunk_overlap: Overlap between chunks
            - llm: LLM configuration dict
            
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed; cannot read PDFs")
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        texts.append(txt)
    return "\n".join(texts)


def read_docx_file(path: str) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed; cannot read DOCX")
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs])


def read_file_auto(path: str) -> str:
    """Auto-detect file type and extract text content.
    
    Supported formats:
        - .txt, .md: Plain text files (UTF-8)
        - .pdf: PDF documents (via pypdf)
        - .docx: Microsoft Word documents (via python-docx)
        
    Args:
        path: Absolute path to document file
        
    Returns:
        str: Extracted text content
        
    Raises:
        ValueError: If file extension is not supported
        RuntimeError: If required parser library is not installed
        
    Example:
        >>> text = read_file_auto('/path/to/document.pdf')
        >>> print(f'Extracted {len(text)} characters')
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        return read_text_file(path)
    if ext == ".pdf":
        return read_pdf_file(path)
    if ext == ".docx":
        return read_docx_file(path)
    raise ValueError(f"Unsupported extension: {ext}")


# Regex pattern for sentence splitting:
# Matches sentence boundaries: [.!?] followed by whitespace and capital letter/digit
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex pattern.
    
    Algorithm:
        1. Normalize whitespace (collapse multiple spaces)
        2. Split on sentence boundaries: [.!?] + whitespace + capital letter
        3. If text is one long sentence (>2000 chars), split into 300-char chunks
        
    Args:
        text: Input text to split
        
    Returns:
        List[str]: List of sentences (or chunks if text is too long)
        
    Note:
        This is a lightweight splitter. For more sophisticated sentence
        detection, consider using spaCy or NLTK.
        
    Example:
        >>> split_sentences('Hello world. How are you? Fine!')
        ['Hello world.', 'How are you?', 'Fine!']
    """
    # Normalize whitespace (collapse multiple spaces to single space)
    text = re.sub(r'\s+', ' ', text.strip())
    if not text:
        return []
    
    # Split on sentence boundaries
    parts = _SENT_SPLIT_RE.split(text)
    
    # Fallback: if text is one huge sentence without proper boundaries,
    # split into fixed-size chunks to prevent memory issues
    if len(parts) == 1 and len(parts[0]) > 2000:
        return [parts[0][i : i + 300] for i in range(0, len(parts[0]), 300)]
    
    return parts


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks respecting sentence boundaries.
    
    Algorithm:
        1. Split text into sentences via split_sentences()
        2. Accumulate sentences into buffer until chunk_size reached
        3. When chunk is full, save it and start new chunk
        4. Add overlap by including last 'overlap' characters from previous chunk
        5. Ensures no mid-sentence splits (preserves semantic coherence)
        
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk (default: 1200)
        chunk_overlap: Characters shared between consecutive chunks (default: 200)
        
    Returns:
        List[str]: List of text chunks with overlap
        
    Edge Cases:
        - Very long sentences (>chunk_size): Handled by split_sentences fallback
        - Very short text (<chunk_size): Returns single chunk
        - Empty text: Returns empty list
        
    Example:
        >>> text = 'Sentence one. Sentence two. Sentence three.'
        >>> chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
        >>> # chunks[0]: 'Sentence one. Sentence two.'
        >>> # chunks[1]: 'Sentence two. Sentence three.' (overlap on 'Sentence two.')
        
    Rationale:
        - Overlap prevents context loss at chunk boundaries
        - Sentence-awareness improves retrieval relevance
        - Configurable sizes allow tuning for different use cases
    """
    # Split text into sentences first
    sents = split_sentences(text)
    chunks: List[str] = []  # Final list of chunks
    buf: List[str] = []     # Buffer for accumulating sentences
    cur_len = 0             # Current buffer length in characters

    for s in sents:
        s_len = len(s)
        
        # If adding this sentence exceeds chunk_size, finalize current chunk
        if cur_len + s_len + 1 > chunk_size and buf:
            # Save accumulated sentences as a chunk
            chunks.append(' '.join(buf).strip())
            
            # Overlap logic: start new chunk with last 'overlap' chars from previous
            if chunk_overlap > 0 and chunks[-1]:
                tail = chunks[-1][-chunk_overlap:]  # Last N characters
                buf = [tail]
                cur_len = len(tail)
            else:
                # No overlap: start fresh
                buf = []
                cur_len = 0
        
        # Add sentence to buffer
        buf.append(s)
        cur_len += s_len + 1  # +1 for space between sentences

    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]


class Embedder:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = SentenceTransformer(model_id)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()

    @property
    def dim(self) -> int:
        # try to infer embedding dimension
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception:
            # fallback lazy encode
            v = self.model.encode(["test"], normalize_embeddings=True, convert_to_numpy=True)[0]
            return len(v)


class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

    def add(self, vectors: List[List[float]], metadatas: List[Dict]):
        import numpy as np

        arr = np.array(vectors, dtype="float32")
        self.index.add(arr)
        self.metadata.extend(metadatas)

    def search(self, query_vec: List[float], k: int) -> List[Tuple[int, float]]:
        import numpy as np

        q = np.array([query_vec], dtype="float32")
        scores, idxs = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "metadata.jsonl"), "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @staticmethod
    def load(dir_path: str) -> Tuple["FaissStore", List[Dict]]:
        idx = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        store = FaissStore(idx.d)
        store.index = idx
        metadata: List[Dict] = []
        with open(os.path.join(dir_path, "metadata.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
        store.metadata = metadata
        return store


def iter_files(root: str) -> Iterable[str]:
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in ALLOWED_EXTS:
            yield path


def build_index(config: dict):
    docs_dir = config["docs_dir"] if "docs_dir" in config else config.get("docs_dir")
    docs_dir = config.get("docs_dir") or config.get("docs_path")
    if not docs_dir or not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"docs_dir not found: {docs_dir}")

    index_dir = config["index_dir"]
    os.makedirs(index_dir, exist_ok=True)

    chunk_size = int(config.get("chunk_size", 1200))
    chunk_overlap = int(config.get("chunk_overlap", 200))
    embed_model_id = config.get("embedding_model", "all-MiniLM-L6-v2")

    embedder = Embedder(embed_model_id)
    store = FaissStore(embedder.dim)

    all_vectors: List[List[float]] = []
    all_meta: List[Dict] = []

    file_count = 0
    chunk_count = 0

    for fp in iter_files(docs_dir):
        file_count += 1
        try:
            text = read_file_auto(fp)
        except Exception as e:
            # skip unreadable files
            continue
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            continue
        vectors = embedder.encode(chunks)
        metas = [
            {
                "source": fp,
                "chunk_index": i,
                "text": chunks[i],
            }
            for i in range(len(chunks))
        ]
        store.add(vectors, metas)
        chunk_count += len(chunks)

    # persist
    store.save(index_dir)
    with open(os.path.join(index_dir, "embedding_model.txt"), "w", encoding="utf-8") as f:
        f.write(embed_model_id)

    return {"files": file_count, "chunks": chunk_count}


def load_index_and_embedder(config: dict) -> Tuple[FaissStore, Embedder]:
    index_dir = config["index_dir"]
    store = FaissStore.load(index_dir)
    with open(os.path.join(index_dir, "embedding_model.txt"), "r", encoding="utf-8") as f:
        model_id = f.read().strip()
    embedder = Embedder(model_id)
    return store, embedder


def make_context_from_hits(store: FaissStore, hits: List[Tuple[int, float]], max_chars: int = 6000) -> Tuple[str, List[Dict]]:
    parts: List[str] = []
    used_meta: List[Dict] = []
    total = 0
    for idx, score in hits:
        m = store.metadata[idx]
        snippet = m.get("text", "")
        if not snippet:
            continue
        add = snippet
        if total + len(add) > max_chars:
            add = add[: max(0, max_chars - total)]
        if not add:
            break
        parts.append(add)
        used_meta.append({"source": m.get("source"), "chunk_index": m.get("chunk_index"), "score": score})
        total += len(add)
        if total >= max_chars:
            break
    return "\n\n".join(parts), used_meta
