# Personal Knowledge Assistant

A document ingestion and semantic search pipeline built in Python. Parses PDF documents, chunks them using multiple strategies (including genuine semantic approaches), generates embeddings with sentence-transformers, and stores them in a FAISS vector index for fast similarity search.

## Architecture

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
│  Parser  │───►│ Chunker  │───►│ Embedder  │───►│  FAISS   │
│ (PyMuPDF)│    │(multiple)│    │(MiniLM-L6)│    │  Store   │
└──────────┘    └──────────┘    └───────────┘    └──────────┘
     PDF          Chunks         384-dim vectors    Indexed
```

**Pipeline:** PDF → Parse into sections → Chunk with configurable strategy → Embed with sentence-transformers → Store in FAISS for cosine similarity search.

## Features

- **PDF parsing** with PyMuPDF — extracts text, page structure, and metadata
- **6 chunking strategies** from simple to semantic:
  - `fixed` — fixed-size with word boundary awareness
  - `sentence` — respects sentence boundaries
  - `embedding_similarity` — detects topic shifts via cosine similarity
  - `density_clustering` — HDBSCAN clustering on embeddings
  - `topic_modeling` — BERTopic topic segmentation
  - `recursive_hierarchical` — multi-level hierarchical chunks
- **Sentence-transformer embeddings** using `all-MiniLM-L6-v2` (384 dimensions)
- **FAISS vector store** with persistence, document filtering, and deletion
- **YAML configuration** for all pipeline parameters
- **Structured logging** throughout

## Installation

```bash
# Clone and set up
git clone <repo-url>
cd personal-knowledge-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.ingestion import KnowledgeBase

# Create a knowledge base
kb = KnowledgeBase(
    index_path="data/index/my_kb",
    chunk_strategy="sentence",
    chunk_size=512,
)

# Ingest a document
kb.ingest("path/to/document.pdf")

# Search
results = kb.search("What is composition in OOP?", top_k=3)
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Source: {result.chunk.get_citation()}")
    print(f"Content: {result.chunk.content[:200]}...")
```

## Project Structure

```
├── configs/
│   └── config.yaml              # Pipeline configuration
├── src/
│   ├── config.py                # Configuration management
│   ├── utils/
│   │   └── logger.py            # Structured logging
│   └── ingestion/
│       ├── knowledge_base.py    # High-level API
│       ├── document_manager.py  # Multi-format document routing
│       ├── document.py          # Document data models
│       ├── parsers/
│       │   ├── base.py          # Abstract parser interface
│       │   └── pdf_parser.py    # PDF parser (PyMuPDF)
│       ├── chunking/
│       │   ├── base_chunker.py  # Abstract chunker interface
│       │   ├── chunk.py         # Chunk data model
│       │   ├── chunk_manager.py # Strategy selector
│       │   ├── fixed_size_chunker.py
│       │   ├── sentence_chunker.py
│       │   ├── embedding_similarity_chunker.py
│       │   ├── density_chunker.py
│       │   ├── topic_chunker.py
│       │   └── recursive_chunker.py
│       ├── embeddings/
│       │   └── embedding_service.py
│       └── storage/
│           ├── vector_store.py  # Abstract store interface
│           └── faiss_store.py   # FAISS implementation
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── examples/
│   ├── basic_usage.py
│   └── advanced_usage.py
└── requirements.txt
```

## Chunking Strategies

| Strategy | Semantic? | Speed | Best For |
|----------|-----------|-------|----------|
| `fixed` | No | Fast | Baseline comparison |
| `sentence` | No | Fast | Well-structured prose |
| `embedding_similarity` | Yes | Medium | Topic boundary detection |
| `density_clustering` | Yes | Slower | Documents with distinct topics |
| `topic_modeling` | Yes | Slower | Multi-topic documents |
| `recursive_hierarchical` | Yes | Medium | Structured documents |

## Configuration

Edit `configs/config.yaml`:

```yaml
embedding:
  model_name: all-MiniLM-L6-v2
  batch_size: 32

chunking:
  strategy: sentence
  chunk_size: 512
  chunk_overlap: 50
  similarity_threshold: 0.5

storage:
  backend: faiss
  index_path: data/index
  auto_save: true
```

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/unit/test_chunking.py -v
```

## Key Design Decisions

- **Composition over inheritance** — The pipeline components (parser, chunker, embedder, store) are composed together in `KnowledgeBase`, not inherited. Each can be swapped independently.
- **Strategy pattern** — `ChunkManager` uses the strategy pattern to select chunking algorithms at runtime without changing client code.
- **Abstract base classes** — `BaseParser`, `BaseChunker`, and `VectorStore` define contracts that concrete implementations must satisfy, enabling extensibility.
- **Lazy loading** — Embedding models and ML dependencies are loaded on first use, keeping imports fast.
- **Factory pattern** — `ChunkManager` acts as a factory, instantiating the correct chunker based on a strategy string.

## Technologies

- **Python 3.10+**
- **PyMuPDF** — PDF text extraction
- **sentence-transformers** — Text embeddings (all-MiniLM-L6-v2)
- **FAISS** — Facebook AI Similarity Search for vector indexing
- **HDBSCAN / UMAP / BERTopic** — Semantic chunking (optional)
- **pytest** — Testing framework
