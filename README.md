# Personal Knowledge Assistant

A document ingestion and semantic search pipeline with retrieval-augmented generation (RAG), built as a first-year Computer Science project demonstrating advanced software design principles.

## Architecture

```
                        ┌─────────────────────────────────┐
                        │         Knowledge Base           │
                        │       (High-Level API)           │
                        └────────────┬────────────────────┘
                                     │
         ┌───────────────┬───────────┼───────────┬─────────────────┐
         ▼               ▼           ▼           ▼                 ▼
   ┌──────────┐   ┌──────────┐ ┌─────────┐ ┌─────────┐    ┌──────────┐
   │  Parse   │   │  Chunk   │ │  Embed  │ │  Store  │    │ Retrieve │
   │          │   │          │ │         │ │         │    │          │
   │ PDF      │   │ Fixed    │ │MiniLM   │ │ FAISS   │    │ Rerank   │
   │ TXT/MD   │   │ Sentence │ │L6-v2    │ │ Index   │    │ Expand   │
   │ DOCX     │   │ Semantic │ │384-dim  │ │ Cosine  │    │ Evaluate │
   └──────────┘   │ Density  │ └─────────┘ └─────────┘    └──────────┘
                  │ Topic    │
                  │ Recursive│                        ┌──────────────┐
                  └──────────┘                        │     RAG      │
                                                      │              │
                                                      │ Claude API   │
                                                      │ Conversation │
                                                      │ Citations    │
                                                      └──────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest a document
python -m src.cli ingest paper.pdf

# Search the knowledge base
python -m src.cli search "What is dependency injection?"

# Interactive RAG chat (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
python -m src.cli chat

# Start the web interface
python -m src.web.app
```

### Python API

```python
from src.ingestion import KnowledgeBase

kb = KnowledgeBase(index_path="data/index/myindex")

# Ingest any supported format
kb.ingest("lecture_notes.pdf")
kb.ingest("README.md")
kb.ingest("report.docx")

# Semantic search
results = kb.search("What is the strategy pattern?", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.chunk.content[:100]}")

# Advanced search with reranking and query expansion
results = kb.advanced_search(
    "dependency injection",
    top_k=5,
    rerank=True,
    expand_query="synonym",
)

# RAG-powered Q&A
from src.rag import RAGPipeline, ClaudeProvider
rag = RAGPipeline(knowledge_base=kb, llm_provider=ClaudeProvider())
response = rag.query("Explain the observer pattern")
print(response.answer)
```

## Project Structure

```
├── pyproject.toml              # Project metadata and tool config
├── requirements.txt            # Pinned dependencies
├── configs/
│   └── config.yaml             # Pipeline configuration
├── src/
│   ├── cli.py                  # Click CLI (ingest/search/chat/eval)
│   ├── config.py               # YAML config loader
│   ├── config_schema.py        # Pydantic validation models
│   ├── ingestion/
│   │   ├── knowledge_base.py   # High-level API
│   │   ├── document_manager.py # Multi-format parser routing
│   │   ├── document.py         # Document/Section data models
│   │   ├── parsers/
│   │   │   ├── base.py         # Abstract BaseParser
│   │   │   ├── pdf_parser.py   # PyMuPDF implementation
│   │   │   ├── text_parser.py  # TXT/Markdown with heading detection
│   │   │   └── docx_parser.py  # Word with table extraction
│   │   ├── chunking/
│   │   │   ├── base_chunker.py # Abstract BaseChunker
│   │   │   ├── chunk_manager.py# Strategy selector
│   │   │   ├── fixed_size_chunker.py
│   │   │   ├── sentence_chunker.py
│   │   │   ├── embedding_similarity_chunker.py
│   │   │   ├── density_chunker.py
│   │   │   ├── topic_chunker.py
│   │   │   └── recursive_chunker.py
│   │   ├── embeddings/
│   │   │   └── embedding_service.py  # Sentence-transformers wrapper
│   │   └── storage/
│   │       ├── vector_store.py  # Abstract VectorStore
│   │       └── faiss_store.py   # FAISS implementation
│   ├── retrieval/
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   ├── query_expansion.py   # Synonym/multi-query/HyDE
│   │   └── evaluation.py        # IR metrics (P@K, MRR, nDCG, MAP)
│   ├── rag/
│   │   ├── llm.py               # LLM provider abstraction + Claude
│   │   └── pipeline.py          # RAG orchestration with memory
│   ├── web/
│   │   ├── app.py               # Flask REST API
│   │   └── templates/
│   │       └── index.html       # Single-page frontend
│   └── utils/
│       └── logger.py            # Structured logging
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── fixtures/
│   │   ├── sample.txt           # Plain text fixture
│   │   ├── sample.md            # Markdown fixture
│   │   └── sample.docx          # Word document fixture
│   └── unit/
│       ├── test_parsers.py      # Parser + DocumentManager tests (43)
│       ├── test_chunking.py     # Chunking strategy tests
│       ├── test_embeddings.py   # Embedding service tests
│       ├── test_vector_store.py # FAISS store tests
│       └── test_retrieval.py    # Query expansion + eval tests (25)
└── data/
    ├── index/                   # Persisted FAISS indices
    └── test_corpus/             # Sample documents
```

## Features

### Document Parsing
Four parsers following the Strategy pattern, routed automatically by `DocumentManager`:

| Format | Parser | Features |
|--------|--------|----------|
| `.pdf` | `PDFParser` | Page-level sections, PDF metadata extraction |
| `.txt` | `TextParser` | Paragraph splitting, encoding fallback (UTF-8 → Latin-1) |
| `.md` | `TextParser` | Heading hierarchy detection (H1–H6), intro sections |
| `.docx` | `DOCXParser` | Word heading styles, table-to-text conversion, author metadata |

### Chunking Strategies
Six strategies from simple to advanced, selectable via configuration:

| Strategy | Semantic | Speed | Best For |
|----------|----------|-------|----------|
| `fixed` | No | Fast | Uniform chunks, baseline |
| `sentence` | No | Fast | Natural text boundaries |
| `embedding_similarity` | Yes | Medium | Coherent topic segments |
| `density_clustering` | Yes | Slow | Discovering natural clusters (HDBSCAN) |
| `topic_modeling` | Yes | Slow | Topic-aligned chunks (BERTopic) |
| `recursive_hierarchical` | Yes | Medium | Hierarchical document structure |

### Retrieval Pipeline
Two-stage retrieve-then-rerank architecture:

1. **FAISS retrieval**: Fast approximate nearest-neighbour search (cosine similarity).
2. **Cross-encoder reranking**: Joint (query, document) scoring for precision (ms-marco-MiniLM).
3. **Query expansion**: Synonym substitution, multi-query rephrasing, or HyDE.
4. **Evaluation**: Precision@K, Recall@K, MRR, nDCG, MAP with JSON-based relevance judgments.

### RAG (Retrieval-Augmented Generation)
Context-grounded question answering via Claude API:

- System prompt enforcing answer grounding with source citations.
- Conversation memory for multi-turn follow-up questions.
- Configurable context window with automatic truncation.
- Source attribution in every response.

### Interfaces
Three ways to interact with the system:

- **CLI** (`python -m src.cli`): Full-featured command line with `ingest`, `search`, `chat`, `compare`, `eval`, `info`, `delete`, `clear`.
- **Web UI** (`python -m src.web.app`): Flask app with search, upload, and chat tabs.
- **Python API**: Direct `KnowledgeBase` and `RAGPipeline` usage.

## CLI Reference

```bash
# Ingest documents
python -m src.cli ingest paper.pdf
python -m src.cli ingest ./docs/ --strategy embedding_similarity

# Search
python -m src.cli search "query" --top-k 10
python -m src.cli search "query" --rerank --expand synonym

# Interactive chat
python -m src.cli chat --rerank

# Index management
python -m src.cli info
python -m src.cli delete <doc_id>
python -m src.cli clear --confirm

# Analysis
python -m src.cli compare document.pdf
python -m src.cli eval judgments.json --top-k 5
```

## Configuration

Edit `configs/config.yaml` or use CLI flags:

```yaml
embedding:
  model_name: all-MiniLM-L6-v2
  batch_size: 32

chunking:
  strategy: sentence
  chunk_size: 512
  chunk_overlap: 50

storage:
  backend: faiss
  index_path: data/index/default

retrieval:
  rerank: false
  query_expansion: null  # synonym | multi_query | hyde
```

## Running Tests

```bash
# All tests
pytest

# Specific test suites
pytest tests/unit/test_parsers.py -v
pytest tests/unit/test_retrieval.py -v

# Skip slow tests (semantic chunking)
pytest -m "not slow"
```

## Design Decisions

This project demonstrates several advanced software design principles:

- **Strategy Pattern**: Chunking strategies, parser selection, and LLM providers are all interchangeable via abstract interfaces.
- **Factory Pattern**: `DocumentManager` routes files to parsers based on extension. `ChunkManager` instantiates chunkers by strategy name.
- **Abstract Base Classes**: `BaseParser`, `BaseChunker`, `VectorStore`, and `LLMProvider` define contracts that implementations must follow.
- **Composition over Inheritance**: `KnowledgeBase` composes a parser, chunker, embedder, and store rather than inheriting from any of them.
- **Lazy Loading**: Embedding models, cross-encoders, and API clients are only loaded when first used.
- **Dependency Inversion**: High-level modules depend on abstractions, not concrete implementations.
- **Mediator Pattern**: `RAGPipeline` coordinates between `KnowledgeBase`, `LLMProvider`, and retrieval enhancements.
- **Open/Closed Principle**: New parsers and chunking strategies can be added without modifying existing code.

## Technology Stack

- **Python 3.10+** with type hints throughout
- **PyMuPDF** for PDF parsing
- **python-docx** for Word document parsing
- **sentence-transformers** (all-MiniLM-L6-v2) for embeddings
- **FAISS** for vector similarity search
- **HDBSCAN / UMAP / BERTopic** for semantic chunking
- **Anthropic Claude API** for RAG generation
- **Flask** for REST API and web frontend
- **Click** for CLI framework
- **Pydantic** for configuration validation
- **pytest** for testing
