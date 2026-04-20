# Personal Knowledge Assistant

A production-grade Retrieval-Augmented Generation (RAG) system for personal document knowledge bases, built as a final-year BSc Computing project at Atlantic Technological University.

The system ingests mixed-format documents (PDF, DOCX, Markdown, plain text), builds hybrid lexical + semantic indices, and answers natural-language questions with source-cited answers grounded in the retrieved context. It extends vanilla RAG with adaptive re-retrieval, semantic caching, HyDE, a spaCy-backed knowledge graph, and two independent evaluation layers (classical IR metrics and a from-scratch implementation of RAGAS-style answer evaluation).

---

## Table of contents

- [Headline results](#headline-results)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Feature overview](#feature-overview)
- [Interfaces](#interfaces)
- [Evaluation](#evaluation)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Design decisions](#design-decisions)
- [Technology stack](#technology-stack)
- [References](#references)

---

## Headline results

Measured over an 8-question test set against the full pipeline, using LLM-as-judge methodology:

| Metric                | Score   | Interpretation                                         |
| --------------------- | ------- | ------------------------------------------------------ |
| Faithfulness          | 0.925   | 92.5% of answer claims are supported by context        |
| Answer Relevancy      | 0.798   | Answers address the question asked                     |
| Context Precision     | 0.731   | Retrieval places relevant chunks first most of the time |
| Composite RAGAS score | 0.772   | Harmonic mean — strong overall production quality      |

The evaluation pipeline also surfaces actionable signal: it correctly identified a topic (CNN coverage) where the ingested corpus was weak and confirmed that the pipeline's grounding layer refused to hallucinate in that case, earning a perfect faithfulness score of 1.0 while correctly reporting low context precision.

Full per-query breakdowns are under `data/eval/results_*.json`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Interfaces                                   │
│   CLI (click) │ Flask Blueprints (6) │ Python API │ Eval CLI         │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                       RAG Pipeline (Mediator)                        │
│                                                                      │
│   semantic cache ─▶ query rewrite ─▶ HyDE ─▶ retrieval ─▶ generation │
│                                        ▲           │                 │
│                                        │     grounding score         │
│                                        │           │                 │
│                                        └── adaptive re-retrieval     │
│                                                    │                 │
│                                          fact verification           │
│                                                    │                 │
│                                    query history + memory            │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                           Retrieval                                  │
│                                                                      │
│   FAISS  +  BM25  ──▶  RRF merge  ──▶  entity boost  ──▶  rerank     │
│   (dense)  (sparse)                    (spaCy NER)      (cross-enc)  │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                          Ingestion                                   │
│                                                                      │
│   parsers (4) ─▶ chunkers (6) ─▶ embeddings ─▶ storage ─▶ registry   │
│   PDF/DOCX/MD/TXT               MiniLM-L6     FAISS/BM25  SHA-256    │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                      Intelligence layer                              │
│                                                                      │
│   knowledge graph │ conflict detection │ document similarity         │
│   study paths     │ quiz generator     │ summariser                  │
└──────────────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────────┐
│                         Evaluation                                   │
│                                                                      │
│   IR metrics              │   Answer evaluation (RAGAS-style)        │
│   Precision@K, MRR, nDCG  │   Faithfulness, Relevancy, Context Prec. │
└──────────────────────────────────────────────────────────────────────┘
```

Each layer depends only on the abstractions below it; concrete implementations are interchangeable.

---

## Quick start

### Prerequisites
- Python 3.10+
- `pip` and a working C toolchain (for `faiss-cpu`)
- Anthropic API key (only required for RAG chat, HyDE, conflict detection, study features, quiz, and answer evaluation; core search and IR evaluation work without one)

### Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # required for NER + knowledge graph
```

Set the API key:

```bash
# Linux / macOS
export ANTHROPIC_API_KEY=sk-ant-...

# Windows (CMD)
set ANTHROPIC_API_KEY=sk-ant-...
```

Or copy `.env.example` to `.env` and fill it in.

### Ingest and search (CLI)

```bash
# Ingest a single document or a directory
python -m src.cli ingest paper.pdf
python -m src.cli ingest ./course_notes/ --strategy sentence

# Semantic search
python -m src.cli search "What is dependency injection?" --top-k 5

# Hybrid search with reranking
python -m src.cli search "neural network training" --rerank --hybrid

# Interactive RAG chat
python -m src.cli chat
```

### Web UI

```bash
python -m src.web.app
# open http://localhost:5000
```

The web app serves a single-page frontend with twelve views (chat, semantic search, upload, study paths, quizzes, summariser, knowledge graph, conflicts, similarity, evaluation dashboard, analytics, and notes) plus standalone templated pages for the graph viewer, quiz runner, study path tool, conflicts analyser, and evaluation dashboard.

### Python API

```python
from src.ingestion import KnowledgeBase
from src.rag import RAGPipeline, ClaudeProvider

kb = KnowledgeBase(index_path="data/index/myindex")

# Ingest any supported format
kb.ingest("lecture_notes.pdf")
kb.ingest("notes.md")

# Advanced search with hybrid retrieval + reranking
results = kb.advanced_search(
    "backpropagation",
    top_k=5,
    rerank=True,
    hybrid=True,
    expand_query="hyde",
)

# Full RAG pipeline with caching, grounding, and fact verification
rag = RAGPipeline(knowledge_base=kb, llm_provider=ClaudeProvider())
response = rag.query("Explain how gradient descent is used to train a neural network.")

print(response.answer)
print(f"Confidence: {response.confidence:.2f}")
print(f"Retrieval attempts: {response.retrieval_attempts}")
print(f"Cache hit: {response.cache_hit}")
for source in response.sources:
    print(f"  - {source.chunk.source_doc_title} (score: {source.score:.3f})")
```

---

## Feature overview

### Document ingestion
- Four parsers selected automatically by extension: `PDFParser` (PyMuPDF), `TextParser` (TXT + Markdown heading detection), `DOCXParser` (styles + table-to-text).
- Deterministic document IDs from file stems, SHA-256 hashing for change detection, silent re-ingestion of modified documents with stale-chunk cleanup.

### Chunking strategies
Six strategies selectable via configuration:

| Strategy | Semantic-aware | Notes |
| --- | --- | --- |
| `fixed` | No | Baseline, uniform sizes |
| `sentence` | No | Natural boundaries, default |
| `recursive` | Partial | Respects heading hierarchy |
| `embedding_similarity` | Yes | Splits where consecutive sentence similarity drops |
| `density_clustering` | Yes | HDBSCAN clustering of sentence embeddings |
| `topic_modeling` | Yes | BERTopic-aligned segments |

### Storage
- **FAISS** for dense cosine-similarity retrieval.
- **BM25** (`rank-bm25`) for lexical retrieval.
- **DocumentRegistry** tracking filename, hash, chunk count, and timestamps per document.
- All three persist atomically to disk as a linked index set (`.faiss`, `.bm25.pkl`, `.meta.json`, `.registry.json`).

### Retrieval
- **Hybrid search** merges dense and sparse candidates via Reciprocal Rank Fusion.
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) refines the candidate pool.
- **Query expansion** via synonym substitution, multi-query paraphrasing, or HyDE (LLM-generated hypothetical documents).
- **Entity boosting** upweights chunks whose spaCy-extracted entities overlap with entities in the query, with optional label filtering (`PERSON`, `ORG`, etc.).
- Cached reranker and expanders avoid the 30-second cold-start penalty on repeated queries.

### RAG pipeline
Mediator over the retrieval stack and the LLM:

1. **Semantic cache lookup.** Similar queries (cosine ≥ 0.92) return cached responses without any LLM or retrieval cost.
2. **Query rewriting.** Conversational follow-ups ("what about it?") are rewritten to self-contained queries using the conversation memory.
3. **HyDE.** An LLM-generated ideal answer is embedded and used for retrieval, catching semantic matches that keyword approaches miss.
4. **Retrieval** through the full hybrid + rerank + boost stack.
5. **Generation** via the `LLMProvider` abstraction (Claude by default).
6. **Grounding scoring.** Every sentence of the answer is scored against the retrieved chunks.
7. **Adaptive re-retrieval.** Low-confidence answers automatically trigger query reformulation and retry (up to `max_retries`).
8. **Fact verification.** Each answer sentence is classified as supported / partial / unverified.
9. **Persistence.** Query history, usage analytics, and sliding-window conversation memory all persist across restarts.

### Intelligence layer
- **Knowledge graph** — spaCy NER extracts entities from every chunk; NetworkX builds a co-occurrence graph; D3.js renders it interactively in the browser.
- **Conflict detection** — surfaces contradictions between documents on a given topic via LLM-guided analysis.
- **Document similarity** — pairwise cosine-similarity matrix over mean document embeddings.
- **Study path generator** — produces a structured learning order from the ingested corpus for a target topic.
- **Quiz generator** — multiple-choice and short-answer questions grounded in the corpus.
- **Summariser** — extractive + abstractive document summaries.
- **Annotations** — persistent per-chunk notes CRUD.

### Evaluation
Two independent evaluation layers:

**IR evaluation** (`src/evaluation/evaluator.py`) — classical metrics over a JSON relevance-judgement file.

- Precision@K
- Mean Reciprocal Rank (MRR)
- Normalised Discounted Cumulative Gain (nDCG@K)

**Answer evaluation** (`src/evaluation/answer_eval.py`) — a from-scratch implementation of the RAGAS methodology using Claude as the judge LLM.

- **Faithfulness** — atomic-claim decomposition and per-claim entailment checking against retrieved context.
- **Answer Relevancy** — reverse-question generation and mean cosine similarity to the original query.
- **Context Precision** — rank-weighted precision using the RAGAS formula `Σ(precision@k × v_k) / total_relevant`.

The implementation is documented in `src/evaluation/answer_eval.py`. LLM prompts, response parsers, and aggregation maths are all in-repo. A JSONL test set under `data/eval/sample_test_set.jsonl` provides eight general AI/ML questions as a starter; swap in your own domain set as needed.

---

## Interfaces

### 1. CLI

```bash
python -m src.cli ingest <file_or_dir> [--strategy <name>]
python -m src.cli search <query> [--top-k 5] [--rerank] [--hybrid] [--expand hyde|synonym|multi_query]
python -m src.cli chat [--rerank] [--hybrid]
python -m src.cli compare <document>
python -m src.cli eval <judgments.json> [--top-k 5]
python -m src.cli info
python -m src.cli delete <doc_id>
python -m src.cli clear --confirm
```

### 2. Web UI

Flask application organised into six blueprints following the Single Responsibility Principle:

| Blueprint | Responsibility |
| --- | --- |
| `core` | Frontend routes, health, info, document management |
| `chat` | RAG Q&A, streaming responses, conversation memory |
| `search` | Semantic and hybrid search, document upload |
| `study` | Study paths, quizzes, summariser |
| `intelligence` | Knowledge graph, conflicts, similarity, IR evaluation |
| `analytics` | Query history, analytics, annotations |

The single-page frontend (`index.html`) is complemented by dedicated templated pages for the graph viewer, quiz runner, study-path tool, conflicts analyser, and the evaluation dashboard (Chart.js).

Security hardening: `secure_filename` on uploads, path-traversal protection on evaluation file paths, 50 MB upload cap, request-timing middleware.

### 3. Python API

The `KnowledgeBase` and `RAGPipeline` classes expose the entire system programmatically — see the Quick Start example above.

### 4. Evaluation CLI

```bash
python -m scripts.run_eval --test-set data/eval/sample_test_set.jsonl
```

Shows estimated API call count, asks for confirmation, then runs the full pipeline on each test case and writes a timestamped JSON report.

---

## Evaluation

### Retrieval quality (IR metrics)

Provide a judgement file with relevant chunk IDs per query:

```json
[
  {"query": "What is gradient descent?", "relevant_chunks": ["chunk_042", "chunk_091"]},
  {"query": "Define backpropagation",    "relevant_chunks": ["chunk_017"]}
]
```

Then:

```bash
python -m src.cli eval data/eval/my_judgments.json --top-k 5
```

Or open `http://localhost:5000/dashboard` for the Chart.js-rendered report.

### Answer quality (RAGAS-style)

A JSONL test set — one JSON object per line with `question` and `ground_truth`:

```json
{"question": "What is overfitting?", "ground_truth": "A model that learns noise in training data..."}
```

Then:

```bash
python -m scripts.run_eval --test-set my_test_set.jsonl
```

The runner prints a per-question table and writes a full JSON report including per-claim faithfulness judgements, reverse questions, and per-chunk relevance labels.

---

## Project structure

```
├── pyproject.toml
├── requirements.txt
├── README.md
├── .env.example
├── configs/
│   └── config.yaml                # pipeline configuration
├── data/
│   ├── eval/                      # test sets and results (JSONL / JSON)
│   ├── graph/                     # cached knowledge graph
│   ├── history/                   # query history
│   ├── index/                     # persisted FAISS + BM25 + registry
│   └── memory/                    # conversation memory
├── examples/
│   ├── basic_usage.py
│   └── advanced_usage.py
├── scripts/
│   └── run_eval.py                # RAGAS-style evaluation CLI
├── src/
│   ├── cli.py                     # click-based CLI
│   ├── config.py                  # dataclass config + YAML loader
│   ├── evaluation/
│   │   ├── answer_eval.py         # RAGAS-style metrics
│   │   ├── evaluator.py           # IR metrics orchestrator
│   │   └── metrics.py             # P@K, MRR, nDCG
│   ├── ingestion/
│   │   ├── chunking/              # 6 chunking strategies + manager
│   │   ├── embeddings/            # sentence-transformer service
│   │   ├── parsers/               # PDF / DOCX / Markdown / TXT
│   │   ├── storage/               # FAISS, BM25, document registry
│   │   ├── document.py
│   │   ├── document_manager.py
│   │   ├── knowledge_base.py      # high-level facade
│   │   ├── ner_extractor.py       # spaCy NER
│   │   └── similarity.py
│   ├── knowledge_graph/
│   │   ├── graph_builder.py       # NetworkX graph construction
│   │   └── graph_store.py         # persistence + D3 serialisation
│   ├── rag/
│   │   ├── annotations.py
│   │   ├── cache.py               # semantic query cache
│   │   ├── conflict_detector.py
│   │   ├── fact_verifier.py       # sentence-level verification
│   │   ├── grounding.py
│   │   ├── llm.py                 # LLMProvider ABC + ClaudeProvider
│   │   ├── memory.py              # persistent conversation memory
│   │   ├── pipeline.py            # the RAG orchestrator
│   │   └── query_history.py
│   ├── retrieval/
│   │   ├── entity_reranker.py
│   │   ├── hybrid_search.py       # RRF fusion
│   │   ├── query_expansion.py     # synonym / multi-query / HyDE
│   │   └── reranker.py            # cross-encoder reranking
│   ├── study/
│   │   ├── path_generator.py
│   │   ├── quiz_generator.py
│   │   └── summariser.py
│   ├── utils/
│   │   └── logger.py
│   └── web/
│       ├── app.py                 # Flask application factory
│       ├── blueprints/            # 6 blueprints + shared dependencies
│       └── templates/             # 6 HTML templates incl. SPA
└── tests/
    ├── conftest.py
    ├── fixtures/                  # sample.pdf, sample.docx, sample.md, sample.txt
    ├── integration/
    └── unit/                      # 20+ unit test modules
```

---

## Configuration

Edit `configs/config.yaml`:

```yaml
embedding:
  model_name: all-MiniLM-L6-v2
  batch_size: 32

chunking:
  strategy: sentence          # fixed | sentence | recursive |
                              # embedding_similarity | density_clustering | topic_modeling
  chunk_size: 512
  chunk_overlap: 50
  similarity_threshold: 0.5   # only used by embedding_similarity

storage:
  backend: faiss
  index_path: data/index/default
  auto_save: true
```

All fields are optional and fall back to the dataclass defaults in `src/config.py`.

---

## Testing

```bash
# Full suite
python -m pytest

# Fast unit tests only
python -m pytest tests/unit/

# Verbose single file
python -m pytest tests/unit/test_answer_eval.py -v

# Skip slow semantic-chunking tests
python -m pytest -m "not slow"
```

**Current test count: 512 passing, 0 failing.** All LLM-dependent tests use a mocked provider (`_ScriptedLLM`), so the full suite runs with no API calls and no cost.

Coverage breakdown:

- Retrieval and IR metrics: `test_hybrid_search.py`, `test_retrieval.py`, `test_metrics.py`
- Chunking, parsing, embeddings, FAISS, BM25: `test_chunking.py`, `test_parsers.py`, `test_embeddings.py`, `test_vector_store.py`, `test_knowledge_base.py`
- RAG pipeline: `test_adaptive.py`, `test_cache.py`, `test_grounding.py`, `test_fact_verifier.py`, `test_memory.py`, `test_pipeline_empty.py`, integration `test_pipeline.py`
- Intelligence: `test_graph.py`, `test_ner.py`, `test_conflict_detector.py`, `test_similarity.py`, `test_study.py`, `test_quiz.py`, `test_summariser.py`
- Evaluation: `test_metrics.py`, `test_answer_eval.py` (40 RAGAS tests)
- Interfaces: `test_api.py` (44 Flask endpoint tests), `test_cli.py`
- Persistence: `test_document_registry.py`, `test_annotations.py`, `test_query_history.py`

---

## Design decisions

The project was designed as an exercise in applying established software engineering patterns to a non-trivial system.

- **Strategy pattern** — chunkers, parsers, and LLM providers all behind abstract interfaces, swappable without changes to calling code.
- **Factory pattern** — `DocumentManager` routes files to parsers by extension; `ChunkManager` instantiates chunkers by strategy name.
- **Mediator pattern** — `RAGPipeline` coordinates the `KnowledgeBase`, `LLMProvider`, grounding scorer, fact verifier, cache, and memory without any of them knowing about each other.
- **Repository pattern** — `AnnotationStore`, `QueryHistory`, and `DocumentRegistry` each own their persistence, exposing a data-access API.
- **Lazy loading** — sentence-transformer models (~90 MB), the cross-encoder (~130 MB), spaCy models (~12 MB), and the Anthropic client all load on first use, keeping startup sub-second.
- **Dependency inversion** — the pipeline depends on `LLMProvider` and `VectorStore` abstractions, not Claude or FAISS specifically.
- **Composition over inheritance** — `KnowledgeBase` composes a parser, chunker, embedder, and store rather than inheriting from any of them.
- **Open/closed principle** — new parsers, chunking strategies, LLM providers, and query expanders can be added without touching existing code.
- **Application-factory pattern** — `create_app()` in `src/web/app.py` allows per-test Flask instances with clean state.
- **Blueprint architecture** — six Flask blueprints split by domain (core, chat, search, study, intelligence, analytics) rather than one monolithic `app.py`.

---

## Technology stack

**Language & runtime**
- Python 3.10+ with type hints
- Flask (application factory + blueprint architecture)
- Click (CLI)

**Retrieval**
- FAISS (dense cosine retrieval)
- `rank-bm25` (sparse lexical retrieval)
- `sentence-transformers` (MiniLM-L6-v2 embeddings + ms-marco-MiniLM cross-encoder)

**NLP**
- spaCy (`en_core_web_sm`) for NER
- NetworkX for graph construction
- NLTK for sentence tokenisation

**LLM**
- Anthropic Claude via the official SDK, behind an `LLMProvider` abstraction

**Data & persistence**
- JSON / JSONL for test sets, judgement files, history, annotations, conversation memory
- Pickle for BM25 index persistence
- FAISS's native binary format for the vector index

**Frontend**
- Vanilla HTML/CSS/JS (no framework)
- D3.js for interactive knowledge-graph rendering
- Chart.js for the evaluation dashboard

**Document parsing**
- PyMuPDF (PDFs)
- python-docx (Word documents)

**Semantic chunking (optional)**
- HDBSCAN, UMAP, BERTopic (under `scikit-learn`)

**Configuration**
- PyYAML, pydantic, python-dotenv

**Testing**
- pytest with fixtures and parameterised tests
- `unittest.mock` for LLM / Flask / KnowledgeBase mocking

---

## References

- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS.
- Es, James, Espinosa-Anke, Schockaert (2023), *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, arXiv:2309.15217.
- Gao et al. (2022), *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE), arXiv:2212.10496.
- Cormack, Clarke, Buettcher (2009), *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR.
- Reimers & Gurevych (2019), *Sentence-BERT*, EMNLP.
- Honnibal & Montani (2017), *spaCy 2* (for industrial-strength NER).

---

## Author

Patrick Clerkin — BSc Computing in Software Development, Atlantic Technological University (2026).