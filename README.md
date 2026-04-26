# Personal Knowledge Assistant

A production-grade Retrieval-Augmented Generation (RAG) system for personal document knowledge bases. Built as a final-year BSc Computing project at Atlantic Technological University.

The system ingests mixed-format documents (PDF, DOCX, Markdown, TXT), builds hybrid lexical + semantic indices, and answers natural-language questions with source-cited answers grounded in retrieved context. It extends vanilla RAG with adaptive re-retrieval, semantic caching, HyDE, a spaCy-backed knowledge graph, and two independent evaluation layers.


---

## Demo

[![Personal Knowledge Assistant — demo](https://img.youtube.com/vi/_1bexu7VyUw/maxresdefault.jpg)](https://youtu.be/_1bexu7VyUw)

Click the thumbnail above for a full walkthrough of the system: ingestion, hybrid search, RAG chat with grounding, the knowledge graph, and the evaluation dashboard.


---

## Headline results

LLM-as-judge evaluation over an 8-question test set against the full pipeline:

| Metric            | Score |
| ----------------- | ----- |
| Faithfulness      | 0.925 |
| Answer Relevancy  | 0.798 |
| Context Precision | 0.731 |
| Composite RAGAS   | 0.772 |

Per-query breakdowns under `data/eval/results_*.json`.

---

## Architecture

```
Documents (PDF, DOCX, MD, TXT)
    │
    ▼
Ingestion — 6 chunkers, sentence-transformer embeddings, SHA-256 registry
    │
    ▼
Hybrid Index — FAISS (dense) + BM25 (sparse) + RRF merge
    │
    ▼
RAG Pipeline — semantic cache → HyDE → retrieve → rerank →
               generate → ground → adaptive re-retrieve → verify
    │
    ├──▶ Intelligence — knowledge graph, conflicts, study paths, quizzes
    └──▶ Evaluation — IR metrics (P@K, MRR, nDCG) + RAGAS-style
```

Each layer depends only on abstractions below it; concrete implementations are swappable.

---

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set the Anthropic API key (required for chat, HyDE, conflict detection, study features, and answer evaluation):

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # Linux / macOS
```

```cmd
set ANTHROPIC_API_KEY=sk-ant-...        REM Windows CMD
```

Or copy `.env.example` to `.env` and fill it in.

**CLI**

```bash
python -m src.cli ingest paper.pdf
python -m src.cli search "What is dependency injection?" --rerank --hybrid
python -m src.cli chat
```

**Web UI**

```bash
python -m src.web.app    # http://localhost:5000
```

**Python API**

```python
from src.ingestion import KnowledgeBase
from src.rag import RAGPipeline, ClaudeProvider

kb = KnowledgeBase(index_path="data/index/myindex")
kb.ingest("lecture_notes.pdf")

rag = RAGPipeline(knowledge_base=kb, llm_provider=ClaudeProvider())
response = rag.query("Explain gradient descent.")
print(response.answer, response.confidence)
```

---

## Features

**Ingestion** — PDF / DOCX / Markdown / TXT parsers; six chunking strategies (`fixed`, `sentence`, `recursive_hierarchical`, `embedding_similarity`, `density_clustering`, `topic_modeling`); SHA-256 change detection with stale-chunk cleanup on re-ingest.

**Retrieval** — FAISS dense + BM25 sparse, merged via Reciprocal Rank Fusion; cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`); query expansion via synonyms, multi-query paraphrasing, or HyDE; spaCy NER entity boosting with optional label filters.

**RAG pipeline** — semantic cache (cosine ≥ 0.92), conversational query rewriting, sentence-level grounding scoring, adaptive re-retrieval on low confidence, sentence-level fact verification, persistent conversation memory.

**Intelligence layer** — interactive knowledge graph (spaCy NER + NetworkX + D3.js), cross-document conflict detection, similarity matrix, study-path generator, quiz generator, summariser, per-chunk annotations.

**Evaluation** — classical IR metrics (Precision@K, MRR, nDCG) plus a from-scratch RAGAS-style answer evaluator (Faithfulness via atomic-claim entailment, Answer Relevancy via reverse-question similarity, Context Precision via rank-weighted relevance).

---

## Interfaces

- **CLI** — `ingest`, `search`, `chat`, `compare`, `eval`, `info`, `delete`, `clear`
- **Web UI** — Flask app with six blueprints (core, chat, search, study, intelligence, analytics); single-page frontend with twelve views plus templated pages for graph, quizzes, study paths, conflicts, and the evaluation dashboard.
- **Python API** — `KnowledgeBase` and `RAGPipeline` expose the full system.
- **Eval CLI** — `python -m scripts.run_eval --test-set <path.jsonl>`

Security: `secure_filename` on uploads, path-traversal protection on evaluation file paths, 50 MB upload cap, request-timing middleware.

---

## Configuration

Edit `configs/config.yaml`. All fields fall back to defaults in `src/config.py`:

```yaml
embedding:
  model_name: all-MiniLM-L6-v2
chunking:
  strategy: sentence          # fixed | sentence | recursive_hierarchical |
                              # embedding_similarity | density_clustering | topic_modeling
  chunk_size: 512
  chunk_overlap: 50
storage:
  index_path: data/index/default
```

---

## Project structure

```
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── configs/
│   └── config.yaml                  # pipeline configuration
├── data/
│   ├── eval/                        # test sets and evaluation results
│   ├── graph/                       # cached knowledge graph
│   ├── history/                     # persisted query history
│   ├── index/                       # FAISS, BM25, registry, metadata
│   └── memory/                      # persisted conversation memory
├── examples/
│   ├── basic_usage.py
│   └── advanced_usage.py
├── scripts/
│   ├── run_eval.py                  # RAGAS-style answer evaluation CLI
│   └── run_eval_naive.py            # naive-RAG baseline for comparison
├── src/
│   ├── cli.py                       # Click-based command-line interface
│   ├── config.py                    # dataclass config + YAML loader
│   ├── evaluation/                  # IR metrics + RAGAS-style answer eval
│   │   ├── answer_eval.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── ingestion/
│   │   ├── knowledge_base.py        # high-level facade
│   │   ├── document.py
│   │   ├── document_manager.py
│   │   ├── ner_extractor.py         # spaCy NER
│   │   ├── similarity.py
│   │   ├── chunking/                # 6 strategies + manager
│   │   ├── embeddings/              # sentence-transformer service
│   │   ├── parsers/                 # PDF / DOCX / Markdown / TXT
│   │   └── storage/                 # FAISS, BM25, DocumentRegistry
│   ├── knowledge_graph/
│   │   ├── graph_builder.py         # NetworkX graph construction
│   │   └── graph_store.py           # persistence + D3 serialisation
│   ├── rag/
│   │   ├── pipeline.py              # the RAG orchestrator (Mediator)
│   │   ├── llm.py                   # LLMProvider ABC + ClaudeProvider
│   │   ├── memory.py                # persistent conversation memory
│   │   ├── grounding.py             # sentence-level grounding scoring
│   │   ├── fact_verifier.py         # sentence-level fact verification
│   │   ├── cache.py                 # semantic query cache
│   │   ├── conflict_detector.py
│   │   ├── annotations.py
│   │   └── query_history.py
│   ├── retrieval/
│   │   ├── hybrid_search.py         # RRF fusion of BM25 + FAISS
│   │   ├── reranker.py              # cross-encoder reranking
│   │   ├── query_expansion.py       # synonym / multi-query / HyDE
│   │   ├── entity_reranker.py       # spaCy NER entity boosting
│   │   └── evaluation.py            # IR evaluator integration
│   ├── study/
│   │   ├── path_generator.py
│   │   ├── quiz_generator.py
│   │   └── summariser.py
│   ├── utils/
│   │   └── logger.py
│   └── web/
│       ├── app.py                   # Flask application factory
│       ├── blueprints/              # 6 blueprints
│       └── templates/               # 6 HTML templates (1 SPA + 5 standalone)
└── tests/
    ├── conftest.py
    ├── fixtures/                    # sample.docx, sample.md, sample.txt
    ├── integration/                 # full-pipeline integration tests
    └── unit/                        # 28 unit-test modules
```

---

## Testing

```bash
python -m pytest                  # full suite — 512 tests, no API calls
python -m pytest tests/unit/      # fast unit tests only
python -m pytest -m "not slow"    # skip semantic-chunking tests
```

All LLM-dependent tests use a mocked provider, so the suite runs cost-free.

---

## Design notes

Built around standard software-engineering patterns: **Strategy** for chunkers, parsers, and LLM providers; **Factory** for parser and chunker selection; **Mediator** for the RAG pipeline coordinating cache, retrieval, generation, grounding, and verification; **Repository** for annotations, query history, and the document registry. Sentence-transformer (~90 MB), cross-encoder (~130 MB), and spaCy models are all lazy-loaded for sub-second startup. The Flask app uses the application-factory + blueprint pattern so each test gets a clean instance.

---

## Technology stack

Python 3.10+ · Flask · Click · FAISS · `rank-bm25` · `sentence-transformers` · spaCy · NetworkX · NLTK · Anthropic Claude SDK · PyMuPDF · python-docx · HDBSCAN / UMAP / BERTopic · D3.js · Chart.js · pytest.

---

## References

- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS.
- Es et al. (2023), *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, arXiv:2309.15217.
- Gao et al. (2022), *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE), arXiv:2212.10496.
- Cormack et al. (2009), *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR.
- Reimers & Gurevych (2019), *Sentence-BERT*, EMNLP.

---

## Author

Patrick Clerkin — BSc Computing in Software Development, Atlantic Technological University (2026).