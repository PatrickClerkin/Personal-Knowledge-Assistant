"""
Advanced usage example showing low-level components.

This example demonstrates:
1. Using individual components (parser, chunker, embedder, store)
2. Comparing different chunking strategies
3. Direct vector similarity operations
"""

from pathlib import Path
import numpy as np

from src.ingestion import (
    PDFParser,
    ChunkManager,
    EmbeddingService,
    FAISSVectorStore,
)

print("=" * 70)
print("ADVANCED USAGE - INDIVIDUAL COMPONENTS")
print("=" * 70)

# 1. Parse a document
print("\n1. PARSING DOCUMENT")
print("-" * 40)

parser = PDFParser()
pdf_path = Path("data/test_corpus/aswdCompositionLab2.pdf")

if not pdf_path.exists():
    print(f"PDF not found: {pdf_path}")
    exit(1)

document = parser.parse(pdf_path)
print(f"Title: {document.metadata.title}")
print(f"Pages: {len(document.sections)}")
print(f"Total characters: {len(document.content):,}")

# 2. Compare chunking strategies
print("\n2. COMPARING CHUNKING STRATEGIES")
print("-" * 40)

strategies = ["fixed", "sentence", "semantic"]
for strategy in strategies:
    chunker = ChunkManager(strategy=strategy, chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_document(document)

    sizes = [len(c.content) for c in chunks]
    avg_size = sum(sizes) / len(sizes)

    print(f"\n{strategy.upper()}:")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Avg size: {avg_size:.0f} chars")
    print(f"  Range: {min(sizes)}-{max(sizes)} chars")

# 3. Generate embeddings
print("\n3. GENERATING EMBEDDINGS")
print("-" * 40)

chunker = ChunkManager(strategy="sentence", chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_document(document)

embedder = EmbeddingService()
print(f"Model: {embedder.model_name}")
print(f"Embedding dimension: {embedder.embedding_dimension}")

embeddings = embedder.embed_chunks(chunks, show_progress=True)
print(f"Generated {embeddings.shape[0]} embeddings")

# 4. Manual similarity search
print("\n4. MANUAL SIMILARITY SEARCH")
print("-" * 40)

query = "What is the difference between composition and inheritance?"
query_embedding = embedder.embed_text(query)

# Compute similarities
similarities = []
for i, chunk in enumerate(chunks):
    sim = embedder.compute_similarity(query_embedding, embeddings[i])
    similarities.append((i, sim, chunk))

# Sort and show top results
similarities.sort(key=lambda x: x[1], reverse=True)

print(f"Query: '{query}'")
print("\nTop 3 matches:")
for idx, sim, chunk in similarities[:3]:
    print(f"\n  Score: {sim:.3f}")
    print(f"  Page: {chunk.page_number}")
    print(f"  Content: {chunk.content[:100]}...")

# 5. Using FAISS store directly
print("\n5. FAISS VECTOR STORE")
print("-" * 40)

store = FAISSVectorStore(embedding_dim=embedder.embedding_dimension)
store.add(chunks, embeddings)

print(f"Store size: {store.size} vectors")
print(f"Documents: {store.get_document_ids()}")

# Search with the store
results = store.search(query_embedding, top_k=3)
print(f"\nSearch results from store:")
for result in results:
    print(f"  Rank {result.rank}: {result.score:.3f} - {result.chunk.content[:50]}...")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
