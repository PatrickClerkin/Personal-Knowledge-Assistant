"""
Basic usage example for the Personal Knowledge Assistant.

This example demonstrates how to:
1. Ingest a PDF document
2. Search for relevant content
3. Save and load the knowledge base
"""

from pathlib import Path
from src.ingestion import KnowledgeBase

# Initialize knowledge base with persistence
kb = KnowledgeBase(
    index_path="data/index/my_knowledge_base",
    embedding_model="all-MiniLM-L6-v2",  # Fast, good quality
    chunk_strategy="sentence",  # Respects sentence boundaries
    chunk_size=512,
    chunk_overlap=50,
)

print("=" * 70)
print("PERSONAL KNOWLEDGE ASSISTANT - BASIC USAGE")
print("=" * 70)

# Ingest a document
pdf_path = Path("data/test_corpus/aswdCompositionLab2.pdf")
if pdf_path.exists():
    print(f"\n1. Ingesting: {pdf_path.name}")
    num_chunks = kb.ingest(pdf_path, show_progress=True)
    print(f"   Created {num_chunks} chunks")
else:
    print(f"   PDF not found: {pdf_path}")
    print("   Please add a PDF to data/test_corpus/")

# Show statistics
print(f"\n2. Knowledge Base Statistics:")
print(f"   Total chunks: {kb.size}")
print(f"   Documents: {len(kb.document_ids)}")

# Search
if kb.size > 0:
    print("\n3. Searching...")
    query = "What is composition in object-oriented programming?"
    print(f"   Query: '{query}'")

    results = kb.search(query, top_k=3)

    print(f"\n   Top {len(results)} Results:")
    for result in results:
        print(f"\n   Rank {result.rank}: Score {result.score:.3f}")
        print(f"   Source: {result.chunk.get_citation()}")
        print(f"   Preview: {result.chunk.content[:150]}...")

print("\n" + "=" * 70)
print("Done! The index is saved to: data/index/my_knowledge_base")
print("=" * 70)
