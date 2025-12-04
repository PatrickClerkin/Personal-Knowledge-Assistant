from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.chunking.chunk_manager import ChunkManager
import time

print("=" * 70)
print("EMBEDDING YOUR PDF CHUNKS")
print("=" * 70)

# Parse and chunk your PDF
print("\n1. Parsing PDF...")
pdf_path = Path("data/test_corpus/aswdCompositionLab2.pdf")
parser = PDFParser()
doc = parser.parse(pdf_path)

print("\n2. Chunking document...")
chunker = ChunkManager(strategy="sentence", chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_document(doc)
print(f"   Created {len(chunks)} chunks")

# Load embedding model
print("\n3. Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed all chunks
print("\n4. Generating embeddings...")
chunk_texts = [chunk.content for chunk in chunks]

start_time = time.time()
embeddings = model.encode(chunk_texts, show_progress_bar=True)
elapsed = time.time() - start_time

print(f"\n✅ Embeddings generated!")
print(f"   Time: {elapsed:.2f} seconds")
print(f"   Speed: {len(chunks) / elapsed:.1f} chunks/second")
print(f"   Shape: {embeddings.shape}")

# Store embeddings back in chunks
for chunk, embedding in zip(chunks, embeddings):
    chunk.embedding = embedding.tolist()

print("\n5. Testing semantic search...")
query = "What is composition in object-oriented programming?"
print(f"   Query: '{query}'")

# Embed the query
query_embedding = model.encode([query])[0]

# Calculate similarities
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = []
for i, chunk in enumerate(chunks):
    sim = cosine_similarity(query_embedding, np.array(chunk.embedding))
    similarities.append((i, sim, chunk))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

# Show top 3 results
print("\n   Top 3 most relevant chunks:")
for rank, (idx, sim, chunk) in enumerate(similarities[:3], 1):
    print(f"\n   Rank {rank}: Chunk {idx} (similarity: {sim:.3f})")
    print(f"   Page: {chunk.page_number}")
    print(f"   Content preview: {chunk.content[:150]}...")

print("\n" + "=" * 70)
print("SUCCESS! Embeddings are working!")
print("Next: Set up vector database for efficient search")
print("=" * 70)