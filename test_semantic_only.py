from pathlib import Path
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.chunking.chunk_manager import ChunkManager

# Parse the PDF
pdf_path = Path("data/test_corpus/aswdCompositionLab2.pdf")
parser = PDFParser()
doc = parser.parse(pdf_path)

print("Testing SEMANTIC (Structure-Based) Chunking")
print("=" * 70)
print(f"Document: {doc.metadata.title}")
print(f"Pages: {len(doc.sections)}")
print(f"Total chars: {len(doc.content):,}")
print()

# Create semantic chunker
chunker = ChunkManager(strategy="semantic", chunk_size=512, chunk_overlap=50)

# Chunk the document
chunks = chunker.chunk_document(doc)

print(f"✅ Chunking complete!")
print(f"Total chunks created: {len(chunks)}")

# Show statistics
chunk_sizes = [len(chunk) for chunk in chunks]
avg_size = sum(chunk_sizes) / len(chunk_sizes)

print(f"\nChunk size statistics:")
print(f"  Average: {avg_size:.0f} chars")
print(f"  Min: {min(chunk_sizes)} chars")
print(f"  Max: {max(chunk_sizes)} chars")

# Show all chunks
print(f"\nAll chunks:")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(f"Page: {chunk.page_number}")
    print(f"Section: {chunk.section_id}")
    print(f"Length: {len(chunk)} chars")
    print(f"Content preview (first 150 chars):")
    print(chunk.content[:150] + "...")

print("\n" + "=" * 70)
print("NOTE: This uses structure-based chunking (by pages/sections)")
print("True semantic chunking will be added in Week 2 with embeddings")
print("=" * 70)