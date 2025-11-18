from pathlib import Path
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.chunking.chunk_manager import ChunkManager

# Parse the PDF
pdf_path = Path("data/test_corpus/aswdCompositionLab2.pdf")
parser = PDFParser()
doc = parser.parse(pdf_path)

print("=" * 70)
print("TESTING CHUNKING STRATEGIES")
print("=" * 70)
print(f"\nDocument: {doc.metadata.title}")
print(f"Original length: {len(doc.content):,} characters")
print(f"Total pages: {len(doc.sections)}")

# Test all three strategies
strategies = ["fixed", "sentence", "semantic"]

results = {}

for strategy in strategies:
    print(f"\n{'='*70}")
    print(f"STRATEGY: {strategy.upper()}")
    print(f"{'='*70}")
    
    # Create chunker
    chunker = ChunkManager(strategy=strategy, chunk_size=512, chunk_overlap=50)
    
    # Chunk the document
    chunks = chunker.chunk_document(doc)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    
    # Show statistics
    chunk_sizes = [len(chunk) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    min_size = min(chunk_sizes)
    max_size = max(chunk_sizes)
    
    print(f"\nChunk size statistics:")
    print(f"  Average: {avg_size:.0f} chars")
    print(f"  Min: {min_size} chars")
    print(f"  Max: {max_size} chars")
    
    results[strategy] = {
        'num_chunks': len(chunks),
        'avg_size': avg_size,
        'chunks': chunks
    }
    
    # Show first 2 chunks
    print(f"\nFirst 2 chunks:")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"Page: {chunk.page_number}")
        print(f"Section: {chunk.section_id}")
        print(f"Length: {len(chunk)} chars")
        print(f"Content preview (first 200 chars):")
        print(chunk.content[:200] + "...")

# Final comparison
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

for strategy, data in results.items():
    print(f"\n{strategy.upper()}:")
    print(f"  Chunks: {data['num_chunks']}")
    print(f"  Avg size: {data['avg_size']:.0f} chars")

print("\n" + "-" * 70)
print("RECOMMENDATIONS:")
print("-" * 70)
print("Fixed-size:  Fast, predictable, but cuts mid-sentence")
print("Sentence:    Respects boundaries, good coherence")
print("Semantic:    Best quality - splits on topic changes (RECOMMENDED)")
print("\nFor production RAG: Use SEMANTIC chunking")
print("=" * 70)