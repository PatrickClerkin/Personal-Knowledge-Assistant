from pathlib import Path
from src.ingestion.parsers.pdf_parser import PDFParser

# Parse your specific PDF
pdf_path = Path("data/test_corpus/aswdCompositionLab2.pdf")
parser = PDFParser()

print("Parsing aswdCompositionLab2.pdf...")
print("=" * 70)

try:
    doc = parser.parse(pdf_path)
    
    # Print metadata
    print("\n📄 METADATA")
    print("-" * 70)
    print(f"Title:       {doc.metadata.title}")
    print(f"Author:      {doc.metadata.author or 'Not specified'}")
    print(f"File type:   {doc.metadata.file_type}")
    print(f"Doc ID:      {doc.metadata.doc_id}")
    print(f"Created:     {doc.metadata.created_date}")
    print(f"Modified:    {doc.metadata.modified_date}")
    
    # Print statistics
    print("\n📊 STATISTICS")
    print("-" * 70)
    print(f"Total characters:  {len(doc.content):,}")
    print(f"Total pages:       {len(doc.sections)}")
    print(f"Total words:       {len(doc.content.split()):,}")
    print(f"Avg chars/page:    {len(doc.content) // len(doc.sections):,}")
    
    # Show page breakdown
    print("\n📑 PAGE BREAKDOWN")
    print("-" * 70)
    for section in doc.sections:
        word_count = len(section.content.split())
        print(f"{section.title}: {len(section.content):,} chars, {word_count:,} words")
    
    # Show first page preview
    print("\n📖 FIRST PAGE PREVIEW (first 500 characters)")
    print("-" * 70)
    print(doc.sections[0].content[:500])
    if len(doc.sections[0].content) > 500:
        print("...")
    
    # Diagnostic checks
    print("\n🔍 DIAGNOSTIC CHECKS")
    print("-" * 70)
    
    if len(doc.content.strip()) < 100:
        print("⚠️  WARNING: Very little text extracted!")
        print("   This might be a scanned/image PDF needing OCR")
    else:
        print("✅ Text extraction successful")
    
    non_ascii = sum(1 for c in doc.content if ord(c) > 127)
    if non_ascii > 0:
        print(f"ℹ️  Contains {non_ascii:,} non-ASCII characters")
    
    if "|" in doc.content or "table" in doc.content.lower():
        print("ℹ️  Document may contain tables")
    
    # Save extracted text
    output_path = Path("data/processed/aswdCompositionLab2_extracted.txt")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"EXTRACTED FROM: {doc.metadata.title}\n")
        f.write("=" * 70 + "\n\n")
        f.write(doc.content)
    
    print(f"\n💾 Full extracted text saved to: {output_path}")
    print("\n✅ SUCCESS: PDF parsed successfully!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()