"""Integration tests for the full ingestion pipeline."""

import pytest
from pathlib import Path
from src.ingestion import (
    PDFParser,
    ChunkManager,
    EmbeddingService,
    FAISSVectorStore,
)


class TestIngestionPipeline:
    """Integration tests for the full document ingestion pipeline."""

    @pytest.fixture
    def pipeline_components(self):
        """Create all pipeline components."""
        return {
            "parser": PDFParser(),
            "chunker": ChunkManager(strategy="sentence", chunk_size=512, chunk_overlap=50),
            "embedder": EmbeddingService(),
            "store": FAISSVectorStore(embedding_dim=384),
        }

    def test_full_pipeline(self, sample_pdf_path, pipeline_components):
        """Test the complete ingestion pipeline."""
        parser = pipeline_components["parser"]
        chunker = pipeline_components["chunker"]
        embedder = pipeline_components["embedder"]
        store = pipeline_components["store"]

        # 1. Parse document
        document = parser.parse(sample_pdf_path)
        assert document is not None
        assert len(document.content) > 0

        # 2. Chunk document
        chunks = chunker.chunk_document(document)
        assert len(chunks) > 0

        # 3. Generate embeddings
        embeddings = embedder.embed_chunks(chunks, store_in_chunks=True)
        assert embeddings.shape[0] == len(chunks)
        assert all(c.has_embedding for c in chunks)

        # 4. Store in vector database
        store.add(chunks, embeddings)
        assert store.size == len(chunks)

        # 5. Search
        query = "What is composition in programming?"
        query_embedding = embedder.embed_text(query)
        results = store.search(query_embedding, top_k=3)

        assert len(results) == 3
        for result in results:
            assert result.chunk is not None
            assert result.score > 0

    def test_pipeline_with_save_load(self, sample_pdf_path, pipeline_components, tmp_path):
        """Test pipeline with saving and loading the index."""
        parser = pipeline_components["parser"]
        chunker = pipeline_components["chunker"]
        embedder = pipeline_components["embedder"]
        store = pipeline_components["store"]

        # Ingest document
        document = parser.parse(sample_pdf_path)
        chunks = chunker.chunk_document(document)
        embeddings = embedder.embed_chunks(chunks)
        store.add(chunks, embeddings)

        # Save index
        index_path = tmp_path / "test_index"
        store.save(str(index_path))

        # Load into new store
        new_store = FAISSVectorStore(embedding_dim=384)
        new_store.load(str(index_path))

        # Verify search still works
        query_embedding = embedder.embed_text("programming concepts")
        results = new_store.search(query_embedding, top_k=2)

        assert len(results) == 2
