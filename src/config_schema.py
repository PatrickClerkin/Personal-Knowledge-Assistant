"""
Application configuration with Pydantic validation.

Replaces the raw YAML dict with typed, validated configuration
models. Provides defaults, type checking, and value constraints
to catch configuration errors early.

Usage:
    config = AppConfig.from_yaml("configs/config.yaml")
    config = AppConfig()  # Uses defaults
"""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding service."""
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model identifier.",
    )
    batch_size: int = Field(
        default=32, ge=1, le=512,
        description="Batch size for embedding generation.",
    )
    show_progress: bool = Field(
        default=True,
        description="Show progress bar during embedding.",
    )


class ChunkingConfig(BaseModel):
    """Configuration for the chunking pipeline."""
    strategy: Literal[
        "fixed", "sentence", "embedding_similarity",
        "density_clustering", "topic_modeling",
        "recursive_hierarchical", "auto",
    ] = Field(
        default="sentence",
        description="Chunking strategy to use.",
    )
    chunk_size: int = Field(
        default=512, ge=50, le=10000,
        description="Target chunk size in characters.",
    )
    chunk_overlap: int = Field(
        default=50, ge=0, le=500,
        description="Character overlap between adjacent chunks.",
    )
    similarity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Similarity threshold for semantic chunking.",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v, info):
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


class StorageConfig(BaseModel):
    """Configuration for vector storage."""
    backend: Literal["faiss"] = Field(
        default="faiss",
        description="Vector store backend.",
    )
    index_path: str = Field(
        default="data/index/default",
        description="Path for persisting the FAISS index.",
    )
    auto_save: bool = Field(
        default=True,
        description="Automatically save after ingestion.",
    )


class RetrievalConfig(BaseModel):
    """Configuration for retrieval enhancements."""
    rerank: bool = Field(
        default=False,
        description="Enable cross-encoder reranking.",
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking.",
    )
    rerank_candidates: int = Field(
        default=20, ge=5, le=100,
        description="Number of candidates to retrieve before reranking.",
    )
    query_expansion: Optional[Literal[
        "synonym", "multi_query", "hyde"
    ]] = Field(
        default=None,
        description="Query expansion strategy (None to disable).",
    )


class WebConfig(BaseModel):
    """Configuration for the web interface."""
    host: str = Field(default="0.0.0.0", description="Server host.")
    port: int = Field(default=5000, ge=1024, le=65535, description="Server port.")
    debug: bool = Field(default=False, description="Flask debug mode.")
    max_upload_mb: int = Field(default=50, ge=1, le=500, description="Max upload size in MB.")


class AppConfig(BaseModel):
    """Root application configuration.

    Aggregates all sub-configurations and provides factory
    methods for loading from YAML or environment variables.
    """
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    data_dir: str = Field(default="data", description="Root data directory.")
    corpus_dir: str = Field(default="data/test_corpus", description="Test corpus path.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load configuration from a YAML file.

        Missing keys use defaults. Extra keys are ignored.
        Invalid values raise ValidationError with details.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(
                "Config file not found: %s. Using defaults.", path
            )
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        config = cls(**raw)
        logger.info("Loaded configuration from %s", path)
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(), f,
                default_flow_style=False, sort_keys=False,
            )
        logger.info("Saved configuration to %s", path)
