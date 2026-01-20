from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding service."""
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    show_progress: bool = True


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    strategy: str = "sentence"  # "fixed", "sentence", or "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_threshold: float = 0.5  # For semantic chunking


@dataclass
class StorageConfig:
    """Configuration for vector storage."""
    backend: str = "faiss"  # Currently only "faiss" supported
    index_path: Path = field(default_factory=lambda: Path("data/index"))
    auto_save: bool = True


@dataclass
class Config:
    """Main configuration class for the Personal Knowledge Assistant."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    corpus_dir: Path = field(default_factory=lambda: Path("data/test_corpus"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create Config from a dictionary."""
        embedding_data = data.get("embedding", {})
        chunking_data = data.get("chunking", {})
        storage_data = data.get("storage", {})

        # Convert storage index_path to Path if present
        if "index_path" in storage_data:
            storage_data["index_path"] = Path(storage_data["index_path"])

        return cls(
            embedding=EmbeddingConfig(**embedding_data),
            chunking=ChunkingConfig(**chunking_data),
            storage=StorageConfig(**storage_data),
            data_dir=Path(data.get("data_dir", "data")),
            corpus_dir=Path(data.get("corpus_dir", "data/test_corpus")),
            processed_dir=Path(data.get("processed_dir", "data/processed")),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "embedding": {
                "model_name": self.embedding.model_name,
                "batch_size": self.embedding.batch_size,
                "show_progress": self.embedding.show_progress,
            },
            "chunking": {
                "strategy": self.chunking.strategy,
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
                "similarity_threshold": self.chunking.similarity_threshold,
            },
            "storage": {
                "backend": self.storage.backend,
                "index_path": str(self.storage.index_path),
                "auto_save": self.storage.auto_save,
            },
            "data_dir": str(self.data_dir),
            "corpus_dir": str(self.corpus_dir),
            "processed_dir": str(self.processed_dir),
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Global default config instance
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _default_config
    if _default_config is None:
        # Try to load from default location
        config_path = Path("configs/config.yaml")
        _default_config = Config.from_yaml(config_path)
    return _default_config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _default_config
    _default_config = config
