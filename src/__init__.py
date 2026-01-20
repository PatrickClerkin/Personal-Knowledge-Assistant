"""Personal Knowledge Assistant - Document ingestion and semantic search."""

from .config import Config, get_config, set_config

__version__ = "0.1.0"

__all__ = [
    "Config",
    "get_config",
    "set_config",
]
