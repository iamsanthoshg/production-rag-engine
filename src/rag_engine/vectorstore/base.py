from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from rag_engine.models import Chunk, SearchResult


class BaseVectorStore(ABC):
    """Abstract interface for vector database implementations (ChromaDB, PGVector, Qdrant, etc.)."""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add chunks and their corresponding embeddings into the vector index."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform similarity search and return top-k SearchResult items sorted by score."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in the collection."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all vectors in the collection."""
        pass

    @abstractmethod
    def get_sources(self) -> List[str]:
        """Return a list of unique document sources currently indexed."""
        pass
