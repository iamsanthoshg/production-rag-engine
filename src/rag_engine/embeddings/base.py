from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingModel(ABC):
    """Abstract base class for generating vector embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the embedding vector dimension."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute vector embeddings for a list of document texts."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Compute vector embedding for a single user query."""
        pass
