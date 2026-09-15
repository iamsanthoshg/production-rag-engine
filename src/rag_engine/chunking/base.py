from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
from rag_engine.models import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for chunking documents."""

    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a single Document into a list of Chunks."""
        pass

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split multiple documents into Chunks."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
