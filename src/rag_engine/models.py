from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a raw document loaded from a source file."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


class Chunk(BaseModel):
    """Represents a discrete, embeddable piece of a document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


class SearchResult(BaseModel):
    """Represents a retrieved chunk with its vector similarity score."""

    chunk: Chunk
    score: float = Field(
        ...,
        description="Similarity score where higher is more relevant (typically cosine similarity between 0 and 1)",
    )


class Citation(BaseModel):
    """Represents an explicit reference cited by the LLM in its grounded answer."""

    citation_index: int
    source: str
    chunk_id: str
    snippet: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """The complete response from the RAG pipeline."""

    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[SearchResult] = Field(default_factory=list)
    latency_seconds: float
    model_name: str
