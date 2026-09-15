import pytest
from rag_engine.models import Chunk, Citation, Document, RAGResponse, SearchResult


def test_document_creation():
    doc = Document(content="Hello world", metadata={"source": "test.md"})
    assert doc.content == "Hello world"
    assert doc.source == "test.md"
    assert doc.id is not None


def test_chunk_creation():
    chunk = Chunk(
        document_id="doc-123",
        content="First chunk content",
        chunk_index=0,
        metadata={"source": "test.md", "page_number": 1},
    )
    assert chunk.document_id == "doc-123"
    assert chunk.source == "test.md"
    assert chunk.chunk_index == 0


def test_rag_response_serialization():
    chunk = Chunk(
        document_id="doc-1",
        content="Pipecat processes frames sequentially.",
        chunk_index=0,
        metadata={"source": "pipecat.md"},
    )
    res = SearchResult(chunk=chunk, score=0.88)
    citation = Citation(
        citation_index=1,
        source="pipecat.md",
        chunk_id=chunk.id,
        snippet=chunk.content[:50],
    )
    response = RAGResponse(
        query="How does Pipecat process data?",
        answer="Pipecat processes data through a pipeline of frames [1].",
        citations=[citation],
        retrieved_chunks=[res],
        latency_seconds=0.45,
        model_name="groq/llama-3.3-70b-versatile",
    )

    data = response.model_dump()
    assert data["query"] == "How does Pipecat process data?"
    assert len(data["citations"]) == 1
    assert data["citations"][0]["citation_index"] == 1
    assert data["retrieved_chunks"][0]["score"] == 0.88
