import pytest
from rag_engine.embeddings.local import SentenceTransformerEmbedding


def test_embedding_dimension():
    embedder = SentenceTransformerEmbedding()
    assert embedder.dimension == 384


def test_embed_documents_and_query():
    embedder = SentenceTransformerEmbedding()
    docs = [
        "FastAPI is a modern web framework for building APIs with Python.",
        "Docker provides containerization and isolation for applications.",
    ]
    embeddings = embedder.embed_documents(docs)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    query_emb = embedder.embed_query("Python web framework")
    assert len(query_emb) == 384
