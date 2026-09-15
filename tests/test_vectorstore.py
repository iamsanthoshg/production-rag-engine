import tempfile
import pytest
from rag_engine.models import Chunk
from rag_engine.vectorstore.chroma import ChromaVectorStore


def test_chroma_vector_store_crud():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = ChromaVectorStore(persist_dir=tmp_dir, collection_name="test_coll")
        assert store.count() == 0

        chunks = [
            Chunk(
                id="c1",
                document_id="d1",
                content="FastAPI dependency injection system.",
                chunk_index=0,
                metadata={"source": "fastapi.md", "topic": "web"},
            ),
            Chunk(
                id="c2",
                document_id="d2",
                content="Docker multi-stage builds reduce image size.",
                chunk_index=0,
                metadata={"source": "docker.md", "topic": "devops"},
            ),
        ]

        # Dummy normalized 3-dim vectors for testing similarity logic
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        store.add_chunks(chunks, embeddings)
        assert store.count() == 2
        assert "fastapi.md" in store.get_sources()
        assert "docker.md" in store.get_sources()

        # Query close to c1
        results = store.similarity_search(query_embedding=[1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].chunk.id == "c1"
        assert results[0].score > results[1].score

        # Test clear
        store.clear()
        assert store.count() == 0
