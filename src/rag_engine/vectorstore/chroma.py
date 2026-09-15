from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from rag_engine.models import Chunk, SearchResult
from rag_engine.vectorstore.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """Local ChromaDB vector store implementation with persistent disk storage."""

    def __init__(self, persist_dir: str = "./data/chroma_db", collection_name: str = "ai_engineering_docs"):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError("chromadb is required. Install via: pip install chromadb")

        self.persist_dir = Path(persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB metadata values must be int, float, str, or bool."""
        clean: Dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (int, float, str, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        ids = [c.id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [self._sanitize_metadata(c.metadata) for c in chunks]

        # Chroma upsert avoids duplicate ID conflicts
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if self.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for cid, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # In cosine space, distance is 1 - cosine_similarity. Thus similarity = 1 - dist
            similarity = max(0.0, 1.0 - float(dist))
            meta_dict = dict(meta) if meta else {}
            doc_id = meta_dict.get("document_id", cid.split("_chunk_")[0] if "_chunk_" in cid else cid)
            chunk_idx = int(meta_dict.get("chunk_index", 0))

            chunk = Chunk(
                id=cid,
                document_id=doc_id,
                content=doc_text,
                chunk_index=chunk_idx,
                metadata=meta_dict,
            )
            search_results.append(SearchResult(chunk=chunk, score=similarity))

        # Sort by similarity score descending
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_sources(self) -> List[str]:
        if self.count() == 0:
            return []
        data = self._collection.get(include=["metadatas"])
        sources = set()
        for m in data["metadatas"] or []:
            if "source" in m:
                sources.add(m["source"])
        return sorted(list(sources))
