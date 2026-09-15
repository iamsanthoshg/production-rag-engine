from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_engine.chunking.base import BaseChunker
from rag_engine.chunking.recursive import RecursiveCharacterChunker
from rag_engine.config import settings
from rag_engine.embeddings.base import BaseEmbeddingModel
from rag_engine.embeddings.local import SentenceTransformerEmbedding
from rag_engine.generation.base import BaseLLMClient
from rag_engine.generation.llm import UnifiedLLMClient
from rag_engine.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from rag_engine.ingestion.loader import DocumentLoader
from rag_engine.models import Citation, Document, RAGResponse, SearchResult
from rag_engine.vectorstore.base import BaseVectorStore
from rag_engine.vectorstore.chroma import ChromaVectorStore


class RAGPipeline:
    """Production-grade RAG orchestrator integrating Ingestion, Chunking, Embeddings, Vector Search, and Grounded Generation."""

    def __init__(
        self,
        chunker: Optional[BaseChunker] = None,
        embedder: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        llm: Optional[BaseLLMClient] = None,
    ):
        self.chunker = chunker or RecursiveCharacterChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedder = embedder or SentenceTransformerEmbedding(
            model_name=settings.embedding_model_name,
        )
        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        self.llm = llm or UnifiedLLMClient()

    def ingest_documents(self, documents: List[Document]) -> int:
        """Chunks, embeds, and stores a list of Documents. Returns total chunks indexed."""
        if not documents:
            return 0

        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            return 0

        texts_to_embed = [c.content for c in chunks]
        embeddings = self.embedder.embed_documents(texts_to_embed)

        self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)
        return len(chunks)

    def ingest_file(self, file_path: str | Path) -> int:
        """Loads and indexes a single document file."""
        docs = DocumentLoader.load_file(file_path)
        return self.ingest_documents(docs)

    def ingest_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Loads and indexes all supported files in a directory."""
        docs = DocumentLoader.load_directory(
            dir_path=dir_path,
            recursive=recursive,
            extensions=extensions,
        )
        total_chunks = self.ingest_documents(docs)
        return {"documents_loaded": len(docs), "chunks_indexed": total_chunks}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Embeds query and retrieves top-k most relevant chunks from the vector store."""
        k = top_k or settings.top_k
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=k,
            where=where,
        )
        # Apply similarity threshold filter
        threshold = settings.similarity_threshold
        filtered = [r for r in results if r.score >= threshold]
        return filtered

    def _extract_citations(
        self, answer: str, retrieved_chunks: List[SearchResult]
    ) -> List[Citation]:
        """Finds citation markers like [1], [2] in answer and maps them back to the source chunks."""
        citations: List[Citation] = []
        found_indices = set(re.findall(r"\[(\d+)\]", answer))

        for idx_str in sorted(found_indices, key=int):
            idx = int(idx_str)
            if 1 <= idx <= len(retrieved_chunks):
                match_result = retrieved_chunks[idx - 1]
                chunk = match_result.chunk
                snippet = chunk.content[:150].replace("\n", " ") + "..."
                citations.append(
                    Citation(
                        citation_index=idx,
                        source=chunk.metadata.get("source", "unknown"),
                        chunk_id=chunk.id,
                        snippet=snippet,
                        metadata=chunk.metadata,
                    )
                )

        return citations

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: float = 0.1,
    ) -> RAGResponse:
        """Executes full RAG workflow: Retrieve → Format Prompt → Generate Answer → Parse Citations."""
        start_time = time.time()

        # Step 1: Retrieval
        retrieved_chunks = self.retrieve(question, top_k=top_k)

        # Step 2: Prompt assembly
        user_prompt = build_user_prompt(question, retrieved_chunks)

        # Step 3: LLM generation
        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        total_latency = time.time() - start_time

        # Step 4: Citation extraction
        citations = self._extract_citations(answer, retrieved_chunks)

        return RAGResponse(
            query=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            latency_seconds=round(total_latency, 3),
            model_name=self.llm.model_name,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the current indexed state."""
        return {
            "total_chunks_indexed": self.vector_store.count(),
            "sources": self.vector_store.get_sources(),
            "embedding_dimension": self.embedder.dimension,
            "embedding_model": settings.embedding_model_name,
            "active_llm": self.llm.model_name,
            "vector_store": self.vector_store.__class__.__name__,
        }

    def clear(self) -> None:
        """Clears the vector store index."""
        self.vector_store.clear()
