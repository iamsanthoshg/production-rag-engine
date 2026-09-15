from __future__ import annotations

from typing import List, Optional
from rag_engine.embeddings.base import BaseEmbeddingModel


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Local, open-source embedding model using sentence-transformers (all-MiniLM-L6-v2 by default)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model on first invocation."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install via: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Trigger lazy load
            _ = self.model
        return self._dimension or 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()
