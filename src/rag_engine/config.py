from __future__ import annotations

import os
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for RAG engine loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider Configuration
    llm_provider: Literal["groq", "openai"] = Field(
        default="groq",
        alias="LLM_PROVIDER",
        description="Choice of LLM provider: 'groq' or 'openai'",
    )

    # Groq Settings
    groq_api_key: Optional[str] = Field(
        default=None,
        alias="GROQ_API_KEY",
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        alias="GROQ_MODEL",
    )

    # OpenAI Settings
    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_MODEL",
    )

    # Embedding Model (Local zero-cost sentence-transformers)
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )

    # Vector Database Settings
    chroma_persist_dir: str = Field(
        default="./data/chroma_db",
        alias="CHROMA_PERSIST_DIR",
    )
    chroma_collection_name: str = Field(
        default="ai_engineering_docs",
        alias="CHROMA_COLLECTION_NAME",
    )

    # Chunking Hyperparameters
    chunk_size: int = Field(
        default=600,
        alias="CHUNK_SIZE",
        description="Target character size for chunks",
    )
    chunk_overlap: int = Field(
        default=100,
        alias="CHUNK_OVERLAP",
        description="Character overlap between consecutive chunks",
    )

    # Retrieval Hyperparameters
    top_k: int = Field(
        default=4,
        alias="TOP_K",
        description="Number of most similar chunks to retrieve",
    )
    similarity_threshold: float = Field(
        default=0.0,
        alias="SIMILARITY_THRESHOLD",
        description="Minimum cosine similarity score threshold (0.0 to 1.0)",
    )

    @property
    def active_model(self) -> str:
        """Returns the active model name depending on the selected provider."""
        if self.llm_provider == "groq":
            return self.groq_model
        return self.openai_model

    @property
    def active_api_key(self) -> Optional[str]:
        """Returns the active API key depending on the selected provider."""
        if self.llm_provider == "groq":
            return self.groq_api_key
        return self.openai_api_key


# Singleton instance
settings = Settings()
