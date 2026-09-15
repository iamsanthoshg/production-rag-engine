from __future__ import annotations

import os
from typing import Optional
from rag_engine.config import settings
from rag_engine.generation.base import BaseLLMClient


class UnifiedLLMClient(BaseLLMClient):
    """Unified client supporting both Groq (free tier) and OpenAI (gpt-4o-mini)."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = (provider or settings.llm_provider).lower()

        # Auto-detect provider if default groq has no key but openai has a key
        if self.provider == "groq" and not (api_key or settings.groq_api_key) and settings.openai_api_key:
            self.provider = "openai"
        elif self.provider == "openai" and not (api_key or settings.openai_api_key) and settings.groq_api_key:
            self.provider = "groq"

        if self.provider == "groq":
            self._model_name = model_name or settings.groq_model
            self.api_key = api_key or settings.groq_api_key
        elif self.provider == "openai":
            self._model_name = model_name or settings.openai_model
            self.api_key = api_key or settings.openai_api_key
        else:
            raise ValueError(f"Unsupported LLM provider: '{self.provider}'. Must be 'groq' or 'openai'.")

        self._client = None

    @property
    def model_name(self) -> str:
        return f"{self.provider}/{self._model_name}"

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self.api_key:
            if self.provider == "groq":
                raise ValueError(
                    "GROQ_API_KEY is not set. Please add GROQ_API_KEY to your .env file. "
                    "You can get a free key in 30 seconds at: https://console.groq.com"
                )
            else:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Please add OPENAI_API_KEY to your .env file."
                )

        if self.provider == "groq":
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("groq package is required. Install via: pip install groq")
            self._client = Groq(api_key=self.api_key)
        else:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package is required. Install via: pip install openai")
            self._client = OpenAI(api_key=self.api_key)

        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        client = self._get_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.provider == "groq":
            response = client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return response.choices[0].message.content or ""
