from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
    """Abstract interface for LLM generation clients."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the active model name."""
        pass

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Generates text from system and user prompts."""
        pass
