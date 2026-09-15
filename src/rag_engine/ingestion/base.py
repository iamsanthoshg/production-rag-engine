from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from rag_engine.models import Document


class BaseLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    def load(self, file_path: Path | str) -> List[Document]:
        """Loads content from file and returns a list of Document objects."""
        pass
