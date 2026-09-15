from __future__ import annotations

from pathlib import Path
from typing import List
from rag_engine.ingestion.base import BaseLoader
from rag_engine.models import Document


class TextLoader(BaseLoader):
    """Loader for plain text (.txt) files."""

    def load(self, file_path: Path | str) -> List[Document]:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return [
            Document(
                content=content,
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "file_type": "text",
                    "char_count": len(content),
                },
            )
        ]
