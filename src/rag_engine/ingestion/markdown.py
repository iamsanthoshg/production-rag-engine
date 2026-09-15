from __future__ import annotations

import re
from pathlib import Path
from typing import List
from rag_engine.ingestion.base import BaseLoader
from rag_engine.models import Document


class MarkdownLoader(BaseLoader):
    """Loader for Markdown (.md) documents with header metadata extraction."""

    def load(self, file_path: Path | str) -> List[Document]:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Extract primary title (# Title) if available
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem.replace("_", " ").title()

        return [
            Document(
                content=content,
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "file_type": "markdown",
                    "title": title,
                    "char_count": len(content),
                },
            )
        ]
