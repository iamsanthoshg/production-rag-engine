from __future__ import annotations

from pathlib import Path
from typing import List
from rag_engine.ingestion.base import BaseLoader
from rag_engine.models import Document


class PDFLoader(BaseLoader):
    """Loader for PDF documents using pypdf, extracting text with page-level metadata."""

    def load(self, file_path: Path | str) -> List[Document]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required to load PDF documents. Install via: pip install pypdf")

        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        reader = PdfReader(str(path))
        documents: List[Document] = []

        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            documents.append(
                Document(
                    content=text,
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "file_type": "pdf",
                        "page_number": idx,
                        "total_pages": len(reader.pages),
                        "char_count": len(text),
                    },
                )
            )

        return documents
