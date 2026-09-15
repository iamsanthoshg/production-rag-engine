from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Type
from rag_engine.ingestion.base import BaseLoader
from rag_engine.ingestion.markdown import MarkdownLoader
from rag_engine.ingestion.pdf import PDFLoader
from rag_engine.ingestion.text import TextLoader
from rag_engine.models import Document


LOADER_MAPPING: Dict[str, Type[BaseLoader]] = {
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".txt": TextLoader,
    ".pdf": PDFLoader,
}


class DocumentLoader:
    """Dispatches files to appropriate loaders based on extension."""

    @classmethod
    def load_file(cls, file_path: Path | str) -> List[Document]:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        loader_cls = LOADER_MAPPING.get(suffix)
        if not loader_cls:
            raise ValueError(f"Unsupported file extension '{suffix}'. Supported: {list(LOADER_MAPPING.keys())}")

        loader = loader_cls()
        return loader.load(path)

    @classmethod
    def load_directory(
        cls,
        dir_path: Path | str,
        recursive: bool = True,
        extensions: List[str] | None = None,
    ) -> List[Document]:
        folder = Path(dir_path).resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"Directory not found: {folder}")

        valid_exts = [e.lower() for e in extensions] if extensions else list(LOADER_MAPPING.keys())
        all_docs: List[Document] = []

        glob_pattern = "**/*" if recursive else "*"
        for path in sorted(folder.glob(glob_pattern)):
            if path.is_file() and path.suffix.lower() in valid_exts:
                docs = cls.load_file(path)
                all_docs.extend(docs)

        return all_docs
