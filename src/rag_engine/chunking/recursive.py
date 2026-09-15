from __future__ import annotations

import re
from typing import List, Optional
from rag_engine.chunking.base import BaseChunker
from rag_engine.models import Chunk, Document


class RecursiveCharacterChunker(BaseChunker):
    """Recursively splits text using semantic separators (paragraphs, newlines, sentences, spaces).

    Maintains chunk overlap across boundaries and attaches rich lineage metadata.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Splits text using the first separator that works, recursing if any resulting segment exceeds chunk_size."""
        final_chunks: List[str] = []

        # Find the appropriate separator
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1 :]
                break

        # Split text by chosen separator
        splits = text.split(separator) if separator != "" else list(text)

        # Merge splits up to chunk_size
        good_splits: List[str] = []
        for s in splits:
            if not s.strip():
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merges small segments together up to chunk_size, applying chunk_overlap between windows."""
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for d in splits:
            _len = len(d)
            if total + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                if total > 0:
                    doc_str = separator.join(current_doc)
                    if doc_str.strip():
                        docs.append(doc_str.strip())

                    # Apply overlap: keep pieces from the tail of current_doc
                    while total > self.chunk_overlap or (
                        total + _len + (len(separator) if current_doc else 0) > self.chunk_size and total > 0
                    ):
                        if not current_doc:
                            break
                        removed = current_doc.pop(0)
                        total -= len(removed) + (len(separator) if current_doc else 0)

            current_doc.append(d)
            total += _len + (len(separator) if len(current_doc) > 1 else 0)

        doc_str = separator.join(current_doc)
        if doc_str.strip():
            docs.append(doc_str.strip())

        return docs

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Splits a Document into Chunks while preserving metadata and document lineage."""
        raw_text = document.content.strip()
        if not raw_text:
            return []

        text_chunks = self._split_text(raw_text, self.separators)
        chunks: List[Chunk] = []

        for idx, text in enumerate(text_chunks):
            chunk_metadata = dict(document.metadata)
            chunk_metadata.update(
                {
                    "chunk_index": idx,
                    "char_count": len(text),
                }
            )

            chunk = Chunk(
                id=f"{document.id}_chunk_{idx}",
                document_id=document.id,
                content=text,
                chunk_index=idx,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)

        return chunks
