import pytest
from rag_engine.chunking.recursive import RecursiveCharacterChunker
from rag_engine.models import Document


def test_chunker_small_text_single_chunk():
    chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
    doc = Document(
        content="This is a short paragraph that easily fits within 500 characters.",
        metadata={"source": "short.txt"},
    )
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].content == doc.content
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["source"] == "short.txt"


def test_chunker_splits_on_paragraphs():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    p1 = "Paragraph 1 has some text that fills up space nicely."
    p2 = "Paragraph 2 is distinct and should be cleanly separated."
    text = f"{p1}\n\n{p2}"

    doc = Document(content=text, metadata={"source": "paragraphs.md"})
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    # Check that chunks preserve source metadata
    assert all(c.source == "paragraphs.md" for c in chunks)


def test_chunker_empty_document():
    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
    doc = Document(content="   \n\n  ", metadata={"source": "empty.md"})
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 0


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)
