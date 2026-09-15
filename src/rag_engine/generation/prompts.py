from __future__ import annotations

from typing import List
from rag_engine.models import SearchResult


SYSTEM_PROMPT = """You are an expert AI Engineering Technical Assistant.
Your mission is to provide accurate, concise, and technically rigorous answers based SOLELY on the provided documentation context.

CRITICAL RULES:
1. GROUNDING: Answer ONLY using facts directly stated in the provided context snippets below. Do NOT use outside knowledge or extrapolate beyond the text.
2. CITATIONS: Whenever you state a claim or fact, cite the context source using bracketed numbers like [1], [2] immediately following the statement. Every key point must have at least one citation.
3. INSUFFICIENT CONTEXT: If the provided context does not contain enough information to answer the question truthfully and completely, you MUST reply:
   "I cannot find sufficient information in the provided documentation to answer this question."
   Do NOT attempt to guess, hallucinate, or answer from general knowledge.
4. TONE: Direct, authoritative, and developer-friendly. Use code blocks when describing code patterns mentioned in the context.
"""


def format_context_for_prompt(retrieved_chunks: List[SearchResult]) -> str:
    """Formats retrieved chunks into numbered context blocks for the prompt."""
    if not retrieved_chunks:
        return "No relevant documentation found."

    formatted_blocks = []
    for idx, item in enumerate(retrieved_chunks, start=1):
        chunk = item.chunk
        source_name = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page_number")
        page_str = f", Page {page}" if page else ""
        section = chunk.metadata.get("title") or chunk.metadata.get("section")
        section_str = f" - {section}" if section else ""

        block = (
            f"--- [Context {idx}] ---\n"
            f"Source: {source_name}{section_str}{page_str}\n"
            f"Relevance Score: {item.score:.3f}\n"
            f"Content:\n{chunk.content}\n"
        )
        formatted_blocks.append(block)

    return "\n".join(formatted_blocks)


def build_user_prompt(query: str, retrieved_chunks: List[SearchResult]) -> str:
    """Combines user question with formatted context."""
    context_str = format_context_for_prompt(retrieved_chunks)
    return (
        f"Context Snippets:\n"
        f"================\n"
        f"{context_str}\n"
        f"================\n\n"
        f"User Question: {query}\n\n"
        f"Please provide your grounded answer with citations [1], [2], etc.:"
    )
