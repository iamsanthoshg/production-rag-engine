#!/usr/bin/env python3
"""Main CLI Entrypoint for the Production RAG Engine (AI Engineering Docs Assistant)."""

import argparse
import sys
from pathlib import Path

# Add src/ to python path so rag_engine can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from rag_engine.config import settings
from rag_engine.pipeline import RAGPipeline

console = Console()


def handle_ingest(args):
    """Ingest documents from directory or file."""
    path = Path(args.path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Path not found: {path}")
        sys.exit(1)

    pipeline = RAGPipeline()

    with console.status("[bold green]Ingesting and indexing documents...", spinner="dots"):
        if path.is_file():
            total = pipeline.ingest_file(path)
            console.print(f"[bold green]✓[/bold green] Ingested file '{path.name}' ({total} chunks indexed).")
        else:
            summary = pipeline.ingest_directory(path)
            console.print(
                f"[bold green]✓[/bold green] Ingested {summary['documents_loaded']} documents "
                f"resulting in [bold cyan]{summary['chunks_indexed']}[/bold cyan] indexed chunks."
            )

    display_stats(pipeline)


def handle_query(args):
    """Run a single grounded query through the RAG pipeline."""
    query_text = args.query.strip()
    if not query_text:
        console.print("[bold red]Error:[/bold red] Query cannot be empty.")
        sys.exit(1)

    pipeline = RAGPipeline()

    if pipeline.vector_store.count() == 0:
        console.print(
            "[bold yellow]Warning:[/bold yellow] Vector database is currently empty!\n"
            "Run [bold cyan]python main.py ingest --path data/docs[/bold cyan] first to index documents."
        )
        sys.exit(1)

    with console.status(f"[bold green]Retrieving context & generating answer with {pipeline.llm.model_name}...", spinner="dots"):
        try:
            response = pipeline.query(query_text, top_k=args.top_k)
        except Exception as e:
            console.print(f"[bold red]Generation Error:[/bold red] {e}")
            sys.exit(1)

    # 1. Display Retrieved Context Table
    table = Table(title=f"Retrieved Context Chunks (Top {len(response.retrieved_chunks)})", show_lines=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Source", style="magenta", width=26)
    table.add_column("Score", style="green", width=8)
    table.add_column("Snippet Preview", style="white")

    for idx, item in enumerate(response.retrieved_chunks, start=1):
        source = item.chunk.metadata.get("source", "unknown")
        snippet = item.chunk.content[:160].replace("\n", " ") + "..."
        table.add_row(str(idx), source, f"{item.score:.3f}", snippet)

    console.print(table)
    console.print()

    # 2. Display Grounded Answer
    console.print(
        Panel(
            Markdown(response.answer),
            title=f"[bold green]Grounded Answer[/bold green] ([dim]{response.model_name} | {response.latency_seconds}s[/dim])",
            border_style="green",
        )
    )

    # 3. Display Citations if present
    if response.citations:
        citation_table = Table(title="Document Citations", show_header=True, header_style="bold cyan")
        citation_table.add_column("Ref", width=6)
        citation_table.add_column("Source Document", width=30)
        citation_table.add_column("Matching Context Excerpt")

        for cit in response.citations:
            citation_table.add_row(
                f"[{cit.citation_index}]",
                cit.source,
                cit.snippet,
            )
        console.print(citation_table)
    else:
        console.print("[dim italic]No explicit citation indices mapped in the response.[/dim italic]")


def handle_chat(args):
    """Start an interactive terminal QA loop."""
    pipeline = RAGPipeline()
    if pipeline.vector_store.count() == 0:
        console.print(
            "[bold yellow]Warning:[/bold yellow] Vector database is empty!\n"
            "Run [bold cyan]python main.py ingest --path data/docs[/bold cyan] first."
        )
        sys.exit(1)

    console.print(
        Panel(
            f"[bold cyan]AI Engineering Documentation Assistant[/bold cyan]\n"
            f"Active Model: [green]{pipeline.llm.model_name}[/green] | Embeddings: [yellow]{settings.embedding_model_name}[/yellow]\n"
            f"Indexed chunks: {pipeline.vector_store.count()}\n"
            f"Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to leave.",
            border_style="cyan",
        )
    )

    while True:
        try:
            query_text = console.input("\n[bold cyan]Ask a question > [/bold cyan]").strip()
            if not query_text:
                continue
            if query_text.lower() in ("exit", "quit", "q"):
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break

            with console.status("[bold green]Thinking...", spinner="dots"):
                response = pipeline.query(query_text)

            console.print(
                Panel(
                    Markdown(response.answer),
                    title=f"[bold green]Answer[/bold green] ([dim]{response.latency_seconds}s[/dim])",
                    border_style="green",
                )
            )

            if response.citations:
                cites = ", ".join(f"[{c.citation_index}] {c.source}" for c in response.citations)
                console.print(f"[dim]Sources: {cites}[/dim]")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Session ended.[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


def display_stats(pipeline=None):
    """Display vector database and engine statistics."""
    if pipeline is None:
        pipeline = RAGPipeline()

    stats = pipeline.get_stats()
    table = Table(title="RAG Engine Status & Index Stats", show_header=False)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Vector Store", stats["vector_store"])
    table.add_row("Total Chunks Indexed", str(stats["total_chunks_indexed"]))
    table.add_row("Indexed Sources", ", ".join(stats["sources"]) if stats["sources"] else "(None)")
    table.add_row("Embedding Model", stats["embedding_model"])
    table.add_row("Embedding Dimension", str(stats["embedding_dimension"]))
    table.add_row("Active LLM", stats["active_llm"])

    console.print(table)


def handle_stats(args):
    display_stats()


def handle_clear(args):
    """Clear all indexed documents."""
    pipeline = RAGPipeline()
    pipeline.clear()
    console.print("[bold green]✓[/bold green] Vector store collection cleared.")


def main():
    parser = argparse.ArgumentParser(
        description="Production RAG Engine - AI Engineering Documentation Assistant"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into vector store")
    ingest_parser.add_argument(
        "--path",
        "-p",
        default="data/docs",
        help="Path to file or directory to ingest (default: data/docs)",
    )
    ingest_parser.set_defaults(func=handle_ingest)

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question against indexed docs")
    query_parser.add_argument("query", type=str, help="The question to ask")
    query_parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Number of context chunks to retrieve (default: 4)",
    )
    query_parser.set_defaults(func=handle_query)

    # Interactive chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive QA session")
    chat_parser.set_defaults(func=handle_chat)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Display vector store and engine stats")
    stats_parser.set_defaults(func=handle_stats)

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all vectors from the store")
    clear_parser.set_defaults(func=handle_clear)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
