# 🚀 Production RAG Engine: AI Engineering Documentation Assistant

A modular, production-grade **Retrieval-Augmented Generation (RAG)** system built without opaque framework bloat. Engineered to provide grounded, hallucination-resistant technical answers with verified source citations.

---

## 📌 Problem & Motivation

Most RAG tutorials rely on heavy abstractions (like standard LangChain/LlamaIndex wrappers) that obscure how chunking boundaries, metadata lineage, vector distance metrics, and prompt constraints actually operate.

This project implements a **clean, transparent, production-ready architecture** from first principles:
- **Clean Domain Models**: Strict Pydantic v2 schemas for Documents, Chunks, SearchResults, Citations, and RAGResponses.
- **Hierarchical Recursive Chunking**: Splits on semantic boundaries (`\n\n` → `\n` → sentences → words) with configurable window overlap and lineage tracking.
- **Zero-Cost Local Embeddings**: Uses `all-MiniLM-L6-v2` via `sentence-transformers` for fast CPU vector embeddings (0 API cost).
- **Decoupled Vector Storage**: ChromaDB persistent store built behind a `BaseVectorStore` interface, ready for drop-in migration to PostgreSQL + `pgvector`.
- **Grounded LLM Generation**: Supports free-tier **Groq** (`llama-3.3-70b-versatile`) and **OpenAI** (`gpt-4o-mini`) with strict citation enforcement `[1]` and anti-hallucination guardrails.

---

## 🏗️ Architecture

```text
  [ Raw Docs: .md, .txt, .pdf ]
                 │
                 ▼
     ┌───────────────────────┐
     │   Document Ingestion  │ ◄── Extracts content, title, page numbers & source metadata
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │  Recursive Chunker    │ ◄── Preserves semantic boundaries & creates chunk IDs
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │   Local Embeddings    │ ◄── all-MiniLM-L6-v2 (384-dim normalized vectors)
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │   Vector Database     │ ◄── ChromaDB (Persistent local HNSW cosine index)
     └───────────┬───────────┘
                 │
                 ▼
        [ User Question ]
                 │
                 ▼
     ┌───────────────────────┐
     │   Vector Retriever    │ ◄── Top-K Cosine Similarity Search + Threshold Filter
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │ Grounded LLM Generator│ ◄── Groq / OpenAI with strict citation formatting
     └───────────┬───────────┘
                 │
                 ▼
  [ Grounded Answer + Explicit Citations [1], [2] + Latency Breakdown ]
```

---

## 📁 Repository Structure

```
├── data/
│   ├── chroma_db/             # Local ChromaDB persistent storage (gitignored)
│   └── docs/                  # Seed AI Engineering Knowledge Base
│       ├── fastapi_architecture.md
│       ├── docker_containerization.md
│       ├── pipecat_pipeline.md
│       └── mcp_protocol.md
├── src/
│   └── rag_engine/
│       ├── config.py          # Pydantic Settings & environment loader
│       ├── models.py          # Core schemas (Document, Chunk, Citation, etc.)
│       ├── pipeline.py        # End-to-end RAG orchestrator
│       ├── chunking/          # BaseChunker & RecursiveCharacterChunker
│       ├── embeddings/        # BaseEmbeddingModel & SentenceTransformerEmbedding
│       ├── generation/        # LLM clients (Groq/OpenAI) & Grounded Prompts
│       ├── ingestion/         # BaseLoader, Text, Markdown & PDF loaders
│       └── vectorstore/       # BaseVectorStore & ChromaVectorStore
├── tests/                     # Comprehensive Pytest suite
├── .env.example               # Environment variables configuration template
├── main.py                    # Interactive Rich CLI application
├── pyproject.toml             # Project build configuration
├── requirements.txt           # Dependency requirements
└── README.md
```

---

## ⚡ Quickstart

### 1. Prerequisites & Installation

Activate your conda environment or create a python virtual environment:

```bash
# Using conda (e.g. voice_agents)
conda activate voice_agents

# Or using standard venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the template and set your preferred LLM provider and API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Choose provider: 'groq' (free tier) or 'openai'
LLM_PROVIDER=groq

# If using Groq (free at https://console.groq.com):
GROQ_API_KEY=gsk_your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# If using OpenAI:
OPENAI_API_KEY=sk-your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## 💻 CLI Usage

### Ingest Documents
Index the seed documentation or any custom folder/file:

```bash
python main.py ingest --path data/docs
```

### Ask Questions (Grounded Retrieval + Citations)
Ask technical questions about your indexed documentation:

```bash
python main.py query "How does a Pipecat pipeline process frames?"
```
```bash
python main.py query "What is the difference between async and sync endpoints in FastAPI?"
```

### Interactive QA Chat
Launch a continuous REPL session:

```bash
python main.py chat
```

### Inspect Database & Model Stats
```bash
python main.py stats
```

### Clear Vector Store
```bash
python main.py clear
```

---

## 🧪 Testing

Run the automated test suite verifying chunking boundaries, embeddings, and vector store operations:

```bash
PYTHONPATH=src pytest tests/ -v
```

---

## 🗺️ Portfolio Evolution Roadmap

- [x] **Phase 1 — MVP Core**: Modular ingestion, recursive chunker, local embeddings, vector DB, grounded LLM generation with citations.
- [ ] **Phase 2 — Backend & Services**: FastAPI asynchronous REST API, PostgreSQL + `pgvector` migration via Docker Compose, Redis caching, streaming SSE responses.
- [ ] **Phase 3 — Advanced RAG**: Hybrid BM25 + dense vector search, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, Query rewriting / Hypothetical Document Embeddings (HyDE).
- [ ] **Phase 4 — Automated Evaluation**: Ragas / TruLens evaluation pipeline measuring Faithfulness, Context Relevance, Answer Correctness, and Latency.
- [ ] **Phase 5 — Observability & Tracing**: OpenTelemetry, OpenInference, and Phoenix/Langfuse tracing for full request inspection.
