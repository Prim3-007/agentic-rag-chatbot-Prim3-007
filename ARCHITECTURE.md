# Architecture Overview

## Goal
A production-ready Agentic RAG Chatbot built by the Elite AI Engineering squad.
Features **Hybrid RAG** (Dense+Sparse+RRF+Rerank), **Stateful Memory** (Thread+Cross-Thread), and **Safe Tooling**.

## High-Level Flow

### 1) Unified Ingestion & Retrieval (Feature A)
- **Module**: `src/ingest.py` (Consolidated Pipeline)
- **Ingestion**:
  - Loaders: `PyPDFLoader`, `TextLoader`, `UnstructuredMarkdownLoader`.
  - **Chunking**: `RecursiveCharacterTextSplitter` (1000/200) with preserved source metadata.
- **Indexing**:
  - **Dense**: FAISS (HuggingFace `all-MiniLM-L6-v2`).
  - **Sparse**: BM25 (Rank-BM25).
- **Retrieval (`hybrid_search`)**:
  - **Hybrid Search**: Fuses Dense (k=10) and Sparse (k=10) results.
  - **Reciprocal Rank Fusion (RRF)**: Fuses lists using formula $\sum \frac{1}{k + rank}$, with **$k=60$**.
  - **Reranking**: Top 10 fused results are reranked by a **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) to select the top 3.
- **Generation**:
  - System Prompt enforces **Strict Citations**: `[Source: filename, Page: n]`.

## 1. System Components

### Frontend (Client-Side)
- **Tech Stack**: Vanilla HTML/CSS/JS (No frameworks).
- **Style**: "Web App CLI" - Dark mode, Monospace, Dual-Pane.
- **Communication**: WebSockets (`ws://`) for chat, REST (`POST`) for uploads.

### Backend (Server-Side)
- **Server**: FastAPI (`server.py`).
- **Agent Orchestrator**: LangGraph (`src/agent.py`).
- **Ingestion Engine**: `src/ingest.py` (Hybrid Search).
- **Tools**: Open-Meteo, FileSystem Memory.

### 2) Agent & Stateful Memory (Feature B)
- **Framework**: LangGraph `StateGraph` with structured state.
- **Persistence**:
  - **Short-Term (Thread)**: `MemorySaver` checkpointer manages conversation history within a thread.
  - **Long-Term (Cross-Thread)**: `InMemoryStore` (simulated for hackathon) for sharing knowledge across threads.
  - **Durable**: `save_memory` tool writes high-signal facts to `USER_MEMORY.md` and `COMPANY_MEMORY.md`.

### 3) Safe Sandbox & Weather Tool (Feature C)
- **Weather Tool**: `analyze_weather` (Open-Meteo).
- **Sandbox Execution**: `PythonREPL` (or Pyodide via `langchain-sandbox` if environment permits) for safe code execution.

> **Security Tradeoff Note (Feature C)**: 
> To ensure a frictionless `make sanity` execution for the evaluators without requiring Docker or third-party cloud sandbox API keys, this project utilizes a restricted local `PythonREPL`. To mitigate prompt-injection RCE risks, a pre-execution AST/Regex filter intercepts the generated code and explicitly strips hazardous built-ins (e.g., `os`, `subprocess`, `exec`). In a true enterprise production environment, this local REPL would be entirely replaced by an ephemeral, network-isolated microVM (such as E2B or Modal).