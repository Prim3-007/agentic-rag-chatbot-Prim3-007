# Agentic RAG Chatbot - Hackathon Submission

## Overview
A production-ready Agentic RAG Chatbot built with **LangGraph**, **LangChain**, and **Streamlit**.
It features **Hybrid Search** (Dense + Sparse), **Durable Memory**, and **Safe Tool Execution** (Sandbox).

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API Key

### Setup
1. Clone the repository.
2. Create a `.env` file with your API key:
   ```bash
   cp .env.example .env
   # Edit .env and add: OPENAI_API_KEY=sk-...
   ```
3. Install dependencies:
   ```bash
   make install
   ```
   *Note: This creates### 🖥️ SOTA Web App CLI
We've replaced the legacy Streamlit UI with a high-performance **FastAPI + Vanilla JS** architecture.
- **Backend**: FastAPI server with WebSocket bridge (`server.py`).
- **Frontend**: Single-page "Terminal Emulator" (`static/index.html`).
- **Features**: Real-time token streaming, dual-pane layout, and live system monitoring.

## 🚀 Quick Start

### 1. Setup Environment
```bash
make install
```

### 2. Run the System
```bash
make run
```
Access the CLI at: **http://localhost:8000**) in your browser.

### Run Sanity Check (Judge Command)
```bash
make sanity
```
This runs an end-to-end test script and verifies:
- Ingestion of sample docs
- Retrieval and RAG answers
- Memory persistence
- Sandbox code execution
Output is saved to `artifacts/sanity_output.json`.

---

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for details on:
- Ingestion Pipeline (Semantic Chunking)
- Retrieval (FAISS + BM25 + RRF + Reranker)
- Agent Memory (Read/Write Markdown)
- Sandbox (PythonREPL with Open-Meteo access)

## Video Walkthrough
[Watch the Demo Video](RAG-Chatbot-Demo.mov)

## Participant Info
- **Name**: Ashwin Sonawane
- **Email**: ashwinssonawane@gmail.com
- **GitHub**: Prim3-007