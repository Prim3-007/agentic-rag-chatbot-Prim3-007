# Agentic RAG Chatbot 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/FastAPI-Backend-green)](https://fastapi.tiangolo.com/)

A production-grade **Agentic RAG System** designed for high-precision retrieval, durable memory, and safe code execution. Built to solve the "Context Window" and "Hallucination" problems in modern LLM applications.

---

## 📖 About The Project

This project was built for the **CDF Hackathon** to demonstrate the power of **Agentic Workflows**. Unlike traditional RAG chatbots that simply "retrieve and answer," this Agent acts as an autonomous reasoning engine.

### Core Philosophy
1.  **Think Before Speaking**: The agent plans its steps (Retrieve -> Analyze -> Answer).
2.  **Remember Everything**: It maintains long-term memory of user preferences and company facts.
3.  **Trust But Verify**: All code execution happens in a secure, sandboxed environment.

---

## ✨ Key Capabilities

### 1. 🧠 Hybrid RAG with Semantic Chunking
-   **Dual Retrieval**: Combines **FAISS (Dense)** for semantic meaning and **BM25 (Sparse)** for keyword precision.
-   **Smart Ingestion**: Uses a custom **Semantic Chunker** that respects document structure (paragraphs, headers) rather than arbitrary usage.
-   **Reranking**: A Cross-Encoder component reranks the top 25 results to ensure the LLM sees only the most relevant context.

### 2. 💾 Durable Memory (The "Brain")
-   **User Memory**: Remembers your name, role, and preferences across sessions (e.g., "I am a Recruiter from Google").
-   **Company Memory**: Stores organizational facts (e.g., "Deployment logs are kept for 30 days").
-   **Routing Logic**: A specialized LLM Router decides *what* is worth remembering vs. what is ephemeral chat.

### 3. 🛡️ Sandboxed Code Execution
-   **Safe Math & Logic**: The agent can write and execute Python code to solve complex math problems (e.g., "Days between two dates").
-   **Security Guardrails**: A custom AST analyzer blocks dangerous imports like `os`, `sys`, or `subprocess` before execution.

### 4. ⚡ High-Performance Architecture
-   **Parallel Ingestion**: Uses `ProcessPoolExecutor` to ingest large documents 4x faster on multi-core CPUs.
-   **Hardware Acceleration**: Automatically detects and uses **MPS (Mac Metal)** or **CUDA (Nvidia)** for embedding generation.

---

## 🛠️ Tech Stack

-   **Orchestration**: LangGraph, LangChain
-   **LLM**: OpenAI GPT-4o
-   **Vector Store**: FAISS (Facebook AI Similarity Search)
-   **Backend**: FastAPI (Python)
-   **Frontend**: Vanilla JS + HTML5 (Terminal Emulator UI)
-   **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API Key

### Installation

1.  **Clone the Repo**
    ```bash
    git clone https://github.com/Prim3-007/agentic-rag-chatbot-Prim3-007.git
    cd agentic-rag-chatbot-Prim3-007
    ```

2.  **Install Dependencies**
    ```bash
    make install
    ```

3.  **Set API Key**
    ```bash
    export OPENAI_API_KEY=sk-proj-your-key-here
    ```

### Running the App
```bash
make run
```
Open **http://localhost:8000** in your browser.

### 🧪 Run Sanity Check (Automated Testing)
The project includes a comprehensive test suite to verify RAG, Memory, and Sandbox functionality.
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

## 🎥 Video Walkthrough

[Watch the Demo Video](RAG-Chatbot-Demo.mov)

> **Note**: This video demonstrates the full flow: Ingestion -> RAG -> Memory -> Sandbox -> Safety Refusal.

---

## 👨‍💻 Participant Info

-   **Name**: Ashwin Sonawane
-   **GitHub**: [Prim3-007](https://github.com/Prim3-007)
-   **Email**: ashwinssonawane@gmail.com

---
*Built with ❤️ for the CDF Hackathon.*
