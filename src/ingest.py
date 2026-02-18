# Standard & Third Party Imports
from concurrent.futures import ProcessPoolExecutor
import streamlit as st
import os
import shutil
import pickle
from typing import List, Tuple, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import numpy as np

# Import worker from lightweight helper to avoid model re-loading in workers
from src.ingest_helper import load_and_split

# Constants
DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")

class RAGPipeline:
    def __init__(self):
        # Determine device
        import torch
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        print(f"🚀 RAG Pipeline initialized on device: {device}")

        # Embeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        model_kwargs = {'device': device}
        # Increased batch size for speed
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': 64}
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Reranker
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Load indices if they exist
        self.vectorstore = None
        self.bm25_retriever = None
        self.load_indices()

    def load_indices(self):
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                self.vectorstore = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load FAISS index: {e}")
        
        if os.path.exists(BM25_INDEX_PATH):
            try:
                with open(BM25_INDEX_PATH, "rb") as f:
                    self.bm25_retriever = pickle.load(f)
            except Exception as e:
                print(f"Failed to load BM25 index: {e}")

    def render_upload_ui(self):
        """Streamlit UI for uploading and ingesting files."""
        # Legacy stub for compatibility
        pass

    def process_documents(self, file_paths: List[str]):
        """Parallel Load, Chunk, and Index."""
        print(f"Propocessing {len(file_paths)} files with parallel workers...")
        
        chunks = []
        # Use ProcessPool for CPU-bound loading/splitting
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(load_and_split, file_paths)
            for res in results:
                chunks.extend(res)
        
        if not chunks:
            print("No new chunks to index.")
            return

        print(f"Generated {len(chunks)} chunks. Updating indexes...")

        # Dense Indexing (GPU Accelerated via embeddings batch_size)
        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(FAISS_INDEX_PATH)

        # Sparse Indexing (Re-build BM25 from ALL vectorstore docs + new chunks)
        # Fix: Previously, we discarded old BM25 data on new upload. Now we merge.
        all_docs_for_bm25 = []
        if self.vectorstore and hasattr(self.vectorstore, "docstore"):
            # Fetch existing docs from FAISS in-memory docstore
            try:
                all_docs_for_bm25.extend(list(self.vectorstore.docstore._dict.values()))
            except Exception as e:
                print(f"Warning: Could not fetch existing docs from FAISS: {e}")
        
        # Add new chunks (if not already in vectorstore, though they just were added)
        # Note: vectorstore.add_documents adds them to docstore too, so they might be duplicated if we blindly add.
        # Actually, self.vectorstore.add_documents(chunks) updates the docstore.
        # So fetching .values() AFTER add_documents should be sufficient!
        
        if not all_docs_for_bm25:
            all_docs_for_bm25 = chunks
            
        self.bm25_retriever = BM25Retriever.from_documents(all_docs_for_bm25)

        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self.bm25_retriever, f)
            
        print(f"Indexed {len(chunks)} new chunks. Total BM25 Docs: {len(all_docs_for_bm25)}")

    def hybrid_search(self, query: str, k_fusion: int = 25, k_final: int = 5) -> List[Document]:
        """
        Execute Hybrid Search (Dense + Sparse) with RRF and Reranking.
        """
        if not self.vectorstore or not self.bm25_retriever:
            return []

        # 1. Retrieve Candidate Lists
        dense_results = self.vectorstore.similarity_search(query, k=k_fusion)
        
        self.bm25_retriever.k = k_fusion
        sparse_results = self.bm25_retriever.invoke(query)

        # 2. Reciprocal Rank Fusion (k=60)
        fused_docs = self._rrf(dense_results, sparse_results, k=60)
        top_fusion = fused_docs[:k_fusion]

        # 3. Cross-Encoder Reranking
        if not top_fusion:
            return []
            
        pairs = [[query, doc.page_content] for doc in top_fusion]
        # Optimization: Use batch_size for faster inference on GPU/MPS
        scores = self.reranker.predict(pairs, batch_size=32)
        
        # Sort by cross-encoder score
        ranked = sorted(zip(top_fusion, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:k_final]]

    def _rrf(self, list1: List[Document], list2: List[Document], k: int = 60) -> List[Document]:
        """Combine two lists using Reciprocal Rank Fusion."""
        scores = {}
        doc_map = {}
        
        # Helper to process a list
        def process_list(doc_list):
            for rank, doc in enumerate(doc_list):
                content = doc.page_content
                if content not in doc_map:
                    doc_map[content] = doc
                scores[content] = scores.get(content, 0.0) + 1.0 / (k + rank + 1)
        
        process_list(list1)
        process_list(list2)
        
        sorted_content = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[c] for c in sorted_content]

# Singleton instance for simple import
rag = RAGPipeline()
