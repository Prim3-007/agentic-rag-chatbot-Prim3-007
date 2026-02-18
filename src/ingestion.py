import os
import shutil
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import pickle

DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")

class IngestionPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        # Semantic chunking relies on embeddings to find break points
        self.text_splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

    def load_file(self, file_path: str) -> List[Document]:
        """Load a file based on its extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return loader.load()

    def ingest_files(self, file_paths: List[str]) -> dict:
        """
        Load files, split them into chunks, and create/update indices.
        Returns a summary of ingestion.
        """
        all_docs = []
        for path in file_paths:
            try:
                docs = self.load_file(path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not all_docs:
            return {"status": "failed", "message": "No documents loaded"}

        # Semantic Chunking
        print("Chunking documents...")
        chunks = self.text_splitter.split_documents(all_docs)
        print(f"Created {len(chunks)} chunks with Semantic Chunking.")

        # Update metadata for chunks (ensure source is persistent)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        # Create/Save FAISS Index
        print("Creating FAISS index...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)

        # Create/Save BM25 Index
        print("Creating BM25 index...")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        # Pickle the BM25 retriever (it's not a standard vector store)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25_retriever, f)

        return {
            "status": "success",
            "chunks_created": len(chunks),
            "files_processed": len(file_paths)
        }

if __name__ == "__main__":
    # Test run
    # Mock some files if needed, or just leave as class definition
    pass
