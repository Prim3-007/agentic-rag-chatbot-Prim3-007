# src/ingest_helper.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_split(path: str) -> List[Document]:
    """
    Worker function to load and split a single file.
    Must be top-level for ProcessPoolExecutor pickling.
    This file intentionally DOES NOT import heavy ML libraries.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(path)
        elif ext == ".docx":
            loader = Docx2txtLoader(path)
        else:
            return []
        
        raw_docs = loader.load()
        
        # optimized "Semantic-ish" Chunking
        # Prioritizes paragraphs (double newline) then sentences
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            add_start_index=True
        )
        chunks = text_splitter.split_documents(raw_docs)
        
        # Clean Metadata
        fname = os.path.basename(path)
        for chunk in chunks:
            chunk.metadata["source"] = fname
            
        return chunks
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return []
