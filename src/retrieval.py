import os
import pickle
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_core.tools import tool
import numpy as np

DATA_DIR = "data"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")

class RetrievalPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        
        # Load FAISS index
        if os.path.exists(FAISS_INDEX_PATH):
            self.vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            self.vectorstore = None
            print("Warning: FAISS index not found.")

        # Load BM25 index
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, "rb") as f:
                self.bm25_retriever = pickle.load(f)
        else:
            self.bm25_retriever = None
            print("Warning: BM25 index not found.")

        # Initialize Cross-Encoder for reranking
        # This will download the model on first run
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def reciprocal_rank_fusion(self, results: List[List[Document]], k=60) -> List[Document]:
        """
        Combine multiple lists of documents using Reciprocal Rank Fusion.
        """
        fused_scores = {}
        doc_map = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = doc.page_content
                if doc_str not in doc_map:
                    doc_map[doc_str] = doc
                
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0.0
                
                fused_scores[doc_str] += 1.0 / (k + rank + 1)

        reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_str] for doc_str, score in reranked_results]

    def retrieve(self, query: str, top_k_fusion: int = 10, top_k_final: int = 3) -> List[Document]:
        """
        Retrieve documents using Hybrid Search (Dense + Sparse) -> RRF -> Reranking.
        """
        if not self.vectorstore or not self.bm25_retriever:
            return []

        # 1. Dense Search (Vector)
        dense_docs = self.vectorstore.similarity_search(query, k=top_k_fusion)
        
        # 2. Sparse Search (BM25)
        # BM25Retriever doesn't support 'k' param in invoke directly easily without kwarg adjustment usually
        # But we can set k on the retriever object if needed, or just take default. 
        # Default is usually 4. Let's force it to top_k_fusion if possible.
        self.bm25_retriever.k = top_k_fusion
        sparse_docs = self.bm25_retriever.invoke(query)

        # 3. Reciprocal Rank Fusion
        hybrid_docs = self.reciprocal_rank_fusion([dense_docs, sparse_docs])
        
        # Slice to top candidate pool for reranking (e.g. top 10 from fusion)
        candidates = hybrid_docs[:top_k_fusion]
        
        if not candidates:
            return []

        # 4. Cross-Encoder Reranking
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # Combine docs with scores
        scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        # Select top 3
        final_docs = [doc for doc, score in scored_docs[:top_k_final]]
        
        return final_docs

@tool
def retrieve_docs(query: str) -> str:
    """
    Search the knowledge base for information. 
    Use this tool when the user asks questions about uploaded documents or specific knowledge.
    """
    pipeline = RetrievalPipeline()
    docs = pipeline.retrieve(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    
    result = "Found the following information:\n\n"
    for i, doc in enumerate(docs):
        # Include citation info in the tool output so the LLM can cite it
        source = doc.metadata.get("source", "Unknown")
        # Attempt to get page number if available (common in PyPDFLoader)
        page = doc.metadata.get("page", None)
        citation_meta = f"Source: {os.path.basename(source)}"
        if page is not None:
            citation_meta += f", Page: {page}"
            
        result += f"--- Document {i+1} ---\nMetadata provided: [{citation_meta}]\nContent:\n{doc.page_content}\n\n"
    
    return result

if __name__ == "__main__":
    # Test logic
    pass
