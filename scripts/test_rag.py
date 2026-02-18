import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import IngestionPipeline
from src.retrieval import RetrievalPipeline
from dotenv import load_dotenv

load_dotenv()

def test_rag():
    print("----------------------------------------------------------------")
    print("Testing Ingestion & Retrieval Pipeline")
    print("----------------------------------------------------------------")

    # 1. Ingest
    print("\n[1] Ingestion Phase")
    ingestor = IngestionPipeline()
    sample_file = "sample_docs/sample.txt"
    
    if not os.path.exists(sample_file):
        print(f"Error: {sample_file} not found.")
        return

    print(f"Ingesting {sample_file}...")
    result = ingestor.ingest_files([sample_file])
    print(f"Ingestion Result: {result}")

    # 2. Retrieve
    print("\n[2] Retrieval Phase")
    try:
        retriever = RetrievalPipeline()
        query = "What technologies does the Agentic RAG Chatbot use?"
        print(f"Query: '{query}'")
        
        results = retriever.retrieve(query)
        
        print(f"\nRetrieved {len(results)} documents:")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            
    except Exception as e:
        print(f"Retrieval failed: {e}")

if __name__ == "__main__":
    test_rag()
