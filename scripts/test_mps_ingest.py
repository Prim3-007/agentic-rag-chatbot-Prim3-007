import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    print("Initialize RAG pipeline...")
    from src.ingest import rag
    print("Success: RAG Pipeline loaded.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
