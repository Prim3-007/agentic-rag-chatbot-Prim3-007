import sys
import os
import shutil

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    print("Initialize RAG pipeline...")
    from src.ingest import rag
    
    # Create dummy files for testing parallel load
    test_dir = "test_parallel_data"
    os.makedirs(test_dir, exist_ok=True)
    files = []
    for i in range(4):
        p = os.path.join(test_dir, f"doc_{i}.txt")
        with open(p, "w") as f:
            f.write(f"This is document number {i}. It has some content related to AI.")
        files.append(p)
        
    print(f"Testing parallel ingestion of {len(files)} files...")
    rag.process_documents(files)
    print("Success: Parallel ingestion completed.")
    
    # Cleanup
    shutil.rmtree(test_dir)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
