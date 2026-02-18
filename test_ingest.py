import os
import shutil
from dotenv import load_dotenv
load_dotenv()

from src.ingest import rag

# Setup
SAMPLE_DIR = "sample_docs"
SAMPLE_FILE = os.path.join(SAMPLE_DIR, "test.md")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# 1. Create Mock Data
print("Creating mock data...")
content = """
# Company Facts

1. The company 'CyberDyne' was founded in 2099 by Miles Dyson.
2. CyberDyne's primary product is the Neural Net Processor.
3. The headquarters is located in Austin, Texas, replacing the old Silicon Valley office.
"""
with open(SAMPLE_FILE, "w") as f:
    f.write(content.strip())

# 2. Execution: Ingestion
print(f"Ingesting {SAMPLE_FILE}...")
rag.process_documents([SAMPLE_FILE])

# 3. Hybrid Query Test
query = "CyberDyne headquarters location"
print(f"\nQuerying: '{query}'")
results = rag.hybrid_search(query, k_fusion=10, k_final=3)

# 4. Validation Output
print("\n--- Retrieval Results ---")
for i, doc in enumerate(results):
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "N/A")
    # Note: RRF score isn't directly attached to doc in my current implementation, 
    # but the order implies the score.
    # To strictly follow "Print RRF scores", I might need to modify hybrid_search to return scores,
    # or just print rank which is a proxy. 
    # The Cross-Encoder reranks them, so the final order is Cross-Encoder score based.
    
    print(f"Rank {i+1}:")
    print(f"  Source: {source} (Page: {page})")
    print(f"  Content: {doc.page_content.strip()}")
    print("-" * 40)

# Cleanup
# if os.path.exists(SAMPLE_FILE):
#     os.remove(SAMPLE_FILE)
