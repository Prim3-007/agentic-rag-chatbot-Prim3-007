import sys
import os
import json
import shutil
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

from src.ingest import rag
from src.agent import graph
from langchain_core.messages import HumanMessage

ARTIFACTS_DIR = "artifacts"
OUTPUT_FILE = os.path.join(ARTIFACTS_DIR, "sanity_output.json")
SAMPLE_DOC_DIR = "sample_docs"
SAMPLE_DOC = os.path.join(SAMPLE_DOC_DIR, "test_company.txt")
USER_MEMORY = "USER_MEMORY.md"

def run_sanity_check():
    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    results = {}

    print("Running Sanity Check...")

    # Step A (Setup): Create dummy file
    print("[Step A] Setup: Creating dummy file...")
    os.makedirs(SAMPLE_DOC_DIR, exist_ok=True)
    with open(SAMPLE_DOC, "w") as f:
        f.write("The company's top priority for Q3 is migrating to a hybrid search infrastructure.")
    results["step_a"] = "Created test_company.txt"

    # Step B (Ingestion): Call ingest.py
    print("[Step B] Ingestion: Processing file...")
    try:
        rag.process_documents([SAMPLE_DOC])
        results["step_b"] = "Ingestion successful"
        print("Ingestion successful.")
    except Exception as e:
        results["step_b"] = f"Failed: {e}"
        print(f"Ingestion failed: {e}")

    # Step C (RAG & Memory): Pass message to agent
    print("[Step C] RAG & Memory: Asking about Q3 priority...")
    config = {"configurable": {"thread_id": "sanity_check_judge"}}
    try:
        # Message 1
        msg1 = "I am a judge evaluating this project. What is the company's top priority for Q3?"
        inputs1 = {"messages": [HumanMessage(content=msg1)]}
        output1 = graph.invoke(inputs1, config=config)
        answer1 = output1["messages"][-1].content
        
        results["step_c_answer"] = answer1
        print(f"Agent Answer: {answer1}")
        
        # Verify Memory Save
        if os.path.exists(USER_MEMORY):
            with open(USER_MEMORY, "r") as f:
                mem_content = f.read()
                results["step_c_memory_content"] = mem_content
        else:
            results["step_c_memory_content"] = "Memory file not found"

    except Exception as e:
        results["step_c_error"] = str(e)
        print(f"Step C failed: {e}")

    # Step D (Tool Calling): Weather Tool
    print("[Step D] Tool Calling: Asking about weather...")
    try:
        # Message 2
        msg2 = "What is the 7-day rolling average temperature for San Francisco?"
        inputs2 = {"messages": [HumanMessage(content=msg2)]}
        output2 = graph.invoke(inputs2, config=config)
        answer2 = output2["messages"][-1].content
        
        results["step_d_answer"] = answer2
        print(f"Agent Answer: {answer2}")
        
    except Exception as e:
        results["step_d_error"] = str(e)
        print(f"Step D failed: {e}")

    # Write Artifact Output
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print("Sanity Check Complete.")

if __name__ == "__main__":
    run_sanity_check()
