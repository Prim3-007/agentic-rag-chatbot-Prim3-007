import os
from langchain_core.tools import tool
from typing import Literal

USER_MEMORY_PATH = "USER_MEMORY.md"
COMPANY_MEMORY_PATH = "COMPANY_MEMORY.md"

def _ensure_files_exist():
    for path in [USER_MEMORY_PATH, COMPANY_MEMORY_PATH]:
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(f"# {os.path.basename(path)}\n")

def read_memory(target: Literal["USER", "COMPANY"]) -> str:
    """Read content from the specified memory file."""
    path = USER_MEMORY_PATH if target == "USER" else COMPANY_MEMORY_PATH
    if not os.path.exists(path):
        return ""
    with open(path, "r") as f:
        return f.read()

@tool
def save_memory(should_write: bool, target: Literal["USER", "COMPANY"], summary: str, confidence: float):
    """
    Save high-signal facts to durable memory.
    
    Args:
        should_write: Set to True if this is a new, important fact worth remembering.
        target: "USER" for user preferences/facts, "COMPANY" for organizational knowledge.
        summary: A concise summary of the fact to store.
        confidence: A score between 0.0 and 1.0 indicating certainty.
    """
    if not should_write:
        return "Skipped: should_write is False."
    
    if confidence < 0.8:
        return f"Skipped: Confidence {confidence} is too low (threshold 0.8)."
        
    _ensure_files_exist()
    
    path = USER_MEMORY_PATH if target == "USER" else COMPANY_MEMORY_PATH
    
    # Naive duplicate check
    current_content = read_memory(target)
    if summary in current_content:
        return "Skipped: Fact already exists in memory."
        
    with open(path, "a") as f:
        f.write(f"\n- {summary}")
        
    return f"Success: Wrote to {target} memory."

@tool
def read_memory_tool(target: Literal["USER", "COMPANY"]):
    """
    Read the current state of memory to avoid duplicates or conflicts.
    """
    return read_memory(target)
