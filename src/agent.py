from typing import Annotated, Literal, TypedDict, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from src.memory import save_memory, read_memory_tool
from src.ingest import rag
from src.tools.sandbox import python_interpreter
from src.tools.weather import analyze_weather

# --- 1. State Definition ---
class AgentState(TypedDict):
    # Messages history
    messages: Annotated[list[BaseMessage], add_messages]
    # Memory context (e.g. recent memories read) - explicitly requested by prompt
    memory_context: str

# --- 2. Memory Router Schema ---
class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether a durable fact should be written to memory.")
    target: Literal["USER", "COMPANY"] = Field(description="The target memory file: USER for preferences, COMPANY for org facts.")
    summary: str = Field(description="A concise summary of the fact.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")

# --- 3. Nodes ---

def memory_router_node(state: AgentState):
    """
    Evaluates the latest user message to decide if we need to write to memory.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    # Structured output binding
    router = model.with_structured_output(MemoryDecision)
    
    # Analyze only the last message (User's input)
    last_msg = state["messages"][-1]
    
    # System prompt for the router
    system_prompt = (
        "You are a Memory Router. Analyze the user's message for durable facts. "
        "ignore ephemeral queries (e.g. 'what is the weather'). "
        "Focus on facts like 'My name is X', 'The company was founded in Y'. "
        "If a fact is found, output should_write=True."
    )
    
    decision = router.invoke([SystemMessage(content=system_prompt), last_msg])
    
    # Logic: Read-Before-Write Execution
    # If decision is to write, we execute the write logic directly here or pass to a tool.
    # The prompt says: "The system must first read ... prevent duplicate ... before appending".
    # Our `save_memory` tool/func in src/memory.py handles the "Read -> Check Duplicate -> Append" logic.
    # So we can just call it if should_write is True.
    
    log = ""
    if decision.should_write and decision.confidence > 0.8:
        # Call the save_memory function directly (it's a tool, but can be called as func if we import the underlying logic or just invoke it)
        # ensure_files_exist checks are inside save_memory
        result = save_memory.invoke({
            "should_write": True,
            "target": decision.target,
            "summary": decision.summary,
            "confidence": decision.confidence
        })
        log = f"[Memory Router] {result}"
    
    # We don't necessarily need to update 'messages' with this log, 
    # but we can update 'memory_context' if we want to pass info to the agent.
    # For now, we arguably just pass.
    return {"memory_context": log}

def agent_node(state: AgentState):
    """
    Main agent node that answers the user.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Bind tools
    tools_list = [read_memory_tool, python_interpreter, analyze_weather, retrieve_docs]
    model = model.bind_tools(tools_list)
    
    system_msg = SystemMessage(content=(
        "You are an expert AI assistant. "
        "Use the available tools to answer questions. "
        "Strict Citations: [Source: <filename>, Page: <page>] when using retrieved docs."
        "Always execute code via the provided tools (e.g. analyze_weather) rather than displaying code blocks to the user."
        "CRITICAL: Answer ONLY using information from the retrieved documents or tools. "
        "If the answer is not in the documents/tools, state 'I cannot answer this based on the available information.' "
        "Do NOT use your pre-existing knowledge for factual queries. "
        "EXCEPTION: You MAY use the python_interpreter tool for general math, logic, or data analysis tasks."
    ))
    
    messages = [system_msg] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- 4. Tool Wrapper ---
# Redefine retrieve_docs to use the Unified Ingest module (since I overwrote agent.py plan)
from langchain_core.tools import tool
import os

@tool
def retrieve_docs(query: str) -> str:
    """
    Search the knowledge base for information. 
    Use this tool when the user asks questions about uploaded documents or specific knowledge.
    """
    docs = rag.hybrid_search(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    
    result = "Found the following information:\n\n"
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        citation_meta = f"Source: {os.path.basename(source)}"
        if page is not None:
            citation_meta += f", Page: {page}"
        result += f"--- Document {i+1} ---\nMetadata provided: [{citation_meta}]\nContent:\n{doc.page_content}\n\n"
    return result

# Update tools list
tools = [retrieve_docs, save_memory, read_memory_tool, python_interpreter, analyze_weather]
tool_node = ToolNode(tools)

# --- 5. Graph Compile ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("memory_router", memory_router_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Edges
# Flow: content -> memory_router -> agent -> (output)
# Note: memory_router runs fast and creates side-effects (saving memory). 
# It passes control to agent.
workflow.add_edge(START, "memory_router")
workflow.add_edge("memory_router", "agent")

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Persistence
checkpointer = MemorySaver()
store = InMemoryStore()

graph = workflow.compile(checkpointer=checkpointer, store=store)
