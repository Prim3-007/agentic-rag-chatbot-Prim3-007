import os
import shutil
import json
from typing import List

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Project Logic
from src.ingest import rag
from src.agent import graph
from langchain_core.messages import HumanMessage

app = FastAPI()

# --- API & WebSocket Routes (Defined FIRST) ---

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Handle file uploads for RAG ingestion.
    """
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_paths = []
    try:
        # Save uploaded files
        for file in files:
            path = os.path.join(temp_dir, file.filename)
            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_paths.append(path)
        
        # Invoke Ingestion Pipeline (Synchronous call)
        rag.process_documents(file_paths)
        
        return JSONResponse({"status": "success", "message": f"Successfully ingested {len(files)} files."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def get_memory_snapshot():
    """Read current memory state from markdown files."""
    user_mem = ""
    comp_mem = ""
    if os.path.exists("USER_MEMORY.md"):
        with open("USER_MEMORY.md", "r") as f: user_mem = f.read()
    if os.path.exists("COMPANY_MEMORY.md"):
        with open("COMPANY_MEMORY.md", "r") as f: comp_mem = f.read()
    return {"user": user_mem, "company": comp_mem}

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Send initial memory state on connection
    await websocket.send_json({"type": "memory", "data": get_memory_snapshot()})
    
    try:
        while True:
            # Wait for user message
            data = await websocket.receive_text()
            user_input = data
            
            # Send "Status" update
            await websocket.send_json({"type": "status", "message": "🧠 Processing Request..."})
            
            # Prepare LangGraph Inputs
            thread_id = "fastapi_cli_user"
            config = {"configurable": {"thread_id": thread_id}}
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # Stream events from LangGraph
            async for event in graph.astream_events(inputs, config=config, version="v1"):
                kind = event["event"]
                name = event["name"]
                
                # 1. Status & Logs
                if kind == "on_chain_start" and name != "LangGraph":
                    pass 
                    
                elif kind == "on_tool_start":
                    await websocket.send_json({"type": "status", "message": f"🔨 Executing {name}..."})
                    
                elif kind == "on_tool_end":
                    tool_output = event["data"].get("output")
                    await websocket.send_json({
                        "type": "log", 
                        "content": f"[{name}] Output:\n{str(tool_output)[:500]}..." 
                    })
                    
                    if name == "save_memory":
                        await websocket.send_json({"type": "memory", "data": get_memory_snapshot()})
                
                # 2. Streaming Tokens (The actual answer)
                elif kind == "on_chat_model_stream":
                    # Filter metadata to ensure we only stream from the 'agent' node
                    # We want to hide the 'memory_router' structured output
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    if node_name == "agent":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content:
                            await websocket.send_json({"type": "token", "chunk": chunk.content})
                        
            # Finalize Turn
            await websocket.send_json({"type": "memory", "data": get_memory_snapshot()})
            await websocket.send_json({"type": "status", "message": "✅ Ready"})
            await websocket.send_json({"type": "end_turn"})
            
    except WebSocketDisconnect:
        print("Client disconnected")

# --- Static Files (Defined LAST to avoid masking routes) ---
app.mount("/", StaticFiles(directory="static", html=True), name="static")
