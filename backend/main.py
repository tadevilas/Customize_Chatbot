import time
import uuid
import json
import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.models import init_db, get_db, ChatSession, Message, Feedback, AgentRun
from backend.database.business_db import init_business_database
from backend.graphs.graph import get_graph
from backend.rag.document_loader import load_document
from backend.rag.chunker import chunk_documents
from backend.rag.vector_store import add_documents
from backend.observability.tracing import setup_tracing, get_run_config, tracing_enabled
from langchain_core.messages import HumanMessage, AIMessage

# Initialize DBs, folders, and tracing on startup
init_db()
init_business_database()
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_DIR, exist_ok=True)
setup_tracing()

app = FastAPI(
    title="Production-Ready Agentic AI Platform",
    description="Agentic platform using Groq LLMs, LangGraph, and SQLite MCP Server",
    version="0.1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request / Response Schemas
class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Unique session ID; created if empty")
    user_id: Optional[str] = Field(default="default_user", description="User identifier")
    message: str = Field(..., description="User question or prompt")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    model_used: str
    tools_used: List[dict]
    latency: float


class FeedbackRequest(BaseModel):
    question: str
    response: str
    rating: str = Field(..., description="'helpful' or 'not_helpful'")
    comments: Optional[str] = None


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "primary_model": settings.PRIMARY_MODEL,
        "fallback_model": settings.FALLBACK_MODEL,
        "mcp_enabled": True,
        "langsmith_tracing": tracing_enabled(),
        "langsmith_project": settings.LANGCHAIN_PROJECT if tracing_enabled() else None,
    }


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    # Ensure session exists in SQLite
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        session = ChatSession(session_id=session_id, user_id=request.user_id)
        db.add(session)
        db.commit()

    # Save user message
    user_msg_record = Message(session_id=session_id, role="user", message=request.message)
    db.add(user_msg_record)
    db.commit()

    # Build LangChain message history
    previous_messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.timestamp.asc()).all()
    lc_messages = [
        HumanMessage(content=m.message) if m.role == "user" else AIMessage(content=m.message)
        for m in previous_messages
    ]

    # Run LangGraph multi-agent pipeline (with LangSmith tracing if enabled)
    graph = get_graph()
    initial_state = {
        "messages":       lc_messages,
        "user_input":     request.message,
        "route":          [],
        "sql_output":     "",
        "search_output":  "",
        "rag_output":     "",
        "final_response": "",
        "tools_used":     [],
        "model_used":     settings.PRIMARY_MODEL,
        "error":          "",
    }
    run_config = get_run_config(session_id, request.user_id or "default_user", run_name="POST /chat")
    result = await graph.ainvoke(initial_state, config=run_config)
    latency = round(time.time() - start_time, 3)

    response_text = result.get("final_response", "No response generated.")
    tools_used    = result.get("tools_used", [])
    model_used    = result.get("model_used", settings.PRIMARY_MODEL)

    # Save assistant message
    db.add(Message(session_id=session_id, role="assistant", message=response_text))

    # Log agent run
    db.add(AgentRun(
        query=request.message,
        tools_used=str([t.get("tool", "") for t in tools_used]),
        latency=latency,
        status="success" if model_used != "error" else "error"
    ))
    db.commit()

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        model_used=model_used,
        tools_used=tools_used,
        latency=latency
    )


@app.post("/chat/stream", tags=["Agent"])
async def chat_stream_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    # Ensure session exists in SQLite
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        session = ChatSession(session_id=session_id, user_id=request.user_id)
        db.add(session)
        db.commit()

    # Save user message
    db.add(Message(session_id=session_id, role="user", message=request.message))
    db.commit()

    # Build LangChain message history
    previous_messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.timestamp.asc()).all()
    lc_messages = [
        HumanMessage(content=m.message) if m.role == "user" else AIMessage(content=m.message)
        for m in previous_messages
    ]

    async def event_generator():
        tools_executed = []
        model_used     = settings.PRIMARY_MODEL

        # Stream supervisor routing event first
        yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'supervisor'})}\n\n"

        # Run LangGraph pipeline (with LangSmith tracing if enabled)
        graph = get_graph()
        initial_state = {
            "messages":       lc_messages,
            "user_input":     request.message,
            "route":          [],
            "sql_output":     "",
            "search_output":  "",
            "rag_output":     "",
            "final_response": "",
            "tools_used":     [],
            "model_used":     settings.PRIMARY_MODEL,
            "error":          "",
        }
        run_config = get_run_config(session_id, request.user_id or "default_user", run_name="POST /chat/stream")
        result = await graph.ainvoke(initial_state, config=run_config)

        # Emit routing info
        route = result.get("route", [])
        yield f"data: {json.dumps({'type': 'route', 'agents': route})}\n\n"

        # Stream tool events
        tools_executed = result.get("tools_used", [])
        for tool_info in tools_executed:
            yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_info})}\n\n"
            yield f"data: {json.dumps({'type': 'tool_end',   'tool': tool_info.get('tool', '')})}\n\n"

        # Stream final response token by token
        full_response = result.get("final_response", "No response generated.")
        model_used    = result.get("model_used", settings.PRIMARY_MODEL)
        chunk_size = 8
        for i in range(0, len(full_response), chunk_size):
            yield f"data: {json.dumps({'type': 'token', 'content': full_response[i:i+chunk_size]})}\n\n"

        latency = round(time.time() - start_time, 3)

        # Save to DB
        db.add(Message(session_id=session_id, role="assistant", message=full_response))
        db.add(AgentRun(
            query=request.message,
            tools_used=str([t.get("tool", "") for t in tools_executed]),
            latency=latency,
            status="success" if model_used != "error" else "error"
        ))
        db.commit()

        yield f"data: {json.dumps({'type': 'done', 'model_used': model_used})}\n\n"
        yield f"data: {json.dumps({'type': 'final_stats', 'session_id': session_id, 'latency': latency})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/feedback", tags=["Feedback"])
def submit_feedback(fb: FeedbackRequest, db: Session = Depends(get_db)):
    feedback_record = Feedback(
        question=fb.question,
        response=fb.response,
        rating=fb.rating,
        comments=fb.comments
    )
    db.add(feedback_record)
    db.commit()
    return {"status": "success", "message": "Feedback recorded."}


@app.get("/sessions", tags=["History"])
def list_recent_sessions(limit: int = 5, db: Session = Depends(get_db)):
    """Return the most recent chat sessions with a short preview from the first user message."""
    sessions = (
        db.query(ChatSession)
        .order_by(ChatSession.created_at.desc())
        .limit(limit)
        .all()
    )
    result = []
    for s in sessions:
        first_msg = (
            db.query(Message)
            .filter(Message.session_id == s.session_id, Message.role == "user")
            .order_by(Message.timestamp.asc())
            .first()
        )
        preview = (first_msg.message[:60] + "…") if first_msg and len(first_msg.message) > 60 else (first_msg.message if first_msg else "Empty session")
        result.append({
            "session_id": s.session_id,
            "created_at": str(s.created_at),
            "preview": preview,
        })
    return result


@app.get("/sessions/{session_id}/messages", tags=["History"])
def get_session_history(session_id: str, db: Session = Depends(get_db)):
    messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.timestamp.asc()).all()
    return [{"role": m.role, "message": m.message, "timestamp": str(m.timestamp)} for m in messages]


@app.post("/upload", tags=["RAG"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT), parse it, chunk it,
    embed it and store it in ChromaDB for RAG retrieval.

    Returns a summary of what was ingested.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: .pdf, .docx, .txt"
        )

    # Save uploaded file to disk
    save_path = os.path.join(settings.UPLOADS_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Load → chunk → embed → store
        docs   = load_document(save_path)
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
        add_documents(chunks, collection_name=settings.RAG_COLLECTION)

        # Count by content type
        from collections import Counter
        type_counts = dict(Counter(c.metadata.get("content_type", "text") for c in chunks))

        return {
            "status": "success",
            "filename": file.filename,
            "saved_to": save_path,
            "docs_extracted": len(docs),
            "chunks_stored": len(chunks),
            "content_types": type_counts,
            "collection": settings.RAG_COLLECTION,
        }

    except Exception as e:
        # Clean up saved file on failure
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/upload/documents", tags=["RAG"])
def list_uploaded_documents():
    """List all documents currently stored in the uploads folder."""
    if not os.path.exists(settings.UPLOADS_DIR):
        return {"documents": []}
    files = [
        {"filename": f, "size_kb": round(os.path.getsize(os.path.join(settings.UPLOADS_DIR, f)) / 1024, 1)}
        for f in os.listdir(settings.UPLOADS_DIR)
        if os.path.isfile(os.path.join(settings.UPLOADS_DIR, f))
    ]
    return {"documents": files, "count": len(files)}


@app.delete("/upload/documents/{filename}", tags=["RAG"])
def delete_document(filename: str):
    """Delete an uploaded document from disk (does not remove its chunks from ChromaDB)."""
    file_path = os.path.join(settings.UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    os.remove(file_path)
    return {"status": "deleted", "filename": filename}
