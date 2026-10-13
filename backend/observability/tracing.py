"""
Phase 6 — LangSmith Observability & Tracing

Activates LangSmith tracing for the entire platform.
When LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY is set in .env,
every LangGraph run, LLM call, and tool invocation is automatically
traced in the LangSmith dashboard.

What gets traced automatically:
  - LangGraph node execution (Supervisor, SQL/RAG/Research agents, Synthesis)
  - Every LLM call (prompt sent, response received, token usage, latency)
  - Every tool call (name, input args, output, latency)
  - Errors and retries

Usage:
  Call setup_tracing() once at app startup.
  Use get_run_config(session_id, user_id) to attach metadata to each graph run.
"""

import os
from typing import Optional, Dict, Any
from backend.config import settings


def setup_tracing() -> bool:
    """
    Activate LangSmith tracing by setting the required environment variables.
    Returns True if tracing is enabled, False if disabled or not configured.
    """
    if not settings.LANGCHAIN_TRACING_V2 or not settings.LANGCHAIN_API_KEY:
        print("[LangSmith] Tracing disabled — set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env")
        return False

    # LangChain reads these env vars automatically
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]     = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"

    print(f"[LangSmith] Tracing enabled → project: '{settings.LANGCHAIN_PROJECT}'")
    return True


def get_run_config(
    session_id: str,
    user_id: str = "default_user",
    run_name: Optional[str] = None,
    tags: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Build a LangChain run config dict that tags each graph invocation
    with session_id and user_id so runs are grouped in LangSmith.

    Pass the returned dict as the `config` argument to graph.ainvoke():
        result = await graph.ainvoke(state, config=get_run_config(session_id, user_id))

    Fields set:
      run_name  → shown as the trace title in LangSmith
      tags      → filter by tag in LangSmith dashboard
      metadata  → session_id, user_id, model — shown in trace details
    """
    return {
        "run_name": run_name or f"chat | session={session_id[:8]}",
        "tags": (tags or []) + [
            "agentic-ai-platform",
            f"session:{session_id[:8]}",
            f"user:{user_id}",
        ],
        "metadata": {
            "session_id":  session_id,
            "user_id":     user_id,
            "primary_model": settings.PRIMARY_MODEL,
            "project":     settings.LANGCHAIN_PROJECT,
        },
    }


def tracing_enabled() -> bool:
    """Returns True if LangSmith tracing is currently active."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
