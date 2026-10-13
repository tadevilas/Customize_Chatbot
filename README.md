# Production-Ready Agentic AI Platform

A production-ready Agentic AI platform built with Groq LLMs, LangGraph orchestration, Model Context Protocol (MCP) servers, FastAPI backend, and Streamlit frontend.

## Features
- Groq-hosted open-source LLMs (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`)
- SQLite MCP Server integration via `langchain-mcp-adapters`
- Document ingestion and RAG with ChromaDB
- Automated evaluation with RAGAS and DeepEval
- Containerized deployment ready for AWS

## Quick Start
```bash
# Run backend
uv run uvicorn backend.main:app --reload --port 8000

# Run Streamlit UI
uv run streamlit run frontend/streamlit_app.py
```
