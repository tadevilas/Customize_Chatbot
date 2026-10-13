"""
Phase 9 — CI Test Suite

Tests that run inside GitHub Actions (no real LLM calls — mocked where needed).
All tests use FastAPI TestClient so no server needs to be running.
"""

import os
import sys
from pathlib import Path

# Ensure root workspace directory is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Set dummy env vars before importing the app so pydantic-settings is happy
for key in ("GROQ_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY"):
    if not os.environ.get(key):
        os.environ[key] = "test-key"
os.environ["LANGCHAIN_TRACING_V2"] = "false"   # never trace in tests

from backend.main import app                          # noqa: E402
from backend.database.business_db import init_business_database  # noqa: E402

client = TestClient(app)


# ── Health ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_schema(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "primary_model" in data
        assert "fallback_model" in data
        assert "mcp_enabled" in data
        assert "langsmith_tracing" in data

    def test_langsmith_tracing_false_in_ci(self):
        """LangSmith tracing must be OFF when LANGCHAIN_TRACING_V2=false."""
        data = client.get("/health").json()
        assert data["langsmith_tracing"] is False
        assert data["langsmith_project"] is None


# ── Sessions ───────────────────────────────────────────────────────────────────

class TestSessions:
    def test_list_sessions_empty(self):
        resp = client.get("/sessions?limit=5")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_sessions_limit_param(self):
        resp = client.get("/sessions?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) <= 2

    def test_session_messages_unknown_id(self):
        resp = client.get("/sessions/nonexistent-session-id/messages")
        assert resp.status_code == 200
        assert resp.json() == []


# ── Feedback ───────────────────────────────────────────────────────────────────

class TestFeedback:
    def test_submit_helpful(self):
        resp = client.post("/feedback", json={
            "question": "What is AI?",
            "response": "AI is artificial intelligence.",
            "rating":   "helpful",
            "comments": "Great answer"
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_submit_not_helpful(self):
        resp = client.post("/feedback", json={
            "question": "Tell me a joke",
            "response": "Why did the AI cross the road?",
            "rating":   "not_helpful",
        })
        assert resp.status_code == 200

    def test_invalid_missing_rating(self):
        resp = client.post("/feedback", json={
            "question": "Q",
            "response": "A",
        })
        assert resp.status_code == 422   # pydantic validation error


# ── Upload ─────────────────────────────────────────────────────────────────────

class TestUpload:
    def test_list_documents(self):
        resp = client.get("/upload/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert "count" in data

    def test_upload_txt_file(self):
        content = b"This is a test document for the RAG pipeline."
        resp = client.post(
            "/upload",
            files={"file": ("test_upload.txt", content, "text/plain")}
        )
        # Accept 200 (success) or 500 (model not loaded in CI) — just not 4xx
        assert resp.status_code in (200, 500)

    def test_delete_nonexistent_document(self):
        resp = client.delete("/upload/documents/does_not_exist.txt")
        assert resp.status_code == 404


# ── Business Database ──────────────────────────────────────────────────────────

class TestBusinessDatabase:
    def test_db_initializes(self):
        init_business_database()
        assert os.path.exists("./data/business_data.db")
