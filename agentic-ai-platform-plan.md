# Production-Ready Agentic AI Platform — Implementation Plan

## Top-Level Overview

This plan defines the step-by-step roadmap to build an enterprise-style, production-grade **Agentic AI Platform**. The system orchestrates multiple specialized agents (SQL, RAG, Web Search, Calculator, and Critic) via **LangGraph**, exposes endpoints via **FastAPI**, presents an interactive UI via **Streamlit**, stores structured metadata in **SQLite** and vector embeddings in **ChromaDB**, and integrates full **RAGAS / DeepEval** quality gates, **LangSmith** observability, containerization with **Docker Compose**, and deployment targeting **AWS EC2**.

The implementation is structured into 10 progressive, decoupled milestones matching the architecture defined in [`Project_Owerview.md`](Project_Owerview.md).

---

## Sub-Tasks

### Sub-Task 1: Project Scaffolding & Basic Chatbot (Phase 1)
- **Intent**: Establish the core repository structure, environment configuration, database models, and a baseline end-to-end chat flow between Groq LLM, FastAPI backend, and Streamlit frontend.
- **Expected Outcomes**:
  - Clean modular directory layout initialized (`backend/`, `frontend/`, `data/`, `docker/`, `tests/`).
  - FastAPI server up with health check (`GET /health`) and basic chat endpoint (`POST /chat`).
  - Streamlit app connecting to FastAPI and rendering streaming/response messages.
  - SQLite database initialized with `chat_sessions` and `messages` tables.
  - Groq API client integration (`llama-3.3-70b-versatile` with `llama-3.1-8b-instant` fallback).
- **Todo List**:
  - [ ] Initialize repository structure, `.gitignore`, `.env.example`, and baseline `requirements.txt`.
  - [ ] Set up SQLite database connection and schemas (`chat_sessions`, `messages`).
  - [ ] Implement Groq LLM wrapper with fallback handling.
  - [ ] Implement FastAPI chat and session management endpoints.
  - [ ] Build basic Streamlit chat interface with session persistence.
  - [ ] Add unit tests verifying API health and LLM connectivity.
- **Relevant Context**: [`Project_Owerview.md`](Project_Owerview.md) sections: Technology Stack, Database Design, API Endpoints, Phase 1.
- **Status**: [ ] pending

---

### Sub-Task 2: RAG Pipeline with ChromaDB & BGE Embeddings (Phase 2)
- **Intent**: Implement document ingestion (PDF, DOCX, PPTX, TXT), chunking, vector embedding storage, retrieval, and reranking.
- **Expected Outcomes**:
  - Document upload endpoint (`POST /upload`) supporting PDF, DOCX, PPTX, TXT.
  - Recursive text chunking (chunk size 1000, overlap 200).
  - Embeddings generated via `BAAI/bge-small-en-v1.5` and stored in local `ChromaDB`.
  - Reranker pipeline using `BAAI/bge-reranker-base` to rerank top-K retrieved candidates.
  - Citations generation with source document names and page references.
- **Todo List**:
  - [ ] Create document parsers for PDF (`PyPDFLoader`), DOCX (`Docx2txtLoader`), and text files.
  - [ ] Implement `RecursiveCharacterTextSplitter` chunking pipeline.
  - [ ] Set up ChromaDB vector store instance with HuggingFace `bge-small-en-v1.5`.
  - [ ] Implement retrieval step fetching top 5 candidates.
  - [ ] Integrate cross-encoder reranking step using `bge-reranker-base`.
  - [ ] Implement synthesis logic generating answers with precise source citations.
  - [ ] Expose document upload and RAG querying via FastAPI and integrate file uploader in Streamlit.
- **Relevant Context**: [`Project_Owerview.md`](Project_Owerview.md) sections: RAG Pipeline, Embeddings, Reranker, Phase 2.
- **Status**: [ ] pending

---

### Sub-Task 3: Standalone Tool Integrations (Phase 3)
- **Intent**: Develop and validate standalone tools for SQLite query generation, Tavily/DuckDuckGo web search, and mathematical calculation.
- **Expected Outcomes**:
  - SQLite tool capable of schema introspection, safe SQL generation, execution, and tabular formatting.
  - Web search tool integrating Tavily API (with DuckDuckGo fallback).
  - Deterministic Python calculator / math tool for financial and statistical calculations.
- **Todo List**:
  - [ ] Implement SQLite introspection and query execution tool with query sandboxing.
  - [ ] Build sample SQLite business database (e.g. company revenue, inventory, sales).
  - [ ] Implement Tavily web search tool with fallback search logic.
  - [ ] Implement math calculation engine for deterministic calculations.
  - [ ] Write isolated unit tests for all three tool modules.
- **Relevant Context**: [`Project_Owerview.md`](Project_Owerview.md) sections: Technology Stack, MCP Tools, Phase 3.
- **Status**: [ ] pending

---

### Sub-Task 4: LangGraph Supervisor & Multi-Agent Orchestration (Phase 4)
- **Intent**: Build the LangGraph state machine orchestrating Supervisor Agent, SQL Agent, Research Agent, RAG Agent, and Synthesis Agent.
- **Expected Outcomes**:
  - Typed state representation for LangGraph conversation state, sub-agent thoughts, tool outputs, and route histories.
  - Supervisor Agent correctly routing complex queries to sub-agents (e.g., dual routing for multi-part questions).
  - Synthesizer combining multi-source answers into a coherent final response.
  - Trace of tool decisions and sub-agent execution persisted in SQLite `agent_runs`.
- **Todo List**:
  - [ ] Define LangGraph State schema (`AgentState`).
  - [ ] Implement Supervisor routing logic with prompt templates.
  - [ ] Wrap SQL Agent, Research Agent, and RAG Agent as LangGraph nodes.
  - [ ] Build Response Synthesis node aggregating multi-agent outputs.
  - [ ] Connect state graph edges and conditional routing logic.
  - [ ] Add persistence to record agent execution metrics and tool usage into `agent_runs`.
  - [ ] Connect multi-agent graph execution into FastAPI `/chat`.
- **Relevant Context**: [`Project_Owerview.md`](Project_Owerview.md) sections: High-Level Architecture, Multi-Agent Design, Phase 4.
- **Status**: [ ] pending

---

### Sub-Task 5: Model Context Protocol (MCP) Server Integrations (Phase 5)
- **Intent**: Decouple tool execution using the Model Context Protocol (MCP), providing standardized SQL, Search, and RAG MCP servers/interfaces.
- **Expected Outcomes**:
  - MCP client & server layer wrapping SQL, Tavily Search, RAG retriever, and Calculator.
  - Standardized JSON-RPC protocol support enabling plug-and-play tool replacement.
- **Todo List**:
  - [ ] Implement MCP server interface / wrappers for SQLite Server.
  - [ ] Implement MCP server interface for RAG Server.
  - [ ] Implement MCP server interface for Web Search Server.
  - [ ] Integrate MCP client protocol handlers into the LangGraph agent tools registry.
- **Relevant Context**: [`Project_Owerview.md`](Project_Owerview.md) sections: MCP Tools, Phase 5.
- **Status**: [ ] pending

---

### Sub-Task 6: LangSmith Observability & Tracing (Phase 6)
- **Intent**: Instrument the entire platform with LangSmith for comprehensive tracing, prompt versioning, token monitoring, and error tracking.
- **Expected Outcomes**:
  - Automatic tracing of all LangGraph runs, tool calls, and LLM requests in LangSmith.
  - Session IDs and user IDs correctly tagged in LangSmith run metadata.
  - Error and latency metrics tracked and surfaced.
- **Todo List**:
  - [ ] Configure LangSmith environment variables and client initialization.
  - [ ] Instrument FastAPI request middleware to propagate LangSmith trace IDs and session tags.
  - [ ] Configure callbacks across LangGraph agents and Groq LLM invocations.
  - [ ] Verify observability traces and latency breakdown on sample queries.
- **Relevant Context**: [`Project_Owerview.md`](Project_Owerview.md) sections: LangSmith Integration, Phase 6.
- **Status**: [ ] pending

---

### Sub-Task 7: Comprehensive Evaluation Architecture (Phase 7)
- **Intent**: Implement automated quality evaluations across 5 layers using RAGAS and DeepEval, alongside a benchmark evaluation dataset and dashboard.
- **Expected Outcomes**:
  - Layer 1 (Retrieval): Context Precision, Context Recall, Context Relevancy.
  - Layer 2 (Answer Quality): Faithfulness, Answer Relevancy, Correctness, Completeness.
  - Layer 3 (Hallucination): DeepEval hallucination metric with target < 20%.
  - Layer 4 (Agent): Tool selection accuracy and tool success rate.
  - Layer 5 (End-to-End): Accuracy, latency, and error rate.
  - Evaluation endpoint (`GET /evaluations`) and benchmark test suite saving results into `evaluation_results` table.
- **Todo List**:
  - [ ] Curate benchmark golden dataset (`eval_data/benchmark_dataset.json`).
  - [ ] Implement RAGAS test suite for retrieval and answer quality evaluation.
  - [ ] Implement DeepEval test suite for faithfulness and hallucination scoring.
  - [ ] Implement Critic Agent node in LangGraph to perform runtime output validation.
  - [ ] Store evaluation runs in `evaluation_results` table.
  - [ ] Build evaluation dashboard views in Streamlit displaying metrics, charts, and trends.
- **Relevant Context**: [`Project_Owerview.md`](Project_Owerview.md) sections: Evaluation Architecture, Critic Agent, Phase 7.
- **Status**: [ ] pending

---

### Sub-Task 8: Human-in-the-Loop & Feedback System (Phase 8)
- **Intent**: Enable human approval gates for critical agent actions and collect end-user feedback (thumbs up/down with comments).
- **Expected Outcomes**:
  - Interactive approval workflow (Approve / Reject / Regenerate) in Streamlit for report/summary drafts.
  - Feedback collection endpoint (`POST /feedback`) persisting ratings and comments into `feedback` table.
  - Feedback analytics view for continuous improvement and dataset curation.
- **Todo List**:
  - [ ] Implement LangGraph interrupt / breakpoint mechanism for human review.
  - [ ] Add approval / rejection review UI in Streamlit for pending agent actions.
  - [ ] Implement `POST /feedback` API endpoint and SQLite `feedback` data access layer.
  - [ ] Add thumbs up / down feedback widget to Streamlit chat interface.
- **Relevant Context**: [`Project_Owerview.md`](Project_Owerview.md) sections: Human-in-the-Loop, Feedback System, Phase 8.
- **Status**: [ ] pending

---

### Sub-Task 9: Docker Containerization & CI/CD Pipeline (Phase 9)
- **Intent**: Package the frontend and backend services into Docker containers, create `docker-compose.yml`, and set up GitHub Actions CI/CD with evaluation quality gates.
- **Expected Outcomes**:
  - Multi-stage Dockerfiles for FastAPI and Streamlit.
  - `docker-compose.yml` spinning up FastAPI, Streamlit, ChromaDB, and SQLite volumes with one command.
  - GitHub Actions workflow running Lint -> Unit Tests -> RAGAS -> DeepEval -> Docker Build.
  - Pipeline fails deployment if Faithfulness < 0.80 or Hallucination > 0.20.
- **Todo List**:
  - [ ] Write `Dockerfile.backend` and `Dockerfile.frontend`.
  - [ ] Create `docker-compose.yml` with volume mounts and environment configs.
  - [ ] Create GitHub Actions workflow (`.github/workflows/ci-cd.yml`).
  - [ ] Add CI evaluation step checking faithfulness and hallucination threshold thresholds.
  - [ ] Verify local container build and service communication.
- **Relevant Context**: [`Project_Owerview.md`](Project_Owerview.md) sections: Docker Strategy, CI/CD Pipeline, Phase 9.
- **Status**: [ ] pending

---

### Sub-Task 10: AWS Deployment & Production Hardening (Phase 10)
- **Intent**: Configure production deployment scripts, Nginx reverse proxy, SSL/TLS, environment secrets handling, and EC2 setup guide.
- **Expected Outcomes**:
  - Nginx reverse proxy configuration routing `/api` to FastAPI and root to Streamlit.
  - Deployment scripts and systemd / docker-compose service configurations for AWS EC2 (t3.micro).
  - Complete `README.md` with setup, local development, evaluation execution, and deployment instructions.
- **Todo List**:
  - [ ] Write Nginx configuration for reverse proxying FastAPI and Streamlit.
  - [ ] Create EC2 setup and bootstrapping scripts (`setup_ec2.sh`).
  - [ ] Implement health check and logging monitoring configuration.
  - [ ] Write comprehensive project `README.md` documenting architecture, API endpoints, evaluation benchmarks, and deployment steps.
- **Relevant Context**: [`Project_Owerview.md`](Project_Owerview.md) sections: AWS Deployment Strategy, Estimated Cost, Phase 10.
- **Status**: [ ] pending
