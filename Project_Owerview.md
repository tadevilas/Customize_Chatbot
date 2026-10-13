# Production-Ready Agentic AI Platform

## Overview

This project aims to build a production-ready Agentic AI Platform using mostly open-source and free technologies while demonstrating modern GenAI engineering practices.

The system will:

- Use Groq-hosted open-source LLMs
- Support RAG (Retrieval-Augmented Generation)
- Support SQL querying through an MCP tool
- Support web search using Tavily
- Use LangGraph for agent orchestration
- Use LangSmith for tracing and observability
- Include automated evaluation using RAGAS and DeepEval
- Support Human-in-the-Loop workflows
- Be containerized using Docker
- Be deployed on AWS
- Use GitHub Actions for CI/CD

---

# Project Objectives

Build an enterprise-style Agentic AI application that can:

1. Answer questions from uploaded documents
2. Search the internet
3. Query databases
4. Route requests to proper tools
5. Evaluate answer quality
6. Monitor agent decisions
7. Collect user feedback
8. Run in production on AWS

---

# High-Level Architecture

```text
                     ┌────────────────┐
                     │ Streamlit UI   │
                     └───────┬────────┘
                             │
                             ▼

                     ┌────────────────┐
                     │ FastAPI API    │
                     └───────┬────────┘
                             │
                             ▼

                     ┌────────────────┐
                     │ LangGraph      │
                     │ Supervisor     │
                     └───────┬────────┘

      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
      ▼                      ▼                      ▼

 SQL Agent             Search Agent           RAG Agent
(SQLite)               (Tavily)             (ChromaDB)

      └──────────────────────┼──────────────────────┘
                             │
                             ▼

                 Response Synthesis Agent

                             │
                             ▼

                   Evaluation Pipeline

                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼

     RAGAS             DeepEval         Human Feedback

                             │
                             ▼

                        LangSmith
```

---

# Technology Stack

## LLM

### Groq

Primary Model:

```text
llama-3.3-70b-versatile
```

Fallback Model:

```text
llama-3.1-8b-instant
```

Benefits:

- Free / low cost
- Very fast inference
- Open-source models
- LangChain integration

---

## Agent Framework

```text
LangGraph
LangChain
```

Purpose:

- Agent orchestration
- Multi-agent workflows
- Tool routing
- Human approval workflows

---

## Backend

```text
FastAPI
```

Responsibilities:

- API layer
- Agent execution
- File uploads
- Evaluation APIs

---

## Frontend

```text
Streamlit
```

Features:

- Chat UI
- File upload
- Agent trace view
- Evaluation dashboards
- Feedback collection

---

## Database

```text
SQLite
```

Purpose:

- Chat history
- Agent runs
- User feedback
- Evaluation metrics

Benefits:

- Free
- Lightweight
- Easy deployment

---

## Vector Database

```text
ChromaDB
```

Purpose:

- Store embeddings
- Retrieve document chunks

Benefits:

- Open source
- Local storage
- Free

---

## Embeddings

Recommended:

```text
BAAI/bge-small-en-v1.5
```

Alternative:

```text
all-MiniLM-L6-v2
```

Cost:

```text
$0
```

---

## Reranker

Recommended:

```text
BAAI/bge-reranker-base
```

Purpose:

- Improve retrieval quality
- Better context ranking

---

## Search Tool

Primary:

```text
Tavily
```

Alternative:

```text
DuckDuckGo Search
```

---

# MCP Tools

## MCP Tool 1: SQL Server

Capabilities:

- List tables
- Describe schema
- Generate SQL
- Execute SQL
- Return results

Database:

```text
SQLite
```

---

## MCP Tool 2: RAG Server

Capabilities:

- Document upload
- Chunking
- Embedding generation
- Retrieval
- Citation generation

Storage:

```text
ChromaDB
```

---

## MCP Tool 3: Search Server

Capabilities:

- Web search
- Research mode
- News search

Tool:

```text
Tavily
```

---

## MCP Tool 4: Calculator

Capabilities:

- Mathematics
- Statistics
- Financial calculations

---

# Multi-Agent Design

## Supervisor Agent

Responsibilities:

- Analyze user request
- Choose appropriate tools
- Coordinate workflow

---

## SQL Agent

Responsibilities:

- Database querying
- SQL generation
- Result summarization

Tools:

```text
SQLite MCP
```

---

## Research Agent

Responsibilities:

- Web research
- Information gathering

Tools:

```text
Tavily MCP
```

---

## RAG Agent

Responsibilities:

- Document retrieval
- Context generation
- Citation extraction

Tools:

```text
Chroma MCP
```

---

## Critic Agent

Responsibilities:

- Detect hallucinations
- Validate outputs
- Evaluate answers

Tools:

```text
DeepEval
RAGAS
```

---

# RAG Pipeline

## Step 1

Document Upload

Supported:

- PDF
- DOCX
- PPTX
- TXT

---

## Step 2

Document Parsing

Libraries:

```text
PyPDFLoader
DocxLoader
Unstructured
```

---

## Step 3

Chunking

Configuration:

```text
Chunk Size: 1000
Overlap: 200
```

Use:

```text
RecursiveCharacterTextSplitter
```

---

## Step 4

Embedding Generation

Model:

```text
BAAI/bge-small-en-v1.5
```

---

## Step 5

Store Embeddings

Database:

```text
ChromaDB
```

---

## Step 6

Retrieval

Top K:

```text
5
```

---

## Step 7

Reranking

Model:

```text
BAAI/bge-reranker-base
```

---

## Step 8

Answer Generation

LLM:

```text
Groq
```

---

# Human-in-the-Loop

Workflow:

```text
User Query
     |
     ▼
Agent Draft
     |
     ▼
Human Review
     |
 ┌───┴────┐
 │        │
Approve  Reject
 │        │
 ▼        ▼
Continue Re-generate
```

Use Cases:

- Email writing
- Report generation
- Executive summaries

---

# Evaluation Architecture

Evaluation is a mandatory component.

## Layer 1: Retrieval Evaluation

Tool:

```text
RAGAS
```

Metrics:

- Context Precision
- Context Recall
- Context Relevancy

---

## Layer 2: Answer Quality Evaluation

Tools:

```text
RAGAS
DeepEval
```

Metrics:

- Faithfulness
- Answer Relevancy
- Correctness
- Completeness

---

## Layer 3: Hallucination Detection

Tool:

```text
DeepEval
```

Metrics:

- Hallucination Score

Goal:

```text
Hallucination < 20%
```

---

## Layer 4: Agent Evaluation

Metrics:

- Tool Selection Accuracy
- Tool Success Rate
- Agent Completion Rate

Example:

Question:

```text
Show IBM revenue and recent AI news
```

Expected:

```text
SQL Tool + Tavily Tool
```

---

## Layer 5: End-to-End Evaluation

Metrics:

- Response Accuracy
- Response Relevancy
- Workflow Completion
- Average Latency
- Error Rate

---

# Feedback System

Create a feedback mechanism.

Users can provide:

```text
👍 Helpful
👎 Not Helpful
```

Store:

```text
Question
Answer
Rating
Comments
Timestamp
```

Purpose:

- Continuous improvement
- Evaluation dataset generation

---

# LangSmith Integration

Track:

- Prompt versions
- Tool calls
- Agent decisions
- Token usage
- Latency
- Errors

Purpose:

- Observability
- Debugging
- Optimization

---

# Database Design

## chat_sessions

```text
id
session_id
user_id
created_at
```

---

## messages

```text
id
session_id
role
message
timestamp
```

---

## feedback

```text
id
question
response
rating
comments
created_at
```

---

## agent_runs

```text
id
query
tools_used
latency
status
created_at
```

---

## evaluation_results

```text
id
query
faithfulness
context_recall
context_precision
hallucination_score
overall_score
```

---

# API Endpoints

## Chat

```http
POST /chat
```

---

## Upload Document

```http
POST /upload
```

---

## Feedback

```http
POST /feedback
```

---

## Evaluation Dashboard

```http
GET /evaluations
```

---

## Health Check

```http
GET /health
```

---

# Docker Strategy

Containers:

```text
FastAPI
Streamlit
```

Use:

```text
Docker Compose
```

For local development.

---

# CI/CD Pipeline

GitHub Actions Workflow

```text
Code Push
    |
    ▼
Lint
    |
    ▼
Unit Tests
    |
    ▼
RAGAS Tests
    |
    ▼
DeepEval Tests
    |
    ▼
Docker Build
    |
    ▼
Deploy
```

Deployment should fail if:

```text
Faithfulness < 0.80
```

or

```text
Hallucination > 0.20
```

---

# Recommended Repository Structure

```text
agentic-ai-platform/
│
├── frontend/
│   └── streamlit_app.py
│
├── backend/
│   ├── api/
│   ├── agents/
│   ├── graphs/
│   ├── tools/
│   ├── mcp/
│   ├── rag/
│   ├── evals/
│   │   ├── ragas/
│   │   ├── deepeval/
│   │   ├── datasets/
│   │   └── reports/
│   ├── database/
│   └── main.py
│
├── chroma_db/
│
├── data/
│   ├── documents/
│   ├── uploads/
│   └── eval_data/
│
├── tests/
│
├── docker/
│
├── .github/
│   └── workflows/
│
├── docker-compose.yml
│
├── requirements.txt
│
└── README.md
```

---

# AWS Deployment Strategy

## Recommended

Use:

```text
EC2 t3.micro
```

Host:

- FastAPI
- Streamlit
- SQLite
- ChromaDB

using:

```text
Docker Compose
```

Benefits:

- Cheapest AWS option
- Easy setup
- Great for portfolios

---

## Future Upgrade

Move to:

```text
ECS Fargate
```

When scaling is required.

---

# Estimated Cost

## Development

```text
Groq           : Free
SQLite         : Free
ChromaDB       : Free
Streamlit      : Free
FastAPI        : Free
Docker         : Free
GitHub Actions : Free
LangGraph      : Free
LangChain      : Free
```

Monthly development cost:

```text
~ $0
```

---

## AWS Deployment

EC2 t3.micro:

```text
~ $8-$12/month
```

Optional domain:

```text
~ $10-15/year
```

---

# Development Roadmap

## Phase 1

Basic Chatbot

- Groq
- FastAPI
- Streamlit

---

## Phase 2

RAG

- ChromaDB
- BGE Embeddings
- PDF Upload

---

## Phase 3

Tools

- SQLite Agent
- Tavily Search
- Calculator

---

## Phase 4

LangGraph

- Supervisor Agent
- Multi-Agent Workflow

---

## Phase 5

MCP

- SQL MCP
- Search MCP
- RAG MCP

---

## Phase 6

Observability

- LangSmith

---

## Phase 7

Evaluation

- RAGAS
- DeepEval
- Hallucination Detection
- Benchmark Dataset

---

## Phase 8

Human-in-the-Loop

- Approval Workflow
- Feedback Collection

---

## Phase 9

Docker & GitHub Actions

- Docker Compose
- CI/CD
- Automated Evals

---

## Phase 10

AWS Deployment

- EC2
- Nginx
- SSL
- Monitoring

---
