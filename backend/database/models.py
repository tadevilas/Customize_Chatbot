import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.config import settings

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

engine = create_engine(
    settings.APP_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.APP_DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(String(100), default="default_user", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(100), index=True, nullable=False)
    role = Column(String(50), nullable=False)  # "user", "assistant", "system"
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    rating = Column(String(20), nullable=False)  # "helpful", "not_helpful"
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    query = Column(Text, nullable=False)
    tools_used = Column(String(255), nullable=True)
    latency = Column(Float, nullable=True)
    status = Column(String(50), default="success")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    query = Column(Text, nullable=False)
    faithfulness = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)
    hallucination_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
