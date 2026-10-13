from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    PRIMARY_MODEL: str = "llama-3.3-70b-versatile"
    FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    TAVILY_API_KEY: str = ""

    APP_DATABASE_URL: str = "sqlite:///./data/app.db"
    BUSINESS_DB_PATH: str = "./data/business_data.db"
    UPLOADS_DIR: str = "./data/uploads"
    CHROMA_DIR: str = "./chroma_db"
    RAG_COLLECTION: str = "rag_documents"
    
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "agentic-ai-platform"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
