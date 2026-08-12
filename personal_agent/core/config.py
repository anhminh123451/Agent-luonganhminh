from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    CHROMA_DB_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "user_data"
    MODEL_LLM: str = "gemini-3.5-flash"  # Model chính cho Gemini
    GROQ_MODEL: str = "llama-3.3-70b-versatile"  # Model cho Groq
    LLM_PROVIDER: str = "gemini"  # Provider chính: "gemini" hoặc "groq"
    LLM_FALLBACK_ENABLED: bool = True  # Tự động fallback khi hết quota
    EMBEDDING_PROVIDER: str = "gemini"  # Provider embedding (gemini, openai, ...)
    EMBEDDING_MODEL: str = "gemini-embedding-2"  # Model embedding cụ thể
    CHECKPOINT_DB_PATH: str = "./data/checkpoints.sqlite"
    MAX_AGENT_STEPS: int = 10
    LOG_LEVEL: str = "INFO"   # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_DIR: str = "logs"      # Thư mục lưu file log
    
    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60*24
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    
    # Database
    DATABASE_URL: str
    
    class Config:
        env_file = ".env"


settings = Settings()


