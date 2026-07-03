from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    CHROMA_DB_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "bank_faq"
    MODEL_LLM: str = "gemini-2.5-flash"  # hoặc "openai/gpt-5.1"
    LLM_PROVIDER: str = "gemini"  # hoặc "groq"
    EMBEDDING_PROVIDER: str = "gemini"  # Provider embedding (gemini, openai, ...)
    EMBEDDING_MODEL: str = "gemini-embedding-2"  # Model embedding cụ thể
    CHECKPOINT_DB_PATH: str = "./data/checkpoints.sqlite"
    MAX_AGENT_STEPS: int = 10
    LOG_LEVEL: str = "INFO"   # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_DIR: str = "logs"      # Thư mục lưu file log

    class Config:
        env_file = ".env"


settings = Settings()


