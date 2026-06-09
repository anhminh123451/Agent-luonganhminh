from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    CHROMA_DB_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "bank_faq"
    LLM_PROVIDER: str = "gemini"  # hoặc "groq"
    MAX_AGENT_STEPS: int = 5
    LOG_LEVEL: str = "DEBUG"   # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_DIR: str = "logs"      # Thư mục lưu file log

    class Config:
        env_file = ".env"


settings = Settings()


