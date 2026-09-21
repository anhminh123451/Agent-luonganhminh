from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DeepSeek LLM
    DEEPSEEK_API_KEY: str
    DEEPSEEK_MODEL: str = "deepseek-flash"  # DeepSeek V4 Flash
    MAX_TOKENS_OUTPUT : int = 4000
    TEMPERATURE: float = 0.7

    # Gemini (chỉ dùng cho Embedding)
    GEMINI_API_KEY: str = ""

    CHROMA_DB_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "user_data"
    # ChromaDB Remote (Fly.io) 
    CHROMA_HOST: str = ""
    CHROMA_PORT: int = 443
    CHROMA_AUTH_TOKEN: str = ""
    
    LLM_PROVIDER:str = "deepseek"
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


