from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    MAX_RESULTS_WEBSEARCH : int = 3
    # DeepSeek LLM
    DEEPSEEK_API_KEY: str
    DEEPSEEK_MODEL: str = "deepseek-flash"  # DeepSeek V4 Flash
    MAX_TOKENS_OUTPUT : int = 4000
    TEMPERATURE: float = 0.7

    # Gemini (chỉ dùng cho Embedding)
    GEMINI_API_KEY: str = ""

    # Tavily Web Search
    TAVILY_API_KEY: str = ""

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

    # ─── Supabase Storage (Tầng Gốc - BM25 Index) ────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    SUPABASE_BM25_BUCKET: str = ""

    # ─── Redis (Tầng 2 - Distributed Cache) ───────────────────────────
    REDIS_URL: str = ""
    REDIS_BM25_TTL: int = 86400          # 24 giờ
    REDIS_LOCK_TIMEOUT: int = 10         # 10 giây chống Cache Stampede

    # ─── Local RAM (Tầng 1 - In-Process Cache) ────────────────────────
    L1_CACHE_MAXSIZE: int = 100          # Tối đa 100 users active
    L1_CACHE_TTL: int = 1800             # 30 phút

    # ─── Hybrid Search Params ─────────────────────────────────────────
    HYBRID_DENSE_WEIGHT: float = 0.7
    HYBRID_SPARSE_WEIGHT: float = 0.3
    HYBRID_RRF_K: int = 60

    # ─── Groq Re-ranker Settings ─────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_RERANK_MODEL: str = "openai/gpt-oss-120b"
    RERANKER_ENABLED: bool = True
    RERANKER_TOP_K_CANDIDATES: int = 10
    RERANKER_MIN_SCORE: float = 5.0
    RERANKER_TIMEOUT_SECONDS: float = 5.0
    N_RESULT_RERANK:int = 5
    N_RESULT_RETRIEVEL:int = 20
    N_RESULT_RRF:int = 10

    class Config:
        env_file = ".env"


settings = Settings()

