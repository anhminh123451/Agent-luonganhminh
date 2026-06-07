# 🏦 Kế Hoạch Xây Dựng Banking AI Agent — Từ Demo Đến Production API

> **Dự án:** Single-Agent Banking Chatbot với kiến trúc ReAct + LangGraph  
> **Mục tiêu:** Chuyển hóa notebook demo thành một project Python chuyên nghiệp, có thể đóng gói thành REST API và đưa vào CV  
> **Tổng thời gian:** 2 tuần (14 ngày)

---

## 📌 Tổng Quan Kiến Trúc Từ Demo

Dựa trên `Single_Agent_demo.ipynb`, hệ thống hiện tại gồm:

| Thành phần | Công nghệ hiện tại | Vấn đề cần refactor |
|---|---|---|
| LLM backbone | Gemini 2.5 Flash / Llama 3.3 70B (Groq) | Hardcode API key, chưa abstraction |
| Vector Store | ChromaDB (in-memory) | Không persistent, mất data khi restart |
| Embedding | Gemini Embedding-2 + ONNX MiniLM | Trộn lẫn, không nhất quán |
| Tool system | Dict mapping + bare functions | Không có error handling, không typed |
| Agent loop | ReAct (THOUGHT → ACTION → ANSWER) | Prompt nằm trong dict, khó maintain |
| Orchestration | LangGraph StateGraph | Tốt — cần giữ lại và mở rộng |
| State | TypedDict | Cần mở rộng thêm fields |
| API | ❌ Không có | Cần thêm FastAPI |
| Tests | ❌ Không có | Cần viết từ đầu |
| Config | ❌ API key hardcode trong code | Cần `.env` + Pydantic Settings |

---

## 🗂️ Cấu Trúc Project Đề Xuất

```
banking-agent/
├── .env                        # API keys (KHÔNG commit lên Git)
├── .env.example                # Template cho người dùng khác
├── .gitignore
├── README.md
├── requirements.txt
├── pyproject.toml              # Metadata project (nếu dùng Poetry)
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entrypoint
│   │
│   ├── core/                   # Module 1: Cấu hình & Core
│   │   ├── config.py           # Pydantic Settings
│   │   ├── logger.py           # Logging setup
│   │   └── exceptions.py      # Custom exceptions
│   │
│   ├── knowledge_base/         # Module 2: Dữ liệu & Vector Store
│   │   ├── loader.py           # Load & preprocess CSV
│   │   ├── embedder.py         # Embedding wrapper
│   │   ├── vector_store.py     # ChromaDB persistent client
│   │   └── indexer.py          # Build/rebuild index
│   │
│   ├── tools/                  # Module 3: Tool Layer
│   │   ├── base.py             # Abstract BaseTool
│   │   ├── faq_tool.py         # FAQ retrieval tool
│   │   ├── branch_tool.py      # Nearest branch tool
│   │   ├── web_search_tool.py  # Web search tool
│   │   └── registry.py         # Tool registry & permission map
│   │
│   ├── agent/                  # Module 4 & 5: Agent Core
│   │   ├── state.py            # AgentState TypedDict
│   │   ├── prompts.py          # Prompt templates
│   │   ├── profiles.py         # Agent profiles & tool lists
│   │   ├── runner.py           # call_agent() logic
│   │   ├── tool_executor.py    # call_tool() logic
│   │   └── graph.py            # LangGraph workflow
│   │
│   ├── api/                    # Module 6: API Layer
│   │   ├── routes/
│   │   │   ├── chat.py         # POST /chat endpoint
│   │   │   └── health.py       # GET /health
│   │   ├── schemas.py          # Request/Response Pydantic models
│   │   └── dependencies.py    # Dependency injection
│   │
│   └── services/               # Module 7: Business Logic
│       └── chat_service.py     # Orchestrate agent invocation
│
├── data/
│   ├── raw/
│   │   ├── BankFAQs.csv
│   │   └── branch_info.csv
│   └── chroma_db/              # Persistent ChromaDB storage
│
├── tests/                      # Module 8: Testing
│   ├── unit/
│   │   ├── test_tools.py
│   │   ├── test_agent.py
│   │   └── test_knowledge_base.py
│   └── integration/
│       └── test_api.py
│
└── Dockerfile
```

---

## 📅 Kế Hoạch Chi Tiết Theo Tuần

---

### 🔧 TUẦN 1 — Xây Dựng Backbone

---

#### Module 1 — Project Setup & Configuration
**⏱️ Thời gian: Ngày 1 (4–6 giờ)**

**Công việc cụ thể:**
- Tạo cấu trúc thư mục theo layout ở trên
- Khởi tạo Git repo, viết `.gitignore` (đặc biệt ignore `.env`, `chroma_db/`)
- Thiết lập `requirements.txt` với version pinning
- Viết `app/core/config.py` dùng Pydantic Settings để đọc `.env`
- Thiết lập logging chuẩn với Python `logging` module
- Viết `app/core/exceptions.py` cho custom exceptions

**Kiến thức cần nắm:**
- **Pydantic BaseSettings** — đọc biến môi trường có type validation, hỗ trợ `.env` file tự động
- **Python `logging` module** — cấu hình logger theo từng module, log levels (DEBUG/INFO/WARNING/ERROR)
- **Git best practices** — `.gitignore`, commit convention (feat/fix/docs), không commit secret
- **`python-dotenv`** — load `.env` vào `os.environ`
- **Dependency pinning** — tại sao cần pin version trong `requirements.txt` để tránh breaking changes

**Ví dụ output (config.py):**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    CHROMA_DB_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "bank_faq"
    LLM_PROVIDER: str = "gemini"  # hoặc "groq"
    MAX_AGENT_STEPS: int = 10

    class Config:
        env_file = ".env"

settings = Settings()
```

---

#### Module 2 — Knowledge Base & Vector Store
**⏱️ Thời gian: Ngày 2–3 (8–10 giờ)**

**Công việc cụ thể:**
- Refactor logic load CSV vào `loader.py` với error handling đầy đủ
- Tách embedding logic ra `embedder.py` dưới dạng class có interface thống nhất
- Chuyển ChromaDB từ **in-memory** sang **persistent client** (lưu xuống `data/chroma_db/`)
- Viết `indexer.py` để build/rebuild index (chỉ index lại khi data thay đổi, kiểm tra bằng hash)
- Unit test cho loader và vector store

**Kiến thức cần nắm:**
- **ChromaDB persistent client** — khác gì `chromadb.Client()` (ephemeral) vs `chromadb.PersistentClient(path=...)` (lưu disk)
- **Embedding fundamentals** — embedding là gì, tại sao cần normalize, cosine similarity vs dot product
- **RAG pipeline** — Retrieval-Augmented Generation: chunk → embed → store → retrieve → generate
- **Data preprocessing** — tạo `combine_text` field từ Question + Answer + Class (đã có trong demo), mở rộng thêm cleaning
- **Content hashing** — dùng `hashlib.md5` để phát hiện khi data thay đổi, tránh index lại vô ích

**Ví dụ output (vector_store.py):**
```python
import chromadb
from app.core.config import settings

class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self._collection = self._client.get_or_create_collection(
            name=settings.COLLECTION_NAME
        )

    def query(self, query_text: str, n_results: int = 3) -> list[dict]:
        results = self._collection.query(
            query_texts=[query_text], n_results=n_results
        )
        return results["documents"][0]

    def add_documents(self, ids, documents, metadatas):
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
```

---

#### Module 3 — Tool Layer
**⏱️ Thời gian: Ngày 3–4 (8–10 giờ)**

**Công việc cụ thể:**
- Tạo `BaseTool` abstract class với interface chuẩn
- Refactor 3 tools từ demo thành 3 class riêng biệt: `FAQTool`, `BranchSearchTool`, `WebSearchTool`
- Xây dựng `ToolRegistry` để quản lý tools và permission theo agent
- Thêm proper error handling cho từng tool (try/except, trả về error message rõ ràng)
- Thay web search mock bằng real API (DuckDuckGo hoặc Tavily)
- Unit test cho từng tool

**Kiến thức cần nắm:**
- **Abstract Base Classes (ABC)** — `from abc import ABC, abstractmethod`, tại sao dùng interface pattern
- **Haversine formula** — tính khoảng cách địa lý giữa 2 tọa độ GPS (đã có trong demo, cần refactor clean hơn)
- **Tool design pattern trong AI** — tại sao tool cần: `name`, `description`, `args_schema`, `run()`
- **Pydantic BaseModel** — validate input arguments cho tool call, tạo typed schemas
- **Tavily Search API / DuckDuckGo API** — tích hợp web search thực thay cho mock data
- **Error handling patterns** — fail-safe returns, không throw exception ra ngoài tool

**Ví dụ output (base.py):**
```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class ToolResult(BaseModel):
    context: str
    source: str
    success: bool = True
    error: str | None = None

class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        pass

    def safe_run(self, **kwargs) -> ToolResult:
        try:
            return self.run(**kwargs)
        except Exception as e:
            return ToolResult(context="", source=self.name, success=False, error=str(e))
```

---

#### Module 4 — Agent Core & Prompt Engineering
**⏱️ Thời gian: Ngày 4–5 (8–10 giờ)**

**Công việc cụ thể:**
- Tách `AgentState` ra `state.py` với full type annotations
- Tách prompt templates ra `prompts.py` (dùng Jinja2 hoặc LangChain PromptTemplate)
- Tách agent profiles ra `profiles.py`, đọc từ YAML file thay vì hardcode
- Refactor `call_agent()` thành `AgentRunner` class
- Refactor `call_tool()` thành `ToolExecutor` class với robust JSON parsing
- Cải thiện `should_continue()` router: thêm HANDOFF handling (để chuẩn bị multi-agent sau này)

**Kiến thức cần nắm:**
- **ReAct (Reasoning + Acting) pattern** — tại sao loop THOUGHT→ACTION→OBSERVATION→THOUGHT hiệu quả
- **LangChain PromptTemplate & ChatPromptTemplate** — cách dùng placeholder, `invoke()`, message roles
- **Jinja2 templating** (tùy chọn) — alternative cho prompt management, dễ edit hơn
- **JSON parsing defensive coding** — brace counting algorithm (đã có trong demo), regex fallback, `json.loads` vs `ast.literal_eval`
- **Agent state management** — tại sao cần immutable state, cách LangGraph copy state giữa các node
- **System prompt design** — few-shot examples trong system prompt, rõ ràng về output format

---

#### Module 5 — LangGraph Workflow
**⏱️ Thời gian: Ngày 5–6 (6–8 giờ)**

**Công việc cụ thể:**
- Tách toàn bộ graph definition ra `graph.py`
- Compile graph một lần (singleton pattern) để không compile lại mỗi request
- Thêm `checkpointer` cho LangGraph để support conversation memory (optional ở base, dễ bật sau)
- Viết `get_graph()` factory function với caching
- Visualize và document luồng chạy của agent bằng Mermaid diagram

**Kiến thức cần nắm:**
- **LangGraph StateGraph** — `add_node()`, `add_edge()`, `add_conditional_edges()`, `compile()`
- **Graph patterns** — tại sao dùng graph thay vì while-loop đơn giản (observability, control flow rõ ràng)
- **Singleton pattern** — compile graph một lần, reuse nhiều lần (tránh overhead)
- **LangGraph checkpointer** — `MemorySaver` cho in-memory persistence, `SqliteSaver` cho disk
- **Mermaid diagram** — vẽ flowchart bằng text, dùng trong README

---

### 🚀 TUẦN 2 — API, Testing & Packaging

---

#### Module 6 — FastAPI REST API
**⏱️ Thời gian: Ngày 7–9 (10–12 giờ)**

**Công việc cụ thể:**
- Setup FastAPI app với lifespan events (khởi tạo graph, vector store khi startup)
- Định nghĩa Pydantic schemas cho Request/Response
- Implement `POST /chat` endpoint nhận query, trả về agent response
- Implement `GET /health` endpoint kiểm tra trạng thái các dependencies
- Implement `POST /index/rebuild` endpoint trigger reindex knowledge base
- Thêm CORS middleware, request ID tracking
- Viết `chat_service.py` tách business logic khỏi route handler
- Test manual với `httpx` hoặc Swagger UI tự động của FastAPI

**Kiến thức cần nắm:**
- **FastAPI fundamentals** — `@app.post()`, path/query/body params, response models, status codes
- **Pydantic v2** — request validation tự động, custom validators, `model_validator`
- **Async vs Sync** — LangGraph invoke là sync, cần dùng `asyncio.run_in_executor()` hoặc `run_sync()` từ Starlette để không block event loop
- **FastAPI lifespan** — `@asynccontextmanager` pattern để init/cleanup resources (thay cho deprecated `@app.on_event`)
- **Dependency Injection** — `Depends()` để inject graph, vector store vào routes
- **HTTP status codes** — 200 OK, 422 Unprocessable Entity, 500 Internal Server Error
- **OpenAPI/Swagger** — FastAPI tự gen, cách viết docstring để docs đẹp

**Ví dụ schemas.py:**
```python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    user_location: str = "Hanoi, Vietnam"
    session_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    num_steps: int
    tool_observations: list[str]
    session_id: str
```

---

#### Module 7 — Testing
**⏱️ Thời gian: Ngày 9–11 (8–10 giờ)**

**Công việc cụ thể:**
- Viết unit tests cho từng tool (`tests/unit/test_tools.py`)
- Viết unit tests cho knowledge base loader và vector store
- Viết unit tests cho agent state transitions và tool executor
- Viết integration tests cho FastAPI endpoints dùng `TestClient`
- Thiết lập `pytest.ini` và coverage report với `pytest-cov`
- Viết fixtures (`conftest.py`) cho mock data và mock LLM

**Kiến thức cần nắm:**
- **pytest** — fixtures, parametrize, conftest.py, marks (unit/integration)
- **unittest.mock** — `patch()`, `MagicMock()`, mock LLM responses để test không tốn API calls
- **FastAPI TestClient** — `from fastapi.testclient import TestClient`, test HTTP endpoints
- **Test coverage** — `pytest-cov`, tại sao cần coverage ≥ 70% cho project CV
- **Test design** — Arrange-Act-Assert pattern, test cả happy path lẫn edge cases

---

#### Module 8 — Packaging & Documentation
**⏱️ Thời gian: Ngày 12–14 (8–10 giờ)**

**Công việc cụ thể:**
- Viết `README.md` đầy đủ: overview, architecture diagram, quickstart, API docs
- Viết `Dockerfile` multi-stage để containerize app
- Viết `docker-compose.yml` để chạy app + ChromaDB (nếu dùng external)
- Tạo GitHub Actions CI pipeline: chạy tests tự động khi push
- Viết `CHANGELOG.md` ghi lại các version
- Tag release `v1.0.0` trên Git

**Kiến thức cần nắm:**
- **Docker fundamentals** — `FROM`, `COPY`, `RUN`, `CMD`, multi-stage builds, `.dockerignore`
- **Docker Compose** — `services`, `volumes`, `environment`, `depends_on`
- **GitHub Actions** — viết `.github/workflows/ci.yml`, trigger on push/PR, cache pip dependencies
- **Semantic Versioning** — MAJOR.MINOR.PATCH, tại sao quan trọng
- **Technical writing** — cách viết README thu hút: badges, GIF demo, clear quickstart

---

## 📊 Timeline Tổng Hợp

```
Tuần 1                                    Tuần 2
Mo  Tu  We  Th  Fr  Sa  Su | Mo  Tu  We  Th  Fr  Sa  Su
[1] [2--3] [3-4] [4--5] [5-6]  | [--7--] [7--9] [9-11] [12---14]
 │    │      │     │      │         │       │       │       │
Setup KB   Tools Agent  Graph     FastAPI Tests  Tests  Docs &
            Store Core  Work-     API     unit   integ  Docker
                        flow
```

| # | Module | Ngày | Giờ ước tính |
|---|--------|------|-------------|
| 1 | Project Setup & Config | 1 | 4–6h |
| 2 | Knowledge Base & Vector Store | 2–3 | 8–10h |
| 3 | Tool Layer | 3–4 | 8–10h |
| 4 | Agent Core & Prompts | 4–5 | 8–10h |
| 5 | LangGraph Workflow | 5–6 | 6–8h |
| 6 | FastAPI REST API | 7–9 | 10–12h |
| 7 | Testing | 9–11 | 8–10h |
| 8 | Packaging & Docs | 12–14 | 8–10h |
| **Total** | | **14 ngày** | **~60–76h** |

> **Lưu ý:** Lịch trên giả định ~5–6 giờ học/code mỗi ngày. Nếu ít thời gian hơn, có thể rút ngắn Module 7 (test ít coverage hơn) và Module 8 (bỏ Docker trước).

---

## 🚀 Gợi Ý Scale Dự Án Sau Khi Hoàn Thành Baseline

Sau khi đã có một production-ready single agent API, đây là các hướng mở rộng từ thấp đến cao về độ phức tạp:

---

### Cấp 1 — Cải Thiện Chất Lượng Agent (1–2 tuần thêm)

#### 1.1 Conversation Memory
Hiện tại mỗi request là stateless. Thêm memory để agent nhớ ngữ cảnh hội thoại:
- Dùng LangGraph `MemorySaver` checkpointer với `thread_id` theo `session_id`
- Lưu conversation history vào Redis hoặc SQLite
- **Kiến thức cần:** Redis basics, session management, context window management

#### 1.2 Hybrid Search (BM25 + Vector)
ChromaDB vector search đôi khi miss khi câu hỏi có keyword cụ thể:
- Kết hợp BM25 (full-text search) với vector search
- Dùng `rank_bm25` library + ChromaDB, merge results bằng Reciprocal Rank Fusion (RRF)
- **Kiến thức cần:** BM25 algorithm, RRF, ensemble retrieval

#### 1.3 Reranking
Sau khi retrieve top-K documents, dùng cross-encoder để rerank:
- Tích hợp `sentence-transformers` cross-encoder hoặc Cohere Rerank API
- **Kiến thức cần:** Bi-encoder vs Cross-encoder, reranking pipeline

---

### Cấp 2 — Multi-Agent Architecture (2–4 tuần thêm)

#### 2.1 Supervisor Pattern
Thêm một `Supervisor Agent` điều phối các specialized agents:
```
User → Supervisor Agent
              ├── FAQ Agent       (câu hỏi thông thường)
              ├── Loan Agent      (tư vấn vay vốn, tính DTI)
              ├── Branch Agent    (tìm chi nhánh)
              └── Escalation Agent (vấn đề phức tạp)
```
- **Kiến thức cần:** Multi-agent patterns (supervisor, hierarchical, collaborative), LangGraph subgraph

#### 2.2 Specialized Loan Agent
Tách Loan logic thành agent riêng với công thức tính DTI (Debt-to-Income ratio):
- Tools riêng: tính lãi suất, tra bảng sản phẩm vay, kiểm tra điều kiện
- **Kiến thức cần:** Financial domain knowledge, tool design cho domain-specific agent

---

### Cấp 3 — Observability & Production Readiness (1–2 tuần thêm)

#### 3.1 LangSmith Tracing
Tích hợp LangSmith để trace từng bước của agent:
- Xem token usage, latency, tool calls theo từng request
- Tạo dataset từ production traces để eval
- **Kiến thức cần:** LangSmith SDK, `@traceable` decorator, tracing concepts

#### 3.2 Evaluation Framework
Xây dựng pipeline đánh giá chất lượng agent tự động:
- RAGAS metrics: faithfulness, answer relevancy, context recall
- Tạo test set từ FAQ data, chạy eval định kỳ
- **Kiến thức cần:** RAGAS framework, LLM-as-judge pattern, evaluation metrics

#### 3.3 Structured Logging & Monitoring
- Dùng `structlog` để log JSON, tích hợp với Grafana/Datadog
- Track metrics: latency, tool_call_count, error_rate, token_usage
- **Kiến thức cần:** Prometheus metrics, structured logging, observability pillars (logs/metrics/traces)

---

### Cấp 4 — Scalability & Advanced Features (4–8 tuần thêm)

#### 4.1 Streaming Response
Thay vì chờ agent chạy xong mới trả response, stream từng token:
- FastAPI `StreamingResponse` + LangGraph streaming
- Frontend nhận `text/event-stream` (Server-Sent Events)
- **Kiến thức cần:** Python generators, SSE protocol, async streaming

#### 4.2 Caching Layer
Cache kết quả FAQ retrieval và LLM response để giảm latency và chi phí:
- Semantic caching với Redis: cache response theo embedding similarity
- **Kiến thức cần:** Redis, semantic similarity caching, cache invalidation strategies

#### 4.3 Authentication & Rate Limiting
Bảo vệ API endpoint trước khi deploy public:
- JWT authentication với `python-jose`
- Rate limiting với `slowapi` (Redis-backed)
- **Kiến thức cần:** JWT, OAuth2, rate limiting algorithms (token bucket, sliding window)

#### 4.4 Fine-tuning & Prompt Optimization
Nâng cao chất lượng model với data thực tế:
- Thu thập feedback từ user (thumbs up/down)
- Fine-tune LLM nhỏ (ví dụ Llama 3.1 8B) trên banking domain
- Dùng DSPy để optimize prompts tự động
- **Kiến thức cần:** PEFT/LoRA fine-tuning, preference data collection, DSPy framework

---

## 💼 Giá Trị CV Từ Dự Án Này

Khi hoàn thành baseline (2 tuần), bạn có thể trình bày:

```
Banking AI Agent — Single Agent với ReAct Architecture
• Xây dựng production-ready AI agent sử dụng LangGraph + Gemini/Groq LLM
• Thiết kế RAG pipeline với ChromaDB persistent vector store và custom embedding
• Implement 3 tools: FAQ retrieval, geospatial branch search (Haversine), web search
• Đóng gói thành REST API với FastAPI, bao gồm CI/CD và Docker containerization
• Viết test suite với pytest, coverage > 70%
Tech stack: Python · LangChain · LangGraph · ChromaDB · FastAPI · Docker · Gemini API
```

Với mỗi cấp scale thêm, bạn có thể bổ sung:
- **Cấp 2:** "Mở rộng sang multi-agent architecture với Supervisor pattern"
- **Cấp 3:** "Tích hợp LangSmith observability và RAGAS evaluation framework"
- **Cấp 4:** "Implement streaming response và semantic caching với Redis"

---

*Kế hoạch được tạo dựa trên phân tích `Single_Agent_demo.ipynb` — Banking AI Chatbot với LangGraph + ChromaDB + Gemini/Groq*
