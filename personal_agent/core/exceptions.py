"""
Custom Exceptions cho Banking AI Agent.

Module này định nghĩa hệ thống exception phân cấp cho toàn bộ project,
giúp xử lý lỗi rõ ràng, có ngữ nghĩa và dễ debug.

Cấu trúc phân cấp:
    BankingAgentError (base)
    ├── ConfigurationError         — Lỗi cấu hình (.env, settings)
    ├── KnowledgeBaseError         — Lỗi liên quan knowledge base
    │   ├── DataLoadError          — Lỗi load/parse CSV
    │   ├── EmbeddingError         — Lỗi tạo embedding
    │   └── VectorStoreError       — Lỗi kết nối/query ChromaDB
    ├── ToolError                  — Lỗi liên quan tool system
    │   ├── ToolNotFoundError      — Tool không tồn tại trong registry
    │   ├── ToolExecutionError     — Lỗi khi chạy tool
    │   └── ToolValidationError    — Lỗi validate input arguments
    ├── AgentError                 — Lỗi agent core
    │   ├── AgentStepLimitError    — Agent vượt quá MAX_AGENT_STEPS
    │   ├── LLMResponseError       — Lỗi parse response từ LLM
    │   └── GraphExecutionError    — Lỗi chạy LangGraph workflow
    └── APIError                   — Lỗi API layer
        ├── InvalidRequestError    — Request không hợp lệ
        └── ServiceUnavailableError — Dependency không khả dụng

Cách sử dụng:
    from core.exceptions import ToolExecutionError, DataLoadError

    # Raise exception với message rõ ràng
    raise ToolExecutionError("FAQ tool failed: ChromaDB connection timeout")

    # Catch theo nhóm module
    try:
        ...
    except KnowledgeBaseError as e:
        logger.error(f"Knowledge base error: {e}")

    # Catch toàn bộ lỗi hệ thống
    try:
        ...
    except BankingAgentError as e:
        logger.error(f"System error: {e}")
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════
# BASE EXCEPTION
# ═══════════════════════════════════════════════════════════════════════

class BankingAgentError(Exception):
    """
    Base exception cho toàn bộ Banking AI Agent.

    Tất cả custom exceptions trong project đều kế thừa từ class này,
    cho phép catch toàn bộ lỗi hệ thống bằng một except duy nhất.

    Attributes:
        message: Mô tả lỗi chi tiết.
        details: Thông tin bổ sung (dict) để hỗ trợ debug/logging.
    """

    def __init__(self, message: str = "An error occurred in Banking Agent", details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message


# ═══════════════════════════════════════════════════════════════════════
# MODULE 1: CORE — Configuration & Logging Errors
# ═══════════════════════════════════════════════════════════════════════

class ConfigurationError(BankingAgentError):
    """
    Lỗi cấu hình hệ thống.

    Raise khi:
    - Thiếu biến môi trường bắt buộc (API key, DB path)
    - File .env không tồn tại hoặc không đọc được
    - Giá trị cấu hình không hợp lệ (ví dụ: MAX_AGENT_STEPS < 0)

    Ví dụ:
        raise ConfigurationError(
            "Missing required API key",
            details={"key": "GEMINI_API_KEY", "source": ".env"}
        )
    """

    def __init__(self, message: str = "Configuration error", details: dict | None = None):
        super().__init__(message, details)


# ═══════════════════════════════════════════════════════════════════════
# MODULE 2: KNOWLEDGE BASE — Data, Embedding & Vector Store Errors
# ═══════════════════════════════════════════════════════════════════════

class KnowledgeBaseError(BankingAgentError):
    """
    Base exception cho tất cả lỗi liên quan Knowledge Base.
    Bao gồm: load data, embedding, vector store operations.
    """

    def __init__(self, message: str = "Knowledge base error", details: dict | None = None):
        super().__init__(message, details)


class DataLoadError(KnowledgeBaseError):
    """
    Lỗi load hoặc parse dữ liệu nguồn.

    Raise khi:
    - File CSV không tồn tại hoặc bị corrupt
    - Thiếu cột bắt buộc trong CSV (Question, Answer, Class)
    - Dữ liệu rỗng sau khi preprocessing

    Ví dụ:
        raise DataLoadError(
            "CSV file is missing required columns",
            details={"file": "BankFAQs.csv", "missing_columns": ["Answer"]}
        )
    """

    def __init__(self, message: str = "Failed to load data", details: dict | None = None):
        super().__init__(message, details)


class EmbeddingError(KnowledgeBaseError):
    """
    Lỗi trong quá trình tạo embedding.

    Raise khi:
    - API embedding (Gemini/ONNX) trả lỗi hoặc timeout
    - Input text rỗng hoặc quá dài
    - Model embedding không load được

    Ví dụ:
        raise EmbeddingError(
            "Gemini embedding API returned error",
            details={"model": "text-embedding-004", "status_code": 429}
        )
    """

    def __init__(self, message: str = "Embedding generation failed", details: dict | None = None):
        super().__init__(message, details)


class VectorStoreError(KnowledgeBaseError):
    """
    Lỗi liên quan ChromaDB vector store.

    Raise khi:
    - Không kết nối được ChromaDB persistent client
    - Collection không tồn tại khi query
    - Lỗi khi add/update/delete documents
    - Path lưu ChromaDB không có quyền ghi

    Ví dụ:
        raise VectorStoreError(
            "Cannot connect to ChromaDB",
            details={"path": "./data/chroma_db", "error": "Permission denied"}
        )
    """

    def __init__(self, message: str = "Vector store operation failed", details: dict | None = None):
        super().__init__(message, details)


# ═══════════════════════════════════════════════════════════════════════
# MODULE 3: TOOLS — Tool System Errors
# ═══════════════════════════════════════════════════════════════════════

class ToolError(BankingAgentError):
    """
    Base exception cho tất cả lỗi liên quan Tool System.
    Bao gồm: tool registry, tool execution, input validation.
    """

    def __init__(self, message: str = "Tool error", details: dict | None = None):
        super().__init__(message, details)


class ToolNotFoundError(ToolError):
    """
    Tool không tồn tại trong registry.

    Raise khi:
    - Agent gọi tool_name không có trong ToolRegistry
    - Tool chưa được đăng ký cho agent profile hiện tại

    Ví dụ:
        raise ToolNotFoundError(
            "Tool not found in registry",
            details={"tool_name": "calculate_loan", "available_tools": ["faq", "branch_search"]}
        )
    """

    def __init__(self, message: str = "Tool not found", details: dict | None = None):
        super().__init__(message, details)


class ToolExecutionError(ToolError):
    """
    Lỗi xảy ra trong quá trình chạy tool.

    Raise khi:
    - Tool gặp exception runtime (API call fail, data error)
    - Tool timeout
    - Tool trả kết quả không hợp lệ

    Ví dụ:
        raise ToolExecutionError(
            "Web search tool timed out",
            details={"tool_name": "web_search", "query": "lãi suất ngân hàng", "timeout": 30}
        )
    """

    def __init__(self, message: str = "Tool execution failed", details: dict | None = None):
        super().__init__(message, details)


class ToolValidationError(ToolError):
    """
    Lỗi validate input arguments của tool.

    Raise khi:
    - Thiếu argument bắt buộc
    - Argument có kiểu dữ liệu sai (ví dụ: latitude không phải float)
    - Argument nằm ngoài range hợp lệ

    Ví dụ:
        raise ToolValidationError(
            "Invalid tool arguments",
            details={"tool_name": "branch_search", "error": "latitude must be a float"}
        )
    """

    def __init__(self, message: str = "Tool input validation failed", details: dict | None = None):
        super().__init__(message, details)


# ═══════════════════════════════════════════════════════════════════════
# MODULE 4 & 5: AGENT — Agent Core & LangGraph Errors
# ═══════════════════════════════════════════════════════════════════════

class AgentError(BankingAgentError):
    """
    Base exception cho tất cả lỗi liên quan Agent Core.
    Bao gồm: agent runner, LLM response parsing, graph execution.
    """

    def __init__(self, message: str = "Agent error", details: dict | None = None):
        super().__init__(message, details)


class AgentStepLimitError(AgentError):
    """
    Agent vượt quá số bước tối đa cho phép (MAX_AGENT_STEPS).

    Raise khi:
    - Agent loop chạy quá MAX_AGENT_STEPS mà không đưa ra ANSWER
    - Phòng tránh infinite loop khi agent bị kẹt

    Ví dụ:
        raise AgentStepLimitError(
            "Agent exceeded maximum steps",
            details={"max_steps": 10, "current_step": 11, "last_action": "THOUGHT"}
        )
    """

    def __init__(self, message: str = "Agent exceeded maximum allowed steps", details: dict | None = None):
        super().__init__(message, details)


class LLMResponseError(AgentError):
    """
    Lỗi parse hoặc xử lý response từ LLM.

    Raise khi:
    - LLM trả response không đúng format (thiếu THOUGHT/ACTION/ANSWER)
    - JSON trong response bị malformed
    - LLM trả response rỗng hoặc bị cắt (truncated)

    Ví dụ:
        raise LLMResponseError(
            "Failed to parse LLM response as JSON",
            details={"raw_response": "...", "parse_error": "Expecting ',' delimiter"}
        )
    """

    def __init__(self, message: str = "Failed to process LLM response", details: dict | None = None):
        super().__init__(message, details)


class GraphExecutionError(AgentError):
    """
    Lỗi trong quá trình chạy LangGraph workflow.

    Raise khi:
    - Graph chưa được compile
    - Lỗi state transition (node không tồn tại, edge bị thiếu)
    - Lỗi invoke graph

    Ví dụ:
        raise GraphExecutionError(
            "Graph invocation failed",
            details={"node": "call_agent", "error": "State missing required key 'query'"}
        )
    """

    def __init__(self, message: str = "LangGraph execution failed", details: dict | None = None):
        super().__init__(message, details)


# ═══════════════════════════════════════════════════════════════════════
# MODULE 6: API — FastAPI Layer Errors
# ═══════════════════════════════════════════════════════════════════════

class APIError(BankingAgentError):
    """
    Base exception cho tất cả lỗi liên quan API Layer.
    Bao gồm: request validation, service availability.
    """

    def __init__(
        self,
        message: str = "API error",
        status_code: int = 500,
        details: dict | None = None,
    ):
        self.status_code = status_code
        super().__init__(message, details)


class InvalidRequestError(APIError):
    """
    Request từ client không hợp lệ.

    Raise khi:
    - Query rỗng hoặc quá dài
    - Session ID format sai
    - Request body thiếu field bắt buộc (ngoài Pydantic validation)

    Ví dụ:
        raise InvalidRequestError(
            "Query exceeds maximum length",
            details={"max_length": 2000, "actual_length": 5000}
        )
    """

    def __init__(self, message: str = "Invalid request", details: dict | None = None):
        super().__init__(message, status_code=400, details=details)


class ServiceUnavailableError(APIError):
    """
    Dependency service không khả dụng.

    Raise khi:
    - ChromaDB không kết nối được khi health check
    - LLM API (Gemini/Groq) không phản hồi
    - Knowledge base chưa được index

    Ví dụ:
        raise ServiceUnavailableError(
            "LLM provider is not responding",
            details={"provider": "gemini", "timeout": 30}
        )
    """

    def __init__(self, message: str = "Service unavailable", details: dict | None = None):
        super().__init__(message, status_code=503, details=details)
