"""
Pydantic Schemas cho API Layer — Module 6.

Module này định nghĩa tất cả Request/Response models cho FastAPI endpoints.
Pydantic v2 tự động validate input, generate OpenAPI docs, và serialize output.

Schemas mapping:

    ┌──────────────────────────────────────────────────────────────┐
    │  Client Request                                              │
    │    ↓                                                         │
    │  ChatRequest  ──→  invoke_agent() ──→  AgentState (dict)     │
    │                                            ↓                 │
    │                                      ChatResponse            │
    │    ↓                                                         │
    │  Client Response                                             │
    └──────────────────────────────────────────────────────────────┘

Endpoints & Schemas:
    POST /chat
        Request:  ChatRequest
        Response: ChatResponse

    GET /health
        Response: HealthResponse

    POST /index/rebuild
        Response: IndexRebuildResponse

Tham khảo:
    - Plan.md Module 6: FastAPI REST API
    - agent/state.py: AgentState, AgentStatus
    - agent/graph.py: invoke_agent() return dict
    - core/config.py: settings (MAX_AGENT_STEPS, etc.)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════════════
# ENUMS — Trạng thái dùng trong Response
# ═══════════════════════════════════════════════════════════════════════

class AgentStatusResponse(str, Enum):
    """
    Trạng thái agent trả về cho client.

    Mapping từ AgentStatus (agent/state.py):
        - done    → agent đã trả lời thành công
        - error   → agent gặp lỗi
        - handoff → agent chuyển giao (multi-agent, tương lai)
    """
    DONE = "done"
    ERROR = "error"
    HANDOFF = "handoff"


# ═══════════════════════════════════════════════════════════════════════
# CHAT SCHEMAS — POST /chat
# ═══════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """
    Request body cho POST /chat endpoint.

    Client gửi câu hỏi và optional metadata để agent xử lý.

    Ví dụ request body:
        {
            "query": "Lãi suất tiết kiệm 12 tháng là bao nhiêu?",
            "session_id": "session-001",
            "agent_profile": "personal_agent",
            "max_steps": 5
        }

    Ví dụ tối giản (chỉ cần query):
        {
            "query": "Chi nhánh nào gần Hà Nội nhất?"
        }
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Câu hỏi từ user. Không được rỗng, tối đa 2000 ký tự.",
        examples=["Lãi suất tiết kiệm 12 tháng là bao nhiêu?"],
    )

    session_id: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "ID phiên hội thoại. Nếu None, server tự tạo UUID mới. "
            "Gửi lại session_id từ response trước để tiếp tục hội thoại."
        ),
        examples=["session-001"],
    )

    agent_profile: str = Field(
        default="personal_agent",
        max_length=64,
        description=(
            "Tên agent profile quyết định tools được phép sử dụng. "
            "Mặc định: 'personal_agent'."
        ),
        examples=["personal_agent"],
    )

    max_steps: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "Giới hạn số bước tối đa cho ReAct loop. "
            "Nếu None, dùng giá trị mặc định từ server config."
        ),
        examples=[5, 10],
    )

    @model_validator(mode="before")
    @classmethod
    def strip_query_whitespace(cls, data: dict) -> dict:
        """Loại bỏ whitespace thừa ở đầu/cuối query trước khi validate."""
        if isinstance(data, dict) and "query" in data:
            query = data.get("query")
            if isinstance(query, str):
                data["query"] = query.strip()
        return data

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Lãi suất tiết kiệm 12 tháng là bao nhiêu?",
                    "session_id": "session-001",
                    "agent_profile": "personal_agent",
                    "max_steps": 5,
                },
                {
                    "query": "Chi nhánh nào gần Hà Nội nhất?",
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    """
    Response body cho POST /chat endpoint.

    Chứa câu trả lời từ agent cùng metadata về quá trình xử lý.

    Ví dụ response:
        {
            "answer": "Lãi suất tiết kiệm 12 tháng hiện tại là 5.5%/năm.",
            "status": "done",
            "session_id": "session-001",
            "num_steps": 3,
            "tool_observations": [
                "[faq_search] Theo dữ liệu FAQ: lãi suất 12 tháng là 5.5%..."
            ],
            "error": null
        }
    """

    answer: str = Field(
        ...,
        description="Câu trả lời cuối cùng từ agent cho user.",
        examples=["Lãi suất tiết kiệm 12 tháng hiện tại là 5.5%/năm."],
    )

    status: AgentStatusResponse = Field(
        ...,
        description="Trạng thái kết thúc của agent (done/error/handoff).",
        examples=["done"],
    )

    session_id: str = Field(
        ...,
        description=(
            "ID phiên hội thoại. Client nên lưu lại và gửi kèm "
            "trong request tiếp theo để duy trì ngữ cảnh hội thoại."
        ),
        examples=["session-001"],
    )

    num_steps: int = Field(
        ...,
        ge=0,
        description="Số bước ReAct loop agent đã thực hiện trong lần invoke này.",
        examples=[3],
    )

    tool_observations: list[str] = Field(
        default_factory=list,
        description=(
            "Danh sách observations từ các tool calls trong session. "
            "Mỗi observation có format: '[tool_name] kết quả...'"
        ),
        examples=[["[faq_search] Theo dữ liệu FAQ: lãi suất 12 tháng là 5.5%..."]],
    )

    error: str | None = Field(
        default=None,
        description="Thông tin lỗi nếu agent gặp vấn đề. None nếu thành công.",
        examples=[None],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Lãi suất tiết kiệm 12 tháng hiện tại là 5.5%/năm.",
                    "status": "done",
                    "session_id": "session-001",
                    "num_steps": 3,
                    "tool_observations": [
                        "[faq_search] Theo dữ liệu FAQ: lãi suất 12 tháng là 5.5%..."
                    ],
                    "error": None,
                },
            ]
        }
    }

    @staticmethod
    def from_agent_result(result: dict, session_id: str) -> "ChatResponse":
        """
        Factory method chuyển đổi AgentState dict → ChatResponse.

        Mapping AgentState fields → ChatResponse fields:
            final_answer     → answer
            status           → status (chỉ lấy done/error/handoff)
            session_id       → session_id
            current_step     → num_steps
            tool_observations → tool_observations
            error            → error

        Args:
            result: Dict kết quả từ invoke_agent() (AgentState).
            session_id: Session ID đã resolved.

        Returns:
            ChatResponse instance.
        """
        # Xử lý status — map sang AgentStatusResponse
        raw_status = result.get("status", "done")
        try:
            status = AgentStatusResponse(raw_status)
        except ValueError:
            # Status không xác định (running, tool_calling, ...) → mặc định error
            status = AgentStatusResponse.ERROR

        # Xử lý answer — nếu error mà không có answer, dùng error message
        answer = result.get("final_answer", "")
        error = result.get("error", "") or None

        if not answer and status == AgentStatusResponse.ERROR:
            answer = error or "Agent gặp lỗi không xác định."

        return ChatResponse(
            answer=answer,
            status=status,
            session_id=session_id,
            num_steps=result.get("current_step", 0),
            tool_observations=result.get("tool_observations", []),
            error=error,
        )


# ═══════════════════════════════════════════════════════════════════════
# HEALTH SCHEMAS — GET /health
# ═══════════════════════════════════════════════════════════════════════

class DependencyStatus(BaseModel):
    """
    Trạng thái của một dependency trong health check.

    Ví dụ:
        {
            "name": "chroma_db",
            "status": "healthy",
            "details": "PersistentClient connected, collection 'bank_faq' has 150 docs"
        }
    """

    name: str = Field(
        ...,
        description="Tên dependency (chroma_db, llm_provider, graph, ...).",
        examples=["chroma_db"],
    )

    status: str = Field(
        ...,
        description="Trạng thái: 'healthy' hoặc 'unhealthy'.",
        examples=["healthy"],
    )

    details: str | None = Field(
        default=None,
        description="Thông tin bổ sung về dependency.",
        examples=["PersistentClient connected, 150 documents indexed"],
    )


class HealthResponse(BaseModel):
    """
    Response body cho GET /health endpoint.

    Kiểm tra trạng thái tổng thể của hệ thống và từng dependency.

    Ví dụ response:
        {
            "status": "healthy",
            "timestamp": "2026-07-13T17:00:00",
            "dependencies": [
                {"name": "graph", "status": "healthy", "details": "Compiled & cached"},
                {"name": "chroma_db", "status": "healthy", "details": "150 documents"},
                {"name": "llm_provider", "status": "healthy", "details": "gemini"}
            ]
        }
    """

    status: str = Field(
        ...,
        description=(
            "Trạng thái tổng thể: 'healthy' (tất cả OK) "
            "hoặc 'unhealthy' (có dependency lỗi)."
        ),
        examples=["healthy"],
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Thời điểm thực hiện health check.",
    )

    dependencies: list[DependencyStatus] = Field(
        default_factory=list,
        description="Trạng thái chi tiết từng dependency.",
    )


# ═══════════════════════════════════════════════════════════════════════
# INDEX REBUILD SCHEMAS — POST /index/rebuild
# ═══════════════════════════════════════════════════════════════════════

class IndexRebuildResponse(BaseModel):
    """
    Response body cho POST /index/rebuild endpoint.

    Trả về kết quả sau khi trigger reindex knowledge base.

    Ví dụ response:
        {
            "success": true,
            "message": "Knowledge base reindexed successfully",
            "documents_indexed": 150,
            "duration_seconds": 12.5
        }
    """

    success: bool = Field(
        ...,
        description="True nếu reindex thành công, False nếu lỗi.",
        examples=[True],
    )

    message: str = Field(
        ...,
        description="Mô tả kết quả reindex.",
        examples=["Knowledge base reindexed successfully"],
    )

    documents_indexed: int | None = Field(
        default=None,
        ge=0,
        description="Số documents đã index. None nếu lỗi.",
        examples=[150],
    )

    duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Thời gian reindex (giây). None nếu lỗi.",
        examples=[12.5],
    )


# ═══════════════════════════════════════════════════════════════════════
# ERROR SCHEMAS — Chuẩn hóa error response
# ═══════════════════════════════════════════════════════════════════════

class ErrorDetail(BaseModel):
    """
    Chi tiết một lỗi cụ thể (cho validation errors).

    Ví dụ:
        {
            "field": "query",
            "message": "String should have at least 1 character",
            "type": "string_too_short"
        }
    """

    field: str | None = Field(
        default=None,
        description="Tên field gây lỗi (nếu có).",
        examples=["query"],
    )

    message: str = Field(
        ...,
        description="Mô tả lỗi.",
        examples=["String should have at least 1 character"],
    )

    type: str | None = Field(
        default=None,
        description="Loại lỗi (Pydantic error type).",
        examples=["string_too_short"],
    )


class ErrorResponse(BaseModel):
    """
    Response body chuẩn hóa cho tất cả error responses.

    Dùng cho cả:
    - Validation errors (422)
    - Business logic errors (400)
    - Server errors (500)
    - Service unavailable (503)

    Ví dụ response (422):
        {
            "error": "Validation Error",
            "message": "Request body không hợp lệ",
            "status_code": 422,
            "details": [
                {
                    "field": "query",
                    "message": "String should have at least 1 character",
                    "type": "string_too_short"
                }
            ],
            "request_id": "req-abc123"
        }
    """

    error: str = Field(
        ...,
        description="Tên loại lỗi.",
        examples=["Validation Error", "Internal Server Error"],
    )

    message: str = Field(
        ...,
        description="Mô tả lỗi ngắn gọn cho user.",
        examples=["Request body không hợp lệ"],
    )

    status_code: int = Field(
        ...,
        description="HTTP status code.",
        examples=[400, 422, 500, 503],
    )

    details: list[ErrorDetail] = Field(
        default_factory=list,
        description="Danh sách chi tiết từng lỗi (nếu có).",
    )

    request_id: str | None = Field(
        default=None,
        description="Request ID để tracking/debugging.",
        examples=["req-abc123"],
    )
