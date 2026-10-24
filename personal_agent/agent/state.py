"""
AgentState cho Personal AI Agent — Native Function Calling Core.

Module này định nghĩa AgentState TypedDict dùng làm state container
cho LangGraph StateGraph, quản lý dữ liệu chảy qua agent loop với
OpenAI-compatible native function calling (DeepSeek).

Kiến trúc: LangGraph State Management (Native Function Calling)
    - AgentState: TypedDict chứa các field cốt lõi
    - LangGraph copy state giữa các node (immutable pattern)
    - Mỗi node nhận state và trả về partial state update

Flow:
    1. User gửi query → khởi tạo AgentState
    2. call_agent: LLM suy luận với tools schema → trả direct content hoặc tool_calls
    3. should_continue: Router kiểm tra pending_tool_calls / auth_required / error / step_limit
    4. call_tool: Thực thi các pending_tool_calls → thêm message role="tool"
    5. Loop lại cho đến khi DONE, AUTH_REQUIRED, hoặc ERROR

Tham khảo:
    - implementation_plan2.md: Phase 1
    - LangGraph StateGraph documentation
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, TypedDict

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ENUMS — Trạng thái của agent
# ═══════════════════════════════════════════════════════════════════════

class AgentStatus(str, Enum):
    """
    Trạng thái hiện tại của agent trong function calling loop.

    Dùng trong should_continue() router và API response:
        - RUNNING → tiếp tục loop (có tool_calls hoặc tool_results cần xử lý)
        - DONE → LLM đã trả lời xong (không còn tool_calls)
        - ERROR → gặp lỗi không recover được
        - AUTH_REQUIRED → cần user xác thực tài khoản bên thứ 3 (Google OAuth)
    """
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    AUTH_REQUIRED = "auth_required"


class MessageRole(str, Enum):
    """
    Vai trò của message trong conversation history theo chuẩn OpenAI.
        - SYSTEM → system prompt
        - USER → input từ user
        - ASSISTANT → câu trả lời hoặc tool_calls từ LLM
        - TOOL → kết quả thực thi tool
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(TypedDict, total=False):
    """
    Cấu trúc message theo chuẩn OpenAI Function Calling.

    Attributes:
        role: system / user / assistant / tool
        content: Nội dung text (None nếu assistant chỉ gọi tool)
        tool_calls: Danh sách function calls do assistant đề xuất
        tool_call_id: ID của tool call khi role="tool"
        name: Tên của tool khi role="tool"
    """
    role: str
    content: str | None
    tool_calls: list[dict] | None
    tool_call_id: str | None
    name: str | None


# ═══════════════════════════════════════════════════════════════════════
# AGENT STATE — TypedDict chính cho LangGraph
# ═══════════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    """
    State container chính cho LangGraph StateGraph.

    Chỉ giữ lại các field cần thiết cho Native Function Calling.
    """

    # ─── INPUT / CONTEXT ─────────────────────────────────────────────
    query: str
    """Câu hỏi hiện tại từ user."""

    user_id: int
    """ID người dùng (multi-tenant isolation, Google OAuth lookup)."""

    session_id: str
    """ID phiên hội thoại (thread_id cho checkpointer)."""

    agent_profile: str
    """Profile của agent (quyết định allowed tools)."""

    # ─── CONVERSATION HISTORY ─────────────────────────────────────────
    messages: list[Message]
    """Toàn bộ messages trong phiên chat (OpenAI format)."""

    # ─── LOOP CONTROL ─────────────────────────────────────────────────
    current_step: int
    """Số bước đã thực hiện trong loop hiện tại."""

    max_steps: int
    """Giới hạn số bước tối đa."""

    status: str
    """AgentStatus value: running / done / error / auth_required."""

    # ─── FUNCTION CALLING ─────────────────────────────────────────────
    pending_tool_calls: list[dict]
    """
    Tool calls từ LLM response đang chờ thực thi.
    Format: [{"id": "...", "name": "...", "args": {...}}]
    """

    tool_results: list[dict]
    """
    Kết quả tool đã chạy.
    Format: [{"tool_call_id": "...", "name": "...", "content": "..."}]
    """

    # ─── AUTH SUSPENSION ──────────────────────────────────────────────
    auth_required: bool
    """True nếu cần user xác thực tài khoản bên thứ 3 (Google OAuth)."""

    auth_url: str
    """URL authorization để FE mở popup xác thực."""

    auth_service: str
    """Tên service cần auth (ví dụ: 'google')."""

    # ─── OUTPUT / ERROR ───────────────────────────────────────────────
    final_answer: str
    """Câu trả lời cuối cùng cho user (trích xuất từ message cuối)."""

    error: str
    """Thông báo lỗi nếu gặp sự cố không recover được."""


# ═══════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION — Khởi tạo AgentState
# ═══════════════════════════════════════════════════════════════════════

def create_initial_state(
    query: str,
    user_id: int,
    session_id: str | None = None,
    agent_profile: str = "personal_agent",
    max_steps: int | None = None,
) -> AgentState:
    """
    Tạo AgentState ban đầu cho một request mới.
    """
    if not query or not query.strip():
        raise ValueError("Query không được rỗng. Vui lòng nhập câu hỏi.")

    if user_id is None:
        raise ValueError("user_id không được rỗng để đảm bảo multi-tenant data isolation.")

    resolved_session_id = session_id or str(uuid.uuid4())
    resolved_max_steps = max_steps if max_steps is not None else settings.MAX_AGENT_STEPS

    state: AgentState = {
        "query": query.strip(),
        "user_id": user_id,
        "session_id": resolved_session_id,
        "agent_profile": agent_profile,
        "messages": [],
        "current_step": 0,
        "max_steps": resolved_max_steps,
        "status": AgentStatus.RUNNING.value,
        "pending_tool_calls": [],
        "tool_results": [],
        "auth_required": False,
        "auth_url": "",
        "auth_service": "",
        "final_answer": "",
        "error": "",
    }

    logger.info(
        f"Created initial AgentState | "
        f"user_id={user_id} | "
        f"session={resolved_session_id[:8]}... | "
        f"profile={agent_profile} | "
        f"max_steps={resolved_max_steps} | "
        f"query='{query[:50]}...'"
    )

    return state


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def add_message(
    state: AgentState,
    role: str,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
    name: str | None = None,
) -> list[Message]:
    """Tạo danh sách messages mới với message được thêm vào cuối."""
    new_msg: Message = {"role": role, "content": content}
    if tool_calls is not None:
        new_msg["tool_calls"] = tool_calls
    if tool_call_id is not None:
        new_msg["tool_call_id"] = tool_call_id
    if name is not None:
        new_msg["name"] = name
    return state.get("messages", []) + [new_msg]


def is_step_limit_reached(state: AgentState) -> bool:
    """Kiểm tra agent đã chạm hoặc vượt quá giới hạn bước chưa."""
    return state.get("current_step", 0) >= state.get("max_steps", settings.MAX_AGENT_STEPS)


def get_state_summary(state: AgentState) -> dict[str, Any]:
    """Tạo summary ngắn gọn của state hiện tại cho logging/debugging."""
    return {
        "user_id": state.get("user_id"),
        "session_id": state.get("session_id", "")[:8] + "...",
        "step": f"{state.get('current_step', 0)}/{state.get('max_steps', 0)}",
        "status": state.get("status"),
        "pending_tool_calls": len(state.get("pending_tool_calls", [])),
        "auth_required": state.get("auth_required", False),
        "n_messages": len(state.get("messages", [])),
        "has_answer": bool(state.get("final_answer")),
        "has_error": bool(state.get("error")),
    }
