"""
AgentState cho Personal AI Agent — Agent Core (Module 4).

Module này định nghĩa AgentState TypedDict dùng làm state container
cho LangGraph StateGraph, quản lý toàn bộ dữ liệu chảy qua agent loop.

Kiến trúc: LangGraph State Management
    - AgentState: TypedDict chứa toàn bộ thông tin cần thiết cho ReAct loop
    - LangGraph tự động copy state giữa các node (immutable pattern)
    - Mỗi node (call_agent, call_tool, should_continue) nhận state,
      xử lý, và trả về partial state update

ReAct Loop Flow:
    1. User gửi query → khởi tạo AgentState
    2. call_agent: LLM suy luận (THOUGHT) → quyết định ACTION hoặc ANSWER
    3. call_tool: Thực thi tool → trả OBSERVATION
    4. should_continue: Router quyết định tiếp tục loop hay kết thúc
    5. Lặp lại 2-4 cho đến khi có ANSWER hoặc vượt MAX_STEPS

State Fields:
    ┌─────────────────────────────────────────────────────────────┐
    │ INPUT                                                       │
    │   query            — Câu hỏi gốc từ user                   │
    │   user_id          — ID người dùng (multi-tenant)           │
    │   session_id       — ID phiên hội thoại                     │
    │   agent_profile    — Profile agent (quyết định tools)       │
    ├─────────────────────────────────────────────────────────────┤
    │ AGENT LOOP STATE                                            │
    │   messages         — Lịch sử messages (system + user + AI)  │
    │   current_step     — Bước hiện tại trong loop               │
    │   max_steps        — Giới hạn bước tối đa                   │
    │   current_action   — Action hiện tại (THOUGHT/ACTION/ANSWER)│
    ├─────────────────────────────────────────────────────────────┤
    │ TOOL EXECUTION                                              │
    │   tool_name        — Tên tool đang được gọi                 │
    │   tool_args        — Arguments cho tool call                │
    │   tool_observations— Danh sách observations từ tool calls   │
    ├─────────────────────────────────────────────────────────────┤
    │ OUTPUT                                                      │
    │   final_answer     — Câu trả lời cuối cùng cho user         │
    │   error            — Thông tin lỗi (nếu có)                 │
    ├─────────────────────────────────────────────────────────────┤
    │ ROUTING & HANDOFF                                           │
    │   status           — Trạng thái agent (running/done/error)  │
    │   handoff_target   — Agent đích nếu cần HANDOFF             │
    │   handoff_reason   — Lý do HANDOFF                          │
    └─────────────────────────────────────────────────────────────┘

Cách sử dụng:
    from agent.state import AgentState, create_initial_state, AgentStatus

    # Khởi tạo state cho request mới
    state = create_initial_state(
        query="Tìm thông tin trong tài liệu của tôi?",
        user_id="user_123",
        agent_profile="personal_agent",
    )

    # Sử dụng trong LangGraph node
    def call_agent(state: AgentState) -> dict:
        # Xử lý...
        return {"current_step": state["current_step"] + 1, ...}

Tham khảo:
    - Plan.md Module 4: Agent Core & Prompt Engineering
    - LangGraph StateGraph documentation
    - ReAct pattern (Yao et al., 2022)
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, TypedDict

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ENUMS — Trạng thái và loại action của agent
# ═══════════════════════════════════════════════════════════════════════

class AgentStatus(str, Enum):
    """
    Trạng thái hiện tại của agent trong ReAct loop.

    Dùng trong should_continue() router để quyết định flow:
        - RUNNING → tiếp tục loop (gọi call_agent)
        - TOOL_CALLING → cần gọi tool (gọi call_tool)
        - DONE → đã có câu trả lời → kết thúc
        - ERROR → gặp lỗi không recover được → kết thúc
        - HANDOFF → chuyển cho agent khác xử lý (multi-agent)
    """
    RUNNING = "running"
    TOOL_CALLING = "tool_calling"
    DONE = "done"
    ERROR = "error"
    HANDOFF = "handoff"


class ActionType(str, Enum):
    """
    Loại action mà LLM quyết định trong mỗi bước ReAct.

    ReAct loop format:
        THOUGHT: Suy luận về câu hỏi
        ACTION: Gọi tool với arguments
        ANSWER: Trả lời cuối cùng cho user
        HANDOFF: Chuyển cho agent khác (multi-agent, tương lai)

    LLM response phải chứa đúng một trong các action types này.
    """
    THOUGHT = "THOUGHT"
    ACTION = "ACTION"
    ANSWER = "ANSWER"
    HANDOFF = "HANDOFF"


# ═══════════════════════════════════════════════════════════════════════
# MESSAGE TYPE — Cấu trúc message trong conversation history
# ═══════════════════════════════════════════════════════════════════════

class MessageRole(str, Enum):
    """
    Vai trò của message trong conversation history.

    Mapping với LLM API message roles:
        - SYSTEM → system prompt (instructions cho agent)
        - USER → câu hỏi/input từ user
        - ASSISTANT → response từ LLM (THOUGHT, ACTION, ANSWER)
        - OBSERVATION → kết quả từ tool execution
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    OBSERVATION = "observation"


class Message(TypedDict):
    """
    Cấu trúc một message trong conversation history.

    Attributes:
        role: Vai trò của message (system/user/assistant/observation).
        content: Nội dung message.
    """
    role: str   # MessageRole value
    content: str


# ═══════════════════════════════════════════════════════════════════════
# AGENT STATE — TypedDict chính cho LangGraph
# ═══════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    State container chính cho LangGraph StateGraph.

    TypedDict được chọn thay vì dataclass/Pydantic vì:
        1. LangGraph yêu cầu state là dict-like object
        2. TypedDict cho phép type checking mà không tạo overhead
        3. LangGraph tự động merge partial updates vào state

    Quy ước:
        - Mỗi node chỉ trả về PARTIAL state update (chỉ các key thay đổi)
        - LangGraph tự merge partial update vào state hiện tại
        - KHÔNG mutate state trực tiếp, luôn trả dict mới

    Ví dụ node:
        def call_agent(state: AgentState) -> dict:
            # Chỉ trả về các key cần update
            return {
                "current_step": state["current_step"] + 1,
                "current_action": ActionType.ACTION,
                "tool_name": "faq_search",
                "tool_args": {"query": "lãi suất"},
            }
    """

    # ─── INPUT (set khi khởi tạo, không đổi trong loop) ──────────────
    query: str
    """Câu hỏi gốc từ user. Không thay đổi trong suốt agent loop."""

    user_id: int
    """
    ID người dùng (multi-tenant). Dùng cho:
    - Filter tài liệu trong VectorStore theo user
    - Tiêm vào tool_args khi gọi DocumentSearchTool
    - Đảm bảo data isolation giữa các users
    Được set khi khởi tạo state và KHÔNG thay đổi trong suốt loop.
    """

    session_id: str
    """
    ID phiên hội thoại. Dùng cho:
    - LangGraph checkpointer (conversation memory)
    - Tracking/logging theo session
    - API response correlation
    """

    agent_profile: str
    """
    Tên agent profile (từ ToolRegistry).
    Quyết định agent được phép dùng tool nào.
    Ví dụ: "personal_agent", "general_agent".
    """

    # ─── CONVERSATION HISTORY ─────────────────────────────────────────
    messages: list[Message]
    """
    Lịch sử conversation đầy đủ, bao gồm:
    - System prompt (role="system")
    - User query (role="user")
    - LLM responses (role="assistant")
    - Tool observations (role="observation")

    Messages được append thêm qua mỗi bước loop.
    Dùng cho LLM context window.
    """

    # ─── AGENT LOOP CONTROL ───────────────────────────────────────────
    current_step: int
    """
    Bước hiện tại trong ReAct loop (bắt đầu từ 0).
    Tăng lên 1 sau mỗi lần call_agent.
    Khi current_step >= max_steps → agent bị dừng (AgentStepLimitError).
    """

    max_steps: int
    """
    Số bước tối đa cho phép. Đọc từ settings.MAX_AGENT_STEPS.
    Phòng tránh infinite loop khi agent bị kẹt.
    """

    current_action: str
    """
    Action type hiện tại mà LLM đã quyết định.
    Giá trị: ActionType enum value (THOUGHT, ACTION, ANSWER, HANDOFF).
    Dùng trong should_continue() router để routing.
    """

    status: str
    """
    Trạng thái tổng thể của agent.
    Giá trị: AgentStatus enum value (running, tool_calling, done, error, handoff).
    Dùng trong should_continue() router để quyết định kết thúc hay tiếp tục.
    """

    # ─── TOOL EXECUTION ───────────────────────────────────────────────
    tool_name: str
    """
    Tên tool mà agent muốn gọi (khi current_action == ACTION).
    Phải khớp với tool đã đăng ký trong ToolRegistry.
    Rỗng ("") khi không có tool call.
    """

    tool_args: dict[str, Any]
    """
    Arguments cho tool call (khi current_action == ACTION).
    Dict chứa key-value pairs khớp với tool's args_schema.
    Ví dụ: {"query": "lãi suất tiết kiệm", "n_results": 3}
    Rỗng ({}) khi không có tool call.
    """

    tool_observations: list[str]
    """
    Danh sách observations từ tất cả tool calls trong session.
    Mỗi observation là string từ ToolResult.to_observation().
    Format: "[tool_name] kết quả..."
    Append thêm sau mỗi tool execution.
    """

    # ─── OUTPUT ───────────────────────────────────────────────────────
    final_answer: str
    """
    Câu trả lời cuối cùng cho user (khi current_action == ANSWER).
    Rỗng ("") khi agent chưa đưa ra câu trả lời.
    Đây là giá trị được trả về cho user qua API.
    """

    error: str
    """
    Thông tin lỗi nếu agent gặp vấn đề không recover được.
    Rỗng ("") khi không có lỗi.
    Set khi status == ERROR.
    """

    # ─── HANDOFF (multi-agent, chuẩn bị cho tương lai) ────────────────
    handoff_target: str
    """
    Tên agent đích khi cần HANDOFF.
    Rỗng ("") khi không có handoff.
    Ví dụ: "loan_agent", "escalation_agent".
    Dùng khi mở rộng sang multi-agent architecture (Cấp 2 trong Plan.md).
    """

    handoff_reason: str
    """
    Lý do HANDOFF cho agent khác.
    Rỗng ("") khi không có handoff.
    Ví dụ: "Câu hỏi về vay vốn cần chuyên gia Loan Agent xử lý."
    """


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

    Factory function đảm bảo mọi field đều được khởi tạo đúng,
    tránh KeyError khi LangGraph truy cập state.

    Args:
        query: Câu hỏi từ user.
        user_id: ID người dùng (bắt buộc, dùng cho multi-tenant filtering).
        session_id: ID session. Nếu None, tự tạo UUID mới.
        agent_profile: Tên agent profile (mặc định "personal_agent").
        max_steps: Giới hạn bước. Nếu None, dùng settings.MAX_AGENT_STEPS.

    Returns:
        AgentState đã khởi tạo đầy đủ, sẵn sàng cho graph.invoke().

    Raises:
        ValueError: Khi query rỗng hoặc chỉ chứa whitespace.
        ValueError: Khi user_id rỗng hoặc chỉ chứa whitespace.

    Ví dụ:
        state = create_initial_state(
            query="Tìm thông tin trong tài liệu của tôi?",
            user_id="user_123",
            agent_profile="personal_agent",
        )
        result = graph.invoke(state)
    """
    # Validate input
    if not query or not query.strip():
        raise ValueError("Query không được rỗng. Vui lòng nhập câu hỏi.")

    if user_id is None:
        raise ValueError(
            "user_id không được rỗng. "
            "user_id bắt buộc để đảm bảo multi-tenant data isolation."
        )

    # Generate session_id nếu chưa có
    resolved_session_id = session_id or str(uuid.uuid4())

    # Resolve max_steps
    resolved_max_steps = max_steps if max_steps is not None else settings.MAX_AGENT_STEPS

    # Chuẩn hóa user_id (đã là int)
    resolved_user_id = user_id

    state: AgentState = {
        # Input
        "query": query.strip(),
        "user_id": resolved_user_id,
        "session_id": resolved_session_id,
        "agent_profile": agent_profile,

        # Conversation history — bắt đầu rỗng
        # System prompt sẽ được thêm bởi call_agent node
        "messages": [],

        # Agent loop control
        "current_step": 0,
        "max_steps": resolved_max_steps,
        "current_action": "",
        "status": AgentStatus.RUNNING.value,

        # Tool execution
        "tool_name": "",
        "tool_args": {},
        "tool_observations": [],

        # Output
        "final_answer": "",
        "error": "",

        # Handoff
        "handoff_target": "",
        "handoff_reason": "",
    }

    logger.info(
        f"Created initial AgentState | "
        f"user_id={resolved_user_id} | "
        f"session={resolved_session_id[:8]}... | "
        f"profile={agent_profile} | "
        f"max_steps={resolved_max_steps} | "
        f"query='{query[:50]}...'" if len(query) > 50 else
        f"Created initial AgentState | "
        f"user_id={resolved_user_id} | "
        f"session={resolved_session_id[:8]}... | "
        f"profile={agent_profile} | "
        f"max_steps={resolved_max_steps} | "
        f"query='{query}'"
    )

    return state


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — Tiện ích thao tác state
# ═══════════════════════════════════════════════════════════════════════

def add_message(state: AgentState, role: str, content: str) -> list[Message]:
    """
    Tạo danh sách messages mới với message được thêm vào cuối.

    KHÔNG mutate state trực tiếp — trả về list mới để LangGraph merge.

    Args:
        state: AgentState hiện tại.
        role: Vai trò message (dùng MessageRole enum value).
        content: Nội dung message.

    Returns:
        List messages mới (copy + append).

    Ví dụ:
        # Trong LangGraph node:
        def call_agent(state: AgentState) -> dict:
            new_messages = add_message(state, MessageRole.ASSISTANT, llm_response)
            return {"messages": new_messages}
    """
    new_message: Message = {"role": role, "content": content}
    return state["messages"] + [new_message]


def add_observation(state: AgentState, observation: str) -> list[str]:
    """
    Tạo danh sách observations mới với observation được thêm vào cuối.

    KHÔNG mutate state trực tiếp — trả về list mới để LangGraph merge.

    Args:
        state: AgentState hiện tại.
        observation: Chuỗi observation từ ToolResult.to_observation().

    Returns:
        List observations mới (copy + append).

    Ví dụ:
        # Trong call_tool node:
        def call_tool(state: AgentState) -> dict:
            result = tool.safe_run(**state["tool_args"])
            new_obs = add_observation(state, result.to_observation())
            return {"tool_observations": new_obs}
    """
    return state["tool_observations"] + [observation]


def is_step_limit_reached(state: AgentState) -> bool:
    """
    Kiểm tra agent đã vượt quá giới hạn bước chưa.

    Args:
        state: AgentState hiện tại.

    Returns:
        True nếu current_step >= max_steps.
    """
    return state["current_step"] >= state["max_steps"]


def get_state_summary(state: AgentState) -> dict[str, Any]:
    """
    Tạo summary ngắn gọn của state hiện tại (cho logging/debugging).

    Args:
        state: AgentState hiện tại.

    Returns:
        Dict chứa thông tin summary.

    Ví dụ:
        summary = get_state_summary(state)
        logger.debug(f"State summary: {summary}")
    """
    return {
        "user_id": state.get("user_id", "unknown"),
        "session_id": state["session_id"][:8] + "...",
        "query_preview": state["query"][:50] + ("..." if len(state["query"]) > 50 else ""),
        "step": f"{state['current_step']}/{state['max_steps']}",
        "status": state["status"],
        "action": state["current_action"] or "none",
        "tool": state["tool_name"] or "none",
        "n_messages": len(state["messages"]),
        "n_observations": len(state["tool_observations"]),
        "has_answer": bool(state["final_answer"]),
        "has_error": bool(state["error"]),
        "handoff_target": state["handoff_target"] or "none",
    }
