"""
LangGraph Workflow cho Personal AI Agent — Native Function Calling.

Module này định nghĩa StateGraph workflow cho Native Function Calling loop,
quản lý routing dựa trên pending_tool_calls và auth_required,
đồng thời tích hợp conversation memory qua SqliteSaver.

Flow:
    START → call_agent → should_continue() ──┬──> call_tool → should_continue_tool() ──> call_agent
                                            │                                        └──> END (AUTH)
                                            ├──> call_agent (loop lại nếu cần)
                                            └──> END (DONE / ERROR / AUTH_REQUIRED)

Tham khảo:
    - implementation_plan2.md: Phase 7 & Phase 10
"""

from __future__ import annotations

import os
import threading
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from core.config import settings
from core.exceptions import GraphExecutionError
from core.logger import get_logger

from agent.state import AgentState, AgentStatus

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# SINGLETON STATE — Thread-safe caching cho compiled graph
# ═══════════════════════════════════════════════════════════════════════

_compiled_graph = None
_graph_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════
# ROUTERS — Routing logic cho conditional edges
# ═══════════════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    """
    Router sau node call_agent.
    """
    # 1. Có tool calls chờ thực thi?
    if state.get("pending_tool_calls"):
        return "call_tool"

    # 2. Kiểm tra lỗi không recover được
    if state.get("error"):
        return END

    # 3. Yêu cầu xác thực Google OAuth
    # if state.get("auth_required"):
    #     return END

    # 4. Đạt giới hạn ngân sách bước
    if state.get("current_step", 0) >= state.get("max_steps", settings.MAX_AGENT_STEPS):
        return END

    # 5. Nếu message cuối cùng là kết quả tool → cần LLM phân tích
    # messages = state.get("messages", [])
    # if messages:
    #     last_msg = messages[-1]
    #     if last_msg.get("role") in ("tool", "observation"):
    #         return "call_agent"

    # 6. Mặc định: LLM đã hoàn tất câu trả lời
    return END


def should_continue_tool(state: AgentState) -> str:
    """
    Router sau node call_tool.
    """
    # Nếu tool executor phát hiện cần OAuth auth → dừng ngay tại END
    if state.get("auth_required"):
        return END

    # Có kết quả tool → loop lại call_agent để LLM đọc kết quả
    return "call_agent"


# ═══════════════════════════════════════════════════════════════════════
# GRAPH BUILDER — Xây dựng StateGraph
# ═══════════════════════════════════════════════════════════════════════

def build_graph() -> Any:
    """
    Xây dựng LangGraph StateGraph cho Native Function Calling loop.
    """
    try:
        from agent.runner import AgentRunner
        from agent.tool_executor import ToolExecutor

        runner = AgentRunner()
        executor = ToolExecutor()

        logger.info("Building LangGraph StateGraph for native function calling...")

        graph = StateGraph(AgentState)

        # ── Thêm nodes ───────────────────────────────────────────────
        graph.add_node("call_agent", runner.call_agent)
        graph.add_node("call_tool", executor.call_tool)

        # ── Entry point ──────────────────────────────────────────────
        graph.set_entry_point("call_agent")

        # ── Conditional edges từ call_agent ──────────────────────────
        graph.add_conditional_edges(
            source="call_agent",
            path=should_continue,
            path_map={
                "call_tool": "call_tool",
                "call_agent": "call_agent",
                END: END,
            },
        )

        # ── Conditional edges từ call_tool ───────────────────────────
        graph.add_conditional_edges(
            source="call_tool",
            path=should_continue_tool,
            path_map={
                "call_agent": "call_agent",
                END: END,
            },
        )

        # ── Checkpointer ─────────────────────────────────────────────
        import sqlite3

        db_path = settings.CHECKPOINT_DB_PATH
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        compiled = graph.compile(checkpointer=checkpointer)

        logger.info("LangGraph StateGraph compiled successfully")
        return compiled

    except Exception as e:
        logger.error(f"Failed to build graph: {e}", exc_info=True)
        raise GraphExecutionError(
            "Failed to build LangGraph workflow",
            details={"error": str(e), "type": type(e).__name__},
        ) from e


def get_graph():
    """Singleton getter cho compiled graph."""
    global _compiled_graph

    if _compiled_graph is not None:
        return _compiled_graph

    with _graph_lock:
        if _compiled_graph is not None:
            return _compiled_graph

        logger.info("Compiling graph singleton...")
        _compiled_graph = build_graph()
        return _compiled_graph


def reset_graph() -> None:
    """Reset cached graph (dùng khi reload hoặc test)."""
    global _compiled_graph
    with _graph_lock:
        _compiled_graph = None
        logger.info("Graph cache cleared")


# ═══════════════════════════════════════════════════════════════════════
# HELPER: EXTRACT LAST ASSISTANT CONTENT
# ═══════════════════════════════════════════════════════════════════════

def _extract_last_assistant_content(messages: list[dict[str, Any]]) -> str:
    """Trích xuất content của assistant message cuối cùng."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


# ═══════════════════════════════════════════════════════════════════════
# INVOKE HELPER — Convenience function để chạy agent
# ═══════════════════════════════════════════════════════════════════════

def invoke_agent(
    query: str,
    user_id: int,
    agent_profile: str = "personal_agent",
    session_id: str | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """
    Convenience function để chạy agent, hỗ trợ duy trì session và
    nhận diện auth suspension.
    """
    import uuid
    from agent.state import create_initial_state

    resolved_session_id = session_id or str(uuid.uuid4())
    resolved_max_steps = (
        max_steps if max_steps is not None else settings.MAX_AGENT_STEPS
    )

    graph = get_graph()
    config = {"configurable": {"thread_id": resolved_session_id}}
    session_preview = resolved_session_id[:8]

    # Kiểm tra state đã tồn tại trên thread_id này chưa
    try:
        existing_state = graph.get_state(config)
        has_existing_state = bool(
            existing_state
            and existing_state.values
            and existing_state.values.get("messages")
        )
    except Exception:
        has_existing_state = False

    if has_existing_state:
        logger.info(
            f"Continuing conversation | "
            f"thread_id={session_preview}... | "
            f"query='{query[:50]}...'"
        )
        input_state = {
            "query": query.strip(),
            "user_id": user_id,
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
    else:
        logger.info(
            f"Starting new conversation | "
            f"thread_id={session_preview}... | "
            f"profile={agent_profile} | "
            f"query='{query[:50]}...'"
        )
        input_state = create_initial_state(
            query=query,
            user_id=user_id,
            agent_profile=agent_profile,
            session_id=resolved_session_id,
            max_steps=resolved_max_steps,
        )

    try:
        result = graph.invoke(input_state, config)

        # Kiểm tra trường hợp cần Auth (Google OAuth)
        if result.get("auth_required"):
            logger.info(
                f"Agent suspended for auth | service={result.get('auth_service')} | "
                f"thread_id={session_preview}..."
            )
            return {
                "status": "auth_required",
                "service": result.get("auth_service", "google"),
                "authorize_url": result.get("auth_url"),
                "message": "Vui lòng kết nối tài khoản Google để sử dụng tính năng Gmail.",
                "session_id": result.get("session_id", resolved_session_id),
                "steps": result.get("current_step", 0),
                "final_answer": "Vui lòng kết nối tài khoản Google để sử dụng tính năng Gmail.",
            }

        # Trả về kết quả bình thường
        answer = result.get("final_answer") or _extract_last_assistant_content(result.get("messages", []))
        return {
            "status": result.get("status", "done"),
            "answer": answer,
            "final_answer": answer,
            "session_id": result.get("session_id", resolved_session_id),
            "steps": result.get("current_step", 0),
            "current_step": result.get("current_step", 0),
            "error": result.get("error", ""),
            "messages": result.get("messages", []),
        }

    except Exception as e:
        logger.error(
            f"Agent invocation failed: {e} | thread_id={session_preview}...",
            exc_info=True,
        )
        raise GraphExecutionError(
            "Agent invocation failed",
            details={
                "query": query[:100],
                "profile": agent_profile,
                "session_id": resolved_session_id,
                "continued": has_existing_state,
                "error": str(e),
            },
        ) from e
