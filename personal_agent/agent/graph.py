"""
LangGraph Workflow cho Personal AI Agent — Module 5.

Module này định nghĩa toàn bộ LangGraph StateGraph workflow,
compile graph một lần (singleton pattern) và cung cấp factory
function get_graph() để reuse trong suốt application lifecycle.

Kiến trúc: LangGraph StateGraph Workflow
    - build_graph(): Xây dựng StateGraph với nodes và edges
    - get_graph(): Factory function với caching (singleton)
    - should_continue(): Router function quyết định flow
    - get_checkpointer(): Factory function cho SqliteSaver (singleton)

Graph Flow (ReAct Loop):
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   START                                                     │
    │     │                                                       │
    │     ▼                                                       │
    │   ┌──────────┐                                              │
    │   │call_agent│ ← LLM suy luận (THOUGHT/ACTION/ANSWER)      │
    │   └────┬─────┘                                              │
    │        │                                                    │
    │        ▼                                                    │
    │   ┌───────────────┐                                         │
    │   │should_continue│ ← Router: quyết định flow tiếp theo     │
    │   └───┬───┬───┬───┘                                         │
    │       │   │   │                                             │
    │       │   │   └──► END (DONE / ERROR)                       │
    │       │   │                                                 │
    │       │   └──────► call_agent (THOUGHT → tiếp tục loop)     │
    │       │                                                     │
    │       ▼                                                     │
    │   ┌─────────┐                                               │
    │   │call_tool│ ← Thực thi tool → trả OBSERVATION             │
    │   └────┬────┘                                               │
    │        │                                                    │
    │        └──────────► call_agent (loop lại với observation)    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

Conditional Routing (should_continue):
    - status == TOOL_CALLING  → "call_tool"    (cần gọi tool)
    - status == RUNNING       → "call_agent"   (tiếp tục suy luận)
    - status == DONE          → END            (có câu trả lời)
    - status == ERROR         → END            (gặp lỗi)
    - status == HANDOFF       → END            (chuyển giao, mở rộng sau)

Persistent Conversation Memory (SqliteSaver):
    - Mỗi session_id tương ứng với 1 thread_id trong checkpointer
    - Graph tự lưu state vào SQLite sau mỗi invoke
    - Khi invoke lại cùng thread_id → checkpointer nạp state cũ
    - current_step được reset mỗi query mới để tránh vượt step limit

Cách sử dụng:
    from agent.graph import invoke_agent

    # Lượt chat 1
    result1 = invoke_agent(
        query="Tên tôi là Minh",
        session_id="session-001",
    )
    print(result1["final_answer"])

    # Lượt chat 2 — agent nhớ ngữ cảnh
    result2 = invoke_agent(
        query="Tên tôi là gì?",
        session_id="session-001",
    )
    print(result2["final_answer"])  # → "Minh"

Tham khảo:
    - Plan.md Module 5: LangGraph Workflow
    - agent/state.py: AgentState TypedDict, AgentStatus
    - agent/runner.py: AgentRunner.call_agent()
    - agent/tool_executor.py: ToolExecutor.call_tool()
    - LangGraph documentation: StateGraph, add_conditional_edges
    - LangGraph checkpoint: SqliteSaver for persistent memory
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
# ROUTER — should_continue() quyết định flow tiếp theo
# ═══════════════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    """
    Router function cho LangGraph conditional edges.

    Kiểm tra trạng thái agent sau mỗi bước call_agent
    để quyết định flow tiếp theo trong graph.

    Routing logic:
        ┌─────────────────────┬──────────────────────────────┐
        │ Status              │ Route đến                    │
        ├─────────────────────┼──────────────────────────────┤
        │ TOOL_CALLING        │ "call_tool" (thực thi tool)  │
        │ RUNNING             │ "call_agent" (tiếp tục loop) │
        │ DONE                │ END (kết thúc, có answer)    │
        │ ERROR               │ END (kết thúc, có lỗi)      │
        │ HANDOFF             │ END (chuyển giao agent khác) │
        │ (unknown)           │ END (fallback an toàn)       │
        └─────────────────────┴──────────────────────────────┘

    Args:
        state: AgentState hiện tại từ LangGraph.

    Returns:
        Tên node tiếp theo ("call_agent", "call_tool") hoặc END.
    """
    status = state.get("status", "")
    current_step = state.get("current_step", 0)
    session_preview = state.get("session_id", "unknown")[:8]

    logger.debug(
        f"[Step {current_step}] should_continue | "
        f"status='{status}' | session={session_preview}..."
    )

    # ── TOOL_CALLING → cần gọi tool ──────────────────────────────────
    if status == AgentStatus.TOOL_CALLING.value:
        tool_name = state.get("tool_name", "")
        logger.info(
            f"[Step {current_step}] Routing → call_tool | "
            f"tool='{tool_name}' | session={session_preview}..."
        )
        return "call_tool"

    # ── RUNNING → tiếp tục suy luận (THOUGHT) ────────────────────────
    if status == AgentStatus.RUNNING.value:
        logger.info(
            f"[Step {current_step}] Routing → call_agent | "
            f"continuing loop | session={session_preview}..."
        )
        return "call_agent"

    # ── DONE → đã có câu trả lời ─────────────────────────────────────
    if status == AgentStatus.DONE.value:
        answer_preview = state.get("final_answer", "")[:50]
        logger.info(
            f"[Step {current_step}] Routing → END (DONE) | "
            f"answer='{answer_preview}...' | session={session_preview}..."
        )
        return END

    # ── ERROR → gặp lỗi không recover được ───────────────────────────
    if status == AgentStatus.ERROR.value:
        error = state.get("error", "unknown error")
        logger.warning(
            f"[Step {current_step}] Routing → END (ERROR) | "
            f"error='{error}' | session={session_preview}..."
        )
        return END

    # ── HANDOFF → chuyển giao cho agent khác ──────────────────────────
    if status == AgentStatus.HANDOFF.value:
        target = state.get("handoff_target", "unknown")
        reason = state.get("handoff_reason", "N/A")
        logger.info(
            f"[Step {current_step}] Routing → END (HANDOFF) | "
            f"target='{target}' | reason='{reason}' | "
            f"session={session_preview}..."
        )
        return END

    # ── Fallback — status không nhận diện được ────────────────────────
    logger.warning(
        f"[Step {current_step}] Unknown status '{status}' — "
        f"routing to END (fallback) | session={session_preview}..."
    )
    return END


# ═══════════════════════════════════════════════════════════════════════
# GRAPH BUILDER — Xây dựng StateGraph
# ═══════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Xây dựng LangGraph StateGraph cho ReAct agent loop.

    Tạo graph với:
        - 2 nodes: call_agent, call_tool
        - 1 entry point: call_agent
        - 1 conditional edge: call_agent → should_continue → routing
        - 1 direct edge: call_tool → call_agent (loop lại)

    Graph structure:
        START → call_agent → should_continue() → call_tool → call_agent
                                               → call_agent (THOUGHT)
                                               → END (DONE/ERROR/HANDOFF)

    Returns:
        Compiled StateGraph sẵn sàng invoke.

    Raises:
        GraphExecutionError: Khi build graph thất bại.

    Lưu ý:
        - AgentRunner và ToolExecutor được khởi tạo bên trong function
          để tránh import circular và đảm bảo lazy initialization.
        - Graph chỉ nên build 1 lần và cache (dùng get_graph()).
    """
    try:
        # Lazy import để tránh circular dependency
        from agent.runner import AgentRunner
        from agent.tool_executor import ToolExecutor

        # ── Khởi tạo node handlers ────────────────────────────────────
        runner = AgentRunner()
        executor = ToolExecutor()

        logger.info("Building LangGraph StateGraph...")

        # ── Tạo StateGraph ────────────────────────────────────────────
        graph = StateGraph(AgentState)

        # ── Thêm nodes ───────────────────────────────────────────────
        graph.add_node("call_agent", runner.call_agent)
        graph.add_node("call_tool", executor.call_tool)

        logger.debug("Added nodes: call_agent, call_tool")

        # ── Set entry point ──────────────────────────────────────────
        graph.set_entry_point("call_agent")

        logger.debug("Set entry point: call_agent")

        # ── Thêm conditional edges từ call_agent ─────────────────────
        # Sau call_agent, should_continue() quyết định đi đâu tiếp
        graph.add_conditional_edges(
            source="call_agent",
            path=should_continue,
            path_map={
                "call_tool": "call_tool",
                "call_agent": "call_agent",
                END: END,
            },
        )

        logger.debug(
            "Added conditional edges: call_agent → "
            "{call_tool, call_agent, END}"
        )

        # ── Thêm direct edge từ call_tool về call_agent ──────────────
        # Sau call_tool (observation), luôn quay lại call_agent
        graph.add_edge("call_tool", "call_agent")

        logger.debug("Added edge: call_tool → call_agent")

        # ── Khởi tạo checkpointer ─────────────────────────────────────
        import sqlite3

        db_path = settings.CHECKPOINT_DB_PATH
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        logger.info(
            f"SqliteSaver checkpointer initialized | "
            f"db_path='{db_path}'"
        )

        # ── Compile graph VỚI checkpointer ────────────────────────────
        compiled = graph.compile(checkpointer=checkpointer)

        logger.info(
            "LangGraph StateGraph compiled successfully | "
            "nodes=['call_agent', 'call_tool'] | "
            "entry_point='call_agent' | "
            "checkpointer=SqliteSaver"
        )

        return compiled

    except Exception as e:
        logger.error(f"Failed to build graph: {e}", exc_info=True)
        raise GraphExecutionError(
            "Failed to build LangGraph workflow",
            details={"error": str(e), "type": type(e).__name__},
        ) from e


# ═══════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION — get_graph() với caching (singleton)
# ═══════════════════════════════════════════════════════════════════════

def get_graph():
    """
    Factory function trả về compiled graph (singleton pattern).

    Graph chỉ được compile 1 lần duy nhất, sau đó cache lại
    để reuse cho tất cả requests. Thread-safe với threading.Lock.

    Tại sao cần singleton:
        - Compile graph tốn thời gian (khởi tạo AgentRunner, ToolExecutor)
        - Graph là stateless — state được truyền qua invoke()
        - Tránh overhead tạo LLM client mỗi request

    Returns:
        Compiled StateGraph instance.

    Raises:
        GraphExecutionError: Khi compile graph thất bại.

    Ví dụ:
        # Trong FastAPI lifespan hoặc bất kỳ đâu
        graph = get_graph()
        result = graph.invoke(initial_state)

        # Gọi lại → trả cached instance (không compile lại)
        graph2 = get_graph()
        assert graph is graph2  # True — cùng instance

    Thread-safety:
        Nhiều requests đồng thời gọi get_graph() lần đầu:
        - Chỉ 1 thread compile graph (acquire lock)
        - Các threads khác đợi → nhận cached instance
    """
    global _compiled_graph

    if _compiled_graph is not None:
        return _compiled_graph

    with _graph_lock:
        # Double-check locking pattern
        # Thread khác có thể đã compile trong khi thread này đợi lock
        if _compiled_graph is not None:
            return _compiled_graph

        logger.info("First call to get_graph() — compiling graph...")
        _compiled_graph = build_graph()
        logger.info("Graph cached successfully (singleton)")

        return _compiled_graph


def reset_graph() -> None:
    """
    Reset cached graph (dùng cho testing hoặc hot-reload).

    Xóa cached instance để lần gọi get_graph() tiếp theo
    sẽ compile graph mới.

    Hữu ích khi:
        - Unit testing: mỗi test cần graph mới
        - Hot-reload: khi thay đổi tool configuration
        - Debugging: force re-initialization

    Ví dụ:
        # Trong test fixture
        @pytest.fixture(autouse=True)
        def reset():
            reset_graph()
            yield
            reset_graph()
    """
    global _compiled_graph

    with _graph_lock:
        _compiled_graph = None
        logger.info("Graph cache cleared — will recompile on next get_graph()")




# ═══════════════════════════════════════════════════════════════════════
# INVOKE HELPER — Convenience function để chạy agent (có checkpointer)
# ═══════════════════════════════════════════════════════════════════════

def invoke_agent(
    query: str,
    user_id: str,
    agent_profile: str = "personal_agent",
    session_id: str | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """
    Convenience function để chạy agent — hỗ trợ duy trì hội thoại.

    Kiểm tra xem thread_id (= session_id) đã có state trong
    checkpointer chưa:
        - Nếu ĐÃ CÓ: Tiếp tục hội thoại — chỉ gửi query mới
          + reset current_step để mỗi query có budget bước riêng.
          + user_id đã được lưu từ lần đầu, không cần truyền lại.
        - Nếu CHƯA CÓ: Tạo initial_state() rồi invoke như mới.

    Graph tự động lưu state vào SQLite sau mỗi invoke nhờ
    SqliteSaver checkpointer.

    Args:
        query: Câu hỏi từ user.
        user_id: ID người dùng (bắt buộc, dùng cho multi-tenant filtering).
        agent_profile: Tên agent profile (mặc định "personal_agent").
        session_id: ID session (optional, tự tạo nếu None).
                    Dùng làm thread_id cho checkpointer.
        max_steps: Giới hạn bước (optional, dùng settings nếu None).

    Returns:
        Dict chứa final AgentState sau khi graph chạy xong.
        Các key quan trọng:
            - final_answer: Câu trả lời cuối cùng
            - tool_observations: Danh sách observations
            - current_step: Số bước đã thực hiện
            - status: Trạng thái cuối (done/error/handoff)
            - error: Thông tin lỗi (nếu có)

    Raises:
        ValueError: Khi query hoặc user_id rỗng.
        GraphExecutionError: Khi graph execution thất bại.

    Ví dụ:
        # Lượt 1
        result1 = invoke_agent(
            "Tên tôi là Minh",
            user_id="user_123",
            session_id="session-001",
        )
        # Lượt 2 — agent nhớ tên
        result2 = invoke_agent(
            "Tên tôi là gì?",
            user_id="user_123",
            session_id="session-001",
        )
    """
    import uuid
    from agent.state import AgentStatus, create_initial_state

    # Resolve session_id
    resolved_session_id = session_id or str(uuid.uuid4())
    resolved_max_steps = (
        max_steps if max_steps is not None else settings.MAX_AGENT_STEPS
    )

    # Lấy compiled graph (có checkpointer)
    graph = get_graph()

    # Config cho checkpointer — thread_id = session_id
    config = {"configurable": {"thread_id": resolved_session_id}}

    session_preview = resolved_session_id[:8]

    # ── Kiểm tra state đã tồn tại trên thread_id này chưa ────────
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
        # ── TIẾP TỤC HỘI THOẠI ──────────────────────────────────
        # State cũ đã có trong checkpointer → chỉ gửi query mới
        # + reset các field loop control cho query mới
        logger.info(
            f"Continuing conversation | "
            f"thread_id={session_preview}... | "
            f"query='{query[:50]}...'"
        )

        input_state = {
            "query": query.strip(),
            # Reset loop control cho query mới
            "current_step": 0,
            "max_steps": resolved_max_steps,
            "status": AgentStatus.RUNNING.value,
            "current_action": "",
            # Reset tool state
            "tool_name": "",
            "tool_args": {},
            # Reset output
            "final_answer": "",
            "error": "",
        }
    else:
        # ── HỘI THOẠI MỚI ────────────────────────────────────────
        # Chưa có state → tạo initial state đầy đủ
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

    # ── Invoke graph với config (checkpointer tự lưu state) ──────
    try:
        result = graph.invoke(input_state, config)

        logger.info(
            f"Agent completed | "
            f"status={result.get('status')} | "
            f"steps={result.get('current_step')} | "
            f"has_answer={bool(result.get('final_answer'))} | "
            f"continued={has_existing_state} | "
            f"thread_id={session_preview}..."
        )

        return result

    except Exception as e:
        logger.error(
            f"Agent invocation failed: {e} | "
            f"thread_id={session_preview}...",
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
