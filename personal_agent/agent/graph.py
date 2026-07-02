"""
LangGraph Workflow cho Personal AI Agent — Module 5.

Module này định nghĩa toàn bộ LangGraph StateGraph workflow,
compile graph một lần (singleton pattern) và cung cấp factory
function get_graph() để reuse trong suốt application lifecycle.

Kiến trúc: LangGraph StateGraph Workflow
    - build_graph(): Xây dựng StateGraph với nodes và edges
    - get_graph(): Factory function với caching (singleton)
    - should_continue(): Router function quyết định flow

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

Cách sử dụng:
    from agent.graph import get_graph
    from agent.state import create_initial_state

    # Lấy compiled graph (singleton, chỉ compile 1 lần)
    graph = get_graph()

    # Tạo initial state và invoke
    state = create_initial_state(
        query="Lãi suất tiết kiệm bao nhiêu?",
        agent_profile="personal_agent",
    )
    result = graph.invoke(state)

    print(result["final_answer"])

Tham khảo:
    - Plan.md Module 5: LangGraph Workflow
    - agent/state.py: AgentState TypedDict, AgentStatus
    - agent/runner.py: AgentRunner.call_agent()
    - agent/tool_executor.py: ToolExecutor.call_tool()
    - LangGraph documentation: StateGraph, add_conditional_edges
"""

from __future__ import annotations

import threading
from typing import Any

from langgraph.graph import END, StateGraph

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

        # ── Compile graph ────────────────────────────────────────────
        compiled = graph.compile()

        logger.info(
            "LangGraph StateGraph compiled successfully | "
            "nodes=['call_agent', 'call_tool'] | "
            "entry_point='call_agent'"
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
# INVOKE HELPER — Convenience function để chạy agent
# ═══════════════════════════════════════════════════════════════════════

def invoke_agent(
    query: str,
    agent_profile: str = "personal_agent",
    session_id: str | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """
    Convenience function để chạy agent từ đầu đến cuối.

    Kết hợp create_initial_state() + get_graph().invoke() thành
    một function call duy nhất cho các use case đơn giản.

    Args:
        query: Câu hỏi từ user.
        agent_profile: Tên agent profile (mặc định "personal_agent").
        session_id: ID session (optional, tự tạo nếu None).
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
        ValueError: Khi query rỗng.
        GraphExecutionError: Khi graph execution thất bại.

    Ví dụ:
        result = invoke_agent("Lãi suất tiết kiệm 12 tháng?")
        print(result["final_answer"])
        print(f"Completed in {result['current_step']} steps")
    """
    from agent.state import create_initial_state

    # Tạo initial state
    state = create_initial_state(
        query=query,
        agent_profile=agent_profile,
        session_id=session_id,
        max_steps=max_steps,
    )

    # Lấy compiled graph
    graph = get_graph()

    # Invoke graph
    session_preview = state["session_id"][:8]
    logger.info(
        f"Invoking agent | query='{query[:50]}...' | "
        f"profile={agent_profile} | session={session_preview}..."
    )

    try:
        result = graph.invoke(state)

        logger.info(
            f"Agent completed | "
            f"status={result.get('status')} | "
            f"steps={result.get('current_step')} | "
            f"has_answer={bool(result.get('final_answer'))} | "
            f"session={session_preview}..."
        )

        return result

    except Exception as e:
        logger.error(
            f"Agent invocation failed: {e} | "
            f"session={session_preview}...",
            exc_info=True,
        )
        raise GraphExecutionError(
            "Agent invocation failed",
            details={
                "query": query[:100],
                "profile": agent_profile,
                "session_id": state["session_id"],
                "error": str(e),
            },
        ) from e
