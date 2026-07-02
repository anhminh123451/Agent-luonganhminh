"""
ToolExecutor cho Personal AI Agent — Agent Core (Module 4).

Module này chứa logic call_tool() — node trong LangGraph
đảm nhiệm việc thực thi tool mà agent đã quyết định gọi.

Kiến trúc: ToolExecutor Pattern
    - ToolExecutor class: Quản lý việc thực thi tool từ AgentState
    - call_tool(): LangGraph node — nhận AgentState, gọi tool, trả observation
    - _validate_tool_permission(): Kiểm tra agent có quyền gọi tool không
    - _format_observation_message(): Chuyển ToolResult → observation message

ReAct Loop — Vai trò của call_tool():
    ┌──────────────────────────────────────────────────────────────┐
    │ call_agent() → ACTION: {"tool": "faq_search", "args": {...}}│
    │     ↓                                                        │
    │ should_continue() → routing đến call_tool node               │
    │     ↓                                                        │
    │ ★ call_tool() ← CHÚNG TA Ở ĐÂY                             │
    │   1. Đọc tool_name + tool_args từ AgentState                 │
    │   2. Validate tool tồn tại trong ToolRegistry                │
    │   3. Validate agent profile có quyền gọi tool                │
    │   4. Gọi tool.safe_run(**tool_args) → ToolResult             │
    │   5. Format observation → append vào messages + observations │
    │   6. Trả partial state update cho LangGraph                  │
    │     ↓                                                        │
    │ should_continue() → routing về call_agent (tiếp tục loop)    │
    └──────────────────────────────────────────────────────────────┘

State Updates từ call_tool():
    - messages += [observation message]       (role=OBSERVATION)
    - tool_observations += [observation text]
    - status = RUNNING                         (sẵn sàng cho call_agent tiếp)
    - tool_name = ""                           (reset)
    - tool_args = {}                           (reset)

Cách sử dụng:
    from agent.tool_executor import ToolExecutor

    # Tạo executor instance (1 lần khi startup)
    executor = ToolExecutor()

    # Sử dụng trong LangGraph node
    def call_tool_node(state: AgentState) -> dict:
        return executor.call_tool(state)

    # Hoặc đăng ký trực tiếp vào graph
    graph.add_node("call_tool", executor.call_tool)

Tham khảo:
    - Plan.md Module 4: Agent Core & Prompt Engineering
    - agent/state.py: AgentState TypedDict
    - agent/prompts.py: format_observation(), format_error_recovery()
    - tools/registry.py: ToolRegistry
    - tools/base.py: BaseTool, ToolResult
"""

from __future__ import annotations

from typing import Any

from core.logger import get_logger

from agent.state import (
    AgentState,
    AgentStatus,
    MessageRole,
    add_message,
    add_observation,
    get_state_summary,
)
from agent.prompts import (
    format_observation,
    format_error_recovery,
)

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# TOOL EXECUTOR — Class chính quản lý call_tool logic
# ═══════════════════════════════════════════════════════════════════════

class ToolExecutor:
    """
    Quản lý việc thực thi tool trong ReAct loop.

    ToolExecutor là LangGraph node — nhận AgentState (sau khi call_agent
    quyết định ACTION), gọi tool tương ứng qua ToolRegistry, và trả
    partial state update chứa observation.

    Nguyên tắc thiết kế:
        1. FAIL-SAFE: Luôn trả observation, kể cả khi tool gặp lỗi
        2. PERMISSION-CHECKED: Validate tool access theo agent profile
        3. OBSERVABLE: Log chi tiết mọi tool execution
        4. STATELESS: Không giữ state nội bộ giữa các lần gọi

    Lifecycle:
        1. Khởi tạo 1 lần khi app startup
        2. call_tool() được gọi mỗi khi agent ra quyết định ACTION
        3. ToolRegistry được truy cập lazy (không inject lúc init)

    Ví dụ:
        executor = ToolExecutor()

        # Dùng trong LangGraph node
        graph.add_node("call_tool", executor.call_tool)
    """

    def __init__(self) -> None:
        """
        Khởi tạo ToolExecutor.

        ToolRegistry không được inject vào constructor vì:
        - Registry là class-level state (ClassVar), truy cập qua classmethod
        - Lazy import tránh circular dependency (agent → tools → agent)
        """
        logger.info("ToolExecutor initialized")

    # ─── Permission validation ────────────────────────────────────────

    def _validate_tool_permission(
        self,
        tool_name: str,
        agent_profile: str,
    ) -> bool:
        """
        Kiểm tra agent có quyền gọi tool này không.

        Truy vấn ToolRegistry.is_tool_allowed() để xác nhận
        tool nằm trong danh sách allowed_tools của agent profile.

        Args:
            tool_name: Tên tool cần kiểm tra.
            agent_profile: Tên agent profile hiện tại.

        Returns:
            True nếu được phép, False nếu không.
        """
        from tools.registry import ToolRegistry

        is_allowed = ToolRegistry.is_tool_allowed(agent_profile, tool_name)

        if not is_allowed:
            logger.warning(
                f"Tool '{tool_name}' is NOT allowed for "
                f"profile '{agent_profile}' | "
                f"allowed: {ToolRegistry.get_tools_for_profile(agent_profile)}"
            )

        return is_allowed

    # ─── Observation formatting ───────────────────────────────────────

    def _build_observation_message(
        self,
        tool_name: str,
        observation_text: str,
        success: bool,
    ) -> str:
        """
        Build observation message hoàn chỉnh cho agent.

        Dùng prompt templates từ prompts.py để format observation
        theo đúng chuẩn mà agent mong đợi trong ReAct loop.

        Args:
            tool_name: Tên tool đã thực thi.
            observation_text: Nội dung observation từ ToolResult.
            success: True nếu tool chạy thành công.

        Returns:
            Chuỗi observation đã format.

        Ví dụ:
            # Success case:
            # "OBSERVATION: [faq_search] Lãi suất tiết kiệm 12 tháng là 5.5%/năm."
            #
            # Error case:
            # "OBSERVATION: [faq_search] ❌ LỖI: Connection timeout
            #  Hãy thử cách tiếp cận khác hoặc dùng tool khác."
        """
        if success:
            return format_observation(
                tool_name=tool_name,
                observation=observation_text,
            )
        else:
            return format_error_recovery(
                tool_name=tool_name,
                error_message=observation_text,
            )

    # ─── Core call_tool logic ─────────────────────────────────────────

    def call_tool(self, state: AgentState) -> dict:
        """
        LangGraph node — thực thi tool và trả observation.

        Đây là bước OBSERVATION trong ReAct loop. Mỗi lần gọi:
        1. Đọc tool_name + tool_args từ AgentState
        2. Validate tool tồn tại trong ToolRegistry
        3. Validate agent profile có quyền gọi tool
        4. Gọi tool.safe_run(**tool_args)
        5. Format observation message
        6. Trả partial state update

        Args:
            state: AgentState hiện tại từ LangGraph.
                   Yêu cầu: tool_name và tool_args đã được set bởi call_agent.

        Returns:
            Dict partial state update cho LangGraph merge:
                - messages: list mới với observation message
                - tool_observations: list mới với observation text
                - tool_name: "" (reset)
                - tool_args: {} (reset)
                - status: AgentStatus.RUNNING (sẵn sàng cho call_agent tiếp)

        Flow diagram:
            ┌─────────────────────────────────────────────────┐
            │ Input: state[tool_name], state[tool_args]       │
            ├─────────────────────────────────────────────────┤
            │ Validate tool_name không rỗng                   │
            │     → Nếu rỗng: trả error observation           │
            ├─────────────────────────────────────────────────┤
            │ Validate tool tồn tại trong ToolRegistry        │
            │     → Nếu không: trả error observation           │
            ├─────────────────────────────────────────────────┤
            │ Validate permission (profile → tool)            │
            │     → Nếu không: trả error observation           │
            ├─────────────────────────────────────────────────┤
            │ Gọi tool.safe_run(**tool_args)                  │
            │     → Luôn trả ToolResult (success hoặc error)  │
            ├─────────────────────────────────────────────────┤
            │ Format observation + update state               │
            └─────────────────────────────────────────────────┘
        """
        tool_name = state.get("tool_name", "")
        tool_args = state.get("tool_args", {})
        agent_profile = state.get("agent_profile", "")
        session_preview = state["session_id"][:8]
        current_step = state["current_step"]

        logger.info(
            f"[Step {current_step}] call_tool | "
            f"tool='{tool_name}' | "
            f"args={tool_args} | "
            f"session={session_preview}..."
        )

        # ── 1. Validate tool_name không rỗng ─────────────────────────
        if not tool_name:
            error_msg = (
                "Agent đã yêu cầu gọi tool nhưng không chỉ định tool_name. "
                "Hãy thử lại với tên tool cụ thể."
            )
            logger.error(
                f"[Step {current_step}] call_tool failed: "
                f"empty tool_name | session={session_preview}..."
            )
            return self._build_error_state_update(
                state=state,
                tool_name="unknown",
                error_msg=error_msg,
            )

        # ── 2. Validate tool tồn tại trong ToolRegistry ──────────────
        from tools.registry import ToolRegistry

        if not ToolRegistry.has(tool_name):
            available = ToolRegistry.available_tools()
            error_msg = (
                f"Tool '{tool_name}' không tồn tại trong hệ thống. "
                f"Các tool khả dụng: {', '.join(available) if available else 'none'}. "
                f"Hãy sử dụng đúng tên tool."
            )
            logger.warning(
                f"[Step {current_step}] Tool not found: '{tool_name}' | "
                f"available={available} | session={session_preview}..."
            )
            return self._build_error_state_update(
                state=state,
                tool_name=tool_name,
                error_msg=error_msg,
            )

        # ── 3. Validate permission ────────────────────────────────────
        if agent_profile and not self._validate_tool_permission(tool_name, agent_profile):
            allowed_tools = [
                t.name for t in ToolRegistry.get_tools_for_profile(agent_profile)
            ]
            error_msg = (
                f"Tool '{tool_name}' không được phép cho profile '{agent_profile}'. "
                f"Các tool được phép: {', '.join(allowed_tools) if allowed_tools else 'none'}. "
                f"Hãy sử dụng tool khác."
            )
            logger.warning(
                f"[Step {current_step}] Permission denied: "
                f"tool='{tool_name}' profile='{agent_profile}' | "
                f"session={session_preview}..."
            )
            return self._build_error_state_update(
                state=state,
                tool_name=tool_name,
                error_msg=error_msg,
            )

        # ── 4. Thực thi tool ──────────────────────────────────────────
        tool = ToolRegistry.get(tool_name)

        logger.info(
            f"[Step {current_step}] Executing tool '{tool_name}' | "
            f"args={tool_args} | session={session_preview}..."
        )

        # safe_run() KHÔNG BAO GIỜ throw exception
        result = tool.safe_run(**tool_args)

        # ── 5. Format observation ─────────────────────────────────────
        if result.success:
            observation_text = result.to_observation()
            observation_msg = self._build_observation_message(
                tool_name=tool_name,
                observation_text=result.context,
                success=True,
            )

            logger.info(
                f"[Step {current_step}] Tool '{tool_name}' SUCCESS | "
                f"observation_length={len(observation_text)} chars | "
                f"session={session_preview}..."
            )
        else:
            error_detail = result.error or "Unknown error"
            observation_text = result.to_observation()
            observation_msg = self._build_observation_message(
                tool_name=tool_name,
                observation_text=error_detail,
                success=False,
            )

            logger.warning(
                f"[Step {current_step}] Tool '{tool_name}' FAILED | "
                f"error='{error_detail}' | "
                f"session={session_preview}..."
            )

        # ── 6. Build partial state update ─────────────────────────────
        new_messages = add_message(
            state,
            MessageRole.OBSERVATION.value,
            observation_msg,
        )
        new_observations = add_observation(state, observation_text)

        state_update: dict[str, Any] = {
            "messages": new_messages,
            "tool_observations": new_observations,
            "tool_name": "",       # Reset sau khi đã thực thi
            "tool_args": {},       # Reset sau khi đã thực thi
            "status": AgentStatus.RUNNING.value,  # Sẵn sàng cho call_agent
        }

        # Log state summary
        from typing import cast
        merged_state = {**state, **state_update}
        summary = get_state_summary(cast(AgentState, merged_state))
        logger.debug(f"State after call_tool: {summary}")

        return state_update

    # ─── Helper: Build error state update ─────────────────────────────

    def _build_error_state_update(
        self,
        state: AgentState,
        tool_name: str,
        error_msg: str,
    ) -> dict[str, Any]:
        """
        Build partial state update khi tool execution gặp lỗi.

        Tạo error observation để agent biết tool đã fail
        và có thể thử cách tiếp cận khác.

        KHÔNG set status=ERROR vì đây là tool error, không phải agent error.
        Agent vẫn có thể tiếp tục loop (thử tool khác hoặc trả ANSWER
        dựa trên thông tin đã có).

        Args:
            state: AgentState hiện tại.
            tool_name: Tên tool bị lỗi.
            error_msg: Mô tả lỗi chi tiết.

        Returns:
            Dict partial state update với error observation.
        """
        # Format error observation
        observation_msg = self._build_observation_message(
            tool_name=tool_name,
            observation_text=error_msg,
            success=False,
        )
        observation_text = f"[{tool_name}] ERROR: {error_msg}"

        new_messages = add_message(
            state,
            MessageRole.OBSERVATION.value,
            observation_msg,
        )
        new_observations = add_observation(state, observation_text)

        return {
            "messages": new_messages,
            "tool_observations": new_observations,
            "tool_name": "",       # Reset
            "tool_args": {},       # Reset
            "status": AgentStatus.RUNNING.value,  # Agent vẫn tiếp tục
        }
