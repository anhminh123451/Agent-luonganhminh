"""
ToolExecutor cho Personal AI Agent — Native Function Calling.

Module này chứa logic call_tool() — node trong LangGraph
thực thi các tool calls được LLM yêu cầu qua pending_tool_calls.
Đồng thời xử lý ngắt phiên (AuthInterrupt) khi gặp tool Gmail mà user chưa authorize.

Tham khảo:
    - implementation_plan2.md: Phase 6 & Phase 8
"""

from __future__ import annotations

from typing import Any, cast

from core.logger import get_logger
from agent.state import AgentState, AgentStatus, MessageRole, get_state_summary
from agent.auth_interrupt import AuthInterrupt
from tools.registry import ToolRegistry
from tools.gmail_mcp_tools import (
    is_gmail_tool,
    check_user_gmail_authenticated,
    execute_gmail_tool,
)
from core.google_oauth import build_authorization_url
from tools.base import ToolResult

logger = get_logger(__name__)


class ToolExecutor:
    """
    Quản lý việc thực thi tool trong Native Function Calling loop.
    """

    def __init__(self) -> None:
        logger.info("ToolExecutor initialized")

    def call_tool(self, state: AgentState) -> dict:
        """
        LangGraph node — thực thi danh sách pending_tool_calls.

        Flow:
        1. Đọc pending_tool_calls từ state
        2. Với mỗi tool call:
           a. Kiểm tra nếu là Gmail tool:
              - Kiểm tra user đã authorize Google chưa (Redis / DB).
              - Nếu chưa → kích hoạt AuthInterrupt và kết thúc ngay.
              - Nếu có → thực thi qua execute_gmail_tool().
           b. Nếu là tool thông thường (document_search, web_search):
              - Tiêm user_id vào args.
              - safe_run(**args) qua ToolRegistry.
           c. Đóng gói kết quả thành Message role="tool".
        3. Append tất cả tool results vào messages.
        4. Clear pending_tool_calls.
        5. Trả về partial state update (status = RUNNING để loop lại call_agent).
        """
        pending_calls = state.get("pending_tool_calls", [])
        user_id = state.get("user_id")
        current_step = state.get("current_step", 0)
        session_preview = state.get("session_id", "unknown")[:8]

        logger.info(
            f"[Step {current_step}] call_tool | "
            f"n_calls={len(pending_calls)} | session={session_preview}..."
        )

        if not pending_calls:
            logger.warning(f"[Step {current_step}] call_tool called with empty pending_tool_calls")
            return {
                "pending_tool_calls": [],
                "status": AgentStatus.RUNNING.value,
            }

        tool_messages: list[dict[str, Any]] = []

        for tc in pending_calls:
            call_id = tc.get("id") or "call_unknown"
            tool_name = tc.get("name", "")
            args = dict(tc.get("args") or {})

            logger.info(
                f"[Step {current_step}] Executing tool '{tool_name}' (id={call_id}) | "
                f"args={args}"
            )

            # ── 1. Kiểm tra nếu là Gmail tool ─────────────────────────
            if is_gmail_tool(tool_name):
                # Kiểm tra quyền truy cập Google OAuth của user
                if not check_user_gmail_authenticated(user_id):
                    logger.info(
                        f"[Step {current_step}] User {user_id} not authenticated with Google — "
                        f"triggering AuthInterrupt for tool '{tool_name}'"
                    )
                    # Tạo URL authorization với state = user_id
                    auth_url = build_authorization_url(state=str(user_id))
                    return AuthInterrupt.build_auth_state_update(
                        authorize_url=auth_url,
                        service="google",
                    )

                # Đã authorize → thực thi Gmail tool qua MCP
                result: ToolResult = execute_gmail_tool(
                    tool_name=tool_name,
                    args=args,
                    user_id=user_id,
                )

            # ── 2. Tool thông thường (document_search, web_search, ...) 
            else:
                if not ToolRegistry.has(tool_name):
                    available = ToolRegistry.available_tools()
                    result = ToolResult(
                        context="",
                        source=tool_name,
                        success=False,
                        error=f"Tool '{tool_name}' không tồn tại. Các tool khả dụng: {', '.join(available)}",
                    )
                else:
                    # Tiêm user_id bảo mật (ngăn chặn cross-user access)
                    if user_id:
                        args["user_id"] = user_id

                    tool = ToolRegistry.get(tool_name)
                    result = tool.safe_run(**args)

            # ── 3. Định dạng message role="tool" ───────────────────────
            if result.success:
                content = result.context
                logger.info(
                    f"[Step {current_step}] Tool '{tool_name}' SUCCESS | "
                    f"result_length={len(content)} chars"
                )
            else:
                content = f"Lỗi khi thực thi tool '{tool_name}': {result.error}"
                logger.warning(
                    f"[Step {current_step}] Tool '{tool_name}' FAILED | "
                    f"error='{result.error}'"
                )

            tool_messages.append({
                "role": MessageRole.TOOL.value,
                "tool_call_id": call_id,
                "name": tool_name,
                "content": content,
            })

        # ── 4. Build partial state update ─────────────────────────────
        new_messages = list(state.get("messages", [])) + tool_messages

        state_update: dict[str, Any] = {
            "messages": new_messages,
            "pending_tool_calls": [],
            "status": AgentStatus.RUNNING.value,
        }

        summary = get_state_summary(cast(AgentState, {**state, **state_update}))
        logger.debug(f"State after call_tool: {summary}")

        return state_update
