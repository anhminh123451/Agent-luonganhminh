"""
Auth Interrupt Pattern cho Personal AI Agent.

Module này cung cấp AuthInterrupt helper để suspend LangGraph session
khi agent cần gọi tool bên thứ 3 (như Gmail) nhưng user chưa cấp quyền OAuth.
Khi đó, agent trả auth_required + authorize_url về FE để mở popup OAuth,
và kết thúc lượt chạy hiện tại. Sau khi user authorize xong, FE tự động retry
cùng session_id để tiếp tục từ checkpoint.

Tham khảo:
    - implementation_plan2.md: Phase 8
"""

from __future__ import annotations

from typing import Any

from agent.state import AgentStatus


class AuthInterrupt:
    """
    Helper dùng để interrupt/suspend agent loop khi cần OAuth authentication.
    """

    AUTH_REQUIRED_KEY = "auth_required"
    AUTH_URL_KEY = "auth_url"
    AUTH_SERVICE_KEY = "auth_service"

    @staticmethod
    def build_auth_state_update(
        authorize_url: str,
        service: str = "google",
    ) -> dict[str, Any]:
        """
        Trả về partial state update để kết thúc graph với thông tin OAuth.
        """
        return {
            "auth_required": True,
            "auth_url": authorize_url,
            "authorize_url": authorize_url,
            "auth_service": service,
            "service": service,
            "status": AgentStatus.AUTH_REQUIRED.value,
            "error": "",  # Không phải lỗi, chỉ là suspend session chờ auth
            "pending_tool_calls": [],
        }
