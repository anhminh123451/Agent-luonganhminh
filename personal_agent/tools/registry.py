"""
Tool Registry cho Banking AI Agent — Tool Layer.

Module này quản lý đăng ký, tra cứu, và phân quyền tools
theo agent profile, đảm bảo dễ mở rộng khi thêm tool mới.

Kiến trúc: Registry Pattern (giống EmbedderRegistry, VectorStoreRegistry)
    - ToolRegistry: Registry trung tâm quản lý mapping tool_name → tool instance
    - Hỗ trợ phân quyền theo agent profile (agent nào được dùng tool nào)
    - Auto-discovery: tự động đăng ký default tools khi khởi tạo

Cách mở rộng khi thêm tool mới:
    1. Tạo class NewTool(BaseTool) trong file riêng (ví dụ: new_tool.py)
    2. Implement name, description, args_schema, run()
    3. Đăng ký vào registry:
       - Cách 1 (thủ công): ToolRegistry.register(NewTool())
       - Cách 2 (tự động): thêm vào _register_default_tools()
    4. (Tùy chọn) Thêm tool vào agent profile:
       ToolRegistry.grant_tool("agent_name", "new_tool_name")

Cách sử dụng:
    from tools.registry import ToolRegistry

    # Đăng ký tool
    ToolRegistry.register(FAQTool())

    # Lấy tool theo tên
    tool = ToolRegistry.get("faq_search")
    result = tool.safe_run(query="savings account")

    # Lấy danh sách tools cho agent profile
    tools = ToolRegistry.get_tools_for_profile("banking_agent")

    # Lấy function declarations cho system prompt
    declarations = ToolRegistry.get_function_declarations()
"""

from __future__ import annotations

from typing import Any, ClassVar

from core.exceptions import ToolError, ToolNotFoundError
from core.logger import get_logger

from tools.base import BaseTool, ToolCategory, ToolResult

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# TOOL REGISTRY — Đăng ký và quản lý tools trung tâm
# ═══════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """
    Registry trung tâm quản lý toàn bộ tool instances.

    Chức năng chính:
        1. Đăng ký/hủy đăng ký tools
        2. Tra cứu tool theo tên
        3. Phân quyền tool theo agent profile
        4. Export function declarations cho LLM system prompt
        5. Liệt kê tools theo category

    Thread Safety:
        Registry sử dụng ClassVar (class-level state). Trong môi trường
        single-process (FastAPI uvicorn), điều này an toàn vì registry
        chỉ được setup 1 lần khi startup.

    Ví dụ:
        # Setup (gọi 1 lần khi app startup)
        ToolRegistry.register(FAQTool())
        ToolRegistry.register(BranchSearchTool())
        ToolRegistry.register(WebSearchTool())

        # Định nghĩa profile
        ToolRegistry.set_profile("banking_agent", ["faq_search", "branch_search"])
        ToolRegistry.set_profile("general_agent", ["faq_search", "web_search"])

        # Sử dụng trong agent
        tool = ToolRegistry.get("faq_search")
        result = tool.safe_run(query="lãi suất tiết kiệm")
    """

    # ─── Internal state ───────────────────────────────────────────────
    # Mapping: tool_name → BaseTool instance
    _registry: ClassVar[dict[str, BaseTool]] = {}

    # Mapping: profile_name → set of tool_names
    # Ví dụ: {"banking_agent": {"faq_search", "branch_search"}}
    _profiles: ClassVar[dict[str, set[str]]] = {}

    # ─── Registration ─────────────────────────────────────────────────

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """
        Đăng ký một tool vào registry.

        Nếu tool name đã tồn tại, sẽ ghi đè (upsert) và log warning.

        Args:
            tool: Instance của BaseTool subclass.

        Raises:
            ToolError: Khi tool không có name hoặc description.

        Ví dụ:
            ToolRegistry.register(FAQTool())
            ToolRegistry.register(BranchSearchTool())
        """
        # Validate tool metadata
        if not getattr(tool, "name", None):
            raise ToolError(
                "Tool must have a 'name' attribute",
                details={"tool_class": tool.__class__.__name__},
            )

        if not getattr(tool, "description", None):
            raise ToolError(
                f"Tool '{tool.name}' must have a 'description' attribute",
                details={"tool_name": tool.name},
            )

        key = tool.name.lower()

        # Warning nếu đăng ký đè
        if key in cls._registry:
            old_tool = cls._registry[key]
            logger.warning(
                f"Overwriting tool '{key}': "
                f"{old_tool.__class__.__name__} → {tool.__class__.__name__}"
            )

        cls._registry[key] = tool

        logger.info(
            f"Registered tool: '{key}' "
            f"({tool.__class__.__name__}, "
            f"category={tool.category.value}, "
            f"version={tool.version})"
        )

    @classmethod
    def register_many(cls, tools: list[BaseTool]) -> None:
        """
        Đăng ký nhiều tools cùng lúc.

        Args:
            tools: Danh sách BaseTool instances.

        Ví dụ:
            ToolRegistry.register_many([
                FAQTool(),
                BranchSearchTool(),
                WebSearchTool(),
            ])
        """
        for tool in tools:
            cls.register(tool)

        logger.info(
            f"Batch registered {len(tools)} tools: "
            f"{[t.name for t in tools]}"
        )

    @classmethod
    def unregister(cls, tool_name: str) -> None:
        """
        Hủy đăng ký một tool khỏi registry.

        Đồng thời xóa tool khỏi tất cả profiles đã cấp quyền.

        Args:
            tool_name: Tên tool cần hủy đăng ký.

        Raises:
            ToolNotFoundError: Tool không tồn tại trong registry.
        """
        key = tool_name.lower()

        if key not in cls._registry:
            raise ToolNotFoundError(
                f"Cannot unregister — tool '{tool_name}' not found",
                details={
                    "tool_name": tool_name,
                    "available_tools": cls.available_tools(),
                },
            )

        del cls._registry[key]

        # Xóa khỏi tất cả profiles
        for profile_tools in cls._profiles.values():
            profile_tools.discard(key)

        logger.info(f"Unregistered tool: '{key}'")

    # ─── Lookup ───────────────────────────────────────────────────────

    @classmethod
    def get(cls, tool_name: str) -> BaseTool:
        """
        Lấy tool instance theo tên.

        Args:
            tool_name: Tên tool (case-insensitive).

        Returns:
            BaseTool instance.

        Raises:
            ToolNotFoundError: Tool không tồn tại trong registry.

        Ví dụ:
            tool = ToolRegistry.get("faq_search")
            result = tool.safe_run(query="lãi suất")
        """
        key = tool_name.lower()
        tool = cls._registry.get(key)

        if tool is None:
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not found in registry",
                details={
                    "requested": tool_name,
                    "available_tools": cls.available_tools(),
                },
            )

        return tool

    @classmethod
    def has(cls, tool_name: str) -> bool:
        """
        Kiểm tra tool đã được đăng ký chưa.

        Args:
            tool_name: Tên tool cần kiểm tra.

        Returns:
            True nếu tool tồn tại, False nếu không.
        """
        return tool_name.lower() in cls._registry

    @classmethod
    def available_tools(cls) -> list[str]:
        """Trả về danh sách tên tất cả tools đã đăng ký."""
        return list(cls._registry.keys())

    @classmethod
    def get_all(cls) -> dict[str, BaseTool]:
        """
        Trả về bản copy của toàn bộ registry.

        Returns:
            Dict mapping tool_name → BaseTool instance.
        """
        return dict(cls._registry)

    @classmethod
    def get_by_category(cls, category: ToolCategory) -> list[BaseTool]:
        """
        Lấy tất cả tools thuộc một category.

        Args:
            category: ToolCategory enum value.

        Returns:
            Danh sách BaseTool instances thuộc category đó.

        Ví dụ:
            retrieval_tools = ToolRegistry.get_by_category(ToolCategory.RETRIEVAL)
        """
        return [
            tool for tool in cls._registry.values()
            if tool.category == category
        ]

    # ─── Profile management (phân quyền tool theo agent) ──────────────

    @classmethod
    def set_profile(cls, profile_name: str, tool_names: list[str]) -> None:
        """
        Định nghĩa một agent profile với danh sách tools được phép dùng.

        Profile cho phép kiểm soát tool nào agent nào được gọi.
        Ví dụ: banking_agent chỉ được dùng faq_search + branch_search,
        không được dùng web_search.

        Args:
            profile_name: Tên profile (ví dụ: "banking_agent").
            tool_names: Danh sách tên tools được phép.

        Raises:
            ToolNotFoundError: Nếu tool_name chưa đăng ký trong registry.

        Ví dụ:
            ToolRegistry.set_profile("banking_agent", [
                "faq_search",
                "branch_search",
            ])
        """
        # Validate: tất cả tools phải đã đăng ký
        validated_names = set()
        for name in tool_names:
            key = name.lower()
            if key not in cls._registry:
                raise ToolNotFoundError(
                    f"Cannot add tool '{name}' to profile '{profile_name}' "
                    f"— tool not registered",
                    details={
                        "tool_name": name,
                        "profile": profile_name,
                        "available_tools": cls.available_tools(),
                    },
                )
            validated_names.add(key)

        cls._profiles[profile_name] = validated_names

        logger.info(
            f"Profile '{profile_name}' defined with tools: "
            f"{sorted(validated_names)}"
        )

    @classmethod
    def grant_tool(cls, profile_name: str, tool_name: str) -> None:
        """
        Thêm 1 tool vào profile đã tồn tại.

        Args:
            profile_name: Tên profile.
            tool_name: Tên tool cần thêm.

        Raises:
            ToolNotFoundError: Tool chưa đăng ký.
        """
        key = tool_name.lower()
        if key not in cls._registry:
            raise ToolNotFoundError(
                f"Cannot grant tool '{tool_name}' — not registered",
                details={"tool_name": tool_name},
            )

        if profile_name not in cls._profiles:
            cls._profiles[profile_name] = set()

        cls._profiles[profile_name].add(key)
        logger.debug(f"Granted tool '{key}' to profile '{profile_name}'")

    @classmethod
    def revoke_tool(cls, profile_name: str, tool_name: str) -> None:
        """
        Xóa 1 tool khỏi profile.

        Args:
            profile_name: Tên profile.
            tool_name: Tên tool cần xóa.
        """
        key = tool_name.lower()
        if profile_name in cls._profiles:
            cls._profiles[profile_name].discard(key)
            logger.debug(f"Revoked tool '{key}' from profile '{profile_name}'")

    @classmethod
    def get_tools_for_profile(cls, profile_name: str) -> list[BaseTool]:
        """
        Lấy danh sách tool instances được phép cho một agent profile.

        Nếu profile không tồn tại, trả về TẤT CẢ tools đã đăng ký
        (behavior mặc định — không giới hạn).

        Args:
            profile_name: Tên agent profile.

        Returns:
            Danh sách BaseTool instances.

        Ví dụ:
            tools = ToolRegistry.get_tools_for_profile("banking_agent")
            for tool in tools:
                print(f"  {tool.name}: {tool.description}")
        """
        if profile_name not in cls._profiles:
            logger.debug(
                f"Profile '{profile_name}' not defined — "
                f"returning all {len(cls._registry)} registered tools"
            )
            return list(cls._registry.values())

        tool_names = cls._profiles[profile_name]
        tools = []
        for name in sorted(tool_names):
            tool = cls._registry.get(name)
            if tool is not None:
                tools.append(tool)
            else:
                logger.warning(
                    f"Tool '{name}' in profile '{profile_name}' "
                    f"but not found in registry — skipping"
                )

        return tools

    @classmethod
    def is_tool_allowed(
        cls,
        profile_name: str,
        tool_name: str,
    ) -> bool:
        """
        Kiểm tra tool có được phép trong profile không.

        Args:
            profile_name: Tên agent profile.
            tool_name: Tên tool cần kiểm tra.

        Returns:
            True nếu được phép (hoặc profile chưa định nghĩa = cho phép tất cả).
        """
        if profile_name not in cls._profiles:
            return True  # Không có profile = không giới hạn

        return tool_name.lower() in cls._profiles[profile_name]

    @classmethod
    def available_profiles(cls) -> list[str]:
        """Trả về danh sách tên tất cả profiles đã định nghĩa."""
        return list(cls._profiles.keys())

    # ─── Function declarations (cho LLM system prompt) ────────────────

    @classmethod
    def get_function_declarations(
        cls,
        profile_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Export function declarations cho tất cả (hoặc một profile) tools.

        Kết quả dùng để nhúng vào system prompt, giúp LLM biết
        cách gọi từng tool với đúng format arguments.

        Args:
            profile_name: Nếu cung cấp, chỉ export tools của profile đó.
                          Nếu None, export tất cả tools.

        Returns:
            Danh sách function declaration dicts.

        Ví dụ:
            declarations = ToolRegistry.get_function_declarations("banking_agent")
            # Output:
            # [
            #     {"name": "faq_search", "description": "...", "parameters": {...}},
            #     {"name": "branch_search", "description": "...", "parameters": {...}},
            # ]
        """
        if profile_name is not None:
            tools = cls.get_tools_for_profile(profile_name)
        else:
            tools = list(cls._registry.values())

        return [tool.to_function_declaration() for tool in tools]

    @classmethod
    def get_tool_descriptions(
        cls,
        profile_name: str | None = None,
    ) -> str:
        """
        Tạo mô tả text cho tất cả tools, dùng cho system prompt.

        Format output dễ đọc cho LLM:
            AVAILABLE TOOLS:
            1. faq_search: Tìm kiếm câu trả lời từ FAQ database.
               Arguments: query (string, required), n_results (integer, default=3)
            2. branch_search: Tìm chi nhánh ngân hàng gần nhất.
               Arguments: latitude (number, required), longitude (number, required)

        Args:
            profile_name: Nếu cung cấp, chỉ mô tả tools của profile đó.

        Returns:
            Chuỗi text mô tả tools.
        """
        if profile_name is not None:
            tools = cls.get_tools_for_profile(profile_name)
        else:
            tools = list(cls._registry.values())

        if not tools:
            return "No tools available."

        lines = ["AVAILABLE TOOLS:"]
        for i, tool in enumerate(tools, 1):
            lines.append(f"{i}. {tool.name}: {tool.description}")

            # Mô tả arguments
            schema = tool.args_schema.model_json_schema()
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))

            if properties:
                args_parts = []
                for prop_name, prop_info in properties.items():
                    prop_type = prop_info.get("type", "any")
                    is_required = prop_name in required
                    default = prop_info.get("default")

                    arg_desc = f"{prop_name} ({prop_type}"
                    if is_required:
                        arg_desc += ", required"
                    elif default is not None:
                        arg_desc += f", default={default}"
                    arg_desc += ")"
                    args_parts.append(arg_desc)

                lines.append(f"   Arguments: {', '.join(args_parts)}")

        return "\n".join(lines)

    # ─── Utility methods ──────────────────────────────────────────────

    @classmethod
    def execute(cls, tool_name: str, **kwargs) -> ToolResult:
        """
        Shortcut: lấy tool và chạy safe_run() trong một lệnh.

        Args:
            tool_name: Tên tool cần chạy.
            **kwargs: Arguments cho tool.

        Returns:
            ToolResult từ tool.safe_run().

        Raises:
            ToolNotFoundError: Tool không tồn tại.

        Ví dụ:
            result = ToolRegistry.execute("faq_search", query="lãi suất")
        """
        tool = cls.get(tool_name)
        return tool.safe_run(**kwargs)

    @classmethod
    def get_info(cls) -> dict[str, Any]:
        """
        Trả về thông tin tổng quan về registry (cho debugging/monitoring).

        Returns:
            Dict chứa số lượng tools, danh sách tools, profiles.
        """
        return {
            "total_tools": len(cls._registry),
            "tools": {
                name: {
                    "class": tool.__class__.__name__,
                    "category": tool.category.value,
                    "version": tool.version,
                    "description": tool.description,
                }
                for name, tool in cls._registry.items()
            },
            "profiles": {
                name: sorted(tools)
                for name, tools in cls._profiles.items()
            },
        }

    @classmethod
    def clear(cls) -> None:
        """Xóa toàn bộ registry và profiles (dùng cho testing)."""
        cls._registry.clear()
        cls._profiles.clear()
        logger.debug("ToolRegistry cleared")


# ═══════════════════════════════════════════════════════════════════════
# ĐĂNG KÝ TOOLS MẶC ĐỊNH
# ═══════════════════════════════════════════════════════════════════════

def _register_default_tools() -> None:
    """
    Đăng ký các tools mặc định vào registry.

    Được gọi lazy — chỉ khi cần mà chưa có tool nào.

    Khi thêm tool mới:
        1. Import tool class
        2. Thêm instance vào danh sách bên dưới
        3. (Tùy chọn) Thêm vào profile phù hợp

    Lưu ý: Dùng try/except cho mỗi tool để 1 tool lỗi
    không ảnh hưởng đến các tools khác.
    """
    if ToolRegistry.available_tools():
        return  # Đã đăng ký rồi, không cần làm lại

    logger.info("Registering default tools...")

    # ── Danh sách tools mặc định ──────────────────────────────────
    # Khi hoàn thiện các tool files (faq_tool.py, branch_tool.py, ...),
    # uncomment các block tương ứng bên dưới.

    # --- FAQ Search Tool ---
    try:
        from tools.faq_tool import FAQTool
        ToolRegistry.register(FAQTool())
    except Exception as e:
        logger.warning(f"Failed to register FAQTool: {e}")

    # --- Branch Search Tool ---
    try:
        from tools.branch_tool import BranchSearchTool
        ToolRegistry.register(BranchSearchTool())
    except Exception as e:
        logger.warning(f"Failed to register BranchSearchTool: {e}")

    # --- Web Search Tool ---
    try:
        from tools.web_search_tool import WebSearchTool
        ToolRegistry.register(WebSearchTool())
    except Exception as e:
        logger.warning(f"Failed to register WebSearchTool: {e}")

    # ── Định nghĩa profiles mặc định ─────────────────────────────
    available = ToolRegistry.available_tools()
    if available:
        ToolRegistry.set_profile("banking_agent", available)
        logger.info(f"Default profile 'banking_agent' set with: {available}")

    registered = ToolRegistry.available_tools()
    logger.info(
        f"Default tool registration complete: "
        f"{len(registered)} tools registered"
    )


def setup_tools() -> None:
    """
    Entry point để khởi tạo Tool System.

    Gọi hàm này trong app startup (main.py) để đăng ký tools
    và thiết lập profiles.

    Ví dụ trong main.py:
        from tools.registry import setup_tools
        setup_tools()
    """
    _register_default_tools()

    # Log summary
    info = ToolRegistry.get_info()
    logger.info(
        f"Tool System ready: "
        f"{info['total_tools']} tools, "
        f"{len(info['profiles'])} profiles"
    )
