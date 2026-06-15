"""
Tools Package — Banking AI Agent.

Module 3: Tool Layer
Quản lý tất cả tools mà agent có thể gọi.

Public API:
    - BaseTool: Abstract base class cho tools
    - ToolResult: Data model kết quả tool
    - ToolArgsSchema: Base Pydantic model cho input validation
    - ToolCategory: Enum phân loại tool
    - ToolRegistry: Registry quản lý tools
    - setup_tools(): Entry point khởi tạo tool system
"""

from tools.base import BaseTool, ToolArgsSchema, ToolCategory, ToolResult
from tools.registry import ToolRegistry, setup_tools

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolArgsSchema",
    "ToolCategory",
    "ToolRegistry",
    "setup_tools",
]
