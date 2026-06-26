"""
Agent Profiles cho Banking AI Agent — Agent Core (Module 4).

Module này quản lý agent profiles — cấu hình định nghĩa vai trò,
tools được phép sử dụng, và metadata cho từng loại agent.

Kiến trúc: Profile Registry Pattern
    - AgentProfile: Dataclass chứa cấu hình cho một agent
    - PROFILES: Dict mapping profile_name → AgentProfile
    - load_profiles_from_yaml(): Load profiles từ YAML file (mở rộng)
    - register_all_profiles(): Đăng ký profiles vào ToolRegistry

Mối quan hệ với các module khác:
    ┌─────────────────────────────────────────────────────────────┐
    │ profiles.py (định nghĩa cấu hình)                          │
    │   ↓                                                         │
    │ ToolRegistry.set_profile() (đăng ký tools cho profile)      │
    │   ↓                                                         │
    │ prompts.py → build_system_prompt_for_profile()              │
    │   ↓                                                         │
    │ state.py → AgentState["agent_profile"]                      │
    │   ↓                                                         │
    │ runner.py → call_agent() sử dụng profile để lấy prompt     │
    └─────────────────────────────────────────────────────────────┘

Cách sử dụng:
    from agent.profiles import get_profile, register_all_profiles, PROFILES

    # Đăng ký tất cả profiles vào ToolRegistry (gọi 1 lần khi startup)
    register_all_profiles()

    # Lấy thông tin profile
    profile = get_profile("banking_agent")
    print(profile.agent_name)       # "Banking Assistant"
    print(profile.allowed_tools)    # ["faq_search", "branch_search", "web_search"]

    # Liệt kê tất cả profiles
    for name, profile in PROFILES.items():
        print(f"{name}: {profile.description}")

Tham khảo:
    - Plan.md Module 4: Agent Core & Prompt Engineering
    - tools/registry.py: ToolRegistry.set_profile()
    - agent/prompts.py: build_system_prompt_for_profile()
    - agent/state.py: AgentState["agent_profile"]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# AGENT PROFILE — Cấu hình cho một agent
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AgentProfile:
    """
    Cấu hình hoàn chỉnh cho một agent profile.

    Mỗi profile định nghĩa:
        - Agent identity (tên, mô tả, vai trò)
        - Tools được phép sử dụng
        - Cấu hình hành vi (max_steps, temperature, ...)

    frozen=True đảm bảo profile immutable sau khi khởi tạo,
    tránh side-effects khi chia sẻ giữa nhiều requests.

    Attributes:
        name: Tên định danh profile (dùng làm key, ví dụ: "banking_agent").
        agent_name: Tên hiển thị của agent (dùng trong system prompt).
        description: Mô tả ngắn về vai trò agent.
        allowed_tools: Danh sách tên tools agent được phép gọi.
        max_steps: Số bước tối đa cho ReAct loop (override settings nếu set).
        system_prompt_extras: Thông tin bổ sung inject vào system prompt.
        metadata: Thông tin mở rộng tùy ý (tags, version, ...).

    Ví dụ:
        profile = AgentProfile(
            name="banking_agent",
            agent_name="Banking Assistant",
            description="Trợ lý ngân hàng đa năng",
            allowed_tools=["faq_search", "branch_search", "web_search"],
        )
    """

    name: str
    agent_name: str
    description: str
    allowed_tools: list[str] = field(default_factory=list)
    max_steps: int | None = None  # None = dùng settings.MAX_AGENT_STEPS
    system_prompt_extras: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate profile sau khi khởi tạo."""
        if not self.name or not self.name.strip():
            raise ValueError("Profile name không được rỗng.")
        if not self.agent_name or not self.agent_name.strip():
            raise ValueError("Agent name không được rỗng.")


# ═══════════════════════════════════════════════════════════════════════
# DEFAULT PROFILES — Profiles hardcode mặc định
# ═══════════════════════════════════════════════════════════════════════

# ── General Agent — Agent có quyền truy cập tất cả tools ──────────────
PERSONAL_AGENT = AgentProfile(
    name="personal_agent",
    agent_name="AI Assistant",
    description=(
        "Trợ lý AI tổng quát với quyền truy cập tất cả tools. "
        "Phù hợp cho các câu hỏi đa dạng, không giới hạn domain."
    ),
    allowed_tools=["faq_search", "branch_search", "web_search"],
    metadata={"tier": "default", "domain": "all"},
)






# ═══════════════════════════════════════════════════════════════════════
# PROFILE REGISTRY — Mapping profile_name → AgentProfile
# ═══════════════════════════════════════════════════════════════════════

PROFILES: dict[str, AgentProfile] = {
    PERSONAL_AGENT.name: PERSONAL_AGENT,
}

# Profile mặc định khi không chỉ định
DEFAULT_PROFILE_NAME: str = "personal_agent"


# ═══════════════════════════════════════════════════════════════════════
# YAML LOADER — Load profiles từ file YAML (mở rộng)
# ═══════════════════════════════════════════════════════════════════════

def load_profiles_from_yaml(yaml_path: str | Path) -> dict[str, AgentProfile]:
    """
    Load agent profiles từ file YAML.

    Cho phép cấu hình profiles bên ngoài code, dễ thay đổi
    mà không cần sửa source code và redeploy.

    Args:
        yaml_path: Đường dẫn đến file YAML chứa profiles.

    Returns:
        Dict mapping profile_name → AgentProfile.

    Raises:
        FileNotFoundError: File YAML không tồn tại.
        ValueError: YAML format không hợp lệ.

    YAML format mong đợi:
        profiles:
          banking_agent:
            agent_name: "Banking Assistant"
            description: "Trợ lý ngân hàng"
            allowed_tools:
              - faq_search
              - branch_search
              - web_search
            max_steps: 5
            metadata:
              tier: primary

    Ví dụ:
        custom_profiles = load_profiles_from_yaml("config/profiles.yaml")
        PROFILES.update(custom_profiles)
    """
    path = Path(yaml_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Profile YAML file not found: {path.absolute()}"
        )

    try:
        import yaml
    except ImportError:
        logger.warning(
            "PyYAML not installed — cannot load YAML profiles. "
            "Install with: pip install pyyaml"
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {path}: {e}") from e

    if not data or "profiles" not in data:
        logger.warning(f"No 'profiles' key found in {path}")
        return {}

    loaded_profiles: dict[str, AgentProfile] = {}

    for profile_name, config in data["profiles"].items():
        try:
            profile = AgentProfile(
                name=profile_name,
                agent_name=config.get("agent_name", profile_name),
                description=config.get("description", ""),
                allowed_tools=config.get("allowed_tools", []),
                max_steps=config.get("max_steps"),
                system_prompt_extras=config.get("system_prompt_extras", ""),
                metadata=config.get("metadata", {}),
            )
            loaded_profiles[profile_name] = profile

            logger.info(
                f"Loaded profile from YAML: '{profile_name}' "
                f"(tools: {profile.allowed_tools})"
            )

        except (ValueError, TypeError) as e:
            logger.warning(
                f"Skipping invalid profile '{profile_name}' in YAML: {e}"
            )

    logger.info(
        f"Loaded {len(loaded_profiles)} profiles from {path.name}"
    )

    return loaded_profiles


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — Tra cứu và quản lý profiles
# ═══════════════════════════════════════════════════════════════════════

def get_profile(profile_name: str) -> AgentProfile:
    """
    Lấy AgentProfile theo tên.

    Nếu profile không tồn tại, trả về DEFAULT profile và log warning.

    Args:
        profile_name: Tên profile cần lấy.

    Returns:
        AgentProfile instance.

    Ví dụ:
        profile = get_profile("banking_agent")
        print(profile.agent_name)  # "Banking Assistant"

        # Profile không tồn tại → trả default
        profile = get_profile("unknown")  # → PERSONAL_AGENT
    """
    if profile_name in PROFILES:
        return PROFILES[profile_name]

    logger.warning(
        f"Profile '{profile_name}' not found — "
        f"falling back to default '{DEFAULT_PROFILE_NAME}'. "
        f"Available profiles: {list(PROFILES.keys())}"
    )
    return PROFILES[DEFAULT_PROFILE_NAME]


def add_profile(profile: AgentProfile) -> None:
    """
    Thêm hoặc cập nhật một profile vào registry.

    Args:
        profile: AgentProfile instance cần thêm.

    Ví dụ:
        loan_agent = AgentProfile(
            name="loan_agent",
            agent_name="Loan Advisor",
            description="Chuyên gia tư vấn vay vốn",
            allowed_tools=["faq_search", "loan_calculator"],
        )
        add_profile(loan_agent)
    """
    if profile.name in PROFILES:
        logger.warning(
            f"Overwriting existing profile: '{profile.name}'"
        )

    PROFILES[profile.name] = profile

    logger.info(
        f"Added profile: '{profile.name}' "
        f"(agent={profile.agent_name}, "
        f"tools={profile.allowed_tools})"
    )


def remove_profile(profile_name: str) -> None:
    """
    Xóa một profile khỏi registry.

    Args:
        profile_name: Tên profile cần xóa.

    Raises:
        ValueError: Không thể xóa default profile.
        KeyError: Profile không tồn tại.
    """
    if profile_name == DEFAULT_PROFILE_NAME:
        raise ValueError(
            f"Cannot remove default profile '{DEFAULT_PROFILE_NAME}'"
        )

    if profile_name not in PROFILES:
        raise KeyError(
            f"Profile '{profile_name}' not found. "
            f"Available: {list(PROFILES.keys())}"
        )

    del PROFILES[profile_name]
    logger.info(f"Removed profile: '{profile_name}'")


def list_profiles() -> list[dict[str, Any]]:
    """
    Liệt kê tất cả profiles dưới dạng list of dicts.

    Tiện cho API endpoint GET /profiles hoặc debugging.

    Returns:
        Danh sách dict chứa thông tin mỗi profile.

    Ví dụ:
        for p in list_profiles():
            print(f"{p['name']}: {p['description']}")
    """
    return [
        {
            "name": profile.name,
            "agent_name": profile.agent_name,
            "description": profile.description,
            "allowed_tools": profile.allowed_tools,
            "max_steps": profile.max_steps,
            "is_default": profile.name == DEFAULT_PROFILE_NAME,
            "metadata": profile.metadata,
        }
        for profile in PROFILES.values()
    ]


def available_profile_names() -> list[str]:
    """Trả về danh sách tên tất cả profiles đã đăng ký."""
    return list(PROFILES.keys())


# ═══════════════════════════════════════════════════════════════════════
# REGISTRATION — Đăng ký profiles vào ToolRegistry
# ═══════════════════════════════════════════════════════════════════════

def register_all_profiles() -> None:
    """
    Đăng ký tất cả agent profiles vào ToolRegistry.

    Gọi ToolRegistry.set_profile() cho mỗi profile, mapping
    profile_name → allowed_tools.

    Gọi hàm này 1 lần khi app startup, SAU KHI tools đã
    được đăng ký vào ToolRegistry.

    Thứ tự khởi tạo đúng:
        1. setup_tools()            → đăng ký tool instances
        2. register_all_profiles()  → đăng ký profiles (tool permissions)
        3. Agent sẵn sàng nhận requests

    Ví dụ trong main.py:
        from tools.registry import setup_tools
        from agent.profiles import register_all_profiles

        # Startup
        setup_tools()               # Bước 1: đăng ký tools
        register_all_profiles()     # Bước 2: đăng ký profiles
    """
    from tools.registry import ToolRegistry

    registered_tools = ToolRegistry.available_tools()
    if not registered_tools:
        logger.warning(
            "No tools registered in ToolRegistry. "
            "Call setup_tools() before register_all_profiles()."
        )
        return

    success_count = 0
    skip_count = 0

    for profile_name, profile in PROFILES.items():
        # Lọc chỉ những tools đã thực sự đăng ký trong registry
        valid_tools = [
            tool for tool in profile.allowed_tools
            if tool in registered_tools
        ]

        skipped_tools = [
            tool for tool in profile.allowed_tools
            if tool not in registered_tools
        ]

        if skipped_tools:
            logger.warning(
                f"Profile '{profile_name}': skipping unregistered tools: "
                f"{skipped_tools}"
            )
            skip_count += len(skipped_tools)

        if valid_tools:
            ToolRegistry.set_profile(profile_name, valid_tools)
            success_count += 1

            logger.info(
                f"Registered profile '{profile_name}' → "
                f"tools: {valid_tools}"
            )
        else:
            logger.warning(
                f"Profile '{profile_name}' has no valid tools — skipping"
            )

    logger.info(
        f"Profile registration complete: "
        f"{success_count}/{len(PROFILES)} profiles registered"
        + (f", {skip_count} tools skipped" if skip_count else "")
    )


def setup_profiles(yaml_path: str | Path | None = None) -> None:
    """
    Entry point để khởi tạo Profile System.

    Workflow:
        1. (Optional) Load profiles từ YAML file
        2. Đăng ký tất cả profiles vào ToolRegistry

    Args:
        yaml_path: Đường dẫn YAML file (optional).
                   Nếu None, chỉ dùng profiles hardcode.

    Ví dụ trong main.py:
        from tools.registry import setup_tools
        from agent.profiles import setup_profiles

        setup_tools()
        setup_profiles()  # Dùng profiles mặc định

        # Hoặc với YAML:
        setup_profiles(yaml_path="config/profiles.yaml")
    """
    # Bước 1: Load từ YAML nếu có
    if yaml_path is not None:
        try:
            yaml_profiles = load_profiles_from_yaml(yaml_path)
            PROFILES.update(yaml_profiles)
            logger.info(
                f"Merged {len(yaml_profiles)} YAML profiles "
                f"into profile registry"
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to load YAML profiles: {e}")

    # Bước 2: Đăng ký vào ToolRegistry
    register_all_profiles()

    logger.info(
        f"Profile System ready: {len(PROFILES)} profiles "
        f"({available_profile_names()})"
    )
