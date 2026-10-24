"""
AgentRunner cho Personal AI Agent — Native Function Calling.

Module này chứa logic chính call_agent() — node trong LangGraph
đảm nhiệm việc gọi DeepSeek LLM với native function calling schemas.
"""

from __future__ import annotations

import json
import time
from typing import Any, cast

from core.config import settings
from core.exceptions import LLMResponseError
from core.logger import get_logger
from agent.llm_provider import DeepSeekProvider

from agent.state import (
    AgentState,
    AgentStatus,
    MessageRole,
    is_step_limit_reached,
    get_state_summary,
)
from agent.prompts import (
    build_system_prompt,
    format_step_limit_warning,
    should_warn_step_limit,
)
from agent.profiles import get_profile
from tools.registry import ToolRegistry

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# LLM CLIENT — Khởi tạo DeepSeek Provider
# ═══════════════════════════════════════════════════════════════════════

def _create_deepseek_provider() -> DeepSeekProvider:
    """Tạo DeepSeekProvider từ settings."""
    provider = DeepSeekProvider(
        api_key=settings.DEEPSEEK_API_KEY,
        model=settings.DEEPSEEK_MODEL,
    )
    logger.info(
        f"DeepSeek Provider initialized | model={provider.model_name}"
    )
    return provider


# ═══════════════════════════════════════════════════════════════════════
# AGENT RUNNER — Quản lý call_agent logic với Native Function Calling
# ═══════════════════════════════════════════════════════════════════════

class AgentRunner:
    """
    Quản lý việc gọi LLM và xử lý response với Native Function Calling.
    """

    def __init__(self) -> None:
        self._llm = _create_deepseek_provider()
        logger.info(
            f"AgentRunner initialized | "
            f"provider={self._llm.provider_name} | "
            f"model={self._llm.model_name}"
        )

    # ─── Build messages cho LLM ───────────────────────────────────────

    def _build_llm_messages(
        self,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        """
        Chuyển đổi AgentState.messages sang format OpenAI API.

        Xử lý 4 loại role:
        - system    → {"role": "system", "content": "..."}
        - user      → {"role": "user", "content": "..."}
        - assistant → {"role": "assistant", "content": "...", "tool_calls": [...]}
        - tool      → {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
        """
        llm_messages = []

        for msg in state.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")

            if role == MessageRole.SYSTEM.value:
                llm_messages.append({
                    "role": "system",
                    "content": content or "",
                })
            elif role == MessageRole.USER.value:
                llm_messages.append({
                    "role": "user",
                    "content": content or "",
                })
            elif role == MessageRole.ASSISTANT.value:
                item: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                }
                if msg.get("tool_calls"):
                    item["tool_calls"] = msg["tool_calls"]
                llm_messages.append(item)
            elif role in (MessageRole.TOOL.value, "observation"):
                llm_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id") or "call_unknown",
                    "name": msg.get("name") or "",
                    "content": content or "",
                })

        return llm_messages

    # ─── Core call_agent logic ────────────────────────────────────────

    def call_agent(self, state: AgentState) -> dict:
        """
        LangGraph node chính — gọi LLM với Native Function Calling.

        Flow:
        1. Kiểm tra step limit → fallback nếu vượt
        2. Build system prompt + user query
        3. Inject step limit warning (nếu gần hết bước)
        4. Lấy tool schemas từ ToolRegistry
        5. Gọi DeepSeek LLM
        6. Kiểm tra response.choices[0].message:
           - Nếu có tool_calls → pending_tool_calls trong state, status=RUNNING
           - Nếu không có tool_calls → final_answer, status=DONE
        """
        current_step = state.get("current_step", 0)
        session_preview = state.get("session_id", "unknown")[:8]

        logger.info(
            f"[Step {current_step}] call_agent | session={session_preview}..."
        )

        # ── 1. Kiểm tra step limit ───────────────────────────────────
        if is_step_limit_reached(state):
            logger.warning(
                f"Step limit reached: {current_step}/{state.get('max_steps')} | "
                f"session={session_preview}..."
            )
            fallback_answer = self._build_step_limit_answer(state)
            new_messages = state.get("messages", []) + [
                {"role": MessageRole.ASSISTANT.value, "content": fallback_answer}
            ]
            return {
                "current_step": current_step + 1,
                "messages": new_messages,
                "pending_tool_calls": [],
                "final_answer": fallback_answer,
                "status": AgentStatus.DONE.value,
                "error": "Agent exceeded maximum steps",
            }

        # ── 2. Xử lý messages và System prompt ──────────────────────
        messages = list(state.get("messages", []))
        query = state.get("query", "")

        if not messages:
            # Cuộc hội thoại mới: inject system prompt + user query
            profile_name = state.get("agent_profile", "personal_agent")
            profile = get_profile(profile_name)
            system_prompt = build_system_prompt(agent_name=profile.agent_name)

            messages = [
                {"role": MessageRole.SYSTEM.value, "content": system_prompt},
                {"role": MessageRole.USER.value, "content": query},
            ]
            logger.debug(
                f"Initialized messages with system prompt | profile={profile_name}"
            )
        else:
            # Cuộc hội thoại tiếp tục (từ checkpoint):
            # Kiểm tra xem user query mới đã được thêm vào cuối chưa
            last_user_msg = None
            for msg in reversed(messages):
                if msg.get("role") == MessageRole.USER.value:
                    last_user_msg = msg.get("content")
                    break

            if last_user_msg != query:
                messages.append({
                    "role": MessageRole.USER.value,
                    "content": query,
                })
                logger.debug(f"Continued conversation — appended new user query | query='{query[:50]}...'")

        # ── 3. Inject step limit warning (nếu gần hết bước) ──────────
        max_steps = state.get("max_steps", settings.MAX_AGENT_STEPS)
        if should_warn_step_limit(current_step, max_steps):
            warning = format_step_limit_warning(
                current_step=current_step,
                max_steps=max_steps,
            )
            messages.append({
                "role": MessageRole.SYSTEM.value,
                "content": warning,
            })
            logger.debug(f"Injected step limit warning at step {current_step}")

        # ── 4. Lấy tool schemas từ ToolRegistry ──────────────────────
        profile_name = state.get("agent_profile", "personal_agent")
        tool_schemas = ToolRegistry.get_openai_tools_schema(profile_name)

        # ── 5. Gọi DeepSeek LLM (với retry khi fail) ────────────────
        llm_messages = self._build_llm_messages({**state, "messages": messages})
        max_retries = 3
        retry_delay = 1
        response = None
        user_id = state.get("user_id")

        for attempt in range(1, max_retries + 1):
            try:
                response = self._llm.generate(
                    messages=llm_messages,
                    tools=tool_schemas if tool_schemas else None,
                    tool_choice="auto" if tool_schemas else None,
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS_OUTPUT,
                    user_id=user_id,
                )
                if response and response.choices:
                    break
            except Exception as e:
                logger.warning(
                    f"LLM call attempt {attempt}/{max_retries} failed: {e}"
                )
                if attempt == max_retries:
                    logger.error("LLM call completely failed after retries", exc_info=True)
                    return {
                        "current_step": current_step + 1,
                        "messages": messages + [{
                            "role": MessageRole.ASSISTANT.value,
                            "content": "Xin lỗi, tôi gặp sự cố khi kết nối hệ thống AI. Vui lòng thử lại sau.",
                        }],
                        "pending_tool_calls": [],
                        "final_answer": "Xin lỗi, tôi gặp sự cố khi kết nối hệ thống AI. Vui lòng thử lại sau.",
                        "status": AgentStatus.ERROR.value,
                        "error": f"LLM call failed: {e}",
                    }
                time.sleep(retry_delay)
                retry_delay *= 2

        if not response or not response.choices:
            raise LLMResponseError("LLM returned empty or invalid response")

        # ── 6. Phân tích response từ LLM ────────────────────────────
        choice = response.choices[0]
        message_obj = choice.message
        content = message_obj.content or ""
        tool_calls = getattr(message_obj, "tool_calls", None)

        if tool_calls and len(tool_calls) > 0:
            # ── Case A: LLM muốn gọi tool ────────────────────────────
            pending = []
            formatted_tool_calls = []

            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args_raw = tc.function.arguments or "{}"
                try:
                    fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                except Exception as e:
                    logger.warning(f"Failed to parse tool arguments for {fn_name}: {e}")
                    fn_args = {}

                pending.append({
                    "id": tc.id,
                    "name": fn_name,
                    "args": fn_args,
                })

                formatted_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args_raw if isinstance(fn_args_raw, str) else json.dumps(fn_args_raw),
                    },
                })

            assistant_msg = {
                "role": MessageRole.ASSISTANT.value,
                "content": content if content else None,
                "tool_calls": formatted_tool_calls,
            }

            logger.info(
                f"[Step {current_step}] LLM requested {len(pending)} tool calls: "
                f"{[p['name'] for p in pending]}"
            )

            state_update: dict[str, Any] = {
                "current_step": current_step + 1,
                "messages": messages + [assistant_msg],
                "pending_tool_calls": pending,
                "status": AgentStatus.RUNNING.value,
            }

        else:
            # ── Case B: LLM trả lời trực tiếp (hoàn thành) ───────────
            assistant_msg = {
                "role": MessageRole.ASSISTANT.value,
                "content": content,
            }

            logger.info(
                f"[Step {current_step}] LLM finished with direct answer | "
                f"length={len(content)} chars"
            )

            state_update = {
                "current_step": current_step + 1,
                "messages": messages + [assistant_msg],
                "pending_tool_calls": [],
                "final_answer": content,
                "status": AgentStatus.DONE.value,
            }

        summary = get_state_summary(cast(AgentState, {**state, **state_update}))
        logger.debug(f"State after call_agent: {summary}")

        return state_update

    # ─── Helper: Build fallback answer khi vượt step limit ────────────

    def _build_step_limit_answer(self, state: AgentState) -> str:
        """Tạo câu trả lời fallback khi agent vượt step limit."""
        tool_results = [
            msg.get("content", "")
            for msg in state.get("messages", [])
            if msg.get("role") in (MessageRole.TOOL.value, "observation") and msg.get("content")
        ]

        if tool_results:
            obs_summary = "\n".join(
                f"- {r[:200]}" for r in tool_results[-3:]
            )
            return (
                f"Dựa trên các thông tin tôi đã tìm được:\n\n"
                f"{obs_summary}\n\n"
                f"Lưu ý: Tôi đã đạt giới hạn số bước xử lý cho câu hỏi này. "
                f"Nếu bạn cần thêm thông tin, vui lòng hỏi chi tiết hơn."
            )

        return (
            "Xin lỗi, tôi không thể hoàn thành việc xử lý câu hỏi "
            "trong giới hạn số bước cho phép. Vui lòng thử hỏi lại "
            "với câu hỏi cụ thể hơn."
        )
