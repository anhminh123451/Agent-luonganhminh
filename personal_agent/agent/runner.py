"""
AgentRunner cho Banking AI Agent — Agent Core (Module 4).

Module này chứa logic chính call_agent() — node trong LangGraph
đảm nhiệm việc gọi LLM để suy luận (ReAct loop).

Kiến trúc: AgentRunner Pattern
    - AgentRunner class: Quản lý LLM client và thực thi call_agent logic
    - call_agent(): LangGraph node — nhận AgentState, gọi LLM, parse response
    - _parse_llm_response(): Defensive JSON parsing cho ACTION format
    - _build_messages_for_llm(): Chuyển đổi AgentState messages → LLM format

ReAct Loop trong call_agent():
    ┌──────────────────────────────────────────────────────────────┐
    │ 1. Kiểm tra step limit (đã vượt max_steps chưa?)            │
    │ 2. Build system prompt (lần đầu) hoặc lấy từ messages       │
    │ 3. Inject step limit warning (nếu gần hết bước)             │
    │ 4. Gọi LLM với messages history                              │
    │ 5. Parse LLM response → xác định action type                 │
    │    - THOUGHT: Ghi nhận suy luận, tiếp tục loop               │
    │    - ACTION: Parse JSON → tool_name + tool_args               │
    │    - ANSWER: Trích xuất final_answer → kết thúc               │
    │    - HANDOFF: Parse target + reason → chuyển giao             │
    │ 6. Update AgentState với partial state                        │
    └──────────────────────────────────────────────────────────────┘

Cách sử dụng:
    from agent.runner import AgentRunner

    # Tạo runner instance (1 lần khi startup)
    runner = AgentRunner()

    # Sử dụng trong LangGraph node
    def call_agent_node(state: AgentState) -> dict:
        return runner.call_agent(state)

Tham khảo:
    - Plan.md Module 4: Agent Core & Prompt Engineering
    - agent/state.py: AgentState TypedDict
    - agent/prompts.py: build_system_prompt_for_profile()
    - tools/registry.py: ToolRegistry
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

from google import genai

from core.config import settings
from core.exceptions import AgentStepLimitError, LLMResponseError
from core.logger import get_logger

from agent.state import (
    AgentState,
    AgentStatus,
    ActionType,
    MessageRole,
    add_message,
    is_step_limit_reached,
    get_state_summary,
)
from agent.prompts import (
    build_system_prompt_for_profile,
    format_step_limit_warning,
    should_warn_step_limit,
)
from agent.profiles import get_profile

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# LLM CLIENT — Khởi tạo Gemini client
# ═══════════════════════════════════════════════════════════════════════

def _create_llm_client() -> genai.Client:
    """
    Tạo Gemini client sử dụng API key từ settings.

    Returns:
        google.genai.Client instance.

    Raises:
        ConfigurationError: Khi API key không hợp lệ.
    """
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    logger.info("Gemini LLM client initialized successfully")
    return client


# ═══════════════════════════════════════════════════════════════════════
# RESPONSE PARSER — Parse LLM response theo ReAct format
# ═══════════════════════════════════════════════════════════════════════

def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Trích xuất JSON object từ text sử dụng brace counting algorithm.

    Khi LLM trả response có chứa JSON (cho ACTION hoặc HANDOFF),
    cần parse chính xác JSON object ngay cả khi có text xung quanh.

    Thuật toán:
        1. Tìm dấu '{' đầu tiên
        2. Đếm { và } để tìm dấu '}' đóng tương ứng
        3. Trích xuất substring và parse JSON

    Args:
        text: Raw text chứa JSON.

    Returns:
        Parsed dict nếu thành công, None nếu không tìm thấy JSON.
    """
    # Tìm vị trí '{' đầu tiên
    start = text.find("{")
    if start == -1:
        return None

    # Brace counting
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                json_str = text[start:i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Thử regex fallback
                    logger.debug(
                        f"JSON parse failed with brace counting, "
                        f"attempting regex fallback"
                    )
                    return None

    return None


def _parse_llm_response(response_text: str) -> dict[str, Any]:
    """
    Parse LLM response theo ReAct format.

    Xác định action type và trích xuất thông tin tương ứng:
        - THOUGHT: Nội dung suy luận
        - ACTION: {"tool": "...", "args": {...}}
        - ANSWER: Câu trả lời cuối cùng
        - HANDOFF: {"target": "...", "reason": "..."}

    Args:
        response_text: Raw text từ LLM response.

    Returns:
        Dict chứa:
            - action_type: ActionType value
            - content: Nội dung parsed (text hoặc dict)
            - raw_response: Response gốc

    Raises:
        LLMResponseError: Khi response không match bất kỳ format nào.
    """
    text = response_text.strip()

    # ── 1. Kiểm tra ANSWER ────────────────────────────────────────────
    answer_match = re.search(
        r"ANSWER:\s*(.+)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if answer_match:
        answer_content = answer_match.group(1).strip()
        logger.info(f"Parsed ANSWER: '{answer_content[:80]}...'")
        return {
            "action_type": ActionType.ANSWER,
            "content": answer_content,
            "raw_response": text,
        }

    # ── 2. Kiểm tra ACTION ───────────────────────────────────────────
    action_match = re.search(
        r"ACTION:\s*(.+)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if action_match:
        action_text = action_match.group(1).strip()
        parsed_json = _extract_json_from_text(action_text)

        if parsed_json and "tool" in parsed_json:
            tool_name = parsed_json.get("tool", "")
            tool_args = parsed_json.get("args", {})

            logger.info(
                f"Parsed ACTION: tool='{tool_name}', "
                f"args={tool_args}"
            )
            return {
                "action_type": ActionType.ACTION,
                "content": {
                    "tool": tool_name,
                    "args": tool_args if isinstance(tool_args, dict) else {},
                },
                "raw_response": text,
            }

        # ACTION có nhưng JSON parse fail → log warning và thử fallback
        logger.warning(
            f"ACTION found but JSON parse failed: '{action_text[:100]}'"
        )

    # ── 3. Kiểm tra HANDOFF ───────────────────────────────────────────
    handoff_match = re.search(
        r"HANDOFF:\s*(.+)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if handoff_match:
        handoff_text = handoff_match.group(1).strip()
        parsed_json = _extract_json_from_text(handoff_text)

        if parsed_json and "target" in parsed_json:
            logger.info(
                f"Parsed HANDOFF: target='{parsed_json.get('target')}', "
                f"reason='{parsed_json.get('reason', 'N/A')}'"
            )
            return {
                "action_type": ActionType.HANDOFF,
                "content": parsed_json,
                "raw_response": text,
            }

    # ── 4. Kiểm tra THOUGHT ──────────────────────────────────────────
    thought_match = re.search(
        r"THOUGHT:\s*(.+)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if thought_match:
        thought_content = thought_match.group(1).strip()

        # Kiểm tra nếu sau THOUGHT có ACTION (multi-line response)
        # Ví dụ: THOUGHT: ... \n ACTION: {...}
        inner_action_match = re.search(
            r"ACTION:\s*(.+)",
            thought_content,
            re.DOTALL | re.IGNORECASE,
        )
        if inner_action_match:
            # Tách THOUGHT và ACTION
            thought_only = thought_content[:inner_action_match.start()].strip()
            action_text = inner_action_match.group(1).strip()
            parsed_json = _extract_json_from_text(action_text)

            if parsed_json and "tool" in parsed_json:
                logger.info(
                    f"Parsed THOUGHT+ACTION: "
                    f"thought='{thought_only[:50]}...', "
                    f"tool='{parsed_json.get('tool')}'"
                )
                return {
                    "action_type": ActionType.ACTION,
                    "content": {
                        "tool": parsed_json.get("tool", ""),
                        "args": parsed_json.get("args", {}),
                        "thought": thought_only,
                    },
                    "raw_response": text,
                }

        # Kiểm tra nếu sau THOUGHT có ANSWER
        inner_answer_match = re.search(
            r"ANSWER:\s*(.+)",
            thought_content,
            re.DOTALL | re.IGNORECASE,
        )
        if inner_answer_match:
            answer_content = inner_answer_match.group(1).strip()
            logger.info(f"Parsed THOUGHT+ANSWER: '{answer_content[:80]}...'")
            return {
                "action_type": ActionType.ANSWER,
                "content": answer_content,
                "raw_response": text,
            }

        logger.info(f"Parsed THOUGHT: '{thought_content[:80]}...'")
        return {
            "action_type": ActionType.THOUGHT,
            "content": thought_content,
            "raw_response": text,
        }

    # ── 5. Fallback — Không match format nào ──────────────────────────
    # Coi như THOUGHT nếu không match (defensive)
    logger.warning(
        f"LLM response doesn't match any ReAct format, "
        f"treating as THOUGHT: '{text[:100]}...'"
    )
    return {
        "action_type": ActionType.THOUGHT,
        "content": text,
        "raw_response": text,
    }


# ═══════════════════════════════════════════════════════════════════════
# AGENT RUNNER — Class chính quản lý call_agent logic
# ═══════════════════════════════════════════════════════════════════════

class AgentRunner:
    """
    Quản lý việc gọi LLM và xử lý response trong ReAct loop.

    AgentRunner là LangGraph node chính — nhận AgentState,
    gọi LLM (Gemini), parse response, và trả về partial state update.

    Attributes:
        _client: Gemini LLM client.
        _model: Tên model LLM sử dụng.

    Lifecycle:
        1. Khởi tạo 1 lần khi app startup
        2. call_agent() được gọi mỗi bước trong ReAct loop
        3. Client được reuse cho tất cả requests

    Ví dụ:
        runner = AgentRunner()

        # Dùng trong LangGraph node
        graph.add_node("call_agent", runner.call_agent)
    """

    def __init__(
        self,
        model: str = settings.MODEL_LLM,
    ) -> None:
        """
        Khởi tạo AgentRunner.

        Args:
            model: Tên model Gemini sử dụng.
                   Mặc định kiểm tra trong CONFIG.
        """
        self._client = _create_llm_client()
        self._model = model

        logger.info(
            f"AgentRunner initialized | "
            f"model={self._model}"
        )

    # ─── Build messages cho LLM ───────────────────────────────────────

    def _build_llm_contents(
        self,
        state: AgentState,
    ) -> list[dict[str, str]]:
        """
        Chuyển đổi AgentState messages thành format cho Gemini API.

        Gemini API sử dụng format:
            [
                {"role": "user", "parts": [{"text": "..."}]},
                {"role": "model", "parts": [{"text": "..."}]},
            ]

        Mapping từ MessageRole:
            - SYSTEM → prepend vào user message đầu tiên
            - USER → role="user"
            - ASSISTANT → role="model"
            - OBSERVATION → role="user" (với prefix)

        Args:
            state: AgentState hiện tại.

        Returns:
            List of content dicts cho Gemini API.
        """
        contents = []
        system_prompt = None

        for msg in state["messages"]:
            role = msg["role"]
            content = msg["content"]

            if role == MessageRole.SYSTEM.value:
                system_prompt = content
            elif role == MessageRole.USER.value:
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })
            elif role == MessageRole.ASSISTANT.value:
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}],
                })
            elif role == MessageRole.OBSERVATION.value:
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })

        return contents, system_prompt

    # ─── Core call_agent logic ────────────────────────────────────────

    def call_agent(self, state: AgentState) -> dict:
        """
        LangGraph node chính — gọi LLM và parse response.

        Đây là heart of the ReAct loop. Mỗi lần gọi:
        1. Kiểm tra step limit
        2. Build/inject system prompt (lần đầu)
        3. Inject step limit warning (nếu cần)
        4. Gọi LLM
        5. Parse response → xác định action type
        6. Trả partial state update

        Args:
            state: AgentState hiện tại từ LangGraph.

        Returns:
            Dict partial state update cho LangGraph merge.

        State Updates theo action type:
            THOUGHT:
                - current_step += 1
                - current_action = THOUGHT
                - messages += [assistant message]
                - status = RUNNING

            ACTION:
                - current_step += 1
                - current_action = ACTION
                - tool_name = parsed tool name
                - tool_args = parsed tool args
                - messages += [assistant message]
                - status = TOOL_CALLING

            ANSWER:
                - current_step += 1
                - current_action = ANSWER
                - final_answer = parsed answer
                - messages += [assistant message]
                - status = DONE

            HANDOFF:
                - current_step += 1
                - current_action = HANDOFF
                - handoff_target = parsed target
                - handoff_reason = parsed reason
                - messages += [assistant message]
                - status = HANDOFF
        """
        current_step = state["current_step"]
        session_preview = state["session_id"][:8]

        logger.info(
            f"[Step {current_step}] call_agent | "
            f"session={session_preview}..."
        )

        # ── 1. Kiểm tra step limit ───────────────────────────────────
        if is_step_limit_reached(state):
            logger.warning(
                f"Step limit reached: {current_step}/{state['max_steps']} | "
                f"session={session_preview}..."
            )

            # Tạo câu trả lời fallback từ thông tin đã thu thập
            fallback_answer = self._build_step_limit_answer(state)

            return {
                "current_step": current_step + 1,
                "current_action": ActionType.ANSWER.value,
                "final_answer": fallback_answer,
                "status": AgentStatus.DONE.value,
                "error": "Agent exceeded maximum steps",
                "messages": add_message(
                    state,
                    MessageRole.ASSISTANT.value,
                    f"ANSWER: {fallback_answer}",
                ),
            }

        # ── 2. Build system prompt (lần đầu tiên) ────────────────────
        messages = state["messages"]

        if not messages:
            # Lần đầu: inject system prompt + user query
            profile = get_profile(state["agent_profile"])

            system_prompt = build_system_prompt_for_profile(
                profile_name=state["agent_profile"],
                agent_name=profile.agent_name,
            )

            messages = [
                {"role": MessageRole.SYSTEM.value, "content": system_prompt},
                {"role": MessageRole.USER.value, "content": state["query"]},
            ]

            logger.debug(
                f"Initialized messages with system prompt | "
                f"profile={state['agent_profile']}"
            )

        # ── 3. Inject step limit warning (nếu gần hết bước) ──────────
        if should_warn_step_limit(current_step, state["max_steps"]):
            warning = format_step_limit_warning(
                current_step=current_step,
                max_steps=state["max_steps"],
            )
            messages = messages + [
                {"role": MessageRole.SYSTEM.value, "content": warning},
            ]
            logger.debug(
                f"Injected step limit warning at step {current_step}"
            )

        # ── 4. Gọi LLM ──────────────────────────────────────────────
        try:
            contents, system_instruction = self._build_llm_contents(
                {**state, "messages": messages}
            )

            # Gọi Gemini API
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config={
                    "system_instruction": system_instruction,
                    "temperature": 0.3,
                    "max_output_tokens": 2048,
                },
            )

            response_text = response.text

            if not response_text:
                raise LLMResponseError(
                    "LLM returned empty response",
                    details={"step": current_step},
                )

            logger.debug(
                f"LLM response received | "
                f"length={len(response_text)} chars"
            )

        except LLMResponseError:
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return {
                "current_step": current_step + 1,
                "current_action": ActionType.ANSWER.value,
                "final_answer": (
                    "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi của bạn. "
                    "Vui lòng thử lại sau."
                ),
                "status": AgentStatus.ERROR.value,
                "error": f"LLM call failed: {type(e).__name__}: {e}",
                "messages": add_message(
                    state,
                    MessageRole.ASSISTANT.value,
                    "ANSWER: Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi. "
                    "Vui lòng thử lại sau.",
                ),
            }

        # ── 5. Parse LLM response ────────────────────────────────────
        parsed = _parse_llm_response(response_text)
        action_type = parsed["action_type"]
        content = parsed["content"]

        # ── 6. Build partial state update ─────────────────────────────
        new_messages = messages + [
            {
                "role": MessageRole.ASSISTANT.value,
                "content": response_text,
            },
        ]

        # Base update (chung cho mọi action type)
        state_update: dict[str, Any] = {
            "current_step": current_step + 1,
            "current_action": action_type.value,
            "messages": new_messages,
        }

        if action_type == ActionType.THOUGHT:
            state_update["status"] = AgentStatus.RUNNING.value
            logger.info(
                f"[Step {current_step}] THOUGHT → continuing loop"
            )

        elif action_type == ActionType.ACTION:
            state_update["status"] = AgentStatus.TOOL_CALLING.value
            state_update["tool_name"] = content["tool"]
            state_update["tool_args"] = content.get("args", {})
            logger.info(
                f"[Step {current_step}] ACTION → "
                f"tool='{content['tool']}'"
            )

        elif action_type == ActionType.ANSWER:
            state_update["status"] = AgentStatus.DONE.value
            state_update["final_answer"] = content
            logger.info(
                f"[Step {current_step}] ANSWER → done | "
                f"answer='{str(content)[:80]}...'"
            )

        elif action_type == ActionType.HANDOFF:
            state_update["status"] = AgentStatus.HANDOFF.value
            state_update["handoff_target"] = content.get("target", "")
            state_update["handoff_reason"] = content.get("reason", "")
            logger.info(
                f"[Step {current_step}] HANDOFF → "
                f"target='{content.get('target')}'"
            )

        # Log state summary
        summary = get_state_summary(cast(AgentState, {**state, **state_update}))
        logger.debug(f"State after call_agent: {summary}")

        return state_update

    # ─── Helper: Build fallback answer khi vượt step limit ────────────

    def _build_step_limit_answer(self, state: AgentState) -> str:
        """
        Tạo câu trả lời fallback khi agent vượt step limit.

        Tổng hợp thông tin từ tool_observations đã thu thập
        để đưa ra câu trả lời tốt nhất có thể.

        Args:
            state: AgentState hiện tại.

        Returns:
            Câu trả lời fallback.
        """
        observations = state.get("tool_observations", [])

        if observations:
            # Có observations → tổng hợp thông tin
            obs_summary = "\n".join(
                f"- {obs}" for obs in observations[-3:]  # Lấy 3 observations gần nhất
            )
            return (
                f"Dựa trên thông tin tôi đã tìm được:\n\n"
                f"{obs_summary}\n\n"
                f"Lưu ý: Tôi đã đạt giới hạn số bước xử lý. "
                f"Nếu cần thêm thông tin chi tiết, "
                f"vui lòng hỏi lại câu hỏi cụ thể hơn."
            )

        # Không có observations
        return (
            "Xin lỗi, tôi không thể hoàn thành việc xử lý câu hỏi "
            "trong giới hạn cho phép. Vui lòng thử hỏi lại "
            "với câu hỏi cụ thể hơn."
        )
