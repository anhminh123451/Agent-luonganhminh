"""
Prompt Templates cho Banking AI Agent — Agent Core (Module 4).

Module này quản lý toàn bộ prompt templates cho agent,
dùng LangChain PromptTemplate để tạo prompt có type-safe,
dễ maintain, và dễ test.

Kiến trúc: Template Pattern với LangChain PromptTemplate
    - SYSTEM_PROMPT_TEMPLATE: System prompt chính cho ReAct agent
    - OBSERVATION_TEMPLATE: Format observation trả về từ tool
    - STEP_LIMIT_WARNING_TEMPLATE: Cảnh báo khi gần hết bước
    - HANDOFF_TEMPLATE: Template cho HANDOFF action (multi-agent)
    - Các helper function build prompt hoàn chỉnh

ReAct Loop Prompt Flow:
    ┌──────────────────────────────────────────────────────────┐
    │ 1. SYSTEM PROMPT (1 lần duy nhất khi khởi tạo)          │
    │    - Vai trò agent + quy tắc ứng xử                     │
    │    - Danh sách AVAILABLE TOOLS                           │
    │    - Output format (THOUGHT / ACTION / ANSWER / HANDOFF) │
    │    - Few-shot examples                                   │
    ├──────────────────────────────────────────────────────────┤
    │ 2. USER MESSAGE (câu hỏi gốc từ user)                   │
    ├──────────────────────────────────────────────────────────┤
    │ 3. ASSISTANT → THOUGHT → ACTION (LLM quyết định)        │
    ├──────────────────────────────────────────────────────────┤
    │ 4. OBSERVATION (kết quả từ tool, format bởi template)    │
    ├──────────────────────────────────────────────────────────┤
    │ 5. Lặp lại 3-4 cho đến khi ANSWER hoặc MAX_STEPS        │
    └──────────────────────────────────────────────────────────┘

Cách sử dụng:
    from agent.prompts import build_system_prompt, format_observation

    # Build system prompt cho agent
    system_prompt = build_system_prompt(
        tool_descriptions="1. faq_search: ...",
        agent_name="Banking Assistant",
    )

    # Format observation từ tool result
    obs_message = format_observation(
        tool_name="faq_search",
        observation="Lãi suất tiết kiệm 12 tháng là 5.5%/năm.",
    )

Tham khảo:
    - Plan.md Module 4: Agent Core & Prompt Engineering
    - LangChain PromptTemplate documentation
    - ReAct pattern (Yao et al., 2022)
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — Prompt chính cho ReAct agent
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["agent_name", "tool_descriptions"],
    template="""\
Bạn là {agent_name}, một trợ lý AI thông minh chuyên hỗ trợ người dùng và bạn có thể duy trì một cuộc hội thoại.  

═══ RULES ═══

1. Bạn trả lời bằng TIẾNG VIỆT, lịch sự và chuyên nghiệp.
2. Khi không chắc chắn, hãy sử dụng tools để tra cứu thông tin chính xác.
3. KHÔNG BAO GIỜ bịa thông tin — nếu không tìm thấy, hãy nói rõ.
4. Trả lời ngắn gọn, rõ ràng, đúng trọng tâm câu hỏi.
5. Nếu câu hỏi nằm ngoài phạm vi ngân hàng, hãy dùng tool web_search.
6. Bạn chỉ được sử dụng đúng những tools nằm trong Available Tools.
7. Luôn ĐỌC và KẾT HỢP thông tin từ các câu thoại trước đó trong lịch sử cuộc hội thoại (messages) để hiểu ngữ cảnh, tránh hỏi lại những thông tin người dùng đã cung cấp hoặc đã được giải quyết ở lượt chat trước.
8. Nếu cảm thấy câu hỏi không cần dùng tool để tìm thông tin thì có thể trả lời trực tiếp.

═══ TOOLS AVAILABLE ═══

{tool_descriptions}

═══ FORMAT OUTPUT ═══

Mỗi lượt trả lời, bạn PHẢI tuân theo ĐÚNG MỘT trong các format sau:

--- Khi cần suy luận ---
THOUGHT: <suy luận của bạn về câu hỏi, phân tích cần dùng tool nào>

--- Khi cần gọi tool ---
ACTION: {{"tool": "<tên_tool>", "args": {{"<arg1>": "<value1>", "<arg2>": "<value2>"}}}}

--- Khi đã có đủ thông tin để trả lời ---
ANSWER: <câu trả lời cuối cùng cho khách hàng>

--- Khi cần chuyển cho agent khác (multi-agent) ---
HANDOFF: {{"target": "<tên_agent_đích>", "reason": "<lý do chuyển>"}}

═══ QUY TRÌNH SUY LUẬN (ReAct) ═══

Bước 1: THOUGHT — Phân tích câu hỏi mới nhất kết hợp với lịch sử cuộc hội thoại phía trên để xác định khách hàng đang muốn gì, thông tin nào ĐÃ CÓ trong lịch sử, và thông tin nào CẦN LẤY THÊM.
Bước 2: ACTION — Gọi tool phù hợp để lấy thông tin
Bước 3: (Nhận OBSERVATION từ tool)
Bước 4: THOUGHT — Phân tích kết quả, quyết định đã đủ thông tin chưa
Bước 5: ANSWER hoặc ACTION tiếp (nếu cần thêm thông tin)

═══ IMPORTANT RULES ═══

- Mỗi lượt CHỈ ĐƯỢC trả về MỘT action (THOUGHT, ACTION, ANSWER, hoặc HANDOFF)
- ACTION phải là JSON hợp lệ với key "tool" và "args"
- Khi nhận OBSERVATION, hãy phân tích kết quả trước khi trả lời
- Nếu tool trả về lỗi, hãy thử cách khác hoặc thông báo cho khách hàng
- KHÔNG được gọi cùng 1 tool với cùng arguments quá 2 lần

═══ EXAMPLES ═══

Ví dụ 1 — Câu hỏi FAQ:
User: "Lãi suất tiết kiệm 12 tháng là bao nhiêu?"
THOUGHT: Khách hàng hỏi về lãi suất tiết kiệm. Tôi cần tra cứu trong FAQ database.
ACTION: {{"tool": "faq_search", "args": {{"query": "lãi suất tiết kiệm 12 tháng"}}}}
OBSERVATION: [faq_search] Lãi suất tiết kiệm kỳ hạn 12 tháng là 5.5%/năm...
ANSWER: Lãi suất tiết kiệm kỳ hạn 12 tháng hiện tại là 5.5%/năm. Quý khách có cần thêm thông tin về các kỳ hạn khác không?

Ví dụ 2 — Tìm chi nhánh: 
User: "Chi nhánh ngân hàng nào gần Hà Nội?"
THOUGHT: Khách hàng cần tìm chi nhánh gần vị trí. Tôi cần dùng branch_search với vị trí Hà Nội.
ACTION: {{"tool": "branch_search", "args": {{"location": "Hà Nội"}}}}
OBSERVATION: [branch_search] Chi nhánh gần nhất: 1. CN Hoàn Kiếm (0.5km)...
ANSWER: Chi nhánh gần khu vực Hà Nội nhất là CN Hoàn Kiếm, cách khoảng 0.5km. Quý khách có muốn biết thêm thông tin liên hệ không?

Ví dụ 3 — Câu hỏi ngoài phạm vi ngân hàng (dùng web_search):
User: "Thời tiết Hà Nội hôm nay thế nào?"
THOUGHT: Đây là câu hỏi về thời tiết, nằm ngoài phạm vi ngân hàng. Tôi sẽ sử dụng web_search để tìm thông tin trên internet.
ACTION: {{"tool": "web_search", "args": {{"query": "thời tiết Hà Nội hôm nay"}}}}
OBSERVATION: [web_search] Thời tiết Hà Nội hôm nay: 32°C, trời nắng, độ ẩm 65%, có mưa rào vào chiều tối...
THOUGHT: Tôi đã có thông tin thời tiết từ web search. Tôi sẽ tổng hợp và trả lời cho người dùng.
ANSWER: Thời tiết Hà Nội hôm nay khoảng 32°C, trời nắng với độ ẩm 65%. Có khả năng mưa rào vào chiều tối, bạn nên mang theo ô nhé!

Ví dụ 4 — Sử dụng thông tin từ lượt thoại trước (Hội thoại nhiều lượt):
User: "Chi nhánh nào gần Cầu Giấy nhất?"
THOUGHT: Khách hàng muốn tìm chi nhánh gần khu vực Cầu Giấy. Tôi sẽ dùng tool branch_search.
ACTION: {{"tool": "branch_search", "args": {{"location": "Cầu Giấy"}}}}
OBSERVATION: [branch_search] Chi nhánh gần nhất: 1. CN Cầu Giấy (địa chỉ 123 Cầu Giấy, cách 0.2km).
ANSWER: Dạ, chi nhánh gần khu vực Cầu Giấy nhất là CN Cầu Giấy tại số 123 Cầu Giấy, cách bạn khoảng 0.2km ạ.
User: "Thế chi nhánh đó hôm nay có mở cửa không?"
THOUGHT: Khách hàng hỏi "chi nhánh đó". Dựa vào câu trả lời (ANSWER) ở lượt thoại ngay trước, "chi nhánh đó" chính là "CN Cầu Giấy". Bây giờ tôi cần tra cứu lịch làm việc của CN Cầu Giấy bằng faq_search.
ACTION: {{"tool": "faq_search", "args": {{"query": "giờ mở cửa chi nhánh Cầu Giấy"}}}}
OBSERVATION: [faq_search] CN Cầu Giấy mở cửa từ Thứ 2 đến Thứ 6 (8h00 - 17h00), Thứ 7 và Chủ Nhật đóng cửa.
THOUGHT: Hôm nay là Chủ Nhật (dựa theo thời gian hệ thống), tool báo chi nhánh đóng cửa vào Chủ Nhật. Tôi sẽ trả lời khách hàng.
ANSWER: Dạ, CN Cầu Giấy hiện tại đóng cửa vào ngày Thứ 7 và Chủ Nhật ạ. Hôm nay là Chủ Nhật nên chi nhánh không làm việc, quý khách có cần em hỗ trợ tìm kiếm dịch vụ trực tuyến nào khác không?
""",
)


# ═══════════════════════════════════════════════════════════════════════
# OBSERVATION TEMPLATE — Format kết quả từ tool
# ═══════════════════════════════════════════════════════════════════════

OBSERVATION_TEMPLATE = PromptTemplate(
    input_variables=["tool_name", "observation"],
    template="""\
OBSERVATION: [{tool_name}] {observation}""",
)


# ═══════════════════════════════════════════════════════════════════════
# STEP LIMIT WARNING — Cảnh báo khi gần hết bước
# ═══════════════════════════════════════════════════════════════════════

STEP_LIMIT_WARNING_TEMPLATE = PromptTemplate(
    input_variables=["current_step", "max_steps", "remaining_steps"],
    template="""\
⚠️ CẢNH BÁO: Bạn đang ở bước {current_step}/{max_steps} (còn {remaining_steps} bước).
Hãy tổng hợp thông tin đã có và đưa ra ANSWER ngay nếu có thể.
Nếu chưa có đủ thông tin, hãy đưa ra câu trả lời tốt nhất với dữ liệu hiện tại.""",
)


# ═══════════════════════════════════════════════════════════════════════
# HANDOFF TEMPLATE — Template cho multi-agent handoff
# ═══════════════════════════════════════════════════════════════════════

HANDOFF_TEMPLATE = PromptTemplate(
    input_variables=["source_agent", "target_agent", "reason", "query"],
    template="""\
[HANDOFF] Agent "{source_agent}" chuyển giao cho "{target_agent}".
Lý do: {reason}
Câu hỏi gốc: {query}

Hãy tiếp tục xử lý câu hỏi trên với chuyên môn của bạn.""",
)


# ═══════════════════════════════════════════════════════════════════════
# ERROR RECOVERY TEMPLATE — Khi tool gặp lỗi
# ═══════════════════════════════════════════════════════════════════════

ERROR_RECOVERY_TEMPLATE = PromptTemplate(
    input_variables=["tool_name", "error_message"],
    template="""\
OBSERVATION: [{tool_name}] ❌ LỖI: {error_message}
Hãy thử cách tiếp cận khác hoặc dùng tool khác để trả lời câu hỏi.""",
)


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — Build prompt hoàn chỉnh
# ═══════════════════════════════════════════════════════════════════════

def build_system_prompt(
    tool_descriptions: str,
    agent_name: str = "AI Assistant",
) -> str:
    """
    Build system prompt hoàn chỉnh cho agent.

    Kết hợp SYSTEM_PROMPT_TEMPLATE với tool descriptions thực tế
    từ ToolRegistry để tạo system prompt đầy đủ.

    Args:
        tool_descriptions: Chuỗi mô tả tools từ ToolRegistry.get_tool_descriptions().
        agent_name: Tên hiển thị của agent (mặc định "Banking Assistant").

    Returns:
        System prompt hoàn chỉnh (string).

    Ví dụ:
        from tools.registry import ToolRegistry

        tool_desc = ToolRegistry.get_tool_descriptions("banking_agent")
        system_prompt = build_system_prompt(tool_desc)
    """
    prompt = SYSTEM_PROMPT_TEMPLATE.invoke({
        "agent_name": agent_name,
        "tool_descriptions": tool_descriptions,
    }).text

    logger.debug(
        f"Built system prompt | agent={agent_name} | "
        f"length={len(prompt)} chars"
    )

    return prompt


def format_observation(
    tool_name: str,
    observation: str,
) -> str:
    """
    Format observation từ tool result thành message cho agent.

    Dùng sau khi tool execution, trả về string để append
    vào messages list trong AgentState.

    Args:
        tool_name: Tên tool đã trả kết quả.
        observation: Nội dung observation (từ ToolResult.to_observation()).

    Returns:
        Chuỗi observation đã format.

    Ví dụ:
        result = tool.safe_run(query="lãi suất")
        obs = format_observation("faq_search", result.context)
        # → "OBSERVATION: [faq_search] Lãi suất tiết kiệm..."
    """
    return OBSERVATION_TEMPLATE.invoke({
        "tool_name": tool_name,
        "observation": observation,
    }).text


def format_step_limit_warning(
    current_step: int,
    max_steps: int,
) -> str:
    """
    Tạo cảnh báo khi agent gần hết bước cho phép.

    Inject vào messages khi current_step >= max_steps - 2
    để nhắc agent tổng hợp và trả lời.

    Args:
        current_step: Bước hiện tại trong ReAct loop.
        max_steps: Số bước tối đa cho phép.

    Returns:
        Chuỗi cảnh báo step limit.

    Ví dụ:
        if state["current_step"] >= state["max_steps"] - 2:
            warning = format_step_limit_warning(
                current_step=state["current_step"],
                max_steps=state["max_steps"],
            )
            # Inject warning vào messages
    """
    remaining = max(0, max_steps - current_step)

    return STEP_LIMIT_WARNING_TEMPLATE.invoke({
        "current_step": str(current_step),
        "max_steps": str(max_steps),
        "remaining_steps": str(remaining),
    }).text


def format_handoff_message(
    source_agent: str,
    target_agent: str,
    reason: str,
    query: str,
) -> str:
    """
    Tạo message handoff khi chuyển giao giữa các agent.

    Dùng trong multi-agent architecture (Cấp 2 trong Plan.md)
    khi agent hiện tại quyết định HANDOFF cho agent khác.

    Args:
        source_agent: Tên agent hiện tại (đang chuyển giao).
        target_agent: Tên agent đích (nhận chuyển giao).
        reason: Lý do chuyển giao.
        query: Câu hỏi gốc từ user.

    Returns:
        Chuỗi handoff message.

    Ví dụ:
        msg = format_handoff_message(
            source_agent="banking_agent",
            target_agent="loan_agent",
            reason="Câu hỏi về vay vốn cần chuyên gia.",
            query="Tôi muốn vay mua nhà 2 tỷ?",
        )
    """
    return HANDOFF_TEMPLATE.invoke({
        "source_agent": source_agent,
        "target_agent": target_agent,
        "reason": reason,
        "query": query,
    }).text


def format_error_recovery(
    tool_name: str,
    error_message: str,
) -> str:
    """
    Format error message khi tool gặp lỗi.

    Giúp agent biết tool đã fail và cần thử cách khác.

    Args:
        tool_name: Tên tool bị lỗi.
        error_message: Mô tả lỗi chi tiết.

    Returns:
        Chuỗi error observation.

    Ví dụ:
        if not result.success:
            error_obs = format_error_recovery("faq_search", result.error)
    """
    return ERROR_RECOVERY_TEMPLATE.invoke({
        "tool_name": tool_name,
        "error_message": error_message,
    }).text


def build_system_prompt_for_profile(
    profile_name: str, 
    agent_name: str = "AI Assistant",
) -> str:
    """
    Build system prompt từ agent profile (convenience function).

    Tự động lấy tool descriptions từ ToolRegistry
    theo profile_name, rồi build system prompt.

    Args:
        profile_name: Tên profile trong ToolRegistry (ví dụ: "banking_agent").
        agent_name: Tên hiển thị của agent.

    Returns:
        System prompt hoàn chỉnh.

    Raises:
        ImportError: Nếu ToolRegistry chưa sẵn sàng.

    Ví dụ:
        # Trong call_agent node:
        system_prompt = build_system_prompt_for_profile(
            profile_name=state["agent_profile"],
            agent_name="AI Assistant",
        )
    """
    from tools.registry import ToolRegistry

    tool_descriptions = ToolRegistry.get_tool_descriptions(profile_name)

    logger.info(
        f"Building system prompt for profile '{profile_name}' | "
        f"agent={agent_name}"
    )

    return build_system_prompt(
        tool_descriptions=tool_descriptions,
        agent_name=agent_name,
    )


def should_warn_step_limit(current_step: int, max_steps: int) -> bool:
    """
    Kiểm tra có nên gửi cảnh báo step limit cho agent không.

    Cảnh báo khi agent còn 2 bước hoặc ít hơn.

    Args:
        current_step: Bước hiện tại.
        max_steps: Số bước tối đa.

    Returns:
        True nếu nên cảnh báo.
    """
    remaining = max_steps - current_step
    return 0 < remaining <= 2
