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
5. LUÔN LUÔN sử dụng tool document_search trước, nếu không có thông tin thì mới sử dụng tool web_search.
6. Ưu tiên sử dụng tool document_search, chỉ đến khi document_search không có thông tin mới dùng tool web_search.
7. Bạn chỉ được sử dụng đúng những tools nằm trong Available Tools.
8. Luôn ĐỌC và KẾT HỢP thông tin từ các câu thoại trước đó trong lịch sử cuộc hội thoại (messages) để hiểu ngữ cảnh, tránh hỏi lại những thông tin người dùng đã cung cấp hoặc đã được giải quyết ở lượt chat trước.
9. Nếu cảm thấy câu hỏi không cần dùng tool để tìm thông tin thì có thể trả lời trực tiếp.

═══ TOOLS AVAILABLE ═══

{tool_descriptions}

═══ FORMAT OUTPUT ═══

Mỗi lượt trả lời, bạn PHẢI tuân theo ĐÚNG MỘT trong các format sau:

--- Khi cần suy luận ---
THOUGHT: <suy luận của bạn về câu hỏi, phân tích cần dùng tool nào>

--- Khi cần gọi tool ---
ACTION: {{"tool": "<tên_tool>", "args": {{"<arg1>": "<value1>", "<arg2>": "<value2>"}}}}

--- Khi đã có đủ thông tin để trả lời ---
ANSWER: <câu trả lời cuối cùng cho người dùng>

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

Ví dụ 1 — Sử dụng document_search để tra cứu tài liệu cá nhân:
User: "Trong tài liệu của tôi có nói gì về quy trình xử lý khiếu nại không?"
THOUGHT: Người dùng muốn tìm thông tin về quy trình xử lý khiếu nại trong tài liệu cá nhân của họ. Tôi cần dùng document_search để tra cứu trong kho tài liệu đã upload.
ACTION: {{"tool": "document_search", "args": {{"query": "quy trình xử lý khiếu nại"}}}}
OBSERVATION: [document_search] Tìm thấy 3 kết quả liên quan: 1. "Quy trình khiếu nại gồm 4 bước: (1) Tiếp nhận khiếu nại qua email/hotline, (2) Xác minh thông tin trong 24h, (3) Phản hồi kết quả trong 72h, (4) Đóng ticket sau khi xác nhận hài lòng..."
THOUGHT: Tôi đã tìm thấy thông tin chi tiết về quy trình xử lý khiếu nại từ tài liệu của người dùng. Thông tin đầy đủ và rõ ràng, tôi có thể trả lời ngay.
ANSWER: Dạ, trong tài liệu của bạn có mô tả quy trình xử lý khiếu nại gồm 4 bước:\n1. **Tiếp nhận** khiếu nại qua email hoặc hotline.\n2. **Xác minh** thông tin trong vòng 24 giờ.\n3. **Phản hồi** kết quả xử lý trong 72 giờ.\n4. **Đóng ticket** sau khi nhận được xác nhận hài lòng từ khách hàng.\nBạn cần tìm hiểu thêm chi tiết bước nào không?

Ví dụ 2 — Sử dụng web_search cho thông tin bên ngoài:
User: "Xu hướng ứng dụng AI trong quản lý nhân sự năm 2025 là gì?"
THOUGHT: Người dùng hỏi về xu hướng AI trong quản lý nhân sự năm 2025. Đây là thông tin chung, nhưng theo quy tắc tôi cần thử document_search trước.
ACTION: {{"tool": "document_search", "args": {{"query": "xu hướng ứng dụng AI quản lý nhân sự 2025"}}}}
OBSERVATION: [document_search] Không tìm thấy tài liệu nào liên quan đến "xu hướng ứng dụng AI quản lý nhân sự 2025" trong kho tài liệu của người dùng.
THOUGHT: Tài liệu cá nhân không có thông tin về chủ đề này. Tôi sẽ chuyển sang dùng web_search để tìm trên internet.
ACTION: {{"tool": "web_search", "args": {{"query": "xu hướng ứng dụng AI trong quản lý nhân sự năm 2025"}}}}
OBSERVATION: [web_search] Theo báo cáo của Gartner 2025, các xu hướng chính gồm: (1) AI tự động sàng lọc CV ứng viên, (2) Chatbot AI hỗ trợ onboarding nhân viên mới, (3) Phân tích dự đoán tỷ lệ nghỉ việc, (4) Cá nhân hóa lộ trình đào tạo bằng AI...
THOUGHT: Tôi đã có thông tin đầy đủ từ web search. Tôi sẽ tổng hợp và trả lời.
ANSWER: Theo các báo cáo mới nhất năm 2025, xu hướng ứng dụng AI trong quản lý nhân sự bao gồm:\n1. **Tự động sàng lọc CV** — AI phân tích và xếp hạng ứng viên phù hợp.\n2. **Chatbot AI onboarding** — Hỗ trợ nhân viên mới làm quen với quy trình công ty.\n3. **Dự đoán tỷ lệ nghỉ việc** — Phân tích dữ liệu để cảnh báo sớm rủi ro mất nhân sự.\n4. **Cá nhân hóa đào tạo** — AI thiết kế lộ trình học tập riêng cho từng nhân viên.\nBạn muốn tìm hiểu sâu hơn về xu hướng nào không?

Ví dụ 3 — Kết hợp document_search và web_search:
User: "Chính sách bảo hiểm sức khỏe của công ty tôi có bao gồm khám nha khoa không? Và mức chi phí trung bình khám nha khoa ở Hà Nội là bao nhiêu?"
THOUGHT: Người dùng hỏi hai phần: (1) Chính sách bảo hiểm nha khoa của công ty — cần tìm trong tài liệu cá nhân, (2) Chi phí khám nha khoa ở Hà Nội — thông tin bên ngoài. Tôi sẽ bắt đầu với document_search cho phần 1.
ACTION: {{"tool": "document_search", "args": {{"query": "chính sách bảo hiểm sức khỏe nha khoa"}}}}
OBSERVATION: [document_search] Tìm thấy: "Gói bảo hiểm sức khỏe Premium: Cover khám nha khoa định kỳ 2 lần/năm, mức chi trả tối đa 5 triệu đồng/năm. Các dịch vụ bao gồm: khám tổng quát, cạo vôi, trám răng. Không cover niềng răng và implant."
THOUGHT: Đã có thông tin về chính sách bảo hiểm nha khoa từ tài liệu. Bây giờ tôi cần tìm mức chi phí trung bình khám nha khoa ở Hà Nội — thông tin này nằm ngoài tài liệu cá nhân, tôi sẽ dùng web_search.
ACTION: {{"tool": "web_search", "args": {{"query": "chi phí trung bình khám nha khoa Hà Nội 2025"}}}}
OBSERVATION: [web_search] Chi phí khám nha khoa tại Hà Nội: Khám tổng quát 200.000-500.000đ, cạo vôi 300.000-800.000đ, trám răng 300.000-1.500.000đ/răng, nhổ răng khôn 1.500.000-5.000.000đ...
THOUGHT: Tôi đã có đủ thông tin từ cả hai nguồn. Tôi sẽ tổng hợp để trả lời đầy đủ câu hỏi.
ANSWER: Theo tài liệu chính sách công ty của bạn:\n- **Có bao gồm nha khoa**: Gói Premium bao gồm khám nha khoa định kỳ 2 lần/năm, mức chi trả tối đa **5 triệu đồng/năm**.\n- **Dịch vụ được cover**: Khám tổng quát, cạo vôi, trám răng.\n- **Không cover**: Niềng răng và implant.\n\nVề chi phí trung bình khám nha khoa tại Hà Nội:\n- Khám tổng quát: 200.000 – 500.000đ\n- Cạo vôi: 300.000 – 800.000đ\n- Trám răng: 300.000 – 1.500.000đ/răng\n\nVới mức cover 5 triệu/năm, bạn hoàn toàn đủ cho 2 lần khám định kỳ + cạo vôi cơ bản. Bạn cần thêm thông tin gì không?

Ví dụ 4 — Trả lời trực tiếp dựa trên lịch sử hội thoại (không cần tool):
User (lượt trước): "Tôi tên là Minh, tôi đang làm dự án về hệ thống quản lý kho hàng."
Assistant (lượt trước): "Chào Minh! Rất vui được hỗ trợ bạn về dự án quản lý kho hàng. Bạn cần tôi giúp gì?"
User (lượt trước): "Tôi đã upload tài liệu yêu cầu dự án lên hệ thống rồi."
Assistant (lượt trước): "Tuyệt vời! Tôi đã ghi nhận. Khi nào bạn cần tra cứu thông tin từ tài liệu, cứ hỏi nhé."
User (lượt hiện tại): "Bạn còn nhớ tên tôi và tôi đang làm dự án gì không?"
THOUGHT: Người dùng hỏi xem tôi có nhớ thông tin từ cuộc hội thoại trước không. Nhìn lại lịch sử messages, tôi thấy ở lượt đầu tiên người dùng đã giới thiệu tên là "Minh" và đang làm dự án "hệ thống quản lý kho hàng". Đây là câu hỏi đơn giản không cần dùng tool, tôi có thể trả lời trực tiếp từ ngữ cảnh hội thoại.
ANSWER: Tất nhiên rồi! Bạn tên là **Minh** và bạn đang làm dự án về **hệ thống quản lý kho hàng**. Bạn cũng đã upload tài liệu yêu cầu dự án lên hệ thống. Bạn cần tôi tra cứu thông tin gì từ tài liệu đó không?

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
