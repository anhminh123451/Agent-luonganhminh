"""
Prompt Templates cho Personal AI Agent — Native Function Calling.

Module này quản lý system prompt và các templates cảnh báo cho agent.
Khi chuyển sang Native Function Calling, system prompt không còn cần
chứa tool descriptions hoặc hướng dẫn format THOUGHT/ACTION/ANSWER
vì tools được truyền trực tiếp qua tham số tools của LLM API.
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from core.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — Chỉ chứa rules và persona
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["agent_name"],
    template="""\
Bạn là {agent_name}, một trợ lý AI thông minh, tận tâm và chuyên nghiệp.

═══ RULES ═══
1. Trả lời bằng TIẾNG VIỆT, lịch sự, ngắn gọn và rõ ràng.
2. KHÔNG BAO GIỜ bịa đặt thông tin — nếu không tìm thấy dữ liệu hoặc không biết, hãy nói rõ.
3. Luôn ĐỌC và KẾT HỢP thông tin từ lịch sử hội thoại trước đó để hiểu đầy đủ ngữ cảnh của người dùng.
4. Khi cần tra cứu tài liệu cá nhân của người dùng, hãy ưu tiên sử dụng tool document_search trước. Nếu không có kết quả hoặc thông tin nằm ngoài phạm vi tài liệu, hãy sử dụng tool web_search.
5. Khi người dùng yêu cầu các thao tác liên quan tới email Gmail (đọc email, soạn thư nháp), hãy sử dụng các tool Gmail tương ứng.
6. Nếu câu hỏi đơn giản (như chào hỏi, trò chuyện thông thường, hoặc thông tin đã có sẵn trong lịch sử trò chuyện), hãy trả lời trực tiếp mà không cần gọi tool.
7. Khi gọi tool, hãy truyền các tham số chính xác và đầy đủ theo mô tả của từng tool.
""",
)


# ═══════════════════════════════════════════════════════════════════════
# STEP LIMIT WARNING — Cảnh báo khi gần hết bước
# ═══════════════════════════════════════════════════════════════════════

STEP_LIMIT_WARNING_TEMPLATE = PromptTemplate(
    input_variables=["current_step", "max_steps", "remaining_steps"],
    template="""\
⚠️ CẢNH BÁO: Bạn đang ở bước {current_step}/{max_steps} (còn {remaining_steps} bước).
Hãy tổng hợp thông tin đã có và đưa ra câu trả lời cuối cùng cho người dùng ngay nếu có thể.
Nếu chưa có đủ dữ liệu, hãy trả lời tốt nhất dựa trên các thông tin hiện tại.""",
)


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def build_system_prompt(agent_name: str = "AI Assistant") -> str:
    """
    Build system prompt cho agent (không cần tool descriptions).
    """
    prompt = SYSTEM_PROMPT_TEMPLATE.invoke({
        "agent_name": agent_name,
    }).text

    logger.debug(
        f"Built system prompt | agent={agent_name} | length={len(prompt)} chars"
    )
    return prompt


def format_step_limit_warning(
    current_step: int,
    max_steps: int,
) -> str:
    """Tạo cảnh báo khi agent gần hết ngân sách bước cho phép."""
    remaining = max(0, max_steps - current_step)
    return STEP_LIMIT_WARNING_TEMPLATE.invoke({
        "current_step": str(current_step),
        "max_steps": str(max_steps),
        "remaining_steps": str(remaining),
    }).text


def should_warn_step_limit(current_step: int, max_steps: int) -> bool:
    """Kiểm tra có nên gửi cảnh báo step limit (còn 2 bước hoặc ít hơn)."""
    remaining = max_steps - current_step
    return 0 < remaining <= 2
