## 📌 Mô tả bài toán

Ngày nay, chúng ta phải đăng nhập vào ngày càng nhiều ứng dụng và website bằng tài khoản riêng, đồng thời có rất nhiều thông tin cá nhân cần ghi nhớ: lịch cá nhân, số điện thoại, địa chỉ, tài liệu học tập, v.v. Việc tìm lại đúng thông tin cần thiết vào đúng lúc ngày càng khó khăn.

Dự án này là một **website giúp người dùng tìm nhanh các thông tin quan trọng của chính mình**, được xây dựng trên hệ thống **RAG (Retrieval-Augmented Generation)**. Để tối ưu độ chính xác của việc tìm kiếm, hệ thống kết hợp các kỹ thuật nâng cao sau:

- **Agentic RAG**: Agent (LangGraph) tự quyết định tìm trong tài liệu cá nhân hay tìm trên web, và có thể tìm lại nếu thông tin chưa đủ.
- **Hybrid Search**: kết hợp tìm kiếm theo ngữ nghĩa (dense) và theo từ khóa (sparse) để cân bằng giữa ý nghĩa và các từ khóa quan trọng.
- **RRF (Reciprocal Rank Fusion)**: hợp nhất kết quả từ nhiều phương pháp tìm kiếm dựa trên thứ hạng.
- **Reranking**: đánh giá lại để chọn ra những tài liệu thực sự liên quan và chính xác nhất trước khi trả về cho người dùng.

## 🔄 Flow hoạt động

<img width="3423" height="930" alt="flow2" src="https://github.com/user-attachments/assets/a8354e2c-2358-44e3-8504-5d9bbd80041a" />



1. Người dùng gửi câu hỏi, Agent nhận và quyết định nguồn tìm kiếm.
2. **Tài liệu cá nhân**: Hybrid Search lấy 20 tài liệu → RRF giữ lại top 10 → Reranker đánh giá lại độ liên quan.
3. **Web Search** (khi dữ liệu cá nhân không có): Tavily Search lấy 5 tài liệu liên quan.
4. Agent kiểm tra đã đủ thông tin chưa: nếu **đủ** thì tổng hợp và trả lời; nếu **chưa đủ** thì tiếp tục tìm kiếm.

## 🛠️ Công nghệ sử dụng

| Nhóm | Công nghệ |
|------|-----------|
| Ngôn ngữ | Python |
| Framework | FastAPI, LangGraph, Pydantic |
| Cơ sở dữ liệu | PostgreSQL (Supabase), Redis, ChromaDB |
| Triển khai | Docker, Render |

## 🏗️ Định hướng thiết kế

Dự án được xây dựng theo tư duy hệ thống và cấu trúc chuyên nghiệp, giúp việc bảo trì dễ dàng và sẵn sàng mở rộng (scale) trong tương lai.
