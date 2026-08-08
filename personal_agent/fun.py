import json

from tools.document_search_tool import DocumentSearchTool

tool = DocumentSearchTool()
llm_res = '{"query": " Điều khoản Pháp lý Mẫu và Quy định Bảo mật Dữ liệu", "user_id": "1", "n_results": 3, "extra_note": "LLM generated this"}'

# Parse JSON string → dict, rồi truyền vào safe_run() dưới dạng **kwargs
parsed_args = json.loads(llm_res)
res = tool.run(**parsed_args)
print(res.context)