import json

from tools.document_search_tool import DocumentSearchTool

tool = DocumentSearchTool()
llm_res = '{"query": "DOC-011,3.1 Điều khoản Bảo mật Thông tin NDA (Khoản 1)", "user_id": "1", "n_results": 3, "extra_note": "LLM generated this"}'

# Parse JSON string → dict, rồi truyền vào safe_run() dưới dạng **kwargs
parsed_args = json.loads(llm_res)
res = tool.run(**parsed_args)
print(res.context)