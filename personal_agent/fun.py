from tools import web_search_tool
from tools.web_search_tool import WebSearchTool

web_search_tool = WebSearchTool()
kwargs = {
    "query" : "messi đã ghi mấy bàn trong trận argentina với Áo ",
    "extract_content" : "True"
}
res = web_search_tool.run(**kwargs)
print(res.context)