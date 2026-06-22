from tools import web_search_tool
from tools.web_search_tool import WebSearchTool

web_search_tool = WebSearchTool()
kwargs = {
    "query" : "giá vàng ở việt nam hôm nay",
    "region" : "vn-vi",
    "extract_content" : "True"
}

res = web_search_tool.run(**kwargs)
print(res.context)