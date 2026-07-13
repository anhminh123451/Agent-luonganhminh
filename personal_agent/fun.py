from api.schemas import ChatRequest,ChatResponse
from agent.graph import invoke_agent
from tools.registry import setup_tools
from agent.profiles import setup_profiles
from fastapi import FastAPI
setup_tools()
setup_profiles(yaml_path="config/profiles.yaml")

# query = "Thời tiết Hà Nội hôm nay bao nhiêu độ C"
# res = invoke_agent(query=query)
# print(res.get("final_answer"))
app = FastAPI()


@app.post("/chat",response_model=ChatResponse)
def chat(request: ChatRequest):
    result = invoke_agent(query=request.query,max_steps=request.max_steps,session_id=request.session_id)
    final = ChatResponse.from_agent_result(result=result,session_id=request.session_id)
    return final


