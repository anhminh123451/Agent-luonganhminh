from api.schemas import ChatRequest,ChatResponse
from agent.graph import invoke_agent
from tools.registry import setup_tools
from agent.profiles import setup_profiles
from fastapi import FastAPI
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router


app = FastAPI()
setup_tools()
setup_profiles(yaml_path="config/profiles.yaml")
app.include_router(
    chat_router
)
app.include_router(
    health_router
)



