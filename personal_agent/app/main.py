from fastapi import FastAPI
from api.routes import auth,chat,document,health
from databases.database import engine, Base
from tools import setup_tools
from agent.profiles import setup_profiles

setup_tools()
setup_profiles()
# Tạo các bảng trong CSDL (nếu chưa có)
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(
    auth.router,
    prefix="/api"
)

app.include_router(
    chat.router,
    prefix="/api"
)


app.include_router(
    document.router,
    prefix="/api"
)


app.include_router(
    health.router,
    prefix="/api"
)

