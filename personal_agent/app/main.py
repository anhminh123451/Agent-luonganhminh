from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.routes import auth, chat, document, health, sessions
from databases.database import engine, Base
from tools import setup_tools
from agent.profiles import setup_profiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from core.limit import limiter
import os



setup_tools()
setup_profiles()
# Tạo các bảng trong CSDL (nếu chưa có)
Base.metadata.create_all(bind=engine)

app = FastAPI()

# ── CORS Middleware — cho phép frontend gọi API cross-origin ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: thay bằng domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter=limiter
app.add_exception_handler(RateLimitExceeded,_rate_limit_exceeded_handler)


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

app.include_router(
    sessions.router,
    prefix="/api"
)

# ── Static Files — serve frontend SPA ─────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

