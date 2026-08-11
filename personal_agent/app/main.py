from fastapi import FastAPI
from api.routes import auth,chat,document,health
from databases.database import engine, Base
from tools import setup_tools
from agent.profiles import setup_profiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from core.limit import limiter



setup_tools()
setup_profiles()
# Tạo các bảng trong CSDL (nếu chưa có)
Base.metadata.create_all(bind=engine)

app = FastAPI()

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

