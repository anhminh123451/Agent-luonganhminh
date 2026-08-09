from fastapi import FastAPI
from api.routes import auth
from databases.database import engine, Base

# Tạo các bảng trong CSDL (nếu chưa có)
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(
    auth.router,
    prefix="/api"
)