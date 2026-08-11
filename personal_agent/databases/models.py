from google.auth import default
from databases.database import Base
from sqlalchemy import Column, Integer, String, DateTime, Boolean,func,ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column,relationship
from datetime import datetime,timezone
from typing import Optional


class Users(Base):
    __tablename__="users"
    user_id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50))
    hashed_password: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    role:Mapped[str] = mapped_column(String(50),default="user",server_default="user")

    documents: Mapped[list["Documents"]]=relationship(back_populates="user",cascade="all, delete-orphan")
    refresh_tokens: Mapped[list["RefreshToken"]]=relationship(back_populates="user",cascade="all, delete-orphan")



class Documents(Base):
    __tablename__="documents"
    doc_id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[Optional[str]] = mapped_column(Text)  
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    # storage_path: Mapped[str] = mapped_column(Text, unique=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), index=True)
    user:Mapped["Users"] = relationship(back_populates="documents")



class RefreshToken(Base):
    __tablename__="refresh"
    token_id: Mapped[int] = mapped_column(primary_key=True, index=True)
    token_key: Mapped[str] = mapped_column(Text, unique=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    user:Mapped["Users"] = relationship(back_populates="refresh_tokens")
    