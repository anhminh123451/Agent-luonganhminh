from databases.database import Base
from sqlalchemy import Column, Integer, String, DateTime, Boolean,func,ForeignKey
from sqlalchemy.orm import Mapped, mapped_column,relationship
from datetime import datetime,timezone


class User(Base):
    __tablename__="users"
    user_id=Column(Integer,primary_key=True,index=True)
    username=Column(String)
    hashed_password=Column(String)
    email=Column(String,unique=True,index=True)
    created_at=Column(DateTime,default=datetime.now(timezone.utc))
    updated_at=Column(DateTime,default=datetime.now(timezone.utc),onupdate=datetime.now(timezone.utc))

    documents: Mapped[list["Documents"]]=relationship(back_populates="user",cascade="all, delete-orphan")
    refresh_tokens: Mapped[list["RefreshToken"]]=relationship(back_populates="user",cascade="all, delete-orphan")

class Documents(Base):
    __tablename__="documents"
    doc_id=Column(Integer,primary_key=True,index=True)
    title=Column(String)
    content=Column(String)
    created_at=Column(DateTime,default=datetime.now(timezone.utc))
    user_id=Column(Integer,ForeignKey("users.user_id"))

    user:Mapped["User"] = relationship(back_populates="documents")



class RefreshToken(Base):
    __tablename__="refresh"
    token_id=Column(Integer,primary_key=True,index=True)
    token_key=Column(String,unique=True)
    user_id=Column(Integer,ForeignKey("users.user_id"))
    created_at=Column(DateTime,default=datetime.now(timezone.utc))
    is_revoked=Column(Boolean,default=False)

    user:Mapped["User"] = relationship(back_populates="refresh_tokens")
    