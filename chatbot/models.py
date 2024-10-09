from sqlalchemy import Column, Integer, VARCHAR, DateTime, String, ForeignKey, Text
from datetime import datetime
from sqlalchemy.orm import relationship
from pydantic import BaseModel
from typing import List, Optional

from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    files = relationship("FileModel", back_populates="user")


class ChatHistory(Base):
    __tablename__ = "chat_histories"
    chat_id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    messages = Column(Text, nullable=False)  # Store as JSON string
    timestamp = Column(String, nullable=False)
    files = relationship("FileModel", back_populates="chat")


class FileModel(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)  # Store the full file path
    upload_time = Column(String, nullable=False)  # Store the upload time as a string
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    chat_id = Column(
        String, ForeignKey("chat_histories.chat_id"), nullable=False
    )  # New field
    user = relationship("User", back_populates="files")
    chat = relationship("ChatHistory", back_populates="files")  # Establish relationship


class FileMetadata(Base):
    __tablename__ = "file_metadata"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chat_histories.chat_id"), nullable=False)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    file_metadata = Column(Text, nullable=False)


class UserCreate(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class ChatMetadata(BaseModel):
    chat_id: str
    timestamp: str
    messages: int