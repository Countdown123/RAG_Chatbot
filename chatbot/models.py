# models.py

from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from pydantic import BaseModel
from typing import Optional

from database import Base

class User(Base):
    """
    Represents a user in the system.
    """
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    files = relationship("FileModel", back_populates="user")

class ChatHistory(Base):
    """
    Represents a chat history for a user.
    """
    __tablename__ = "chat_histories"
    chat_id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    messages = Column(Text, nullable=False)  # Store as JSON string
    timestamp = Column(String, nullable=False)
    files = relationship("FileModel", back_populates="chat")

class FileModel(Base):
    """
    Represents a file uploaded by a user.
    """
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
    """
    Represents metadata associated with a file.
    """
    __tablename__ = "file_metadata"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chat_histories.chat_id"), nullable=False)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    file_metadata = Column(Text, nullable=False)

class UserCreate(BaseModel):
    """
    Pydantic model for user creation.
    """
    email: str
    password: str

class Token(BaseModel):
    """
    Pydantic model for authentication tokens.
    """
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """
    Pydantic model for token data.
    """
    email: Optional[str] = None

class ChatMetadata(BaseModel):
    """
    Pydantic model for chat metadata.
    """
    chat_id: str
    timestamp: str
    messages: int
