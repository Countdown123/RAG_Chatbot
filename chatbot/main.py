# main.py

import os
import json
import uvicorn
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Depends,
    HTTPException,
    status,
    Form,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
import models

from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from passlib.context import CryptContext
from jose import JWTError, jwt

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
import logging

from database import engine, get_db, SessionLocal
models.Base.metadata.create_all(bind=engine)
from user.user_crud import *
from models import *

from metadata import SQLChatbot

from graph import process_files,create_qa_workflow ,GraphState,process_query,create_file_processing_workflow

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable LangChain tracing to avoid LangSmith authentication errors
os.environ["LANGCHAIN_TRACING"] = "false"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "your-secret-key"  # Replace with a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# SQLChatbot setup
sql_chatbot = None

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = get_user(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user


# Ensure the 'uploads' directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Ensure the 'data' directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# FastAPI app initialization
app = FastAPI()

# CORS middleware (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Client connected")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("Client disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        logger.info(f"Sent message to client: {message}")


manager = ConnectionManager()


# Helper functions for chat and file management
def get_chat_directory(user_id: int, chat_id: str) -> Path:
    """
    Returns the Path object for a user's specific chat directory.
    """
    return Path(f"uploads/{user_id}/{chat_id}")


def save_chat_history_to_file(user_id: int, chat_id: str, messages: List[dict]):
    """
    Saves the chat messages to a JSON file in the specified format.
    """
    chat_dir = get_chat_directory(user_id, chat_id)
    chat_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    chat_history_file = chat_dir / f"{user_id}-{chat_id}-chat_history.json"
    with open(chat_history_file, "w") as f:
        json.dump(messages, f, indent=4)
    logger.info(f"Chat history saved to {chat_history_file}")


def load_chat_history_from_file(user_id: int, chat_id: str) -> List[dict]:
    """
    Loads the chat messages from the JSON file.
    """
    chat_history_file = (
        get_chat_directory(user_id, chat_id) / f"{user_id}-{chat_id}-chat_history.json"
    )
    if not chat_history_file.exists():
        return []
    with open(chat_history_file, "r") as f:
        messages = json.load(f)
    return messages


def save_uploaded_file(user_id: int, chat_id: str, file: UploadFile) -> str:
    """
    Saves the uploaded file to the appropriate chat directory and returns the file path.
    """
    chat_dir = get_chat_directory(user_id, chat_id)
    files_dir = chat_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)  # Ensure files directory exists
    file_path = files_dir / file.filename
    with open(file_path, "wb") as f:
        content = file.file.read()
        f.write(content)
    logger.info(f"File {file.filename} saved to {file_path}")
    return str(file_path)


# Routes
@app.post("/users/")
async def create_user_endpoint(
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Endpoint to create a new user.
    Validates that password and confirm_password match.
    """
    if password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    db_user = get_user(db, email=email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_create = UserCreate(email=email, password=password)
    create_user(db, user_create)
    return {"message": "User created successfully"}


@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Endpoint for user login.
    Returns a JWT access token upon successful authentication.
    """
    user = authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@app.get("/chats/", response_model=List[ChatMetadata])
async def get_chats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Endpoint to retrieve all chat sessions for the current user.
    Returns a list of chat metadata including chat_id, timestamp, and message count.
    """
    chats = db.query(ChatHistory).filter(ChatHistory.user_id == current_user.id).all()
    chat_list = [
        ChatMetadata(
            chat_id=chat.chat_id,
            timestamp=chat.timestamp,
            messages=len(json.loads(chat.messages)),
        )
        for chat in chats
    ]
    return chat_list

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        db = SessionLocal()
        user = get_user(db, email=email)
        if user is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    except JWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.connect(websocket)
    chat_id = None
    messages = []

    # Retrieve the OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API Key not found.")
        await websocket.send_text("OpenAI API Key is not set.")
        return

    qa_workflow = create_qa_workflow()  # 워크플로우를 한 번만 생성
    graph_state = None  # 그래프 상태를 저장할 변수

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message from client: {data}")
            if data.startswith("new_chat:"):
                # Extract chat_id from the message
                chat_id = data.split("new_chat:")[1]
                # Create a new ChatHistory in the database
                db_chat = ChatHistory(
                    chat_id=chat_id,
                    user_id=user.id,
                    messages=json.dumps([]),
                    timestamp=datetime.now().isoformat(),
                )
                db.add(db_chat)
                db.commit()
                db.refresh(db_chat)
                logger.info(f"Started new chat session with ID: {chat_id}")
                # Initialize messages list
                messages = []
                # Save the empty chat history to JSON file
                save_chat_history_to_file(user.id, chat_id, messages)
                # Send confirmation to client
                await websocket.send_text(f"Chat {chat_id} started.")
                graph_state = None  # 새 채팅 시작 시 그래프 상태 초기화
                logger.info(f"Graph state initialized to None for new chat: {chat_id}")
            else:
                # Handle normal chat messages
                if chat_id:
                    # Retrieve the chat from the database
                    db_chat = (
                        db.query(ChatHistory)
                        .filter(
                            ChatHistory.chat_id == chat_id,
                            ChatHistory.user_id == user.id,
                        )
                        .first()
                    )
                    if db_chat:
                        messages = json.loads(db_chat.messages)
                        # Append the sent message
                        sent_message = {
                            "type": "sent",
                            "content": data,
                            "timestamp": datetime.now().isoformat(),
                        }
                        messages.append(sent_message)

                        # 파일 타입 확인
                        file_query = db.query(FileModel).filter(FileModel.chat_id == chat_id).first()
                        if file_query:
                            file_extension = file_query.filename.split('.')[-1].lower()
                            logger.info(f"File type for chat {chat_id}: {file_extension}")
                            if file_extension in ['csv', 'xlsx', 'xls'] and sql_chatbot:
                                try:
                                    response = sql_chatbot.ask_question(data)
                                except Exception as e:
                                    logger.error(f"Error processing question: {e}")
                                    response = "Sorry, I encountered an error while processing your question."
                            elif file_extension == 'pdf':
                                try:
                                    metadata_record = db.query(FileMetadata).filter(FileMetadata.file_id == file_query.id).first()
                                    if metadata_record:
                                        file_metadata = json.loads(metadata_record.file_metadata)
                                        index_name = file_metadata.get('index_name')
                                        logger.info(f"index_name 값: {index_name}")
                                        logger.info(f"Full metadata for file: {json.dumps(file_metadata, indent=2)}")

                                        if not index_name:
                                            raise HTTPException(status_code=400, detail="Pinecone index name is missing.")

                                        if index_name:
                                            if graph_state is None or isinstance(graph_state, dict):
                                                graph_state = GraphState(
                                                    question=data,
                                                    db=index_name,
                                                    search_filters=[],
                                                    next_node="chat_interface",
                                                    metadata=file_metadata.get('metadata', {})  # 여기에 메타데이터 추가

                                                )
                                                logger.info(f"New graph state created: {graph_state}")
                                            else:
                                                graph_state.question = data
                                                graph_state.next_node = "chat_interface"
                                                graph_state.metadata = file_metadata.get('metadata', {})  # 메타데이터 업데이트

                                                logger.info(f"Updated graph state: {graph_state}")

                                            logger.info(f"Processing query with graph state: {graph_state}")
                                            result = process_query(graph_state)
                                            
                                            answer = result.get("answer", "")
                                            page_numbers = result.get("page_numbers", [])
                                            speakers = result.get("speakers", [])
                                            quotes = result.get("quotes", [])
                                            response = f"""답변:
                                            {answer}

                                            확인된 페이지:
                                            {', '.join(map(str, page_numbers))}

                                            발언자:
                                            {', '.join(speakers)}

                                            인용문:
                                            """ + "\n\n".join(f"{i+1}. {quote}" for i, quote in enumerate(quotes))
                                        else:
                                            response = "Error: Pinecone index name not found for this PDF file."
                                    else:
                                        response = "Error: Metadata not found for this PDF file."
                                except Exception as e:
                                    logger.error(f"Error processing PDF question: {str(e)}", exc_info=True)
                                    response = f"Sorry, I encountered an error while processing your question for the PDF: {str(e)}"
                            else:
                                response = "Unsupported file type for this chat session."
                        else:
                            response = "Please upload a data file before asking questions."

                        await websocket.send_text(response)

                        received_message = {
                            "type": "received",
                            "content": response,
                            "timestamp": datetime.now().isoformat(),
                        }
                        messages.append(received_message)

                        # Update the database
                        db_chat.messages = json.dumps(messages)
                        db.commit()
                        # Save the updated chat history to JSON file
                        save_chat_history_to_file(user.id, chat_id, messages)
                        logger.info(f"Chat history updated for chat {chat_id}")
                    else:
                        error_msg = "Error: Chat session not found."
                        await websocket.send_text(error_msg)
                        logger.error("Error: Chat session not found.")
                else:
                    # No chat_id yet, send an error message
                    error_msg = "Error: No chat session established."
                    await websocket.send_text(error_msg)
                    logger.error("Error: No chat session established.")
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user.email}")
        manager.disconnect(websocket)
        db.close()
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        manager.disconnect(websocket)
        db.close()

@app.post("/upload/")
async def upload_file(
    files: List[UploadFile] = File(...),
    chat_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Verify that the chat_id exists for the user
    db_chat = (
        db.query(ChatHistory)
        .filter(
            ChatHistory.chat_id == chat_id, ChatHistory.user_id == current_user.id
        )
        .first()
    )

    global sql_chatbot

    if not db_chat:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    # Retrieve existing files for the chat
    existing_files = (
        db.query(FileModel)
        .filter(FileModel.user_id == current_user.id, FileModel.chat_id == chat_id)
        .all()
    )

    # Count existing files of each type
    existing_pdf_count = 0
    existing_csv_xlsx_count = 0

    for f in existing_files:
        extension = f.filename.split(".")[-1].lower()
        if extension == "pdf":
            existing_pdf_count += 1
        elif extension in ["xlsx", "xls", "csv"]:
            existing_csv_xlsx_count += 1

    # Count the files being uploaded
    uploading_pdf_count = 0
    uploading_csv_xlsx_count = 0

    for file in files:
        filename = file.filename
        extension = filename.split(".")[-1].lower()

        if extension == "pdf":
            uploading_pdf_count += 1
        elif extension in ["xlsx", "xls", "csv"]:
            uploading_csv_xlsx_count += 1
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")

    # Enforce that the user can only upload PDF files or CSV/XLSX files, not both
    if uploading_pdf_count > 0 and uploading_csv_xlsx_count > 0:
        raise HTTPException(status_code=400, detail="You can only upload PDF files or a single CSV/XLSX file per chat, not both.")

    # Enforce the new constraints
    if existing_csv_xlsx_count > 0:
        # If a CSV/XLSX file is already uploaded, prevent uploading any new files
        raise HTTPException(status_code=400, detail="Cannot upload additional files when a CSV/XLSX file is already uploaded in this chat.")
    if existing_pdf_count > 0 and uploading_csv_xlsx_count > 0:
        # If PDF files are already uploaded, prevent uploading CSV/XLSX files
        raise HTTPException(status_code=400, detail="Cannot upload a CSV/XLSX file when PDF files are already uploaded in this chat.")

    # Enforce the limits
    if existing_pdf_count + uploading_pdf_count > 5:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 5 PDF files per chat.")
    if existing_csv_xlsx_count + uploading_csv_xlsx_count > 1:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 1 CSV/XLSX file per chat.")

    # Proceed with uploading files
    file_list = []

    for file in files:
        filename = file.filename
        extension = filename.split(".")[-1].lower()

        # Save the uploaded file to the chat's files directory
        try:
            file_path = save_uploaded_file(current_user.id, chat_id, file)
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

        # Create a new File record in the database
        db_file = FileModel(
            filename=filename,
            filepath=file_path,
            upload_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_id=current_user.id,
            chat_id=chat_id,
        )
        try:
            db.add(db_file)
            db.commit()
            db.refresh(db_file)
        except Exception as e:
            logger.error(f"Error saving file to database: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to save file information."
            )

        # Process the file based on its type
        if extension in ["xlsx", "xls", "csv"]:
            try:
                sanitized_filename = "".join(
                    e for e in filename if e.isalnum() or e in [".", "_"]
                )
                sql_chatbot = SQLChatbot(file_path, sanitized_filename)
            except ValueError as e:
                logger.error(f"Error initializing SQLChatbot: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Unexpected error initializing SQLChatbot: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="An unexpected error occurred while processing the file.",
                )
        
        elif extension == "pdf":
            try:
                file_processing_app = create_file_processing_workflow()
                
                state = GraphState(
                    file_paths=[file_path],
                    question="",
                    answer="",
                    verification_result=False,
                    page_numbers=[],
                    db="",
                    pdfs_loaded=True,
                    verification_count=0,
                    excluded_pages=set(),
                    next_question=True,
                    file_types=[extension],
                    next_node="",
                    processed_data="",
                    max_pages=0,
                    metadata={"file_names": [], "speakers": [], "max_pages": 0}
                )

                for output in file_processing_app.stream(state):
                    current_node = next(iter(output))
                    new_state = output[current_node]

                    if isinstance(new_state, dict):
                        logger.debug(f"New state from {current_node}: {new_state}")
                        state.update(new_state)
                    else:
                        raise ValueError(f"Unexpected result type from processing node: {type(new_state)}")

                    if current_node == "vector_storage" and 'db' in state:
                        logger.debug("Vector storage completed")
                        break
                    elif 'db' not in state:
                        raise ValueError("db field is missing in the state object")

                logger.info(f"PDF file processed and vectors stored for {filename}")

                result = {
                    "index_name": state.get("db", ""),
                    "max_pages": state.get("max_pages", 0),
                    "metadata": state.get("metadata", {}),
                }

                new_metadata = FileMetadata(
                    file_id=db_file.id,
                    chat_id=chat_id,
                    file_metadata=json.dumps(result)
                )
                db.add(new_metadata)
                db.commit()
                logger.info(f"PDF metadata saved to database for file ID: {db_file.id}")
                logger.info(f"Metadata content: {json.dumps(result, indent=2)}")

            except Exception as e:
                logger.error(f"Error processing PDF file: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to process PDF file.")

        # 파일 리스트에 추가
        file_list.append({"id": db_file.id, "filename": db_file.filename, "upload_time": db_file.upload_time})

    # 최종적으로 클라이언트로 응답 반환
    logger.info(f"File upload completed. File list: {file_list}")
    return {"fileList": file_list, "message": "File processed successfully"}


@app.get("/files/")
async def get_uploaded_files(
    chat_id: Optional[str] = None,  # Optional chat_id to filter files
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Endpoint to retrieve the list of files uploaded by the current user.
    If chat_id is provided, filters files for that specific chat.
    """
    query = db.query(FileModel).filter(FileModel.user_id == current_user.id)
    if chat_id:
        query = query.filter(FileModel.chat_id == chat_id)
    files = query.all()
    file_list = [
        {"id": f.id, "filename": f.filename, "upload_time": f.upload_time}
        for f in files
    ]
    return {"fileList": file_list}


@app.get("/file/{file_id}")
async def get_file_data(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info(f"Fetching file data for file_id: {file_id}, user_id: {current_user.id}")

    db_file = (
        db.query(FileModel)
        .filter(FileModel.id == file_id, FileModel.user_id == current_user.id)
        .first()
    )

    if not db_file:
        logger.warning(f"File not found: file_id={file_id}, user_id={current_user.id}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(f"File found: {db_file.filename}")

    filename = db_file.filename
    filepath = db_file.filepath
    extension = filename.split(".")[-1].lower()

    if extension == "pdf":
        metadata_record = (
            db.query(FileMetadata).filter(FileMetadata.file_id == db_file.id).first()
        )
        if not metadata_record:
            raise HTTPException(
                status_code=404, detail=f"Metadata not found for file ID: {file_id}"
            )
        return {"metadata": json.loads(metadata_record.file_metadata)}
    elif extension in ["xlsx", "xls", "csv"]:
        try:
            if extension in ["xlsx", "xls"]:
                df = pd.read_excel(filepath)
            else:  # csv
                df = pd.read_csv(filepath)

            # Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Replace NaN with 0
            df.fillna(0, inplace=True)

            # Convert DataFrame to dictionary
            data = df.to_dict(orient="records")
            columns = df.columns.tolist()

            return JSONResponse(content={"columns": columns, "data": data})
        except Exception as e:
            logger.error(f"Error processing file '{filename}': {e}")
            raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")


@app.get("/history/{chat_id}")
async def get_chat_history(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Endpoint to retrieve the chat history and associated files for a specific chat session.
    """
    db_chat = (
        db.query(ChatHistory)
        .filter(ChatHistory.chat_id == chat_id, ChatHistory.user_id == current_user.id)
        .first()
    )
    if db_chat:
        messages = json.loads(db_chat.messages)
        # Retrieve associated files
        files = (
            db.query(FileModel)
            .filter(FileModel.user_id == current_user.id, FileModel.chat_id == chat_id)
            .all()
        )
        file_list = [
            {"id": f.id, "filename": f.filename, "upload_time": f.upload_time}
            for f in files
        ]
        return {
            "chat_id": chat_id,
            "messages": messages,
            "files": file_list,
            "timestamp": db_chat.timestamp,
        }
    else:
        raise HTTPException(status_code=404, detail="Chat not found")

@app.get("/metadata/{file_id}")
async def get_metadata(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        # 파일 메타데이터를 조회
        metadata_record = (
            db.query(FileMetadata).filter(FileMetadata.file_id == file_id).first()
        )
        if not metadata_record:
            logger.error(f"Metadata not found for file ID: {file_id}")
            raise HTTPException(
                status_code=404, detail=f"Metadata not found for file ID: {file_id}"
            )
        
        # 파일 기록을 조회
        file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
        if not file_record:
            logger.error(f"File record not found for file ID: {file_id}")
            raise HTTPException(
                status_code=404, detail=f"File record not found for file ID: {file_id}"
            )
        
        logger.info(f"Metadata record: {metadata_record}")
        logger.info(f"File record: {file_record}")

        # 메타데이터를 JSON 형태로 변환
        metadata = json.loads(metadata_record.file_metadata)
        file_extension = file_record.filename.split('.')[-1].lower()

        # PDF 파일에 대한 메타데이터 처리
        if file_extension == 'pdf':
            # 메타데이터를 변환하여 반환
            transformed_metadata = {
                "파일명": file_record.filename,
                "발언자": ", ".join(metadata.get('metadata', {}).get('speakers', [])),
                "총 페이지": metadata.get('max_pages', 'N/A'),
                "인덱스 이름": metadata.get('index_name', '')  # 기존 'index_name'을 '인덱스 이름'으로 표시
            }

            # 추가 메타데이터 처리
            for key, value in metadata.get('metadata', {}).items():
                if key not in ['speakers', 'file_names', 'max_pages']:
                    transformed_metadata[key] = value

            return {"metadata": transformed_metadata}

        # PDF 외의 파일에 대한 메타데이터 처리
        else:
            return {"metadata": metadata}

    except Exception as e:
        logger.error(f"Error fetching metadata for file ID: {file_id}, error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching metadata. Please try again.")




# Serve HTML pages
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", encoding='UTF-8') as f:
        return f.read()


@app.get("/index.html", response_class=HTMLResponse)
async def read_index_html():
    with open("templates/index.html", encoding='UTF-8') as f:
        return f.read()


@app.get("/login.html", response_class=HTMLResponse)
async def read_login():
    with open("templates/login.html", encoding='UTF-8') as f:
        return f.read()


@app.get("/signup.html", response_class=HTMLResponse)
async def read_signup():
    with open("templates/signup.html", encoding='UTF-8') as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=3939, reload=True)
