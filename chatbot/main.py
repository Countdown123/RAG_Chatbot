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
                        # Check for uploaded data files
                        uploaded_files = (
                            db.query(FileModel)
                            .filter(
                                FileModel.chat_id == chat_id,
                                FileModel.user_id == user.id,
                            )
                            .all()
                        )
                        valid_extensions = ["xlsx", "xls", "csv"]
                        data_files = [
                            f
                            for f in uploaded_files
                            if f.filename.split(".")[-1].lower() in valid_extensions
                        ]
                        if data_files:
                            # Initialize the agent
                            llm = ChatOpenAI(
                                model_name="gpt-3.5-turbo",
                                temperature=0,
                                openai_api_key=openai_api_key,
                            )
                            db_path = "data/chatbot_data.db"
                            db_sql = SQLDatabase.from_uri(f"sqlite:///{db_path}")

                            agent_executor = create_sql_agent(
                                llm=llm,
                                db=db_sql,
                                agent_type="openai-tools",
                                verbose=True,
                            )
                            # Process the message with the agent
                            try:
                                # Since agent_executor.run might be blocking, use run_in_executor
                                loop = asyncio.get_event_loop()
                                response = await loop.run_in_executor(
                                    None, agent_executor.run, data
                                )
                            except Exception as e:
                                logger.error(f"Error during agent execution: {e}")
                                response = "Sorry, I could not process your request."

                            await websocket.send_text(response)
                            # Append the received message
                            received_message = {
                                "type": "received",
                                "content": response,
                                "timestamp": datetime.now().isoformat(),
                            }
                            messages.append(received_message)
                        else:
                            # No data files uploaded; handle chat without agent
                            response = "You have no data files uploaded. You can still chat, but advanced functionalities are limited."
                            await websocket.send_text(response)
                            # Append the system response
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
        manager.disconnect(websocket)
        db.close()
    except Exception as e:
        logger.error(f"Error: {e}")
        manager.disconnect(websocket)
        db.close()


@app.post("/upload/")
async def upload_file(
    files: List[UploadFile] = File(...),
    chat_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    allowed_extensions = ["xlsx", "xls", "csv", "pdf"]

    # Verify that the chat_id exists for the user
    db_chat = (
        db.query(ChatHistory)
        .filter(
            ChatHistory.chat_id == chat_id, ChatHistory.user_id == current_user.id
        )
        .first()
    )
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
    pdf_count = 0
    csv_xlsx_count = 0

    for file in files:
        filename = file.filename
        extension = filename.split(".")[-1].lower()

        if extension == "pdf":
            pdf_count += 1
        elif extension in ["xlsx", "xls", "csv"]:
            csv_xlsx_count += 1
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")

    # Check limits
    if existing_pdf_count + pdf_count > 5:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 5 PDF files per chat.")
    if existing_csv_xlsx_count + csv_xlsx_count > 1:
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
            # Process data files
            try:
                if extension == "csv":
                    df = pd.read_csv(file_path, header=None)
                else:
                    df = pd.read_excel(file_path, header=None)
                # Assign default column names
                df.columns = [f"column_{i}" for i in range(len(df.columns))]
                # Clean column names
                df.columns = [str(col).strip().replace(" ", "_") for col in df.columns]
                # Replace infinite values with NaN
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                # Replace NaN with 0
                df.fillna(0, inplace=True)
                # Create an engine to the SQLite database
                data_db_path = "data/chatbot_data.db"
                data_engine = create_engine(f"sqlite:///{data_db_path}", echo=False)
                # Replace hyphens with underscores in chat_id
                sanitized_chat_id = chat_id.replace("-", "_")
                # Provide a table name with sanitized chat_id
                table_name = f"table_{current_user.id}_{sanitized_chat_id}_{db_file.id}"
                # Write the DataFrame into the SQLite database
                df.to_sql(table_name, con=data_engine, if_exists="replace", index=False)
                logger.info(
                    f"Data from file '{filename}' loaded into table '{table_name}'"
                )
            except Exception as e:
                logger.error(f"Error processing file '{filename}': {e}")
                raise HTTPException(
                    status_code=500, detail=f"Error processing file: {e}"
                )
        elif extension == "pdf":
            # For PDF files, generate and store metadata
            metadata = generate_pdf_metadata(file_path)
            new_metadata = FileMetadata(
                file_id=db_file.id,
                chat_id=db_file.chat_id,
                file_metadata=json.dumps(metadata),
            )
            db.add(new_metadata)
            db.commit()
            logger.info(f"Generated and saved metadata for file ID: {db_file.id}")

        # Append to file_list to return
        file_list.append({"id": db_file.id, "filename": db_file.filename, "upload_time": db_file.upload_time})

    # Return the updated list of files
    files_in_chat = (
        db.query(FileModel)
        .filter(FileModel.user_id == current_user.id, FileModel.chat_id == chat_id)
        .all()
    )
    file_list = [
        {"id": f.id, "filename": f.filename, "upload_time": f.upload_time}
        for f in files_in_chat
    ]
    return {"fileList": file_list}


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
    logger.info(
        f"Fetching file data for file_id: {file_id}, user_id: {current_user.id}"
    )

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
            logger.warning(
                f"Metadata not found for file ID: {file_id}. Generating new metadata."
            )
            metadata = generate_pdf_metadata(filepath)

            new_metadata = FileMetadata(
                file_id=db_file.id,
                chat_id=db_file.chat_id,
                file_metadata=json.dumps(metadata),
            )
            db.add(new_metadata)
            db.commit()
            logger.info(f"Generated and saved new metadata for file ID: {file_id}")
        else:
            metadata = json.loads(metadata_record.file_metadata)
            raise HTTPException(status_code=400, detail="Unsupported file type")

        return {"metadata": metadata}
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
    elif extension == "pdf":
        # PDF 파일에 대한 메타데이터 반환
        metadata_record = (
            db.query(FileMetadata).filter(FileMetadata.file_id == db_file.id).first()
        )
        if not metadata_record:
            raise HTTPException(
                status_code=404, detail=f"Metadata not found for file ID: {file_id}"
            )
        return {"metadata": json.loads(metadata_record.file_metadata)}
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")


# PDF 메타데이터 생성 함수
def generate_pdf_metadata(filepath):
    try:
        pdf_loader = PyPDFLoader(filepath)
        docs = pdf_loader.load()

        full_text = "\n".join([doc.page_content for doc in docs])
        llm = ChatOpenAI(model="gpt-3.5-turbo")  # ChatGPT 3.5 사용 (더 빠르고 저렴)

        prompt = f"""
        Analyze the following PDF content and extract the metadata with the following structure:
        
        - Document Title: Title of the document
        - Authors: List of authors (if available)
        - Section Headings: Key section headings from the document
        - Key Topics: Major topics discussed in the document
        - Notable Data Points: Any significant statistics, numbers, or facts in the text
        - Summary: Brief summary of the document content

        Use the following PDF text to extract the metadata:

        PDF Text: {full_text[:1500]}  # 처음 1500자만 사용

        Return the metadata as a structured key-value format.
        """

        metadata_response = llm.invoke(prompt)

        metadata = {}
        if metadata_response and hasattr(metadata_response, "content"):
            metadata_text = metadata_response.content

            lines = metadata_text.split("\n")
            for line in lines:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    metadata[key.strip()] = value.strip()
        else:
            metadata = {"Error": "Failed to generate metadata"}

        return metadata
    except Exception as e:
        logger.error(f"Error generating PDF metadata: {e}")
        return {"Error": str(e)}


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


@app.post("/upload_pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    chat_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    pdf_path = f"uploads/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    db_file = (
        db.query(FileModel)
        .filter(FileModel.chat_id == chat_id, FileModel.filename == file.filename)
        .first()
    )

    if db_file:
        existing_metadata = (
            db.query(FileMetadata).filter(FileMetadata.file_id == db_file.id).first()
        )
        if not existing_metadata:
            pdf_loader = PyPDFLoader(pdf_path)
            docs = pdf_loader.load()

            full_text = "\n".join([doc.page_content for doc in docs])
            llm = ChatOpenAI(model="gpt-4")

            prompt = f"""
            Analyze the following PDF content and extract the metadata with the following structure:
            
            - Document Title: Title of the document
            - Authors: List of authors
            - Section Headings: Key section headings from the document
            - Key Topics: Major topics discussed in the document
            - Notable Data Points: Any significant statistics, numbers, or facts in the text
            - Conclusion: Summary of the conclusion, if available

            Use the following PDF text to extract the metadata:

            PDF Text: {full_text[:1000]}

            Return the metadata as a structured key-value format.
            """

            metadata_response = llm.invoke(prompt)

            metadata = {}
            if metadata_response and hasattr(metadata_response, "content"):
                metadata_text = metadata_response.content

                lines = metadata_text.split("\n")
                for line in lines:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        metadata[key.strip()] = value.strip()
            else:
                metadata = {
                    "Error": (
                        "Failed to parse metadata from GPT response"
                        if metadata_response
                        else "No metadata generated"
                    )
                }

            db_metadata = FileMetadata(
                chat_id=chat_id, file_id=db_file.id, file_metadata=json.dumps(metadata)
            )
            db.add(db_metadata)
            db.commit()

    files = (
        db.query(FileModel)
        .filter(FileModel.user_id == current_user.id, FileModel.chat_id == chat_id)
        .all()
    )
    file_list = [
        {"id": f.id, "filename": f.filename, "upload_time": f.upload_time}
        for f in files
    ]
    return {"fileList": file_list}


@app.get("/metadata/{file_id}")
async def get_metadata(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info(f"Fetching metadata for file_id: {file_id}, user_id: {current_user.id}")

    db_file = (
        db.query(FileModel)
        .filter(FileModel.id == file_id, FileModel.user_id == current_user.id)
        .first()
    )

    if not db_file:
        logger.warning(f"File not found: file_id={file_id}, user_id={current_user.id}")
        raise HTTPException(status_code=404, detail="File not found")

    metadata_record = (
        db.query(FileMetadata).filter(FileMetadata.file_id == file_id).first()
    )

    if not metadata_record:
        logger.warning(
            f"Metadata not found for file ID: {file_id}. Generating new metadata."
        )
        if db_file.filename.lower().endswith(".pdf"):
            metadata = generate_pdf_metadata(db_file.filepath)
            new_metadata = FileMetadata(
                file_id=file_id,
                chat_id=db_file.chat_id,
                file_metadata=json.dumps(metadata),
            )
            db.add(new_metadata)
            db.commit()
            logger.info(f"Generated and saved new metadata for file ID: {file_id}")
            return {"metadata": metadata}
        else:
            raise HTTPException(
                status_code=400,
                detail="Metadata generation is only supported for PDF files",
            )

    return {"metadata": json.loads(metadata_record.file_metadata)}


@app.get("/metadata/{file_id}")
async def get_metadata(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 파일 ID로 메타데이터 조회
    metadata_record = (
        db.query(FileMetadata).filter(FileMetadata.file_id == file_id).first()
    )

    if not metadata_record:
        raise HTTPException(
            status_code=404, detail=f"Metadata not found for file ID: {file_id}"
        )

    return {"metadata": json.loads(metadata_record.file_metadata)}


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
