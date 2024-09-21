# main.py

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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn
import json
import pandas as pd
import os
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    chats = relationship("ChatHistory", back_populates="user")
    files = relationship("FileModel", back_populates="user")


class ChatHistory(Base):
    __tablename__ = "chat_histories"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    messages = Column(Text)  # JSON serialized as text
    user = relationship("User", back_populates="chats")


class FileModel(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    filepath = Column(String)
    upload_time = Column(String)  # Store the upload time as a string
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="files")


Base.metadata.create_all(bind=engine)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "your-secret-key"  # Replace with a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Pydantic models
class UserCreate(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: str = None


# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Helper functions
def get_db():
    """
    Dependency function to get a database session.
    Closes the session after the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, user: UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, email: str, password: str):
    user = get_user(db, email)
    if user is None:
        return False
    if not pwd_context.verify(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
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
        print("Client connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print("Client disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        print(f"Sent message to client: {message}")


manager = ConnectionManager()


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication.
    Authenticates the user using the JWT token passed as a query parameter.
    """
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
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received message from client: {data}")
            if data.startswith("new_chat:"):
                # Extract chat_id from the message
                chat_id = data.split("new_chat:")[1]
                # Create a new ChatHistory in the database
                db_chat = ChatHistory(
                    chat_id=chat_id, user_id=user.id, messages=json.dumps([])
                )
                db.add(db_chat)
                db.commit()
                db.refresh(db_chat)
                print(f"Started new chat session with ID: {chat_id}")
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
                        messages.append({"type": "sent", "content": data})
                        # Echo the message back or implement chatbot logic here
                        response = f"Echo: {data}"
                        await websocket.send_text(response)
                        # Add the response to the messages
                        messages.append({"type": "received", "content": response})
                        db_chat.messages = json.dumps(messages)
                        db.commit()
                    else:
                        error_msg = "Error: Chat session not found."
                        await websocket.send_text(error_msg)
                        print("Error: Chat session not found.")
                else:
                    # No chat_id yet, send an error message
                    error_msg = "Error: No chat session established."
                    await websocket.send_text(error_msg)
                    print("Error: No chat session established.")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        db.close()
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)
        db.close()


@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Endpoint to upload files. Accepts .xlsx, .csv, and .pdf files.
    Stores the file and updates the database.
    """
    allowed_extensions = ["xlsx", "csv", "pdf"]
    filename = file.filename
    extension = filename.split(".")[-1].lower()

    if extension in allowed_extensions:
        # Save the uploaded file to a directory
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_location = os.path.join(upload_dir, filename)
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

        # Create a new File record in the database
        db_file = FileModel(
            filename=filename,
            filepath=file_location,
            upload_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            user_id=current_user.id,
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)

        # Retrieve the updated list of files
        files = db.query(FileModel).filter(FileModel.user_id == current_user.id).all()
        file_list = [
            {"id": f.id, "filename": f.filename, "upload_time": f.upload_time}
            for f in files
        ]
        return {"fileList": file_list}
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")


@app.get("/files/")
async def get_uploaded_files(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    Endpoint to retrieve the list of files uploaded by the current user.
    """
    files = db.query(FileModel).filter(FileModel.user_id == current_user.id).all()
    file_list = [{"filename": f.filename, "upload_time": f.upload_time} for f in files]
    return {"fileList": file_list}


@app.get("/file/{file_id}")
async def get_file_data(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Endpoint to retrieve the contents of an uploaded file.
    For .xlsx and .csv files, returns data as JSON.
    For .pdf files, returns a message.
    """
    db_file = (
        db.query(FileModel)
        .filter(FileModel.id == file_id, FileModel.user_id == current_user.id)
        .first()
    )
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")

    filename = db_file.filename
    filepath = db_file.filepath
    extension = filename.split(".")[-1].lower()

    if extension in ["xlsx", "csv"]:
        try:
            if extension == "xlsx":
                df = pd.read_excel(filepath)
            else:  # csv
                df = pd.read_csv(filepath)
            data = df.to_dict(orient="records")
            columns = df.columns.tolist()
            return {"columns": columns, "data": data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    elif extension == "pdf":
        return {"message": "PDF file viewing is not supported in this application."}
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")


@app.get("/history/{chat_id}")
async def get_chat_history(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Endpoint to retrieve the chat history for a specific chat session.
    """
    db_chat = (
        db.query(ChatHistory)
        .filter(ChatHistory.chat_id == chat_id, ChatHistory.user_id == current_user.id)
        .first()
    )
    if db_chat:
        messages = json.loads(db_chat.messages)
        return {"chat_id": chat_id, "messages": messages}
    else:
        raise HTTPException(status_code=404, detail="Chat not found")


# Serve HTML pages
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html") as f:
        return f.read()


@app.get("/index.html", response_class=HTMLResponse)
async def read_index_html():
    with open("templates/index.html") as f:
        return f.read()


@app.get("/login.html", response_class=HTMLResponse)
async def read_login():
    with open("templates/login.html") as f:
        return f.read()


@app.get("/signup.html", response_class=HTMLResponse)
async def read_signup():
    with open("templates/signup.html") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=3939, reload=True)
