from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from typing import List

app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# In-memory chat history for simplicity (you can use a database for persistence)
chat_history = {}


# HTML response for serving the main page
@app.get("/", response_class=HTMLResponse)
async def get():
    with open("templates/index.html") as f:
        return f.read()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


# Instantiate the connection manager
manager = ConnectionManager()


# WebSocket chat endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("new_chat:"):
                # New chat session, reset chat history
                chat_id = data.split(":")[1]
                chat_history[chat_id] = []
                await websocket.send_text(f"Chat {chat_id} started.")
            else:
                # Append to chat history and broadcast the message
                current_chat_id = list(chat_history.keys())[
                    -1
                ]  # Get the last (current) chat
                chat_history[current_chat_id].append({"type": "sent", "content": data})
                await manager.broadcast(f"Message from user: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Endpoint to upload Excel files
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".xlsx"):
        # Read the Excel file and convert it to HTML table
        df = pd.read_excel(file.file)
        table_html = df.to_html()
        return {"tableData": table_html}
    return {"error": "Unsupported file type"}


# Chat history retrieval endpoint (for debugging or future expansion)
@app.get("/history/{chat_id}")
async def get_chat_history(chat_id: str):
    return chat_history.get(chat_id, {"error": "Chat not found"})
