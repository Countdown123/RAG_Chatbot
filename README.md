# 🤖 RetrieveR
<p align="center">
  <img src="https://github.com/Countdown123/RAG_Chatbot/blob/main/chatbot/static/retriever.png" width="384" height="384"/>
</p>


"RetrieveR is an AI-powered document analysis service that transforms unstructured data into actionable insights through intelligent, real-time interactions."

## 📄 Description
RetrieveR is a document analysis service that leverages AI technology to analyze user-uploaded documents and provide accurate answers. It supports various types of unstructured data, such as meeting minutes, business Excel files, and CSV files. Based on the uploaded files, it offers a chatbot-style interaction that allows users to ask questions and retrieve the necessary information. By understanding the data within the files, users can quickly access the information they need in a conversational manner, making it easy to gain insights.

## ⏳ Development period
2024.08.26 ~ 2024.10.18

## ✨ Key Features
### Support for various data formats
Analyzes unstructured data such as meeting minutes, Excel, CSV, etc., to extract key information. In particular, for meeting minutes, AI automatically extracts metadata and provides it in an easily viewable format.
### AI-based analysis
Leverages artificial intelligence to understand document content and provide customized answers for users.
### Intuitive user interface
Offers an easy-to-use interface, even for users without data or IT knowledge.
### Chatbot interaction
Users can engage in relevant questions with the chatbot based on the uploaded documents. Through these questions, users can effectively extract the necessary information from the document and gain insights.

## 📝 How to Use
### 1. Upload your documents
You can upload documents in Excel, CSV, or PDF formats.
### 2. AI processes the documents
The AI automatically analyzes the content of the uploaded documents and provides metadata for PDFs.
### 3. Interact with the chatbot
Ask questions through the interactive chatbot, and receive real-time answers based on the analyzed data. This allows you to explore the documents further and uncover valuable insights.

## 🛠️ Tech Stack
- **Backend**: FastAPI, Python
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite, Pinecone
- **AI/ML**: GPT, LangChain, LangGraph
- **Real-time Communication**: WebSocket

## ⚙️ Installation and Execution
### 1. Clone the repository

    git clone https://github.com/Countdown123/RAG_Chatbot.git

### 2. Navigate to the project directory

    cd RAG_Chatbot/chatbot

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Set up runtime environment
Check the runtime.txt file to see the required Python version and set up the environment accordingly.

### 5. Replace API Keys
Paste your API keys in `chatbot/.env`

    OPENAI_API_KEY = ""
    PINECONE_API_KEY = ""

### 6. Start the server
Run the FastAPI server locally.

    python main.py

### 7. Get started on web browser
Open http://127.0.0.1:3939/

## 📸 Screenshots
<p align="center">
  <img src="https://github.com/Countdown123/RAG_Chatbot/blob/main/chatbot/static/retriver_screenshot_1.png"/>
</p>
<p align="center">
  <img src="https://github.com/Countdown123/RAG_Chatbot/blob/main/chatbot/static/retriver_screenshot_3.png"/>
</p>
<p align="center">
  <img src="https://github.com/Countdown123/RAG_Chatbot/blob/main/chatbot/static/retriver_screenshot_4.png"/>
</p>