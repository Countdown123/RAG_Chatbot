# RetrieveR

"RetrieveR is an AI-powered document analysis service that transforms unstructured data into actionable insights through intelligent, real-time interactions."

## Description
RetrieveR is a document analysis service that leverages AI technology to analyze user-uploaded documents and provide accurate answers. It supports various formats of unstructured data, such as meeting minutes, business Excel files, and CSV files. Through artificial intelligence, it understands the content of the documents and automatically extracts relevant information.

## Development period
2024.08.26 ~ 2024.10.18

## Key Features
- **Support for various data formats**: Analyzes unstructured data such as meeting minutes, Excel, CSV, etc., to extract key information.
- **AI-based analysis**: Leverages artificial intelligence to understand document content and provide customized answers for users.
- **Intuitive user interface**: Provides an easy-to-use interface, even for users without data or IT knowledge.
- **Automation of business processes**: Automates information provision through document analysis, maximizing decision-making and operational efficiency.
- **Chatbot and Data Visualization**: Users can interact with the chatbot to explore information based on the analyzed data, and easily understand the data through interactive dashboards.

## How to Use
### 1. Upload documents
You can upload Excel, CSV, or PDF files.
### 2. AI analyzes the documents 
The AI analyzes the content of the uploaded documents and provides relevant answers.
### 3. Interact through the chatbot 
Ask questions via the interactive chatbot and receive real-time answers based on the information within the documents. During this process, you can explore the data and gain the insights you need.

## Tech Stack
- **Backend**: FastAPI, Python
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite, Pinecone
- **AI/ML**: GPT, LangChain, LangGraph
- **Real-time Communication**: WebSocket

## Installation and Execution
### 1. Project clone

    git clone https://github.com/Countdown123/RAG_Chatbot.git

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Set up runtime environment
Check the runtime.txt file to see the required Python version and set up the environment accordingly.

### 4. Start the server
Run the FastAPI server locally.

    python main.py

