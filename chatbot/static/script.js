// static/script.js

// Utility function to generate UUID (for unique chat IDs)
function generateUUID() { // Public Domain/MIT
    var d = new Date().getTime();//Timestamp
    var d2 = (performance && performance.now && (performance.now()*1000)) || 0;//Time in microseconds since page-load or 0 if unsupported
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16;//random number between 0 and 16
        if(d > 0){
            r = (d + r)%16 | 0;
            d = Math.floor(d/16);
        } else {
            r = (d2 + r)%16 | 0;
            d2 = Math.floor(d2/16);
        }
        return (c==='x' ? r : (r&0x3|0x8)).toString(16);
    });
}

// Initialize the application
document.addEventListener("DOMContentLoaded", function () {
    const accessToken = localStorage.getItem("accessToken");

    // Check if the token is available
    if (!accessToken) {
        redirectToLogin();
    } else {
        initializeWebSocket(accessToken);  // Pass the token when initializing the WebSocket
        fetchChatList();                   // Fetch and display chat list
        setupLogoutButton();
        setupSendMessageForm();            // Set up the message form submit handler
        setupUploadForm();                 // Set up the file upload form submit handler
        setupNewChatButton();              // Set up the new chat button
    }
});

// Redirect to login page
function redirectToLogin() {
    window.location.href = "/login.html";
}

// Handle logout
function setupLogoutButton() {
    const logoutBtn = document.getElementById("logoutBtn");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", function () {
            saveChatHistory();  // Save chat history when the user logs out
            localStorage.removeItem("accessToken");
            redirectToLogin();
        });
    }
}

// Initialize WebSocket connection with the token
function initializeWebSocket(token) {
    // const ws = new WebSocket(`ws://127.0.0.1:3939/ws?token=${token}`);
    const ws = new WebSocket(`wss://rag-chatbot-t20p.onrender.com/ws?token=${token}`);
    window.ws = ws;

    ws.onopen = () => {
        console.log("Connected to the WebSocket server");
        // Optionally, you can start a default chat or wait for user action
        // For now, we'll wait for user to start a chat
    };

    ws.onmessage = (event) => {
        const data = event.data;
        if (data.startsWith("Chat ") && data.endsWith(" started.")) {
            // Extract chat_id from the message
            const chat_id = data.split("Chat ")[1].split(" started.")[0];
            currentChatId = chat_id;
            chatHistories[currentChatId] = [];  // Initialize chat history
            // Create a new chat history entry in the UI
            createNewChatHistoryEntry(currentChatId, new Date().toLocaleString());
        } else {
            receiveMessage(data);
        }
    };

    ws.onclose = (event) => {
        console.log("WebSocket closed:", event);
        if (currentChatId && chatHistories[currentChatId]) {
            saveChatHistory(currentChatId);  // Save chat history on WebSocket disconnection
        }
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        alert("WebSocket connection failed. Please try again later.");
    };
}

var currentChatId = null;  // Track the current chat session
let chatHistories = {};    // Store chat histories for different sessions

// Fetch the list of chat sessions for the user
function fetchChatList() {
    const token = localStorage.getItem("accessToken");
    fetch("/chats/", {
        method: "GET",
        headers: {
            "Authorization": `Bearer ${token}`
        }
    })
        .then(response => {
            if (response.status === 401) {
                alert("Session expired. Please log in again.");
                localStorage.removeItem("accessToken");
                redirectToLogin();
                throw new Error("Unauthorized");
            }
            if (!response.ok) {
                throw new Error("Failed to fetch chat list.");
            }
            return response.json();
        })
        .then(data => {
            if (data) {
                updateChatListDisplay(data);
            }
        })
        .catch(error => {
            console.error("Error fetching chat list:", error);
            alert("Error fetching chat list. Please try again.");
        });
}

// Update the chat list display
function updateChatListDisplay(chatList) {
    const chatHistoryList = document.getElementById("chatHistory");
    if (!chatHistoryList) return;

    chatHistoryList.innerHTML = "";  // Clear existing list

    if (chatList.length === 0) {
        const noChatsItem = document.createElement("li");
        noChatsItem.textContent = "No previous chats found.";
        chatHistoryList.appendChild(noChatsItem);
        return;
    }

    chatList.forEach(chat => {
        const listItem = document.createElement("li");
        listItem.classList.add("history-item");
        listItem.id = `chat-${chat.chat_id}`;
        const timestamp = new Date(chat.timestamp).toLocaleString();
        listItem.textContent = `Chat from ${timestamp} (${chat.messages} messages)`;
        listItem.onclick = () => loadChatContent(chat.chat_id);
        chatHistoryList.appendChild(listItem);
    });
}

// Load a specific chat's content into the chat window (including files)
function loadChatContent(chatId) {
    currentChatId = chatId;

    // Fetch chat history and associated files from the backend
    fetch(`/history/${chatId}`, {
        method: "GET",
        headers: {
            "Authorization": `Bearer ${localStorage.getItem("accessToken")}`
        }
    })
        .then(response => {
            if (response.status === 401) {
                alert("Session expired. Please log in again.");
                localStorage.removeItem("accessToken");
                redirectToLogin();
                throw new Error("Unauthorized");
            }
            if (!response.ok) {
                throw new Error(`Chat history for chat ID ${chatId} not found`);
            }
            return response.json();
        })
        .then(data => {
            if (data.messages) {
                displayChatMessages(data.messages);  // Load and display chat messages
                if (data.files) {
                    displayChatFiles(data.files);      // Load and display associated files
                }
                chatHistories[currentChatId] = data.messages; // Update local chat history
                const messageCount = data.messages.length;
                const timestamp = new Date(data.timestamp).toLocaleString();
                updateChatHistoryEntry(chatId, timestamp, messageCount);
            }
        })
        .catch(error => {
            console.error("Error loading chat history:", error);
            alert("Error loading chat history. Please try again.");
        });
}

// Handle received messages and display them
function receiveMessage(message) {
    if (!currentChatId) {
        alert("Received a message without an active chat session.");
        return;
    }
    addMessageToDisplay(message, "received");  // Display the received message
}

// Append a message to the chat window and save in chat history
function addMessageToDisplay(text, type) {
    const messagesContainer = document.getElementById("messages");
    if (!messagesContainer) return;

    const messageElement = document.createElement("li");
    messageElement.classList.add(type);  // 'sent' or 'received'
    messageElement.textContent = text;
    messagesContainer.appendChild(messageElement);
    scrollToBottom(messagesContainer);  // Ensure the chat window scrolls to the latest message

    // Save the message in the current chat history
    if (currentChatId) {
        if (!chatHistories[currentChatId]) {
            chatHistories[currentChatId] = [];
        }
        chatHistories[currentChatId].push({ type, content: text, timestamp: new Date().toISOString().toLocaleString() });
        updateChatHistoryEntry(currentChatId, new Date().toLocaleString(), chatHistories[currentChatId].length);
    }
}

// Scroll the chat window to the bottom to show the latest message
function scrollToBottom(container) {
    container.scrollTop = container.scrollHeight;
}

// Create a new entry in the chat history list
function createNewChatHistoryEntry(chatId, timestamp) {
    const chatHistoryList = document.getElementById("chatHistory");
    if (!chatHistoryList) return;

    // Create a new history item
    const historyItem = document.createElement("li");
    historyItem.classList.add("history-item");
    historyItem.id = `chat-${chatId}`;
    historyItem.textContent = `Chat from ${timestamp.toLocaleString()} (0 messages)`;
    historyItem.onclick = () => loadChatContent(chatId);
    chatHistoryList.insertBefore(historyItem, chatHistoryList.firstChild);
}

// Update an existing chat history entry with the number of messages
function updateChatHistoryEntry(chatId, timestamp, messageCount) {
    const historyItem = document.getElementById(`chat-${chatId}`);
    if (historyItem) {
        historyItem.textContent = `Chat from ${timestamp.toLocaleString()} (${messageCount} messages)`;
    }
}

// Start a new chat session
function startNewChat() {
    if (!window.ws || window.ws.readyState !== WebSocket.OPEN) {
        alert("WebSocket is not connected. Please try again later.");
        return;
    }

    const newChatId = generateUUID();  // Generate a unique chat ID
    currentChatId = newChatId;
    chatHistories[currentChatId] = [];  // Initialize the new chat history

    // Clear the current chat display and input field
    document.getElementById("messages").innerHTML = "";
    document.getElementById("messageText").value = "";

    // Notify the server about the new chat session
    window.ws.send(`new_chat:${newChatId}`);

    // Display the welcome message in the UI
    addMessageToDisplay(welcomeMessage, "received");

    // Add the new chat to the history list
    const timestamp = new Date().toLocaleString();
    createNewChatHistoryEntry(currentChatId, timestamp);

    // Refresh the chat list
    fetchChatList();
    
    loadChatContent(currentChatId);
}


// Save chat history to the backend
function saveChatHistory(chatId) {
    if (!chatId || !chatHistories[chatId]) return;

    // Optionally, implement logic to save chat history if not already handled by WebSocket
    // Currently, chat history is saved via WebSocket on message reception and chat initiation
    console.log(`Chat history for ${chatId} saved.`);
}

// Set up the message form submit functionality
function setupSendMessageForm() {
    const messageForm = document.getElementById("messageForm");
    if (messageForm) {
        messageForm.addEventListener("submit", function (event) {
            event.preventDefault();  // Prevent the form from submitting normally
            sendMessage(event);
        });
    }
}

// Send a message via WebSocket
function sendMessage(event) {
    event.preventDefault();
    const messageInput = document.getElementById("messageText");
    if (!messageInput) return;

    const messageText = messageInput.value.trim();

    if (messageText === "") return;  // Do nothing if message is empty

    if (!currentChatId) {
        alert("No active chat session. Please start a new chat.");
        return;
    }

    // Send the message to the WebSocket server
    window.ws.send(messageText);

    // Display the sent message in the chat window
    addMessageToDisplay(messageText, "sent");

    // Clear the input field
    messageInput.value = "";
}

// Set up the file upload form submit functionality
function setupUploadForm() {
    const uploadForm = document.getElementById("uploadForm");
    if (uploadForm) {
        uploadForm.addEventListener("submit", function (event) {
            event.preventDefault();  // Prevent the form from submitting normally
            uploadFile();
        });
    }
}

// Handle file upload and associate it with the current chat
async function uploadFile() {
    const accessToken = localStorage.getItem("accessToken");
    const uploadForm = document.getElementById("uploadForm");
    if (!uploadForm) return;

    if (!currentChatId) {
        alert("Please start a chat before uploading files.");
        return;
    }

    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Please select at least one file to upload.");
        return;
    }

    // Count the number of files of each type being uploaded
    let uploadingPdfCount = 0;
    let uploadingCsvXlsxCount = 0;

    for (let i = 0; i < fileInput.files.length; i++) {
        const file = fileInput.files[i];
        const extension = file.name.split('.').pop().toLowerCase();
        if (extension === 'pdf') {
            uploadingPdfCount += 1;
        } else if (['csv', 'xlsx', 'xls'].includes(extension)) {
            uploadingCsvXlsxCount += 1;
        } else {
            alert(`Unsupported file type: ${extension}`);
            return;
        }
    }

    // Enforce that the user can only upload PDF files or CSV/XLSX files, not both
    if (uploadingPdfCount > 0 && uploadingCsvXlsxCount > 0) {
        alert("You can only upload PDF files or a single CSV/XLSX file per chat, not both.");
        return;
    }

    try {
        // Fetch the existing files for the current chat
        const response = await fetch(`/files/?chat_id=${currentChatId}`, {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${accessToken}`
            },
        });

        if (response.status === 401) {
            alert("Session expired. Please log in again.");
            localStorage.removeItem("accessToken");
            redirectToLogin();
            return;
        }

        if (!response.ok) {
            throw new Error("Error fetching existing files.");
        }

        const data = await response.json();

        if (data.fileList) {
            // Count existing files
            let existingPdfCount = 0;
            let existingCsvXlsxCount = 0;

            data.fileList.forEach(file => {
                const extension = file.filename.split('.').pop().toLowerCase();
                if (extension === 'pdf') {
                    existingPdfCount += 1;
                } else if (['csv', 'xlsx', 'xls'].includes(extension)) {
                    existingCsvXlsxCount += 1;
                }
            });

            // Enforce the new constraints
            // If there is at least one CSV/XLSX file, prevent uploading any new files
            if (existingCsvXlsxCount > 0) {
                alert("You cannot upload additional files when a CSV/XLSX file is already uploaded in this chat.");
                return;
            }

            // If there is at least one PDF file, prevent uploading any CSV/XLSX files
            if (existingPdfCount > 0 && uploadingCsvXlsxCount > 0) {
                alert("You cannot upload a CSV/XLSX file when PDF files are already uploaded in this chat.");
                return;
            }

            // Enforce the limits
            if ((existingPdfCount + uploadingPdfCount) > 5) {
                alert("You can upload a maximum of 5 PDF files per chat.");
                return;
            }

            if ((existingCsvXlsxCount + uploadingCsvXlsxCount) > 1) {
                alert("You can upload a maximum of 1 CSV/XLSX file per chat.");
                return;
            }

            // Proceed with the upload
            const formData = new FormData();
            formData.append("chat_id", currentChatId);

            // Append each selected file to the FormData
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append("files", fileInput.files[i]);
            }

            // Show the loading screen
            showLoadingScreen();

            const uploadResponse = await fetch("/upload/", {
                method: "POST",
                headers: { 
                    "Authorization": `Bearer ${accessToken}`
                },
                body: formData,
            });

            handleFileUploadResponse(uploadResponse);

        } else {
            alert("Error fetching existing files. Please try again.");
        }
    } catch (error) {
        console.error("Error in uploadFile:", error);
        alert("An error occurred. Please try again.");
    } finally {
        // Hide the loading screen regardless of success or failure
        hideLoadingScreen();
    }
}

// 파일 업로드 후 처리 함수 수정 (메타데이터 표시 제거)
function handleFileUploadResponse(response) {
    if (response.status === 401) {
        alert("Session expired. Please log in again.");
        localStorage.removeItem("accessToken");
        redirectToLogin();
        return;
    }
    return response.json().then((data) => {
        console.log("Upload response data:", data);
        if (data.fileList) {
            updateFileListDisplay(data.fileList);
            alert("File(s) uploaded successfully!");
            document.getElementById("uploadForm").reset();
        } else if (data.detail) {
            console.error("Error:", data.detail);
            alert("Error: " + data.detail);
        } else {
            console.error("Unexpected response:", data);
            alert("An unexpected error occurred during file upload.");
        }
    }).catch(error => {
        console.error("Error parsing upload response:", error);
        alert("An error occurred while processing the upload response.");
    });
}


// Fetch the list of uploaded files for the current chat
function fetchFileList(chatId = null) {
    var accessToken = localStorage.getItem("accessToken");
    let url = "/files/";
    if (chatId) {
        url += `?chat_id=${chatId}`;
    }
    fetch(url, {
        method: "GET",
        headers: {
            "Authorization": `Bearer ${accessToken}`
        },
    })
        .then((response) => {
            if (response.status === 401) {
                // Token might have expired
                alert("Session expired. Please log in again.");
                localStorage.removeItem("accessToken");
                redirectToLogin();
            }
            return response.json();
        })
        .then((data) => {
            if (data.fileList) {
                updateFileListDisplay(data.fileList);
            } else if (data.detail) {
                console.error("Error:", data.detail);
                alert("Error: " + data.detail);
            } else {
                console.error("Unexpected response:", data);
                alert("An unexpected error occurred while fetching file list.");
            }
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

// Update the file list display
function updateFileListDisplay(fileList) {
    console.log("Updating file list display:", fileList);
    const fileListContainer = document.getElementById("fileList");
    if (!fileListContainer) return;

    fileListContainer.innerHTML = "";  // Clear the file list display

    if (fileList.length === 0) {
        const noFilesItem = document.createElement("li");
        noFilesItem.textContent = "No files uploaded for this chat.";
        fileListContainer.appendChild(noFilesItem);
        return;
    }

    fileList.forEach(file => {
        const listItem = createFileListItem(file);
        fileListContainer.appendChild(listItem);
    });

    // Ensure the file list container is visible
    document.getElementById("fileListContainer").style.display = "block";  
}

// 파일 리스트 아이템 생성 함수 수정
function createFileListItem(file) {
    const listItem = document.createElement("li");
    listItem.textContent = `${file.filename} (Uploaded on ${file.upload_time})`;
    listItem.style.cursor = 'pointer';

    if (file.filename.toLowerCase().endsWith('.pdf')) {
        listItem.onclick = () => fetchAndShowMetadata(file.id, file.filename);
    } else {
        listItem.onclick = () => fetchFileData(file.id, file.filename);
    }

    return listItem;
}

// PDF 파일 클릭 시 메타데이터 조회 및 표시
function fetchAndShowMetadata(fileId, filename) {
    const accessToken = localStorage.getItem("accessToken");

    console.log("Fetching metadata for fileId:", fileId);

    clearMetadataTable();
    showLoadingScreen();  // Show loading screen before fetch

    fetch(`/metadata/${fileId}`, {
        method: "GET",
        headers: {
            "Authorization": `Bearer ${accessToken}`,
        },
    })
        .then(response => {
            if (response.status === 401) {
                hideLoadingScreen();
                alert("Session expired. Please log in again.");
                localStorage.removeItem("accessToken");
                redirectToLogin();
                throw new Error("Unauthorized");
            }
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`Failed to fetch metadata: ${text}`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.metadata) {
                showMetadataTable(data.metadata);
            } else {
                alert("No metadata found for this file.");
            }
        })
        .catch(error => {
            console.error("Error fetching metadata:", error);
            alert("Error fetching metadata. Please try again.");
        })
        .finally(() => {
            hideLoadingScreen();  // Always hide loading screen when done
        });
}

// Fetch file data when a file is clicked
function fetchFileData(fileId, filename) {
    const accessToken = localStorage.getItem("accessToken");
    console.log(`Fetching data for file: ${filename} (ID: ${fileId})`);
    
    showLoadingScreen("Fetching file data...");  // Show loading screen

    fetch(`/file/${fileId}`, {
        method: "GET",
        headers: {
            "Authorization": `Bearer ${accessToken}`,
        },
    })
        .then(response => {
            console.log(`Response status: ${response.status}`);
            if (response.status === 401) {
                hideLoadingScreen();
                alert("Session expired. Please log in again.");
                localStorage.removeItem("accessToken");
                redirectToLogin();
                throw new Error("Unauthorized");
            }
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`Failed to fetch file data: ${text}`);
                });
            }
            return response.json();
        })
        .then(data => {
            hideLoadingScreen();  // Hide loading screen
            console.log("Received data:", data);
            if (data.columns && data.data) {
                displayTable(data.columns, data.data, filename);
            } else if (data.metadata) {
                showMetadataTable(data.metadata);
            } else {
                alert("An error occurred while retrieving the file data.");
            }
        })
        .catch(error => {
            hideLoadingScreen();  // Hide loading screen
            console.error("Error fetching file data:", error);
            if (error.message.includes("Metadata not found")) {
                alert("Metadata is being generated for this file. Please try again in a few moments.");
            } else {
                alert(`Error fetching file data: ${error.message}`);
            }
        });
}

// Display data in a modal table
function displayTable(columns, data, filename) {
    const modal = createModal();
    const modalContent = createModalContent(filename);

    const table = createTable(columns, data);

    modalContent.appendChild(table);
    modal.appendChild(modalContent);
    document.body.appendChild(modal);

    // Initialize DataTables on the newly created table
    $(table).DataTable({
        responsive: true,
        paging: true,
        searching: true,
        ordering: true,
        // Add any additional DataTables configurations here
    });
}

// Create the modal structure
function createModal() {
    const existingModal = document.getElementById("tableModal");
    if (existingModal) {
        existingModal.style.display = "none"; // Hide existing modal if any
        existingModal.remove(); // Remove it to create a fresh one
    }

    const modal = document.createElement("div");
    modal.classList.add("modal");
    modal.id = "tableModal"; // Assign ID for consistency
    modal.style.display = "block"; // Show the modal
    return modal;
}

// Create the modal content
function createModalContent(filename) {
    const modalContent = document.createElement("div");
    modalContent.classList.add("modal-content");

    const closeButton = createCloseButton(modalContent);
    const title = document.createElement("h2");
    title.textContent = filename;

    modalContent.appendChild(closeButton);
    modalContent.appendChild(title);

    return modalContent;
}

// Create the modal close button
function createCloseButton(modalContent) {
    const closeButton = document.createElement("span");
    closeButton.classList.add("close-button");
    closeButton.innerHTML = "&times;";
    closeButton.onclick = () => {
        modalContent.parentElement.style.display = "none";
        modalContent.parentElement.remove();
    };
    return closeButton;
}

// Create a table with columns and data
function createTable(columns, data) {
    const table = document.createElement("table");
    table.classList.add("display", "responsive", "nowrap", "data-table");
    table.style.width = "100%";
    table.id = "dataTable"; // Assign ID for DataTables

    const thead = createTableHeader(columns);
    const tbody = createTableBody(columns, data);

    table.appendChild(thead);
    table.appendChild(tbody);

    return table;
}

// Create the table header
function createTableHeader(columns) {
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    columns.forEach(col => {
        const th = document.createElement("th");
        th.textContent = col;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    return thead;
}

// Create the table body
function createTableBody(columns, data) {
    const tbody = document.createElement("tbody");

    data.forEach(row => {
        const tr = document.createElement("tr");
        columns.forEach(col => {
            const td = document.createElement("td");
            td.textContent = row[col];
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });

    return tbody;
}

// Generic error handler
function showErrorAlert(error) {
    console.error("Error:", error);
    alert("An unexpected error occurred.");
}

// Set up the new chat button
function setupNewChatButton() {
    const newChatBtn = document.getElementById("newChatBtn");
    if (newChatBtn) {
        newChatBtn.addEventListener("click", function () {
            startNewChat();
        });
    }
}

// Display chat messages
function displayChatMessages(messages) {
    const messagesContainer = document.getElementById("messages");
    if (!messagesContainer) return;

    messagesContainer.innerHTML = "";  // Clear the chat display

    messages.forEach(msg => {
        const messageElement = document.createElement("li");
        messageElement.classList.add(msg.type);  // 'sent' or 'received'
        messageElement.textContent = msg.content;
        messagesContainer.appendChild(messageElement);
    });

    scrollToBottom(messagesContainer);
}

// Display associated files with chat history
function displayChatFiles(files) {
    const fileListContainer = document.getElementById("fileList");
    if (!fileListContainer) return;

    fileListContainer.innerHTML = "";  // Clear the file list display

    if (files.length === 0) {
        const noFilesItem = document.createElement("li");
        noFilesItem.textContent = "No files associated with this chat.";
        fileListContainer.appendChild(noFilesItem);
        return;
    }

    files.forEach(file => {
        const listItem = createFileListItem(file);
        fileListContainer.appendChild(listItem);
    });

    // Ensure the file list container is visible
    document.getElementById("fileListContainer").style.display = "block";  
}

// 메타데이터 테이블 초기화 함수
function clearMetadataTable() {
    const existingTable = document.getElementById("metadataTable");
    if (existingTable) {
        existingTable.remove();
    }

    const existingModal = document.querySelector('.modal');
    if (existingModal) {
        existingModal.remove();
    }
}

// 메타데이터 테이블 표시 함수
function showMetadataTable(metadata) {
    const popup = document.createElement("div");
    popup.classList.add("modal");

    const popupContent = document.createElement("div");
    popupContent.classList.add("modal-content");

    const closeButton = document.createElement("button");
    closeButton.textContent = "Close";
    closeButton.onclick = function () {
        document.body.removeChild(popup);
    };

    const tableElement = document.createElement("table");
    tableElement.id = "metadataTable";

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    const thKey = document.createElement("th");
    thKey.textContent = "Key";
    const thValue = document.createElement("th");
    thValue.textContent = "Value";
    headerRow.appendChild(thKey);
    headerRow.appendChild(thValue);
    thead.appendChild(headerRow);
    tableElement.appendChild(thead);

    const tbody = document.createElement("tbody");

    for (const key in metadata) {
        if (metadata.hasOwnProperty(key)) {
            const row = document.createElement("tr");

            const tdKey = document.createElement("td");
            tdKey.textContent = key;

            const tdValue = document.createElement("td");
            if (Array.isArray(metadata[key])) {
                tdValue.innerHTML = metadata[key].join("<br>");
            } else if (typeof metadata[key] === "object") {
                tdValue.innerHTML = JSON.stringify(metadata[key], null, 2).replace(/\n/g, "<br>").replace(/ /g, "&nbsp;");
            } else {
                tdValue.textContent = metadata[key];
            }

            row.appendChild(tdKey);
            row.appendChild(tdValue);
            tbody.appendChild(row);
        }
    }

    tableElement.appendChild(tbody);
    popupContent.appendChild(tableElement);
    popupContent.appendChild(closeButton);
    popup.appendChild(popupContent);
    document.body.appendChild(popup);

    popup.style.display = 'block';
}

function showLoadingScreen() {
    hideLoadingScreen();  // Remove any existing loading screen
    
    const loadingScreen = document.createElement("div");
    loadingScreen.className = "loading-screen";
    loadingScreen.innerHTML = `<div class="spinner"></div>`;
    document.body.appendChild(loadingScreen);
    console.log("Loading screen shown");
    
    // Force a reflow to ensure the loading screen is rendered
    void loadingScreen.offsetWidth;
}

function hideLoadingScreen() {
    const loadingScreen = document.querySelector(".loading-screen");
    if (loadingScreen) {
        loadingScreen.style.opacity = '0';
        setTimeout(() => {
            loadingScreen.remove();
            console.log("Loading screen hidden");
        }, 300);  // Delay removal to allow for fade-out effect
    }
}

function checkLoadingScreen() {
    setTimeout(() => {
        const loadingScreen = document.querySelector('.loading-screen');
        if (!loadingScreen) {
            console.error('Loading screen not found in DOM');
        } else {
            console.log('Loading screen confirmed in DOM');
            console.log('Loading screen style:', window.getComputedStyle(loadingScreen));
        }
    }, 100);
}

// Call this after showLoadingScreen() in fetchAndShowMetadata
// showLoadingScreen();
// checkLoadingScreen();