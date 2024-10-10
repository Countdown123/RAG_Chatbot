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
    const ws = new WebSocket(`ws://127.0.0.1:3939/ws?token=${token}`);
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

    // Display a welcome message for the new chat
    addMessageToDisplay("New chat started. How can I help you?", "received");

    // Add the new chat to the history list
    const timestamp = new Date().toLocaleString();
    createNewChatHistoryEntry(currentChatId, timestamp);

    // Refresh the chat list
    fetchChatList();
    
    loadChatContent(currentChatId);
    // fetchChatList();

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
function uploadFile() {
    const accessToken = localStorage.getItem("accessToken");
    const uploadForm = document.getElementById("uploadForm");
    if (!uploadForm) return;

    const formData = new FormData(uploadForm);

    // Include the current chat ID in the upload request
    if (!currentChatId) {
        alert("Please start a chat before uploading files.");
        return;
    }
    formData.append("chat_id", currentChatId);

    fetch("/upload/", {
        method: "POST",
        headers: { 
            "Authorization": `Bearer ${accessToken}`
            // Note: Do not set 'Content-Type' header when sending FormData
        },
        body: formData,
    })
        .then(response => handleFileUploadResponse(response))
        .catch(error => console.error("File upload error:", error));
}

// Handle the file upload response
function handleFileUploadResponse(response) {
    if (response.status === 401) {
        alert("Session expired. Please log in again.");
        localStorage.removeItem("accessToken");
        redirectToLogin();
    }
    return response.json().then((data) => {
        console.log("Upload response data:", data); // Debugging
        if (data.fileList) {
            updateFileListDisplay(data.fileList);
            // Optionally, display a success message
            alert("File uploaded successfully!");
            // Clear the upload form
            document.getElementById("uploadForm").reset();
        } else if (data.detail) {
            console.error("Error:", data.detail);
            alert("Error: " + data.detail);
        } else {
            console.error("Unexpected response:", data);
            alert("An unexpected error occurred during file upload.");
        }
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

// Create a list item for each file
function createFileListItem(file) {
    const listItem = document.createElement("li");
    listItem.textContent = `${file.filename} (Uploaded on ${file.upload_time})`;
    listItem.style.cursor = 'pointer';
    listItem.onclick = () => fetchFileData(file.id, file.filename);

    // PDF 파일일 경우 클릭 시 메타데이터 조회
    if (file.filename.endsWith(".pdf")) {
        listItem.onclick = () => fetchAndShowMetadata(file.id, file.filename);
    } else {
        listItem.onclick = () => fetchFileData(file.id, file.filename); // 기존 파일 데이터 조회 함수
    }

    return listItem;
}

// PDF 파일 클릭 시 메타데이터 조회 및 표시
function fetchAndShowMetadata(fileId, filename) {
    const accessToken = localStorage.getItem("accessToken");

    console.log("Fetching metadata for fileId:", fileId); // fileId 출력 확인

    // 메타데이터 테이블 초기화
    clearMetadataTable();

    // 서버에서 메타데이터 가져오기
    fetch(`/metadata/${fileId}`, {
        method: "GET",
        headers: {
            "Authorization": `Bearer ${accessToken}`,
        },
    })
        .then(response => {
            if (response.status === 401) {
                alert("Session expired. Please log in again.");
                localStorage.removeItem("accessToken");
                redirectToLogin();
                throw new Error("Unauthorized");
            }
            if (!response.ok) {
                throw new Error("Failed to fetch metadata.");
            }
            return response.json();
        })
        .then(data => {
            if (data.metadata) {
                showMetadataTable(data.metadata);  // 메타데이터 테이블로 표시
            } else {
                alert("No metadata found for this file.");
            }
        })
        .catch(error => {
            console.error("Error fetching metadata:", error);
            alert("Error fetching metadata. Please try again.");
        });
}


// Fetch file data when a file is clicked
function fetchFileData(fileId, filename) {
    const accessToken = localStorage.getItem("accessToken");
    fetch(`/file/${fileId}`, {
        method: "GET",
        headers: {
            "Authorization": `Bearer ${accessToken}`,
        },
    })
        .then(response => {
            if (response.status === 401) {
                alert("Session expired. Please log in again.");
                localStorage.removeItem("accessToken");
                redirectToLogin();
                throw new Error("Unauthorized");
            }
            if (!response.ok) {
                throw new Error("Failed to fetch file data.");
            }
            return response.json();
        })
        .then(data => {
            if (data.columns && data.data) {
                displayTable(data.columns, data.data, filename);
            } else if (data.message) {
                alert(data.message);
            } else {
                alert("An error occurred while retrieving the file data.");
            }
        })
        .catch(error => {
            console.error("Error fetching file data:", error);
            alert("Error fetching file data. Please try again.");
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

// 파일 업로드 후 메타데이터 테이블 표시
document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('chat_id', currentChatId);  

    // JWT 토큰 가져오기
    const accessToken = localStorage.getItem('accessToken');

    const response = await fetch('/upload_pdf/', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${accessToken}`, 
        },
        body: formData
    });

    const data = await response.json();

    if (response.ok) {
        // 기존 메타데이터 초기화
        clearMetadataTable();  // 기존 메타데이터 초기화 함수 호출
    
        // 새로운 메타데이터 바로 표시
        showMetadataTable(data.metadata);  // 동적으로 받은 메타데이터 표시
    } else {
        console.error("Error uploading PDF:", data);
        alert(`Error: ${data.detail}`);
    }
    
});

// 메타데이터 테이블 초기화 함수
function clearMetadataTable() {
    const existingTable = document.getElementById("metadataTable");
    if (existingTable) {
        existingTable.remove();  // 기존 테이블을 제거
    }

    // 기존 모달 제거
    const existingModal = document.querySelector('.modal');
    if (existingModal) {
        existingModal.remove();  
    }
}

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
    tableElement.id = "metadataTable";  // 새 메타데이터 테이블

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

    // 메타데이터를 테이블에 추가
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

var fileNo = 0;
var filesArr = [];

/* 첨부파일 추가 */
function addFile(obj) {
    var maxPdfFileCnt = 5;  // PDF 파일 최대 개수
    var attPdfFileCnt = filesArr.filter(file => file.type === 'application/pdf').length; // 기존 추가된 PDF 파일 개수
    var remainPdfFileCnt = maxPdfFileCnt - attPdfFileCnt;  // 추가로 첨부가능한 PDF 파일 개수
    var curFileCnt = obj.files.length;  // 현재 선택된 첨부파일 개수

    for (var i = 0; i < curFileCnt; i++) {
        const file = obj.files[i];

        // 첨부파일 검증
        if (validation(file)) {
            if (file.type === 'application/pdf') {
                if (attPdfFileCnt >= maxPdfFileCnt) {
                    alert("PDF 파일은 최대 " + maxPdfFileCnt + "개 까지 첨부 가능합니다.");
                    break;
                }
                attPdfFileCnt++;
            } else if (file.type === 'text/csv' || file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
                if (filesArr.some(f => f.type === 'text/csv' || f.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')) {
                    alert("CSV 또는 XLSX 파일은 하나만 첨부 가능합니다.");
                    continue;
                }
            } else {
                alert("허용되지 않은 파일 형식입니다.");
                continue;
            }

            // 파일 배열에 담기
            var reader = new FileReader();
            reader.onload = function () {
                filesArr.push(file);
            };
            reader.readAsDataURL(file);

            // 목록 추가
            let htmlData = '';
            htmlData += '<div id="file' + fileNo + '" class="filebox">';
            htmlData += '   <p class="name">' + file.name + '</p>';
            htmlData += '   <a class="delete" onclick="deleteFile(' + fileNo + ');"><i class="far fa-minus-square"></i></a>';
            htmlData += '</div>';
            $('.file-list').append(htmlData);
            fileNo++;
        }
    }

    // 초기화
    document.querySelector("input[type=file]").value = "";
}

/* 첨부파일 검증 */
function validation(file) {
    const fileTypes = ['application/pdf', 'text/csv', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    if (file.name.length > 100) {
        alert("파일명이 100자 이상인 파일은 제외되었습니다.");
        return false;
    } else if (file.size > (100 * 1024 * 1024)) {
        alert("최대 파일 용량인 100MB를 초과한 파일은 제외되었습니다.");
        return false;
    } else if (file.name.lastIndexOf('.') == -1) {
        alert("확장자가 없는 파일은 제외되었습니다.");
        return false;
    } else if (!fileTypes.includes(file.type)) {
        alert("첨부가 불가능한 파일은 제외되었습니다.");
        return false;
    } else {
        return true;
    }
}

/* 첨부파일 삭제 */
function deleteFile(num) {
    document.querySelector("#file" + num).remove();
    filesArr[num].is_delete = true;
}

/* 폼 전송 */
function submitForm() {
    // 폼데이터 담기
    var form = document.querySelector("form");
    var formData = new FormData(form);
    for (var i = 0; i < filesArr.length; i++) {
        // 삭제되지 않은 파일만 폼데이터에 담기
        if (!filesArr[i].is_delete) {
            formData.append("attach_file", filesArr[i]);
        }
    }

    $.ajax({
        method: 'POST',
        url: '/register',
        dataType: 'json',
        data: formData,
        async: true,
        timeout: 30000,
        cache: false,
        headers: { 'cache-control': 'no-cache', 'pragma': 'no-cache' },
        success: function () {
            alert("파일업로드 성공");
        },
        error: function (xhr, desc, err) {
            alert('에러가 발생 하였습니다.');
            return;
        }
    });
}
