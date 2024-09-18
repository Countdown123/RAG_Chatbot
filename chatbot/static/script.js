// static/script.js

// Check for the JWT token on page load
document.addEventListener("DOMContentLoaded", function () {
  var accessToken = localStorage.getItem("accessToken");
  if (!accessToken) {
    // Redirect to login page if not authenticated
    window.location.href = "/login.html";
  } else {
    // Initialize WebSocket connection with the token
    initializeWebSocket(accessToken);
    // Fetch and display the list of uploaded files
    fetchFileList();
  }
});

// Initialize WebSocket connection with token
function initializeWebSocket(token) {
  var ws = new WebSocket("ws://127.0.0.1:3939/ws?token=" + token);

  // Store the WebSocket instance for use in other functions
  window.ws = ws;

  ws.onopen = function () {
    console.log("Connected to the WebSocket server");
    initializeChat();  // Start a new chat on connection
  };

  ws.onmessage = function (event) {
    receiveMessage(event.data);  // Handle incoming messages
  };

  ws.onclose = function (event) {
    console.log("WebSocket closed:", event);
    // Optionally, handle reconnection or inform the user
  };

  ws.onerror = function (error) {
    console.error("WebSocket error:", error);
    alert("WebSocket connection failed. Please try again later.");
  };
}

var currentChatId = null;  // Track the current chat session
let chatHistories = {};    // Store chat histories for different sessions

// Initialize the chat after WebSocket is connected
function initializeChat() {
  if (!currentChatId) {
    startNewChat();  // Start a new chat when the WebSocket is open
  }
}

// Handle received messages from the WebSocket server
function receiveMessage(message) {
  console.log("Received message:", message);
  if (!currentChatId) {
    startNewChat();  // Start a new chat if none is active
  }
  addMessageToDisplay(message, "received");  // Display the message as 'received'
}

// Send message via WebSocket
function sendMessage(event) {
  event.preventDefault();
  var input = document.getElementById("messageText");
  var message = input.value.trim();  // Get the typed message
  
  if (message) {
    if (!currentChatId) {
      startNewChat();  // Ensure a new chat is started if none exists
    }
    window.ws.send(message);  // Send the message to the server
    addMessageToDisplay(message, "sent");  // Display the message as 'sent'
    input.value = "";  // Clear the input field
  }
  var messages = document.getElementById("messages");
  messages.scrollTop = messages.scrollHeight;  // Scroll to the bottom of the chat
}

// Append a message to the chat window and save it in chat history
function addMessageToDisplay(text, type) {
  var messages = document.getElementById("messages");
  var messageElement = document.createElement("li");
  messageElement.classList.add(type);  // Add either 'sent' or 'received' class
  messageElement.textContent = text;
  messages.appendChild(messageElement);
  messages.scrollTop = messages.scrollHeight;  // Scroll to the latest message

  // Save the message in the current chat history
  if (currentChatId) {
    if (!chatHistories[currentChatId]) {
      chatHistories[currentChatId] = [];
    }
    chatHistories[currentChatId].push({type: type, content: text});
  }

  updateChatHistoryEntry();  // Update the chat history list
}

// Create a new entry in the chat history list
function createNewChatHistoryEntry() {
  var chatHistory = document.getElementById("chatHistory");
  var historyItem = document.createElement("li");
  historyItem.classList.add("history-item");
  historyItem.id = `chat-${currentChatId}`;

  var thisChatId = currentChatId;  // Capture the chatId for this history item

  var timestamp = new Date().toLocaleString();
  historyItem.textContent = `Chat from ${timestamp}`;
  
  // When a chat history item is clicked, display that chat's content
  historyItem.onclick = function() {
    loadChatContent(thisChatId);
  };

  // Insert the new chat history at the top of the history list
  chatHistory.insertBefore(historyItem, chatHistory.firstChild);
}

// Update the display of an existing chat history entry
function updateChatHistoryEntry() {
  var historyItem = document.getElementById(`chat-${currentChatId}`);
  if (historyItem && chatHistories[currentChatId]) {
    historyItem.textContent = `Chat from ${new Date().toLocaleString()} (${chatHistories[currentChatId].length} messages)`;
  }
}

// Load a specific chat's content into the chat window (clear current chat and load selected)
function loadChatContent(chatId) {
  var messages = document.getElementById("messages");
  messages.innerHTML = '';  // Clear the current chat display

  // Display the messages of the selected chat
  if (chatHistories[chatId]) {
    chatHistories[chatId].forEach(msg => {
      var messageElement = document.createElement("li");
      messageElement.classList.add(msg.type);  // Add either 'sent' or 'received' class
      messageElement.textContent = msg.content;
      messages.appendChild(messageElement);
    });
  }

  // Set the current chat ID to the selected chat
  currentChatId = chatId;
}

// Start a new chat session and clear the chat window
function startNewChat() {
  document.getElementById("messages").innerHTML = '';  // Clear current chat display
  currentChatId = Date.now();  // Use the current timestamp as the new chat ID
  chatHistories[currentChatId] = [];  // Initialize a new history entry
  createNewChatHistoryEntry();  // Add the new chat to the history list
  document.getElementById("messageText").value = '';  // Clear the input field
  window.ws.send("new_chat:" + currentChatId);  // Notify the server of the new chat
  addMessageToDisplay("New chat started. How can I help you?", "received");  // Display a welcome message
}

// Handle file upload and display the Excel data in a popup
function uploadFile() {
  var accessToken = localStorage.getItem("accessToken");
  var formData = new FormData(document.getElementById("uploadForm"));
  fetch("/upload/", {
    method: "POST",
    headers: {
      "Authorization": "Bearer " + accessToken,
    },
    body: formData,
  })
    .then((response) => {
      if (response.status === 401) {
        // Token might have expired
        alert("Session expired. Please log in again.");
        localStorage.removeItem("accessToken");
        window.location.href = "/login.html";
      }
      return response.json();
    })
    .then((data) => {
      console.log("Upload response data:", data); // Debugging
      if (data.tableData) {
        showTablePopup(data.tableData);  // Show the uploaded table in a popup
        // Update the file list
        if (data.fileList) {
          updateFileListDisplay(data.fileList);
        }
      } else if (data.detail) {
        console.error("Error:", data.detail);
        alert("Error: " + data.detail);
      } else {
        console.error("Unexpected response:", data);
        alert("An unexpected error occurred during file upload.");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

// Show the table data (from the Excel file) in a popup modal
function showTablePopup(tableData) {
  const popup = document.createElement("div");
  popup.classList.add("modal");

  const popupContent = document.createElement("div");
  popupContent.classList.add("modal-content");
  popupContent.innerHTML = tableData;

  const closeButton = document.createElement("button");
  closeButton.textContent = "Close";
  closeButton.onclick = function() {
    document.body.removeChild(popup);
  };

  popup.appendChild(popupContent);
  popup.appendChild(closeButton);
  document.body.appendChild(popup);
}

// Fetch and display the list of uploaded files
function fetchFileList() {
  var accessToken = localStorage.getItem("accessToken");
  fetch("/files/", {
    method: "GET",
    headers: {
      "Authorization": "Bearer " + accessToken,
    },
  })
    .then((response) => {
      if (response.status === 401) {
        // Token might have expired
        alert("Session expired. Please log in again.");
        localStorage.removeItem("accessToken");
        window.location.href = "/login.html";
      }
      return response.json();
    })
    .then((data) => {
      console.log("Received data from /files/:", data); // Debugging
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
  var fileListContainer = document.getElementById("fileList");
  fileListContainer.innerHTML = ''; // Clear the current list

  fileList.forEach((file) => {
    var listItem = document.createElement("li");
    listItem.textContent = `${file.filename} (Uploaded on ${file.upload_time})`;
    fileListContainer.appendChild(listItem);
  });
}

// Logout functionality
document.getElementById("logoutBtn")?.addEventListener("click", function () {
  localStorage.removeItem("accessToken");
  window.location.href = "/login.html";
});
