// WebSocket connection
var ws = new WebSocket("ws://127.0.0.1:3939/ws");

var currentChatId = null;  // Track the current chat session
let chatHistories = {};    // Store chat histories for different sessions

ws.onopen = function () {
  console.log("Connected to the WebSocket server");
  initializeChat();  // Start a new chat on connection
};

ws.onmessage = function (event) {
  receiveMessage(event.data);  // Handle incoming messages
};

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
    ws.send(message);  // Send the message to the server
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
    loadChatContent(thisChatId);  // Use the captured chatId
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
  ws.send("new_chat:" + currentChatId);  // Notify the server of the new chat
  addMessageToDisplay("New chat started. How can I help you?", "received");  // Display a welcome message
}

// Handle file upload and display the Excel data in a popup
function uploadFile() {
  var formData = new FormData(document.getElementById("uploadForm"));
  fetch("/upload/", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      showTablePopup(data.tableData);  // Show the uploaded table in a popup
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

// Initialize the chat on page load
function initializeChat() {
  if (!currentChatId) {
    startNewChat();  // Start a new chat when the page loads
  }
}
