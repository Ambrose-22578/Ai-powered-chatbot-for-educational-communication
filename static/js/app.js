document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const userInputField = document.getElementById('user-input');
    const userInput = userInputField.value.trim();
    if (userInput === '') return;

    displayMessage(userInput, 'user');
    userInputField.value = '';

    fetch('/get_response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: userInput })  // Ensure key matches Flask
    })
    .then(response => response.json())
    .then(data => {
        displayMessage(data.bot_response, 'bot');  // Match JSON key returned by Flask
    })
    .catch(error => {
        console.error('Error:', error);
        displayMessage("Error: Unable to connect to the server.", 'bot');
    });
}

function displayMessage(message, sender) {
    const chatWindow = document.getElementById('chat-window');
    const messageElem = document.createElement('div');
    messageElem.classList.add('message', sender);
    messageElem.textContent = message;
    chatWindow.appendChild(messageElem);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}
