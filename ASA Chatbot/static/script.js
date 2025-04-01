function getResponse() {
    var userText = document.getElementById("userInput").value;
    if (userText.trim() === "") return;

    var chatbox = document.getElementById("chatbox");

    var userMessage = <p><strong>User:</strong> ${userText}</p>;
    chatbox.innerHTML += userMessage;

    var typingMessage = <p><strong>ASA:</strong> <span id="typing">...</span></p>;
    chatbox.innerHTML += typingMessage;

    fetch(/get?msg=${encodeURIComponent(userText)})
        .then(response => response.json())
        .then(data => {
            document.getElementById("typing").innerText = data.response;
        });

    chatbox.scrollTop = chatbox.scrollHeight;
    document.getElementById("userInput").value = "";
}