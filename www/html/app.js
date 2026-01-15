// app.js

let socket = null;
const transcriptDiv = document.getElementById('transcript');
const connectionStatus = document.getElementById('connectionStatus');
const connectBtn = document.getElementById('connectBtn');
let isConnected = false;

function getSelectedLanguages() {
    const checkboxes = document.querySelectorAll('input[name="lang"]:checked');
    const languages = Array.from(checkboxes).map(cb => cb.value);
    return languages.join(',');
}

function toggleConnection() {
    if (isConnected) {
        disconnect();
    } else {
        connect();
    }
}

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const languages = getSelectedLanguages();
    const wsUrl = `${protocol}//${host}/listener?lang=${languages}`;

    console.log(`Connecting to ${wsUrl}`);
    
    // UI Updates
    connectBtn.disabled = true;
    connectBtn.textContent = 'Connecting...';
    transcriptDiv.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding-top: 2rem;">Connecting...</div>';

    try {
        socket = new WebSocket(wsUrl);

        socket.onopen = function() {
            console.log('WebSocket Connected');
            isConnected = true;
            updateStatus(true);
            transcriptDiv.innerHTML = ''; // Clear "Connecting..." message
            appendSystemMessage('Connected to server.');
        };

        socket.onclose = function(event) {
            console.log('WebSocket Closed', event);
            isConnected = false;
            updateStatus(false);
            socket = null;
            if (event.wasClean) {
                 appendSystemMessage(`Disconnected cleanly.`);
            } else {
                 appendSystemMessage(`Connection lost.`);
            }
        };

        socket.onerror = function(error) {
            console.error('WebSocket Error', error);
            // On error, onclose is usually called shortly after
        };

        socket.onmessage = function(event) {
            try {
                // Determine if message is text or binary
                // The handler sends TextMessage, so it should be string
                // But let's handle if it sends valid JSON string
                // Note: The Go code sends base64 encoded JSON if it was using some specific library, 
                // but checking handlers.go: c.Conn.WriteMessage(websocket.TextMessage, message)
                // where message comes from json.Marshal(message)
                
                // However, the JSON from Go might be just the JSON object string.
                const data = JSON.parse(event.data);
                appendMessage(data);
            } catch (e) {
                console.error('Error parsing message:', e, event.data);
                appendSystemMessage('Received invalid data format.');
            }
        };

    } catch (e) {
        console.error('Connection failed:', e);
        isConnected = false;
        updateStatus(false);
    }
}

function disconnect() {
    if (socket) {
        socket.close();
    }
}

function updateStatus(connected) {
    if (connected) {
        connectionStatus.classList.add('connected');
        connectionStatus.querySelector('span').textContent = 'Connected';
        connectBtn.textContent = 'Disconnect';
        connectBtn.classList.remove('primary'); 
        connectBtn.style.backgroundColor = '#ef4444'; // Red for disconnect
        connectBtn.disabled = false;
        
        // Disable checkboxes while connected
        document.querySelectorAll('input[name="lang"]').forEach(cb => cb.disabled = true);
    } else {
        connectionStatus.classList.remove('connected');
        connectionStatus.querySelector('span').textContent = 'Disconnected';
        connectBtn.textContent = 'Connect';
        connectBtn.style.backgroundColor = ''; // Reset to default
        connectBtn.disabled = false;

        // Enable checkboxes
        document.querySelectorAll('input[name="lang"]').forEach(cb => cb.disabled = false);
    }
}

function appendMessage(data) {
    // data structure based on hub.go SpeakPayload:
    // { rooms, user, text, timestamp, language }

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message';

    const timestamp = data.timestamp || new Date().toLocaleTimeString();
    const langTag = data.language ? `[${data.language.toUpperCase()}] ` : '';

    msgDiv.innerHTML = `
        <div class="message-meta">
            <span>${langTag}${data.user || 'Unknown'}</span>
            <span>${timestamp}</span>
        </div>
        <div class="message-content">${escapeHtml(data.text)}</div>
    `;

    transcriptDiv.appendChild(msgDiv);
    
    // Auto-scroll to bottom
    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
}

function appendSystemMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message';
    msgDiv.style.borderLeftColor = 'var(--text-secondary)';
    msgDiv.style.color = 'var(--text-secondary)';
    msgDiv.innerHTML = `<i>${text}</i>`;
    transcriptDiv.appendChild(msgDiv);
    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
}

function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Initialize timestamp for speaker form if needed (though backend might handle it better)
// But let's just leave it empty and let backend handling if it wants, 
// or fill it client side.
document.body.addEventListener('htmx:configRequest', function(evt) {
    // Add timestamp to form data if not present?
    // The form has hidden input named 'timestamp'.
    // We can populate it before submit.
    if (evt.target.tagName === 'FORM' && evt.target.getAttribute('hx-post') === '/speak') {
        // Find the input
        const tsInput = evt.target.querySelector('input[name="timestamp"]');
        if (tsInput) {
            tsInput.value = new Date().toISOString();
        }
    }
});
