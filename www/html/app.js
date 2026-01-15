// app.js

let socket = null;
const transcriptDiv = document.getElementById('transcript');
const connectionStatus = document.getElementById('connectionStatus');
const connectBtn = document.getElementById('connectBtn');
let isConnected = false;
let shouldReconnect = false;
let reconnectTimeout = null;
let heartbeatInterval = null;
const HEARTBEAT_TIME_MS = 30000; // 30 seconds

function getSelectedLanguages() {
    const checkboxes = document.querySelectorAll('input[name="lang"]:checked');
    const languages = Array.from(checkboxes).map(cb => cb.value);
    return languages.join(',');
}

function toggleConnection() {
    if (isConnected) {
        disconnect();
    } else {
        shouldReconnect = true; // User explicitly wants to connect
        connect();
    }
}

function connect() {
    // Prevent multiple connection attempts
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const languages = getSelectedLanguages();
    const wsUrl = `${protocol}//${host}/listener?lang=${languages}`;

    console.log(`Connecting to ${wsUrl}`);
    
    // UI Updates
    connectBtn.disabled = true;
    connectBtn.textContent = 'Connecting...';
    // Only show connecting message if transcript is empty to avoid cluttering during reconnect
    if (transcriptDiv.children.length === 0) {
        transcriptDiv.innerHTML = '<div id="connecting-msg" style="text-align: center; color: var(--text-secondary); padding-top: 2rem;">Connecting...</div>';
    } else {
         appendSystemMessage('Reconnecting...');
    }

    try {
        socket = new WebSocket(wsUrl);

        socket.onopen = function() {
            console.log('WebSocket Connected');
            isConnected = true;
            updateStatus(true);
            
            // Remove "Connecting..." placeholder if it exists
            const connectingMsg = document.getElementById('connecting-msg');
            if (connectingMsg) {
                connectingMsg.remove();
            }
            // If transcript was just "Connecting...", clear it
            if (transcriptDiv.innerHTML.includes('padding-top: 2rem;">Connecting...</div>')) {
                transcriptDiv.innerHTML = '';
            }

            appendSystemMessage('Connected to server.');
            startHeartbeat();
        };

        socket.onclose = function(event) {
            console.log('WebSocket Closed', event);
            cleanupConnection();

            if (shouldReconnect) {
                appendSystemMessage('Connection lost. Retrying in 3 seconds...');
                reconnectTimeout = setTimeout(connect, 3000);
            } else {
                if (event.wasClean) {
                     appendSystemMessage(`Disconnected cleanly.`);
                }
            }
        };

        socket.onerror = function(error) {
            console.error('WebSocket Error', error);
            // On error, onclose is usually called shortly after
        };

        socket.onmessage = function(event) {
            resetHeartbeat(); // Received message, so connection is alive
            try {
                // Determine if message is text or binary
                const data = JSON.parse(event.data);
                appendMessage(data);
            } catch (e) {
                console.error('Error parsing message:', e, event.data);
                appendSystemMessage('Received invalid data format.');
            }
        };

    } catch (e) {
        console.error('Connection failed:', e);
        cleanupConnection();
        if (shouldReconnect) {
             reconnectTimeout = setTimeout(connect, 3000);
        }
    }
}

function disconnect() {
    shouldReconnect = false; // User explicitly wants to disconnect
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }
    stopHeartbeat();
    if (socket) {
        socket.close();
    }
}

function cleanupConnection() {
    isConnected = false;
    socket = null;
    stopHeartbeat();
    updateStatus(false);
}

// --- Heartbeat Logic ---

function startHeartbeat() {
    stopHeartbeat(); // Ensure no duplicates
    // Set a timer to send a ping if no message received for HEARTBEAT_TIME_MS
    // Actually, usually we send ping periodically.
    // But requirement says: "If certain time no message received, please auto send heart beat"
    // So we reset the timer on every message.
    resetHeartbeat();
}

function stopHeartbeat() {
    if (heartbeatInterval) {
        clearTimeout(heartbeatInterval);
        heartbeatInterval = null;
    }
}

function resetHeartbeat() {
    stopHeartbeat();
    heartbeatInterval = setTimeout(() => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            console.log('Sending heartbeat...');
            socket.send('ping'); // Send heartbeat
            // Restart timer to keep checking
            resetHeartbeat();
        }
    }, HEARTBEAT_TIME_MS);
}

// -----------------------

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

    // Newest on top
    transcriptDiv.prepend(msgDiv);
    // Remove old messages if too many? (Optional optimization)
    if (transcriptDiv.children.length > 200) {
        transcriptDiv.lastElementChild.remove();
    }
}

function appendSystemMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message';
    msgDiv.style.borderLeftColor = 'var(--text-secondary)';
    msgDiv.style.color = 'var(--text-secondary)';
    msgDiv.innerHTML = `<i>${text}</i>`;
    
    // Newest on top
    transcriptDiv.prepend(msgDiv);
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

// Initialize timestamp for speaker form if needed
document.body.addEventListener('htmx:configRequest', function(evt) {
    if (evt.target.tagName === 'FORM' && evt.target.getAttribute('hx-post') === '/speak') {
        const tsInput = evt.target.querySelector('input[name="timestamp"]');
        if (tsInput) {
            tsInput.value = new Date().toISOString();
        }
    }
});
