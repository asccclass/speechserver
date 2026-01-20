// app.js

let socket = null;
const transDiv = document.getElementById('transcript-translation');
const sourceDiv = document.getElementById('transcript-source');
const connectionStatus = document.getElementById('connectionStatus');
const connectBtn = document.getElementById('connectBtn');
let isConnected = false;
let shouldReconnect = false;
let reconnectTimeout = null;
let heartbeatInterval = null;
const HEARTBEAT_TIME_MS = 30000; // 30 seconds

// Modal Controls
function openSpeakerModal() {
    document.getElementById('speakerModal').classList.remove('hidden');
}

function closeSpeakerModal() {
    document.getElementById('speakerModal').classList.add('hidden');
}

// Close modal when clicking outside
window.onclick = function (event) {
    const modal = document.getElementById('speakerModal');
    if (event.target === modal) {
        closeSpeakerModal();
    }
}

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

    // Clear placeholders if they exist
    clearPlaceholders();

    appendSystemMessage('Connecting...');

    try {
        socket = new WebSocket(wsUrl);
        socket.onopen = function () {
            console.log('WebSocket Connected');
            isConnected = true;
            updateStatus(true);
            appendSystemMessage('Connected to server.');
            startHeartbeat();
        };

        socket.onclose = function (event) {
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

        socket.onerror = function (error) {
            console.error('WebSocket Error', error);
            // On error, onclose is usually called shortly after
        };

        socket.onmessage = function (event) {
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
    stopHeartbeat();
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

function clearPlaceholders() {
    const placeholders = document.querySelectorAll('.placeholder-text');
    placeholders.forEach(el => el.remove());
}

function appendMessage(data) {
    // data structure based on hub.go SpeakPayload:
    // { rooms, user, text, timestamp, language, translation }

    const timestamp = data.timestamp || new Date().toLocaleTimeString();
    const langTag = data.language ? `[${data.language.toUpperCase()}] ` : '';
    const user = data.user || 'Unknown';

    // 1. Handle Translation Column (Left)
    if (data.translation) {
        const transHtml = `<div class="message-content">${escapeHtml(data.translation)}</div>`;
        const transMeta = `
            <div class="message-meta">
                <span>${timestamp}</span>
            </div>`;
        addEntryToColumn(transDiv, transHtml, transMeta);
    } else {
        // If no translation, add a spacer or duplicate? User requested "Translation Result".
        // If no translation logic exists yet, maybe just show original?
        // For now, if translation is empty, we show a spacer or nothing.
        // But checking requirements: "Translation result with original text ... simultaneously appear".
        // If there is no translation (e.g. speaking same language), usually we prefer to see the text.
        // I will display the original text in Translation column if translation is missing,
        // but visually styled differently or just the text.
        // Actually, let's keep it empty or show a placeholder if strictly separating.
        // However, standard translation UI usually shows Source -> Target.
        // If Source == Target, it might not send translation.
        // Let's assume we simply don't print to left if no translation.
        // BUT, to keep alignment, we might want to print a "blank" block?
        // No, chat interfaces usually behave like independent streams or linked bubbles.
        // Let's just add to Source column if no translation.
        // However, the request implies they come together.
        // Let's assume they are independent logs.
    }

    // 2. Handle Source Column (Right)
    if (data.text) {
        const sourceHtml = `<div class="message-content">${escapeHtml(data.text)}</div>`;
        const sourceMeta = `
            <div class="message-meta">
                <span>${langTag}${user}</span>
                <span>${timestamp}</span>
            </div>`;
        addEntryToColumn(sourceDiv, sourceHtml, sourceMeta);
    }
}

function addEntryToColumn(container, contentHtml, metaHtml) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message';
    msgDiv.innerHTML = `${metaHtml}${contentHtml}`;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;

    // Limit history
    if (container.children.length > 200) {
        container.firstElementChild.remove();
    }
}

function appendSystemMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message';
    msgDiv.style.borderLeftColor = 'var(--text-secondary)';
    msgDiv.style.color = 'var(--text-secondary)';
    msgDiv.innerHTML = `<i>${text}</i>`;

    // Append to BOTH columns for visibility
    [transDiv, sourceDiv].forEach(div => {
        if (div) {
            // Clone for the second append
            const clone = msgDiv.cloneNode(true);
            div.appendChild(clone);
            div.scrollTop = div.scrollHeight;
        }
    });
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
document.body.addEventListener('htmx:configRequest', function (evt) {
    if (evt.target.tagName === 'FORM' && evt.target.getAttribute('hx-post') === '/speaker') {
        const tsInput = evt.target.querySelector('input[name="timestamp"]');
        if (tsInput) {
            tsInput.value = new Date().toISOString();
        }
    }
});
