package main

import (
	"encoding/json"
	"strings"
	"sync"
)

// SpeakPayload represents the JSON received from /speak
type SpeakPayload struct {
	Rooms     string `json:"rooms"`
	User      string `json:"user"`
	Text      string `json:"text"`
	Timestamp string `json:"timestamp"`
	// Language is optional. If not provided, it might default or match all.
	Language string `json:"language,omitempty"`
}

// Client represents a connected WebSocket client
type Client struct {
	Hub       *Hub
	Conn      *WebSocketConn // Interface to allow mocking or specific library wrapper
	Send      chan []byte
	Languages []string // Languages this client is interested in
}

// Hub maintains the set of active clients and broadcasts messages to clients.
type Hub struct {
	Clients    map[*Client]bool
	Broadcast  chan *SpeakPayload
	Register   chan *Client
	Unregister chan *Client
	History    []*SpeakPayload // Buffer for recent messages
	mu         sync.Mutex
}

func NewHub() *Hub {
	return &Hub{
		Broadcast:  make(chan *SpeakPayload),
		Register:   make(chan *Client),
		Unregister: make(chan *Client),
		Clients:    make(map[*Client]bool),
		History:    make([]*SpeakPayload, 0),
	}
}

func (h *Hub) Run() {
	for {
		select {
		case client := <-h.Register:
			h.mu.Lock()
			h.Clients[client] = true
			h.mu.Unlock()
		case client := <-h.Unregister:
			h.mu.Lock()
			if _, ok := h.Clients[client]; ok {
				delete(h.Clients, client)
				close(client.Send)
			}
			h.mu.Unlock()
		case message := <-h.Broadcast:
			// Buffer the message
			h.mu.Lock()
			h.History = append(h.History, message)
			if len(h.History) > 100 { // Keep last 100 messages
				h.History = h.History[1:]
			}
			clients := h.getAllClients() // Get copy to avoid deadlock during send
			h.mu.Unlock()

			encoded, _ := json.Marshal(message)

			for client := range clients {
				// Filter based on language
				if h.shouldSend(client, message) {
					select {
					case client.Send <- encoded:
					default:
						h.Unregister <- client
					}
				}
			}
		}
	}
}

func (h *Hub) getAllClients() map[*Client]bool {
	// Assumes caller holds lock or is safe
	// Actually for safety, let's just make a shallow copy if we were doing this carefully
	// But inside the select loop, we have exclusive access unless we used a mutex inside the loop, 
	// which we did. The mutex protects map access.
	// Since we are inside the 'case', we are the only one modifying the map via Register/Unregister channels?
	// No, Register/Unregister handlers run in this same goroutine.
	// So we can iterate directly.
	// Wait, we unlocked mu before the loop. So we need to copy.
	
	// Re-lock is needed? No, we are in the single Hub.Run goroutine.
	// Register/Unregister are channels. The modification happens inside this loop.
	// So h.Clients is safe to read here IF we didn't unlock.
	// But I unlocked to allow history append? No, I unlocked after history append.
	
	// Let's refine locking.
	// The Hub.Run is single threaded. The only way h.Clients changes is via the channels.
	// So we don't strictly need a mutex for h.Clients *inside this loop* if only this loop touches it.
	// But other reads (like stats) might need it.
	// Let's copy for safety during iteration especially if send blocks (it sends to channel, so non-blocking usually, but 'default' handles it).
	
	copies := make(map[*Client]bool)
	for k, v := range h.Clients {
		copies[k] = v
	}
	return copies
}

func (h *Hub) shouldSend(client *Client, msg *SpeakPayload) bool {
	if len(client.Languages) == 0 {
		return true // Send everything if no filter specified
	}
	msgLang := strings.ToLower(msg.Language)
	if msgLang == "" {
		return true // Message has no language, send to all
	}
	for _, l := range client.Languages {
		if strings.ToLower(l) == msgLang {
			return true
		}
	}
	return false
}
