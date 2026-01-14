package main

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	// Allow all origins for now
	CheckOrigin: func(r *http.Request) bool { return true },
}

// Wrapper for websocket connection to allow simple Interface usage in Client if needed,
// but for now we just hold the *websocket.Conn
type WebSocketConn = websocket.Conn

// HandleSpeak processes the POST /speak request
func HandleSpeak(hub *Hub, w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var payload SpeakPayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Basic validation
	if payload.Text == "" {
		// Just log, maybe keep alive?
	}

	// Send to Hub
	hub.Broadcast <- &payload

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

// HandleListener processes the GET /listener request (WebSocket)
func HandleListener(hub *Hub, w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Upgrade error:", err)
		return
	}

	// Parse languages from query param: ?lang=cn,en
	langParam := r.URL.Query().Get("lang")
	var languages []string
	if langParam != "" {
		parts := strings.Split(langParam, ",")
		for _, p := range parts {
			languages = append(languages, strings.TrimSpace(p))
		}
	}

	client := &Client{
		Hub:       hub,
		Conn:      conn,
		Send:      make(chan []byte, 256),
		Languages: languages,
	}

	client.Hub.Register <- client

	// Start two goroutines for the client: read (pump) and write (pump)
	go client.writePump()
	go client.readPump()
}

// readPump pumps messages from the websocket connection to the hub.
// For a listener-only client, this primarily handles close messages/pings.
func (c *Client) readPump() {
	defer func() {
		c.Hub.Unregister <- c
		c.Conn.Close()
	}()
	for {
		_, _, err := c.Conn.ReadMessage()
		if err != nil {
			break
		}
	}
}

// writePump pumps messages from the hub to the websocket connection.
func (c *Client) writePump() {
	defer func() {
		c.Conn.Close()
	}()
	for {
		select {
		case message, ok := <-c.Send:
			if !ok {
				c.Conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.Conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			if err := w.Close(); err != nil {
				return
			}
		}
	}
}
