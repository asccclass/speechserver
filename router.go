// router.go
package main

import (
	"net/http"

	SherryServer "github.com/asccclass/sherryserver"
)

// Create your Router function
func NewRouter(srv *SherryServer.Server, documentRoot string) *http.ServeMux {
	router := http.NewServeMux()

	// Static File server
	staticfileserver := SherryServer.StaticFileServer{documentRoot, "index.html"}
	staticfileserver.AddRouter(router)

	// Hub for WebSockets
	hub := NewHub()
	go hub.Run()

	// API Endpoints
	router.HandleFunc("/speak", func(w http.ResponseWriter, r *http.Request) {
		HandleSpeak(hub, w, r)
	})
	router.HandleFunc("/listener", func(w http.ResponseWriter, r *http.Request) {
		HandleListener(hub, w, r)
	})

	return router
}
