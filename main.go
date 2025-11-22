package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/gorilla/websocket"
)

const (
	PythonServiceURL = "http://127.0.0.1:8000/step"
	Port             = ":8080"
)

type ControlParams struct {
	CmdVx  float64 `json:"cmd_vx"`
	CmdVy  float64 `json:"cmd_vy"`
	CmdVz  float64 `json:"cmd_vz"`
	DriftX float64 `json:"drift_x"`
	DriftY float64 `json:"drift_y"`
	DriftZ float64 `json:"drift_z"`
	Noise  float64 `json:"noise"`
}

type ResponseData map[string]interface{}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func main() {
	fs := http.FileServer(http.Dir("./static"))
	http.Handle("/", fs)
	http.HandleFunc("/ws", handleWebSocket)

	fmt.Printf("Server started at http://localhost%s\n", Port)
	log.Fatal(http.ListenAndServe(Port, nil))
}

func handleWebSocket(w http.ResponseWriter, r *http.Request) {
	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println(err)
		return
	}
	defer ws.Close()

	for {
		_, message, err := ws.ReadMessage()
		if err != nil {
			break
		}

		resp, err := http.Post(PythonServiceURL, "application/json", bytes.NewBuffer(message))
		if err != nil {
			log.Println("Python service unreachable:", err)
			continue
		}

		var data ResponseData
		if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
			log.Println("Decode error:", err)
			resp.Body.Close()
			continue
		}
		resp.Body.Close()

		if err := ws.WriteJSON(data); err != nil {
			break
		}
	}
}
