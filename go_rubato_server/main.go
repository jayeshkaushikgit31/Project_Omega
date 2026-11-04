package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// Input structure for the Python client
type Request struct {
	Operation string    `json:"operation"`
	Data      []float64 `json:"data"`
	TileSize  int       `json:"tile_size"`
}

// Output structure for the Python client
type Response struct {
	Data       []float64 `json:"data"`
	Bandwidth  int       `json:"bandwidth"`
	ServerTime float64   `json:"server_time"`
}

func main() {
	// Read JSON input from stdin (Python will send this)
	decoder := json.NewDecoder(bufio.NewReader(os.Stdin))
	var req Request
	if err := decoder.Decode(&req); err != nil {
		fmt.Fprintf(os.Stderr, "Error decoding input: %v\n", err)
		os.Exit(1)
	}

	// Measure server processing time
	startTime := time.Now()

	// Simulate the Rubato Transciphering & Homomorphic Operation
	processedData := make([]float64, len(req.Data))
	if req.Operation == "pixel" {
		for i, v := range req.Data {
			processedData[i] = 255.0 - v
		}
	} else {
		// For convolution, since this is a simplified Go test server,
		// we do a placeholder operation. In the real MT-PRO repo,
		// this is where the im2col CKKS math happens.
		for i, v := range req.Data {
			processedData[i] = v * 0.95
		}
	}

	// Simulate network bandwidth (8 bytes per float64 + 16 bytes overhead)
	simulatedBandwidth := len(req.Data) * 8

	// Calculate elapsed time
	elapsed := time.Since(startTime).Seconds()

	// Prepare the response
	resp := Response{
		Data:       processedData,
		Bandwidth:  simulatedBandwidth,
		ServerTime: elapsed,
	}

	// Send JSON response back to Python
	encoder := json.NewEncoder(os.Stdout)
	if err := encoder.Encode(resp); err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding output: %v\n", err)
		os.Exit(1)
	}
}