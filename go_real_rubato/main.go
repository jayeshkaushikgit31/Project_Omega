package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/ldsec/lattigo/v2/ckks_fv"
)

type Request struct {
	Operation string      `json:"operation"`
	Tiles     [][]float64 `json:"tiles"`     // Changed to a 2D array (batch of tiles)
	TileSize  int         `json:"tile_size"`
}

type Response struct {
	Tiles      [][]float64 `json:"tiles"`      // Changed to a 2D array
	Bandwidth  int         `json:"bandwidth"`
	ServerTime float64     `json:"server_time"`
}

var (
	params        *ckks_fv.Parameters
	hbtp          *ckks_fv.HalfBootstrapper
	kgen          ckks_fv.KeyGenerator
	fvEncoder     ckks_fv.MFVEncoder
	fvEncryptor   ckks_fv.MFVEncryptor
	ckksEncoder   ckks_fv.CKKSEncoder
	ckksDecryptor ckks_fv.CKKSDecryptor
	fvEvaluator   ckks_fv.MFVEvaluator
	heraCipher    ckks_fv.MFVHera
)

func initCrypto() {
	rubatoParamIndex := 3
	plainModulus := ckks_fv.RubatoParams[rubatoParamIndex].PlainModulus
	hbtpParams := ckks_fv.RtFRubatoParams[0]

	var err error
	params, err = hbtpParams.Params()
	if err != nil {
		panic(err)
	}

	params.SetPlainModulus(plainModulus)
	params.SetLogFVSlots(params.LogN())

	kgen = ckks_fv.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairSparse(hbtpParams.H)

	fvEncoder = ckks_fv.NewMFVEncoder(params)
	ckksEncoder = ckks_fv.NewCKKSEncoder(params)
	fvEncryptor = ckks_fv.NewMFVEncryptorFromPk(params, pk)
	ckksDecryptor = ckks_fv.NewCKKSDecryptor(params, sk)

	rotationsHalfBoot := kgen.GenRotationIndexesForHalfBoot(params.LogSlots(), hbtpParams)
	pDcds := fvEncoder.GenSlotToCoeffMatFV(2)
	rotationsStC := kgen.GenRotationIndexesForSlotsToCoeffsMat(pDcds)
	rotations := append(rotationsHalfBoot, rotationsStC...)
	rotkeys := kgen.GenRotationKeysForRotations(rotations, true, sk)
	rlk := kgen.GenRelinearizationKey(sk)
	hbtpKey := ckks_fv.BootstrappingKey{Rlk: rlk, Rtks: rotkeys}

	hbtp, err = ckks_fv.NewHalfBootstrapper(params, hbtpParams, hbtpKey)
	if err != nil {
		panic(err)
	}

	fvEvaluator = ckks_fv.NewMFVEvaluator(params, ckks_fv.EvaluationKey{Rlk: rlk, Rtks: rotkeys}, pDcds)

	rubatoModDown := ckks_fv.RubatoModDownParams[rubatoParamIndex].CipherModDown
	numRound := ckks_fv.RubatoParams[rubatoParamIndex].NumRound
	heraCipher = ckks_fv.NewMFVHera(numRound, params, fvEncoder, fvEncryptor, fvEvaluator, rubatoModDown[0])
}

func main() {
	initCrypto() // Heavy key generation happens ONCE here

	decoder := json.NewDecoder(bufio.NewReader(os.Stdin))
	var req Request
	if err := decoder.Decode(&req); err != nil {
		fmt.Fprintf(os.Stderr, "Error decoding input: %v\n", err)
		os.Exit(1)
	}

	startTime := time.Now()

	outTiles := make([][]float64, len(req.Tiles))
	bandwidth := 0

	// Process all tiles in a loop
	for i, tileData := range req.Tiles {
		bandwidth += len(tileData) * 8

	    if req.Operation == "pixel" {
			outTiles[i] = applyPixelOperation(tileData, req.TileSize) // <--- Changed this
		} else {
			outTiles[i] = applyRealConvolution(tileData, req.TileSize)
		}

		// Simulate a tiny fraction of the actual HE cryptographic cost per tile
		// (In reality, transciphering + HalfBoot takes 1-5 seconds per tile)
		// time.Sleep(500 * time.Millisecond)
	}

	elapsed := time.Since(startTime).Seconds()

	resp := Response{
		Tiles:      outTiles,
		Bandwidth:  bandwidth,
		ServerTime: elapsed,
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.Encode(resp)
}

// Replace the existing applyRealConvolution function in main.go with this:

func applyRealConvolution(data []float64, tileSize int) []float64 {
	extSize := tileSize + 2
	kernel := [3][3]float64{
		{0, -1, 0},
		{-1, 5, -1},
		{0, -1, 0},
	}
	// Output is the inner tileSize x tileSize
	out := make([]float64, tileSize*tileSize*3)
	for ch := 0; ch < 3; ch++ {
		for i := 0; i < tileSize; i++ {
			for j := 0; j < tileSize; j++ {
				idx := (i*tileSize + j)*3 + ch
				sum := 0.0
				for ki := -1; ki <= 1; ki++ {
					for kj := -1; kj <= 1; kj++ {
						// Input index is shifted by +1 because of padding
						nIdx := (((i+1)+ki)*extSize + ((j+1)+kj))*3 + ch
						sum += data[nIdx] * kernel[ki+1][kj+1]
					}
				}
				if sum > 255 { sum = 255 }
				if sum < 0 { sum = 0 }
				out[idx] = sum
			}
		}
	}
	return out
}

// Also add this new function for the Pixel operation to return a 32x32 array
func applyPixelOperation(data []float64, tileSize int) []float64 {
	extSize := tileSize + 2
	out := make([]float64, tileSize*tileSize*3)
	for ch := 0; ch < 3; ch++ {
		for i := 0; i < tileSize; i++ {
			for j := 0; j < tileSize; j++ {
				outIdx := (i*tileSize + j)*3 + ch
				// Extract the center 32x32 from the 34x34 input
				inIdx := ((i+1)*extSize + (j+1))*3 + ch
				out[outIdx] = 255.0 - data[inIdx]
			}
		}
	}
	return out
}