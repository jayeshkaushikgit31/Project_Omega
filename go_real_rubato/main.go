package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"time"
	"runtime"

	"github.com/ldsec/lattigo/v2/ckks_fv"
)

type Request struct {
	Operation string      `json:"operation"`
	Tiles     [][]float64 `json:"tiles"`
	TileSize  int         `json:"tile_size"`
}

type Response struct {
	Tiles      [][]float64 `json:"tiles"`
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

	W := 34 // tileSize + 2
	convRotations := []int{-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1}
	rotations := append(rotationsHalfBoot, rotationsStC...)
	rotations = append(rotations, convRotations...)

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
	initCrypto()

	decoder := json.NewDecoder(bufio.NewReader(os.Stdin))
	var req Request
	if err := decoder.Decode(&req); err != nil {
		fmt.Fprintf(os.Stderr, "Error decoding input: %v\n", err)
		os.Exit(1)
	}

	startTime := time.Now()
	outTiles := make([][]float64, len(req.Tiles))
	bandwidth := 0

	for i, tileData := range req.Tiles {
		bandwidth += len(tileData) * 8

		if req.Operation == "pixel" {
			out := make([]float64, len(tileData))
			for j, v := range tileData {
				out[j] = 255.0 - v
			}
			// Extract the valid 32x32 center from the 34x34 tile
			outTiles[i] = extractCenter(out, req.TileSize)
		} else {
			pt := ckks_fv.NewPlaintextFVLvl(params, params.MaxLevel())
			ckksEncoder.EncodeCoeffs(tileData, pt)
			ct := fvEncryptor.EncryptNew(pt)

			ctResult := applyHomomorphicConvolution(ct, req.TileSize)

			ptResult := ckksDecryptor.DecryptNew(ctResult)
			resData := ckksEncoder.DecodeCoeffs(ptResult)

			// Truncate result to original padded tile length
			if len(resData) >= len(tileData) {
				resData = resData[:len(tileData)]
			}
			// Extract the valid 32x32 center
			outTiles[i] = extractCenter(resData, req.TileSize)
		}
        // Force Go to free up memory from the previous tile
		runtime.GC()
	}

	elapsed := time.Since(startTime).Seconds()
	resp := Response{Tiles: outTiles, Bandwidth: bandwidth, ServerTime: elapsed}
	encoder := json.NewEncoder(os.Stdout)
	encoder.Encode(resp)
}

func applyHomomorphicConvolution(ct *ckks_fv.Ciphertext, tileSize int) *ckks_fv.Ciphertext {
	W := tileSize + 2
	ctOut := fvEvaluator.MulScalarNew(ct, 5)

	neighbors := []struct {
		rotation int
	}{
		{-W - 1}, {-W}, {-W + 1}, {-1}, {1}, {W - 1}, {W}, {W + 1},
	}

	for _, n := range neighbors {
		ctRot := fvEvaluator.RotateColumnsNew(ct, n.rotation)
		ctNeg := fvEvaluator.NegNew(ctRot)
		fvEvaluator.Add(ctOut, ctNeg, ctOut)
	}

	return ctOut
}

// extractCenter takes a flattened 34x34x3 array and returns the valid 32x32x3 center
func extractCenter(data []float64, tileSize int) []float64 {
	extSize := tileSize + 2
	out := make([]float64, tileSize*tileSize*3)
	for ch := 0; ch < 3; ch++ {
		for i := 0; i < tileSize; i++ {
			for j := 0; j < tileSize; j++ {
				outIdx := (i*tileSize + j)*3 + ch
				inIdx := ((i+1)*extSize + (j+1))*3 + ch
				out[outIdx] = data[inIdx]
			}
		}
	}
	return out
}