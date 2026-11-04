package ckks_fv

import (
	"fmt"
	"math/rand"
)

type ContextFC struct {
	ID     string
	PK     *PublicKey
	SK     *SecretKey
	Nonces [][]byte

	NumRound   int
	ParamIndex int
	Radix      int
	FullCoeffs bool

	//BootstrappingKey BootstrappingKey
}

type SaveFileFC interface {
	PrintContext()
	GetContextID() string
	GetAuthServerParameters() (*SecretKey, *PublicKey)
	GetSharedParameters() (int, int, int, bool, [][]byte)
}

func CreateNewContextFC(shared Shared, Pk *PublicKey, Sk *SecretKey) SaveFileFC {
	context := new(ContextFC)
	context.ID = "0"
	context.SK = Sk
	context.PK = Pk
	context.Nonces = shared.GetNonces()
	context.NumRound = shared.GetNumRounds()
	context.Radix = shared.GetRadix()
	context.ParamIndex = shared.GetParamIndex()
	context.FullCoeffs = shared.GetFullCoeffs()
	fmt.Println("Context Created")
	return context
}
func (context *ContextFC) GetContextID() string {
	return context.ID
}
func (context *ContextFC) PrintContext() {
	fmt.Println("ID: ", context.ID)
	fmt.Println("SK: ", context.SK)
	fmt.Println("PK: ", context.PK)
	fmt.Println("NumRound: ", context.NumRound)
	fmt.Println("Radix: ", context.Radix)
	fmt.Println("Nonces (len): ", len(context.Nonces))
}

func (context *ContextFC) GetAuthServerParameters() (*SecretKey, *PublicKey) {
	return context.SK, context.PK
}

func (context *ContextFC) GetSharedParameters() (int, int, int, bool, [][]byte) {
	var nonces [][]byte
	if len(context.Nonces) == 0 {
		println("Nonce is empty")
		nonces := make([][]byte, 3)
		for i := 0; i < 3; i++ {
			nonces[i] = make([]byte, 64)
			rand.Read(nonces[i])
		}
	} else {
		nonces = context.Nonces
	}

	//return context.NumRound, context.ParamIndex, context.Radix, context.FullCoeffs, context.PDcds, context.Nonces
	return context.NumRound, context.ParamIndex, context.Radix, context.FullCoeffs, nonces
}
