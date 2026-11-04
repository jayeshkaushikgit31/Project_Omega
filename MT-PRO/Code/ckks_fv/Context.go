package ckks_fv

import (
	"fmt"
	"math/rand"
)

type Context struct {
	ID         string
	PK         *PublicKey
	SK         *SecretKey
	Nonces     [][]byte
	PDcds      [][]*PtDiagMatrixT
	NumRound   int
	ParamIndex int
	Radix      int
	FullCoeffs bool
	RelinKey   *RelinearizationKey
	//BootstrappingKey BootstrappingKey
}

type SaveFile interface {
	PrintContext()
	GetContextID() string
	GetClientParameters() (*SecretKey, *PublicKey)
	GetSharedParameters() (int, int, int, bool, [][]*PtDiagMatrixT, [][]byte)
	//TODO: Add rotation key set here
	GetRelinKey() *RelinearizationKey
}

func CreateNewContext(shared Shared, Pk *PublicKey, Sk *SecretKey) SaveFile {
	context := new(Context)
	context.ID = "0"
	context.SK = Sk
	context.PK = Pk
	context.Nonces = shared.GetNonces()
	context.PDcds = shared.GetPDcds()
	context.NumRound = shared.GetNumRounds()
	context.Radix = shared.GetRadix()
	context.ParamIndex = shared.GetParamIndex()
	context.FullCoeffs = shared.GetFullCoeffs()

	//context.rotKeys = RotKeys.RotationKeySet
	//_, RotKeys, RelinKey, BootstrappingKey := shared.GetAllPublicKeys()
	//context.RotationKeys = RotKeys
	//_, _, RelinKey, BootstrappingKey := shared.GetAllPublicKeys()
	_, _, RelinKey, _ := shared.GetAllPublicKeys()
	context.RelinKey = RelinKey
	//context.BootstrappingKey = BootstrappingKey
	fmt.Println("Context Created")
	return context
}
func (context *Context) GetContextID() string {
	return context.ID
}
func (context *Context) PrintContext() {
	fmt.Println("ID: ", context.ID)
	fmt.Println("SK: ", context.SK)
	fmt.Println("PK: ", context.PK)
	//fmt.Println("Nonces: ", context.Nonces)
	fmt.Println("NumRound: ", context.NumRound)
	fmt.Println("Radix: ", context.Radix)
	fmt.Println("Relin: ", context.RelinKey)
	fmt.Println("PCDS: ", context.PDcds)
	//fmt.Println("Bootstrapping Key: ", context.BootstrappingKey)
	//fmt.Println("Rotation Keys: ", context.rotKeys)
	//fmt.Println("FullCoeffs: ", context.FullCoeffs)
}

func (context *Context) GetClientParameters() (*SecretKey, *PublicKey) {
	return context.SK, context.PK
}

func (context *Context) GetRelinKey() *RelinearizationKey {
	return context.RelinKey
}

func (context *Context) GetSharedParameters() (int, int, int, bool, [][]*PtDiagMatrixT, [][]byte) {

	nonces := make([][]byte, 3)
	for i := 0; i < 3; i++ {
		nonces[i] = make([]byte, 64)
		rand.Read(nonces[i])
	}
	//return context.NumRound, context.ParamIndex, context.Radix, context.FullCoeffs, context.PDcds, context.Nonces
	return context.NumRound, context.ParamIndex, context.Radix, context.FullCoeffs, context.PDcds, nonces
}
