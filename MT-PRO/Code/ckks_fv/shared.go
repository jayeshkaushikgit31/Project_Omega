package ckks_fv

import (
	"crypto/rand"
	"fmt"

	"github.com/ldsec/lattigo/v2/utils"
)

type Shared interface {
	GetParams() *Parameters
	GetParamIndex() int
	GetRadix() int
	GetHammingWeight() int
	GetFullCoeffs() bool
	GetMessagesSize() int
	GetNumRounds() int
	GetMessageScaling() float64
	GetNonces() [][]byte
	GetHeraModDown() []int
	GetStcModDown() []int
	GetPDcds() [][]*PtDiagMatrixT
	GetHbtpParams() *HalfBootParameters
	GetFvEncryptor() MFVEncryptor
	GetFvEvaluator() MFVEvaluator
	GetCKKSEvaluator() CKKSEvaluator
	GetRotationKeys() *RotationKeySet
	GetFvEncoder() MFVEncoder

	GetAllPublicKeys() (*PublicKey, *RotationKeySet, *RelinearizationKey, BootstrappingKey)
	GetCKKSEncoder() CKKSEncoder
	GetCKKSEncryptor() CKKSEncryptor
	CreateRandomData() [][]float64
	CreateSpecificData() [][]float64
	DataToComplex(data [][]float64) []complex128
	DataToComplexFull(data [][]float64) [][]complex128
	CreateMask(index int) [][]float64
	CreateMaskSingle(index int) []float64

	SetPublicKeys(pk *PublicKey, rotKeys *RotationKeySet, relinKey *RelinearizationKey, bootstrappingKey BootstrappingKey)
	SetRotationKeyset(rotKeys *RotationKeySet)
	SetNonces(nonce [][]byte)
}

// bootstrappingKey BootstrappingKey
type Ishared struct {
	Params     *Parameters
	ParamIndex int
	radix      int
	HbtpParams *HalfBootParameters

	FvEncryptor MFVEncryptor
	FvEvaluator MFVEvaluator
	FvEncoder   MFVEncoder

	CkksEvaluator CKKSEvaluator
	CkksEncryptor CKKSEncryptor
	CkksEncoder   CKKSEncoder

	Nonces         [][]byte
	MessageSize    int
	MessageScaling float64
	PDcds          [][]*PtDiagMatrixT
	HammingWeight  int
	NumRound       int
	FullCoeffs     bool

	Pk               *PublicKey
	RotKeys          *RotationKeySet
	RelinKey         *RelinearizationKey
	BootstrappingKey BootstrappingKey

	HeraModDown []int
	stcModDown  []int
}

func Init_From_Context(context SaveFile) Shared {
	numRound, paramIndex, radix, fullCoeffs, pDcds, nonces := context.GetSharedParameters()
	//pk, ror, rel, boot := context.GetPublicAndComputationKeys()
	var s Ishared
	s.Nonces = nonces
	s.NumRound = numRound
	s.radix = radix
	hbtpParams := RtFHeraParams[paramIndex]
	s.HbtpParams = hbtpParams
	s.ParamIndex = paramIndex
	params, err := hbtpParams.Params()
	if err != nil {
		panic(err)
	}
	//server.paramIndex = paramIndex

	if numRound == 4 {
		s.HeraModDown = HeraModDownParams80[paramIndex].CipherModDown
		s.stcModDown = HeraModDownParams80[paramIndex].StCModDown
	} else {
		s.HeraModDown = HeraModDownParams128[paramIndex].CipherModDown
		s.stcModDown = HeraModDownParams128[paramIndex].StCModDown
	}
	if fullCoeffs {
		s.MessageSize = params.N()
		params.SetLogFVSlots(params.LogN())
	} else {
		s.MessageSize = params.Slots()
		params.SetLogFVSlots(params.LogSlots())
	}

	s.Params = params
	s.HammingWeight = hbtpParams.H
	s.MessageScaling = (float64(params.PlainModulus()) / (hbtpParams.MessageRatio * 2))
	s.MessageScaling = (float64(params.PlainModulus()) / hbtpParams.MessageRatio)

	//is this really needed?
	s.FvEncoder = NewMFVEncoder(params)
	s.CkksEncoder = NewCKKSEncoder(params)

	s.FullCoeffs = fullCoeffs

	s.PDcds = pDcds

	//s.SetPublicKeys(pk, ror, rel, boot)
	return &s
}
func Init(numRound int, paramIndex int, radix int, fullCoeffs bool) Shared {
	var s Ishared
	s.NumRound = numRound
	s.radix = radix
	hbtpParams := RtFHeraParams[paramIndex]
	s.HbtpParams = hbtpParams
	s.ParamIndex = paramIndex
	params, err := hbtpParams.Params()
	if err != nil {
		panic(err)
	}
	//server.paramIndex = paramIndex

	s.HammingWeight = hbtpParams.H
	if fullCoeffs {
		s.MessageSize = params.N()
		params.SetLogFVSlots(params.LogN())
	} else {
		s.MessageSize = params.Slots()
		params.SetLogFVSlots(params.LogSlots())
	}
	//s.MessageScaling = (float64(params.PlainModulus()) / (hbtpParams.MessageRatio * 2))
	s.MessageScaling = (float64(params.PlainModulus()) / hbtpParams.MessageRatio)
	s.Params = params

	//is this really needed?
	s.FvEncoder = NewMFVEncoder(params)
	s.CkksEncoder = NewCKKSEncoder(params)

	s.FullCoeffs = fullCoeffs

	s.PDcds = s.FvEncoder.GenSlotToCoeffMatFV(radix)

	if numRound == 4 {
		s.HeraModDown = HeraModDownParams80[paramIndex].CipherModDown
		s.stcModDown = HeraModDownParams80[paramIndex].StCModDown
	} else {
		s.HeraModDown = HeraModDownParams128[paramIndex].CipherModDown
		s.stcModDown = HeraModDownParams128[paramIndex].StCModDown
	}

	return &s
}
func (shared *Ishared) GetParamIndex() int {
	return shared.ParamIndex
}
func (shared *Ishared) GetRadix() int {
	return shared.radix
}
func (s *Ishared) SetPublicKeys(pk *PublicKey, rotKeys *RotationKeySet, relinKey *RelinearizationKey, bootstrappingKey BootstrappingKey) {
	s.Pk = pk
	s.RotKeys = rotKeys
	s.RelinKey = relinKey
	s.BootstrappingKey = bootstrappingKey
	s.FvEvaluator = NewMFVEvaluator(s.Params, EvaluationKey{Rlk: relinKey, Rtks: rotKeys}, s.PDcds)
	s.FvEncryptor = NewMFVEncryptorFromPk(s.Params, pk)
	s.CkksEvaluator = NewCKKSEvaluator(s.Params, EvaluationKey{Rlk: relinKey, Rtks: rotKeys})
	s.CkksEncryptor = NewCKKSEncryptorFromPk(s.Params, pk)
}
func (s *Ishared) SetRotationKeyset(rotKeys *RotationKeySet) {
	s.RotKeys = rotKeys
}

func (s *Ishared) GetFvEncryptor() MFVEncryptor {
	return s.FvEncryptor
}

func (s *Ishared) GetAllPublicKeys() (*PublicKey, *RotationKeySet, *RelinearizationKey, BootstrappingKey) {
	return s.Pk, s.RotKeys, s.RelinKey, s.BootstrappingKey
}

func (s *Ishared) GetSizeAndScaling() (int, float64) {
	return s.MessageSize, s.MessageScaling
}
func (s *Ishared) GetHeraModDown() []int {
	return s.HeraModDown
}
func (s *Ishared) GetStcModDown() []int {
	return s.stcModDown
}
func (s *Ishared) GetNonces() [][]byte {
	//var nonces [][]byte
	if len(s.Nonces) > 0 {
		return s.Nonces
	}
	nonces := make([][]byte, s.MessageSize)
	for i := 0; i < s.MessageSize; i++ {
		nonces[i] = make([]byte, 64)
		rand.Read(nonces[i])
	}
	s.Nonces = nonces
	return nonces
}

func (s *Ishared) GetParams() *Parameters {
	return s.Params
}
func (s *Ishared) GetHammingWeight() int {
	return s.HammingWeight
}
func (s *Ishared) GetFullCoeffs() bool {
	return s.FullCoeffs
}

func (s *Ishared) GetPDcds() [][]*PtDiagMatrixT {
	return s.PDcds
}
func (s *Ishared) GetHbtpParams() *HalfBootParameters {
	return s.HbtpParams
}

// if fullCoeffs size = params.N
// else size = params.slots
func (shared *Ishared) CreateRandomData() [][]float64 {
	fmt.Println("Message Size: 16x", shared.MessageSize)
	data := make([][]float64, 16)
	for s := 0; s < 16; s++ {
		data[s] = make([]float64, shared.MessageSize)

		for i := 0; i < shared.MessageSize; i++ {
			data[s][i] = utils.RandFloat64(-1, 1)
		}
	}
	return data
}

//Fill all rows with range from 0-16 but [1][0] will be 1 to test euclidean distance
func (shared *Ishared) CreateSpecificData() [][]float64 {
	data := make([][]float64, 16)

	for s := 0; s < 16; s++ {
		data[s] = make([]float64, shared.MessageSize)
		for i := 0; i < shared.MessageSize; i++ {
			data[s][i] = float64(s*16 + i)
		}
	}
	return data
}
func (shared *Ishared) CreateMask(index int) [][]float64 {
	data := make([][]float64, 16)

	for s := 0; s < 16; s++ {
		data[s] = make([]float64, shared.MessageSize)
		for i := 0; i < shared.MessageSize; i++ {
			if i == index {
				data[0][i] = 1.0
			} else {
				data[s][i] = 0.0
			}

		}
	}
	return data
}
func (shared *Ishared) CreateMaskSingle(index int) []float64 {
	msgSize := shared.MessageSize
	if shared.FullCoeffs {
		msgSize = shared.GetMessagesSize() / 2
	}
	data := make([]float64, msgSize)
	for i := 0; i < msgSize; i++ {
		if i == index {
			data[i] = 1.0
		} else {
			data[i] = 0.0
		}

	}
	return data
}

func (shared *Ishared) GetMessagesSize() int {
	return shared.MessageSize
}
func (shared *Ishared) DataToComplex(data [][]float64) []complex128 {
	valuesWant := make([]complex128, shared.Params.Slots())
	for i := 0; i < shared.Params.Slots(); i++ {
		//valuesWant[i] = complex(data[i][0], 0)
		valuesWant[i] = complex(data[0][i], 0)
	}
	return valuesWant
}
func (shared *Ishared) DataToComplexFull(data [][]float64) [][]complex128 {
	valuesWant := make([][]complex128, shared.Params.Slots())
	for r := 0; r < 16; r++ {
		valuesWant[r] = make([]complex128, 16)
		for i := 0; i < shared.Params.Slots(); i++ {
			//valuesWant[i] = complex(data[i][0], 0)
			valuesWant[r][i] = complex(data[r][i], 0)
		}
	}

	return valuesWant
}

func (shared *Ishared) GetFvEncoder() MFVEncoder {
	return shared.FvEncoder
}

func (shared *Ishared) GetFvEvaluator() MFVEvaluator {
	return shared.FvEvaluator
}

func (shared *Ishared) GetCKKSEvaluator() CKKSEvaluator {
	return shared.CkksEvaluator
}

func (shared *Ishared) GetNumRounds() int {
	return shared.NumRound
}
func (shared *Ishared) GetMessageScaling() float64 {
	return shared.MessageScaling
}

func (shared *Ishared) GetRotationKeys() *RotationKeySet {
	return shared.RotKeys
}
func (shared *Ishared) GetCKKSEncoder() CKKSEncoder {
	return shared.CkksEncoder
}

func (shared *Ishared) GetCKKSEncryptor() CKKSEncryptor {
	return shared.CkksEncryptor
}

func (shared *Ishared) SetNonces(nonce [][]byte) {
	shared.Nonces = nonce
}
