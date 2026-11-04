package ckks_fv

import (
	"fmt"
)

type ServerAuth interface {
	DecryptMessage(ciphertext *Ciphertext) []complex128
	DecryptFullMessage(ciphertext []*Ciphertext) [][]complex128
	DecryptNMessages(ciphertext []*Ciphertext, n int) [][]complex128
	GetSaveContext() SaveFile
	GetSaveContextFC() SaveFileFC
	TestMask()
}

type serverAuth struct {
	sk            *SecretKey
	pk            *PublicKey
	ckksDecryptor CKKSDecryptor
	ckksEncoder   CKKSEncoder
	shared        Shared
	context       SaveFile
}

func (server *serverAuth) containsKey(rotations []int, r int) bool {
	for i, _ := range rotations {
		if rotations[i] == r {
			return true
		}
	}
	return false
}

func NewServerAuthFromContext(context SaveFile, shared Shared, Rotkeys *RotationKeySet, bootKey BootstrappingKey) ServerAuth {
	server := new(serverAuth)
	server.shared = shared
	sk, pk := context.GetClientParameters()
	server.context = context
	server.sk = sk
	server.pk = pk
	server.ckksDecryptor = NewCKKSDecryptor(server.shared.GetParams(), server.sk)
	server.ckksEncoder = NewCKKSEncoder(server.shared.GetParams())
	kgen := NewKeyGenerator(shared.GetParams())
	rotationsHalfBoot := kgen.GenRotationIndexesForHalfBoot(shared.GetParams().LogSlots(), shared.GetHbtpParams())
	rotationsStC := kgen.GenRotationIndexesForSlotsToCoeffsMat(shared.GetPDcds())
	rotations := append(rotationsHalfBoot, rotationsStC...)

	if !shared.GetFullCoeffs() {
		rotations = append(rotations, shared.GetParams().Slots()/2)
	}
	for i := 1; i < 16; i++ {
		if !server.containsKey(rotations, i) {
			rotations = append(rotations, i)
		}
	}
	fmt.Printf("\nRotations: %v\n", rotations)
	//dauert ewig?
	//rotationKeys := kgen.GenRotationKeysForRotations(rotations, true, server.sk)
	//rel, boot := context.GetRelinAndBootstrappingKeys()
	shared.SetPublicKeys(pk, Rotkeys, context.GetRelinKey(), bootKey)
	return server
}

func NewServerAuthFromContextFC(shared Shared, context SaveFileFC) ServerAuth {
	server := new(serverAuth)

	server.shared = shared
	kgen := NewKeyGenerator(shared.GetParams())
	sk, pk := context.GetAuthServerParameters()
	fmt.Printf("\nSK: %v. Pk: %v", sk, pk)
	server.sk = sk
	server.pk = pk

	rotationsHalfBoot := kgen.GenRotationIndexesForHalfBoot(shared.GetParams().LogSlots(), shared.GetHbtpParams())

	rotationsStC := kgen.GenRotationIndexesForSlotsToCoeffsMat(shared.GetPDcds())
	rotations := append(rotationsHalfBoot, rotationsStC...)

	if !shared.GetFullCoeffs() {
		rotations = append(rotations, shared.GetParams().Slots()/2)
		// for i := 1; i < 16; i++ {
		// 	if !server.containsKey(rotations, i) {
		// 		rotations = append(rotations, i)
		// 	}
		// }
	} else {
		// for i := 1; i < 250; i++ {
		// 	if !server.containsKey(rotations, i) {
		// 		rotations = append(rotations, i)
		// 	}
		// }
	}

	//fmt.Println("Rotations: ", rotations, "HalfBootRot: ", rotationsHalfBoot, "rotationsStC: ", rotationsStC)
	fmt.Printf("\nRotations: %v\n", rotations)
	//dauert ewig?
	rotationKeys := kgen.GenRotationKeysForRotations(rotations, true, server.sk)
	//rotationKeys := kgen.GenRotationKeysForRotations(rotations, false, server.sk)
	fmt.Println("Rotation Keys added")
	relinKey := kgen.GenRelinearizationKey(server.sk)
	fmt.Println("Relin Keys added")
	hbtpKey := BootstrappingKey{Rlk: relinKey, Rtks: rotationKeys}
	bootstrappingKey := hbtpKey
	fmt.Println("Bootstrapping Keys added")

	shared.SetPublicKeys(pk, rotationKeys, relinKey, bootstrappingKey)
	fmt.Println("Public Keys Keys set to shared")
	server.ckksDecryptor = NewCKKSDecryptor(server.shared.GetParams(), server.sk)
	server.ckksEncoder = NewCKKSEncoder(server.shared.GetParams())
	return server
}

func NewServerAuth(shared Shared) ServerAuth {

	server := new(serverAuth)

	server.shared = shared
	kgen := NewKeyGenerator(shared.GetParams())
	sk, pk := kgen.GenKeyPairSparse(shared.GetHammingWeight())
	fmt.Printf("\nSK: %v. Pk: %v", sk, pk)
	server.sk = sk
	server.pk = pk
	rotationsHalfBoot := kgen.GenRotationIndexesForHalfBoot(shared.GetParams().LogSlots(), shared.GetHbtpParams())

	rotationsStC := kgen.GenRotationIndexesForSlotsToCoeffsMat(shared.GetPDcds())
	rotations := append(rotationsHalfBoot, rotationsStC...)

	if !shared.GetFullCoeffs() {
		rotations = append(rotations, shared.GetParams().Slots()/2)
		// for i := 1; i < 16; i++ {
		// 	if !server.containsKey(rotations, i) {
		// 		rotations = append(rotations, i)
		// 	}
		// }
	} else {
		// for i := 1; i < 128; i++ {
		// 	if !server.containsKey(rotations, i) {
		// 		rotations = append(rotations, i)
		// 	}
		// }
	}

	//fmt.Println("Rotations: ", rotations, "HalfBootRot: ", rotationsHalfBoot, "rotationsStC: ", rotationsStC)
	fmt.Printf("\nRotations: %v\n", rotations)
	//dauert ewig?
	rotationKeys := kgen.GenRotationKeysForRotations(rotations, true, server.sk)
	//rotationKeys := kgen.GenRotationKeysForRotations(rotations, false, server.sk)
	fmt.Println("Rotation Keys added")
	relinKey := kgen.GenRelinearizationKey(server.sk)
	fmt.Println("Relin Keys added")
	hbtpKey := BootstrappingKey{Rlk: relinKey, Rtks: rotationKeys}
	bootstrappingKey := hbtpKey
	fmt.Println("Bootstrapping Keys added")

	shared.SetPublicKeys(pk, rotationKeys, relinKey, bootstrappingKey)
	fmt.Println("Public Keys Keys set to shared")
	server.ckksDecryptor = NewCKKSDecryptor(server.shared.GetParams(), server.sk)
	server.ckksEncoder = NewCKKSEncoder(server.shared.GetParams())
	return server
}
func (server *serverAuth) DecryptMessage(ciphertext *Ciphertext) []complex128 {

	valuesTest := server.ckksEncoder.DecodeComplex(server.ckksDecryptor.DecryptNew(ciphertext), server.shared.GetParams().LogSlots())
	return valuesTest
}

func (server *serverAuth) DecryptFullMessage(ciphertext []*Ciphertext) [][]complex128 {

	var result [][]complex128
	for i := 0; i < 16; i++ {
		valuesTest := server.ckksEncoder.DecodeComplex(server.ckksDecryptor.DecryptNew(ciphertext[i]), server.shared.GetParams().LogSlots())
		result = append(result, valuesTest)
	}
	return result
}

func (server *serverAuth) DecryptNMessages(ciphertext []*Ciphertext, n int) [][]complex128 {
	var result [][]complex128
	for i := 0; i < n; i++ {
		valuesTest := server.ckksEncoder.DecodeComplex(server.ckksDecryptor.DecryptNew(ciphertext[i]), server.shared.GetParams().LogSlots())
		result = append(result, valuesTest)
	}
	return result
}

func (server *serverAuth) generateSaveContext() SaveFile {
	context := CreateNewContext(server.shared, server.pk, server.sk)
	return context
}

func (server *serverAuth) GetSaveContext() SaveFile {
	if server.context == nil {
		return server.generateSaveContext()
	}
	return server.context
}

func (server *serverAuth) GetSaveContextFC() SaveFileFC {
	context := CreateNewContextFC(server.shared, server.pk, server.sk)
	return context
}

func (server *serverAuth) TestMask() {
	message := server.shared.CreateMaskSingle(0)
	//coeffs := server.generateCoeffsFromData(message)
	//plainCKKSRingTs := server.encCoeffsKeystreamToCkksPtRing(coeffs)
	msgSize := server.shared.GetMessagesSize()
	logSlots := server.shared.GetParams().logSlots
	if server.shared.GetFullCoeffs() {
		msgSize = server.shared.GetMessagesSize() / 2
		logSlots = (server.shared.GetParams().LogN() - 1)
	}
	cmplx := make([]complex128, msgSize)

	for i, _ := range message {
		cmplx[i] = complex(message[i], 0.0)
	}

	fmt.Println(logSlots, len(message), server.shared.GetParams().logSlots)
	pt := server.ckksEncoder.EncodeComplexNew(cmplx, logSlots)
	ckksEncryptor := NewCKKSEncryptorFromPk(server.shared.GetParams(), server.pk)
	enc := ckksEncryptor.EncryptNew(pt)
	res := server.DecryptMessage(enc)
	//fmt.Println(res)
	for _, c := range res {
		fmt.Printf("%6.3f,", real(c))
	}
	fmt.Println("")
	//ckksEncoder := NewCKKSEncoder(server.shared.GetParams())

}
