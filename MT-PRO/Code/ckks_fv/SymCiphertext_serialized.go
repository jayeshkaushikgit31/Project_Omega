package ckks_fv

//import "fmt"

type SCT struct {
	Ciphertext []*SPlaintextRingT
	ContextId  string
}

type SymCiphertext_Serialized interface {
}

func NewSymCiphertextSerialized(context SaveFile, ptr []*PlaintextRingT) SymCiphertext_Serialized {

	ct := new(SCT)
	ct.ContextId = context.GetContextID()
	sPtr := make([]*SPlaintextRingT, 0)
	for i, _ := range ptr {
		akt := ToSPoly(ptr[i])
		sPtr = append(sPtr, akt)
	}

	ct.Ciphertext = sPtr

	return ct
}

