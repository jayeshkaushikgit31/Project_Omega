package ckks_fv

import "fmt"

type CT struct {
	Ciphertext []*SCiphertext
	ContextId  string
}

type Ciphertext_Serialized interface {
	PrintCiphertext()
	ConvertToRegularCiphertext() []*Ciphertext
	GetCiphertext() []*SCiphertext
	//GetRegularCiphertext() []*Ciphertext
}

func NewCiphertextSerialized(context SaveFile, ciphertext []*Ciphertext) Ciphertext_Serialized {
	ct := new(CT)
	ct.ContextId = context.GetContextID()
	actualCt := make([]*SCiphertext, 0)

	for i := 0; i < 16; i++ {
		actualCt = append(actualCt, ConvertTosCiphertext(ciphertext[i]))
	}
	ct.Ciphertext = actualCt

	return ct
}
func NewSingleCiphertextSerialized(context string, ciphertext *Ciphertext) Ciphertext_Serialized {
	ct := new(CT)
	ct.ContextId = context
	actualCt := make([]*SCiphertext, 0)
	actualCt = append(actualCt, ConvertTosCiphertext(ciphertext))
	ct.Ciphertext = actualCt
	return ct
}
func (ct *CT) PrintCiphertext() {
	fmt.Println("Context ID: ", ct.ContextId)
	fmt.Println("Ciphertext: ", ct.Ciphertext)
}

func (ct *CT) GetCiphertext() []*SCiphertext {
	return ct.Ciphertext
}

func (ct *CT) ConvertToRegularCiphertext() []*Ciphertext {
	actualCt := make([]*Ciphertext, 0)
	for i := 0; i < 16; i++ {
		if i == len(ct.Ciphertext) {
			break
		}
		c := ConvertToCiphertext(ct.Ciphertext[i])
		actualCt = append(actualCt, c)
	}
	return actualCt
}
