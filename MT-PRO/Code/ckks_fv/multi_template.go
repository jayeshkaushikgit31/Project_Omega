package ckks_fv

import (
	"errors"
	"fmt"
	"strings"
)

type MTmpl struct {
	data         []Template
	name         string
	subjectID    string
	instructions []*TemplateInstruction
	ct           *Ciphertext
}

type MultiTemplate interface {
	PrintTemplate()
	Distance(other MultiTemplate) float64
	ToMessage() []float64
	GetTemplates() []Template
	GetName() string
	GetSubjectID() string
	SetCiphertext(ct *Ciphertext)
	GetCiphertext() (*Ciphertext, error)
}

func NewMultiTemplate(templates []Template, instructions []*TemplateInstruction, subjectId string) MultiTemplate {
	nmt := new(MTmpl)
	nmt.data = templates
	nmt.instructions = instructions
	nmt.subjectID = subjectId
	names := []string{subjectId, "_"}
	for _, t := range templates {
		names = append(names, t.GetName())
	}
	nmt.name = strings.Join(names, "")
	print(nmt.name)
	return nmt
}

func (nmt *MTmpl) Distance(other MultiTemplate) float64 {
	sum := 0.0
	for i, t := range nmt.data {
		if i == 1 {
			sum += 2 * t.SquaredEuclideanDistance(other.GetTemplates()[i])
		} else {
			sum += t.SquaredEuclideanDistance(other.GetTemplates()[i])
		}

	}
	return sum
}

func (nmt *MTmpl) PrintTemplate() {
	fmt.Printf("\nID: %v Amount: %v\n", nmt.name, len(nmt.data))
	fmt.Printf("\nName0: %v len: %v, Name1: %v len: %v\n", nmt.data[0].GetName(), len(nmt.data[0].GetData()), nmt.data[1].GetName(), len(nmt.data[1].GetData()))

}

func (nmt *MTmpl) GetName() string {
	return nmt.name
}
func (nmt *MTmpl) GetSubjectID() string {
	return nmt.subjectID
}
func (nmt *MTmpl) ToMessage() []float64 {
	return ConnectTemplates(nmt.data)
}

func (nmt *MTmpl) GetTemplates() []Template {
	return nmt.data
}

func (nmt *MTmpl) SetCiphertext(ct *Ciphertext) {
	nmt.ct = ct
}
func (nmt *MTmpl) GetCiphertext() (*Ciphertext, error) {
	if nmt.ct == nil {
		return nil, errors.New("No ciphertext for this multitemplate")
	}
	return nmt.ct, nil
}
