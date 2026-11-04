package ckks_fv

type TemplateInstruction struct {
	Index          int
	TemplateLength int
	Binary         bool
	Mask           *Ciphertext
}

func NewFingerShortTemplateInstruction(index int) *TemplateInstruction {
	ti := new(TemplateInstruction)
	ti.Index = index
	ti.TemplateLength = 256
	ti.Binary = false
	return ti
}

func NewIrisTemplateInstruction(index int) *TemplateInstruction {
	ti := new(TemplateInstruction)
	ti.Index = index
	ti.TemplateLength = 5120
	ti.Binary = true
	return ti
}
func NewIrisShortTemplateInstruction(index int) *TemplateInstruction {
	ti := new(TemplateInstruction)
	ti.Index = index
	ti.TemplateLength = 512
	ti.Binary = true
	return ti
}

func NewFaceTemplateInstruction(index int) *TemplateInstruction {
	ti := new(TemplateInstruction)
	ti.Index = index
	ti.TemplateLength = 512
	ti.Binary = false
	//log.Println("Added FaceTemplate Instruction")
	return ti
}
func NewFingerValentinaTemplateInstruction(index int) *TemplateInstruction {
	ti := new(TemplateInstruction)
	ti.Index = index
	ti.TemplateLength = 256
	ti.Binary = false
	//log.Println("Added FaceTemplate Instruction")
	return ti
}
func NewIrisValentinaTemplateInstruction(index int) *TemplateInstruction {
	ti := new(TemplateInstruction)
	ti.Index = index
	ti.TemplateLength = 512
	ti.Binary = false
	//log.Println("Added Iris(Valentina) Instruction")
	return ti
}

func NewFingerLongTemplateInstruction(index int) *TemplateInstruction {
	ti := new(TemplateInstruction)
	ti.Index = index
	ti.TemplateLength = 640
	ti.Binary = false
	return ti
}
