package ckks_fv

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path"
	"strconv"
	"strings"

	"github.com/kshedden/gonpy"

	"github.com/ldsec/lattigo/v2/utils"
)

type Tmpl struct {
	data []float64
	name string
}

type Template interface {
	PrintTemplate()
	SquaredEuclideanDistance(other Template) float64
	HammingDistance(other Template) float64
	DecryptedDeviationDistance(decrypted [][]complex128) float64
	DecryptedDeviationDistanceFullCoeffs(decrypted []complex128) float64
	GetData() []float64
	GetName() string
}

func NewBinaryMockTemplate(size int) Template {
	nTemplate := new(Tmpl)
	nTemplate.data = make([]float64, size)
	nTemplate.name = "Binary"
	for i := 0; i < size; i++ {
		if rand.Intn(2) == 1 {
			nTemplate.data[i] = 1.0
		} else {
			nTemplate.data[i] = 0.0
		}
	}
	return nTemplate
}

func NewMockTemplate(size int) Template {
	nTemplate := new(Tmpl)
	nTemplate.data = make([]float64, size)
	nTemplate.name = "RndmFinger"
	for i := 0; i < size; i++ {
		nTemplate.data[i] = utils.RandFloat64(-1, 1)
	}
	return nTemplate
}
func NewTemplate(pathToFile string, half ...bool) Template {
	nTemplate := new(Tmpl)
	if _, err := os.Stat(pathToFile); err == nil {

	} else if errors.Is(err, os.ErrNotExist) {
		fmt.Println("File does not exist")
	}
	file, err := os.Open(pathToFile)

	if err != nil {
		println(err)
		return nil
	}
	_, fileName := path.Split(pathToFile)
	name := strings.TrimSuffix(fileName, ".af")
	nTemplate.name = strings.TrimSuffix(name, ".npy")
	if strings.HasSuffix(fileName, "af") {
		scanner := bufio.NewScanner(file)
		scanner.Split(bufio.ScanLines)
		for scanner.Scan() {
			//fmt.Println(scanner.Text())
			parsed, _ := strconv.ParseFloat(scanner.Text(), 64)
			nTemplate.data = append(nTemplate.data, parsed)
		}
	} else {
		r, x := gonpy.NewFileReader(pathToFile)
		data, e := r.GetFloat32()
		if e != nil || x != nil {
			fmt.Print(e, x)
		}
		for i, _ := range data {
			if len(half) > 0 && i == len(data)/2 {
				break
			}
			nTemplate.data = append(nTemplate.data, float64(data[i]))
		}
	}

	file.Close()
	fmt.Println("Sucessfully converted to Template: ", nTemplate.name)
	return nTemplate
}

func (template *Tmpl) PrintTemplate() {
	fmt.Printf("Template %v\n", template.name)
	for i := 0; i < len(template.data); i++ {
		fmt.Printf("%6.10f;", template.data[i])
	}
	fmt.Println()
}

func (template *Tmpl) SquaredEuclideanDistance(other Template) float64 {
	otherData := other.GetData()
	if len(template.data) != len(otherData) {
		panic("templates need to have same length")
	}
	accum := 0.0
	for i, _ := range template.data {
		accum += math.Pow(template.data[i]-otherData[i], 2)
	}
	return accum

}
func (template *Tmpl) HammingDistance(other Template) float64 {
	otherData := other.GetData()
	if len(template.data) != len(otherData) {
		panic("templates need to have same length")
	}
	accum := 0.0
	for i, _ := range template.data {
		if template.data[i] != other.GetData()[i] {
			accum += 1
		}

	}
	return accum

}

func (template *Tmpl) DecryptedDeviationDistance(decrypted [][]complex128) float64 {

	deviation := 0.0
	var i int
	for r, _ := range decrypted {
		for c, _ := range decrypted[r] {
			i = r*len(decrypted[r]) + c
			if i < len(template.data) {
				deviation += math.Pow(real(decrypted[r][c])-template.data[i], 2)
			} else {
				deviation += math.Pow(real(decrypted[r][c])-0, 2)
			}

		}
	}
	return math.Sqrt(deviation)

}

func (template *Tmpl) DecryptedDeviationDistanceFullCoeffs(decrypted []complex128) float64 {

	deviation := 0.0
	for i, _ := range template.data {
		deviation += math.Pow(real(decrypted[i])-template.data[i], 2)
	}
	return math.Sqrt(deviation)

}

func (Template *Tmpl) GetData() []float64 {
	return Template.data
}

func (Template *Tmpl) GetName() string {
	return Template.name
}
