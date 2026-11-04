package ckks_fv

import (
	"fmt"
	"io"
	"math"
	"math/big"
	"math/bits"
	"os"
	"path"
	"sort"
	"strings"

	"github.com/ldsec/lattigo/v2/ring"
)

// Returns uniform random value in (0,q) by rejection sampling
func SampleZqx(rand io.Reader, q uint64) (res uint64) {
	bitLen := bits.Len64(q - 2)
	byteLen := (bitLen + 7) / 8
	b := bitLen % 8
	if b == 0 {
		b = 8
	}

	bytes := make([]byte, byteLen)
	for {
		_, err := io.ReadFull(rand, bytes)
		if err != nil {
			panic(err)
		}
		bytes[byteLen-1] &= uint8((1 << b) - 1)

		res = 0
		for i := 0; i < byteLen; i++ {
			res += uint64(bytes[i]) << (8 * i)
		}

		if res < q {
			return
		}
	}
}

// StandardDeviation computes the scaled standard deviation of the input vector.
func StandardDeviation(vec []float64, scale float64) (std float64) {
	// We assume that the error is centered around zero
	var err, tmp, mean, n float64

	n = float64(len(vec))

	for _, c := range vec {
		mean += c
	}

	mean /= n

	for _, c := range vec {
		tmp = c - mean
		err += tmp * tmp
	}

	return math.Sqrt(err/n) * scale
}

func scaleUpExact(value float64, n float64, q uint64) (res uint64) {

	var isNegative bool
	var xFlo *big.Float
	var xInt *big.Int

	isNegative = false
	if value < 0 {
		isNegative = true
		xFlo = big.NewFloat(-n * value)
	} else {
		xFlo = big.NewFloat(n * value)
	}

	xFlo.Add(xFlo, big.NewFloat(0.5))

	xInt = new(big.Int)
	xFlo.Int(xInt)
	xInt.Mod(xInt, ring.NewUint(q))

	res = xInt.Uint64()

	if isNegative {
		res = q - res
	}

	return
}

func scaleUpVecExact(values []float64, n float64, moduli []uint64, coeffs [][]uint64) {

	var isNegative bool
	var xFlo *big.Float
	var xInt *big.Int
	tmp := new(big.Int)

	for i := range values {

		if n*math.Abs(values[i]) > 1.8446744073709552e+19 {

			isNegative = false
			if values[i] < 0 {
				isNegative = true
				xFlo = big.NewFloat(-n * values[i])
			} else {
				xFlo = big.NewFloat(n * values[i])
			}

			xFlo.Add(xFlo, big.NewFloat(0.5))

			xInt = new(big.Int)
			xFlo.Int(xInt)

			for j := range moduli {
				tmp.Mod(xInt, ring.NewUint(moduli[j]))
				if isNegative {
					coeffs[j][i] = moduli[j] - tmp.Uint64()
				} else {
					coeffs[j][i] = tmp.Uint64()
				}
			}
		} else {

			if values[i] < 0 {
				for j := range moduli {
					coeffs[j][i] = moduli[j] - (uint64(-n*values[i]+0.5) % moduli[j])
				}
			} else {
				for j := range moduli {
					coeffs[j][i] = uint64(n*values[i]+0.5) % moduli[j]
				}
			}
		}
	}
}

func scaleUpVecExactBigFloat(values []*big.Float, scale float64, moduli []uint64, coeffs [][]uint64) {

	prec := int(values[0].Prec())

	xFlo := ring.NewFloat(0, prec)
	xInt := new(big.Int)
	tmp := new(big.Int)

	zero := ring.NewFloat(0, prec)

	scaleFlo := ring.NewFloat(scale, prec)
	half := ring.NewFloat(0.5, prec)

	for i := range values {

		xFlo.Mul(scaleFlo, values[i])

		if values[i].Cmp(zero) < 0 {
			xFlo.Sub(xFlo, half)
		} else {
			xFlo.Add(xFlo, half)
		}

		xFlo.Int(xInt)

		for j := range moduli {

			Q := ring.NewUint(moduli[j])

			tmp.Mod(xInt, Q)

			if values[i].Cmp(zero) < 0 {
				tmp.Add(tmp, Q)
			}

			coeffs[j][i] = tmp.Uint64()
		}
	}
}

// Divides x by n^2, returns a float
func scaleDown(coeff *big.Int, n float64) (x float64) {

	x, _ = new(big.Float).SetInt(coeff).Float64()
	x /= n

	return
}

func genBigIntChain(Q []uint64) (bigintChain []*big.Int) {

	bigintChain = make([]*big.Int, len(Q))
	bigintChain[0] = ring.NewUint(Q[0])
	for i := 1; i < len(Q); i++ {
		bigintChain[i] = ring.NewUint(Q[i])
		bigintChain[i].Mul(bigintChain[i], bigintChain[i-1])
	}
	return
}

// GenSwitchkeysRescalingParams generates the parameters for rescaling the switching keys
func GenSwitchkeysRescalingParams(Q, P []uint64) (params []uint64) {

	params = make([]uint64, len(Q))

	PBig := ring.NewUint(1)
	for _, pj := range P {
		PBig.Mul(PBig, ring.NewUint(pj))
	}

	tmp := ring.NewUint(0)

	for i := 0; i < len(Q); i++ {

		params[i] = tmp.Mod(PBig, ring.NewUint(Q[i])).Uint64()
		params[i] = ring.ModExp(params[i], int(Q[i]-2), Q[i])
		params[i] = ring.MForm(params[i], Q[i], ring.BRedParams(Q[i]))
	}

	return
}

func sliceBitReverseInPlaceComplex128(slice []complex128, N int) {

	var bit, j int

	for i := 1; i < N; i++ {

		bit = N >> 1

		for j >= bit {
			j -= bit
			bit >>= 1
		}

		j += bit

		if i < j {
			slice[i], slice[j] = slice[j], slice[i]
		}
	}
}

func sliceBitReverseInPlaceRingComplex(slice []*ring.Complex, N int) {

	var bit, j int

	for i := 1; i < N; i++ {

		bit = N >> 1

		for j >= bit {
			j -= bit
			bit >>= 1
		}

		j += bit

		if i < j {
			slice[i], slice[j] = slice[j], slice[i]
		}
	}
}
func Print_context(shared Shared) {

	fmt.Printf("\nMessage Size: %v\n", shared.GetMessagesSize())
	pk, rotK, rln, bstKey := shared.GetAllPublicKeys()
	fmt.Println("Pk")
	fmt.Println(*pk)
	fmt.Println("Rln")
	fmt.Println(*rln)
	fmt.Println("RotK")
	fmt.Println(*rotK)
	fmt.Println("BootstrappingKey")
	fmt.Println(bstKey)

}

func Print_decrypted_template(dec [][]complex128) {
	for i := 0; i < len(dec)-1; i++ {
		for j := 0; j < len(dec[i])-1; j++ {
			fmt.Printf("%6.3f,", dec[i][j])
		}
		fmt.Print("\n\n")
	}
}
func Print_decrypted_message(dec []complex128) {
	for i := 0; i < len(dec)-1; i++ {
		if i >= 100 {
			break
		}
		fmt.Printf("[%v]%6.3f,", i, dec[i])
	}
	fmt.Print("\n")
}

func Print_decrypted_template_fullCoeffs(dec []complex128, length int) {
	for i := 0; i < length; i++ {
		fmt.Printf("[%v]%6.3f,", i, dec[i])
	}
	fmt.Print("\n")
}
func Print_decrypted_template_fullCoeffs_WithCompare(dec []complex128, template Template) {
	for i := 0; i < 128; i++ {
		fmt.Printf("[%v]%6.3f==%6.3f,", i, dec[i], template.GetData()[i])
	}
	fmt.Print("\n")
}

func SaveContext(contextPath string, shared Shared, serverAuth ServerAuth) {
	fmt.Println("******************************")
	fmt.Println("   Context Serialization")
	fmt.Println("******************************")

	fmt.Println("Context Path: ", contextPath)

	if shared.GetFullCoeffs() {
		ctx := serverAuth.GetSaveContextFC()
		SerializeContextFC(ctx, contextPath)
		fromFile := DeserializeContextFC(contextPath)
		ctx.PrintContext()
		fmt.Println("Context From File:")
		fromFile.PrintContext()
	} else {
		ctx := serverAuth.GetSaveContext()
		SerializeContext(ctx, contextPath)
		fromFile := DeserializeContext(contextPath)
		ctx.PrintContext()
		fmt.Println("Context From File:")
		fromFile.PrintContext()

		_, Rotkeys, _, BootKeys := shared.GetAllPublicKeys()
		SerializeBootstrapKeys(BootKeys, "./bootKeys.gob")
		bootFromFile := DeserializeBootstrappingKeys("./bootKeys.gob")
		fmt.Println("Bootkeys From File:")
		fmt.Println(bootFromFile)
		SerializeRotationKeys(Rotkeys, "./rotkeys.gob")
		rotfromFile := DeserializeRotationKeys("./rotkeys.gob")
		fmt.Println("RotKeys From File:")
		println(rotfromFile)
	}

}

func FileOrFolderExists(path string) bool {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return false
	}
	return true
}

func GetAllFileFromFolder(folder string) []string {
	dir, err := os.Open(folder)
	if err != nil {
		panic(err)
	}
	defer dir.Close()
	names, _ := dir.Readdirnames(0)
	for i, name := range names {
		names[i] = path.Join(dir.Name(), name)
	}

	return names
}
func GetAllFileFromSubFolder(folder string, subFolder string) []string {
	folder = path.Join(folder, subFolder)
	dir, err := os.Open(folder)
	if err != nil {
		panic(err)
	}
	defer dir.Close()
	names, _ := dir.Readdirnames(0)
	for i, name := range names {
		names[i] = path.Join(dir.Name(), name)
	}

	return names
}

func GetAllSubjectIDsFromFolder(folder string) []string {
	dir, err := os.Open(folder)
	if err != nil {
		panic(err)
	}
	defer dir.Close()
	names, _ := dir.Readdirnames(0)
	sort.Strings(names)
	return names
}

func AddSubjectFolder(folder string, subject string) {
	folder = path.Join(folder, subject)
	if FileOrFolderExists(folder) {
		return
	}
	os.Mkdir(folder, os.ModePerm)
}
func ConnectTemplates(templates []Template) []float64 {
	data := make([]float64, 0)
	for _, template := range templates {
		data = append(data, template.GetData()...)
	}
	return data
}

func ProbePath(encryptedFolder string, subject string, probeID string) string {
	x := strings.Join([]string{probeID, "enc"}, ".")
	return path.Join(encryptedFolder, subject, x)
}

func MatedPath(encryptedFolder string, subject string) string {
	subject = "mated_" + subject
	x := strings.Join([]string{subject, "comp"}, ".")
	return path.Join(encryptedFolder, x)
}
func NonMatedPath(encryptedFolder string, subject string) string {
	subject = "NonMated_" + subject
	x := strings.Join([]string{subject, "comp"}, ".")
	return path.Join(encryptedFolder, x)
}

func WriteResultsToFile(path string, encrypted []float64, plain []float64) {
	f, err := os.Create(path)
	if err != nil {
		fmt.Println(err)
	}
	// remember to close the file
	defer f.Close()

	for i, e := range encrypted {
		line := fmt.Sprintf("%6.9f;%6.9f\n", e, plain[i])
		_, err := f.WriteString(line)
		if err != nil {
			fmt.Println(err)
		}
	}
	fmt.Println("Added Comparison file: ", path)

}
