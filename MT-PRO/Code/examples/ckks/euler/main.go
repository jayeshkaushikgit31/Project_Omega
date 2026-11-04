package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
)

func euclideanDistance(encReference *ckks.Ciphertext, encProbe *ckks.Ciphertext, templateSize int, evaluator ckks.Evaluator) *ckks.Ciphertext {
	temp := evaluator.SubNew(encReference, encProbe)
	tempSqaured := evaluator.MulRelinNew(temp, temp)

	rt := evaluator.RotateNew(tempSqaured, 1)
	evaluator.Add(tempSqaured, rt, tempSqaured)

	for i := 0; i < templateSize-1; i++ {
		rt = evaluator.RotateNew(rt, 1)
		evaluator.Add(tempSqaured, rt, tempSqaured)
	}
	return tempSqaured
}

func example() {

	var start time.Time
	var err error

	LogN := 14
	LogSlots := 13

	LogModuli := ckks.LogModuli{
		LogQi: []int{55, 40, 40, 40, 40, 40, 40, 40},
		LogPi: []int{45, 45},
	}

	Scale := float64(1 << 40)

	params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
	if err != nil {
		panic(err)
	}
	params.SetScale(Scale)
	params.SetLogSlots(LogSlots)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	kgen := ckks.NewKeyGenerator(params)

	sk := kgen.GenSecretKey()

	rlk := kgen.GenRelinearizationKey(sk)

	encryptor := ckks.NewEncryptorFromSk(params, sk)

	decryptor := ckks.NewDecryptor(params, sk)

	encoder := ckks.NewEncoder(params)

	evaluator := ckks.NewEvaluator(params, ckks.EvaluationKey{Rlk: rlk})

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= %f, sigma = %f \n", params.LogN(), params.LogSlots(), params.LogQP(), params.MaxLevel()+1, params.Scale(), params.Sigma())

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("           PLAINTEXT CREATION            ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	r := float64(16)

	pi := 3.141592653589793

	slots := params.Slots()

	values := make([]complex128, slots)
	for i := range values {
		values[i] = complex(2*pi, 0)
	}

	plaintext := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()/r)
	encoder.Encode(plaintext, values, params.LogSlots())

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              ENCRYPTION                 ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	ciphertext := encryptor.EncryptNew(plaintext)

	fmt.Printf("Done in %s \n", time.Since(start))

	printDebug(params, ciphertext, values, decryptor, encoder, false)

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Printf("        EVALUATION OF i*x on %d values\n", slots)
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()

	evaluator.MultByi(ciphertext, ciphertext)

	fmt.Printf("Done in %s \n", time.Since(start))

	for i := range values {
		values[i] *= complex(0, 1)
	}

	printDebug(params, ciphertext, values, decryptor, encoder, false)

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Printf("       EVALUATION of x/r on %d values\n", slots)
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()

	ciphertext.MulScale(r)

	fmt.Printf("Done in %s \n", time.Since(start))

	for i := range values {
		values[i] /= complex(r, 0)
	}

	printDebug(params, ciphertext, values, decryptor, encoder, false)

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Printf("       EVALUATION of e^x on %d values\n", slots)
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()

	coeffs := []complex128{
		complex(1.0, 0),
		complex(1.0, 0),
		complex(1.0/2, 0),
		complex(1.0/6, 0),
		complex(1.0/24, 0),
		complex(1.0/120, 0),
		complex(1.0/720, 0),
		complex(1.0/5040, 0),
	}

	poly := ckks.NewPoly(coeffs)

	if ciphertext, err = evaluator.EvaluatePoly(ciphertext, poly, ciphertext.Scale()); err != nil {
		panic(err)
	}

	fmt.Printf("Done in %s \n", time.Since(start))

	for i := range values {
		values[i] = cmplx.Exp(values[i])
	}

	printDebug(params, ciphertext, values, decryptor, encoder, false)

	fmt.Println()
	fmt.Println("===============================================")
	fmt.Printf("       EVALUATION of x^r on %d values\n", slots)
	fmt.Println("===============================================")
	fmt.Println()

	start = time.Now()

	evaluator.Power(ciphertext, int(r), ciphertext)

	fmt.Printf("Done in %s \n", time.Since(start))

	for i := range values {
		values[i] = cmplx.Pow(values[i], complex(r, 0))
	}

	printDebug(params, ciphertext, values, decryptor, encoder, false)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         DECRYPTION & DECODING           ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Printf("Done in %s \n", time.Since(start))

	printDebug(params, ciphertext, values, decryptor, encoder, false)

}
func euclideanDistanceTest() {

	var start time.Time
	var err error

	LogN := 14
	LogSlots := 13

	LogModuli := ckks.LogModuli{
		LogQi: []int{55, 40, 40, 40, 40, 40, 40, 40},
		LogPi: []int{45, 45},
	}

	Scale := float64(1 << 40)

	params, err := ckks.NewParametersFromLogModuli(LogN, &LogModuli)
	if err != nil {
		panic(err)
	}
	params.SetScale(Scale)
	params.SetLogSlots(LogSlots)

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         INSTANTIATING SCHEME            ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	kgen := ckks.NewKeyGenerator(params)

	sk := kgen.GenSecretKey()

	rlk := kgen.GenRelinearizationKey(sk)

	encryptor := ckks.NewEncryptorFromSk(params, sk)

	decryptor := ckks.NewDecryptor(params, sk)

	encoder := ckks.NewEncoder(params)

	slots := params.Slots()
	fmt.Println(slots)
	rot := make([]int, 12)

	//for i := 0; i < 10; i++ {
	//	rot[i] = 1
	//}
	rot[0] = 1
	//rot[1] = 1

	rotKeySet := kgen.GenRotationKeysForRotations(rot, false, sk)
	evaluator := ckks.NewEvaluator(params, ckks.EvaluationKey{Rlk: rlk, Rtks: rotKeySet})

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Printf("CKKS parameters: logN = %d, logSlots = %d, logQP = %d, levels = %d, scale= %f, sigma = %f \n", params.LogN(), params.LogSlots(), params.LogQP(), params.MaxLevel()+1, params.Scale(), params.Sigma())

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("           PLAINTEXT CREATION            ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	r := float64(16)

	//values := make([]complex128, slots)
	//for i := range values {
	//	values[i] = complex(float64(i), 0)
	//}
	values := make([]complex128, 16)
	for i := range values {
		values[i] = complex(float64(i), 0)
	}
	values2 := make([]complex128, 16)
	for i := range values {
		values2[i] = complex(float64(i), 0)
	}
	values2[0] = complex(3.0, 0)

	plaintext := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()/r)
	plaintext2 := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale()/r)

	encoder.Encode(plaintext, values, params.LogSlots())

	encoder.Encode(plaintext2, values2, params.LogSlots())

	fmt.Printf("Done in %s \n", time.Since(start))

	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("              ENCRYPTION                 ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()

	EncReference := encryptor.EncryptNew(plaintext)
	EncProbe := encryptor.EncryptNew(plaintext2)

	fmt.Printf("Done in %s \n", time.Since(start))

	printDebug(params, EncReference, values, decryptor, encoder, false)
	printDebug(params, EncProbe, values2, decryptor, encoder, false)
	fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         ROTATE          ")
	fmt.Println("=========================================")
	fmt.Println()
	// sub, multiply
	//roatate and add(sq+rotateted)
	temp := evaluator.SubNew(EncReference, EncProbe)
	fmt.Println("         ref-probe          ")
	printDebug(params, temp, values, decryptor, encoder, false)

	tempSqaured := evaluator.MulNew(temp, temp)
	fmt.Println("         squared          ")
	printDebug(params, tempSqaured, values, decryptor, encoder, false)

	evaluator.Relinearize(tempSqaured, tempSqaured)
	fmt.Println("         squared  (relin)        ")
	printDebug(params, tempSqaured, values, decryptor, encoder, false)

	rt := evaluator.RotateNew(tempSqaured, 1)
	//rt := evaluator.RotateNew(EncReference, 1)
	fmt.Println("         1. rot          ")
	printDebug(params, rt, values, decryptor, encoder, false)

	evaluator.Add(tempSqaured, rt, tempSqaured)
	fmt.Println("         add          ")
	printDebug(params, tempSqaured, values, decryptor, encoder, false)

	for i := 0; i < 9; i++ {
		rt = evaluator.RotateNew(rt, 1)
		evaluator.Add(tempSqaured, rt, tempSqaured)
	}
	fmt.Println("         result:          ")
	printDebug(params, tempSqaured, values, decryptor, encoder, false)
	EncReference0 := encryptor.EncryptNew(plaintext)
	EncProbe1 := encryptor.EncryptNew(plaintext2)

	ot := euclideanDistance(EncReference0, EncProbe1, 16, evaluator)
	fmt.Println("         res eucl()          ")
	printDebug(params, ot, values, decryptor, encoder, false)

	/*fmt.Println()
	fmt.Println("=========================================")
	fmt.Println("         DECRYPTION & DECODING           ")
	fmt.Println("=========================================")
	fmt.Println()

	start = time.Now()}*/

	// rotated := encoder.Decode(decryptor.DecryptNew(rt), params.LogSlots())
	// fmt.Printf("Done in %s \n", time.Since(start))
	// fmt.Printf("ValuesTest: %6.10f %6.10f %6.10f %6.10f...\n", rotated[0], rotated[1], rotated[2], rotated[3])

	//printDebug(params, ciphertext, values, decryptor, encoder)
	// printDebug(params, ciphertext, values, decryptor, encoder)
}
func printDebug(params *ckks.Parameters, ciphertext *ckks.Ciphertext, valuesWant []complex128, decryptor ckks.Decryptor, encoder ckks.Encoder, printDetails bool) (valuesTest []complex128) {

	valuesTest = encoder.Decode(decryptor.DecryptNew(ciphertext), params.LogSlots())

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", math.Log2(ciphertext.Scale()))
	fmt.Printf("Euclidean Distance: %6.10f\n", valuesTest[0])
	fmt.Printf("ValuesTest: %6.10f %6.10f %6.10f %6.10f...\n", valuesTest[0], valuesTest[1], valuesTest[2], valuesTest[3])
	fmt.Printf("ValuesWant: %6.10f %6.10f %6.10f %6.10f...\n", valuesWant[0], valuesWant[1], valuesWant[2], valuesWant[3])
	if !printDetails {
		return
	}
	for i := 0; i < 10; i++ {
		fmt.Printf("ct:%v = %v\n", math.Round(real(valuesTest[i])), math.Round(real(valuesWant[i])))
	}

	fmt.Println()

	//precStats := ckks.GetPrecisionStats(params, nil, nil, valuesWant, valuesTest, params.LogSlots(), 0)

	//fmt.Println(precStats.String())

	return
}

func main() {
	euclideanDistanceTest()
	//example()
}
