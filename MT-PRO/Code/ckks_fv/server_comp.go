package ckks_fv

import (
	"fmt"
	"math"
	"time"
)

type ServerComp interface {
	TranscipherFirstMessage(symCiphertext []*PlaintextRingT, kCt []*Ciphertext) *Ciphertext
	TranscipherFullMessage(symCiphertext []*PlaintextRingT, kCt []*Ciphertext) []*Ciphertext
	TranscipherFirstNMessages(symCiphertext []*PlaintextRingT, kCt []*Ciphertext, n int) []*Ciphertext
	TranscipherFirstMessageTimings(symCiphertext []*PlaintextRingT, kCt []*Ciphertext) (*Ciphertext, float64, float64)
	ComputeEuclideanDistanceSingle(probe *Ciphertext, reference *Ciphertext, tmplateLen int) *Ciphertext
	ComputeEuclideanDistanceFull(probe []*Ciphertext, reference []*Ciphertext, templateLen int) *Ciphertext
	ComputeHammingDistanceSingle(probe *Ciphertext, reference *Ciphertext, templateLen int) *Ciphertext
	AccumulateDistances(left *Ciphertext, right *Ciphertext) *Ciphertext
	Mask(probe *Ciphertext, i int) *Ciphertext
	ProcessTemplateInstructionsSingleCT(instructions []*TemplateInstruction, probe *Ciphertext, reference *Ciphertext) *Ciphertext
	EncryptProbeDirectlySingleCiphertext(data []float64) *Ciphertext
}

type serverComp struct {
	shared    Shared
	hera      MFVHera
	hbtp      *HalfBootstrapper
	needReset bool
}

func NewServerComp(shared Shared) ServerComp {

	server := new(serverComp)
	server.needReset = false
	server.shared = shared
	server.hera = NewMFVHera(shared.GetNumRounds(), shared.GetParams(), shared.GetFvEncoder(), shared.GetFvEncryptor(), shared.GetFvEvaluator(), shared.GetHeraModDown()[0])
	_, rotkeys, rlk, _ := shared.GetAllPublicKeys()
	hbtpKey := BootstrappingKey{Rlk: rlk, Rtks: rotkeys}
	server.hbtp, _ = NewHalfBootstrapper(server.shared.GetParams(), server.shared.GetHbtpParams(), hbtpKey)

	return server
}
func (server *serverComp) generateScaledPlaintext(plainCKKSRingTs []*PlaintextRingT) *Ciphertext {
	//var plaintexts []*Plaintext
	plaintexts := make([]*Plaintext, 16)

	for s := 0; s < 16; s++ {
		plaintexts[s] = NewPlaintextFVLvl(server.shared.GetParams(), 0)
		// ring -> in - plaintext -> out
		server.shared.GetFvEncoder().FVScaleUp(plainCKKSRingTs[s], plaintexts[s])
	}
	ciphertext := NewCiphertextFVLvl(server.shared.GetParams(), 1, 0)
	ciphertext.Value()[0] = plaintexts[0].Value()[0].CopyNew()
	return ciphertext
	//return plaintexts
}
func (server *serverComp) generateScaledFullPlaintext(plainCKKSRingTs []*PlaintextRingT) []*Ciphertext {
	//var plaintexts []*Plaintext
	plaintexts := make([]*Plaintext, 16)

	for s := 0; s < 16; s++ {
		plaintexts[s] = NewPlaintextFVLvl(server.shared.GetParams(), 0)
		// ring -> in - plaintext -> out
		server.shared.GetFvEncoder().FVScaleUp(plainCKKSRingTs[s], plaintexts[s])
	}
	var ciphertexts []*Ciphertext

	for i := 0; i < 16; i++ {
		ciphertext := NewCiphertextFVLvl(server.shared.GetParams(), 1, 0)
		ciphertext.Value()[0] = plaintexts[i].Value()[0].CopyNew()
		ciphertexts = append(ciphertexts, ciphertext)
	}

	return ciphertexts
	//return plaintexts
}
func (server *serverComp) reset() {
	if server.needReset {
		fmt.Println("Reset Hera")
		server.hera.Reset(server.shared.GetHeraModDown()[0])
	} else {
		server.needReset = true
	}
}

func (server *serverComp) TranscipherFullMessage(symCiphertext []*PlaintextRingT, kCt []*Ciphertext) []*Ciphertext {
	var ctBoot *Ciphertext
	var result []*Ciphertext
	//var ciphertext *Ciphertext
	server.reset()
	ciphertexts := server.generateScaledFullPlaintext(symCiphertext)

	fvKeystreams := server.hera.Crypt(server.shared.GetNonces(), kCt, server.shared.GetHeraModDown())
	fmt.Println("HERA Offline")
	for i := 0; i < 1; i++ {
		//stcModDown is empty?? -> added check for length
		fvKeystreams[i] = server.shared.GetFvEvaluator().SlotsToCoeffs(fvKeystreams[i], server.shared.GetStcModDown())
		server.shared.GetFvEvaluator().ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
	}

	for i := 1; i < 16; i++ {
		fvKeystreams[i] = server.shared.GetFvEvaluator().SlotsToCoeffs(fvKeystreams[i], server.shared.GetStcModDown())
		server.shared.GetFvEvaluator().ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
	}
	for i := 0; i < 16; i++ {
		fmt.Printf("Round %v \n", i)
		fmt.Println("Sub")
		server.shared.GetFvEvaluator().Sub(ciphertexts[i], fvKeystreams[i], ciphertexts[i])
		fmt.Println("NTT")
		server.shared.GetFvEvaluator().TransformToNTT(ciphertexts[i], ciphertexts[i])
		fmt.Println("Scale")
		ciphertexts[i].SetScale(math.Exp2(math.Round(math.Log2(float64(server.shared.GetParams().Qi()[0]) / float64(server.shared.GetParams().PlainModulus()) * server.shared.GetMessageScaling()))))
		fmt.Println("HalfBoot")
		if server.shared.GetFullCoeffs() {
			ctBoot, _ = server.hbtp.HalfBoot(ciphertexts[i], false)
		} else {
			ctBoot, _ = server.hbtp.HalfBoot(ciphertexts[i], true)
		}
		result = append(result, ctBoot)

	}

	return result
}

func (server *serverComp) TranscipherFirstNMessages(symCiphertext []*PlaintextRingT, kCt []*Ciphertext, n int) []*Ciphertext {
	var ctBoot *Ciphertext
	var result []*Ciphertext
	server.reset()
	//var ciphertext *Ciphertext
	ciphertexts := server.generateScaledFullPlaintext(symCiphertext)
	// if we want to transcipher another message kCt doesn't have enough levels??? aka the encrypted key

	fvKeystreams := server.hera.Crypt(server.shared.GetNonces(), kCt, server.shared.GetHeraModDown())
	fmt.Println("HERA Offline")
	for i := 0; i < 1; i++ {
		//stcModDown is empty?? -> added check for length
		fvKeystreams[i] = server.shared.GetFvEvaluator().SlotsToCoeffs(fvKeystreams[i], server.shared.GetStcModDown())
		server.shared.GetFvEvaluator().ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
	}

	for i := 1; i < 16; i++ {
		fvKeystreams[i] = server.shared.GetFvEvaluator().SlotsToCoeffs(fvKeystreams[i], server.shared.GetStcModDown())
		server.shared.GetFvEvaluator().ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
	}
	for i := 0; i < n; i++ {
		fmt.Printf("Round %v \n", i)
		fmt.Println("Sub")
		server.shared.GetFvEvaluator().Sub(ciphertexts[i], fvKeystreams[i], ciphertexts[i])
		fmt.Println("NTT")
		server.shared.GetFvEvaluator().TransformToNTT(ciphertexts[i], ciphertexts[i])
		fmt.Println("Scale")
		ciphertexts[i].SetScale(math.Exp2(math.Round(math.Log2(float64(server.shared.GetParams().Qi()[0]) / float64(server.shared.GetParams().PlainModulus()) * server.shared.GetMessageScaling()))))
		fmt.Println("HalfBoot")
		if server.shared.GetFullCoeffs() {
			ctBoot, _ = server.hbtp.HalfBoot(ciphertexts[i], false)
		} else {
			ctBoot, _ = server.hbtp.HalfBoot(ciphertexts[i], true)
		}
		result = append(result, ctBoot)

	}

	return result
}

func (server *serverComp) TranscipherFirstMessage(symCiphertext []*PlaintextRingT, kCt []*Ciphertext) *Ciphertext {
	var ctBoot *Ciphertext
	var ciphertext *Ciphertext
	server.reset()
	//Measure this time
	ciphertext = server.generateScaledPlaintext(symCiphertext)
	//to this
	fvKeystreams := server.hera.Crypt(server.shared.GetNonces(), kCt, server.shared.GetHeraModDown())
	fmt.Println("HERA Offline")
	for i := 0; i < 1; i++ {
		//stcModDown is empty?? -> added check for length
		//fvKeystreams[i] = server.shared.GetFvEvaluator().SlotsToCoeffs(fvKeystreams[i], server.shared.GetStcModDown())
		//server.shared.GetFvEvaluator().ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
		fmt.Printf("round %v\n", i)
		fvKeystreams[i] = server.shared.GetFvEvaluator().SlotsToCoeffs(fvKeystreams[i], server.shared.GetStcModDown())
		server.shared.GetFvEvaluator().ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
	}

	fmt.Println("Sub")
	server.shared.GetFvEvaluator().Sub(ciphertext, fvKeystreams[0], ciphertext)
	fmt.Println("NTT")
	server.shared.GetFvEvaluator().TransformToNTT(ciphertext, ciphertext)
	fmt.Println("Scale")
	ciphertext.SetScale(math.Exp2(math.Round(math.Log2(float64(server.shared.GetParams().Qi()[0]) / float64(server.shared.GetParams().PlainModulus()) * server.shared.GetMessageScaling()))))
	fmt.Println("HalfBoot")

	if server.shared.GetFullCoeffs() {
		ctBoot, _ = server.hbtp.HalfBoot(ciphertext, false)
	} else {
		ctBoot, _ = server.hbtp.HalfBoot(ciphertext, true)
	}
	return ctBoot
}
func (server *serverComp) TranscipherFirstMessageTimings(symCiphertext []*PlaintextRingT, kCt []*Ciphertext) (*Ciphertext, float64, float64) {
	var ctBoot *Ciphertext
	var ciphertext *Ciphertext
	server.reset()
	//Measure this time
	start := time.Now()
	ciphertext = server.generateScaledPlaintext(symCiphertext)
	timingCT := time.Since(start).Seconds()
	//to this
	start = time.Now()
	fvKeystreams := server.hera.Crypt(server.shared.GetNonces(), kCt, server.shared.GetHeraModDown())
	fmt.Println("HERA Offline")
	for i := 0; i < 1; i++ {
		//stcModDown is empty?? -> added check for length
		//fvKeystreams[i] = server.shared.GetFvEvaluator().SlotsToCoeffs(fvKeystreams[i], server.shared.GetStcModDown())
		//server.shared.GetFvEvaluator().ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
		fmt.Printf("round %v\n", i)
		fvKeystreams[i] = server.shared.GetFvEvaluator().SlotsToCoeffs(fvKeystreams[i], server.shared.GetStcModDown())
		server.shared.GetFvEvaluator().ModSwitchMany(fvKeystreams[i], fvKeystreams[i], fvKeystreams[i].Level())
	}

	fmt.Println("Sub")
	server.shared.GetFvEvaluator().Sub(ciphertext, fvKeystreams[0], ciphertext)
	fmt.Println("NTT")
	server.shared.GetFvEvaluator().TransformToNTT(ciphertext, ciphertext)
	fmt.Println("Scale")
	ciphertext.SetScale(math.Exp2(math.Round(math.Log2(float64(server.shared.GetParams().Qi()[0]) / float64(server.shared.GetParams().PlainModulus()) * server.shared.GetMessageScaling()))))
	fmt.Println("HalfBoot")

	if server.shared.GetFullCoeffs() {
		ctBoot, _ = server.hbtp.HalfBoot(ciphertext, false)
	} else {
		ctBoot, _ = server.hbtp.HalfBoot(ciphertext, true)
	}
	timingTransciphering := time.Since(start).Seconds()
	return ctBoot, timingCT, timingTransciphering
}
func (server *serverComp) ComputeEuclideanDistanceSingle(probe *Ciphertext, reference *Ciphertext, templateLen int) *Ciphertext {
	temp := server.shared.GetCKKSEvaluator().SubNew(reference, probe)
	tempSquared := server.shared.GetCKKSEvaluator().MulNew(temp, temp)
	server.shared.GetCKKSEvaluator().Relinearize(tempSquared, tempSquared)

	rt := server.shared.GetCKKSEvaluator().RotateNew(tempSquared, 1)
	//rt := server.shared.CKKSEvaluator().RotateNew(reference, 1)
	server.shared.GetCKKSEvaluator().Add(tempSquared, rt, tempSquared)

	for i := 0; i < templateLen-2; i++ {
		rt = server.shared.GetCKKSEvaluator().RotateNew(rt, 1)
		server.shared.GetCKKSEvaluator().Add(tempSquared, rt, tempSquared)
	}

	return tempSquared
}

func (server *serverComp) ComputeEuclideanDistanceFull(probe []*Ciphertext, reference []*Ciphertext, templateLen int) *Ciphertext {
	accumulated := server.ComputeEuclideanDistanceSingle(probe[0], reference[0], templateLen)
	for i := 1; i < len(probe); i++ {
		tempSquared := server.ComputeEuclideanDistanceSingle(probe[i], reference[i], templateLen)
		server.shared.GetCKKSEvaluator().Add(accumulated, tempSquared, accumulated)
	}
	return accumulated
}

func (server *serverComp) AccumulateDistances(left *Ciphertext, right *Ciphertext) *Ciphertext {
	return server.shared.GetCKKSEvaluator().AddNew(left, right)
}

func (server *serverComp) ComputeHammingDistanceSingle(probe *Ciphertext, reference *Ciphertext, templateLen int) *Ciphertext {
	//Sum(Ai+Bi - 2*AiBi)
	AiBiAdd := server.shared.GetCKKSEvaluator().AddNew(probe, reference)
	AiBiMul := server.shared.GetCKKSEvaluator().MulNew(probe, reference)
	server.shared.GetCKKSEvaluator().Relinearize(AiBiMul, AiBiMul)
	AiBiMulTwo := server.shared.GetCKKSEvaluator().MultByConstNew(AiBiMul, 2.0)
	//server.shared.GetCKKSEvaluator().Relinearize(AiBiMulTwo, AiBiMulTwo)
	temp := server.shared.GetCKKSEvaluator().SubNew(AiBiAdd, AiBiMulTwo)
	//Sum up temp into first slot
	rt := server.shared.GetCKKSEvaluator().RotateNew(temp, 1)
	server.shared.GetCKKSEvaluator().Add(temp, rt, temp)
	for i := 0; i < templateLen-2; i++ {
		rt = server.shared.GetCKKSEvaluator().RotateNew(rt, 1)
		server.shared.GetCKKSEvaluator().Add(temp, rt, temp)
	}
	return temp
}
func (server *serverComp) Mask(text *Ciphertext, i int) *Ciphertext {
	mask := server.GetEncryptedMaskSingle(i)
	res := server.shared.GetCKKSEvaluator().MulNew(mask, text)
	return res
}
func (server *serverComp) GetEncryptedMask(templateInstruction *TemplateInstruction) *Ciphertext {

	message := server.shared.CreateMaskSingle(templateInstruction.Index)
	//coeffs := server.generateCoeffsFromData(message)
	//plainCKKSRingTs := server.encCoeffsKeystreamToCkksPtRing(coeffs)
	cmplx := make([]complex128, server.shared.GetMessagesSize())
	fmt.Println(message)
	for i, _ := range message {
		cmplx[i] = complex(message[i], 0.0)
	}
	logSlots := server.shared.GetParams().logSlots
	if server.shared.GetFullCoeffs() {
		logSlots = server.shared.GetParams().logFVSlots
	}
	pt := server.shared.GetCKKSEncoder().EncodeComplexNew(cmplx, logSlots)
	enc := server.shared.GetCKKSEncryptor().EncryptNew(pt)
	return enc
}

func (server *serverComp) EncryptProbeDirectlySingleCiphertext(data []float64) *Ciphertext {
	msgSize := server.shared.GetMessagesSize()
	logSlots := server.shared.GetParams().logSlots
	if server.shared.GetFullCoeffs() {
		msgSize = server.shared.GetMessagesSize() / 2
		logSlots = (server.shared.GetParams().LogN() - 1)
	}
	cmplx := make([]complex128, msgSize)
	//fmt.Println(message)
	for i, _ := range data {
		cmplx[i] = complex(data[i], 0.0)
	}
	pt := server.shared.GetCKKSEncoder().EncodeComplexNew(cmplx, logSlots)
	enc := server.shared.GetCKKSEncryptor().EncryptNew(pt)
	return enc

}
func (server *serverComp) GetEncryptedMaskSingle(index int) *Ciphertext {

	message := server.shared.CreateMaskSingle(index)
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
	pt := server.shared.GetCKKSEncoder().EncodeComplexNew(cmplx, logSlots)
	enc := server.shared.GetCKKSEncryptor().EncryptNew(pt)
	return enc

}
func (server *serverComp) MaxTemplateLen(instructionsDic map[int][]*TemplateInstruction) int {
	max := 0
	for k, _ := range instructionsDic {
		if k > max {
			max = k
		}
	}
	return max
}

func (server *serverComp) EuclideanDistanceTemplateInstructions(instructionsDic map[int][]*TemplateInstruction, probe *Ciphertext, reference *Ciphertext) {
	maxLen := server.MaxTemplateLen(instructionsDic)

	temp := server.shared.GetCKKSEvaluator().SubNew(reference, probe)
	tempSquared := server.shared.GetCKKSEvaluator().MulNew(temp, temp)
	server.shared.GetCKKSEvaluator().Relinearize(tempSquared, tempSquared)

	rt := server.shared.GetCKKSEvaluator().RotateNew(tempSquared, 1)
	//rt := server.shared.CKKSEvaluator().RotateNew(reference, 1)
	server.shared.GetCKKSEvaluator().Add(tempSquared, rt, tempSquared)

	for i := 1; i < maxLen-1; i++ {
		//for a templen of 512, 511 rotations are needed thats why we look at i+1 in the dic
		val, exists := instructionsDic[(i + 1)]
		if exists {
			for _, ins := range val {

				fmt.Println("index: ", ins.Index)
				fmt.Println("ct: ", tempSquared)
				mask := server.Mask(tempSquared, ins.Index)
				server.shared.GetCKKSEvaluator().Relinearize(mask, mask)
				ins.Mask = mask
			}
		}
		rt = server.shared.GetCKKSEvaluator().RotateNew(rt, 1)
		server.shared.GetCKKSEvaluator().Add(tempSquared, rt, tempSquared)
	}
	val, exists := instructionsDic[maxLen]
	if exists {
		for _, ins := range val {
			mask := server.Mask(tempSquared, ins.Index)
			ins.Mask = mask
		}
	}

}

func (server *serverComp) HammingDistanceTemplateInstructions(instructionsDic map[int][]*TemplateInstruction, probe *Ciphertext, reference *Ciphertext) {
	maxLen := server.MaxTemplateLen(instructionsDic)

	AiBiAdd := server.shared.GetCKKSEvaluator().AddNew(probe, reference)
	AiBiMul := server.shared.GetCKKSEvaluator().MulNew(probe, reference)
	server.shared.GetCKKSEvaluator().Relinearize(AiBiMul, AiBiMul)
	AiBiMulTwo := server.shared.GetCKKSEvaluator().MultByConstNew(AiBiMul, 2.0)
	//server.shared.GetCKKSEvaluator().Relinearize(AiBiMulTwo, AiBiMulTwo)
	temp := server.shared.GetCKKSEvaluator().SubNew(AiBiAdd, AiBiMulTwo)
	//Sum up temp into first slot
	rt := server.shared.GetCKKSEvaluator().RotateNew(temp, 1)
	server.shared.GetCKKSEvaluator().Add(temp, rt, temp)

	for i := 1; i < maxLen-1; i++ {
		//for a templen of 512, 511 rotations are needed thats why we look at i+1 in the dic
		val, exists := instructionsDic[(i + 1)]
		if exists {
			for _, ins := range val {

				fmt.Println("index: ", ins.Index)
				fmt.Println("ct: ", temp)
				mask := server.Mask(temp, ins.Index)
				server.shared.GetCKKSEvaluator().Relinearize(mask, mask)
				ins.Mask = mask
			}
		}
		rt = server.shared.GetCKKSEvaluator().RotateNew(rt, 1)
		server.shared.GetCKKSEvaluator().Add(temp, rt, temp)
	}
	val, exists := instructionsDic[maxLen]
	if exists {
		for _, ins := range val {
			mask := server.Mask(temp, ins.Index)
			ins.Mask = mask
		}
	}
}

func (server *serverComp) ProcessTemplateInstructionsSingleCT(instructions []*TemplateInstruction, probe *Ciphertext, reference *Ciphertext) *Ciphertext {
	bin_dic := make(map[int][]*TemplateInstruction)
	float_dic := make(map[int][]*TemplateInstruction)
	for _, instruction := range instructions {
		if instruction.Binary {
			_, ok := bin_dic[instruction.TemplateLength]
			if !ok {
				bin_dic[instruction.TemplateLength] = make([]*TemplateInstruction, 0)
			}
			bin_dic[instruction.TemplateLength] = append(bin_dic[instruction.TemplateLength], instruction)
		} else {
			_, ok := float_dic[instruction.TemplateLength]
			if !ok {
				float_dic[instruction.TemplateLength] = make([]*TemplateInstruction, 0)
			}
			float_dic[instruction.TemplateLength] = append(float_dic[instruction.TemplateLength], instruction)
		}
	}

	server.EuclideanDistanceTemplateInstructions(float_dic, probe, reference)
	server.HammingDistanceTemplateInstructions(bin_dic, probe, reference)
	//add masks together for final result
	var final_ct *Ciphertext
	firstAdded := false
	for _, v := range float_dic {
		for _, ti := range v {
			if !firstAdded {
				rel := ti.Mask
				if ti.Mask.Degree() == 2 {
					rel = server.shared.GetCKKSEvaluator().RelinearizeNew(ti.Mask)
				}
				final_ct = rel
				firstAdded = true
			} else {
				rel := ti.Mask
				if ti.Mask.Degree() == 2 {
					rel = server.shared.GetCKKSEvaluator().RelinearizeNew(ti.Mask)
				}
				server.shared.GetCKKSEvaluator().Add(final_ct, rel, final_ct)
			}
		}
	}
	for _, v := range bin_dic {
		for _, ti := range v {
			if !firstAdded {
				rel := ti.Mask
				if ti.Mask.Degree() == 2 {
					rel = server.shared.GetCKKSEvaluator().RelinearizeNew(ti.Mask)
				}
				final_ct = rel
				firstAdded = true
			} else {
				rel := ti.Mask
				if ti.Mask.Degree() == 2 {
					rel = server.shared.GetCKKSEvaluator().RelinearizeNew(ti.Mask)
				}
				server.shared.GetCKKSEvaluator().Add(final_ct, rel, final_ct)
			}
		}
	}
	return final_ct

}
