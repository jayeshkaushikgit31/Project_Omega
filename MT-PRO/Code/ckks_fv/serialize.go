package ckks_fv

import (
	"encoding/gob"
	"fmt"
	"os"
)

func SerializeShared(shared Shared) {
	fmt.Println("=====Serialize Shared=====")
	gob.Register(Ishared{})
	f, err := os.Create("./shared.gob")
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if enc.Encode(shared); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}
	/*err := writeGob("./shared.gob", shared)
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("Done")
	}*/
}

func DeserializeShared() Shared {
	fmt.Println("=====Deserialize Shared=====")
	shared := new(Shared)
	f, err := os.Open("./shared.gob")
	if err != nil {
		fmt.Println(err)
	}
	gob.Register(Ishared{})
	dec := gob.NewDecoder(f)
	if dec.Decode(&shared); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}

	//err := readGob(, &shared)

	return *shared
}

func SerializeContext(context SaveFile, path string) {
	fmt.Println("=====Serialize Context=====")
	//gob.Register(BootstrappingKey{})
	gob.Register(Context{})
	f, err := os.Create(path)
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if enc.Encode(context); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}
}

func DeserializeContext(path string) SaveFile {
	fmt.Println("=====Deserialize Context=====")
	context := new(Context)
	f, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	gob.Register(Context{})
	//gob.Register(BootstrappingKey{})
	dec := gob.NewDecoder(f)
	if dec.Decode(&context); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}

	return context
}

func SerializeContextFC(context SaveFileFC, path string) {
	fmt.Println("=====Serialize Context=====")
	//gob.Register(BootstrappingKey{})
	gob.Register(ContextFC{})
	f, err := os.Create(path)
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if enc.Encode(context); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}
}

func DeserializeContextFC(path string) SaveFileFC {
	fmt.Println("=====Deserialize Context=====")
	context := new(ContextFC)
	f, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	gob.Register(ContextFC{})
	//gob.Register(BootstrappingKey{})
	dec := gob.NewDecoder(f)
	if dec.Decode(&context); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}

	return context
}
func SerializeRotationKeys(rtks *RotationKeySet, path string) {
	fmt.Println("=====Serialize RotKeys=====")
	gob.Register(RotationKeySet{})
	f, err := os.Create(path)
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if enc.Encode(rtks); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}
}

func DeserializeRotationKeys(path string) *RotationKeySet {
	fmt.Println("=====Deserialize RotKeys=====")
	ct := new(RotationKeySet)
	f, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	gob.Register(RotationKeySet{})
	dec := gob.NewDecoder(f)
	if dec.Decode(&ct); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}

	return ct
}

func SerializeBootstrapKeys(rtks BootstrappingKey, path string) {
	fmt.Println("=====Serialize BootstrappingKey=====")
	gob.Register(BootstrappingKey{})
	f, err := os.Create(path)
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if enc.Encode(rtks); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}
}

func DeserializeBootstrappingKeys(path string) BootstrappingKey {
	fmt.Println("=====Deserialize BootstrappingKey=====")
	var ct BootstrappingKey
	f, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	gob.Register(BootstrappingKey{})
	dec := gob.NewDecoder(f)
	if dec.Decode(&ct); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}

	return ct
}

func SerializeSymCT(pt SymCiphertext_Serialized, path string) {
	fmt.Println("=====Serialize SymCT=====")
	gob.Register(SCT{})
	f, err := os.Create(path)
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if enc.Encode(pt); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}
}

func SerializeCiphertext(cts Ciphertext_Serialized, path string) {
	fmt.Println("=====Serialize Ciphertext=====")
	gob.Register(CT{})
	f, err := os.Create(path)
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	if enc.Encode(cts); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}
}
func DeserializeCiphertext(path string) Ciphertext_Serialized {
	fmt.Println("=====Deserialize Ciphertext=====")
	ct := new(CT)
	f, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
	}
	gob.Register(CT{})
	dec := gob.NewDecoder(f)
	if dec.Decode(&ct); err != nil {
		fmt.Println(err)
	} else {
		println("Done")
	}

	return ct
}
