# MT-Pro 
The MT-Pro Code is derived from the [RtF Transciphering Framework](https://github.com/KAIST-CryptLab/RtF-Transciphering). To start the code use the go file in [examples/ckks_fv](./Code/examples/ckks_fv). The Code might be adjusted from [ckks_fv/main.go](./Code/ckks_fv/main.go). 
When directly running - first adjust all the paths to your needs. 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
Below here the contents of the [RTF Transciphering Framework README.md](https://github.com/KAIST-CryptLab/RtF-Transciphering/blob/master/README.md) file are stated: 

# RtF Transciphering Framework
This is an implementation of the RtF transciphering framework along with the HERA cipher (proposed in [Transciphering Framework for Approximate Homomorphic Encryption](https://eprint.iacr.org/2020/1335)) and the Rubato cipher (proposed in [Rubato: Noisy Ciphers for Approximate Homomorphic Encryption](https://eprint.iacr.org/2022/537)) using the `lattigo` library.

## New Package
We implement the hybrid framework in [ckks_fv](./Code/ckks_fv), which contains the following functionalities.
- CKKS scheme (the same as [ckks](./Code/ckks))
- FV scheme supporting multi-level operations (named as `mfv`)
- Halfboot operation
- Evaluation of the HERA cipher in the FV scheme

An example of finding modulus switching parameter in the RtF framework is given in [examples/ckks_fv](./Code/examples/ckks_fv).

## Benchmark
You can run the benchmark of RtF transciphering framework along with HERA or Rubato using [RtF_bench_test.go](./Code/ckks_fv/RtF_bench_test.go).
The benchmark parameters are given in [rtf_params.go](./Code/ckks_fv/rtf_params.go).
To benchmark all the parameters, run the following command in [ckks_fv](./Code/ckks_fv) directory.

```go test -timeout=0s -bench=. ```

You can also benchmark specific parameter.
For example, to run benchmark with HERA, run the following command.

``` go test -timeout=0s -bench=BenchmarkRtFHera```

To run the HERA parameter `80as` in the paper, run the following command.

```go test -timeout=0s -bench=BenchmarkRtFHera80as```

Benchmark with Rubato is also possible in similar way.
