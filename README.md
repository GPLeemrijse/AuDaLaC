# AuDaLaC
This repository contains AuDaLaC, the compiler for the Autonomous Data Language (AuDaLa) that I wrote as part of my 2023 Embedded Systems master thesis ["Towards relaxed memory semantics for the Autonomous Data Language"](https://github.com/GPLeemrijse/AuDaLaC/blob/master/Towards_relaxed_memory_semantics_for_the_Autonomous_Data_Language.pdf). This work was done as part of the [Formal System Analysis](https://fsa.win.tue.nl/) research group of the [Technical University Eindhoven](https://www.tue.nl/en/). I was supervised by [dr. ir. Thomas Neele](https://tneele.com/).

## Requirements
The [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) needs to be installed. The GPU should support at least architecture `-arch=sm_60` with support for [Unified Virtual Address Space](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-virtual-address-space).
The Rust programming language should be installed (with cargo) to build the compiler.

## Usage
Basic usage:

```cargo run -- MyAuDaLaFile.adl -o MyAuDaLaFile.cu```
The resulting CUDA file (.cu) can be compiled with NVCC. See the [Makefiles](https://github.com/GPLeemrijse/AuDaLaC/blob/master/tests/benchmarks/prefix_sum/Makefile) for examples. The resulting binary will ask for a `.init` file, see [this file](https://github.com/GPLeemrijse/AuDaLaC/blob/master/tests/benchmarks/SPM/testcases/keiren_ex1_pg_min.init) for an example that defines the parity game on slide 23 of [these slides](https://www.win.tue.nl/~timw/teaching/amc/2009/college14.pdf).
Useful options include `--memorder`, `--fp_strat`, `--schedule_strat`, and `--time`. When debugging a program that will not terminate, the `--printunstable` option can be useful.

When a program creates more instances than specified in the supplied `.init` file, memory should be allocated for these instances through the `--nrofinstances` option. Like so:
```
cd tests/benchmarks/SCC
cargo run -- SCC_FB.adl -o SCC_FB.cu --nrofinstances NodeSet=10000
```


Run `cargo run -- -h` to see all CLI options.

## Running example algorithms
### SCC
```
cd tests/benchmarks/SCC
cargo run -- SCC_FB.adl -o SCC_FB.cu --nrofinstances NodeSet=10000
make SCC_FB.out
./SCC_FB.out testcases/random_fb_1000_1289.init

cargo run -- SCC_COL.adl -o SCC_COL.cu
make SCC_COL.out
./SCC_COL.out testcases/random_col_1000_1289.init
```
### SPM
```
cd tests/benchmarks/SPM
cargo run -- SPM.adl -o SPM.cu
make SPM.out
./SPM.out testcases/dining/dining_4.invariantly_inevitably_eat.init
```

### Prefix Sum
```
cd tests/benchmarks/prefix_sum
cargo run -- prefix_sum.adl -o prefix_sum.cu
make prefix_sum.out
./prefix_sum.out testcases/prefix_sum_1000.init
```

### SCS
```
cd tests/benchmarks/synthesis
cargo run -- synthesis.adl -o synthesis.cu
make synthesis.out
./synthesis.out testcases/synthesis_1000_1280.init
```

## Running tests and benchmarks
To run the unittests, simply run `cargo test`. If the benchmarks should also be run, run:
```
export BENCHMARK=true
cargo run -- --nocapture --test-threads=1
```
