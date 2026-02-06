# CLAUDE.md — ariel

## What This Is

ariel is an R front-end for the Triton GPU compiler. It takes torchlang IR
graphs and compiles them to GPU kernels via Triton's MLIR infrastructure,
bypassing Python entirely.

Named after the spirit in The Tempest — fast, airy, does the real work.

## Architecture

```
torchlang IR (ir_graph)
      │
      ▼
Phase 5a: emit_ttir()          ← Pure R, string generation
      │                            Emits Triton MLIR textual IR
      ▼
Triton MLIR (.mlir text)
      │
      ▼
Phase 5b: mlir_compile()        ← Rcpp bindings via .Call()
      │                            Links against Triton's C++ MLIR libs
      │                            Runs: tt → ttg → LLVM → PTX
      ▼
PTX / CUBIN
      │
      ▼
Phase 5c: gpu_launch()          ← C bindings via .Call()
      │                            CUDA driver API
      ▼
GPU execution
```

## Triton MLIR Dialect Quick Reference

Triton IR (TTIR) uses these MLIR dialects:

- `tt` (Triton): `tt.load`, `tt.store`, `tt.dot`, `tt.reduce`, `tt.splat`,
  `tt.make_range`, `tt.get_program_id`, `tt.addptr`
- `arith`: `arith.addf`, `arith.mulf`, `arith.subf`, `arith.divf`,
  `arith.cmpf`, `arith.select`, `arith.constant`
- `math`: `math.exp`, `math.log`, `math.tanh`, `math.sqrt`, `math.abs`
- `scf`: `scf.for` (tiled matmul K-loop)
- `tensor`: tensor types

Triton's compilation pipeline:
1. TTIR (tt dialect) — what we emit
2. TTGIR (tt + ttg dialect) — GPU layout annotations (Triton adds these)
3. LLVM IR — via MLIR's LLVM dialect lowering
4. PTX — via LLVM's NVPTX backend
5. CUBIN — via ptxas

## Relationship to torchlang

ariel imports torchlang for the `ir_graph` / `ir_node` types and
optimization passes. The interface is:

```r
library(torchlang)
library(ariel)

# torchlang captures and optimizes
ir <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
ir <- optimize_graph(ir)

# ariel compiles to GPU
result <- emit_ttir(ir, group_id = 1L)     # Phase 5a
compiled <- mlir_compile(result, sm = 80L) # Phase 5b (needs Triton)
compiled$ptx                               # PTX assembly
# gpu_launch(compiled, x_gpu)              # Phase 5c (planned)
```

## Package Conventions

- Base R only (tinyverse), Rcpp for C++ bindings
- tinytest for testing
- Phase 5a tests work without GPU (pure string output)
- Phase 5b tests guarded by Triton availability
- Phase 5c tests guarded by GPU availability

## Phased Development

### Phase 5a — MLIR Textual IR Emission (complete)
- `R/ir_to_mlir.R`: torchlang IR → Triton MLIR text
- Pure R, no system dependencies
- Testable anywhere

### Phase 5b — MLIR Compilation via Rcpp (complete)
- `src/triton_compile.cpp`: Rcpp bindings wrapping Triton's C++ MLIR pipeline
- `src/triton_stub.cpp`: Fallback when Triton not available
- `R/compile.R`: R wrapper `mlir_compile()`
- `configure`: Finds TRITON_HOME, LLVM_DIR, writes Makevars
- Mirrors pass ordering from `triton/third_party/nvidia/backend/compiler.py`
- Pipeline: parse MLIR → TTIR passes → TTGIR → LLVM dialect → LLVM IR → PTX
- Needs: Triton built from source + LLVM/MLIR

### Phase 5c — GPU Launch (planned)
- `src/cuda_launch.c`: CUDA driver API bindings
- `R/launch.R`: kernel execution, memory management
- Needs: CUDA toolkit

## Building Triton for Phase 5b

Triton must be built from source to get C++ headers + libraries:

```bash
git clone https://github.com/triton-lang/triton.git ~/triton
cd ~/triton
pip install -e python  # builds C++ libs + downloads LLVM/MLIR
```

This produces:
- **Headers**: `~/triton/include/triton/` (source) + build dir (tablegen'd)
- **Libraries**: Object files in build dir
- **LLVM/MLIR**: Auto-downloaded to `~/.triton/llvm/{hash}/`

Then reinstall ariel:
```bash
TRITON_HOME=~/triton r -e 'tinypkgr::install()'
```

Without Triton, the package installs with a stub: Phase 5a works,
`mlir_compile()` returns a clear error message.

### Key libraries linked (when Triton available)
- Triton: TritonIR, TritonGPUIR, TritonToTritonGPU, TritonGPUToLLVM,
  TritonNVIDIAGPUToLLVM, NVGPUToLLVM, TritonAnalysis, etc.
- MLIR: MLIRPass, MLIRParser, MLIRIR, dialect libs (arith, math, scf, gpu, etc.)
- LLVM: NVPTXCodeGen, Passes, Core, Support, etc.

## Tested Emission Patterns (Phase 5a)

From `torchlang/inst/scripts/whisper_bench.R`:

### Elementwise fusion (`emit_ttir`)
- `fused_add_div` — mel postprocessing (2 inputs, add+div chain)
- `fused_add_gelu` — FFN bias+activation (2 inputs, add then GELU decomposition)
- All compound ops tested: relu, sigmoid, silu, gelu, sign

### Matmul with epilogue (`emit_ttir_matmul`)
- `matmul_kernel` — plain tiled matmul (works)
- `ffn_matmul_gelu_kernel` — matmul + GELU epilogue fusion (works)
- `matmul + mul(scale)` — **does not work**: `emit_ttir_matmul` only supports unary epilogues, not binary ops that need a second operand (like scaling by a constant)

### Known limitation: binary epilogues
`emit_ttir_matmul(epilogue_ops = "mul")` returns NULL because `mul` is a binary op but the epilogue codegen only passes `current_ssa` (one input). To support `matmul * scale`, would need either:
1. A special `scale` epilogue that takes a scalar attribute
2. A general binary epilogue mechanism with a second pointer argument

## Relationship to Whisper

The benchmark script at `torchlang/inst/scripts/whisper_bench.R` demonstrates ariel on 5 real Whisper-tiny computation patterns:
- FFN block: matmul + GELU + matmul
- Mel postprocessing: log10 + add + div
- GELU standalone
- Attention scores: matmul * scale
- Residual add

torchlang captures these as expressions, optimizes to IR, and ariel emits
TTIR for each fusible group. With Triton built from source, `mlir_compile()`
compiles to PTX. GPU launch (Phase 5c) is planned.
