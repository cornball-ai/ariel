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
Phase 5b: mlir_compile()        ← C bindings via .Call()
      │                            Links against Triton's MLIR libs
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
mlir_text <- emit_ttir(ir, group_id = 1L)  # Phase 5a
# kernel <- mlir_compile(mlir_text)          # Phase 5b
# result <- gpu_launch(kernel, x_gpu)        # Phase 5c
```

## Package Conventions

- Base R only (tinyverse)
- C (not C++) for MLIR/CUDA bindings where possible
- tinytest for testing
- Phase 5a tests work without GPU (pure string output)
- Phase 5b/5c tests guarded by GPU availability

## Phased Development

### Phase 5a — MLIR Textual IR Emission (current)
- `R/ir_to_mlir.R`: torchlang IR → Triton MLIR text
- Pure R, no system dependencies
- Testable anywhere

### Phase 5b — MLIR C API Compilation
- `src/mlir_bindings.c`: R ↔ MLIR C API
- Links against Triton's MLIR libraries
- Needs: Triton installed from source (for libTritonIR, etc.)

### Phase 5c — GPU Launch
- `src/cuda_launch.c`: CUDA driver API bindings
- `R/launch.R`: kernel execution, memory management
- Needs: CUDA toolkit

## Building Triton for C API Access

Triton must be built from source to get the MLIR libraries:

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -e python  # builds C++ libraries
# Libraries end up in triton/_C/ or build/
```

Key libraries needed:
- libTritonIR (Triton MLIR dialects)
- libTritonTransforms (optimization passes)
- libMLIR (core MLIR infrastructure)

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

torchlang captures these as expressions, optimizes to IR, and ariel emits TTIR for each fusible group. Currently text-only (Phase 5a) — no GPU compilation yet.
