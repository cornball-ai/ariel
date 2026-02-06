# ariel

R front-end for the [Triton](https://github.com/triton-lang/triton) GPU compiler. Takes [torchlang](https://github.com/cornball-ai/torchlang) IR graphs and compiles them to GPU kernels via Triton's MLIR infrastructure, bypassing Python entirely.

Named after *The Little Mermaid* — just like King Triton, but [part of R world](https://www.youtube.com/watch?v=SXKlJuO07eM).

## Installation

```r
# Install from GitHub
remotes::install_github("cornball-ai/ariel")
```

Phase 5a (MLIR text emission) works out of the box with no system dependencies. Phase 5b (compilation to PTX) requires building Triton from source — see [Building Triton](#building-triton) below.

## Usage

```r
library(torchlang)
library(ariel)

# Capture and optimize a torch expression
ir <- lower_to_ir(list(quote(x$relu()$sigmoid())))
ir <- optimize_graph(ir)

# Phase 5a: Emit Triton MLIR text (works everywhere)
result <- emit_ttir(ir, group_id = 1L)
cat(result$mlir_text)

# Phase 5b: Compile to PTX (requires Triton)
compiled <- mlir_compile(result, sm = 80L)
compiled$ptx          # PTX assembly string
compiled$kernel_name  # "fused_relu_sigmoid"
compiled$shared_mem   # shared memory bytes
```

### Matmul kernels

```r
ir <- lower_to_ir(list(quote(x %*% y)))
ir <- optimize_graph(ir)
result <- emit_ttir_matmul(ir, group_id = 1L)
compiled <- mlir_compile(result, sm = 80L)
```

### GPU execution

```r
# Create input on GPU
x <- torch_randn(c(16, 16))$cuda()

# Allocate output
output <- torch_zeros_like(x)

# Launch kernel
result <- gpu_launch(
  ptx = compiled$ptx,
  kernel_name = compiled$kernel_name,
  inputs = list(x),
  output = output,
  grid = as.integer(c(1, 1, 1)),
  block = as.integer(c(128, 1, 1)),  # Must match .reqntid in PTX
  shared_mem = as.integer(compiled$shared_mem)
)
```

## Architecture

```
torchlang IR (ir_graph)
      |
      v
Phase 5a: emit_ttir()           Pure R, no dependencies
      |                          Generates Triton MLIR text
      v
Phase 5b: mlir_compile()        Rcpp, links Triton C++ libs
      |                          TTIR -> TTGIR -> LLVM -> PTX
      v
PTX assembly
      |
      v
Phase 5c: gpu_launch()          Rcpp + CUDA driver API
      |                          Launches compiled kernels
      v
GPU execution
```

## Building Triton

`mlir_compile()` requires Triton's C++ MLIR libraries built from source. Phase 5a functions (`emit_ttir`, `emit_ttir_matmul`) work without this.

### Prerequisites

```bash
sudo apt install cmake ninja-build clang lld
```

### Step 1: Clone Triton

```bash
git clone https://github.com/triton-lang/triton.git ~/triton
```

### Step 2: Build LLVM/MLIR

Triton pins a specific LLVM commit. Use the included build script:

```bash
cd ~/triton
# This builds LLVM/MLIR to ~/triton/llvm-project/build/
# Takes ~30 minutes on 8 cores
scripts/build-llvm-project.sh
```

### Step 3: Build Triton C++ libraries

```bash
cd ~/triton
mkdir -p build && cd build
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_LIBRARY_DIR=$HOME/triton/llvm-project/build/lib \
  -DMLIR_DIR=$HOME/triton/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/triton/llvm-project/build/lib/cmake/llvm \
  -DTRITON_CODEGEN_BACKENDS="nvidia" \
  -DTRITON_BUILD_PYTHON_MODULE=OFF \
  -DTRITON_BUILD_UT=OFF \
  -DTRITON_BUILD_PROTON=OFF
ninja
```

Some `bin/` targets may fail due to missing AMD headers — this is expected when building with only the NVIDIA backend. The core libraries will compile successfully.

### Step 4: Create static archive

Bundle the Triton object files into a single archive:

```bash
cd ~/triton/build
mkdir -p lib_static
ar rcs lib_static/libTritonAll.a \
  $(find lib third_party/nvidia third_party/f2reduce -name "*.o" | sort)
```

### Step 5: Install ariel with Triton support

```bash
TRITON_HOME=$HOME/triton R CMD INSTALL path/to/ariel
```

The `configure` script auto-detects `TRITON_HOME`, the build directory, and the LLVM installation. It generates `src/Makevars` with the correct include paths and library flags.

### Verifying the build

```r
library(ariel)

# If this returns a parse error (not "Triton is not available"),
# Triton is linked correctly:
tryCatch(
  mlir_compile(list(mlir_text = "invalid")),
  error = function(e) message(e$message)
)
# "Failed to parse MLIR text..." = Triton working
# "Triton is not available..."   = stub only
```

## Triton Kernel Signature

Triton-compiled PTX kernels have extra parameters beyond what's declared in the TTIR:

**TTIR signature** (3 parameters):
```mlir
tt.func @kernel(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %n_elements: i32)
```

**PTX signature** (5 parameters):
```ptx
.visible .entry kernel(
  .param .u64 .ptr .global param_0,  // input pointer
  .param .u64 .ptr .global param_1,  // output pointer
  .param .u32 param_2,                // n_elements
  .param .u64 .ptr .global param_3,  // metadata (pass NULL)
  .param .u64 .ptr .global param_4   // metadata (pass NULL)
)
```

`gpu_launch()` handles this automatically by passing NULL for the extra metadata parameters.

**Thread count**: Triton kernels specify required thread count via `.reqntid` directive in PTX. The `block` parameter must match this value (typically 128).

## Supported operations

### Elementwise (emit_ttir)

Unary: `neg`, `exp`, `log`, `log2`, `tanh`, `sqrt`, `abs`, `floor`, `ceil`, `sin`, `cos`, `relu`, `sigmoid`, `silu`, `gelu`, `sign`

Binary: `add`, `sub`, `mul`, `div`

### Matmul (emit_ttir_matmul)

Tiled matrix multiplication with optional unary epilogue fusion (e.g., matmul + GELU).

## Development

```bash
# Format, document, install, test
r -e 'rformat::rformat_dir("R"); tinyrox::document(); tinypkgr::install(); tinytest::test_package("ariel")'

# R CMD check
r -e 'tinypkgr::check()'
```

## License

Apache License 2.0
