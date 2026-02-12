% tinyrox says don't edit this manually, but it can't stop you!
\name{emit_ttir_matmul}
\alias{emit_ttir_matmul}
\title{Emit Triton MLIR for Tiled Matmul with Epilogue Fusion}
\usage{
emit_ttir_matmul(
  epilogue_ops = character(),
  func_name = "matmul_kernel",
  dtype = "f32",
  has_bias = FALSE,
  block_m = 64L,
  block_n = 64L,
  block_k = 64L
)
}
\arguments{
\item{epilogue_ops}{Character vector of elementwise ops to fuse}

\item{func_name}{Kernel function name}

\item{dtype}{MLIR element type ("f32" or "f16")}

\item{block_m}{Tile size M}

\item{block_n}{Tile size N}

\item{block_k}{Tile size K}
}
\value{
List with mlir_text, func_name, epilogue_ops, dtype,
  block_m, block_n, block_k.
  NULL if epilogue contains unsupported ops.
}
\description{
Generates MLIR for a tiled matrix multiplication following Triton's
canonical matmul pattern: 2D grid, \code{expand_dims} + \code{broadcast}
for address computation, \code{scf.for} K-loop with pointer advancement,
\code{tt.dot} accumulation, optional fused epilogue, masked store.
Modeled after \code{triton/test/TritonGPU/matmul.mlir}.
}
