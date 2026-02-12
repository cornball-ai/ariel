% tinyrox says don't edit this manually, but it can't stop you!
\name{emit_ttir_softmax}
\alias{emit_ttir_softmax}
\title{Emit Triton MLIR for Fused Row-Wise Softmax}
\usage{
emit_ttir_softmax(
  func_name = "softmax_kernel",
  dtype = "f32",
  block_size = 1024L
)
}
\arguments{
\item{func_name}{Kernel function name}

\item{dtype}{MLIR element type ("f32" or "f16")}

\item{block_size}{Max columns per row (must be power of 2, >= n_cols)}
}
\value{
List with mlir_text, func_name, dtype, block_size, kernel_type.
}
\description{
Generates a numerically stable softmax kernel: one program per row.
Pattern: max(row) -> exp(x - max) -> sum(exp) -> divide.
Handles rows up to \code{block_size} columns. For longer rows,
use a larger block_size (must be power of 2).
Kernel signature: \code{(in_ptr, out_ptr, n_cols)}.
Launch with grid = (n_rows, 1, 1).
}
