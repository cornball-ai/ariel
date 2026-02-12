% tinyrox says don't edit this manually, but it can't stop you!
\name{emit_ttir_layer_norm}
\alias{emit_ttir_layer_norm}
\title{Emit Triton MLIR for Fused Layer Normalization}
\usage{
emit_ttir_layer_norm(
  func_name = "layer_norm_kernel",
  dtype = "f32",
  block_size = 1024L
)
}
\arguments{
\item{func_name}{Kernel function name}

\item{dtype}{MLIR element type ("f32" or "f16")}

\item{block_size}{Max columns per row (power of 2)}
}
\value{
List with mlir_text, func_name, dtype, block_size, kernel_type.
}
\description{
Generates a fused layer_norm kernel: one program per row.
Pattern: mean(row) -> var(row) -> (x - mean) / sqrt(var + eps) * weight + bias.
Handles rows up to \code{block_size} columns.
Kernel signature: \code{(in_ptr, weight_ptr, bias_ptr, out_ptr, n_cols, eps)}.
Launch with grid = (n_rows, 1, 1).
}
