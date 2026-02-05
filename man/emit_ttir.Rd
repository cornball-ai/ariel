% tinyrox says don't edit this manually, but it can't stop you!
\name{emit_ttir}
\alias{emit_ttir}
\title{Emit Triton MLIR for an Elementwise Fusion Group}
\usage{
emit_ttir(graph, group_id, func_name = NULL, dtype = "f32", block_size = 1024L)
}
\arguments{
\item{graph}{An ir_graph (from torchlang) with fusion annotations}

\item{group_id}{Integer fusion group ID}

\item{func_name}{Optional kernel function name}

\item{dtype}{MLIR element type ("f32" or "f16")}

\item{block_size}{Integer block size (must be power of 2)}
}
\value{
List with mlir_text, func_name, n_inputs, external_input_ids,
  output_id, group_node_ids, dtype, block_size.
  NULL if group contains unsupported ops.
}
\description{
Translates a torchlang IR fusion group to Triton's MLIR textual format.
The output is a complete MLIR module containing a \code{tt.func} with
the pointer-based load/compute/store pattern used by Triton.
}
