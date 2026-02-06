% tinyrox says don't edit this manually, but it can't stop you!
\name{gpu_launch}
\alias{gpu_launch}
\title{Launch a GPU Kernel from Compiled PTX}
\usage{
gpu_launch(ptx, kernel_name, inputs, output, grid, block, shared_mem = 0L)
}
\arguments{
\item{ptx}{Character, PTX assembly from mlir_compile()}

\item{kernel_name}{Character, entry point function name}

\item{inputs}{List of torch_tensor inputs (must be on CUDA)}

\item{output}{torch_tensor pre-allocated output (must be on CUDA)}

\item{grid}{Integer vector of length 3, grid dimensions (blocks)}

\item{block}{Integer vector of length 3, block dimensions (threads)}

\item{shared_mem}{Integer, dynamic shared memory bytes}
}
\value{
The output tensor (same as input)
}
\description{
Launch a GPU Kernel from Compiled PTX
}
