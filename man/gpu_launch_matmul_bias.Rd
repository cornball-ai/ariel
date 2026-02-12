% tinyrox says don't edit this manually, but it can't stop you!
\name{gpu_launch_matmul_bias}
\alias{gpu_launch_matmul_bias}
\title{Launch a Matmul+Bias GPU Kernel from Compiled PTX}
\usage{
gpu_launch_matmul_bias(
  ptx,
  kernel_name,
  A,
  B,
  bias,
  C,
  M,
  N,
  K,
  stride_am,
  stride_ak,
  stride_bk,
  stride_bn,
  stride_cm,
  stride_cn,
  grid,
  block,
  shared_mem = 0L
)
}
\arguments{
\item{ptx}{Character, PTX assembly from mlir_compile()}

\item{kernel_name}{Character, entry point function name}

\item{A}{torch_tensor, matrix A (must be on CUDA)}

\item{B}{torch_tensor, matrix B (must be on CUDA)}

\item{bias}{torch_tensor, 1D bias vector (must be on CUDA)}

\item{C}{torch_tensor, pre-allocated output matrix (must be on CUDA)}

\item{M}{Integer, rows of A / rows of C}

\item{N}{Integer, cols of B / cols of C}

\item{K}{Integer, cols of A / rows of B}

\item{stride_am}{Integer, stride of A along M dimension}

\item{stride_ak}{Integer, stride of A along K dimension}

\item{stride_bk}{Integer, stride of B along K dimension}

\item{stride_bn}{Integer, stride of B along N dimension}

\item{stride_cm}{Integer, stride of C along M dimension}

\item{stride_cn}{Integer, stride of C along N dimension}

\item{grid}{Integer vector of length 3, grid dimensions}

\item{block}{Integer vector of length 3, block dimensions}

\item{shared_mem}{Integer, dynamic shared memory bytes}
}
\value{
The output tensor C
}
\description{
Like gpu_launch_matmul but with an extra bias pointer parameter.
The kernel layout is: 4 pointer args (A, B, bias, C), 9 i32 scalars
(M, N, K, strides), and 2 null metadata pointers added by Triton.
}
