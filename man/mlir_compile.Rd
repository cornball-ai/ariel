% tinyrox says don't edit this manually, but it can't stop you!
\name{mlir_compile}
\alias{mlir_compile}
\title{Compile Triton MLIR to PTX}
\usage{
mlir_compile(
  result,
  sm = 80L,
  num_warps = 4L,
  num_ctas = 1L,
  ptx_version = NULL
)
}
\arguments{
\item{result}{List returned by \code{emit_ttir()} or \code{emit_ttir_matmul()},
must contain a \code{mlir_text} element.}

\item{sm}{Integer compute capability (e.g., 80 for A100, 90 for H100).}

\item{num_warps}{Integer number of warps per block.}

\item{num_ctas}{Integer number of CTAs in a cluster.}

\item{ptx_version}{Integer PTX ISA version (e.g., 83 for PTX 8.3).
If NULL, defaults based on compute capability.}
}
\value{
List with:
  \describe{
    \item{ptx}{Character, the PTX assembly string.}
    \item{kernel_name}{Character, the entry point function name.}
    \item{shared_mem}{Integer, shared memory bytes required.}
  }
}
\description{
Takes the output of \code{emit_ttir()} or \code{emit_ttir_matmul()} and
compiles it through Triton's full MLIR pipeline to produce PTX assembly.
Requires Triton built from source (see Details).
}
\details{
Triton must be built from source to use this function:
\preformatted{
git clone https://github.com/triton-lang/triton.git ~/triton
cd ~/triton && pip install -e python
}
Then reinstall ariel with \code{TRITON_HOME=~/triton} set.
Without Triton, this function stops with an informative error.
Phase 5a functions (\code{emit_ttir}, \code{emit_ttir_matmul}) work
regardless.
}
