#' Compile Triton MLIR to PTX
#'
#' Takes the output of \code{emit_ttir()} or \code{emit_ttir_matmul()} and
#' compiles it through Triton's full MLIR pipeline to produce PTX assembly.
#' Requires Triton built from source (see Details).
#'
#' @param result List returned by \code{emit_ttir()} or \code{emit_ttir_matmul()},
#'   must contain a \code{mlir_text} element.
#' @param sm Integer compute capability (e.g., 80 for A100, 90 for H100).
#' @param num_warps Integer number of warps per block.
#' @param num_ctas Integer number of CTAs in a cluster.
#' @param ptx_version Integer PTX ISA version (e.g., 83 for PTX 8.3).
#'   If NULL, defaults based on compute capability.
#' @return List with:
#'   \describe{
#'     \item{ptx}{Character, the PTX assembly string.}
#'     \item{kernel_name}{Character, the entry point function name.}
#'     \item{shared_mem}{Integer, shared memory bytes required.}
#'   }
#' @details
#' Triton must be built from source to use this function:
#' \preformatted{
#' git clone https://github.com/triton-lang/triton.git ~/triton
#' cd ~/triton && pip install -e python
#' }
#' Then reinstall ariel with \code{TRITON_HOME=~/triton} set.
#' Without Triton, this function stops with an informative error.
#' Phase 5a functions (\code{emit_ttir}, \code{emit_ttir_matmul}) work
#' regardless.
#'
#' @useDynLib ariel, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @export
mlir_compile <- function(result, sm = 80L, num_warps = 4L, num_ctas = 1L,
                          ptx_version = NULL) {
  if (!is.list(result) || is.null(result$mlir_text))
    stop("Expected a list with 'mlir_text' (from emit_ttir or emit_ttir_matmul)",
         call. = FALSE)

  sm <- as.integer(sm)
  num_warps <- as.integer(num_warps)
  num_ctas <- as.integer(num_ctas)

  if (is.null(ptx_version)) {
    # Default PTX versions by compute capability
    ptx_version <- if (sm >= 90L) 83L else if (sm >= 80L) 81L else 75L
  }
  ptx_version <- as.integer(ptx_version)

  compile_mlir_to_ptx(result$mlir_text, sm, num_warps, num_ctas, ptx_version)
}
