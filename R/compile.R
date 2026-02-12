#' Compile Triton MLIR to PTX
#'
#' Takes the output of \code{emit_ttir()} or \code{emit_ttir_matmul()} and
#' compiles it through Triton's full MLIR pipeline to produce PTX assembly.
#' Requires Triton built from source (see Details).
#'
#' Results are cached on disk so repeated compilations of the same kernel
#' (same MLIR text, same target parameters) are instant.
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

  # Check disk cache
  cache_key <- .ptx_cache_key(result$mlir_text, sm, num_warps, num_ctas,
                               ptx_version)
  cached <- .ptx_cache_get(cache_key)
  if (!is.null(cached)) return(cached)

  # Compile
  compiled <- compile_mlir_to_ptx(result$mlir_text, sm, num_warps, num_ctas,
                                   ptx_version)

  # Cache result
  .ptx_cache_set(cache_key, compiled)

  compiled
}


# PTX disk cache directory
.ptx_cache_dir <- function() {
  d <- file.path(tools::R_user_dir("ariel", "cache"), "ptx")
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  d
}

# Hash for cache key
.ptx_cache_key <- function(mlir_text, sm, num_warps, num_ctas, ptx_version) {
  tmp <- tempfile()
  on.exit(unlink(tmp))
  writeLines(paste(mlir_text, sm, num_warps, num_ctas, ptx_version, sep = "|"),
             tmp)
  unname(tools::md5sum(tmp))
}

# Retrieve from disk cache
.ptx_cache_get <- function(key) {
  meta_path <- file.path(.ptx_cache_dir(), paste0(key, ".rds"))
  if (!file.exists(meta_path)) return(NULL)
  tryCatch(readRDS(meta_path), error = function(e) NULL)
}

# Store to disk cache
.ptx_cache_set <- function(key, compiled) {
  cache_dir <- .ptx_cache_dir()
  meta_path <- file.path(cache_dir, paste0(key, ".rds"))
  tryCatch(saveRDS(compiled, meta_path), error = function(e) NULL)
}


#' Load Pre-compiled PTX into GPU Kernel Cache
#'
#' Loads a PTX string directly into ariel's in-memory kernel cache,
#' bypassing compilation. Used by artifact loaders to restore compiled
#' kernels from disk.
#'
#' @param ptx Character, PTX assembly text.
#' @param kernel_name Character, the kernel entry point name.
#' @param shared_mem Integer, shared memory bytes required.
#' @return Invisibly returns a list with ptx, kernel_name, shared_mem.
#' @export
gpu_cache_load_ptx <- function(ptx, kernel_name, shared_mem = 0L) {
  result <- list(ptx = ptx, kernel_name = kernel_name,
                 shared_mem = as.integer(shared_mem))
  invisible(result)
}


#' Clear PTX Disk Cache
#'
#' @return Number of cached entries cleared (invisibly).
#' @export
clear_ptx_cache <- function() {
  cache_dir <- .ptx_cache_dir()
  if (!dir.exists(cache_dir)) return(invisible(0L))
  files <- list.files(cache_dir, full.names = TRUE)
  n <- length(files)
  if (n > 0L) unlink(files)
  invisible(n)
}


#' PTX Cache Statistics
#'
#' @return List with n_cached and cache_dir.
#' @export
ptx_cache_stats <- function() {
  cache_dir <- .ptx_cache_dir()
  files <- if (dir.exists(cache_dir)) list.files(cache_dir) else character()
  list(n_cached = length(files), cache_dir = cache_dir)
}
