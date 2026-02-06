// triton_stub.cpp — Fallback when Triton is not available
//
// Provides the same Rcpp-exported function signature so R CMD check passes
// and the package installs cleanly without Triton. Calling mlir_compile()
// gives a clear error message.

#ifndef HAS_TRITON

#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List compile_mlir_to_ptx(std::string mlir_text,
                                int compute_capability,
                                int num_warps,
                                int num_ctas,
                                int ptx_version) {
  Rcpp::stop(
    "Triton is not available. "
    "Phase 5b (MLIR compilation) requires Triton built from source. "
    "See ?mlir_compile for setup instructions."
  );
}

#endif // !HAS_TRITON
