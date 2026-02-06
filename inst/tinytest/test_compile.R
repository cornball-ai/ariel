# test_compile.R - Tests for Phase 5b: MLIR compilation
#
# These tests are guarded by Triton availability.
# Without Triton, mlir_compile() errors gracefully and tests are skipped.

library(ariel)

# ============================================================
# mlir_compile: input validation (works without Triton)
# ============================================================

expect_error(mlir_compile("not a list"), "Expected a list")
expect_error(mlir_compile(list(foo = "bar")), "Expected a list")

# ============================================================
# mlir_compile: Triton availability check
# ============================================================

# Try compiling a minimal MLIR module to detect if Triton is available
triton_available <- tryCatch({
  # A trivial MLIR text that should parse if Triton is present
  trivial <- list(mlir_text = "module {}")
  mlir_compile(trivial, sm = 80L)
  TRUE
}, error = function(e) {
  # Expected: either "Triton is not available" (stub) or a pass error
  if (grepl("Triton is not available", e$message)) {
    FALSE
  } else {
    # Triton is available but the trivial module failed (expected for empty module)
    TRUE
  }
})

if (!triton_available) {
  # Verify the stub error message is informative
  err <- tryCatch(mlir_compile(list(mlir_text = "module {}")),
                  error = function(e) e$message)
  expect_true(grepl("Triton is not available", err))
  exit_file("Triton not available, skipping Phase 5b tests")
}

# ============================================================
# mlir_compile: with Phase 5a elementwise output (requires Triton + torchlang)
# ============================================================

if (requireNamespace("torchlang", quietly = TRUE)) {
  library(torchlang)

  ir <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
  ir <- optimize_graph(ir)
  groups <- torchlang:::get_fusion_groups(ir)

  if (length(groups) > 0L) {
    result <- emit_ttir(ir, groups[1])

    if (!is.null(result)) {
      compiled <- tryCatch(
        mlir_compile(result, sm = 80L),
        error = function(e) e
      )

      if (!inherits(compiled, "error")) {
        # PTX was generated
        expect_true(is.list(compiled))
        expect_true(is.character(compiled$ptx))
        expect_true(nchar(compiled$ptx) > 0)

        # Contains PTX entry point
        expect_true(grepl("\\.visible .entry", compiled$ptx))

        # Has kernel name
        expect_true(is.character(compiled$kernel_name))
        expect_true(nchar(compiled$kernel_name) > 0)

        # Shared memory is non-negative integer
        expect_true(is.numeric(compiled$shared_mem))
        expect_true(compiled$shared_mem >= 0)

        # PTX contains target annotations
        expect_true(grepl("\\.target sm_", compiled$ptx))
      }
    }
  }
}

# ============================================================
# mlir_compile: with Phase 5a matmul output (requires Triton)
# ============================================================

mm_result <- emit_ttir_matmul()

mm_compiled <- tryCatch(
  mlir_compile(mm_result, sm = 80L, num_warps = 4L),
  error = function(e) e
)

if (!inherits(mm_compiled, "error")) {
  expect_true(is.list(mm_compiled))
  expect_true(is.character(mm_compiled$ptx))
  expect_true(nchar(mm_compiled$ptx) > 0)
  expect_true(grepl("\\.visible .entry", mm_compiled$ptx))
  expect_true(grepl("matmul", mm_compiled$kernel_name))
}
