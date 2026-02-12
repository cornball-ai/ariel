# test_ir_to_mlir.R - Tests for Phase 5a: TTIR emission

library(ariel)
library(torchlang)

# ============================================================
# triton_op_supported
# ============================================================

expect_true(triton_op_supported("relu"))
expect_true(triton_op_supported("sigmoid"))
expect_true(triton_op_supported("tanh"))
expect_true(triton_op_supported("exp"))
expect_true(triton_op_supported("add"))
expect_true(triton_op_supported("mul"))
expect_true(triton_op_supported("silu"))
expect_true(triton_op_supported("gelu"))
expect_false(triton_op_supported("matmul"))
expect_false(triton_op_supported("imaginary_op"))

# ============================================================
# emit_ttir: simple unary chain (relu -> sigmoid -> tanh)
# ============================================================

ir <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
ir <- optimize_graph(ir)
groups <- torchlang:::get_fusion_groups(ir)

expect_true(length(groups) >= 1L)

result <- emit_ttir(ir, groups[1])

expect_false(is.null(result))
expect_true(is.character(result$mlir_text))
expect_true(nchar(result$mlir_text) > 100)

# Contains expected MLIR patterns (validated against triton/test/TritonGPU/matmul.mlir)
expect_true(grepl("tt.func public", result$mlir_text))
expect_true(grepl("tt.get_program_id x : i32", result$mlir_text))
expect_true(grepl("tt.make_range", result$mlir_text))
expect_true(grepl("tt.load", result$mlir_text))
expect_true(grepl("tt.store", result$mlir_text))
expect_true(grepl("tt.return", result$mlir_text))
expect_true(grepl("tt.splat", result$mlir_text))
expect_true(grepl("tt.addptr", result$mlir_text))
expect_true(grepl("module", result$mlir_text))

# Block size is concrete integer, not SSA variable
expect_true(grepl("tensor<1024x", result$mlir_text))
expect_true(grepl("arith.constant 1024 : i32", result$mlir_text))

# Contains relu pattern (arith.cmpf + arith.select)
expect_true(grepl("arith.cmpf", result$mlir_text))
expect_true(grepl("arith.select", result$mlir_text))

# Contains sigmoid pattern (math.exp + arith.divf)
expect_true(grepl("math.exp", result$mlir_text))
expect_true(grepl("arith.divf", result$mlir_text))

# Contains tanh (decomposed to 2*sigmoid(2x)-1, no math.tanh)
expect_true(grepl("arith.mulf", result$mlir_text))

# Metadata
expect_equal(result$n_inputs, 1L)
expect_equal(length(result$external_input_ids), 1L)
expect_equal(length(result$group_node_ids), 3L)
expect_true(grepl("^fused_", result$func_name))

# ============================================================
# emit_ttir: custom function name
# ============================================================

custom <- emit_ttir(ir, groups[1], func_name = "my_kernel")
expect_equal(custom$func_name, "my_kernel")
expect_true(grepl("@my_kernel", custom$mlir_text))

# ============================================================
# emit_ttir: binary ops (add)
# ============================================================

ir_bin <- lower_to_ir(list(
  quote(a <- x$relu()),
  quote(a + y)
))
ir_bin <- optimize_graph(ir_bin)
bin_groups <- torchlang:::get_fusion_groups(ir_bin)

if (length(bin_groups) > 0L) {
  bin_result <- emit_ttir(ir_bin, bin_groups[1])
  if (!is.null(bin_result)) {
    expect_true(grepl("arith.addf", bin_result$mlir_text))
    expect_true(bin_result$n_inputs >= 1L)
  }
}

# ============================================================
# emit_ttir: unsupported op returns NULL
# ============================================================

fake_ir <- torchlang:::ir_graph(
  nodes = list(
    "1" = torchlang:::ir_node(1L, "input", attrs = list(name = "x")),
    "2" = torchlang:::ir_node(2L, "imaginary_op", inputs = 1L,
                              attrs = list(fusion_group = 99L)),
    "3" = torchlang:::ir_node(3L, "relu", inputs = 2L,
                              attrs = list(fusion_group = 99L))
  ),
  input_ids = 1L,
  output_ids = 3L
)
null_result <- emit_ttir(fake_ir, 99L)
expect_true(is.null(null_result))

# ============================================================
# emit_ttir: dtype parameter
# ============================================================

result_f16 <- emit_ttir(ir, groups[1], dtype = "f16")
expect_true(grepl("f16", result_f16$mlir_text))
expect_equal(result_f16$dtype, "f16")

# ============================================================
# emit_ttir: silu compound op
# ============================================================

ir_silu <- lower_to_ir(list(quote(x$silu()$relu())))
ir_silu <- optimize_graph(ir_silu)
silu_groups <- torchlang:::get_fusion_groups(ir_silu)

if (length(silu_groups) > 0L) {
  silu_result <- emit_ttir(ir_silu, silu_groups[1])
  if (!is.null(silu_result)) {
    # silu decomposes to subf(0,x) + exp + addf + divf
    expect_true(grepl("arith.subf", silu_result$mlir_text))
    expect_true(grepl("math.exp", silu_result$mlir_text))
  }
}

# ============================================================
# emit_ttir: gelu compound op
# ============================================================

ir_gelu <- lower_to_ir(list(quote(x$gelu()$relu())))
ir_gelu <- optimize_graph(ir_gelu)
gelu_groups <- torchlang:::get_fusion_groups(ir_gelu)

if (length(gelu_groups) > 0L) {
  gelu_result <- emit_ttir(ir_gelu, gelu_groups[1])
  if (!is.null(gelu_result)) {
    # gelu decomposes to mulf + sigmoid approx (no math.tanh)
    expect_true(grepl("arith.mulf", gelu_result$mlir_text))
    expect_true(grepl("math.exp", gelu_result$mlir_text))
  }
}

# ============================================================
# emit_ttir: pointer types in signature
# ============================================================

expect_true(grepl("!tt.ptr<f32>", result$mlir_text))

# ============================================================
# emit_ttir: custom block size
# ============================================================

result_bs <- emit_ttir(ir, groups[1], block_size = 512L)
expect_equal(result_bs$block_size, 512L)
expect_true(grepl("tensor<512x", result_bs$mlir_text))
expect_true(grepl("arith.constant 512 : i32", result_bs$mlir_text))

# ============================================================
# emit_ttir_matmul: basic matmul
# ============================================================

mm_result <- emit_ttir_matmul()

expect_false(is.null(mm_result))
expect_true(is.character(mm_result$mlir_text))
expect_true(nchar(mm_result$mlir_text) > 100)

# Contains matmul patterns (modeled after triton/test/TritonGPU/matmul.mlir)
expect_true(grepl("tt.func public @matmul_kernel", mm_result$mlir_text))
expect_true(grepl("tt.dot", mm_result$mlir_text))
expect_true(grepl("scf.for", mm_result$mlir_text))
expect_true(grepl("scf.yield", mm_result$mlir_text))
expect_true(grepl("tt.get_program_id x", mm_result$mlir_text))
expect_true(grepl("tt.get_program_id y", mm_result$mlir_text))

# 2D address computation: expand_dims + broadcast (real Triton pattern)
expect_true(grepl("tt.expand_dims", mm_result$mlir_text))
expect_true(grepl("tt.broadcast", mm_result$mlir_text))

# K-loop carries 3 iter_args: acc, a_ptrs, b_ptrs
expect_true(grepl("iter_args", mm_result$mlir_text))

# Pointer advancement inside loop
expect_true(grepl("a_ptrs_next", mm_result$mlir_text))
expect_true(grepl("b_ptrs_next", mm_result$mlir_text))

# Boundary mask with arith.andi
expect_true(grepl("arith.andi", mm_result$mlir_text))

# tt.divisibility annotations on pointer args
expect_true(grepl("tt.divisibility = 16", mm_result$mlir_text))

# ============================================================
# emit_ttir_matmul: with relu epilogue
# ============================================================

mm_relu <- emit_ttir_matmul(epilogue_ops = "relu", func_name = "matmul_relu")
expect_false(is.null(mm_relu))
expect_true(grepl("@matmul_relu", mm_relu$mlir_text))
expect_true(grepl("arith.cmpf", mm_relu$mlir_text))  # relu pattern
expect_equal(mm_relu$epilogue_ops, "relu")

# ============================================================
# emit_ttir_matmul: unsupported epilogue returns NULL
# ============================================================

mm_bad <- emit_ttir_matmul(epilogue_ops = "imaginary_op")
expect_true(is.null(mm_bad))

# ============================================================
# emit_ttir_matmul: custom tile sizes
# ============================================================

mm_custom <- emit_ttir_matmul(block_m = 128L, block_n = 128L, block_k = 64L)
expect_equal(mm_custom$block_m, 128L)
expect_equal(mm_custom$block_n, 128L)
expect_equal(mm_custom$block_k, 64L)

# ============================================================
# emit_ttir: input validation
# ============================================================

expect_error(emit_ttir("not a graph", 1L), "Expected an ir_graph")
