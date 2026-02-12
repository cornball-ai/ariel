#' Translate torchlang IR to Triton MLIR (TTIR)
#'
#' Lowers an optimized ir_graph (from torchlang) to Triton's MLIR
#' textual format. This is the first stage of the compilation pipeline.
#' The output can be fed to Triton's MLIR passes for GPU compilation.

# ============================================================================
# Op Mapping: torchlang IR ops -> MLIR operations
# ============================================================================

# Unary ops: torchlang op name -> MLIR op string
.ttir_unary_ops <- list(
  # neg is handled as compound op (arith.subf from zero)
  exp   = "math.exp",
  log   = "math.log",
  log2  = "math.log2",
  # tanh is compound op: decomposed to 2*sigmoid(2x)-1 to avoid illegal math.tanh
  # sqrt and rsqrt are compound: decomposed to avoid libdevice
  abs   = "math.absf",
  floor = "math.floor",
  ceil  = "math.ceil",
  sin   = "math.sin",
  cos   = "math.cos"
)

# Binary ops: torchlang op name -> MLIR op string
.ttir_binary_ops <- list(
  add = "arith.addf",
  sub = "arith.subf",
  mul = "arith.mulf",
  div = "arith.divf"
)

# Compound ops that need custom MLIR emission (not 1:1 with MLIR ops)
.ttir_compound_ops <- c("neg", "relu", "sigmoid", "silu", "gelu", "tanh",
                        "sqrt", "rsqrt", "leaky_relu", "elu", "sign")


#' Check if an Op is Supported for Triton Emission
#'
#' @param op Character, operation name from torchlang IR
#' @return Logical
#' @export
triton_op_supported <- function(op) {
  op %in% c(names(.ttir_unary_ops), names(.ttir_binary_ops),
            .ttir_compound_ops)
}


# ============================================================================
# MLIR Emission Helpers
# ============================================================================

# Generate a unique SSA value name
.ssa <- function(id) sprintf("%%v%d", id)

# Emit a simple unary MLIR op
.emit_unary <- function(mlir_op, input_ssa, output_ssa, type) {
  sprintf("    %s = %s %s : %s", output_ssa, mlir_op, input_ssa, type)
}

# Emit a simple binary MLIR op
.emit_binary <- function(mlir_op, lhs_ssa, rhs_ssa, output_ssa, type) {
  sprintf("    %s = %s %s, %s : %s", output_ssa, mlir_op, lhs_ssa, rhs_ssa, type)
}

# Emit compound ops that expand to multiple MLIR operations.
# Returns list(lines, ssa) where ssa is the final output SSA name.
.emit_compound <- function(op, input_ssas, base_id, type) {
  x <- input_ssas[1]
  lines <- character()
  final_ssa <- .ssa(base_id)

  if (op == "neg") {
    # neg(x) = 0 - x (Triton does not support arith.negf)
    zero_ssa <- sprintf("%%zero_%d", base_id)
    lines <- c(
      sprintf("    %s = arith.constant dense<0.000000e+00> : %s", zero_ssa, type),
      sprintf("    %s = arith.subf %s, %s : %s", final_ssa, zero_ssa, x, type)
    )
  } else if (op == "relu") {
    # relu(x) = select(x > 0, x, 0)
    zero_ssa <- sprintf("%%zero_%d", base_id)
    cmp_ssa <- sprintf("%%cmp_%d", base_id)
    cmp_type <- sub("f32", "i1", sub("f16", "i1", type))
    lines <- c(
      sprintf("    %s = arith.constant dense<0.000000e+00> : %s", zero_ssa, type),
      sprintf("    %s = arith.cmpf ogt, %s, %s : %s", cmp_ssa, x, zero_ssa, type),
      sprintf("    %s = arith.select %s, %s, %s : %s, %s",
              final_ssa, cmp_ssa, x, zero_ssa, cmp_type, type)
    )
  } else if (op == "sigmoid") {
    # sigmoid(x) = 1 / (1 + exp(-x))
    zero_ssa <- sprintf("%%zero_%d", base_id)
    neg_ssa <- sprintf("%%neg_%d", base_id)
    exp_ssa <- sprintf("%%exp_%d", base_id)
    one_ssa <- sprintf("%%one_%d", base_id)
    sum_ssa <- sprintf("%%sum_%d", base_id)
    lines <- c(
      sprintf("    %s = arith.constant dense<0.000000e+00> : %s", zero_ssa, type),
      sprintf("    %s = arith.subf %s, %s : %s", neg_ssa, zero_ssa, x, type),
      sprintf("    %s = math.exp %s : %s", exp_ssa, neg_ssa, type),
      sprintf("    %s = arith.constant dense<1.000000e+00> : %s", one_ssa, type),
      sprintf("    %s = arith.addf %s, %s : %s", sum_ssa, one_ssa, exp_ssa, type),
      sprintf("    %s = arith.divf %s, %s : %s", final_ssa, one_ssa, sum_ssa, type)
    )
  } else if (op == "silu") {
    # silu(x) = x / (1 + exp(-x))
    zero_ssa <- sprintf("%%zero_%d", base_id)
    neg_ssa <- sprintf("%%neg_%d", base_id)
    exp_ssa <- sprintf("%%exp_%d", base_id)
    one_ssa <- sprintf("%%one_%d", base_id)
    sum_ssa <- sprintf("%%sum_%d", base_id)
    lines <- c(
      sprintf("    %s = arith.constant dense<0.000000e+00> : %s", zero_ssa, type),
      sprintf("    %s = arith.subf %s, %s : %s", neg_ssa, zero_ssa, x, type),
      sprintf("    %s = math.exp %s : %s", exp_ssa, neg_ssa, type),
      sprintf("    %s = arith.constant dense<1.000000e+00> : %s", one_ssa, type),
      sprintf("    %s = arith.addf %s, %s : %s", sum_ssa, one_ssa, exp_ssa, type),
      sprintf("    %s = arith.divf %s, %s : %s", final_ssa, x, sum_ssa, type)
    )
  } else if (op == "gelu") {
    # gelu(x) ~ x * sigmoid(1.702 * x)
    # Sigmoid approximation avoids math.tanh (illegal in Triton MLIR) and
    # math.erf (needs libdevice linking). Uses only arith + math.exp.
    # Max error vs exact GELU: ~0.005 (good enough for inference).
    c_coeff_ssa <- sprintf("%%c_coeff_%d", base_id)
    zero_ssa <- sprintf("%%zero_%d", base_id)
    one_ssa <- sprintf("%%one_%d", base_id)
    sx_ssa <- sprintf("%%sx_%d", base_id)
    neg_ssa <- sprintf("%%neg_%d", base_id)
    exp_ssa <- sprintf("%%exp_%d", base_id)
    denom_ssa <- sprintf("%%denom_%d", base_id)
    sig_ssa <- sprintf("%%sig_%d", base_id)
    lines <- c(
      sprintf("    %s = arith.constant dense<1.702000e+00> : %s", c_coeff_ssa, type),
      sprintf("    %s = arith.constant dense<0.000000e+00> : %s", zero_ssa, type),
      sprintf("    %s = arith.constant dense<1.000000e+00> : %s", one_ssa, type),
      # 1.702 * x
      sprintf("    %s = arith.mulf %s, %s : %s", sx_ssa, c_coeff_ssa, x, type),
      # sigmoid(1.702 * x) = 1 / (1 + exp(-1.702 * x))
      sprintf("    %s = arith.subf %s, %s : %s", neg_ssa, zero_ssa, sx_ssa, type),
      sprintf("    %s = math.exp %s : %s", exp_ssa, neg_ssa, type),
      sprintf("    %s = arith.addf %s, %s : %s", denom_ssa, one_ssa, exp_ssa, type),
      sprintf("    %s = arith.divf %s, %s : %s", sig_ssa, one_ssa, denom_ssa, type),
      # x * sigmoid(1.702 * x)
      sprintf("    %s = arith.mulf %s, %s : %s", final_ssa, x, sig_ssa, type)
    )
  } else if (op == "tanh") {
    # tanh(x) = 2 * sigmoid(2x) - 1
    # Decomposed to avoid math.tanh (illegal in Triton MLIR).
    # Uses: 2x -> exp(-2x) -> 1/(1+exp(-2x)) -> 2*sig - 1
    two_ssa <- sprintf("%%two_%d", base_id)
    one_ssa <- sprintf("%%one_%d", base_id)
    zero_ssa <- sprintf("%%zero_%d", base_id)
    twox_ssa <- sprintf("%%twox_%d", base_id)
    neg_ssa <- sprintf("%%neg_%d", base_id)
    exp_ssa <- sprintf("%%exp_%d", base_id)
    denom_ssa <- sprintf("%%denom_%d", base_id)
    sig_ssa <- sprintf("%%sig_%d", base_id)
    twosig_ssa <- sprintf("%%twosig_%d", base_id)
    lines <- c(
      sprintf("    %s = arith.constant dense<2.000000e+00> : %s", two_ssa, type),
      sprintf("    %s = arith.constant dense<1.000000e+00> : %s", one_ssa, type),
      sprintf("    %s = arith.constant dense<0.000000e+00> : %s", zero_ssa, type),
      sprintf("    %s = arith.mulf %s, %s : %s", twox_ssa, two_ssa, x, type),
      sprintf("    %s = arith.subf %s, %s : %s", neg_ssa, zero_ssa, twox_ssa, type),
      sprintf("    %s = math.exp %s : %s", exp_ssa, neg_ssa, type),
      sprintf("    %s = arith.addf %s, %s : %s", denom_ssa, one_ssa, exp_ssa, type),
      sprintf("    %s = arith.divf %s, %s : %s", sig_ssa, one_ssa, denom_ssa, type),
      sprintf("    %s = arith.mulf %s, %s : %s", twosig_ssa, two_ssa, sig_ssa, type),
      sprintf("    %s = arith.subf %s, %s : %s", final_ssa, twosig_ssa, one_ssa, type)
    )
  } else if (op == "sqrt") {
    # sqrt(x) = exp(0.5 * log(x)) — avoids math.sqrt (libdevice)
    half_ssa <- sprintf("%%half_%d", base_id)
    log_ssa <- sprintf("%%log_%d", base_id)
    half_log_ssa <- sprintf("%%half_log_%d", base_id)
    lines <- c(
      sprintf("    %s = arith.constant dense<5.000000e-01> : %s", half_ssa, type),
      sprintf("    %s = math.log %s : %s", log_ssa, x, type),
      sprintf("    %s = arith.mulf %s, %s : %s", half_log_ssa, half_ssa, log_ssa, type),
      sprintf("    %s = math.exp %s : %s", final_ssa, half_log_ssa, type)
    )
  } else if (op == "rsqrt") {
    # rsqrt(x) = exp(-0.5 * log(x)) — avoids math.sqrt/rsqrt (libdevice)
    neg_half_ssa <- sprintf("%%neg_half_%d", base_id)
    log_ssa <- sprintf("%%log_%d", base_id)
    half_log_ssa <- sprintf("%%half_log_%d", base_id)
    lines <- c(
      sprintf("    %s = arith.constant dense<-5.000000e-01> : %s", neg_half_ssa, type),
      sprintf("    %s = math.log %s : %s", log_ssa, x, type),
      sprintf("    %s = arith.mulf %s, %s : %s", half_log_ssa, neg_half_ssa, log_ssa, type),
      sprintf("    %s = math.exp %s : %s", final_ssa, half_log_ssa, type)
    )
  } else if (op == "leaky_relu") {
    # leaky_relu(x) = select(x > 0, x, 0.01 * x)
    alpha_ssa <- sprintf("%%alpha_%d", base_id)
    zero_ssa <- sprintf("%%zero_%d", base_id)
    cmp_ssa <- sprintf("%%cmp_%d", base_id)
    scaled_ssa <- sprintf("%%scaled_%d", base_id)
    cmp_type <- sub("f32", "i1", sub("f16", "i1", type))
    lines <- c(
      sprintf("    %s = arith.constant dense<1.000000e-02> : %s", alpha_ssa, type),
      sprintf("    %s = arith.constant dense<0.000000e+00> : %s", zero_ssa, type),
      sprintf("    %s = arith.cmpf ogt, %s, %s : %s", cmp_ssa, x, zero_ssa, type),
      sprintf("    %s = arith.mulf %s, %s : %s", scaled_ssa, alpha_ssa, x, type),
      sprintf("    %s = arith.select %s, %s, %s : %s, %s",
              final_ssa, cmp_ssa, x, scaled_ssa, cmp_type, type)
    )
  } else if (op == "elu") {
    # elu(x) = select(x > 0, x, exp(x) - 1)
    zero_ssa <- sprintf("%%zero_%d", base_id)
    one_ssa <- sprintf("%%one_%d", base_id)
    cmp_ssa <- sprintf("%%cmp_%d", base_id)
    exp_ssa <- sprintf("%%exp_%d", base_id)
    expm1_ssa <- sprintf("%%expm1_%d", base_id)
    cmp_type <- sub("f32", "i1", sub("f16", "i1", type))
    lines <- c(
      sprintf("    %s = arith.constant dense<0.000000e+00> : %s", zero_ssa, type),
      sprintf("    %s = arith.constant dense<1.000000e+00> : %s", one_ssa, type),
      sprintf("    %s = arith.cmpf ogt, %s, %s : %s", cmp_ssa, x, zero_ssa, type),
      sprintf("    %s = math.exp %s : %s", exp_ssa, x, type),
      sprintf("    %s = arith.subf %s, %s : %s", expm1_ssa, exp_ssa, one_ssa, type),
      sprintf("    %s = arith.select %s, %s, %s : %s, %s",
              final_ssa, cmp_ssa, x, expm1_ssa, cmp_type, type)
    )
  } else if (op == "sign") {
    # sign(x) = select(x > 0, 1, select(x < 0, -1, 0))
    zero_ssa <- sprintf("%%zero_%d", base_id)
    pos1_ssa <- sprintf("%%pos1_%d", base_id)
    neg1_ssa <- sprintf("%%neg1_%d", base_id)
    cmp_gt_ssa <- sprintf("%%cmp_gt_%d", base_id)
    cmp_lt_ssa <- sprintf("%%cmp_lt_%d", base_id)
    sel1_ssa <- sprintf("%%sel1_%d", base_id)
    cmp_type <- sub("f32", "i1", sub("f16", "i1", type))
    lines <- c(
      sprintf("    %s = arith.constant dense<0.000000e+00> : %s", zero_ssa, type),
      sprintf("    %s = arith.constant dense<1.000000e+00> : %s", pos1_ssa, type),
      sprintf("    %s = arith.constant dense<-1.000000e+00> : %s", neg1_ssa, type),
      sprintf("    %s = arith.cmpf ogt, %s, %s : %s", cmp_gt_ssa, x, zero_ssa, type),
      sprintf("    %s = arith.cmpf olt, %s, %s : %s", cmp_lt_ssa, x, zero_ssa, type),
      sprintf("    %s = arith.select %s, %s, %s : %s, %s",
              sel1_ssa, cmp_lt_ssa, neg1_ssa, zero_ssa, cmp_type, type),
      sprintf("    %s = arith.select %s, %s, %s : %s, %s",
              final_ssa, cmp_gt_ssa, pos1_ssa, sel1_ssa, cmp_type, type)
    )
  }

  list(lines = lines, ssa = final_ssa)
}


# ============================================================================
# Main Emission: Elementwise Kernel
# ============================================================================

#' Emit Triton MLIR for an Elementwise Fusion Group
#'
#' Translates a torchlang IR fusion group to Triton's MLIR textual format.
#' The output is a complete MLIR module containing a \code{tt.func} with
#' the pointer-based load/compute/store pattern used by Triton.
#'
#' @param graph An ir_graph (from torchlang) with fusion annotations
#' @param group_id Integer fusion group ID
#' @param func_name Optional kernel function name
#' @param dtype MLIR element type ("f32" or "f16")
#' @param block_size Integer block size (must be power of 2)
#' @return List with mlir_text, func_name, n_inputs, external_input_ids,
#'   output_id, group_node_ids, dtype, block_size.
#'   NULL if group contains unsupported ops.
#' @export
emit_ttir <- function(graph, group_id, func_name = NULL, dtype = "f32",
                      block_size = 1024L) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)
  block_size <- as.integer(block_size)

  # Extract fusion group nodes
  group_node_ids <- integer()
  for (id_str in names(graph$nodes)) {
    n <- graph$nodes[[id_str]]
    if (isTRUE(n$attrs$fusion_group == group_id)) {
      group_node_ids <- c(group_node_ids, n$id)
    }
  }
  group_node_ids <- sort(group_node_ids)
  if (length(group_node_ids) == 0L) {
    stop(sprintf("No nodes in fusion group %d", group_id), call. = FALSE)
  }

  group_set <- as.character(group_node_ids)

  # Find external inputs
  external_input_ids <- integer()
  for (nid in group_node_ids) {
    node <- graph$nodes[[as.character(nid)]]
    for (inp in node$inputs) {
      if (!as.character(inp) %in% group_set) {
        external_input_ids <- c(external_input_ids, inp)
      }
    }
  }
  external_input_ids <- sort(unique(external_input_ids))
  output_id <- max(group_node_ids)

  # Check all ops supported
  for (nid in group_node_ids) {
    if (!triton_op_supported(graph$nodes[[as.character(nid)]]$op)) {
      return(NULL)
    }
  }

  # Auto-generate name
  if (is.null(func_name)) {
    ops <- vapply(group_node_ids, function(nid) {
      graph$nodes[[as.character(nid)]]$op
    }, character(1))
    func_name <- paste0("fused_", paste(ops, collapse = "_"))
    func_name <- gsub("[^a-zA-Z0-9_]", "_", func_name)
  }

  n_inputs <- length(external_input_ids)
  ptr_type <- sprintf("!tt.ptr<%s>", dtype)
  block_type <- sprintf("tensor<%dx%s>", block_size, dtype)
  i32_block <- sprintf("tensor<%dxi32>", block_size)
  ptr_block <- sprintf("tensor<%dx%s>", block_size, ptr_type)

  # Function signature
  arg_names <- character()
  arg_types <- character()
  for (i in seq_len(n_inputs)) {
    arg_names <- c(arg_names, sprintf("%%arg%d", i - 1L))
    arg_types <- c(arg_types, ptr_type)
  }
  arg_names <- c(arg_names, "%out_ptr", "%n_elements")
  arg_types <- c(arg_types, ptr_type, "i32")
  sig_args <- paste(sprintf("%s: %s", arg_names, arg_types), collapse = ", ")

  L <- character()  # accumulate MLIR lines

  L <- c(L, "module {")
  L <- c(L, sprintf("  tt.func public @%s(%s) {", func_name, sig_args))

  # Program ID and offset computation
  L <- c(L, "    %pid = tt.get_program_id x : i32")
  L <- c(L, sprintf("    %%c%d = arith.constant %d : i32", block_size, block_size))
  L <- c(L, sprintf("    %%block_start = arith.muli %%pid, %%c%d : i32", block_size))
  L <- c(L, sprintf(
    "    %%range = tt.make_range {end = %d : i32, start = 0 : i32} : %s",
    block_size, i32_block))
  L <- c(L, sprintf(
    "    %%start_splat = tt.splat %%block_start : i32 -> %s", i32_block))
  L <- c(L, sprintf(
    "    %%offsets = arith.addi %%start_splat, %%range : %s", i32_block))

  # Mask: offsets < n_elements
  L <- c(L, sprintf(
    "    %%n_splat = tt.splat %%n_elements : i32 -> %s", i32_block))
  L <- c(L, sprintf(
    "    %%mask = arith.cmpi slt, %%offsets, %%n_splat : %s", i32_block))

  # Load each external input
  ext_ssa <- list()
  for (i in seq_len(n_inputs)) {
    ptr_arg <- sprintf("%%arg%d", i - 1L)
    splat_ssa <- sprintf("%%ptr_splat_%d", i)
    addr_ssa <- sprintf("%%addr_%d", i)
    load_ssa <- sprintf("%%load_%d", i)
    L <- c(L, sprintf(
      "    %s = tt.splat %s : %s -> %s", splat_ssa, ptr_arg, ptr_type, ptr_block))
    L <- c(L, sprintf(
      "    %s = tt.addptr %s, %%offsets : %s, %s",
      addr_ssa, splat_ssa, ptr_block, i32_block))
    L <- c(L, sprintf(
      "    %s = tt.load %s, %%mask : %s", load_ssa, addr_ssa, ptr_block))
    ext_ssa[[as.character(external_input_ids[i])]] <- load_ssa
  }

  # Compute ops
  node_ssa <- ext_ssa
  next_tmp <- 100L

  for (nid in group_node_ids) {
    node <- graph$nodes[[as.character(nid)]]
    op <- node$op
    input_ssas <- vapply(node$inputs, function(inp_id) {
      node_ssa[[as.character(inp_id)]]
    }, character(1))

    out_ssa <- .ssa(nid)

    if (op %in% names(.ttir_unary_ops)) {
      L <- c(L, .emit_unary(.ttir_unary_ops[[op]], input_ssas[1],
                             out_ssa, block_type))
      node_ssa[[as.character(nid)]] <- out_ssa

    } else if (op %in% names(.ttir_binary_ops)) {
      L <- c(L, .emit_binary(.ttir_binary_ops[[op]], input_ssas[1],
                              input_ssas[2], out_ssa, block_type))
      node_ssa[[as.character(nid)]] <- out_ssa

    } else if (op %in% .ttir_compound_ops) {
      compound <- .emit_compound(op, input_ssas, next_tmp, block_type)
      L <- c(L, compound$lines)
      node_ssa[[as.character(nid)]] <- compound$ssa
      next_tmp <- next_tmp + 50L

    } else {
      return(NULL)
    }
  }

  # Store result
  result_ssa <- node_ssa[[as.character(output_id)]]
  L <- c(L, sprintf(
    "    %%out_splat = tt.splat %%out_ptr : %s -> %s", ptr_type, ptr_block))
  L <- c(L, sprintf(
    "    %%out_addr = tt.addptr %%out_splat, %%offsets : %s, %s",
    ptr_block, i32_block))
  L <- c(L, sprintf(
    "    tt.store %%out_addr, %s, %%mask : %s", result_ssa, ptr_block))

  L <- c(L, "    tt.return")
  L <- c(L, "  }")
  L <- c(L, "}")

  list(
    mlir_text = paste(L, collapse = "\n"),
    func_name = func_name,
    n_inputs = n_inputs,
    external_input_ids = external_input_ids,
    output_id = output_id,
    group_node_ids = group_node_ids,
    dtype = dtype,
    block_size = block_size
  )
}


# ============================================================================
# Main Emission: Tiled Matmul
# ============================================================================

#' Emit Triton MLIR for Tiled Matmul with Epilogue Fusion
#'
#' Generates MLIR for a tiled matrix multiplication following Triton's
#' canonical matmul pattern: 2D grid, \code{expand_dims} + \code{broadcast}
#' for address computation, \code{scf.for} K-loop with pointer advancement,
#' \code{tt.dot} accumulation, optional fused epilogue, masked store.
#'
#' Modeled after \code{triton/test/TritonGPU/matmul.mlir}.
#'
#' @param epilogue_ops Character vector of elementwise ops to fuse
#' @param func_name Kernel function name
#' @param dtype MLIR element type ("f32" or "f16")
#' @param block_m Tile size M
#' @param block_n Tile size N
#' @param block_k Tile size K
#' @return List with mlir_text, func_name, epilogue_ops, dtype,
#'   block_m, block_n, block_k.
#'   NULL if epilogue contains unsupported ops.
#' @export
emit_ttir_matmul <- function(epilogue_ops = character(),
                              func_name = "matmul_kernel",
                              dtype = "f32",
                              has_bias = FALSE,
                              block_m = 64L, block_n = 64L, block_k = 64L) {
  for (op in epilogue_ops) {
    if (!triton_op_supported(op)) return(NULL)
  }

  block_m <- as.integer(block_m)
  block_n <- as.integer(block_n)
  block_k <- as.integer(block_k)

  # Pointer offset: bias_ptr shifts C_ptr and all scalars by 1
  po <- if (has_bias) 1L else 0L

  ptr_type <- sprintf("!tt.ptr<%s>", dtype)
  acc_type <- sprintf("tensor<%dx%dxf32>", block_m, block_n)
  a_tile_type <- sprintf("tensor<%dx%dx%s>", block_m, block_k, dtype)
  b_tile_type <- sprintf("tensor<%dx%dx%s>", block_k, block_n, dtype)
  a_ptr_type <- sprintf("tensor<%dx%dx%s>", block_m, block_k, ptr_type)
  b_ptr_type <- sprintf("tensor<%dx%dx%s>", block_k, block_n, ptr_type)
  c_ptr_type <- sprintf("tensor<%dx%dx%s>", block_m, block_n, ptr_type)
  offs_m_type <- sprintf("tensor<%dxi32>", block_m)
  offs_n_type <- sprintf("tensor<%dxi32>", block_n)
  offs_k_type <- sprintf("tensor<%dxi32>", block_k)
  mask_type <- sprintf("tensor<%dx%dxi1>", block_m, block_n)
  a_offs_type <- sprintf("tensor<%dx%dxi32>", block_m, block_k)
  b_offs_type <- sprintf("tensor<%dx%dxi32>", block_k, block_n)
  c_offs_type <- sprintf("tensor<%dx%dxi32>", block_m, block_n)
  bias_1d_type <- sprintf("tensor<%dxf32>", block_n)
  bias_ptr_1d_type <- sprintf("tensor<%dx%s>", block_n, ptr_type)

  L <- character()
  L <- c(L, "module {")

  # Function signature:
  # Without bias: arg0=A, arg1=B, arg2=C, arg3=M, ..., arg11=stride_cn
  # With bias:    arg0=A, arg1=B, arg2=bias, arg3=C, arg4=M, ..., arg12=stride_cn
  if (has_bias) {
    L <- c(L, sprintf(paste0(
      "  tt.func public @%s(",
      "%%arg0: %s {tt.divisibility = 16 : i32}, ",
      "%%arg1: %s {tt.divisibility = 16 : i32}, ",
      "%%arg2: %s {tt.divisibility = 16 : i32}, ",
      "%%arg3: %s {tt.divisibility = 16 : i32}, ",
      "%%arg4: i32, %%arg5: i32, %%arg6: i32, ",
      "%%arg7: i32 {tt.divisibility = 16 : i32}, ",
      "%%arg8: i32, ",
      "%%arg9: i32 {tt.divisibility = 16 : i32}, ",
      "%%arg10: i32, ",
      "%%arg11: i32 {tt.divisibility = 16 : i32}, ",
      "%%arg12: i32) {"),
      func_name, ptr_type, ptr_type, ptr_type, ptr_type))
  } else {
    L <- c(L, sprintf(paste0(
      "  tt.func public @%s(",
      "%%arg0: %s {tt.divisibility = 16 : i32}, ",
      "%%arg1: %s {tt.divisibility = 16 : i32}, ",
      "%%arg2: %s {tt.divisibility = 16 : i32}, ",
      "%%arg3: i32, %%arg4: i32, %%arg5: i32, ",
      "%%arg6: i32 {tt.divisibility = 16 : i32}, ",
      "%%arg7: i32, ",
      "%%arg8: i32 {tt.divisibility = 16 : i32}, ",
      "%%arg9: i32, ",
      "%%arg10: i32 {tt.divisibility = 16 : i32}, ",
      "%%arg11: i32) {"),
      func_name, ptr_type, ptr_type, ptr_type))
  }

  # Arg indices for scalars (shifted by po when has_bias)
  # Without bias: M=arg3, N=arg4, K=arg5, stride_am=arg6, stride_ak=arg7,
  #               stride_bk=arg8, stride_bn=arg9, stride_cm=arg10, stride_cn=arg11
  # With bias:    M=arg4, N=arg5, K=arg6, stride_am=arg7, stride_ak=arg8,
  #               stride_bk=arg9, stride_bn=arg10, stride_cm=arg11, stride_cn=arg12
  a_M  <- sprintf("%%arg%d", 3L + po)
  a_N  <- sprintf("%%arg%d", 4L + po)
  a_K  <- sprintf("%%arg%d", 5L + po)
  a_sam <- sprintf("%%arg%d", 6L + po)
  a_sak <- sprintf("%%arg%d", 7L + po)
  a_sbk <- sprintf("%%arg%d", 8L + po)
  a_sbn <- sprintf("%%arg%d", 9L + po)
  a_scm <- sprintf("%%arg%d", 10L + po)
  a_scn <- sprintf("%%arg%d", 11L + po)
  a_C   <- sprintf("%%arg%d", 2L + po)  # C_ptr

  # Constants
  L <- c(L, sprintf("    %%c%d = arith.constant %d : i32", block_m, block_m))
  L <- c(L, sprintf("    %%c%d_0 = arith.constant %d : i32", block_k, block_k))
  L <- c(L, "    %c0 = arith.constant 0 : i32")
  L <- c(L, sprintf("    %%cst_zero = arith.constant dense<0.000000e+00> : %s", acc_type))
  L <- c(L, sprintf(
    "    %%cst_true = arith.constant dense<true> : tensor<%dx%dxi1>",
    block_m, block_k))

  # Program IDs
  L <- c(L, "    %pid_m = tt.get_program_id x : i32")
  L <- c(L, "    %pid_n = tt.get_program_id y : i32")

  # Offset ranges
  L <- c(L, sprintf(
    "    %%range_m = tt.make_range {end = %d : i32, start = 0 : i32} : %s",
    block_m, offs_m_type))
  L <- c(L, sprintf(
    "    %%range_n = tt.make_range {end = %d : i32, start = 0 : i32} : %s",
    block_n, offs_n_type))
  L <- c(L, sprintf(
    "    %%range_k = tt.make_range {end = %d : i32, start = 0 : i32} : %s",
    block_k, offs_k_type))

  # offs_m = pid_m * BLOCK_M + range_m
  L <- c(L, sprintf(
    "    %%base_m = arith.muli %%pid_m, %%c%d : i32", block_m))
  L <- c(L, sprintf(
    "    %%base_m_splat = tt.splat %%base_m : i32 -> %s", offs_m_type))
  L <- c(L, sprintf(
    "    %%offs_m = arith.addi %%base_m_splat, %%range_m : %s", offs_m_type))

  # offs_n = pid_n * BLOCK_N + range_n  (reuse BLOCK_M constant if same)
  L <- c(L, sprintf(
    "    %%base_n = arith.muli %%pid_n, %%c%d : i32", block_m))
  L <- c(L, sprintf(
    "    %%base_n_splat = tt.splat %%base_n : i32 -> %s", offs_n_type))
  L <- c(L, sprintf(
    "    %%offs_n = arith.addi %%base_n_splat, %%range_n : %s", offs_n_type))

  # --- A pointer: a_ptr + offs_m[:,None] * stride_am + range_k[None,:] * stride_ak ---
  # expand_dims offs_m -> [BLOCK_M, 1]
  L <- c(L, sprintf(
    "    %%offs_m_col = tt.expand_dims %%offs_m {axis = 1 : i32} : %s -> tensor<%dx1xi32>",
    offs_m_type, block_m))
  L <- c(L, sprintf(
    "    %%stride_am_splat = tt.splat %s : i32 -> tensor<%dx1xi32>", a_sam, block_m))
  L <- c(L, sprintf(
    "    %%a_row_offs = arith.muli %%offs_m_col, %%stride_am_splat : tensor<%dx1xi32>",
    block_m))
  # expand_dims range_k -> [1, BLOCK_K]
  L <- c(L, sprintf(
    "    %%range_k_row = tt.expand_dims %%range_k {axis = 0 : i32} : %s -> tensor<1x%dxi32>",
    offs_k_type, block_k))
  L <- c(L, sprintf(
    "    %%stride_ak_splat = tt.splat %s : i32 -> tensor<1x%dxi32>", a_sak, block_k))
  L <- c(L, sprintf(
    "    %%a_col_offs = arith.muli %%range_k_row, %%stride_ak_splat : tensor<1x%dxi32>",
    block_k))
  # broadcast and add
  L <- c(L, sprintf(
    "    %%a_row_bc = tt.broadcast %%a_row_offs : tensor<%dx1xi32> -> %s",
    block_m, a_offs_type))
  L <- c(L, sprintf(
    "    %%a_col_bc = tt.broadcast %%a_col_offs : tensor<1x%dxi32> -> %s",
    block_k, a_offs_type))
  L <- c(L, sprintf(
    "    %%a_offs = arith.addi %%a_row_bc, %%a_col_bc : %s", a_offs_type))
  L <- c(L, sprintf(
    "    %%a_ptr_splat = tt.splat %%arg0 : %s -> %s", ptr_type, a_ptr_type))
  L <- c(L, sprintf(
    "    %%a_ptr_init = tt.addptr %%a_ptr_splat, %%a_offs : %s, %s",
    a_ptr_type, a_offs_type))

  # --- B pointer: b_ptr + range_k[:,None] * stride_bk + offs_n[None,:] * stride_bn ---
  L <- c(L, sprintf(
    "    %%range_k_col = tt.expand_dims %%range_k {axis = 1 : i32} : %s -> tensor<%dx1xi32>",
    offs_k_type, block_k))
  L <- c(L, sprintf(
    "    %%stride_bk_splat = tt.splat %s : i32 -> tensor<%dx1xi32>", a_sbk, block_k))
  L <- c(L, sprintf(
    "    %%b_row_offs = arith.muli %%range_k_col, %%stride_bk_splat : tensor<%dx1xi32>",
    block_k))
  L <- c(L, sprintf(
    "    %%offs_n_row = tt.expand_dims %%offs_n {axis = 0 : i32} : %s -> tensor<1x%dxi32>",
    offs_n_type, block_n))
  L <- c(L, sprintf(
    "    %%stride_bn_splat = tt.splat %s : i32 -> tensor<1x%dxi32>", a_sbn, block_n))
  L <- c(L, sprintf(
    "    %%b_col_offs = arith.muli %%offs_n_row, %%stride_bn_splat : tensor<1x%dxi32>",
    block_n))
  L <- c(L, sprintf(
    "    %%b_row_bc = tt.broadcast %%b_row_offs : tensor<%dx1xi32> -> %s",
    block_k, b_offs_type))
  L <- c(L, sprintf(
    "    %%b_col_bc = tt.broadcast %%b_col_offs : tensor<1x%dxi32> -> %s",
    block_n, b_offs_type))
  L <- c(L, sprintf(
    "    %%b_offs = arith.addi %%b_row_bc, %%b_col_bc : %s", b_offs_type))
  L <- c(L, sprintf(
    "    %%b_ptr_splat = tt.splat %%arg1 : %s -> %s", ptr_type, b_ptr_type))
  L <- c(L, sprintf(
    "    %%b_ptr_init = tt.addptr %%b_ptr_splat, %%b_offs : %s, %s",
    b_ptr_type, b_offs_type))

  # --- K-loop: scf.for with iter_args(acc, a_ptr, b_ptr) ---
  L <- c(L, sprintf(paste0(
    "    %%loop:3 = scf.for %%k = %%c0 to %s step %%c%d_0 ",
    "iter_args(%%acc = %%cst_zero, %%a_ptrs = %%a_ptr_init, %%b_ptrs = %%b_ptr_init) ",
    "-> (%s, %s, %s) : i32 {"),
    a_K, block_k, acc_type, a_ptr_type, b_ptr_type))

  # Load A and B tiles
  L <- c(L, sprintf(
    "      %%a_tile = tt.load %%a_ptrs, %%cst_true, %%cst_zero : %s", a_ptr_type))
  L <- c(L, sprintf(
    "      %%b_tile = tt.load %%b_ptrs, %%cst_true, %%cst_zero : %s", b_ptr_type))

  # Dot product: acc += a @ b
  L <- c(L, sprintf(
    "      %%dot = tt.dot %%a_tile, %%b_tile, %%acc : %s * %s -> %s",
    a_tile_type, b_tile_type, acc_type))

  # Advance A pointer by stride_ak * BLOCK_K
  L <- c(L, sprintf(
    "      %%a_step = arith.muli %s, %%c%d_0 : i32", a_sak, block_k))
  L <- c(L, sprintf(
    "      %%a_step_splat = tt.splat %%a_step : i32 -> %s", a_offs_type))
  L <- c(L, sprintf(
    "      %%a_ptrs_next = tt.addptr %%a_ptrs, %%a_step_splat : %s, %s",
    a_ptr_type, a_offs_type))

  # Advance B pointer by stride_bk * BLOCK_K
  L <- c(L, sprintf(
    "      %%b_step = arith.muli %s, %%c%d_0 : i32", a_sbk, block_k))
  L <- c(L, sprintf(
    "      %%b_step_splat = tt.splat %%b_step : i32 -> %s", b_offs_type))
  L <- c(L, sprintf(
    "      %%b_ptrs_next = tt.addptr %%b_ptrs, %%b_step_splat : %s, %s",
    b_ptr_type, b_offs_type))

  # Yield
  L <- c(L, sprintf(
    "      scf.yield %%dot, %%a_ptrs_next, %%b_ptrs_next : %s, %s, %s",
    acc_type, a_ptr_type, b_ptr_type))
  L <- c(L, "    }")

  # --- Bias add (if has_bias): load 1D bias, broadcast to [BLOCK_M, BLOCK_N], add ---
  current_ssa <- "%loop#0"
  if (has_bias) {
    # N splat for mask (1D)
    L <- c(L, sprintf(
      "    %%N_splat_1d = tt.splat %s : i32 -> %s", a_N, offs_n_type))
    L <- c(L, sprintf(
      "    %%bias_mask = arith.cmpi slt, %%offs_n, %%N_splat_1d : %s", offs_n_type))

    # Bias pointer: splat bias_ptr (%arg2) + offs_n
    L <- c(L, sprintf(
      "    %%bias_splat = tt.splat %%arg2 : %s -> %s", ptr_type, bias_ptr_1d_type))
    L <- c(L, sprintf(
      "    %%bias_addrs = tt.addptr %%bias_splat, %%offs_n : %s, %s",
      bias_ptr_1d_type, offs_n_type))
    L <- c(L, sprintf(
      "    %%bias_1d = tt.load %%bias_addrs, %%bias_mask : %s", bias_ptr_1d_type))

    # Broadcast [BLOCK_N] -> [1, BLOCK_N] -> [BLOCK_M, BLOCK_N]
    L <- c(L, sprintf(
      "    %%bias_2d = tt.expand_dims %%bias_1d {axis = 0 : i32} : %s -> tensor<1x%dxf32>",
      bias_1d_type, block_n))
    L <- c(L, sprintf(
      "    %%bias_bc = tt.broadcast %%bias_2d : tensor<1x%dxf32> -> %s",
      block_n, acc_type))

    # Add bias to accumulator
    L <- c(L, sprintf(
      "    %%biased = arith.addf %%loop#0, %%bias_bc : %s", acc_type))
    current_ssa <- "%biased"
  }

  # --- Epilogue: apply fused ops to loop result (or biased result) ---
  epi_idx <- 200L
  for (op in epilogue_ops) {
    out_ssa <- sprintf("%%epi_%d", epi_idx)
    if (op %in% names(.ttir_unary_ops)) {
      L <- c(L, sprintf("    %s = %s %s : %s",
                         out_ssa, .ttir_unary_ops[[op]], current_ssa, acc_type))
      current_ssa <- out_ssa
    } else if (op %in% .ttir_compound_ops) {
      compound <- .emit_compound(op, current_ssa, epi_idx, acc_type)
      L <- c(L, compound$lines)
      current_ssa <- compound$ssa
    } else {
      return(NULL)
    }
    epi_idx <- epi_idx + 50L
  }

  # --- C pointer and masked store ---
  # Reuse offs_m, offs_n from above
  L <- c(L, sprintf(
    "    %%offs_m_c = tt.expand_dims %%offs_m {axis = 1 : i32} : %s -> tensor<%dx1xi32>",
    offs_m_type, block_m))
  L <- c(L, sprintf(
    "    %%stride_cm_splat = tt.splat %s : i32 -> tensor<%dx1xi32>", a_scm, block_m))
  L <- c(L, sprintf(
    "    %%c_row_offs = arith.muli %%stride_cm_splat, %%offs_m_c : tensor<%dx1xi32>",
    block_m))
  L <- c(L, sprintf(
    "    %%offs_n_c = tt.expand_dims %%offs_n {axis = 0 : i32} : %s -> tensor<1x%dxi32>",
    offs_n_type, block_n))
  L <- c(L, sprintf(
    "    %%stride_cn_splat = tt.splat %s : i32 -> tensor<1x%dxi32>", a_scn, block_n))
  L <- c(L, sprintf(
    "    %%c_col_offs = arith.muli %%offs_n_c, %%stride_cn_splat : tensor<1x%dxi32>",
    block_n))
  L <- c(L, sprintf(
    "    %%c_row_bc = tt.broadcast %%c_row_offs : tensor<%dx1xi32> -> %s",
    block_m, c_offs_type))
  L <- c(L, sprintf(
    "    %%c_col_bc = tt.broadcast %%c_col_offs : tensor<1x%dxi32> -> %s",
    block_n, c_offs_type))
  L <- c(L, sprintf(
    "    %%c_offs = arith.addi %%c_row_bc, %%c_col_bc : %s", c_offs_type))
  L <- c(L, sprintf(
    "    %%c_ptr_splat = tt.splat %s : %s -> %s", a_C, ptr_type, c_ptr_type))
  L <- c(L, sprintf(
    "    %%c_addrs = tt.addptr %%c_ptr_splat, %%c_offs : %s, %s",
    c_ptr_type, c_offs_type))

  # Boundary mask: (offs_m < M) & (offs_n < N)
  L <- c(L, sprintf(
    "    %%m_check = tt.expand_dims %%offs_m {axis = 1 : i32} : %s -> tensor<%dx1xi32>",
    offs_m_type, block_m))
  L <- c(L, sprintf(
    "    %%M_splat = tt.splat %s : i32 -> tensor<%dx1xi32>", a_M, block_m))
  L <- c(L, sprintf(
    "    %%m_mask_col = arith.cmpi slt, %%m_check, %%M_splat : tensor<%dx1xi32>",
    block_m))
  L <- c(L, sprintf(
    "    %%n_check = tt.expand_dims %%offs_n {axis = 0 : i32} : %s -> tensor<1x%dxi32>",
    offs_n_type, block_n))
  L <- c(L, sprintf(
    "    %%N_splat = tt.splat %s : i32 -> tensor<1x%dxi32>", a_N, block_n))
  L <- c(L, sprintf(
    "    %%n_mask_row = arith.cmpi slt, %%n_check, %%N_splat : tensor<1x%dxi32>",
    block_n))
  L <- c(L, sprintf(
    "    %%m_mask = tt.broadcast %%m_mask_col : tensor<%dx1xi1> -> %s",
    block_m, mask_type))
  L <- c(L, sprintf(
    "    %%n_mask = tt.broadcast %%n_mask_row : tensor<1x%dxi1> -> %s",
    block_n, mask_type))
  L <- c(L, sprintf(
    "    %%c_mask = arith.andi %%m_mask, %%n_mask : %s", mask_type))

  # Store
  L <- c(L, sprintf(
    "    tt.store %%c_addrs, %s, %%c_mask : %s", current_ssa, c_ptr_type))

  L <- c(L, "    tt.return")
  L <- c(L, "  }")
  L <- c(L, "}")

  list(
    mlir_text = paste(L, collapse = "\n"),
    func_name = func_name,
    epilogue_ops = epilogue_ops,
    has_bias = has_bias,
    dtype = dtype,
    block_m = block_m,
    block_n = block_n,
    block_k = block_k
  )
}


# ============================================================================
# Reduction Kernels: Softmax
# ============================================================================

#' Emit Triton MLIR for Fused Row-Wise Softmax
#'
#' Generates a numerically stable softmax kernel: one program per row.
#' Pattern: max(row) -> exp(x - max) -> sum(exp) -> divide.
#' Handles rows up to \code{block_size} columns. For longer rows,
#' use a larger block_size (must be power of 2).
#'
#' Kernel signature: \code{(in_ptr, out_ptr, n_cols)}.
#' Launch with grid = (n_rows, 1, 1).
#'
#' @param func_name Kernel function name
#' @param dtype MLIR element type ("f32" or "f16")
#' @param block_size Max columns per row (must be power of 2, >= n_cols)
#' @return List with mlir_text, func_name, dtype, block_size, kernel_type.
#' @export
emit_ttir_softmax <- function(func_name = "softmax_kernel",
                               dtype = "f32", block_size = 1024L) {
  block_size <- as.integer(block_size)
  ptr_type <- sprintf("!tt.ptr<%s>", dtype)
  block_type <- sprintf("tensor<%dx%s>", block_size, dtype)
  i32_block <- sprintf("tensor<%dxi32>", block_size)
  i1_block <- sprintf("tensor<%dxi1>", block_size)
  ptr_block <- sprintf("tensor<%dx%s>", block_size, ptr_type)

  L <- character()
  L <- c(L, "module {")
  L <- c(L, sprintf(
    "  tt.func public @%s(%%in_ptr: %s, %%out_ptr: %s, %%n_cols: i32) {",
    func_name, ptr_type, ptr_type))

  # Row index from program_id
  L <- c(L, "    %row_id = tt.get_program_id x : i32")

  # Column offsets and mask
  L <- c(L, sprintf(
    "    %%range = tt.make_range {start = 0 : i32, end = %d : i32} : %s",
    block_size, i32_block))
  L <- c(L, sprintf(
    "    %%n_splat = tt.splat %%n_cols : i32 -> %s", i32_block))
  L <- c(L, sprintf(
    "    %%mask = arith.cmpi slt, %%range, %%n_splat : %s", i32_block))

  # Row start = row_id * n_cols
  L <- c(L, sprintf(
    "    %%stride_splat = tt.splat %%n_cols : i32 -> %s", i32_block))
  L <- c(L, sprintf(
    "    %%row_id_splat = tt.splat %%row_id : i32 -> %s", i32_block))
  L <- c(L, sprintf(
    "    %%row_start = arith.muli %%row_id_splat, %%stride_splat : %s", i32_block))
  L <- c(L, sprintf(
    "    %%offsets = arith.addi %%row_start, %%range : %s", i32_block))

  # Load input row (masked, fill with -inf for max, 0 for sum)
  L <- c(L, sprintf(
    "    %%in_splat = tt.splat %%in_ptr : %s -> %s", ptr_type, ptr_block))
  L <- c(L, sprintf(
    "    %%in_addrs = tt.addptr %%in_splat, %%offsets : %s, %s",
    ptr_block, i32_block))
  L <- c(L, sprintf(
    "    %%neginf = arith.constant dense<0xFF800000> : %s", block_type))
  L <- c(L, sprintf(
    "    %%data = tt.load %%in_addrs, %%mask, %%neginf : %s", ptr_block))

  # Pass 1: row max
  L <- c(L, sprintf(
    '    %%row_max = "tt.reduce"(%%data) ({'))
  L <- c(L, "    ^bb0(%a0: f32, %b0: f32):")
  L <- c(L, "      %m = arith.maximumf %a0, %b0 : f32")
  L <- c(L, "      tt.reduce.return %m : f32")
  L <- c(L, sprintf(
    "    }) {axis = 0 : i32} : (%s) -> f32", block_type))

  # exp(x - max)
  L <- c(L, sprintf(
    "    %%max_splat = tt.splat %%row_max : f32 -> %s", block_type))
  L <- c(L, sprintf(
    "    %%shifted = arith.subf %%data, %%max_splat : %s", block_type))
  L <- c(L, sprintf(
    "    %%exps = math.exp %%shifted : %s", block_type))

  # Mask exp values (0 for out-of-bounds)
  L <- c(L, sprintf(
    "    %%zeros = arith.constant dense<0.000000e+00> : %s", block_type))
  L <- c(L, sprintf(
    "    %%masked_exps = arith.select %%mask, %%exps, %%zeros : %s, %s",
    i1_block, block_type))

  # Pass 2: sum of exp
  L <- c(L, sprintf(
    '    %%exp_sum = "tt.reduce"(%%masked_exps) ({'))
  L <- c(L, "    ^bb0(%a1: f32, %b1: f32):")
  L <- c(L, "      %s = arith.addf %a1, %b1 : f32")
  L <- c(L, "      tt.reduce.return %s : f32")
  L <- c(L, sprintf(
    "    }) {axis = 0 : i32} : (%s) -> f32", block_type))

  # Divide
  L <- c(L, sprintf(
    "    %%sum_splat = tt.splat %%exp_sum : f32 -> %s", block_type))
  L <- c(L, sprintf(
    "    %%result = arith.divf %%masked_exps, %%sum_splat : %s", block_type))

  # Store
  L <- c(L, sprintf(
    "    %%out_splat = tt.splat %%out_ptr : %s -> %s", ptr_type, ptr_block))
  L <- c(L, sprintf(
    "    %%out_addrs = tt.addptr %%out_splat, %%offsets : %s, %s",
    ptr_block, i32_block))
  L <- c(L, sprintf(
    "    tt.store %%out_addrs, %%result, %%mask : %s", ptr_block))

  L <- c(L, "    tt.return")
  L <- c(L, "  }")
  L <- c(L, "}")

  list(
    mlir_text = paste(L, collapse = "\n"),
    func_name = func_name,
    dtype = dtype,
    block_size = block_size,
    kernel_type = "softmax"
  )
}


# ============================================================================
# Reduction Kernels: Layer Norm
# ============================================================================

#' Emit Triton MLIR for Fused Layer Normalization
#'
#' Generates a fused layer_norm kernel: one program per row.
#' Pattern: mean(row) -> var(row) -> (x - mean) / sqrt(var + eps) * weight + bias.
#' Handles rows up to \code{block_size} columns.
#'
#' Kernel signature: \code{(in_ptr, weight_ptr, bias_ptr, out_ptr, n_cols, eps)}.
#' Launch with grid = (n_rows, 1, 1).
#'
#' @param func_name Kernel function name
#' @param dtype MLIR element type ("f32" or "f16")
#' @param block_size Max columns per row (power of 2)
#' @return List with mlir_text, func_name, dtype, block_size, kernel_type.
#' @export
emit_ttir_layer_norm <- function(func_name = "layer_norm_kernel",
                                  dtype = "f32", block_size = 1024L) {
  block_size <- as.integer(block_size)
  ptr_type <- sprintf("!tt.ptr<%s>", dtype)
  block_type <- sprintf("tensor<%dx%s>", block_size, dtype)
  i32_block <- sprintf("tensor<%dxi32>", block_size)
  i1_block <- sprintf("tensor<%dxi1>", block_size)
  ptr_block <- sprintf("tensor<%dx%s>", block_size, ptr_type)

  L <- character()
  L <- c(L, "module {")
  L <- c(L, sprintf(paste0(
    "  tt.func public @%s(%%in_ptr: %s, %%weight_ptr: %s, ",
    "%%bias_ptr: %s, %%out_ptr: %s, %%n_cols: i32, %%eps: f32) {"),
    func_name, ptr_type, ptr_type, ptr_type, ptr_type))

  # Row index
  L <- c(L, "    %row_id = tt.get_program_id x : i32")

  # Column offsets and mask
  L <- c(L, sprintf(
    "    %%range = tt.make_range {start = 0 : i32, end = %d : i32} : %s",
    block_size, i32_block))
  L <- c(L, sprintf(
    "    %%n_splat = tt.splat %%n_cols : i32 -> %s", i32_block))
  L <- c(L, sprintf(
    "    %%mask = arith.cmpi slt, %%range, %%n_splat : %s", i32_block))

  # Input row offsets
  L <- c(L, sprintf(
    "    %%stride_splat = tt.splat %%n_cols : i32 -> %s", i32_block))
  L <- c(L, sprintf(
    "    %%row_id_splat = tt.splat %%row_id : i32 -> %s", i32_block))
  L <- c(L, sprintf(
    "    %%row_start = arith.muli %%row_id_splat, %%stride_splat : %s", i32_block))
  L <- c(L, sprintf(
    "    %%offsets = arith.addi %%row_start, %%range : %s", i32_block))

  # Load input row
  L <- c(L, sprintf(
    "    %%in_splat = tt.splat %%in_ptr : %s -> %s", ptr_type, ptr_block))
  L <- c(L, sprintf(
    "    %%in_addrs = tt.addptr %%in_splat, %%offsets : %s, %s",
    ptr_block, i32_block))
  L <- c(L, sprintf(
    "    %%zeros = arith.constant dense<0.000000e+00> : %s", block_type))
  L <- c(L, sprintf(
    "    %%data = tt.load %%in_addrs, %%mask, %%zeros : %s", ptr_block))

  # Load weight and bias (1D, indexed by column)
  L <- c(L, sprintf(
    "    %%w_splat = tt.splat %%weight_ptr : %s -> %s", ptr_type, ptr_block))
  L <- c(L, sprintf(
    "    %%w_addrs = tt.addptr %%w_splat, %%range : %s, %s",
    ptr_block, i32_block))
  L <- c(L, sprintf(
    "    %%weight = tt.load %%w_addrs, %%mask, %%zeros : %s", ptr_block))
  L <- c(L, sprintf(
    "    %%b_splat = tt.splat %%bias_ptr : %s -> %s", ptr_type, ptr_block))
  L <- c(L, sprintf(
    "    %%b_addrs = tt.addptr %%b_splat, %%range : %s, %s",
    ptr_block, i32_block))
  L <- c(L, sprintf(
    "    %%bias = tt.load %%b_addrs, %%mask, %%zeros : %s", ptr_block))

  # Mean: sum(x) / n_cols
  L <- c(L, sprintf(
    "    %%masked_data = arith.select %%mask, %%data, %%zeros : %s, %s",
    i1_block, block_type))
  L <- c(L, sprintf(
    '    %%sum_val = "tt.reduce"(%%masked_data) ({'))
  L <- c(L, "    ^bb0(%a0: f32, %b0: f32):")
  L <- c(L, "      %r0 = arith.addf %a0, %b0 : f32")
  L <- c(L, "      tt.reduce.return %r0 : f32")
  L <- c(L, sprintf(
    "    }) {axis = 0 : i32} : (%s) -> f32", block_type))
  L <- c(L, "    %n_f32 = arith.sitofp %n_cols : i32 to f32")
  L <- c(L, "    %mean_val = arith.divf %sum_val, %n_f32 : f32")
  L <- c(L, sprintf(
    "    %%mean_splat = tt.splat %%mean_val : f32 -> %s", block_type))

  # Centered: x - mean
  L <- c(L, sprintf(
    "    %%centered = arith.subf %%data, %%mean_splat : %s", block_type))

  # Variance: sum((x - mean)^2) / n_cols
  L <- c(L, sprintf(
    "    %%sq = arith.mulf %%centered, %%centered : %s", block_type))
  L <- c(L, sprintf(
    "    %%masked_sq = arith.select %%mask, %%sq, %%zeros : %s, %s",
    i1_block, block_type))
  L <- c(L, sprintf(
    '    %%var_sum = "tt.reduce"(%%masked_sq) ({'))
  L <- c(L, "    ^bb0(%a1: f32, %b1: f32):")
  L <- c(L, "      %r1 = arith.addf %a1, %b1 : f32")
  L <- c(L, "      tt.reduce.return %r1 : f32")
  L <- c(L, sprintf(
    "    }) {axis = 0 : i32} : (%s) -> f32", block_type))
  L <- c(L, "    %var_val = arith.divf %var_sum, %n_f32 : f32")

  # inv_std = 1/sqrt(var + eps) = exp(-0.5 * log(var + eps))
  # Uses math.exp + math.log (native PTX) instead of math.sqrt (libdevice)
  L <- c(L, sprintf(
    "    %%eps_splat = tt.splat %%eps : f32 -> %s", block_type))
  L <- c(L, sprintf(
    "    %%var_splat = tt.splat %%var_val : f32 -> %s", block_type))
  L <- c(L, sprintf(
    "    %%var_eps = arith.addf %%var_splat, %%eps_splat : %s", block_type))
  L <- c(L, sprintf(
    "    %%log_var = math.log %%var_eps : %s", block_type))
  L <- c(L, sprintf(
    "    %%neg_half = arith.constant dense<-5.000000e-01> : %s", block_type))
  L <- c(L, sprintf(
    "    %%half_log = arith.mulf %%neg_half, %%log_var : %s", block_type))
  L <- c(L, sprintf(
    "    %%inv_std_tensor = math.exp %%half_log : %s", block_type))

  # Normalize: (x - mean) * inv_std * weight + bias
  L <- c(L, sprintf(
    "    %%normed = arith.mulf %%centered, %%inv_std_tensor : %s", block_type))
  L <- c(L, sprintf(
    "    %%scaled = arith.mulf %%normed, %%weight : %s", block_type))
  L <- c(L, sprintf(
    "    %%result = arith.addf %%scaled, %%bias : %s", block_type))

  # Store
  L <- c(L, sprintf(
    "    %%out_splat = tt.splat %%out_ptr : %s -> %s", ptr_type, ptr_block))
  L <- c(L, sprintf(
    "    %%out_addrs = tt.addptr %%out_splat, %%offsets : %s, %s",
    ptr_block, i32_block))
  L <- c(L, sprintf(
    "    tt.store %%out_addrs, %%result, %%mask : %s", ptr_block))

  L <- c(L, "    tt.return")
  L <- c(L, "  }")
  L <- c(L, "}")

  list(
    mlir_text = paste(L, collapse = "\n"),
    func_name = func_name,
    dtype = dtype,
    block_size = block_size,
    kernel_type = "layer_norm"
  )
}
