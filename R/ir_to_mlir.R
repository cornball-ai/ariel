#' Translate torchlang IR to Triton MLIR (TTIR)
#'
#' Lowers an optimized ir_graph (from torchlang) to Triton's MLIR
#' textual format. This is the first stage of the compilation pipeline.
#' The output can be fed to Triton's MLIR passes for GPU compilation.

# ============================================================================
# Op Mapping: torchlang IR ops → MLIR operations
# ============================================================================

# Unary ops: torchlang op name → MLIR op string
# These operate on tensor<BLOCKxf32> types in Triton MLIR
.ttir_unary_ops <- list(
  # arith dialect
  neg   = "arith.negf",

  # math dialect
  exp   = "math.exp",
  log   = "math.log",
  log2  = "math.log2",
  tanh  = "math.tanh",
  sqrt  = "math.sqrt",
  abs   = "math.absf",
  floor = "math.floor",
  ceil  = "math.ceil",
  sin   = "math.sin",
  cos   = "math.cos"
)

# Binary ops: torchlang op name → MLIR op string
.ttir_binary_ops <- list(
  add = "arith.addf",
  sub = "arith.subf",
  mul = "arith.mulf",
  div = "arith.divf"
)

# Compound ops that need custom MLIR emission (not 1:1 with MLIR ops)
.ttir_compound_ops <- c("relu", "sigmoid", "silu", "gelu", "sign")


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

# MLIR type string for a Triton block tensor
.block_type <- function(dtype = "f32", block_var = "%BLOCK_SIZE") {
  sprintf("tensor<%sx%s>", block_var, dtype)
}

# Emit a simple unary MLIR op
.emit_unary <- function(mlir_op, input_ssa, output_ssa, type) {
  sprintf("    %s = %s %s : %s", output_ssa, mlir_op, input_ssa, type)
}

# Emit a simple binary MLIR op
.emit_binary <- function(mlir_op, lhs_ssa, rhs_ssa, output_ssa, type) {
  sprintf("    %s = %s %s, %s : %s", output_ssa, mlir_op, lhs_ssa, rhs_ssa, type)
}

# Emit compound ops that expand to multiple MLIR operations
# Returns a character vector of MLIR lines and the final SSA name
.emit_compound <- function(op, input_ssas, base_id, type) {
  x <- input_ssas[1]
  lines <- character()
  final_ssa <- .ssa(base_id)

  if (op == "relu") {
    # relu(x) = max(x, 0) = select(x > 0, x, 0)
    zero_ssa <- sprintf("%%zero_%d", base_id)
    cmp_ssa <- sprintf("%%cmp_%d", base_id)
    cmp_type <- sub("f32", "i1", sub("f16", "i1", type))
    lines <- c(
      sprintf("    %s = arith.constant dense<0.0> : %s", zero_ssa, type),
      sprintf("    %s = arith.cmpf ogt, %s, %s : %s", cmp_ssa, x, zero_ssa, type),
      sprintf("    %s = arith.select %s, %s, %s : %s, %s",
              final_ssa, cmp_ssa, x, zero_ssa, cmp_type, type)
    )
  } else if (op == "sigmoid") {
    # sigmoid(x) = 1 / (1 + exp(-x))
    neg_ssa <- sprintf("%%neg_%d", base_id)
    exp_ssa <- sprintf("%%exp_%d", base_id)
    one_ssa <- sprintf("%%one_%d", base_id)
    sum_ssa <- sprintf("%%sum_%d", base_id)
    lines <- c(
      sprintf("    %s = arith.negf %s : %s", neg_ssa, x, type),
      sprintf("    %s = math.exp %s : %s", exp_ssa, neg_ssa, type),
      sprintf("    %s = arith.constant dense<1.0> : %s", one_ssa, type),
      sprintf("    %s = arith.addf %s, %s : %s", sum_ssa, one_ssa, exp_ssa, type),
      sprintf("    %s = arith.divf %s, %s : %s", final_ssa, one_ssa, sum_ssa, type)
    )
  } else if (op == "silu") {
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    neg_ssa <- sprintf("%%neg_%d", base_id)
    exp_ssa <- sprintf("%%exp_%d", base_id)
    one_ssa <- sprintf("%%one_%d", base_id)
    sum_ssa <- sprintf("%%sum_%d", base_id)
    lines <- c(
      sprintf("    %s = arith.negf %s : %s", neg_ssa, x, type),
      sprintf("    %s = math.exp %s : %s", exp_ssa, neg_ssa, type),
      sprintf("    %s = arith.constant dense<1.0> : %s", one_ssa, type),
      sprintf("    %s = arith.addf %s, %s : %s", sum_ssa, one_ssa, exp_ssa, type),
      sprintf("    %s = arith.divf %s, %s : %s", final_ssa, x, sum_ssa, type)
    )
  } else if (op == "gelu") {
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c1_ssa <- sprintf("%%c_half_%d", base_id)
    c2_ssa <- sprintf("%%c_coeff_%d", base_id)
    c3_ssa <- sprintf("%%c_sqrt2pi_%d", base_id)
    one_ssa <- sprintf("%%one_%d", base_id)
    x2_ssa <- sprintf("%%x2_%d", base_id)
    x3_ssa <- sprintf("%%x3_%d", base_id)
    cx3_ssa <- sprintf("%%cx3_%d", base_id)
    inner_ssa <- sprintf("%%inner_%d", base_id)
    scaled_ssa <- sprintf("%%scaled_%d", base_id)
    tanh_ssa <- sprintf("%%tanh_%d", base_id)
    plus1_ssa <- sprintf("%%plus1_%d", base_id)
    half_x_ssa <- sprintf("%%half_x_%d", base_id)
    lines <- c(
      sprintf("    %s = arith.constant dense<0.5> : %s", c1_ssa, type),
      sprintf("    %s = arith.constant dense<4.471500e-02> : %s", c2_ssa, type),
      sprintf("    %s = arith.constant dense<7.978845e-01> : %s", c3_ssa, type),
      sprintf("    %s = arith.constant dense<1.0> : %s", one_ssa, type),
      sprintf("    %s = arith.mulf %s, %s : %s", x2_ssa, x, x, type),
      sprintf("    %s = arith.mulf %s, %s : %s", x3_ssa, x2_ssa, x, type),
      sprintf("    %s = arith.mulf %s, %s : %s", cx3_ssa, c2_ssa, x3_ssa, type),
      sprintf("    %s = arith.addf %s, %s : %s", inner_ssa, x, cx3_ssa, type),
      sprintf("    %s = arith.mulf %s, %s : %s", scaled_ssa, c3_ssa, inner_ssa, type),
      sprintf("    %s = math.tanh %s : %s", tanh_ssa, scaled_ssa, type),
      sprintf("    %s = arith.addf %s, %s : %s", plus1_ssa, one_ssa, tanh_ssa, type),
      sprintf("    %s = arith.mulf %s, %s : %s", half_x_ssa, c1_ssa, x, type),
      sprintf("    %s = arith.mulf %s, %s : %s", final_ssa, half_x_ssa, plus1_ssa, type)
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
      sprintf("    %s = arith.constant dense<0.0> : %s", zero_ssa, type),
      sprintf("    %s = arith.constant dense<1.0> : %s", pos1_ssa, type),
      sprintf("    %s = arith.constant dense<-1.0> : %s", neg1_ssa, type),
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
# Main Emission Functions
# ============================================================================

#' Emit Triton MLIR for an Elementwise Fusion Group
#'
#' Translates a torchlang IR fusion group to Triton's MLIR textual format.
#' The output is a complete MLIR module containing a `tt.func` with
#' pointer-based load/compute/store pattern.
#'
#' @param graph An ir_graph (from torchlang) with fusion annotations
#' @param group_id Integer fusion group ID
#' @param func_name Optional kernel function name
#' @param dtype MLIR type string ("f32" or "f16")
#' @return List with mlir_text (character string), func_name, n_inputs,
#'   external_input_ids, output_id, group_node_ids.
#'   NULL if group contains unsupported ops.
#' @export
emit_ttir <- function(graph, group_id, func_name = NULL, dtype = "f32") {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

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
  block_type <- .block_type(dtype)

  # --- Build the tt.func ---

  # Function args: n_inputs pointer args + output pointer + n_elements
  # In Triton MLIR, pointers are !tt.ptr<f32> and scalars are i32
  ptr_type <- sprintf("!tt.ptr<%s>", dtype)
  arg_names <- character()
  arg_types <- character()
  for (i in seq_len(n_inputs)) {
    arg_names <- c(arg_names, sprintf("%%arg%d", i - 1L))
    arg_types <- c(arg_types, ptr_type)
  }
  arg_names <- c(arg_names, "%out_ptr", "%n_elements")
  arg_types <- c(arg_types, ptr_type, "i32")

  sig_args <- paste(sprintf("%s: %s", arg_names, arg_types), collapse = ", ")

  lines <- character()

  # Module header
  lines <- c(lines, "#loc = loc(unknown)")
  lines <- c(lines, "module {")
  lines <- c(lines, sprintf("  tt.func public @%s(%s) {", func_name, sig_args))

  # Program ID and offset computation
  lines <- c(lines, "    %pid = tt.get_program_id x : i32")
  lines <- c(lines, "    %BLOCK_SIZE = arith.constant 1024 : i32")
  lines <- c(lines, "    %block_start = arith.muli %pid, %BLOCK_SIZE : i32")
  lines <- c(lines,
    sprintf("    %%range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>"))
  lines <- c(lines,
    "    %block_start_splat = tt.splat %block_start : i32 -> tensor<1024xi32>")
  lines <- c(lines,
    "    %offsets = arith.addi %block_start_splat, %range : tensor<1024xi32>")

  # Mask: offsets < n_elements
  lines <- c(lines,
    "    %n_splat = tt.splat %n_elements : i32 -> tensor<1024xi32>")
  lines <- c(lines,
    "    %mask = arith.cmpi slt, %offsets, %n_splat : tensor<1024xi32>")

  # Load each external input
  ext_ssa <- list()  # external_input_id -> SSA name after load
  for (i in seq_len(n_inputs)) {
    ptr_arg <- sprintf("%%arg%d", i - 1L)
    splat_ssa <- sprintf("%%ptr_splat_%d", i)
    addr_ssa <- sprintf("%%addr_%d", i)
    load_ssa <- sprintf("%%load_%d", i)
    lines <- c(lines,
      sprintf("    %s = tt.splat %s : %s -> tensor<1024x%s>",
              splat_ssa, ptr_arg, ptr_type, ptr_type))
    lines <- c(lines,
      sprintf("    %s = tt.addptr %s, %s : tensor<1024x%s>, tensor<1024xi32>",
              addr_ssa, splat_ssa, "%offsets", ptr_type))
    lines <- c(lines,
      sprintf("    %s = tt.load %s, %s : tensor<1024x%s>",
              load_ssa, addr_ssa, "%mask", ptr_type))
    ext_ssa[[as.character(external_input_ids[i])]] <- load_ssa
  }

  # Emit compute ops for each node in the fusion group
  node_ssa <- ext_ssa  # maps node id (as character) to SSA name
  next_tmp <- 100L

  for (nid in group_node_ids) {
    node <- graph$nodes[[as.character(nid)]]
    op <- node$op
    input_ssas <- vapply(node$inputs, function(inp_id) {
      node_ssa[[as.character(inp_id)]]
    }, character(1))

    out_ssa <- .ssa(nid)

    if (op %in% names(.ttir_unary_ops)) {
      mlir_op <- .ttir_unary_ops[[op]]
      lines <- c(lines, .emit_unary(mlir_op, input_ssas[1], out_ssa, block_type))
      node_ssa[[as.character(nid)]] <- out_ssa

    } else if (op %in% names(.ttir_binary_ops)) {
      mlir_op <- .ttir_binary_ops[[op]]
      lines <- c(lines, .emit_binary(mlir_op, input_ssas[1], input_ssas[2],
                                     out_ssa, block_type))
      node_ssa[[as.character(nid)]] <- out_ssa

    } else if (op %in% .ttir_compound_ops) {
      compound <- .emit_compound(op, input_ssas, next_tmp, block_type)
      lines <- c(lines, compound$lines)
      node_ssa[[as.character(nid)]] <- compound$ssa
      next_tmp <- next_tmp + 50L  # leave room for intermediate SSAs

    } else {
      return(NULL)
    }
  }

  # Store result
  result_ssa <- node_ssa[[as.character(output_id)]]
  lines <- c(lines,
    sprintf("    %%out_splat = tt.splat %%out_ptr : %s -> tensor<1024x%s>",
            ptr_type, ptr_type))
  lines <- c(lines,
    sprintf("    %%out_addr = tt.addptr %%out_splat, %%offsets : tensor<1024x%s>, tensor<1024xi32>",
            ptr_type))
  lines <- c(lines,
    sprintf("    tt.store %%out_addr, %s, %%mask : tensor<1024x%s>",
            result_ssa, ptr_type))

  # Close function and module
  lines <- c(lines, "    tt.return")
  lines <- c(lines, "  }")
  lines <- c(lines, "}")

  mlir_text <- paste(lines, collapse = "\n")

  list(
    mlir_text = mlir_text,
    func_name = func_name,
    n_inputs = n_inputs,
    external_input_ids = external_input_ids,
    output_id = output_id,
    group_node_ids = group_node_ids,
    dtype = dtype
  )
}


#' Emit Triton MLIR for Tiled Matmul with Epilogue Fusion
#'
#' Generates MLIR for a tiled matrix multiplication using tt.dot,
#' with optional fused epilogue operations applied before the store.
#'
#' @param epilogue_ops Character vector of elementwise ops to fuse
#' @param func_name Kernel function name
#' @param dtype MLIR element type ("f32" or "f16")
#' @param block_m Tile size M
#' @param block_n Tile size N
#' @param block_k Tile size K
#' @return List with mlir_text, func_name, epilogue_ops.
#'   NULL if epilogue contains unsupported ops.
#' @export
emit_ttir_matmul <- function(epilogue_ops = character(),
                              func_name = "matmul_kernel",
                              dtype = "f32",
                              block_m = 64L, block_n = 64L, block_k = 32L) {
  for (op in epilogue_ops) {
    if (!triton_op_supported(op)) return(NULL)
  }

  ptr_type <- sprintf("!tt.ptr<%s>", dtype)
  acc_type <- sprintf("tensor<%dx%dxf32>", block_m, block_n)
  a_tile_type <- sprintf("tensor<%dx%dx%s>", block_m, block_k, dtype)
  b_tile_type <- sprintf("tensor<%dx%dx%s>", block_k, block_n, dtype)
  offs_m_type <- sprintf("tensor<%dxi32>", block_m)
  offs_n_type <- sprintf("tensor<%dxi32>", block_n)
  offs_k_type <- sprintf("tensor<%dxi32>", block_k)

  lines <- character()
  lines <- c(lines, "#loc = loc(unknown)")
  lines <- c(lines, "module {")
  lines <- c(lines, sprintf(paste0(
    "  tt.func public @%s(",
    "%%a_ptr: %s, %%b_ptr: %s, %%c_ptr: %s, ",
    "%%M: i32, %%N: i32, %%K: i32, ",
    "%%stride_am: i32, %%stride_ak: i32, ",
    "%%stride_bk: i32, %%stride_bn: i32, ",
    "%%stride_cm: i32, %%stride_cn: i32) {"),
    func_name, ptr_type, ptr_type, ptr_type))

  # Program IDs
  lines <- c(lines, "    %pid_m = tt.get_program_id x : i32")
  lines <- c(lines, "    %pid_n = tt.get_program_id y : i32")

  # Block size constants
  lines <- c(lines, sprintf("    %%BLOCK_M = arith.constant %d : i32", block_m))
  lines <- c(lines, sprintf("    %%BLOCK_N = arith.constant %d : i32", block_n))
  lines <- c(lines, sprintf("    %%BLOCK_K = arith.constant %d : i32", block_k))

  # Offset ranges
  lines <- c(lines, sprintf(
    "    %%range_m = tt.make_range {start = 0 : i32, end = %d : i32} : %s",
    block_m, offs_m_type))
  lines <- c(lines, sprintf(
    "    %%range_n = tt.make_range {start = 0 : i32, end = %d : i32} : %s",
    block_n, offs_n_type))
  lines <- c(lines, sprintf(
    "    %%range_k = tt.make_range {start = 0 : i32, end = %d : i32} : %s",
    block_k, offs_k_type))

  # offs_m = pid_m * BLOCK_M + range_m
  lines <- c(lines, sprintf(
    "    %%base_m = arith.muli %%pid_m, %%BLOCK_M : i32"))
  lines <- c(lines, sprintf(
    "    %%base_m_splat = tt.splat %%base_m : i32 -> %s", offs_m_type))
  lines <- c(lines, sprintf(
    "    %%offs_m = arith.addi %%base_m_splat, %%range_m : %s", offs_m_type))

  # offs_n = pid_n * BLOCK_N + range_n
  lines <- c(lines,
    "    %base_n = arith.muli %pid_n, %BLOCK_N : i32")
  lines <- c(lines, sprintf(
    "    %%base_n_splat = tt.splat %%base_n : i32 -> %s", offs_n_type))
  lines <- c(lines, sprintf(
    "    %%offs_n = arith.addi %%base_n_splat, %%range_n : %s", offs_n_type))

  # Zero accumulator
  lines <- c(lines, sprintf(
    "    %%acc_init = arith.constant dense<0.0> : %s", acc_type))

  # K-loop via scf.for
  lines <- c(lines, "    %c0 = arith.constant 0 : i32")
  lines <- c(lines, sprintf(
    "    %%acc_final = scf.for %%k = %%c0 to %%K step %%BLOCK_K iter_args(%%acc = %%acc_init) -> (%s) {",
    acc_type))

  # Load A tile and B tile (simplified — full address computation)
  lines <- c(lines, "      // A tile load: a_ptr + offs_m * stride_am + (k + range_k) * stride_ak")
  lines <- c(lines, sprintf(
    "      // B tile load: b_ptr + (k + range_k) * stride_bk + offs_n * stride_bn"))
  lines <- c(lines, sprintf(
    "      // [Tile address computation elided for clarity]"))
  lines <- c(lines, sprintf(
    "      %%a_tile = tt.load %%a_addr, %%mask_a : tensor<%dx%dx%s>",
    block_m, block_k, ptr_type))
  lines <- c(lines, sprintf(
    "      %%b_tile = tt.load %%b_addr, %%mask_b : tensor<%dx%dx%s>",
    block_k, block_n, ptr_type))

  # Dot product
  lines <- c(lines, sprintf(
    "      %%dot = tt.dot %%a_tile, %%b_tile, %%acc : %s * %s -> %s",
    a_tile_type, b_tile_type, acc_type))
  lines <- c(lines, sprintf("      scf.yield %%dot : %s", acc_type))
  lines <- c(lines, "    }")

  # Epilogue: apply fused ops to acc_final
  current_ssa <- "%acc_final"
  epi_idx <- 200L
  for (op in epilogue_ops) {
    out_ssa <- sprintf("%%epi_%d", epi_idx)
    if (op %in% names(.ttir_unary_ops)) {
      mlir_op <- .ttir_unary_ops[[op]]
      lines <- c(lines, sprintf("    %s = %s %s : %s",
                                out_ssa, mlir_op, current_ssa, acc_type))
    } else if (op %in% .ttir_compound_ops) {
      compound <- .emit_compound(op, current_ssa, epi_idx, acc_type)
      lines <- c(lines, compound$lines)
      out_ssa <- compound$ssa
    } else {
      return(NULL)
    }
    current_ssa <- out_ssa
    epi_idx <- epi_idx + 50L
  }

  # Store result
  lines <- c(lines, "    // Store C tile: c_ptr + offs_m * stride_cm + offs_n * stride_cn")
  lines <- c(lines, sprintf("    tt.store %%c_addr, %s, %%mask_c : tensor<%dx%dx%s>",
                             current_ssa, block_m, block_n, ptr_type))

  lines <- c(lines, "    tt.return")
  lines <- c(lines, "  }")
  lines <- c(lines, "}")

  mlir_text <- paste(lines, collapse = "\n")

  list(
    mlir_text = mlir_text,
    func_name = func_name,
    epilogue_ops = epilogue_ops,
    dtype = dtype,
    block_m = block_m,
    block_n = block_n,
    block_k = block_k
  )
}
