// cuda_launch.cpp - CUDA driver API bindings for GPU kernel execution
//
// Phase 5c: Takes compiled PTX from Phase 5b, extracts device pointers from
// R torch tensors, loads PTX module, and launches kernel on GPU.

#include <Rcpp.h>
#include <R.h>
#include <Rinternals.h>
#include <cuda.h>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

// Helper: check CUDA driver API error
#define CUDA_CHECK(call) do { \
  CUresult err = call; \
  if (err != CUDA_SUCCESS) { \
    const char *name, *msg; \
    cuGetErrorName(err, &name); \
    cuGetErrorString(err, &msg); \
    throw std::runtime_error(std::string("CUDA error: ") + name + ": " + msg); \
  } \
} while(0)

// Global CUDA state (lazy init)
static bool cuda_initialized = false;
static CUcontext cuda_context = nullptr;

static void ensure_cuda_initialized() {
  if (cuda_initialized) return;

  CUDA_CHECK(cuInit(0));

  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device, 0));

  CUDA_CHECK(cuCtxCreate(&cuda_context, 0, device));
  cuda_initialized = true;
}

// Extract CUDA device pointer from R torch tensor
// R torch_tensor structure: EXTPTRSXP -> shared_ptr<void>* -> torch::Tensor*
static CUdeviceptr get_tensor_device_ptr(SEXP tensor_sexp) {
  // Verify it's a torch_tensor
  if (TYPEOF(tensor_sexp) != EXTPTRSXP) {
    throw std::runtime_error("Expected torch_tensor (external pointer)");
  }

  // Get raw external pointer without Rcpp's XPtr template
  void* xptr_addr = R_ExternalPtrAddr(tensor_sexp);
  if (!xptr_addr) {
    throw std::runtime_error("NULL torch tensor pointer");
  }

  // Navigate: shared_ptr<void>* -> torch::Tensor*
  // The external pointer holds a shared_ptr<void>* that points to a Tensor
  auto shared_ptr_ptr = static_cast<std::shared_ptr<void>*>(xptr_addr);
  auto tensor_ptr = static_cast<torch::Tensor*>(shared_ptr_ptr->get());

  if (!tensor_ptr) {
    throw std::runtime_error("NULL tensor in shared_ptr");
  }

  // Check tensor is on CUDA
  if (!tensor_ptr->is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA device (.cuda())");
  }

  // Get device pointer
  return reinterpret_cast<CUdeviceptr>(tensor_ptr->data_ptr());
}

//' Launch a GPU Kernel from Compiled PTX
//'
//' @param ptx Character, PTX assembly from mlir_compile()
//' @param kernel_name Character, entry point function name
//' @param inputs List of torch_tensor inputs (must be on CUDA)
//' @param output torch_tensor pre-allocated output (must be on CUDA)
//' @param grid Integer vector of length 3, grid dimensions (blocks)
//' @param block Integer vector of length 3, block dimensions (threads)
//' @param shared_mem Integer, dynamic shared memory bytes
//' @return The output tensor (same as input)
//' @export
// [[Rcpp::export]]
SEXP gpu_launch(std::string ptx, std::string kernel_name,
                Rcpp::List inputs, SEXP output,
                Rcpp::IntegerVector grid, Rcpp::IntegerVector block,
                int shared_mem = 0) {
  try {
    ensure_cuda_initialized();

    // Load PTX module
    CUmodule module;
    CUDA_CHECK(cuModuleLoadData(&module, ptx.c_str()));

    // Get kernel function
    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));

    // Extract device pointers from input tensors
    std::vector<CUdeviceptr> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
      input_ptrs.push_back(get_tensor_device_ptr(inputs[i]));
    }

    // Get output device pointer
    CUdeviceptr output_ptr = get_tensor_device_ptr(output);

    // Get output tensor to compute n_elements
    void* output_raw = R_ExternalPtrAddr(output);
    auto output_shared_ptr = static_cast<std::shared_ptr<void>*>(output_raw);
    auto output_tensor_ptr = static_cast<torch::Tensor*>(output_shared_ptr->get());
    int n_elements = output_tensor_ptr->numel();

    // Build kernel args array: [input_ptrs..., output_ptr, n_elements, metadata...]
    // Triton-compiled PTX has extra metadata parameters beyond the TTIR signature
    std::vector<void*> args;
    args.reserve(inputs.size() + 4);  // inputs + output + n_elements + 2 metadata ptrs
    for (size_t i = 0; i < input_ptrs.size(); i++) {
      args.push_back(&input_ptrs[i]);
    }
    args.push_back(&output_ptr);
    args.push_back(&n_elements);

    // Triton adds extra pointer parameters - pass NULL for now
    CUdeviceptr null_ptr = 0;
    args.push_back(&null_ptr);
    args.push_back(&null_ptr);

    // Launch kernel
    CUDA_CHECK(cuLaunchKernel(
      kernel,
      grid[0], grid[1], grid[2],       // grid dimensions
      block[0], block[1], block[2],    // block dimensions
      shared_mem,                       // shared memory bytes
      nullptr,                          // stream (NULL = default stream)
      args.data(),                      // kernel arguments
      nullptr                           // extra (unused)
    ));

    // Synchronize to ensure kernel completes
    CUDA_CHECK(cuCtxSynchronize());

    // Unload module
    CUDA_CHECK(cuModuleUnload(module));

    return output;

  } catch (std::exception &e) {
    Rcpp::stop(e.what());
  }
}
