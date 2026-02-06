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
#include <unordered_map>

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

// ---- PTX Module Cache ----
// Caches CUmodule + CUfunction per kernel_name to avoid repeated
// cuModuleLoadData() on every launch (~0.25ms overhead per call).

struct CachedKernel {
  CUmodule module;
  CUfunction function;
};

static std::unordered_map<std::string, CachedKernel> kernel_cache;

static CUfunction get_cached_kernel(const std::string& ptx,
                                    const std::string& kernel_name) {
  auto it = kernel_cache.find(kernel_name);
  if (it != kernel_cache.end()) {
    return it->second.function;
  }

  CUmodule module;
  CUDA_CHECK(cuModuleLoadData(&module, ptx.c_str()));

  CUfunction function;
  CUDA_CHECK(cuModuleGetFunction(&function, module, kernel_name.c_str()));

  kernel_cache[kernel_name] = {module, function};
  return function;
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

    CUfunction kernel = get_cached_kernel(ptx, kernel_name);

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

    return output;

  } catch (std::exception &e) {
    Rcpp::stop(e.what());
  }
}


//' Launch a Matmul GPU Kernel from Compiled PTX
//'
//' Matmul kernels have a different parameter layout than elementwise kernels:
//' 3 pointer args (A, B, C), 9 i32 scalars (M, N, K, strides), and 2 null
//' metadata pointers added by Triton. This function handles that layout.
//'
//' @param ptx Character, PTX assembly from mlir_compile()
//' @param kernel_name Character, entry point function name
//' @param A torch_tensor, matrix A (must be on CUDA)
//' @param B torch_tensor, matrix B (must be on CUDA)
//' @param C torch_tensor, pre-allocated output matrix (must be on CUDA)
//' @param M Integer, rows of A / rows of C
//' @param N Integer, cols of B / cols of C
//' @param K Integer, cols of A / rows of B
//' @param stride_am Integer, stride of A along M dimension
//' @param stride_ak Integer, stride of A along K dimension
//' @param stride_bk Integer, stride of B along K dimension
//' @param stride_bn Integer, stride of B along N dimension
//' @param stride_cm Integer, stride of C along M dimension
//' @param stride_cn Integer, stride of C along N dimension
//' @param grid Integer vector of length 3, grid dimensions
//' @param block Integer vector of length 3, block dimensions
//' @param shared_mem Integer, dynamic shared memory bytes
//' @return The output tensor C
//' @export
// [[Rcpp::export]]
SEXP gpu_launch_matmul(std::string ptx, std::string kernel_name,
                       SEXP A, SEXP B, SEXP C,
                       int M, int N, int K,
                       int stride_am, int stride_ak,
                       int stride_bk, int stride_bn,
                       int stride_cm, int stride_cn,
                       Rcpp::IntegerVector grid, Rcpp::IntegerVector block,
                       int shared_mem = 0) {
  try {
    ensure_cuda_initialized();

    CUfunction kernel = get_cached_kernel(ptx, kernel_name);

    // Extract device pointers from tensors
    CUdeviceptr a_ptr = get_tensor_device_ptr(A);
    CUdeviceptr b_ptr = get_tensor_device_ptr(B);
    CUdeviceptr c_ptr = get_tensor_device_ptr(C);

    // Triton adds 2 extra metadata pointer parameters beyond the TTIR signature
    CUdeviceptr null_ptr = 0;

    // Build kernel args: 3 ptrs + 9 i32 scalars + 2 null metadata ptrs = 14 params
    void* args[] = {
      &a_ptr, &b_ptr, &c_ptr,
      &M, &N, &K,
      &stride_am, &stride_ak,
      &stride_bk, &stride_bn,
      &stride_cm, &stride_cn,
      &null_ptr, &null_ptr
    };

    // Launch kernel
    CUDA_CHECK(cuLaunchKernel(
      kernel,
      grid[0], grid[1], grid[2],
      block[0], block[1], block[2],
      shared_mem,
      nullptr,
      args,
      nullptr
    ));

    // Synchronize
    CUDA_CHECK(cuCtxSynchronize());

    return C;

  } catch (std::exception &e) {
    Rcpp::stop(e.what());
  }
}


//' Clear the GPU Kernel Cache
//'
//' Unloads all cached PTX modules and clears the kernel cache.
//' Call this when you want to free GPU resources or reload modified kernels.
//'
//' @return List with n_cleared (number of cached kernels removed)
//' @export
// [[Rcpp::export]]
Rcpp::List gpu_cache_clear() {
  int n = kernel_cache.size();
  for (auto& kv : kernel_cache) {
    cuModuleUnload(kv.second.module);
  }
  kernel_cache.clear();
  return Rcpp::List::create(Rcpp::Named("n_cleared") = n);
}


//' Get GPU Kernel Cache Statistics
//'
//' @return List with n_cached (number of kernels in cache) and
//'   kernel_names (character vector of cached kernel names)
//' @export
// [[Rcpp::export]]
Rcpp::List gpu_cache_stats() {
  std::vector<std::string> names;
  names.reserve(kernel_cache.size());
  for (auto& kv : kernel_cache) {
    names.push_back(kv.first);
  }
  return Rcpp::List::create(
    Rcpp::Named("n_cached") = (int)kernel_cache.size(),
    Rcpp::Named("kernel_names") = names
  );
}
