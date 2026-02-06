// triton_compile.cpp — Rcpp bindings for Triton MLIR compilation pipeline
//
// Mirrors the pass ordering from:
//   triton/third_party/nvidia/backend/compiler.py
//
// Pipeline: TTIR text → parse → TTIR passes → TTGIR → LLVM dialect → LLVM IR → PTX

#ifdef HAS_TRITON

#include <Rcpp.h>
#include <string>
#include <memory>

// MLIR core
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Support/LLVM.h"

// MLIR dialect includes
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

// MLIR translation
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

// MLIR conversion passes
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

// Triton dialects
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"

// Triton passes
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

// NVIDIA-specific
#include "Dialect/NVGPU/IR/Dialect.h"
#include "NVGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"

// LLVM backend
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Support/SourceMgr.h"

namespace {

// Collect MLIR diagnostics into a string (errors only)
std::string collectDiagnostics(mlir::MLIRContext &ctx,
                                std::function<mlir::LogicalResult()> fn) {
  std::string errorStr;
  mlir::ScopedDiagnosticHandler handler(
      &ctx, [&](mlir::Diagnostic &diag) {
        if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
          llvm::raw_string_ostream os(errorStr);
          diag.print(os);
          os << "\n";
        }
        // Warnings and notes are silently discarded
        return mlir::success();
      });
  auto result = fn();
  if (mlir::failed(result) && errorStr.empty())
    errorStr = "MLIR pass pipeline failed (no diagnostic)";
  return errorStr;
}

// Initialize LLVM targets (once)
void initLLVMTargets() {
  static bool initialized = false;
  if (!initialized) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    initialized = true;
  }
}

} // namespace

Rcpp::List compile_mlir_to_ptx(std::string mlir_text,
                                int compute_capability,
                                int num_warps,
                                int num_ctas,
                                int ptx_version) {
  // ---- 1. Create MLIRContext and register all dialects ----

  auto ctx = std::make_unique<mlir::MLIRContext>(
      mlir::MLIRContext::Threading::DISABLED);

  mlir::DialectRegistry registry;
  registry.insert<
      mlir::triton::TritonDialect,
      mlir::triton::gpu::TritonGPUDialect,
      mlir::triton::instrument::TritonInstrumentDialect,
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
      mlir::math::MathDialect,
      mlir::arith::ArithDialect,
      mlir::scf::SCFDialect,
      mlir::gpu::GPUDialect,
      mlir::cf::ControlFlowDialect,
      mlir::LLVM::LLVMDialect,
      mlir::ub::UBDialect,
      mlir::triton::gluon::GluonDialect,
      mlir::triton::nvgpu::NVGPUDialect>();

  mlir::LLVM::registerInlinerInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);

  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();

  // ---- 2. Parse MLIR text into ModuleOp ----

  auto moduleRef = mlir::parseSourceString<mlir::ModuleOp>(mlir_text, ctx.get());
  if (!moduleRef) {
    Rcpp::stop("Failed to parse MLIR text. Check that emit_ttir() output is valid.");
  }
  mlir::ModuleOp module = *moduleRef;

  // ---- 3. TTIR normalization passes (make_ttir) ----
  {
    mlir::PassManager pm(ctx.get());
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::triton::createTritonRewriteTensorPointer());
    if (compute_capability / 10 < 9)
      pm.addPass(mlir::triton::createTritonRewriteTensorDescriptorToPointer());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::triton::createTritonCombineOps());
    pm.addPass(mlir::triton::createTritonReorderBroadcast());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::triton::createTritonLoopUnroll());

    std::string diag = collectDiagnostics(*ctx, [&]() {
      return pm.run(module);
    });
    if (!diag.empty())
      Rcpp::stop("TTIR normalization failed: " + diag);
  }

  // ---- 4. TTIR → TTGIR conversion (make_ttgir) ----
  {
    mlir::PassManager pm(ctx.get());

    std::string target = "cuda:" + std::to_string(compute_capability);
    pm.addPass(mlir::triton::createConvertTritonToTritonGPU(
        {target, num_warps, 32, num_ctas}));

    pm.addPass(mlir::triton::gpu::createTritonGPUCoalesce());
    bool emuTF32 = (compute_capability / 10 >= 8);
    pm.addPass(mlir::triton::gpu::createTritonGPUF32DotTC({emuTF32}));
    pm.addPass(mlir::triton::nvidia_gpu::createTritonNvidiaGPUPlanCTAPass());
    pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
    pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());
    pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands(
        {compute_capability >= 80}));
    pm.addPass(
        mlir::triton::nvidia_gpu::createTritonNvidiaGPUOptimizeDescriptorEncodingPass());

    // Capability-specific scheduling/pipelining
    if (compute_capability / 10 == 8 || compute_capability / 10 == 9) {
      pm.addPass(mlir::triton::createTritonLoopInvariantCodeMotion());
    } else {
      pm.addPass(mlir::triton::createTritonLoopInvariantCodeMotion());
    }

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::triton::createTritonLoopAwareCSE());
    pm.addPass(mlir::triton::gpu::createTritonGPUPrefetch());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands(
        {compute_capability >= 80}));
    pm.addPass(mlir::triton::gpu::createTritonGPUCoalesceAsyncCopy());
    pm.addPass(
        mlir::triton::nvidia_gpu::createTritonNvidiaGPUOptimizeTMemLayoutsPass());
    if (compute_capability / 10 >= 9)
      pm.addPass(mlir::triton::nvidia_gpu::createTritonNvidiaGPUTMALoweringPass());
    pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
    pm.addPass(
        mlir::triton::nvidia_gpu::createTritonNvidiaGPUInterleaveTMemPass());
    pm.addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
    pm.addPass(mlir::triton::gpu::createTritonGPUReorderInstructions());
    pm.addPass(mlir::triton::createTritonLoopAwareCSE());
    pm.addPass(mlir::createSymbolDCEPass());

    // Fence insertion
    {
      mlir::triton::nvidia_gpu::TritonGPUFenceInsertionOptions fenceOpts;
      fenceOpts.computeCapability = compute_capability;
      pm.addPass(mlir::triton::nvidia_gpu::createTritonGPUFenceInsertion(fenceOpts));
    }
    pm.addPass(mlir::triton::nvidia_gpu::createTritonNvidiaGPUMMALoweringPass());

    pm.addPass(mlir::createSCCPPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());

    std::string diag = collectDiagnostics(*ctx, [&]() {
      return pm.run(module);
    });
    if (!diag.empty())
      Rcpp::stop("TTGIR conversion failed: " + diag);
  }

  // ---- 5. TTGIR → LLVM dialect (make_llir) ----
  {
    mlir::PassManager pm(ctx.get());

    pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(mlir::triton::gpu::createTritonGPUAllocateWarpGroups());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::triton::gluon::createGluonInline());
    pm.addPass(mlir::triton::createAllocateSharedMemoryNvPass(
        compute_capability, ptx_version));
    pm.addPass(
        mlir::triton::nvidia_gpu::createTritonTensorMemoryAllocationPass());
    pm.addPass(
        mlir::triton::nvidia_gpu::createTritonNvidiaGPUCheckMatmulTwoCTAPass());
    pm.addPass(
        mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());

    // Proxy fence insertion
    {
      mlir::triton::nvidia_gpu::TritonGPUProxyFenceInsertionOptions proxyOpts;
      proxyOpts.computeCapability = compute_capability;
      pm.addPass(
          mlir::triton::nvidia_gpu::createTritonGPUProxyFenceInsertion(proxyOpts));
    }

    pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(
        compute_capability, ptx_version));
    pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::triton::gpu::createCanonicalizeLLVMIR());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::triton::createConvertNVGPUToLLVM());
    pm.addPass(mlir::triton::createConvertWarpSpecializeToLLVM());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createConvertNVVMToLLVMPass());

    std::string diag = collectDiagnostics(*ctx, [&]() {
      return pm.run(module);
    });
    if (!diag.empty())
      Rcpp::stop("LLVM lowering failed: " + diag);
  }

  // ---- 6. Extract metadata from module ----
  int shared_mem = 0;
  if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("ttg.shared"))
    shared_mem = attr.getInt();

  // ---- 7. MLIR LLVM dialect → LLVM IR ----
  initLLVMTargets();

  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmMod =
      mlir::translateModuleToLLVMIR(module, llvmCtx);
  if (!llvmMod)
    Rcpp::stop("Failed to translate MLIR to LLVM IR");

  // Set nvptx-short-ptr option
  {
    auto options = llvm::cl::getRegisteredOptions();
    auto it = options.find("nvptx-short-ptr");
    if (it != options.end()) {
      auto *opt = static_cast<llvm::cl::opt<bool> *>(it->second);
      opt->setValue(true);
    }
  }

  // Set target triple and data layout
  std::string proc = "sm_" + std::to_string(compute_capability);
  std::string triple = "nvptx64-nvidia-cuda";

  // Compute features string from PTX version
  int llvm_ptx = std::min(86, ptx_version);
  std::string features = "+ptx" + std::to_string(llvm_ptx);

  {
    std::string error;
    llvm::Triple targetTriple(triple);
    auto *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target)
      Rcpp::stop("LLVM target lookup failed: " + error);

    llvm::TargetOptions topt;
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
        targetTriple, proc, features, topt, llvm::Reloc::PIC_,
        std::nullopt, llvm::CodeGenOptLevel::None));
    llvmMod->setDataLayout(machine->createDataLayout());
  }

  llvmMod->setTargetTriple(llvm::Triple(triple));

  // ---- 8. Optimize LLVM IR ----
  {
    // Inline everything
    for (llvm::Function &f : llvmMod->functions())
      if (!f.hasFnAttribute(llvm::Attribute::NoInline))
        f.addFnAttr(llvm::Attribute::AlwaysInline);

    llvm::legacy::PassManager pm;
    pm.add(llvm::createAlwaysInlinerLegacyPass());
    pm.add(llvm::createVerifierPass());
    pm.run(*llvmMod);
  }

  // New pass manager optimization (O3)
  {
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    std::unique_ptr<llvm::TargetMachine> machine;
    {
      std::string error;
      auto *target = llvm::TargetRegistry::lookupTarget(
          llvmMod->getTargetTriple(), error);
      llvm::TargetOptions topt;
      topt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
      topt.NoNaNsFPMath = true;
      topt.TrapUnreachable = true;
      machine.reset(target->createTargetMachine(
          llvmMod->getTargetTriple(), proc, features, topt,
          llvm::Reloc::PIC_, std::nullopt,
          llvm::CodeGenOptLevel::Aggressive));
    }

    llvm::PipelineTuningOptions tuningOpts;
    tuningOpts.LoopUnrolling = true;
    tuningOpts.LoopInterleaving = true;
    tuningOpts.LoopVectorization = true;
    tuningOpts.SLPVectorization = true;

    llvm::PassBuilder pb(machine.get(), tuningOpts);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm =
        pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    mpm.run(*llvmMod, mam);
  }

  // ---- 9. LLVM IR → PTX assembly ----
  std::string ptxStr;
  {
    std::string error;
    auto *target = llvm::TargetRegistry::lookupTarget(
        llvmMod->getTargetTriple(), error);
    if (!target)
      Rcpp::stop("PTX target lookup failed: " + error);

    llvm::TargetOptions topt;
    topt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    topt.NoNaNsFPMath = true;
    topt.TrapUnreachable = true;
    topt.MCOptions.AsmVerbose = true;
    topt.MCOptions.PreserveAsmComments = true;

    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
        llvmMod->getTargetTriple(), proc, features, topt,
        llvm::Reloc::PIC_, std::nullopt,
        llvm::CodeGenOptLevel::Aggressive));

    llvmMod->setDataLayout(machine->createDataLayout());

    llvm::raw_string_ostream stream(ptxStr);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager pass;
    machine->addPassesToEmitFile(pass, pstream, nullptr,
                                  llvm::CodeGenFileType::AssemblyFile);
    pass.run(*llvmMod);
  }

  // ---- 10. Extract kernel name from PTX ----
  std::string kernelName;
  {
    // Look for ".visible .entry KERNEL_NAME"
    std::string marker = ".visible .entry ";
    auto pos = ptxStr.find(marker);
    if (pos != std::string::npos) {
      auto nameStart = pos + marker.size();
      auto nameEnd = ptxStr.find_first_of("( \t\n", nameStart);
      if (nameEnd != std::string::npos)
        kernelName = ptxStr.substr(nameStart, nameEnd - nameStart);
    }
  }

  // Post-process PTX: patch version and target
  {
    std::string versionStr = std::to_string(ptx_version / 10) + "." +
                              std::to_string(ptx_version % 10);
    // Replace .version X.Y
    std::string versionPrefix = ".version ";
    auto vpos = ptxStr.find(versionPrefix);
    if (vpos != std::string::npos) {
      auto vend = ptxStr.find('\n', vpos);
      ptxStr.replace(vpos, vend - vpos,
                      versionPrefix + versionStr);
    }
    // Replace .target sm_XX
    std::string targetPrefix = ".target sm_";
    auto tpos = ptxStr.find(targetPrefix);
    if (tpos != std::string::npos) {
      auto tend = ptxStr.find_first_of(" ,\n", tpos + targetPrefix.size());
      ptxStr.replace(tpos, tend - tpos,
                      targetPrefix + std::to_string(compute_capability));
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("ptx") = ptxStr,
      Rcpp::Named("kernel_name") = kernelName,
      Rcpp::Named("shared_mem") = shared_mem
  );
}

#endif // HAS_TRITON
