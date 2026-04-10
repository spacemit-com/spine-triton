//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton-shared/Conversion/XSMTToLinalg/XSMTToLinalg.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define DEBUG_TYPE "xsmt-to-linalg"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_XSMTTOLINALG
#include "triton-shared/Conversion/XSMTToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class XSMTToLinalgPass
    : public triton::impl::XSMTToLinalgBase<XSMTToLinalgPass> {
  using XSMTToLinalgBase<XSMTToLinalgPass>::XSMTToLinalgBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                triton::TritonDialect, ttx::TritonTilingExtDialect,
                tts::TritonStructuredDialect, mlir::LLVM::LLVMDialect,
                mlir::xsmt::XSMTDialect, mlir::xsmt_async::XSMTAsyncDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    bool hasXSMTUser = false;
    moduleOp.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("xsmt.") ||
          op->getName().getStringRef().starts_with("xsmt_async.")) {
        hasXSMTUser = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasXSMTUser) {
      RewritePatternSet patterns(&getContext());
      triton::LoopParallelizationConversionPatterns(patterns);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
      }
      return;
    }

    RewritePatternSet patterns0(&getContext());
    RewritePatternSet patterns1(&getContext());
    RewritePatternSet patterns2(&getContext());
    RewritePatternSet patterns3(&getContext());
    RewritePatternSet patterns4(&getContext());
    RewritePatternSet patterns5(&getContext());
    RewritePatternSet patterns6(&getContext());

    triton::populateXSMTOptimizationAndValidationPatterns(patterns0);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns0)))) {
      signalPassFailure();
    }

    triton::populateXSMTToLinalgConversionPatterns(patterns1);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns1)))) {
      signalPassFailure();
    }

    triton::ConvertMMT4DAddConversionPatterns(patterns2);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns2)))) {
      signalPassFailure();
    }

    triton::MMT4DOpConversionPatterns(patterns3);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns3)))) {
      signalPassFailure();
    }

    triton::LoopParallelizationConversionPatterns(patterns4);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns4)))) {
      signalPassFailure();
    }

    triton::BufferizationCleanupConversionPatterns(patterns5);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns5)))) {
      signalPassFailure();
    }

    triton::populateCanonicalizationPatterns(patterns6);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns6)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createXSMTToLinalgPass() {
  return std::make_unique<XSMTToLinalgPass>();
}
