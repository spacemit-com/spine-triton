//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcileLlvmPtrCasts.h"

#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_RECONCILELLVMPTRCASTS
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct PromoteMemrefToPtrArg : public OpRewritePattern<func::FuncOp> {
  PromoteMemrefToPtrArg(MLIRContext *context)
      : OpRewritePattern<func::FuncOp>(context) {}

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    // Skip function declarations (no body) — e.g. spine_print_unranked_memref
    if (funcOp.isDeclaration())
      return failure();

    Block &entryBlock = funcOp.getBody().front();
    MLIRContext *ctx = funcOp->getContext();
    SmallVector<UnrealizedConversionCastOp> castOps;

    for (BlockArgument arg : entryBlock.getArguments()) {
      for (OpOperand &use : arg.getUses()) {
        if (auto castOp =
                dyn_cast<UnrealizedConversionCastOp>(use.getOwner())) {
          if (castOp.getNumResults() == 1 &&
              isLLVMPtrType(castOp.getResult(0).getType())) {
            castOps.push_back(castOp);
          }
        }
      }
    }

    if (castOps.empty())
      return failure();

    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type> newArgTypes(oldType.getInputs());

    for (auto &castOp : castOps) {
      newArgTypes.push_back(castOp.getResult(0).getType());
    }

    auto newFuncType =
        FunctionType::get(ctx, newArgTypes, oldType.getResults());

    Location loc = funcOp.getLoc();
    func::FuncOp newFunc =
        func::FuncOp::create(rewriter, loc, funcOp.getName(), newFuncType);

    newFunc->setAttrs(funcOp->getAttrs());

    newFunc.setType(newFuncType);

    Block *newEntryBlock = rewriter.createBlock(&newFunc.getBody());
    for (Type type : newArgTypes) {
      newEntryBlock->addArgument(type, loc);
    }

    for (int i = 0, e = oldType.getNumInputs(); i < e; i++) {
      if (DictionaryAttr attrs = funcOp.getArgAttrDict(i)) {
        newFunc.setArgAttrs(i, attrs.getValue());
      }
    }
    IRMapping mapper;

    unsigned numOrigArgs = oldType.getNumInputs();
    for (unsigned i = 0; i < numOrigArgs; i++) {
      mapper.map(entryBlock.getArgument(i), newEntryBlock->getArgument(i));
    }

    for (unsigned i = 0; i < castOps.size(); i++) {
      mapper.map(castOps[i].getResult(0),
                 newEntryBlock->getArgument(numOrigArgs + i));
    }

    for (Operation &op : entryBlock.getOperations()) {
      if (isa<UnrealizedConversionCastOp>(op)) {
        continue;
      }
      rewriter.clone(op, mapper);
    }

    rewriter.replaceOp(funcOp, newFunc->getResults());

    return success();
  }

private:
  bool isLLVMPtrType(Type type) const {
    if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type)) {
      return true;
    }
    return false;
  }
};

class ReconcileLlvmPtrCastsPass
    : public triton::impl::ReconcileLlvmPtrCastsBase<
          ReconcileLlvmPtrCastsPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<PromoteMemrefToPtrArg>(&getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createReconcileLlvmPtrCastsPass() {
  return std::make_unique<ReconcileLlvmPtrCastsPass>();
}
