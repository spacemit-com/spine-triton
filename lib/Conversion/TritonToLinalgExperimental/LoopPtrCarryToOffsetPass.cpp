//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/LoopPtrCarryToOffset.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace triton;

namespace mlir::triton {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_LOOPPTRCARRYTOOFFSET
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace mlir::triton

namespace {

static bool isScalarPtrType(Type t) {
  return isa<triton::PointerType>(t);
}

static bool isDefinedOutsideOf(Value v, Operation *scope) {
  if (auto blockArg = dyn_cast<BlockArgument>(v))
    return !scope->isAncestor(blockArg.getOwner()->getParentOp());
  Operation *defOp = v.getDefiningOp();
  return defOp && !scope->isAncestor(defOp);
}

struct RewriteInfo {
  // Stride added to the pointer each iteration; std::nullopt means the
  // pointer is forwarded unchanged.
  std::optional<Value> stride;
  // Integer type of the replacement offset argument.
  Type offsetType;
};

struct WhilePtrCarryToOffsetPattern : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    Block &before = whileOp.getBefore().front();
    Block &after = whileOp.getAfter().front();
    auto condOp = cast<scf::ConditionOp>(before.getTerminator());
    auto yieldOp = cast<scf::YieldOp>(after.getTerminator());
    ArrayRef<BlockArgument> beforeArgs = before.getArguments();
    ArrayRef<BlockArgument> afterArgs = after.getArguments();

    // The before block must be a plain "cmp + condition" head: at most one
    // non-terminator op, which must be an integer compare that does not touch
    // the carried pointers.
    Operation *cmpOp = nullptr;
    for (Operation &op : before.without_terminator()) {
      if (cmpOp)
        return failure();
      if (!isa<arith::CmpIOp>(op))
        return failure();
      cmpOp = &op;
    }

    // The condition must forward the before args unchanged (standard while
    // shape, not a do-while variant).
    if (condOp.getArgs().size() != beforeArgs.size())
      return failure();
    for (auto [fwd, arg] : llvm::zip(condOp.getArgs(), beforeArgs)) {
      if (fwd != arg)
        return failure();
    }

    // Identify the scalar-pointer iteration args and how each one is updated.
    SmallVector<size_t> ptrSlots;
    DenseMap<size_t, RewriteInfo> rewrites;
    for (auto [idx, arg] : llvm::enumerate(beforeArgs)) {
      if (!isScalarPtrType(arg.getType()))
        continue;
      ptrSlots.push_back(idx);
      Value newYield = yieldOp.getOperands()[idx];
      RewriteInfo info;
      if (newYield == afterArgs[idx]) {
        // Forwarded unchanged: keep the offset unchanged as well.
        info.stride = std::nullopt;
        info.offsetType = inferForwardedOffsetType(afterArgs[idx]);
      } else if (auto addPtrOp =
                     newYield.getDefiningOp<triton::AddPtrOp>()) {
        if (addPtrOp.getPtr() != afterArgs[idx])
          return failure();
        Value stride = addPtrOp.getOffset();
        if (!isDefinedOutsideOf(stride, whileOp))
          return failure();
        info.stride = stride;
        info.offsetType = stride.getType();
      } else {
        whileOp.emitRemark()
            << "LoopPtrCarryToOffset: unsupported update of carried pointer #"
            << idx << ", leaving loop unchanged";
        return failure();
      }
      rewrites[idx] = info;
    }
    if (ptrSlots.empty())
      return failure();
    if (cmpOp) {
      for (Value opnd : cmpOp->getOperands()) {
        for (size_t slot : ptrSlots) {
          if (opnd == beforeArgs[slot])
            return failure();
        }
      }
    }

    Location loc = whileOp.getLoc();
    SmallVector<Value> newInits;
    SmallVector<Type> newResultTypes;
    rewriter.setInsertionPoint(whileOp);
    for (auto [idx, init] : llvm::enumerate(whileOp.getInits())) {
      auto it = rewrites.find(idx);
      if (it == rewrites.end()) {
        newInits.push_back(init);
        newResultTypes.push_back(whileOp.getResult(idx).getType());
      } else {
        Type offsetType = it->second.offsetType;
        newInits.push_back(arith::ConstantOp::create(
            rewriter, loc, rewriter.getIntegerAttr(offsetType, 0)));
        newResultTypes.push_back(offsetType);
      }
    }

    scf::WhileOp newWhile = scf::WhileOp::create(
        rewriter, loc, newResultTypes, newInits,
        [&](OpBuilder &b, Location l, ValueRange newBeforeArgs) {
          IRMapping mapping;
          for (auto [oldArg, newArg] :
               llvm::zip(before.getArguments(), newBeforeArgs))
            mapping.map(oldArg, newArg);
          Value condVal;
          if (cmpOp) {
            Operation *newCmp = b.clone(*cmpOp, mapping);
            condVal = newCmp->getResult(0);
          } else {
            // before block only held the condition terminator: the loop is
            // infinite or the condition is a constant folded away. We cannot
            // get here in practice (scf.while requires a condition value),
            // so fall back to a true constant.
            condVal = arith::ConstantOp::create(b, l, b.getBoolAttr(true));
          }
          scf::ConditionOp::create(b, l, condVal, newBeforeArgs);
        },
        [&](OpBuilder &b, Location l, ValueRange newAfterArgs) {
          IRMapping mapping;
          for (auto [oldArg, newArg] : llvm::zip(after.getArguments(), newAfterArgs))
            mapping.map(oldArg, newArg);
          // Rebuild each carried pointer at the top of the body.
          for (size_t slot : ptrSlots) {
            Value rebuilt = triton::AddPtrOp::create(
                b, l, after.getArgument(slot).getType(),
                whileOp.getInits()[slot], newAfterArgs[slot]);
            mapping.map(after.getArgument(slot), rebuilt);
          }
          for (Operation &op : after.without_terminator())
            b.clone(op, mapping);
          SmallVector<Value> newYieldOperands;
          for (auto [idx, oldYieldVal] :
               llvm::enumerate(yieldOp.getOperands())) {
            auto it = rewrites.find(idx);
            if (it == rewrites.end()) {
              newYieldOperands.push_back(mapping.lookupOrNull(oldYieldVal)
                                             ? mapping.lookup(oldYieldVal)
                                             : oldYieldVal);
            } else if (it->second.stride) {
              newYieldOperands.push_back(arith::AddIOp::create(
                  b, l, newAfterArgs[idx], it->second.stride.value()));
            } else {
              newYieldOperands.push_back(newAfterArgs[idx]);
            }
          }
          scf::YieldOp::create(b, l, newYieldOperands);
        });

    // Rebuild pointers for uses of the loop results after the loop.
    SmallVector<Value> replacements;
    rewriter.setInsertionPointAfter(newWhile);
    for (auto [idx, result] : llvm::enumerate(whileOp.getResults())) {
      auto it = rewrites.find(idx);
      if (it == rewrites.end()) {
        replacements.push_back(newWhile.getResult(idx));
      } else {
        replacements.push_back(triton::AddPtrOp::create(
            rewriter, loc, result.getType(), whileOp.getInits()[idx],
            newWhile.getResult(idx)));
      }
    }
    rewriter.replaceOp(whileOp, replacements);
    return success();
  }

private:
  static Type inferForwardedOffsetType(BlockArgument ptrArg) {
    // A forwarded pointer carries no stride information; recover the offset
    // width from a scalar tt.addptr inside the body, default to i32.
    for (Operation &op : ptrArg.getOwner()->getOperations()) {
      auto addPtrOp = dyn_cast<triton::AddPtrOp>(op);
      if (addPtrOp && addPtrOp.getPtr() == ptrArg &&
          !isa<ShapedType>(addPtrOp.getOffset().getType()))
        return addPtrOp.getOffset().getType();
    }
    return IntegerType::get(ptrArg.getContext(), 32);
  }
};

struct LoopPtrCarryToOffsetPass
    : public triton::impl::LoopPtrCarryToOffsetBase<LoopPtrCarryToOffsetPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<WhilePtrCarryToOffsetPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createLoopPtrCarryToOffsetPass() {
  return std::make_unique<LoopPtrCarryToOffsetPass>();
}
