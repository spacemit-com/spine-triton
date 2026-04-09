//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/XSMTToLinalg/XSMTToLinalg.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "xsmt-to-linalg"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::memref;
using namespace mlir::tensor;
using namespace mlir::bufferization;
using namespace mlir::xsmt;

#include "triton-shared/Conversion/XSMTToLinalg/Passes.h.inc"

Value createZeroConstant(PatternRewriter &rewriter, Location loc,
                         Type elementType) {
  if (elementType.isF32()) {
    return arith::ConstantFloatOp::create(rewriter, loc, rewriter.getF32Type(),
                                          APFloat(0.0f));
  } else if (elementType.isF64()) {
    return arith::ConstantFloatOp::create(rewriter, loc, rewriter.getF64Type(),
                                          APFloat(0.0));
  } else if (elementType.isF16()) {
    return arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getF16Type(),
        APFloat::getZero(rewriter.getF16Type().getFloatSemantics()));
  } else if (elementType.isBF16()) {
    return arith::ConstantFloatOp::create(
        rewriter, loc, rewriter.getBF16Type(),
        APFloat::getZero(rewriter.getBF16Type().getFloatSemantics()));
  } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
    return arith::ConstantIntOp::create(rewriter, loc, 0, intType.getWidth());
  }
  return nullptr;
}

static bool isZeroOFRLocal(OpFoldResult ofr) {
  if (auto attr = llvm::dyn_cast<Attribute>(ofr)) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt() == 0;
    return false;
  }
  if (auto val = llvm::dyn_cast<Value>(ofr)) {
    if (auto cst = val.getDefiningOp<arith::ConstantIndexOp>())
      return cst.value() == 0;
    if (auto cst = val.getDefiningOp<arith::ConstantIntOp>())
      return cst.value() == 0;
    if (auto cast = val.getDefiningOp<arith::IndexCastOp>()) {
      if (auto cst = cast.getIn().getDefiningOp<arith::ConstantIntOp>())
        return cst.value() == 0;
    }
  }
  return false;
}

static bool isUnitOFRLocal(OpFoldResult ofr) {
  if (auto attr = llvm::dyn_cast<Attribute>(ofr)) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt() == 1;
    return false;
  }
  if (auto val = llvm::dyn_cast<Value>(ofr)) {
    if (auto cst = val.getDefiningOp<arith::ConstantIndexOp>())
      return cst.value() == 1;
    if (auto cst = val.getDefiningOp<arith::ConstantIntOp>())
      return cst.value() == 1;
    if (auto cast = val.getDefiningOp<arith::IndexCastOp>()) {
      if (auto cst = cast.getIn().getDefiningOp<arith::ConstantIntOp>())
        return cst.value() == 1;
    }
  }
  return false;
}

static bool sameValueOrEquivalentCast(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;

  if (auto cast = lhs.getDefiningOp<arith::IndexCastOp>())
    if (cast.getIn() == rhs)
      return true;
  if (auto cast = rhs.getDefiningOp<arith::IndexCastOp>())
    if (cast.getIn() == lhs)
      return true;

  return false;
}

static std::optional<int64_t> getConstantIntLike(Value v) {
  if (!v)
    return std::nullopt;
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return getConstantIntLike(cast.getIn());
  return std::nullopt;
}

static bool sameBoundedSizeExpr(Value lhs, Value rhs);

static Value peelTrivialIndexExpr(Value v) {
  while (v) {
    if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
      v = cast.getIn();
      continue;
    }
    if (auto minOp = v.getDefiningOp<arith::MinSIOp>()) {
      // min(x, C) where v is already known-bounded often appears repeatedly.
      // Keep the canonical SSA value rather than peeling further here.
      return v;
    }
    break;
  }
  return v;
}

static bool sameMinSIOp(arith::MinSIOp lhsMin, arith::MinSIOp rhsMin) {
  Value lhsA = peelTrivialIndexExpr(lhsMin.getLhs());
  Value lhsB = peelTrivialIndexExpr(lhsMin.getRhs());
  Value rhsA = peelTrivialIndexExpr(rhsMin.getLhs());
  Value rhsB = peelTrivialIndexExpr(rhsMin.getRhs());

  return (sameBoundedSizeExpr(lhsA, rhsA) && sameBoundedSizeExpr(lhsB, rhsB)) ||
         (sameBoundedSizeExpr(lhsA, rhsB) && sameBoundedSizeExpr(lhsB, rhsA));
}

static bool sameSubIOp(arith::SubIOp lhsSub, arith::SubIOp rhsSub) {
  return sameBoundedSizeExpr(lhsSub.getLhs(), rhsSub.getLhs()) &&
         sameBoundedSizeExpr(lhsSub.getRhs(), rhsSub.getRhs());
}

static bool sameBoundedSizeExpr(Value lhs, Value rhs) {
  lhs = peelTrivialIndexExpr(lhs);
  rhs = peelTrivialIndexExpr(rhs);

  if (!lhs || !rhs)
    return false;
  if (sameValueOrEquivalentCast(lhs, rhs))
    return true;

  if (auto lhsCst = getConstantIntLike(lhs)) {
    if (auto rhsCst = getConstantIntLike(rhs))
      return *lhsCst == *rhsCst;
  }

  auto lhsMin = lhs.getDefiningOp<arith::MinSIOp>();
  auto rhsMin = rhs.getDefiningOp<arith::MinSIOp>();
  if (lhsMin && rhsMin)
    return sameMinSIOp(lhsMin, rhsMin);

  auto lhsSub = lhs.getDefiningOp<arith::SubIOp>();
  auto rhsSub = rhs.getDefiningOp<arith::SubIOp>();
  if (lhsSub && rhsSub)
    return sameSubIOp(lhsSub, rhsSub);

  return false;
}

static bool sameOFR(OpFoldResult lhs, OpFoldResult rhs) {
  auto lhsAttr =
      llvm::dyn_cast_if_present<Attribute>(lhs.dyn_cast<Attribute>());
  auto rhsAttr =
      llvm::dyn_cast_if_present<Attribute>(rhs.dyn_cast<Attribute>());
  if (lhsAttr && rhsAttr)
    return lhsAttr == rhsAttr;

  Value lhsVal = lhs.dyn_cast<Value>();
  Value rhsVal = rhs.dyn_cast<Value>();
  if (lhsVal && rhsVal)
    return sameBoundedSizeExpr(lhsVal, rhsVal);

  if (lhsAttr && rhsVal) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(lhsAttr)) {
      if (auto cst = rhsVal.getDefiningOp<arith::ConstantIndexOp>())
        return intAttr.getInt() == cst.value();
      if (auto cst = rhsVal.getDefiningOp<arith::ConstantIntOp>())
        return intAttr.getInt() == cst.value();
    }
  }
  if (lhsVal && rhsAttr)
    return sameOFR(rhs, lhs);

  return false;
}

static bool canElideDescriptorSubview(Value rankedMemRef, Value off0,
                                      Value off1, Value size0, Value size1) {
  (void)size0;
  (void)size1;
  if (!isZeroOFRLocal(off0) || !isZeroOFRLocal(off1))
    return false;

  auto memrefTy = llvm::dyn_cast<MemRefType>(rankedMemRef.getType());
  return memrefTy && memrefTy.getRank() == 2;
}

static bool canElideIdentitySubview(Value memrefBaseValue,
                                    ArrayRef<OpFoldResult> sizes,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> strides) {
  auto memrefTy = llvm::dyn_cast<MemRefType>(memrefBaseValue.getType());
  if (!memrefTy)
    return false;
  if (memrefTy.getRank() != static_cast<int64_t>(offsets.size()) ||
      memrefTy.getRank() != static_cast<int64_t>(strides.size()) ||
      memrefTy.getRank() != static_cast<int64_t>(sizes.size()))
    return false;

  if (!llvm::all_of(offsets,
                    [](OpFoldResult ofr) { return isZeroOFRLocal(ofr); }) ||
      !llvm::all_of(strides,
                    [](OpFoldResult ofr) { return isUnitOFRLocal(ofr); }))
    return false;

  if (auto reinterpretCast =
          memrefBaseValue.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto baseSizes = reinterpretCast.getMixedSizes();
    if (baseSizes.size() != sizes.size())
      return false;
    return llvm::all_of(llvm::zip(sizes, baseSizes), [](auto it) {
      return sameOFR(std::get<0>(it), std::get<1>(it));
    });
  }

  return true;
}

inline bool isInsideLoop(mlir::Operation *op) {
  mlir::Operation *parent = op->getParentOp();
  while (parent) {
    if (llvm::isa<mlir::scf::ForOp, mlir::scf::WhileOp, mlir::scf::ParallelOp>(
            parent)) {
      return true;
    }
    parent = parent->getParentOp();
  }
  return false;
}

struct MMT4DBindFusionPattern : public OpRewritePattern<xsmt::BindOp> {
  MMT4DBindFusionPattern(MLIRContext *context)
      : OpRewritePattern<xsmt::BindOp>(context) {}

  LogicalResult matchAndRewrite(xsmt::BindOp bindOp,
                                PatternRewriter &rewriter) const override {

    Value mmt4dResult = bindOp.getDest();
    Value accumulator = bindOp.getSrc();

    auto mmt4dOp = mmt4dResult.getDefiningOp<xsmt::MMT4DOp>();
    if (!mmt4dOp) {
      return failure();
    }

    if (mmt4dOp.getC()) {
      return failure();
    }

    rewriter.setInsertionPoint(bindOp);

    auto newMMT4DOp = xsmt::MMT4DOp::create(
        rewriter, bindOp.getLoc(), bindOp.getResult().getType(), mmt4dOp.getA(),
        mmt4dOp.getB(), accumulator);

    rewriter.replaceOp(bindOp, newMMT4DOp.getResult());

    if (mmt4dOp->use_empty()) {
      rewriter.eraseOp(mmt4dOp);
    }

    return success();
  }
};

template <typename OpTy>
struct MBarrierLoopCheckPattern : public OpRewritePattern<OpTy> {
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Operation *currentParent = op->getParentOp();

    while (currentParent) {
      if (isInsideLoop(op.getOperation())) {
        return op.emitError() << "'" << op->getName()
                              << "' operation cannot be used inside a loop. ";
      }
      if (isa<FunctionOpInterface>(currentParent)) {
        break;
      }
      currentParent = currentParent->getParentOp();
    }
    return failure();
  }
};

template <typename OpTy>
struct CheckGlobalMBarrierInLoopPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value barrier = op.getMbarrier();

    Operation *definingOp = barrier.getDefiningOp();
    if (!definingOp) {
      return failure();
    }

    if (!llvm::isa<xsmt::GlobalMBarrierInitOp>(definingOp)) {
      return failure();
    }

    if (!isInsideLoop(op.getOperation())) {
      return failure();
    }

    op.emitError()
        << "'" << op->getName() << "' operation with barrier defined by "
        << "'xsmt.global_mbarrier_init' cannot be used inside a loop. "
        << "Global barriers are not designed for loop-level synchronization.";

    return failure();
  }
};

struct FillOpPattern : public OpRewritePattern<linalg::FillOp> {
public:
  FillOpPattern(MLIRContext *context)
      : OpRewritePattern<linalg::FillOp>(context) {}

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {

    Location loc = fillOp.getLoc();
    Value fillValue = fillOp.getInputs()[0];
    Value fillOutput = fillOp.getOutputs()[0];

    auto outputType = dyn_cast<TensorType>(fillOutput.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(fillOp,
                                         "FAIL: Output is not a tensor type");
    }

    SmallVector<int64_t> shape(outputType.getShape().begin(),
                               outputType.getShape().end());

    auto allocOp = memref::AllocOp::create(
        rewriter, loc, MemRefType::get(shape, outputType.getElementType()));
    PatternRewriter::InsertionGuard guard(rewriter);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    SmallVector<Value> ivs;

    scf::ForOp outermostLoop;

    for (size_t i = 0; i < shape.size(); ++i) {
      Value dimLimit = arith::ConstantIndexOp::create(rewriter, loc, shape[i]);
      auto loop = scf::ForOp::create(rewriter, loc, c0, dimLimit, c1);

      if (i == 0) {
        outermostLoop = loop;
      }

      ivs.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    memref::StoreOp::create(rewriter, loc, fillValue, allocOp, ivs);
    if (outermostLoop) {
      rewriter.setInsertionPointAfter(outermostLoop);
    }

    auto toTensorOp = bufferization::ToTensorOp::create(
        rewriter, loc, outputType, allocOp, /*restrict=*/true);

    rewriter.replaceOp(fillOp, toTensorOp.getResult());
    return success();
  }
};

struct ViewOpPattern : public OpRewritePattern<xsmt::ViewOp> {
public:
  ViewOpPattern(MLIRContext *context)
      : OpRewritePattern<xsmt::ViewOp>(context) {}

  LogicalResult matchAndRewrite(xsmt::ViewOp viewOp,
                                PatternRewriter &rewriter) const override {

    Location loc = viewOp.getLoc();
    Value base = viewOp.getBase();
    ValueRange offsets = viewOp.getOffsets();
    ArrayRef<int32_t> shape = viewOp.getShape();
    ArrayRef<int32_t> microSize = viewOp.getMicroSize();

    auto resultType = dyn_cast<TensorType>(viewOp.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(viewOp, "Result is not a tensor type");
    }
    Type elementType = resultType.getElementType();
    if (shape.size() != 2 || microSize.size() != 2) {
      return rewriter.notifyMatchFailure(
          viewOp, "Expected shape and micro_size to have 2 elements each");
    }
    if (offsets.size() != 2) {
      return rewriter.notifyMatchFailure(viewOp,
                                         "Expected exactly 2 offset values");
    }
    auto baseType = dyn_cast<TensorType>(base.getType());
    if (!baseType) {
      return rewriter.notifyMatchFailure(viewOp, "Base is not a tensor type");
    }

    // Optimization: for FillOp base (e.g. zero-initialized tensor),
    // skip the redundant alloc → materialize → subview → to_tensor chain.
    // We can directly pack the fill result tensor without materializing to
    // memref. This avoids generating:
    //   %alloc = memref.alloc()
    //   bufferization.materialize_in_destination %fill in writable %alloc
    //   %cast = memref.cast %alloc
    //   %t = bufferization.to_tensor %cast restrict
    //   linalg.pack %t ...
    // Instead, we generate:
    //   linalg.pack %fill ...
    if (auto fillOp = base.getDefiningOp<linalg::FillOp>()) {
      auto tensorType = cast<RankedTensorType>(fillOp.getResult(0).getType());
      auto shapes = tensorType.getShape();
      SmallVector<OpFoldResult> sizes;
      for (int64_t s : shapes) {
        sizes.push_back(rewriter.getIndexAttr(s));
      }
      return convertFillDirect(viewOp, base, sizes, rewriter);
    }

    auto memrefType =
        MemRefType::get(baseType.getShape(), baseType.getElementType());
    Value memref = memref::AllocOp::create(rewriter, loc, memrefType);

    auto materialize = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, base, memref);
    materialize.setWritable(true);

    bool hasMicroSize = false;
    bool isSameMicroSize = false;
    ArrayRef<int32_t> preMicroSize;

    if (auto ViewOp = base.getDefiningOp<xsmt::ViewOp>()) {
      preMicroSize = ViewOp.getMicroSize();
    } else if (auto loadOp = base.getDefiningOp<xsmt::DescriptorLoadViewOp>()) {
      preMicroSize = loadOp.getMicroSize();
    }

    if (!preMicroSize.empty()) {
      hasMicroSize = true;
      if (preMicroSize == microSize) {
        isSameMicroSize = true;
      }
    }

    SmallVector<OpFoldResult> sizes;
    if (auto ViewOp = base.getDefiningOp<xsmt::ViewOp>()) {
      auto shapes = ViewOp.getShape();
      for (int64_t shape : shapes) {
        sizes.push_back(rewriter.getIndexAttr(shape));
      }
    } else {
      if (baseType.getRank() == 2) {
        auto shapes = baseType.getShape();
        for (int64_t s : shapes) {
          sizes.push_back(rewriter.getIndexAttr(s));
        }
        return convertNormal(viewOp, memref, sizes, rewriter);
      } else if (baseType.getRank() == 4) {
        return convertIsDiffMicroSize(viewOp, memref, rewriter);
      } else {
        return failure();
      }
    }

    if (!hasMicroSize && !isSameMicroSize) {
      return convertNormal(viewOp, memref, sizes, rewriter);
    } else if (hasMicroSize && isSameMicroSize) {
      return convertIsSameMicroSize(viewOp, memref, sizes, rewriter);
    } else if (hasMicroSize && !isSameMicroSize) {
      return convertIsDiffMicroSize(viewOp, memref, rewriter);
    } else {
      return failure();
    }
  }

private:
  /// Directly pack a FillOp result tensor without going through
  /// alloc → materialize → subview → to_tensor. This eliminates the
  /// redundant bufferization roundtrip for zero-initialized tensors.
  LogicalResult convertFillDirect(xsmt::ViewOp viewOp, Value baseTensor,
                                  SmallVector<OpFoldResult> sizes,
                                  PatternRewriter &rewriter) const {
    Location loc = viewOp.getLoc();
    ValueRange offsets = viewOp.getOffsets();
    ArrayRef<int32_t> shape = viewOp.getShape();
    ArrayRef<int32_t> microSize = viewOp.getMicroSize();

    auto resultType = cast<RankedTensorType>(viewOp.getResult().getType());
    Type elementType = resultType.getElementType();

    SmallVector<OpFoldResult> offsetValues;
    for (auto offset : offsets) {
      offsetValues.push_back(ensureIndexType(loc, offset, rewriter));
    }

    // Compute actual slice size: min(baseSize - offset, shape)
    auto size_m1 = subOFRs(sizes[0], offsetValues[0], loc, rewriter);
    Value size_m2 = ofrToIndexValue(size_m1, loc, rewriter);
    Value shape0 = ConstantIndexOp::create(rewriter, loc, shape[0]);
    Value size_m3 = MinSIOp::create(rewriter, loc, size_m2, shape0);

    auto size_n1 = subOFRs(sizes[1], offsetValues[1], loc, rewriter);
    Value size_n2 = ofrToIndexValue(size_n1, loc, rewriter);
    Value shape1 = ConstantIndexOp::create(rewriter, loc, shape[1]);
    Value size_n3 = MinSIOp::create(rewriter, loc, size_n2, shape1);

    // Extract the slice from the fill result tensor directly (no memref
    // roundtrip)
    SmallVector<int64_t, 2> sliceStaticShape = {ShapedType::kDynamic,
                                                ShapedType::kDynamic};
    auto sliceType = RankedTensorType::get(sliceStaticShape, elementType);
    Value slice = tensor::ExtractSliceOp::create(
        rewriter, loc, sliceType, baseTensor,
        ArrayRef<OpFoldResult>{offsetValues[0], offsetValues[1]},
        ArrayRef<OpFoldResult>{size_m3, size_n3},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                               rewriter.getIndexAttr(1)});

    // Pack directly
    Value tile0Value = ConstantIndexOp::create(rewriter, loc, microSize[0]);
    Value tile1Value = ConstantIndexOp::create(rewriter, loc, microSize[1]);
    Value packedRows = createCeilDivUI(rewriter, loc, size_m3, tile0Value);
    Value packedCols = createCeilDivUI(rewriter, loc, size_n3, tile1Value);

    SmallVector<int64_t, 4> shapeDims;
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(microSize[0]);
    shapeDims.push_back(microSize[1]);
    auto emptyType = RankedTensorType::get(shapeDims, elementType);
    Value emptyTensor = EmptyOp::create(rewriter, loc, emptyType,
                                        ValueRange{packedRows, packedCols});

    Value paddingValue = createZeroConstant(rewriter, loc, elementType);
    if (!paddingValue) {
      return rewriter.notifyMatchFailure(
          viewOp, "Unsupported element type for padding");
    }

    Value packedResult =
        linalg::PackOp::create(
            rewriter, loc, slice, emptyTensor, ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(microSize[0]),
                                   rewriter.getIndexAttr(microSize[1])},
            paddingValue, ArrayRef<int64_t>{0, 1})
            .getResult();

    Value castResult =
        tensor::CastOp::create(rewriter, loc, resultType, packedResult);
    rewriter.replaceOp(viewOp, castResult);
    return success();
  }

  LogicalResult convertNormal(xsmt::ViewOp viewOp, Value memrefBaseValue,
                              SmallVector<OpFoldResult> sizes,
                              PatternRewriter &rewriter) const {
    Location loc = viewOp.getLoc();
    Value base = viewOp.getBase();
    ValueRange offsets = viewOp.getOffsets();
    ArrayRef<int32_t> shape = viewOp.getShape();
    ArrayRef<int32_t> microSize = viewOp.getMicroSize();

    auto resultType = cast<TensorType>(viewOp.getResult().getType());
    Type elementType = resultType.getElementType();

    SmallVector<OpFoldResult> offsetValues;

    for (auto offset : offsets) {
      offsetValues.push_back(ensureIndexType(loc, offset, rewriter));
    }

    Value c1 = ConstantIndexOp::create(rewriter, loc, 1);
    auto size_m1 = subOFRs(sizes[0], offsetValues[0], loc, rewriter);
    Value size_m2 = ofrToIndexValue(size_m1, loc, rewriter);
    Value shape0 = ConstantIndexOp::create(rewriter, loc, shape[0]);
    Value size_m3 = MinSIOp::create(rewriter, loc, size_m2, shape0);

    auto size_n1 = subOFRs(sizes[1], offsetValues[1], loc, rewriter);
    Value size_n2 = ofrToIndexValue(size_n1, loc, rewriter);
    Value shape1 = ConstantIndexOp::create(rewriter, loc, shape[1]);
    Value size_n3 = MinSIOp::create(rewriter, loc, size_n2, shape1);

    SmallVector<OpFoldResult> strideValues{c1, c1};
    Value toTensorSource = memrefBaseValue;
    auto tensorType = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, elementType);

    if (!canElideIdentitySubview(memrefBaseValue,
                                 ArrayRef<OpFoldResult>{size_m3, size_n3},
                                 offsetValues, strideValues)) {
      auto subview = memref::SubViewOp::create(
          rewriter, loc, memrefBaseValue, offsetValues,
          ArrayRef<OpFoldResult>{size_m3, size_n3}, strideValues);
      toTensorSource = subview.getResult();
      auto subviewType = cast<MemRefType>(subview.getType());
      tensorType = RankedTensorType::get(subviewType.getShape(),
                                         subviewType.getElementType());
    }

    Value toTensorOp2 = bufferization::ToTensorOp::create(
        rewriter, loc, tensorType, toTensorSource,
        /*restrict=*/rewriter.getUnitAttr());

    Value tile0Value = ConstantIndexOp::create(rewriter, loc, microSize[0]);
    Value tile1Value = ConstantIndexOp::create(rewriter, loc, microSize[1]);
    Value packedRows = createCeilDivUI(rewriter, loc, size_m3, tile0Value);
    Value packedCols = createCeilDivUI(rewriter, loc, size_n3, tile1Value);

    SmallVector<int64_t, 4> shapeDims;
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(microSize[0]);
    shapeDims.push_back(microSize[1]);
    auto emptyType =
        RankedTensorType::get(shapeDims, tensorType.getElementType());
    Value emptyTensor = EmptyOp::create(rewriter, loc, emptyType,
                                        ValueRange{packedRows, packedCols});

    Value paddingValue = createZeroConstant(rewriter, loc, elementType);
    if (!paddingValue) {
      return rewriter.notifyMatchFailure(
          viewOp, "Unsupported element type for padding");
    }

    Value packedResult =
        linalg::PackOp::create(
            rewriter, loc, toTensorOp2, emptyTensor, ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(microSize[0]),
                                   rewriter.getIndexAttr(microSize[1])},
            paddingValue, ArrayRef<int64_t>{0, 1})
            .getResult();

    auto originalResultType =
        cast<RankedTensorType>(viewOp.getResult().getType());
    Value castResult =
        tensor::CastOp::create(rewriter, loc, originalResultType, packedResult);
    rewriter.replaceOp(viewOp, castResult);
    return success();
  }

  LogicalResult convertIsSameMicroSize(xsmt::ViewOp viewOp,
                                       Value memrefBaseValue,
                                       SmallVector<OpFoldResult> sizes,
                                       PatternRewriter &rewriter) const {

    Location loc = viewOp.getLoc();
    Value base = viewOp.getBase();
    ValueRange offsets = viewOp.getOffsets();
    ArrayRef<int32_t> shape = viewOp.getShape();
    ArrayRef<int32_t> microSize = viewOp.getMicroSize();

    auto resultType = cast<TensorType>(viewOp.getResult().getType());
    Type elementType = resultType.getElementType();

    SmallVector<OpFoldResult> offsetValues;

    for (auto offset : offsets) {
      offsetValues.push_back(ensureIndexType(loc, offset, rewriter));
    }

    Value c1 = ConstantIndexOp::create(rewriter, loc, 1);
    auto size_m1 = subOFRs(sizes[0], offsetValues[0], loc, rewriter);
    Value size_m2 = ofrToIndexValue(size_m1, loc, rewriter);
    Value shape0 = ConstantIndexOp::create(rewriter, loc, shape[0]);
    Value size_m3 = MinSIOp::create(rewriter, loc, size_m2, shape0);

    auto size_n1 = subOFRs(sizes[1], offsetValues[1], loc, rewriter);
    Value size_n2 = ofrToIndexValue(size_n1, loc, rewriter);
    Value shape1 = ConstantIndexOp::create(rewriter, loc, shape[1]);
    Value size_n3 = MinSIOp::create(rewriter, loc, size_n2, shape1);

    Value tile0Value = ConstantIndexOp::create(rewriter, loc, microSize[0]);
    Value tile1Value = ConstantIndexOp::create(rewriter, loc, microSize[1]);
    Value size0 = createCeilDivUI(rewriter, loc, size_m3, tile0Value);
    Value size1 = createCeilDivUI(rewriter, loc, size_n3, tile1Value);

    Value offset0 = createCeilDivUI(rewriter, loc, offsets[0], tile0Value);
    Value offset1 = createCeilDivUI(rewriter, loc, offsets[1], tile1Value);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);

    auto subview = memref::SubViewOp::create(
        rewriter, loc, memrefBaseValue,
        ArrayRef<OpFoldResult>{offset0, offset1, c0, c0},
        ArrayRef<OpFoldResult>{size0, size1, tile0Value, tile1Value},
        ArrayRef<OpFoldResult>{c1, c1, c1, c1});

    auto subviewMemRefType = subview.getType();
    auto subviewTensorType = RankedTensorType::get(
        subviewMemRefType.getShape(), subviewMemRefType.getElementType());

    Value newTensor = bufferization::ToTensorOp::create(
        rewriter, loc, subviewTensorType, subview.getResult(),
        /*restrict=*/rewriter.getUnitAttr());

    Value castResult =
        tensor::CastOp::create(rewriter, loc, resultType, newTensor);
    rewriter.replaceOp(viewOp, castResult);

    return success();
  }

  LogicalResult convertIsDiffMicroSize(xsmt::ViewOp viewOp,
                                       Value memrefBaseValue,
                                       PatternRewriter &rewriter) const {
    Location loc = viewOp.getLoc();
    Value base = viewOp.getBase();
    ValueRange offsets = viewOp.getOffsets();
    ArrayRef<int32_t> shape = viewOp.getShape();
    ArrayRef<int32_t> microSize = viewOp.getMicroSize();

    auto resultType = cast<RankedTensorType>(viewOp.getResult().getType());
    Type elementType = resultType.getElementType();

    bool other = false;

    ArrayRef<int32_t> preMicroSize;
    if (auto prevViewOp = base.getDefiningOp<xsmt::ViewOp>()) {
      preMicroSize = prevViewOp.getMicroSize();
    } else if (auto loadOp = base.getDefiningOp<xsmt::DescriptorLoadViewOp>()) {
      preMicroSize = loadOp.getMicroSize();
    } else {
      other = true;
    }

    Value packedTensor = memrefBaseValue;
    ShapedType shapedTy = cast<ShapedType>(memrefBaseValue.getType());
    if (!isa<RankedTensorType>(memrefBaseValue.getType())) {
      auto tensorTy =
          RankedTensorType::get(shapedTy.getShape(), shapedTy.getElementType());
      packedTensor = bufferization::ToTensorOp::create(
          rewriter, loc, tensorTy, memrefBaseValue,
          rewriter.getUnitAttr()); // restrict=true
    }

    if (microSize[0] == 1 && microSize[1] == 1) {
      Value flatTensor = packedTensor;

      Value dim0 = tensor::DimOp::create(rewriter, loc, packedTensor, 0);
      Value dim1 = tensor::DimOp::create(rewriter, loc, packedTensor, 1);

      int64_t dim2 = shapedTy.getDimSize(2);
      int64_t dim3 = shapedTy.getDimSize(3);

      if (other) {
        Value flatRows = arith::MulIOp::create(
            rewriter, loc, dim0,
            arith::ConstantIndexOp::create(rewriter, loc, dim2));
        Value flatCols = arith::MulIOp::create(
            rewriter, loc, dim1,
            arith::ConstantIndexOp::create(rewriter, loc, dim3));

        auto flatTy = RankedTensorType::get(
            {ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
        Value empty = tensor::EmptyOp::create(rewriter, loc, flatTy,
                                              ValueRange{flatRows, flatCols});

        flatTensor =
            linalg::UnPackOp::create(
                rewriter, loc, packedTensor, empty, ArrayRef<int64_t>{0, 1},
                ArrayRef<OpFoldResult>{rewriter.getIndexAttr(dim2),
                                       rewriter.getIndexAttr(dim3)})
                .getResult();
      } else {
        Value flatRows = arith::MulIOp::create(
            rewriter, loc, dim0,
            arith::ConstantIndexOp::create(rewriter, loc, preMicroSize[0]));
        Value flatCols = arith::MulIOp::create(
            rewriter, loc, dim1,
            arith::ConstantIndexOp::create(rewriter, loc, preMicroSize[1]));

        auto flatTy = RankedTensorType::get(
            {ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
        Value empty = tensor::EmptyOp::create(rewriter, loc, flatTy,
                                              ValueRange{flatRows, flatCols});

        flatTensor =
            linalg::UnPackOp::create(
                rewriter, loc, packedTensor, empty, ArrayRef<int64_t>{0, 1},
                ArrayRef<OpFoldResult>{rewriter.getIndexAttr(preMicroSize[0]),
                                       rewriter.getIndexAttr(preMicroSize[1])})
                .getResult();
      }

      Value off0 = ensureIndexType(loc, offsets[0], rewriter);
      Value off1 = ensureIndexType(loc, offsets[1], rewriter);
      Value reqRows = arith::ConstantIndexOp::create(rewriter, loc, shape[0]);
      Value reqCols = arith::ConstantIndexOp::create(rewriter, loc, shape[1]);

      Value flatRows = tensor::DimOp::create(rewriter, loc, flatTensor, 0);
      Value flatCols = tensor::DimOp::create(rewriter, loc, flatTensor, 1);
      Value availRows = arith::SubIOp::create(rewriter, loc, flatRows, off0);
      Value availCols = arith::SubIOp::create(rewriter, loc, flatCols, off1);
      Value actualRows =
          arith::MinUIOp::create(rewriter, loc, reqRows, availRows);
      Value actualCols =
          arith::MinUIOp::create(rewriter, loc, reqCols, availCols);

      Value slice = tensor::ExtractSliceOp::create(
          rewriter, loc, flatTensor, ArrayRef<Value>{off0, off1},
          ArrayRef<Value>{actualRows, actualCols},
          ArrayRef<Value>{arith::ConstantIndexOp::create(rewriter, loc, 1),
                          arith::ConstantIndexOp::create(rewriter, loc, 1)});

      Value result = tensor::CastOp::create(rewriter, loc, resultType, slice);
      rewriter.replaceOp(viewOp, result);
      return success();
    }

    Value dim0 = tensor::DimOp::create(rewriter, loc, packedTensor, 0);
    Value dim1 = tensor::DimOp::create(rewriter, loc, packedTensor, 1);

    int64_t dim2 = shapedTy.getDimSize(2);
    int64_t dim3 = shapedTy.getDimSize(3);

    Value flatRows = arith::MulIOp::create(
        rewriter, loc, dim0,
        arith::ConstantIndexOp::create(rewriter, loc, dim2));
    Value flatCols = arith::MulIOp::create(
        rewriter, loc, dim1,
        arith::ConstantIndexOp::create(rewriter, loc, dim3));

    auto flatTy = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
    Value flatEmpty = tensor::EmptyOp::create(rewriter, loc, flatTy,
                                              ValueRange{flatRows, flatCols});

    Value flatTensor =
        linalg::UnPackOp::create(
            rewriter, loc, packedTensor, flatEmpty, ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(dim2),
                                   rewriter.getIndexAttr(dim3)})
            .getResult();

    Value off0 = ensureIndexType(loc, offsets[0], rewriter);
    Value off1 = ensureIndexType(loc, offsets[1], rewriter);
    Value reqRows = arith::ConstantIndexOp::create(rewriter, loc, shape[0]);
    Value reqCols = arith::ConstantIndexOp::create(rewriter, loc, shape[1]);

    Value availRows = arith::SubIOp::create(
        rewriter, loc, tensor::DimOp::create(rewriter, loc, flatTensor, 0),
        off0);
    Value availCols = arith::SubIOp::create(
        rewriter, loc, tensor::DimOp::create(rewriter, loc, flatTensor, 1),
        off1);
    Value actualRows =
        arith::MinUIOp::create(rewriter, loc, reqRows, availRows);
    Value actualCols =
        arith::MinUIOp::create(rewriter, loc, reqCols, availCols);

    Value slice = tensor::ExtractSliceOp::create(
        rewriter, loc, flatTensor, ArrayRef<Value>{off0, off1},
        ArrayRef<Value>{actualRows, actualCols},
        ArrayRef<Value>{arith::ConstantIndexOp::create(rewriter, loc, 1),
                        arith::ConstantIndexOp::create(rewriter, loc, 1)});

    Value tile0 = arith::ConstantIndexOp::create(rewriter, loc, microSize[0]);
    Value tile1 = arith::ConstantIndexOp::create(rewriter, loc, microSize[1]);
    Value outerRows = createCeilDivUI(rewriter, loc, actualRows, tile0);
    Value outerCols = createCeilDivUI(rewriter, loc, actualCols, tile1);

    SmallVector<int64_t> packedShape = {
        ShapedType::kDynamic, ShapedType::kDynamic, microSize[0], microSize[1]};
    auto packedTy = RankedTensorType::get(packedShape, elementType);
    Value packedEmpty = tensor::EmptyOp::create(
        rewriter, loc, packedTy, ValueRange{outerRows, outerCols});

    Value padding = createZeroConstant(rewriter, loc, elementType);
    if (!padding)
      return rewriter.notifyMatchFailure(viewOp, "Cannot create padding value");

    Value packed =
        linalg::PackOp::create(
            rewriter, loc, slice, packedEmpty, ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(microSize[0]),
                                   rewriter.getIndexAttr(microSize[1])},
            padding)
            .getResult();

    Value result = tensor::CastOp::create(rewriter, loc, resultType, packed);
    rewriter.replaceOp(viewOp, result);
    return success();
  }
};

struct LowerXSMTMMT4D : public OpRewritePattern<xsmt::MMT4DOp> {
  using OpRewritePattern::OpRewritePattern;

  /// Returns true if the memref type is known to be contiguous in the last
  /// dimension (i.e. last stride == 1). We use this as a conservative guard for
  /// the "unpack directly into output buffer" optimization because downstream
  /// lowering (e.g. spestruct.unpack) may require the destination/result memref
  /// types to match exactly.
  static bool hasUnitStrideInLastDim(MemRefType ty) {
    if (!ty.hasStaticShape()) {
      // Shape dynamic is fine; we only care about layout.
    }
    if (!ty.getLayout().isIdentity()) {
      int64_t offset = 0;
      SmallVector<int64_t, 4> strides;
      if (failed(ty.getStridesAndOffset(strides, offset)))
        return false;
      if (strides.empty())
        return false;
      return strides.back() == 1;
    }
    // Identity layout implies last-dim stride is 1.
    return true;
  }

  LogicalResult matchAndRewrite(xsmt::MMT4DOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value a = op.getA();
    Value b = op.getB();
    Value c = op.getC();

    if (auto cast = a.getDefiningOp<tensor::CastOp>())
      a = cast.getSource();
    if (auto cast = b.getDefiningOp<tensor::CastOp>())
      b = cast.getSource();

    bool unpack = false;

    if (c) {
      if (c.hasOneUse() && op.getResult().use_empty())
        unpack = true;
      if (unpack) {
        if (auto cast = c.getDefiningOp<tensor::CastOp>())
          c = cast.getSource();
        auto cPackOp = c.getDefiningOp<linalg::PackOp>();
        if (!cPackOp)
          return rewriter.notifyMatchFailure(op,
                                             "C must be result of linalg.pack");

        Value packInput = cPackOp.getSource();
        auto toTensorOp = packInput.getDefiningOp<bufferization::ToTensorOp>();
        if (!toTensorOp)
          return rewriter.notifyMatchFailure(
              op, "Packed input must come from to_tensor");

        Value subview = toTensorOp->getOperand(0);

        auto cType = dyn_cast<RankedTensorType>(c.getType());
        auto zeroAttr = FloatAttr::get(cType.getElementType(), 0.0);
        auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);

        auto fillOp =
            linalg::FillOp::create(rewriter, loc, TypeRange{c.getType()},
                                   ValueRange{zero}, ValueRange{c});

        auto mmt4dOp = linalg::Mmt4DOp::create(rewriter, loc, ValueRange{a, b},
                                               ValueRange{fillOp.getResult(0)});

        // Unpack directly into the output buffer (subview) as in
        // /home/share/nfs_share/zuoweixia/ir_offline/mm_kernel_linalg1.mlir.
        //
        // IMPORTANT: Use the *same* memref subview that backs the pack input
        // (operand of bufferization.to_tensor feeding linalg.pack). This keeps
        // dest/result types stable for downstream lowerings.
        Value destination = subview;
        auto destMemrefTy = cast<MemRefType>(destination.getType());

        // Create a dynamic-shaped destination tensor view backed by the memref.
        auto elemTy = destMemrefTy.getElementType();
        auto dstTensorTy = RankedTensorType::get(
            {ShapedType::kDynamic, ShapedType::kDynamic}, elemTy);
        auto dstTensor = bufferization::ToTensorOp::create(
            rewriter, loc, dstTensorTy, destination,
            /*restrict=*/rewriter.getUnitAttr(),
            /*writable=*/rewriter.getUnitAttr());

        // Unpack into the destination tensor (should alias the output buffer).
        ArrayRef<int64_t> staticInnerTiles = cPackOp.getStaticInnerTiles();
        if (staticInnerTiles.empty())
          return rewriter.notifyMatchFailure(op, "static_inner_tiles is empty");
        SmallVector<OpFoldResult> innerTiles;
        for (int64_t tile : staticInnerTiles)
          innerTiles.push_back(rewriter.getIndexAttr(tile));

        auto unpackOp = linalg::UnPackOp::create(
            rewriter, loc, mmt4dOp.getResult(0), dstTensor.getResult(),
            cPackOp.getInnerDimsPos(), innerTiles, cPackOp.getOuterDimsPerm());

        // Keep an explicit anchor write-back. If aliasing is proven, this
        // should become a no-op; otherwise it preserves semantics.
        auto materializeOp = bufferization::MaterializeInDestinationOp::create(
            rewriter, unpackOp.getLoc(), unpackOp.getResult(), destination);
        materializeOp.setWritable(true);

        rewriter.replaceOp(op, unpackOp.getResult());
        return success();
      } else {
        if (auto cast = c.getDefiningOp<tensor::CastOp>())
          c = cast.getSource();
        auto toTensorOp = c.getDefiningOp<bufferization::ToTensorOp>();
        if (!toTensorOp)
          return rewriter.notifyMatchFailure(op, "c must come from to_tensor");

        auto subview = toTensorOp->getOperand(0);

        auto cType = dyn_cast<RankedTensorType>(c.getType());
        auto zeroAttr = FloatAttr::get(cType.getElementType(), 0.0);
        auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);

        auto fillOp =
            linalg::FillOp::create(rewriter, loc, TypeRange{c.getType()},
                                   ValueRange{zero}, ValueRange{c});

        auto mmt4dOp = linalg::Mmt4DOp::create(rewriter, loc, ValueRange{a, b},
                                               ValueRange{fillOp.getResult(0)});

        auto materializeOp = bufferization::MaterializeInDestinationOp::create(
            rewriter, mmt4dOp.getLoc(), mmt4dOp.getResult(0), subview);
        materializeOp.setWritable(true);
        rewriter.replaceOp(op, mmt4dOp.getResult(0));
        return success();
      }
    } else {
      auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
      Value initTensor = tensor::EmptyOp::create(
          rewriter, loc, resultType.getShape(), resultType.getElementType());
      auto zeroAttr = FloatAttr::get(resultType.getElementType(), 0.0);
      auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);

      auto fillOp = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                           ValueRange{initTensor});

      auto mmt4dOp = linalg::Mmt4DOp::create(rewriter, loc, ValueRange{a, b},
                                             ValueRange{fillOp.getResult(0)});
      rewriter.replaceOp(op, mmt4dOp.getResult(0));
      return success();
    }
  }
  Value traceDestinationFromOutput(Value output) const {
    if (auto packOp = output.getDefiningOp<linalg::PackOp>()) {
      Value packInput = packOp.getSource();
      if (auto toTensorOp =
              packInput.getDefiningOp<bufferization::ToTensorOp>()) {
        return toTensorOp->getOperand(0);
      }

      return packInput;
    }
    if (auto toTensorOp = output.getDefiningOp<bufferization::ToTensorOp>()) {
      return toTensorOp->getOperand(0);
    }
    return Value();
  }
};

struct ConvertMMT4DAddPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    for (auto it : genericOp.getIteratorTypesArray()) {
      if (it != utils::IteratorType::parallel)
        return rewriter.notifyMatchFailure(genericOp, "non-parallel iterators");
    }

    auto maps = genericOp.getIndexingMaps();
    if (maps.size() != 3)
      return rewriter.notifyMatchFailure(genericOp, "wrong number of maps");

    auto &block = genericOp.getRegion().front();
    if (!llvm::hasSingleElement(block.without_terminator())) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "multiple block operations");
    }

    auto addOp = dyn_cast<arith::AddFOp>(*block.begin());
    if (!addOp)
      return rewriter.notifyMatchFailure(genericOp, "not an addf operation");
    if (addOp.getLhs() != block.getArgument(0) ||
        addOp.getRhs() != block.getArgument(1)) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "addf not using block args");
    }

    Value mmt4dResult;
    Value otherInput;
    if (auto mmt4d = genericOp.getInputs()[0].getDefiningOp<MMT4DOp>()) {
      mmt4dResult = genericOp.getInputs()[0];
      otherInput = genericOp.getInputs()[1];
    } else if (auto mmt4d = genericOp.getInputs()[1].getDefiningOp<MMT4DOp>()) {
      mmt4dResult = genericOp.getInputs()[1];
      otherInput = genericOp.getInputs()[0];
    } else {
      return rewriter.notifyMatchFailure(genericOp, "no xsmt.mmt4d input");
    }

    auto mmt4dOp = cast<MMT4DOp>(mmt4dResult.getDefiningOp());
    if (mmt4dOp.getNumOperands() != 2) {
      return rewriter.notifyMatchFailure(genericOp, "mmt4d has accumulator");
    }

    Value accTensor = genericOp.getOutputs()[0];
    if (accTensor != otherInput) {
      return rewriter.notifyMatchFailure(
          genericOp, "output not matching accumulation target");
    }

    auto resType = dyn_cast<RankedTensorType>(mmt4dResult.getType());
    if (!resType || resType.getRank() != 4 || !resType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "non-static 4D result type");
    }
    auto shape = resType.getShape();

    Location loc = genericOp.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(genericOp);

    Value a = mmt4dOp.getA();
    Value b = mmt4dOp.getB();

    if (auto cast = a.getDefiningOp<tensor::CastOp>())
      a = cast.getSource();
    if (auto cast = b.getDefiningOp<tensor::CastOp>())
      b = cast.getSource();

    SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes;

    Value dim0 = tensor::DimOp::create(rewriter, loc, a, 0);
    Value dim1 = tensor::DimOp::create(rewriter, loc, b, 0);
    sizes.push_back(dim0);
    sizes.push_back(dim1);
    sizes.push_back(rewriter.getIndexAttr(shape[2]));
    sizes.push_back(rewriter.getIndexAttr(shape[3]));

    SmallVector<int64_t> sliceShape = {
        ShapedType::kDynamic, ShapedType::kDynamic, shape[2], shape[3]};
    auto sliceType =
        RankedTensorType::get(sliceShape, resType.getElementType());

    auto extractOp = tensor::ExtractSliceOp::create(
        rewriter, loc, sliceType, accTensor, offsets, sizes, strides);

    auto mmt4dResType =
        RankedTensorType::get(sliceShape, resType.getElementType());
    auto newMmt4dOp =
        linalg::Mmt4DOp::create(rewriter, loc, mmt4dResType, ValueRange{a, b},
                                ValueRange{extractOp.getResult()});

    auto insertOp =
        tensor::InsertSliceOp::create(rewriter, loc, newMmt4dOp.getResult(0),
                                      accTensor, offsets, sizes, strides);

    rewriter.replaceOp(genericOp, insertOp.getResult());
    rewriter.eraseOp(mmt4dOp);

    return success();
  }
};

class ForToForallPattern : public OpRewritePattern<scf::ForOp> {
public:
  ForToForallPattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto bindSubBlockAttr = forOp->getAttrOfType<BoolAttr>("bind_sub_block");
    if (!bindSubBlockAttr || !bindSubBlockAttr.getValue()) {
      return failure();
    }

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value step = forOp.getStep();
    Value indVar = forOp.getInductionVar();
    auto convertToIndex = [&](Value val) -> Value {
      if (val.getType().isIndex())
        return val;
      return arith::IndexCastOp::create(rewriter, forOp.getLoc(),
                                        rewriter.getIndexType(), val);
    };

    SmallVector<OpFoldResult> lbs = {convertToIndex(lb)};
    SmallVector<OpFoldResult> ubs = {convertToIndex(ub)};
    SmallVector<OpFoldResult> steps = {convertToIndex(step)};
    SmallVector<Value> outputs;
    auto initArgs = forOp.getInitArgs();
    for (auto initArg : initArgs) {
      outputs.push_back(initArg);
    }

    auto forallOp = scf::ForallOp::create(rewriter, forOp.getLoc(), lbs, ubs,
                                          steps, outputs, std::nullopt);

    Block &forallBlock = forallOp.getRegion().front();
    Block &forBody = *forOp.getBody();
    auto inParallelOp = cast<scf::InParallelOp>(forallBlock.getTerminator());
    rewriter.setInsertionPoint(inParallelOp);
    IRMapping mapping;

    Value forallIndVar = forallBlock.getArgument(0);
    auto castIndVar = arith::IndexCastOp::create(
        rewriter, forOp.getLoc(), indVar.getType(), forallIndVar);
    mapping.map(indVar, castIndVar);

    unsigned numInitArgs = initArgs.size();
    for (unsigned i = 0; i < numInitArgs; ++i) {
      mapping.map(forBody.getArgument(i + 1), forallBlock.getArgument(i + 1));
    }

    for (auto &op : forBody.without_terminator()) {
      rewriter.clone(op, mapping);
    }
    rewriter.replaceOp(forOp, forallOp.getResults());
    return success();
  }
};

class ReplaceRedundantMaterializePattern1
    : public OpRewritePattern<bufferization::MaterializeInDestinationOp> {
public:
  using OpRewritePattern<
      bufferization::MaterializeInDestinationOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(bufferization::MaterializeInDestinationOp materializeOp,
                  PatternRewriter &rewriter) const override {

    auto toTensorOp =
        materializeOp.getSource().getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp)
      return failure();

    auto destAlloc = materializeOp.getDest().getDefiningOp<memref::AllocOp>();
    if (!destAlloc)
      return failure();

    auto srcAlloc = toTensorOp->getOperand(0).getDefiningOp<memref::AllocOp>();
    if (!srcAlloc)
      return failure();

    if (srcAlloc.getType() != destAlloc.getType()) {
      return failure();
    }

    Value srcMemref = srcAlloc.getResult();
    Value destMemref = destAlloc.getResult();

    bool hasOtherUses = false;
    for (OpOperand &use : llvm::make_early_inc_range(destMemref.getUses())) {
      if (use.getOwner() == materializeOp)
        continue;

      rewriter.modifyOpInPlace(use.getOwner(), [&]() { use.set(srcMemref); });
      hasOtherUses = true;
    }

    rewriter.eraseOp(materializeOp);

    if (!hasOtherUses && destAlloc->use_empty()) {
      rewriter.eraseOp(destAlloc);
    }

    return success();
  }
};

/// Rewrite the common tail pattern:
///   %t0 = tensor.empty() : tensor<128x128xf16>
///   %u  = linalg.unpack %packed ... into %t0 : ... -> tensor<128x128xf16>
///   %s  = tensor.extract_slice %u[0,0] [%m,%n] [1,1] : tensor<128x128xf16> to
///   tensor<?x?xf16> %sv = memref.subview %out[...] [%m,%n] ...
///   bufferization.materialize_in_destination %s in writable %sv
/// into an in-place form (as in mm_kernel_linalg1.mlir):
///   %sv' = (optional) memref.cast %sv : memref<?x?xf16, strided<[?,?],...>> to
///   memref<?x?xf16, strided<[?,1],...>> %dst = bufferization.to_tensor %sv'
///   restrict writable : ... -> tensor<?x?xf16> %u2  = linalg.unpack %packed
///   ... into %dst : ... -> tensor<?x?xf16>
///   bufferization.materialize_in_destination %u2 in writable %sv'
///
/// This avoids the intermediate 128x128 tensor and the extract_slice copy for
/// tail blocks.
class MaterializeUnpackIntoSubviewPattern
    : public OpRewritePattern<bufferization::MaterializeInDestinationOp> {
public:
  MaterializeUnpackIntoSubviewPattern(MLIRContext *context)
      : OpRewritePattern<bufferization::MaterializeInDestinationOp>(
            context,
            /*benefit=*/2) {}

private:
  static bool isZeroIndex(Value value) {
    APInt intValue;
    return matchPattern(value, m_ConstantInt(&intValue)) && intValue.isZero();
  }

  static bool isIdentitySubview(memref::SubViewOp subview) {
    if (!hasZeroOffsets(subview.getOffsets()))
      return false;
    if (!hasUnitStrides(subview.getStrides()))
      return false;

    auto srcTy = dyn_cast<MemRefType>(subview.getSource().getType());
    auto dstTy = dyn_cast<MemRefType>(subview.getType());
    if (!srcTy || !dstTy || srcTy.getRank() != dstTy.getRank())
      return false;

    for (auto [srcDim, dstDim] :
         llvm::zip(srcTy.getShape(), dstTy.getShape())) {
      if (srcDim != ShapedType::kDynamic && dstDim != ShapedType::kDynamic &&
          srcDim != dstDim)
        return false;
    }
    return true;
  }

  static bool comesFromDimOf(Value value, Value source) {
    auto dimOp = value.getDefiningOp<tensor::DimOp>();
    return dimOp && dimOp.getSource() == source;
  }

  static bool isBoundedByUnpackDim(Value size, Value unpackResult) {
    if (comesFromDimOf(size, unpackResult))
      return true;

    auto minOp = size.getDefiningOp<arith::MinUIOp>();
    if (!minOp)
      return false;

    return comesFromDimOf(minOp.getLhs(), unpackResult) ||
           comesFromDimOf(minOp.getRhs(), unpackResult);
  }

  static bool isStaticSliceWithinUnpackDim(OpFoldResult size,
                                           int64_t unpackDim) {
    auto attr =
        llvm::dyn_cast_if_present<IntegerAttr>(size.dyn_cast<Attribute>());
    return attr && unpackDim != ShapedType::kDynamic &&
           attr.getInt() <= unpackDim;
  }

  static bool isSliceSizeCompatible(OpFoldResult size, Value unpackResult,
                                    int64_t unpackDim) {
    if (isStaticSliceWithinUnpackDim(size, unpackDim))
      return true;

    Value sizeVal = size.dyn_cast<Value>();
    return sizeVal && isBoundedByUnpackDim(sizeVal, unpackResult);
  }

  static bool hasUnitStrides(ValueRange strides) {
    return llvm::all_of(
        strides, [](Value stride) { return isConstantIntValue(stride, 1); });
  }

  static bool hasZeroOffsets(ValueRange offsets) {
    return llvm::all_of(offsets, isZeroIndex);
  }

  LogicalResult matchAndRewrite(bufferization::MaterializeInDestinationOp mat,
                                PatternRewriter &rewriter) const override {
    auto fail = [&](StringRef reason) -> LogicalResult {
      (void)reason;
      return failure();
    };

    if (!mat.getWritable())
      return failure();

    // Match the actual post-bufferization shape:
    // materialize_in_destination(extract_slice(cast(extract_slice(unpack(...)))))
    auto outerSlice = mat.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!outerSlice)
      return fail("source is not outer extract_slice");

    if (outerSlice.getMixedOffsets().size() != 2 ||
        outerSlice.getMixedSizes().size() != 2 ||
        outerSlice.getMixedStrides().size() != 2)
      return fail("expected rank-2 outer slice");

    if (!hasZeroOffsets(outerSlice.getOffsets()))
      return fail("expected zero outer slice offsets");

    if (!hasUnitStrides(outerSlice.getStrides()))
      return fail("expected unit outer slice strides");

    auto cast = outerSlice.getSource().getDefiningOp<tensor::CastOp>();
    if (!cast)
      return fail("outer slice source is not tensor.cast");

    auto innerSlice = cast.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!innerSlice)
      return fail("cast source is not inner extract_slice");

    auto unpack = innerSlice.getSource().getDefiningOp<linalg::UnPackOp>();
    if (!unpack)
      return fail("inner slice source is not linalg.unpack");

    if (innerSlice.getMixedOffsets().size() != 2 ||
        innerSlice.getMixedSizes().size() != 2 ||
        innerSlice.getMixedStrides().size() != 2)
      return fail("expected rank-2 inner slice");

    if (!hasZeroOffsets(innerSlice.getOffsets()))
      return fail("expected zero inner slice offsets");

    if (!hasUnitStrides(innerSlice.getStrides()))
      return fail("expected unit inner slice strides");

    auto unpackTy = dyn_cast<RankedTensorType>(unpack.getResult().getType());
    if (!unpackTy || unpackTy.getRank() != 2)
      return fail("expected rank-2 unpack result");

    Value unpackResult = unpack.getResult();
    ArrayRef<int64_t> unpackShape = unpackTy.getShape();
    if (!llvm::all_of(llvm::zip(innerSlice.getMixedSizes(), unpackShape),
                      [&](auto it) {
                        return isSliceSizeCompatible(
                            std::get<0>(it), unpackResult, std::get<1>(it));
                      }))
      return fail("inner slice sizes must be derived from unpack dimensions");

    // Destination subview memref.
    Value dstMemref = mat.getDest();
    if (auto subview = dstMemref.getDefiningOp<memref::SubViewOp>()) {
      if (isIdentitySubview(subview)) {
        dstMemref = subview.getSource();
      }
    }
    auto dstTy = dyn_cast<MemRefType>(dstMemref.getType());
    if (!dstTy || dstTy.getRank() != 2)
      return fail("destination is not rank-2 memref");

    if (outerSlice.getMixedSizes().size() !=
        static_cast<size_t>(dstTy.getRank()))
      return fail("outer slice rank does not match destination rank");

    Location loc = mat.getLoc();
    rewriter.setInsertionPoint(mat);

    // Keep destination memref type unchanged. We don't try to force a
    // particular strided layout here; downstream can reason about the actual
    // strides if needed.
    Value castedDst = dstMemref;

    // Create writable tensor view of the destination.
    auto dstTensorTy =
        RankedTensorType::get(dstTy.getShape(), dstTy.getElementType());
    Value dstTensor =
        bufferization::ToTensorOp::create(rewriter, loc, dstTensorTy, castedDst,
                                          /*restrict=*/rewriter.getUnitAttr(),
                                          /*writable=*/rewriter.getUnitAttr());

    // Recreate unpack with destination init = dstTensor.
    auto newUnpack = linalg::UnPackOp::create(
        rewriter, loc, unpack.getSource(), dstTensor, unpack.getInnerDimsPos(),
        unpack.getMixedTiles(), unpack.getOuterDimsPerm());

    // Replace materialize to use the unpack result directly.
    auto newMat = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, newUnpack.getResult(), castedDst);
    newMat.setWritable(true);

    rewriter.eraseOp(mat);
    // Old slice/unpack/empty will be DCE'd if no longer used.
    return success();
  }
};
struct ReplaceRedundantMaterializePattern2
    : public OpRewritePattern<bufferization::MaterializeInDestinationOp> {
  using OpRewritePattern<
      bufferization::MaterializeInDestinationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::MaterializeInDestinationOp matOp,
                                PatternRewriter &rewriter) const override {
    auto allocOp = matOp.getDest().getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();

    auto loop = dyn_cast<scf::ForOp>(matOp->getParentOp());
    if (!loop)
      return failure();

    Value sourceTensor = matOp.getSource();
    if (!sourceTensor.getDefiningOp() ||
        loop->isAncestor(sourceTensor.getDefiningOp()))
      return failure();

    MemRefType memrefType = allocOp.getType();
    if (memrefType.getRank() != 4 || !memrefType.hasStaticShape())
      return failure();

    rewriter.setInsertionPoint(loop);
    auto hoistedAlloc =
        memref::AllocOp::create(rewriter, loop.getLoc(), memrefType);
    auto hoistMat = bufferization::MaterializeInDestinationOp::create(
        rewriter, loop.getLoc(), sourceTensor, hoistedAlloc.getResult());
    hoistMat->setAttr("writable", rewriter.getUnitAttr());

    Value hoistedBuffer = hoistedAlloc.getResult();

    matOp.getDest().replaceAllUsesWith(hoistedBuffer);

    rewriter.eraseOp(matOp);
    rewriter.eraseOp(allocOp);

    SmallVector<Operation *> deadOps;
    for (auto &use : llvm::make_early_inc_range(sourceTensor.getUses())) {
      auto redundant =
          dyn_cast<bufferization::MaterializeInDestinationOp>(use.getOwner());
      if (!redundant)
        continue;
      if (redundant.getSource() != sourceTensor)
        continue;
      if (loop->isAncestor(redundant))
        continue;

      Value deadMemref = redundant.getDest();
      if (auto deadAlloc = deadMemref.getDefiningOp<memref::AllocOp>()) {
        for (auto *user : llvm::make_early_inc_range(deadMemref.getUsers())) {
          if (user == redundant)
            continue;
          rewriter.modifyOpInPlace(user, [&] {
            for (unsigned i = 0; i < user->getNumOperands(); ++i) {
              if (user->getOperand(i) == deadMemref) {
                user->setOperand(i, hoistedBuffer);
              }
            }
          });
        }
      }
    }
    return success();
  }
};

struct RemoveUnusedAllocPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value memrefResult = allocOp.getMemref();

    SmallVector<Operation *> opsToErase;

    for (Operation *user : memrefResult.getUsers()) {

      if (auto matOp = dyn_cast<MaterializeInDestinationOp>(user)) {
        if (matOp.getDest() == memrefResult) {
          opsToErase.push_back(user);
          continue;
        }
      }

      if (isa<memref::DeallocOp>(user)) {
        opsToErase.push_back(user);
        continue;
      }

      return failure();
    }
    for (Operation *op : opsToErase) {
      rewriter.eraseOp(op);
    }
    rewriter.eraseOp(allocOp);

    return success();
  }
};

struct RemoveRedundantBufferizationRoundtrip
    : public OpRewritePattern<ToTensorOp> {
  using OpRewritePattern<ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToTensorOp toTensorOp,
                                PatternRewriter &rewriter) const override {
    Value memref = toTensorOp->getOperand(0);

    auto allocOp = memref.getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
      return failure();
    }

    MaterializeInDestinationOp matOp;
    Operation *deallocOp = nullptr;

    for (Operation *user : memref.getUsers()) {
      if (user == toTensorOp)
        continue;

      if (auto m = dyn_cast<MaterializeInDestinationOp>(user)) {
        if (m.getDest() == memref) {
          if (matOp) {
            return failure();
          }
          matOp = m;
          continue;
        }
      }

      if (isa<memref::DeallocOp>(user)) {
        deallocOp = user;
        continue;
      }
      return failure();
    }

    if (!matOp) {
      return failure();
    }

    Value sourceTensor = matOp.getSource();

    if (sourceTensor.getType() != toTensorOp.getResult().getType()) {
      return failure();
    }

    rewriter.replaceOp(toTensorOp, sourceTensor);
    if (deallocOp) {
      rewriter.eraseOp(deallocOp);
    }
    rewriter.eraseOp(matOp);
    rewriter.eraseOp(allocOp);

    return success();
  }
};

struct GetThreadOpToLLVMCallPattern
    : public OpRewritePattern<xsmt::GetThreadOp> {
  using OpRewritePattern<xsmt::GetThreadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmt::GetThreadOp getThreadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = getThreadOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = getThreadOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return failure();
    }

    StringRef funcName = "spine_get_stream_threads";
    Type resultType = rewriter.getI32Type();

    auto llvmFuncOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!llvmFuncOp) {
      auto funcType = LLVM::LLVMFunctionType::get(resultType, /*params=*/{},
                                                  /*isVarArg=*/false);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      llvmFuncOp = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcType);
      llvmFuncOp.setLinkage(LLVM::Linkage::External);
    }

    auto callOp =
        LLVM::CallOp::create(rewriter, loc, resultType, funcName, ValueRange{});

    rewriter.replaceOp(getThreadOp, callOp.getResults());

    return success();
  }
};

// Pattern to convert proton::RecordOp to RISC-V rdtime inline assembly +
// runtime call Note: rdcycle is disabled in user mode on many RISC-V systems,
// so we use rdtime instead
struct ProtonRecordOpPattern : public OpRewritePattern<proton::RecordOp> {
  using OpRewritePattern<proton::RecordOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(proton::RecordOp recordOp,
                                PatternRewriter &rewriter) const override {
    Location loc = recordOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = recordOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return failure();
    }

    // 1. Create rdtime inline assembly to read time counter
    // Note: rdcycle is often disabled in user mode, rdtime is more portable
    Type i64Type = rewriter.getI64Type();

    // RISC-V rdtime instruction: reads the time CSR (available in user mode)
    auto asmDialectAttr =
        LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);

    auto inlineAsmOp = LLVM::InlineAsmOp::create(
        rewriter, loc,
        /*resultTypes=*/i64Type,
        /*operands=*/ValueRange{},
        /*asm_string=*/"rdtime $0",
        /*constraints=*/"=r",
        /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        /*tail_call_kind=*/LLVM::tailcallkind::TailCallKind::None,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr{});

    Value cycleValue = inlineAsmOp.getResult(0);

    // 2. Declare and call runtime function: proton_record(const char* name,
    // int64_t cycle, int is_start)
    StringRef funcName = "proton_record";

    // Get or create the runtime function declaration
    auto llvmFuncOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!llvmFuncOp) {
      // Function signature: void proton_record(const char* name, int64_t cycle,
      // int32_t is_start)
      Type voidType = LLVM::LLVMVoidType::get(ctx);
      Type ptrType = LLVM::LLVMPointerType::get(ctx);
      Type i32Type = rewriter.getI32Type();

      auto funcType = LLVM::LLVMFunctionType::get(
          voidType, {ptrType, i64Type, i32Type}, /*isVarArg=*/false);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      llvmFuncOp = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcType);
      llvmFuncOp.setLinkage(LLVM::Linkage::External);
    }

    // 3. Create global string constant for the scope name
    std::string scopeName = recordOp.getName().str();
    std::string globalName = "proton_scope_" + scopeName;

    // Check if global already exists
    auto globalOp = moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (!globalOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      // Create null-terminated string
      std::string strWithNull = scopeName + '\0';
      auto strAttr = rewriter.getStringAttr(strWithNull);
      auto arrayType = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8),
                                                strWithNull.size());

      globalOp = LLVM::GlobalOp::create(
          rewriter, loc, arrayType,
          /*isConstant=*/true, LLVM::Linkage::Internal, globalName, strAttr);
    }

    // 4. Get pointer to the global string
    Type ptrType = LLVM::LLVMPointerType::get(ctx);
    auto addrOp = LLVM::AddressOfOp::create(rewriter, loc, ptrType, globalName);

    // 5. Create is_start constant (1 for start, 0 for end)
    int32_t isStartVal = recordOp.getIsStart() ? 1 : 0;
    auto isStartConst =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(isStartVal));

    // 6. Call the runtime function
    LLVM::CallOp::create(
        rewriter, loc, TypeRange{}, funcName,
        ValueRange{addrOp.getResult(), cycleValue, isStartConst.getResult()});

    // 7. Erase the original op
    rewriter.eraseOp(recordOp);

    return success();
  }
};

struct InsertMBarrierReleasePattern
    : public mlir::OpRewritePattern<xsmt_async::MBarrierAllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(xsmt_async::MBarrierAllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value root = op.getResult();

    if (hasExistingRelease(root))
      return mlir::failure();

    mlir::Block *allocBlock = op->getBlock();

    llvm::SmallPtrSet<mlir::Value, 16> aliasSet;
    collectAliasClosure(root, aliasSet);

    mlir::Operation *lastUser = nullptr;
    bool escaped = false;
    findLastUser(aliasSet, allocBlock, op.getOperation(), lastUser, escaped);

    if (escaped || !lastUser)
      return mlir::failure();

    insertRelease(rewriter, op.getLoc(), root, lastUser);

    return mlir::success();
  }

private:
  bool hasExistingRelease(mlir::Value barrier) const {
    for (mlir::Operation *user : barrier.getUsers()) {
      if (mlir::isa<xsmt_async::MBarrierReleaseOp>(user))
        return true;
    }
    return false;
  }

  bool
  isBarrierForwardingOp(mlir::Operation *user, mlir::Value v,
                        llvm::SmallVectorImpl<mlir::Value> &newAliases) const {
    if (auto fromElements = dyn_cast<tensor::FromElementsOp>(user)) {
      for (Value operand : fromElements.getOperands()) {
        if (operand == v) {
          newAliases.push_back(fromElements.getResult());
          return true;
        }
      }
    }

    if (auto extract = dyn_cast<tensor::ExtractOp>(user)) {
      if (extract.getTensor() == v) {
        newAliases.push_back(extract.getResult());
        return true;
      }
    }

    if (auto cast = dyn_cast<tensor::CastOp>(user)) {
      if (cast.getSource() == v) {
        newAliases.push_back(cast.getResult());
        return true;
      }
    }

    if (auto ucc = dyn_cast<UnrealizedConversionCastOp>(user)) {
      for (Value input : ucc.getInputs()) {
        if (input == v) {
          for (Value output : ucc.getOutputs())
            newAliases.push_back(output);
          return true;
        }
      }
    }

    if (auto select = dyn_cast<arith::SelectOp>(user)) {
      if (select.getTrueValue() == v || select.getFalseValue() == v) {
        newAliases.push_back(select.getResult());
        return true;
      }
    }

    if (auto insert = dyn_cast<tensor::InsertOp>(user)) {
      if (insert.getScalar() == v || insert.getDest() == v) {
        newAliases.push_back(insert.getResult());
        return true;
      }
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(user)) {
      for (auto &region : ifOp->getRegions()) {
        if (region.empty())
          continue;
        auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
        if (!yield)
          continue;
        for (auto [idx, operand] : llvm::enumerate(yield.getOperands())) {
          if (operand == v && idx < ifOp.getNumResults()) {
            newAliases.push_back(ifOp.getResult(idx));
          }
        }
      }
      if (!newAliases.empty())
        return true;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
        if (initArg == v) {
          newAliases.push_back(forOp.getRegionIterArg(idx));
          newAliases.push_back(forOp.getResult(idx));
        }
      }
      if (!newAliases.empty())
        return true;
    }

    return false;
  }

  void collectAliasClosure(mlir::Value root,
                           llvm::SmallPtrSetImpl<mlir::Value> &aliasSet) const {
    llvm::SmallVector<mlir::Value, 16> worklist;

    aliasSet.insert(root);
    worklist.push_back(root);

    while (!worklist.empty()) {
      mlir::Value current = worklist.pop_back_val();

      for (mlir::Operation *user : current.getUsers()) {
        llvm::SmallVector<mlir::Value, 4> newAliases;
        isBarrierForwardingOp(user, current, newAliases);

        for (mlir::Value alias : newAliases) {
          if (aliasSet.insert(alias).second) {
            worklist.push_back(alias);
          }
        }
      }
    }
  }

  void findLastUser(const llvm::SmallPtrSetImpl<mlir::Value> &aliasSet,
                    mlir::Block *allocBlock, mlir::Operation *allocOp,
                    mlir::Operation *&lastUser, bool &escaped) const {
    lastUser = allocOp;
    escaped = false;

    for (mlir::Value v : aliasSet) {
      for (mlir::Operation *user : v.getUsers()) {
        if (mlir::isa<xsmt_async::MBarrierReleaseOp>(user))
          continue;
        mlir::Operation *ancestor = allocBlock->findAncestorOpInBlock(*user);

        if (!ancestor) {
          escaped = true;
          return;
        }
        if (lastUser->isBeforeInBlock(ancestor)) {
          lastUser = ancestor;
        }
      }
    }
  }

  void insertRelease(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Value barrier, mlir::Operation *lastUser) const {
    if (lastUser->hasTrait<mlir::OpTrait::IsTerminator>()) {
      rewriter.setInsertionPoint(lastUser);
    } else {
      rewriter.setInsertionPointAfter(lastUser);
    }

    xsmt_async::MBarrierReleaseOp::create(rewriter, loc, barrier);
  }
};

struct DescriptorLoadViewOpPattern
    : public OpRewritePattern<DescriptorLoadViewOp> {
  using OpRewritePattern<DescriptorLoadViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorLoadViewOp op,
                                PatternRewriter &rewriter) const override {
    Value result = op.getResult();
    bool hasTranspose = false;
    linalg::TransposeOp transposeOp;

    for (OpOperand &use : result.getUses()) {
      if (auto t = dyn_cast<linalg::TransposeOp>(use.getOwner())) {
        if (hasTranspose)
          return failure();
        hasTranspose = true;
        transposeOp = t;
      }
    }

    if (hasTranspose) {
      if (!result.hasOneUse())
        return failure();
      if (transposeOp.getInput() != result)
        return failure();
    }
    Value sourcePtr = op.getBase();
    auto unrealizedCast = sourcePtr.getDefiningOp<UnrealizedConversionCastOp>();
    if (!unrealizedCast)
      return failure();
    Value rankedMemRef = unrealizedCast.getOperand(0);

    // Trace through memref.cast to get the original reinterpret_cast if present
    // This allows us to work with the actual dynamic sizes from
    // reinterpret_cast instead of the static sizes from memref.cast
    if (auto castOp = rankedMemRef.getDefiningOp<memref::CastOp>()) {
      rankedMemRef = castOp.getSource();
    }

    auto memrefTy = dyn_cast<MemRefType>(rankedMemRef.getType());
    if (!memrefTy)
      return failure();

    if (memrefTy.getRank() == 4) {
      if (hasTranspose)
        return failure();
      return convertRank4ToUnpack(op, rankedMemRef, rewriter);
    }

    if (memrefTy.getRank() != 2)
      return failure();

    Location loc = op.getLoc();

    Value dim0Value, dim1Value;
    if (auto rc = rankedMemRef.getDefiningOp<memref::ReinterpretCastOp>()) {
      // Get actual sizes from reinterpret_cast
      auto sizes = rc.getMixedSizes();
      if (sizes.size() != 2)
        return failure();
      dim0Value = ofrToIndexValue(loc, sizes[0], rewriter);
      dim1Value = ofrToIndexValue(loc, sizes[1], rewriter);
    } else {
      dim0Value =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/0, rewriter);
      dim1Value =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/1, rewriter);
    }

    auto microSizeAttr = op.getMicroSize();
    if (microSizeAttr.size() != 2)
      return failure();
    int32_t tile0 = microSizeAttr[0];
    int32_t tile1 = microSizeAttr[1];

    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto offsets = op.getOffsets();
    Value offset0 = offsets[0];
    Value offset1 = offsets[1];

    Value indexOffset0 = ensureIndexType(loc, offset0, rewriter);
    Value indexOffset1 = ensureIndexType(loc, offset1, rewriter);
    Value indexDim0Value = ensureIndexType(loc, dim0Value, rewriter);
    Value indexDim1Value = ensureIndexType(loc, dim1Value, rewriter);

    auto shapes = op.getShape();
    Value shape0 = arith::ConstantIndexOp::create(rewriter, loc, shapes[0]);
    Value shape1 = arith::ConstantIndexOp::create(rewriter, loc, shapes[1]);

    Value size_m1 =
        arith::SubIOp::create(rewriter, loc, indexDim0Value, indexOffset0);
    Value size_m2 = arith::MinSIOp::create(rewriter, loc, size_m1, shape0);

    Value size_k1 =
        arith::SubIOp::create(rewriter, loc, indexDim1Value, indexOffset1);
    Value size_k2 = arith::MinSIOp::create(rewriter, loc, size_k1, shape1);

    SmallVector<OpFoldResult> subviewOffsets{indexOffset0, indexOffset1};
    SmallVector<OpFoldResult> subviewSizes{size_m2, size_k2};
    SmallVector<OpFoldResult> subviewStrides{c1, c1};

    Value toTensorSource = rankedMemRef;
    auto rankedMemRefType = cast<MemRefType>(rankedMemRef.getType());
    auto tensorType = RankedTensorType::get(rankedMemRefType.getShape(),
                                            rankedMemRefType.getElementType());

    bool elideSubview =
        canElideDescriptorSubview(rankedMemRef, indexOffset0, indexOffset1,
                                  size_m2, size_k2) ||
        canElideIdentitySubview(rankedMemRef, subviewSizes, subviewOffsets,
                                subviewStrides);

    if (!elideSubview) {
      Value subview = memref::SubViewOp::create(rewriter, loc, rankedMemRef,
                                                /*offsets=*/subviewOffsets,
                                                /*sizes=*/subviewSizes,
                                                /*strides=*/subviewStrides);

      auto subviewType = cast<MemRefType>(subview.getType());
      tensorType = RankedTensorType::get(subviewType.getShape(),
                                         subviewType.getElementType());
      toTensorSource = subview;
    }

    Value tensor = bufferization::ToTensorOp::create(
        rewriter, loc, tensorType, toTensorSource,
        /*restrict=*/rewriter.getUnitAttr());

    if (!hasTranspose) {
      auto shapes = op.getShape();
      return convertWithoutTranspose(op, tensor, size_m2, size_k2, shapes[0],
                                     shapes[1], tile0, tile1, rewriter);
    }

    auto permutation = transposeOp.getPermutation();
    if (permutation.size() != 4 || permutation[0] != 1 || permutation[1] != 0 ||
        permutation[2] != 3 || permutation[3] != 2) {
      return failure();
    }

    return convertWithTranspose(op, transposeOp, tensor, size_m2, size_k2,
                                shapes[0], shapes[1], tile0, tile1, rewriter);
  }

private:
  LogicalResult convertRank4ToUnpack(DescriptorLoadViewOp op,
                                     Value rankedMemRef,
                                     PatternRewriter &rewriter) const {
    Location loc = op.getLoc();

    auto memrefTy = dyn_cast<MemRefType>(rankedMemRef.getType());
    if (!memrefTy || memrefTy.getRank() != 4)
      return failure();

    Value outer0, outer1, tile0Val, tile1Val;
    if (auto rc = rankedMemRef.getDefiningOp<memref::ReinterpretCastOp>()) {
      auto sizes = rc.getMixedSizes();
      if (sizes.size() != 4)
        return failure();
      outer0 = ofrToIndexValue(loc, sizes[0], rewriter);
      outer1 = ofrToIndexValue(loc, sizes[1], rewriter);
      tile0Val = ofrToIndexValue(loc, sizes[2], rewriter);
      tile1Val = ofrToIndexValue(loc, sizes[3], rewriter);
    } else {
      outer0 =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/0, rewriter);
      outer1 =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/1, rewriter);
      tile0Val =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/2, rewriter);
      tile1Val =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/3, rewriter);
    }

    int64_t tile0Static = memrefTy.getDimSize(2);
    int64_t tile1Static = memrefTy.getDimSize(3);
    if (ShapedType::isDynamic(tile0Static) ||
        ShapedType::isDynamic(tile1Static))
      return failure();

    auto tensor4DType =
        RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());
    Value packedTensor = bufferization::ToTensorOp::create(
        rewriter, loc, tensor4DType, rankedMemRef, rewriter.getUnitAttr());

    Value flatRows = arith::MulIOp::create(rewriter, loc, outer0, tile0Val);
    Value flatCols = arith::MulIOp::create(rewriter, loc, outer1, tile1Val);

    auto elementType = memrefTy.getElementType();
    auto flatTy = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
    Value empty = tensor::EmptyOp::create(rewriter, loc, flatTy,
                                          ValueRange{flatRows, flatCols});

    Value flatTensor =
        linalg::UnPackOp::create(
            rewriter, loc, packedTensor, empty, ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile0Static),
                                   rewriter.getIndexAttr(tile1Static)})
            .getResult();

    auto offsets = op.getOffsets();
    if (offsets.size() != 2)
      return failure();

    Value off0 = ensureIndexType(loc, offsets[0], rewriter);
    Value off1 = ensureIndexType(loc, offsets[1], rewriter);

    auto shapes = op.getShape();
    if (shapes.size() != 2)
      return failure();

    Value reqRows = arith::ConstantIndexOp::create(rewriter, loc, shapes[0]);
    Value reqCols = arith::ConstantIndexOp::create(rewriter, loc, shapes[1]);

    Value flatDim0 = tensor::DimOp::create(rewriter, loc, flatTensor, 0);
    Value flatDim1 = tensor::DimOp::create(rewriter, loc, flatTensor, 1);

    Value availRows = arith::SubIOp::create(rewriter, loc, flatDim0, off0);
    Value availCols = arith::SubIOp::create(rewriter, loc, flatDim1, off1);
    Value actualRows =
        arith::MinSIOp::create(rewriter, loc, reqRows, availRows);
    Value actualCols =
        arith::MinSIOp::create(rewriter, loc, reqCols, availCols);

    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value slice = tensor::ExtractSliceOp::create(
        rewriter, loc, flatTensor, ArrayRef<Value>{off0, off1},
        ArrayRef<Value>{actualRows, actualCols}, ArrayRef<Value>{c1, c1});

    auto originalResultType = cast<RankedTensorType>(op.getResult().getType());
    Value castResult =
        tensor::CastOp::create(rewriter, loc, originalResultType, slice);

    rewriter.replaceOp(op, castResult);
    cleanupUnusedOps(op, rewriter);
    return success();
  }

  Value getDimValueFromTypeOrDimOp(Location loc, Value memref, int64_t dim,
                                   PatternRewriter &rewriter) const {
    auto ty = cast<MemRefType>(memref.getType());
    if (!ty.isDynamicDim(dim)) {
      return arith::ConstantIndexOp::create(rewriter, loc, ty.getDimSize(dim));
    }
    return memref::DimOp::create(rewriter, loc, memref, dim);
  }

  LogicalResult convertWithoutTranspose(DescriptorLoadViewOp op, Value tensor,
                                        Value dim0, Value dim1, int32_t shape0,
                                        int32_t shape1, int32_t tile0,
                                        int32_t tile1,
                                        PatternRewriter &rewriter) const {
    Location loc = op.getLoc();

    // Use expected shape (not actual data size) to calculate packed dimensions
    // This ensures the output has the expected static shape, with padding for
    // boundary cases
    int64_t packedRowsStatic = (shape0 + tile0 - 1) / tile0;
    int64_t packedColsStatic = (shape1 + tile1 - 1) / tile1;

    SmallVector<int64_t, 4> shapeDims = {packedRowsStatic, packedColsStatic,
                                         tile0, tile1};

    auto tensorType = cast<RankedTensorType>(tensor.getType());
    Type elementType = tensorType.getElementType();
    auto emptyType = RankedTensorType::get(shapeDims, elementType);

    // Static shape, no dynamic sizes needed
    Value emptyTensor =
        tensor::EmptyOp::create(rewriter, loc, emptyType, ValueRange{});
    Value paddingValue = createZeroConstant(rewriter, loc, elementType);

    Value packedResult =
        linalg::PackOp::create(
            rewriter, loc, tensor, emptyTensor,
            /*innerDimsPos=*/ArrayRef<int64_t>{0, 1},
            /*innerTiles=*/
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile0),
                                   rewriter.getIndexAttr(tile1)},
            /*paddingValue=*/paddingValue,
            /*outerDimsPerm=*/ArrayRef<int64_t>{0, 1})
            .getResult();

    auto originalResultType = cast<RankedTensorType>(op.getResult().getType());
    Value castResult =
        tensor::CastOp::create(rewriter, loc, originalResultType, packedResult);

    rewriter.replaceOp(op, castResult);
    cleanupUnusedOps(op, rewriter);
    return success();
  }

  LogicalResult convertWithTranspose(DescriptorLoadViewOp op,
                                     linalg::TransposeOp transposeOp,
                                     Value tensor, Value dim0, Value dim1,
                                     int32_t shape0, int32_t shape1,
                                     int32_t tile0, int32_t tile1,
                                     PatternRewriter &rewriter) const {
    Location loc = op.getLoc();

    // Use expected shape (not actual data size) to calculate packed dimensions
    // This ensures static output shapes for downstream operations
    // For transpose case: output shape is [ceil(shape1/tile1),
    // ceil(shape0/tile0), tile1, tile0]
    int64_t packedRowsStatic = (shape1 + tile1 - 1) / tile1;
    int64_t packedColsStatic = (shape0 + tile0 - 1) / tile0;

    auto tensorType = cast<RankedTensorType>(tensor.getType());
    Type elementType = tensorType.getElementType();

    SmallVector<int64_t, 4> shapeDims = {packedRowsStatic, packedColsStatic,
                                         tile1, tile0};

    auto emptyType = RankedTensorType::get(shapeDims, elementType);

    // Static shape, no dynamic sizes needed
    Value emptyTensor =
        tensor::EmptyOp::create(rewriter, loc, emptyType, ValueRange{});
    Value paddingValue = createZeroConstant(rewriter, loc, elementType);

    Value packedResult =
        linalg::PackOp::create(
            rewriter, loc, tensor, emptyTensor,
            /*innerDimsPos=*/ArrayRef<int64_t>{1, 0},
            /*innerTiles=*/
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile1),
                                   rewriter.getIndexAttr(tile0)},
            /*paddingValue=*/paddingValue,
            /*outerDimsPerm=*/ArrayRef<int64_t>{1, 0})
            .getResult();

    auto transposeResultType =
        cast<RankedTensorType>(transposeOp->getResult(0).getType());
    Value castResult = tensor::CastOp::create(
        rewriter, loc, transposeResultType, packedResult);

    rewriter.replaceOp(transposeOp, castResult);
    rewriter.eraseOp(op);
    cleanupUnusedOps(op, rewriter);
    return success();
  }

  void cleanupUnusedOps(DescriptorLoadViewOp op,
                        PatternRewriter &rewriter) const {
    Value sourcePtr = op.getBase();
    auto unrealized = sourcePtr.getDefiningOp<UnrealizedConversionCastOp>();
    if (!unrealized)
      return;

    Value v = unrealized.getOperand(0);
    memref::CastOp castOp = v.getDefiningOp<memref::CastOp>();
    if (castOp)
      v = castOp.getSource();
    memref::ReinterpretCastOp rcOp =
        v.getDefiningOp<memref::ReinterpretCastOp>();

    if (unrealized->use_empty())
      rewriter.eraseOp(unrealized);

    if (castOp && castOp->use_empty())
      rewriter.eraseOp(castOp);

    if (rcOp && rcOp->use_empty())
      rewriter.eraseOp(rcOp);
  }
};

struct FoldAllocCopyToTensor final
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToTensorOp toTensor,
                                PatternRewriter &rewriter) const override {

    Value allocMemref = toTensor->getOperand(0);

    auto alloc = allocMemref.getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    memref::CopyOp copyOp;
    memref::DeallocOp deallocOp;

    for (Operation *user : allocMemref.getUsers()) {
      if (user == toTensor.getOperation())
        continue;

      if (auto c = dyn_cast<memref::CopyOp>(user)) {
        if (c.getTarget() != allocMemref)
          return failure();
        if (copyOp)
          return failure();
        copyOp = c;
        continue;
      }

      if (auto d = dyn_cast<memref::DeallocOp>(user)) {
        if (deallocOp)
          return failure();
        deallocOp = d;
        continue;
      }

      return failure();
    }

    if (!copyOp)
      return failure();

    if (copyOp->getBlock() == toTensor->getBlock()) {
      if (!copyOp->isBeforeInBlock(toTensor))
        return failure();
    } else {
      DominanceInfo dom(toTensor->getParentOp());
      if (!dom.dominates(copyOp, toTensor))
        return failure();
    }

    Value src = copyOp.getSource();

    auto resTy = dyn_cast<RankedTensorType>(toTensor.getType());
    auto srcTy = dyn_cast<MemRefType>(src.getType());
    if (!resTy || !srcTy)
      return failure();
    if (srcTy.getRank() != resTy.getRank())
      return failure();
    if (srcTy.getElementType() != resTy.getElementType())
      return failure();

    auto newToTensor = bufferization::ToTensorOp::create(
        rewriter, toTensor.getLoc(), resTy, src);

    newToTensor->setAttrs(toTensor->getAttrs());

    rewriter.replaceOp(toTensor, newToTensor.getResult());

    rewriter.eraseOp(copyOp);
    if (deallocOp)
      rewriter.eraseOp(deallocOp);
    rewriter.eraseOp(alloc);

    return success();
  }
};

void mlir::triton::populateXSMTOptimizationAndValidationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MMT4DBindFusionPattern>(patterns.getContext());
  patterns.add<MBarrierLoopCheckPattern<xsmt_async::MBarrierAllocOp>>(
      patterns.getContext());
  patterns.add<MBarrierLoopCheckPattern<xsmt::GlobalMBarrierInitOp>>(
      patterns.getContext());
  patterns.add<CheckGlobalMBarrierInLoopPattern<xsmt_async::MBarrierArriveOp>>(
      patterns.getContext());
  patterns.add<CheckGlobalMBarrierInLoopPattern<xsmt_async::MBarrierWaitOp>>(
      patterns.getContext());
}

void mlir::triton::populateXSMTToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  // patterns.add<FillOpPattern>(patterns.getContext());
  patterns.add<DescriptorLoadViewOpPattern>(patterns.getContext());
  patterns.add<ViewOpPattern>(patterns.getContext());
  patterns.add<InsertMBarrierReleasePattern>(patterns.getContext());
  patterns.add<GetThreadOpToLLVMCallPattern>(patterns.getContext());
  patterns.add<ProtonRecordOpPattern>(patterns.getContext());
}

void mlir::triton::ConvertMMT4DAddConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConvertMMT4DAddPattern>(patterns.getContext());
}

void mlir::triton::MMT4DOpConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerXSMTMMT4D>(patterns.getContext());
}

void mlir::triton::LoopParallelizationConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ForToForallPattern>(patterns.getContext());
  patterns.add<ReplaceRedundantMaterializePattern1>(patterns.getContext());
  patterns.add<ReplaceRedundantMaterializePattern2>(patterns.getContext());
}

void mlir::triton::BufferizationCleanupConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<RemoveUnusedAllocPattern>(patterns.getContext());
  patterns.add<RemoveRedundantBufferizationRoundtrip>(patterns.getContext());
  patterns.add<FoldAllocCopyToTensor>(patterns.getContext());
}

void mlir::triton::populateCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MaterializeUnpackIntoSubviewPattern>(patterns.getContext());
}
