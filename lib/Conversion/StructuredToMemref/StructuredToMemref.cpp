//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Types.h"

#include "include/triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "include/triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

static const std::string WRAP_SIDE_BY_SIDE = "wrap_side_by_side";
static const std::string WRAP_STACKED = "wrap_stacked";

static memref::SubViewOp getSubview(int rank, ArrayRef<OpFoldResult> dims,
                                    Value source, Location loc, OpBuilder &b) {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return memref::SubViewOp::create(b, loc, cast<MemRefType>(dstType), source,
                                   offsets, dims, strides);
}

static Type getElementTypeStructuredPtr(tts::MakeTensorPtrOp op) {
  assert(!op.isBlockPtr());
  // tensor<1024x!tt.ptr<f32>>
  auto ptrType = cast<triton::PointerType>(
      cast<RankedTensorType>(op.getType()).getElementType());
  return ptrType.getPointeeType();
}

static Type getElementTypeBlockPtr(tts::MakeTensorPtrOp op) {
  assert(op.isBlockPtr());
  // !tt.ptr<tensor<128x64xbf16>, 1>
  auto shapedType = cast<ShapedType>(
      cast<triton::PointerType>(op.getType()).getPointeeType());
  return shapedType.getElementType();
}

static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                      ArrayRef<int64_t> staticStrides,
                                      ArrayRef<int64_t> resultShape) {
  auto layout = StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
  Type elemType;
  if (op.isBlockPtr()) {
    elemType = getElementTypeBlockPtr(op);
  } else {
    elemType = getElementTypeStructuredPtr(op);
  }
  return MemRefType::get(resultShape, elemType, layout);
}

static MemRefType getResultMemrefType(tts::MakeGatherScatterTensorPtrOp op,
                                      int64_t offset,
                                      ArrayRef<int64_t> staticStrides,
                                      ArrayRef<int64_t> resultShape) {
  auto layout = StridedLayoutAttr::get(op.getContext(), offset, staticStrides);

  auto ptrType = cast<triton::PointerType>(op.getType());
  Type elemType = ptrType.getPointeeType();

  Type realEltTy = cast<RankedTensorType>(elemType).getElementType();
  return MemRefType::get(resultShape, realEltTy, layout);
}

// If there are dimensions with size 1 and stride 0, replace 0 stride with
// the product of sizes of all lower dimensions. This avoids creating memref
// with zero stride.
template <class OpType>
llvm::SmallVector<OpFoldResult> getMixedStridesForMemref(OpType op,
                                                         OpBuilder &b) {
  llvm::SmallVector<OpFoldResult> strides;
  auto accumulate = 1;
  for (auto [size, stride] :
       llvm::reverse(llvm::zip(op.getSizes(), op.getMixedStrides()))) {
    auto strideIntAttr = getIntAttr(stride);
    if (size == 1 && strideIntAttr && strideIntAttr.value() == 0) {
      strides.push_back(b.getIndexAttr(accumulate));
    } else if (auto v = llvm::dyn_cast_if_present<Value>(stride)) {
      OpFoldResult result = getAsOpFoldResult(v);
      strides.push_back(result);
    } else {
      strides.push_back(stride);
    }
    accumulate *= size;
  }
  std::reverse(strides.begin(), strides.end());
  return strides;
}

static OpFoldResult accumulateTargetOffset(Location loc,
                                           ArrayRef<OpFoldResult> offsets,
                                           OpBuilder &b) {
  OpFoldResult targetOffset = b.getIndexAttr(0);
  for (auto o : offsets) {
    targetOffset = addOFRs(targetOffset, o, loc, b);
  }
  return targetOffset;
}

static OpFoldResult accumulateTargetOffset(Location loc,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> strides,
                                           int gatherDim, OpBuilder &b) {
  OpFoldResult targetOffset = b.getIndexAttr(0);
  for (int i = 0; i < offsets.size(); i++) {

    OpFoldResult offset = offsets[i];
    // If this is the gather dimension, multiply the offset by the stride.
    // Non-gather dimensions are already multiplied by the stride
    // in the offsets in PtrAnalysis.
    if (i == gatherDim) {
      OpFoldResult stride = strides[i];
      offset = mulOFRs(offset, stride, loc, b);
    }
    targetOffset = addOFRs(targetOffset, offset, loc, b);
  }
  return targetOffset;
}

static Value rewriteGatherScatterPtrElement(
    ArrayRef<int64_t> resultShape, tts::MakeGatherScatterTensorPtrOp op,
    Value basePtr, Value gatherOffsetElt, int gatherDim,
    ConversionPatternRewriter &rewriter) {

  auto mixedStrides = getMixedStridesForMemref(op, rewriter);
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

  auto offsets = op.getMixedOffsets();
  offsets[gatherDim] = gatherOffsetElt;
  auto targetOffset = accumulateTargetOffset(op.getLoc(), offsets, mixedStrides,
                                             gatherDim, rewriter);

  auto staticTargetOffset = getIntAttr(targetOffset);
  auto resultType =
      getResultMemrefType(op, staticTargetOffset.value_or(ShapedType::kDynamic),
                          staticStrides, resultShape);

  std::vector<int64_t> staticSizes = op.getSizes();
  staticSizes[gatherDim] = 1;
  SmallVector<Value> dynSizes; // sizes are always static
  auto sizes = mlir::getMixedValues(staticSizes, dynSizes, rewriter);

  auto castOp = memref::ReinterpretCastOp::create(
      rewriter, op.getLoc(), resultType, basePtr, targetOffset, sizes,
      mixedStrides);

  return castOp.getResult();
}

// Fill load destination with other value for mask.
static void fillWithValue(Location loc, Value alloc, Value other,
                          ArrayRef<int64_t> shape,
                          SmallVector<OpFoldResult> &&mixedDims,
                          ArrayRef<int64_t> staticMaskDims,
                          ConversionPatternRewriter &rewriter) {
  // Fill load destination with other value
  // For each dimension check if dims[i] < shape[i], or-accumulate
  // the result
  auto accBase =
      arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false))
          .getResult();
  for (size_t i = 0; i < shape.size(); i++) {
    auto shapei = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getIndexAttr(shape[i]));

    Value dimi = dyn_cast<Value>(mixedDims[i]);
    if (!dimi) {
      dimi = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexAttr(staticMaskDims[i]));
    }

    Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                      dimi, shapei);
    accBase = arith::OrIOp::create(rewriter, loc, accBase, cmp);
  }

  // condition the memset on the or-accumulation
  // initialize with padding prior to CopyOp
  scf::IfOp::create(rewriter, loc, accBase, [&](OpBuilder &b, Location loc) {
    linalg::FillOp::create(b, loc, ValueRange{other}, ValueRange{alloc});
    scf::YieldOp::create(b, loc);
  });
}

namespace {

struct MakeTensorPtrConverter
    : public OpConversionPattern<tts::MakeTensorPtrOp> {
private:
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  static Type getElementTypeStructuredPtr(tts::MakeTensorPtrOp op) {
    assert(!op.isBlockPtr());
    // tensor<1024x!tt.ptr<f32>>
    auto ptrType = cast<triton::PointerType>(
        cast<RankedTensorType>(op.getType()).getElementType());
    return ptrType.getPointeeType();
  }

  static Type getElementTypeBlockPtr(tts::MakeTensorPtrOp op) {
    assert(op.isBlockPtr());
    // !tt.ptr<tensor<128x64xbf16>, 1>
    auto shapedType = cast<ShapedType>(
        cast<triton::PointerType>(op.getType()).getPointeeType());
    return shapedType.getElementType();
  }

  static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                        ArrayRef<int64_t> staticStrides,
                                        ArrayRef<int64_t> resultShape) {
    auto layout =
        StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
    Type elemType;
    if (op.isBlockPtr()) {
      elemType = getElementTypeBlockPtr(op);
    } else {
      elemType = getElementTypeStructuredPtr(op);
    }
    return MemRefType::get(resultShape, elemType, layout);
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createSideBySideCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    auto targetOffset = ofrToIndexValue(
        accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter),
        loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling side-by-side wraparound
    //
    // Note: We do not support cases where the target has already overflown the
    // number of columns! This is because in PtrAnalysis, the offset has already
    // been collapsed into a single dimension, so it is ambiguous to determine
    // whether the offset actually overflows or just refers to an element on the
    // subsequent rows.
    //
    // Same limitations apply to the stacked wraparound case.
    //
    ////////////////////////////////////////////////////////////////////////////
    //
    //    nextOffset - targetOffset = colSize
    //    d1 + d2 = colSize
    //                          N
    //                                x            clampedOffset
    //      --------------------------*----------------*-----*
    //      |                                          |     nextOffset (might
    //      |                    targetOffset          |             overflow)
    //  y   *-----                    *----------------|
    //      |    |                    |                |
    //  M   |-----                    -----------------|
    //      | d2                              d1       |
    //      --------------------------------------------
    //
    //    x = targetOffset % N
    //    nextOffset = x + colSize
    //    clampedOffset = min(nextOffset, N)
    //    d1 = clampedOffset - x
    //
    ////////////////////////////////////////////////////////////////////////////

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        /* result shape */
        SmallVector<int64_t>{

            // Row stays the same, but mlir doesn't allow this anymore. Put
            // dynamic.
            ShapedType::kDynamic,

            // Column is dynamic, in most cases, this
            // should be the same as the original column.
            // The last chunk may be smaller due to
            // wrapping around.
            ShapedType::kDynamic});

    Value rowSize = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value modN = ofrToIndexValue(op.getMixedShape()[1], loc, rewriter);

    Value x = arith::RemSIOp::create(rewriter, loc, targetOffset, modN);
    Value y = arith::SubIOp::create(rewriter, loc, targetOffset, x);

    SmallVector<Value> strideVals =
        ofrsToIndexValues(op.getMixedStrides(), loc, rewriter);

    // First chunk
    Value nextOffset = arith::AddIOp::create(rewriter, loc, x, colSize);
    Value clampedOffset =
        arith::MinSIOp::create(rewriter, loc, nextOffset, modN);
    Value d1 = arith::SubIOp::create(rewriter, loc, clampedOffset, x);
    SmallVector<Value> sizes1{rowSize, d1};

    auto cast1 = memref::ReinterpretCastOp::create(
        rewriter, loc, resultType, adaptor.getBase(), targetOffset, sizes1,
        strideVals);

    // Second chunk
    Value d2 = arith::SubIOp::create(rewriter, loc, colSize, d1);
    SmallVector<Value> sizes2{rowSize, d2};

    auto cast2 = memref::ReinterpretCastOp::create(
        rewriter, loc, resultType, adaptor.getBase(), y, sizes2, strideVals);

    return {cast1, cast2};
  }

  std::pair<memref::ReinterpretCastOp, memref::ReinterpretCastOp>
  createStackedCastOps(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {

    auto loc = op->getLoc();
    auto resultShape = cast<RankedTensorType>(op.getType()).getShape();

    assert(resultShape.size() == 2);

    auto targetOffset = ofrToIndexValue(
        accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter),
        loc, rewriter);

    ////////////////////////////////////////////////////////////////////////////
    //
    // Handling stacked wraparound
    //
    // We do not support cases where the target offset has already overflown the
    // number of rows. See side-by-side wraparound for details.
    //
    ////////////////////////////////////////////////////////////////////////////
    //    We're loading a tensor of dim (rowSize, colSize)
    //    d1 + d2 = rowSize
    //    d2 is the number of rows that overflow
    //
    //                       cols
    //
    //               wrappedAroundOff
    //      --------------*------------*--------
    //      |        d2   |            |       |
    //      |             |------------|       |
    //  rows|                                  |
    //      |                                  |
    //      |           targetOffset           |
    //      |             *------------|       |
    //      |             |            |       |
    //      |         d1  |            |       |
    //      |             | clampedOff |       |
    //      --------------*---------------------
    //                    |  overflow  |
    //                    *-------------
    //                 nextOff
    //
    //    wrappedAroundOff = targetOffset % cols
    //    clampedOff = (rows * strideRows) + wrappedAroundOff
    //                  ~~~~~~~~~~~~~~~~~
    //                         ^
    //                         |
    //          We have already computed
    //          rows * strideRows = modRow = shape[1]
    //          in TritonToStructured
    //
    //          clampedOff - targetOffset
    //    d1 = --------------------
    //              strideRows

    auto resultType = getResultMemrefType(
        op, /* offset */ ShapedType::kDynamic,
        /* staticStrides */
        SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
        /* result shape */
        SmallVector<int64_t>{
            // Row is dynamic, in most cases, this should
            // be the same as the original row. The last
            // chunk may be smaller due to wrapping
            // around.
            ShapedType::kDynamic,

            // Col stays the same, which is resultShape[1], but mlir doesn't
            // allow this anymore. So we put dynamic instead.
            ShapedType::kDynamic});

    Value rowSize = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(op.getSizes()[0]));
    Value colSize = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(op.getSizes()[1]));

    Value strideRow = ofrToIndexValue(op.getMixedStrides()[0], loc, rewriter);
    Value strideCol = ofrToIndexValue(op.getMixedStrides()[1], loc, rewriter);

    Value modRow = ofrToIndexValue(op.getMixedShape()[0], loc, rewriter);

    // First chunk
    Value wrappedAroundOff =
        arith::RemSIOp::create(rewriter, loc, targetOffset, strideRow);
    Value clampedOff =
        arith::AddIOp::create(rewriter, loc, modRow, wrappedAroundOff);
    Value d1 = arith::SubIOp::create(rewriter, loc, clampedOff, targetOffset);
    d1 = arith::DivSIOp::create(rewriter, loc, d1, strideRow);

    SmallVector<Value> sizes1{d1, colSize};
    memref::ReinterpretCastOp cast1 = memref::ReinterpretCastOp::create(
        rewriter, loc, resultType, adaptor.getBase(), targetOffset, sizes1,
        ValueRange{strideRow, strideCol});

    // Second chunk
    Value d2 = arith::SubIOp::create(rewriter, loc, rowSize, d1);
    SmallVector<Value> sizes2{d2, colSize};
    memref::ReinterpretCastOp cast2 = memref::ReinterpretCastOp::create(
        rewriter, loc, resultType, adaptor.getBase(), wrappedAroundOff, sizes2,
        ValueRange{strideRow, strideCol});

    return {cast1, cast2};
  }

  LogicalResult rewriteSplitPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto parentShape = op.getStaticShape();
    assert(parentShape.size() == 2 &&
           "Only support split pointer for 2D tensors only");
    SmallVector<Value> casts;
    StringRef wrapType;

    // For split pointers, a split dimension is either a dynamic or a non-zero
    // value. The other dimension must be zero.
    auto isSplitDimension = [](int64_t dim) {
      return dim == ShapedType::kDynamic || dim != 0;
    };

    if (isSplitDimension(parentShape[0])) {
      // Stacked case
      assert(parentShape[1] == 0);
      auto [cast1, cast2] = createStackedCastOps(op, adaptor, rewriter);
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_STACKED;
    } else if (isSplitDimension(parentShape[1])) {
      assert(parentShape[0] == 0);
      auto [cast1, cast2] = createSideBySideCastOps(op, adaptor, rewriter);
      casts = {cast1.getResult(), cast2.getResult()};
      wrapType = WRAP_SIDE_BY_SIDE;
    } else {
      llvm_unreachable("Unexpected split pointer shape");
    }

    auto combinedCast = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), op.getType(), casts);

    combinedCast->setAttr(wrapType, rewriter.getUnitAttr());

    rewriter.replaceOp(op, combinedCast);

    return success();
  }

  LogicalResult rewritePtr(ArrayRef<int64_t> resultShape, bool isBlockPtr,
                           tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {

    auto mixedStrides = getMixedStridesForMemref(op, rewriter);
    SmallVector<int64_t> staticStrides;
    SmallVector<Value> dynamicStrides;
    dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

    auto targetOffset =
        accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter);
    auto staticTargetOffset = getIntAttr(targetOffset);
    auto loc = op->getLoc();
    auto mixShapes = op.getMixedShape();
    auto mixSizes = op.getMixedSizes();
    auto mixOffsets = op.getMixedOffsets();
    auto mixOrigOffsets = op.getMixedOriginalOffsets();
    bool isBlockPtr1 = false;
    SmallVector<OpFoldResult> actualSizes;

    if (hasConstZero(mixShapes[0])) {
      isBlockPtr1 = true;
    }
    memref::ReinterpretCastOp castOp;

    // Track whether we used actualSizes (which may be smaller than resultShape
    // due to boundary)
    bool usedActualSizes = false;
    if (!isBlockPtr1 && mixSizes.size() == mixOffsets.size() &&
        mixShapes.size() == mixOffsets.size() &&
        mixSizes.size() == mixShapes.size()) {
      for (int32_t i = 0; i < mixSizes.size(); i++) {
        auto offset = mixOffsets[i];
        auto actualOffset = mixOrigOffsets[i];
        auto remaining = subOFRs(mixShapes[i], actualOffset, loc, rewriter);
        auto actualSize = minOFRs(mixSizes[i], remaining, loc, rewriter);
        actualSizes.push_back(actualSize);
      }
      MemRefType resultType;
      if (mixSizes.size() == 1) {
        resultType = getResultMemrefType(
            op, staticTargetOffset.value_or(ShapedType::kDynamic),
            staticStrides, ShapedType::kDynamic);
      } else {
        resultType = getResultMemrefType(
            op, ShapedType::kDynamic,
            SmallVector<int64_t>(resultShape.size(), ShapedType::kDynamic),
            SmallVector<int64_t>(mixSizes.size(), ShapedType::kDynamic));
      }
      castOp = memref::ReinterpretCastOp::create(
          rewriter, op.getLoc(), resultType, adaptor.getBase(), targetOffset,
          actualSizes, mixedStrides);
      usedActualSizes = true;
    } else {
      auto resultType = getResultMemrefType(
          op, staticTargetOffset.value_or(ShapedType::kDynamic), staticStrides,
          resultShape);
      castOp = memref::ReinterpretCastOp::create(
          rewriter, op.getLoc(), resultType, adaptor.getBase(), targetOffset,
          mixSizes, mixedStrides);
    }

    Value result = castOp.getResult();
    auto resultTy = cast<MemRefType>(result.getType());
    bool needsCast = false;
    // When usedActualSizes is true, the actual size may be smaller than
    // resultShape (e.g., boundary case where tensor dim < block size), so we
    // should NOT cast to static resultShape which would cause cast
    // incompatibility error.
    if (!usedActualSizes &&
        resultTy.getRank() == static_cast<int64_t>(resultShape.size())) {
      for (size_t i = 0; i < resultShape.size(); ++i) {
        if (resultShape[i] != ShapedType::kDynamic &&
            resultTy.isDynamicDim(i)) {
          needsCast = true;
          break;
        }
      }
    }

    if (needsCast) {
      auto staticType =
          MemRefType::get(resultShape, resultTy.getElementType(),
                          resultTy.getLayout(), resultTy.getMemorySpace());

      auto cast =
          memref::CastOp::create(rewriter, op.getLoc(), staticType, result);
      rewriter.replaceOp(op, cast);
    } else {
      rewriter.replaceOp(op, castOp);
    }

    return success();
  }

  LogicalResult
  rewriteStructuredPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
    return rewritePtr(resultShape, false, op, adaptor, rewriter);
  }

  LogicalResult rewriteBlockPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Block pointers are basically the same as structured pointers except that
    // the return types are !tt.ptr<tensor<AxBxCxbf16>> instead of
    // tensor<AxBxCx!tt.ptr<bf16>>
    ArrayRef<int64_t> resultShape =
        cast<ShapedType>(
            cast<triton::PointerType>(op.getType()).getPointeeType())
            .getShape();
    return rewritePtr(resultShape, true, op, adaptor, rewriter);
  }

public:
  MakeTensorPtrConverter(const TypeConverter &typeConverter,
                         MLIRContext *context)
      : OpConversionPattern<tts::MakeTensorPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::is_sorted(op.getOrder(), std::greater<>())) {
      emitError(op.getLoc()) << "non-decreasing dimension order on tensor "
                                "pointers are not yet supported";
      return failure();
    }

    if (op.isBlockPtr()) {
      return rewriteBlockPtr(op, adaptor, rewriter);
    }

    if (op.isStructuredPtr()) {
      return rewriteStructuredPtr(op, adaptor, rewriter);
    }

    if (op.isSplitPtr()) {
      return rewriteSplitPtr(op, adaptor, rewriter);
    }

    return failure();
  }
};

struct MakeGatherScatterTensorPtrConverter
    : public OpConversionPattern<tts::MakeGatherScatterTensorPtrOp> {
private:
  using OpConversionPattern<
      tts::MakeGatherScatterTensorPtrOp>::OpConversionPattern;

public:
  MakeGatherScatterTensorPtrConverter(const TypeConverter &typeConverter,
                                      MLIRContext *context)
      : OpConversionPattern<tts::MakeGatherScatterTensorPtrOp>(typeConverter,
                                                               context) {}

  LogicalResult
  matchAndRewrite(tts::MakeGatherScatterTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The gatherScatterPtr is rewritten as separate rows during load/store
    // operations. Therefore, no action is needed here except saving
    // adaptor.getBase().
    rewriter.replaceOp(op, adaptor.getBase());
    return success();
  }
};

struct LoadConverter : public OpConversionPattern<tts::LoadOp> {
private:
  using OpConversionPattern<tts::LoadOp>::OpConversionPattern;

  void createSideBySideCopies(Value block1, Value block2, Value dst,
                              Location loc,
                              ConversionPatternRewriter &rewriter) const {

    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));

    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

    Value block1Row = memref::DimOp::create(rewriter, loc, block1, 0);
    Value block1Col = memref::DimOp::create(rewriter, loc, block1, 1);

    Value block2Row = memref::DimOp::create(rewriter, loc, block2, 0);
    Value block2Col = memref::DimOp::create(rewriter, loc, block2, 1);

    auto block1Dst = memref::SubViewOp::create(rewriter, loc, dst, /* offsets */
                                               ValueRange{zero, zero},
                                               /* sizes */
                                               ValueRange{block1Row, block1Col},
                                               /* strides */
                                               ValueRange{one, one});

    auto block2Dst = memref::SubViewOp::create(rewriter, loc, dst,
                                               /* offsets */
                                               ValueRange{zero, block1Col},
                                               /* sizes */
                                               ValueRange{block2Row, block2Col},
                                               /* strides */
                                               ValueRange{one, one});

    memref::CopyOp::create(rewriter, loc, block1, block1Dst);
    memref::CopyOp::create(rewriter, loc, block2, block2Dst);
  }

  void createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                           ConversionPatternRewriter &rewriter) const {

    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

    Value block1Row = memref::DimOp::create(rewriter, loc, block1, 0);
    Value block1Col = memref::DimOp::create(rewriter, loc, block1, 1);

    Value block2Row = memref::DimOp::create(rewriter, loc, block2, 0);
    Value block2Col = memref::DimOp::create(rewriter, loc, block2, 1);

    auto block1Dst = memref::SubViewOp::create(rewriter, loc, dst, /* offsets */
                                               ValueRange{zero, zero},
                                               /* sizes */
                                               ValueRange{block1Row, block1Col},
                                               /* strides */
                                               ValueRange{one, one});

    auto block2Dst = memref::SubViewOp::create(rewriter, loc, dst,
                                               /* offsets */
                                               ValueRange{block1Row, zero},
                                               /* sizes */
                                               ValueRange{block2Row, block2Col},
                                               /* strides */
                                               ValueRange{one, one});

    memref::CopyOp::create(rewriter, loc, block1, block1Dst);
    memref::CopyOp::create(rewriter, loc, block2, block2Dst);
  }

  memref::SubViewOp createSubview(Value src, ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides, Location loc,
                                  ConversionPatternRewriter &rewriter) const {
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType =
        memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
    return memref::SubViewOp::create(rewriter, loc, cast<MemRefType>(dstType),
                                     src, offsets, sizes, strides);
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getSideBySideSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                        Location loc,
                        ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult col1 =
        memref::DimOp::create(rewriter, loc, block1, 1).getResult();
    OpFoldResult subviewCol1 = minOFRs(col1, subviewColFull, loc, rewriter);
    OpFoldResult subviewCol2 =
        subOFRs(subviewColFull, subviewCol1, loc, rewriter);

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRowFull, subviewCol1},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRowFull, subviewCol2},
                             strides, loc, rewriter);

    return {sv1, sv2};
  }

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getStackedSubviews(ArrayRef<OpFoldResult> dims, Value block1, Value block2,
                     const Location loc,
                     ConversionPatternRewriter &rewriter) const {
    OpFoldResult subviewRowFull = dims[0];
    OpFoldResult subviewColFull = dims[1];
    OpFoldResult row1 =
        memref::DimOp::create(rewriter, loc, block1, 0).getResult();
    OpFoldResult subviewRow1 = minOFRs(row1, subviewRowFull, loc, rewriter);
    OpFoldResult subviewRow2 =
        subOFRs(subviewRowFull, subviewRow1, loc, rewriter);

    SmallVector<OpFoldResult> offsets(dims.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(dims.size(), rewriter.getIndexAttr(1));
    auto sv1 = createSubview(block1, offsets, {subviewRow1, subviewColFull},
                             strides, loc, rewriter);
    auto sv2 = createSubview(block2, offsets, {subviewRow2, subviewColFull},
                             strides, loc, rewriter);
    return {sv1, sv2};
  }

  LogicalResult
  rewriteStructuredLoad(tts::LoadOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    assert(!op.hasMask());

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();
    auto other = op.getOther();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();

    auto alloc = memref::AllocOp::create(
        rewriter, loc, MemRefType::get(tensorType.getShape(), elemType));

    // No mask
    assert(!other && "other value used in non-masked load");

    auto ptrDefiningOp = ptr.getDefiningOp();
    // If ptr is an unrealized_conversion_cast that is NOT a wraparound
    // (side-by-side/stacked), unwrap it to use the source memref directly
    // so subsequent SubView/Copy ops don't keep the cast live.
    if (auto unrealizedCast =
            dyn_cast_or_null<UnrealizedConversionCastOp>(ptrDefiningOp)) {
      if (!unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE) &&
          !unrealizedCast->hasAttr(WRAP_STACKED)) {
        // Unwrap to use the source memref directly. Do NOT erase the
        // unrealized materialization created by the conversion framework;
        // removing it can cause assertion failures when it is an
        // unresolved materialization. Leave cleanup to the conversion
        // infrastructure or later canonicalization passes.
        ptr = unrealizedCast.getInputs()[0];
        ptrDefiningOp = ptr.getDefiningOp();
      }
    }

    if (ptrDefiningOp->hasAttr(WRAP_SIDE_BY_SIDE) ||
        ptrDefiningOp->hasAttr(WRAP_STACKED)) {

      auto unrealizedCast = cast<UnrealizedConversionCastOp>(ptrDefiningOp);
      auto memrefs = unrealizedCast.getOperands();
      assert(memrefs.size() == 2);
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];

      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        createSideBySideCopies(block1, block2, alloc, loc, rewriter);
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        createStackedCopies(block1, block2, alloc, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }
      // Create tensor from alloc and replace the original load op.
      Value tensor = bufferization::ToTensorOp::create(
          rewriter, loc, tensorType, alloc, true /* restrict */,
          true /* writable */);
      rewriter.replaceOp(op, tensor);
      // Do NOT erase the unrealized conversion here; it will be cleaned
      // up once it has no users.
    } else if (!op.getBoundaryCheck().empty()) {
      SmallVector<OpFoldResult> sizes;
      if (auto ReinterpretCastOp =
              ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
        sizes = ReinterpretCastOp.getMixedSizes();
      } else if (auto allocOp = ptr.getDefiningOp<memref::AllocOp>()) {
        auto memrefType = allocOp.getType();
        auto shape = memrefType.getShape();
        for (int64_t dim : shape) {
          sizes.push_back(rewriter.getIndexAttr(dim));
        }
      } else if (auto transposeOp = ptr.getDefiningOp<memref::TransposeOp>()) {
        auto memrefType = transposeOp.getType();
        auto shape = memrefType.getShape();
        for (int64_t dim : shape) {
          sizes.push_back(rewriter.getIndexAttr(dim));
        }
      } else if (auto subView = ptr.getDefiningOp<memref::SubViewOp>()) {
        for (OpFoldResult ofr : subView.getMixedSizes())
          sizes.push_back(ofr);
      } else if (auto CastOp = ptr.getDefiningOp<memref::CastOp>()) {
        // CastOp may cast from dynamic size to static size, we need to get
        // the actual dynamic sizes from the source operation
        auto source = CastOp.getSource();
        if (auto srcReinterpretCast =
                source.getDefiningOp<memref::ReinterpretCastOp>()) {
          sizes = srcReinterpretCast.getMixedSizes();
        } else {
          auto memrefType = CastOp.getType();
          auto shape = memrefType.getShape();
          for (int64_t dim : shape) {
            sizes.push_back(rewriter.getIndexAttr(dim));
          }
        }
      } else if (auto unrealizedCast =
                     ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
        // Handle unrealized_conversion_cast - get sizes from the source memref
        auto source = unrealizedCast.getInputs()[0];
        if (auto srcReinterpretCast =
                source.getDefiningOp<memref::ReinterpretCastOp>()) {
          sizes = srcReinterpretCast.getMixedSizes();
        } else {
          // Get sizes from the source memref type, using dim ops for dynamic
          // dims
          auto srcMemrefType = cast<MemRefType>(source.getType());
          auto shape = srcMemrefType.getShape();
          for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == ShapedType::kDynamic) {
              Value dimVal = memref::DimOp::create(rewriter, loc, source, i);
              sizes.push_back(dimVal);
            } else {
              sizes.push_back(rewriter.getIndexAttr(shape[i]));
            }
          }
        }
        // Use the source memref directly instead of the cast result
        ptr = source;
      } else {
        // Default: get sizes from the memref type
        auto memrefType = cast<MemRefType>(ptr.getType());
        auto shape = memrefType.getShape();
        for (size_t i = 0; i < shape.size(); ++i) {
          if (shape[i] == ShapedType::kDynamic) {
            // For dynamic dimensions, use memref.dim
            Value dimVal = memref::DimOp::create(rewriter, loc, ptr, i);
            sizes.push_back(dimVal);
          } else {
            sizes.push_back(rewriter.getIndexAttr(shape[i]));
          }
        }
      }

      auto paddingAttr = op.getPadding();
      if (paddingAttr.has_value()) {
        Value paddingValue;
        if (auto floatType = dyn_cast<FloatType>(elemType)) {
          const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
          if (paddingAttr.value() == triton::PaddingOption::PAD_NEG_INF) {
            paddingValue = arith::ConstantFloatOp::create(
                rewriter, loc, floatType,
                APFloat::getInf(semantics, /*Negative=*/true));
          } else if (paddingAttr.value() == triton::PaddingOption::PAD_INF) {
            paddingValue = arith::ConstantFloatOp::create(
                rewriter, loc, floatType, APFloat::getInf(semantics));
          } else {
            paddingValue = arith::ConstantFloatOp::create(
                rewriter, loc, floatType, APFloat::getZero(semantics));
          }
        } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
          if (paddingAttr.value() == triton::PaddingOption::PAD_NEG_INF) {
            paddingValue = arith::ConstantIntOp::create(
                rewriter, loc, intType,
                intType.isSigned()
                    ? APInt::getSignedMinValue(intType.getWidth())
                    : APInt::getMinValue(intType.getWidth()));
          } else if (paddingAttr.value() == triton::PaddingOption::PAD_INF) {
            paddingValue = arith::ConstantIntOp::create(
                rewriter, loc, intType,
                intType.isSigned()
                    ? APInt::getSignedMaxValue(intType.getWidth())
                    : APInt::getMaxValue(intType.getWidth()));
          } else {
            paddingValue =
                arith::ConstantIntOp::create(rewriter, loc, intType, 0);
          }
        } else {
          llvm_unreachable("Unsupported element type used for fill");
        }

        fillWithValue(loc, alloc, paddingValue, tensorType.getShape(),
                      std::move(sizes), tensorType.getShape(), rewriter);
      } else {
        Value zeroValue;
        if (auto floatType = dyn_cast<FloatType>(elemType)) {
          const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
          zeroValue = arith::ConstantFloatOp::create(
              rewriter, loc, floatType, APFloat::getZero(semantics));
        } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
          zeroValue = arith::ConstantIntOp::create(rewriter, loc, intType, 0);
        } else {
          llvm_unreachable("Unsupported element type used for fill");
        }

        fillWithValue(loc, alloc, zeroValue, tensorType.getShape(),
                      std::move(sizes), tensorType.getShape(), rewriter);
      }
      memref::SubViewOp srcSubview =
          getSubview(tensorType.getRank(), sizes, ptr, loc, rewriter);
      memref::SubViewOp dstSubview =
          getSubview(tensorType.getRank(), sizes, alloc, loc, rewriter);
      memref::CopyOp::create(rewriter, loc, srcSubview, dstSubview);
      Value tensor = bufferization::ToTensorOp::create(
          rewriter, loc, tensorType, alloc, true /* restrict */,
          true /* writable */);
      rewriter.replaceOp(op, tensor);
    } else {
      Value tensor = bufferization::ToTensorOp::create(
          rewriter, loc, tensorType, ptr, true /* restrict */,
          true /* writable */);
      rewriter.replaceOp(op, tensor);
    }

    return success();
  }

  LogicalResult rewriteMaskedLoad(tts::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    assert(op.hasMask());
    if (!op.getBoundaryCheck().empty()) {
      return op.emitError("masked load cannot have boundary_check");
    }

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();

    auto alloc = memref::AllocOp::create(
        rewriter, loc, MemRefType::get(tensorType.getShape(), elemType));

    SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();

    // Fill load destination with other value
    if (Value other = op.getOther()) {
      fillWithValue(loc, alloc, other, tensorType.getShape(),
                    op.getMixedMaskDims(), op.getStaticMaskDims(), rewriter);
    }

    auto ptrDefiningOp = ptr.getDefiningOp();
    // If ptr is an unrealized_conversion_cast that is NOT a wraparound
    // (side-by-side/stacked), unwrap it to use the source memref directly
    // so subsequent SubView/Copy ops don't keep the cast live.
    if (auto unrealizedCast =
            dyn_cast_or_null<UnrealizedConversionCastOp>(ptrDefiningOp)) {
      if (!unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE) &&
          !unrealizedCast->hasAttr(WRAP_STACKED)) {
        // Unwrap to the source memref; do NOT erase the unrealized
        // conversion materialization here.
        ptr = unrealizedCast.getInputs()[0];
        ptrDefiningOp = ptr.getDefiningOp();
      }
    }

    if (ptrDefiningOp->hasAttr(WRAP_SIDE_BY_SIDE) ||
        ptrDefiningOp->hasAttr(WRAP_STACKED)) {

      auto unrealizedCast = cast<UnrealizedConversionCastOp>(ptrDefiningOp);

      auto memrefs = unrealizedCast.getOperands();
      assert(memrefs.size() == 2);
      auto block1 = memrefs[0];
      auto block2 = memrefs[1];

      if (unrealizedCast->hasAttr(WRAP_SIDE_BY_SIDE)) {
        auto [subview1, subview2] =
            getSideBySideSubviews(mixedDims, block1, block2, loc, rewriter);
        createSideBySideCopies(subview1, subview2, alloc, loc, rewriter);
      } else if (unrealizedCast->hasAttr(WRAP_STACKED)) {
        auto [subview1, subview2] =
            getStackedSubviews(mixedDims, block1, block2, loc, rewriter);
        createStackedCopies(subview1, subview2, alloc, loc, rewriter);
      } else {
        llvm_unreachable("unexpected wraparound type");
      }

      rewriter.eraseOp(unrealizedCast);

    } else {
      // If ptr is a memref.cast materialization, use its source memref
      // for the SubView/Copy. Do NOT erase the cast op here.
      Value actualPtr = ptr;
      if (auto castOp = ptr.getDefiningOp<memref::CastOp>())
        actualPtr = castOp.getSource();

      memref::SubViewOp srcSubview =
          getSubview(tensorType.getRank(), mixedDims, actualPtr, loc, rewriter);
      memref::SubViewOp dstSubview =
          getSubview(tensorType.getRank(), mixedDims, alloc, loc, rewriter);
      memref::CopyOp::create(rewriter, loc, srcSubview, dstSubview);
    }

    Value tensor = bufferization::ToTensorOp::create(rewriter, loc, tensorType,
                                                     alloc, true /* restrict */,
                                                     true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }

  LogicalResult rewriteGather(tts::MakeGatherScatterTensorPtrOp ptr,
                              tts::LoadOp op, Value memRefPtr,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();

    Value gatherOffset = ptr.getGatherScatterOffset();
    // Cast gatherOffset to index
    auto offsetShapedType = cast<ShapedType>(gatherOffset.getType());
    unsigned offsetSize = offsetShapedType.getShape()[0];
    auto indexOffsetTy = RankedTensorType::get(offsetShapedType.getShape(),
                                               rewriter.getIndexType());
    gatherOffset =
        arith::IndexCastOp::create(rewriter, loc, indexOffsetTy, gatherOffset)
            .getResult();

    int gatherDim = ptr.getGatherScatterDim();

    auto offsets = ptr.getMixedOffsets();
    auto strides = ptr.getMixedStrides();

    std::vector<int64_t> staticSizes = ptr.getSizes();
    staticSizes[gatherDim] = 1;
    SmallVector<Value> dynSizes; // sizes are always static
    auto sizes = mlir::getMixedValues(staticSizes, dynSizes, rewriter);

    // Create alloc to save the result.
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto allocType =
        MemRefType::get(resultType.getShape(), resultType.getElementType());
    auto alloc = memref::AllocOp::create(rewriter, loc, allocType);

    auto allocStrides = mlir::getMixedValues(
        allocType.getStridesAndOffset().first, dynSizes, rewriter);
    // Fill load destination with other value
    if (Value other = op.getOther()) {
      fillWithValue(loc, alloc, other, resultType.getShape(),
                    op.getMixedMaskDims(), op.getStaticMaskDims(), rewriter);
    }

    // Create loop to iterate every offset in gatherOffset.
    auto lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upperBound =
        arith::ConstantIndexOp::create(rewriter, loc, offsetSize).getResult();
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      OpFoldResult gatherMaskDim = mixedDims[gatherDim];
      // If gatherMaskDim is a immediate, we can just update the offsetSize
      // to the value of gatherMaskDim.
      // Otherwise, we will need to compare the induction variable with
      // gatherMaskDim to guard the load.
      if (auto gatherMaskDimIndex = getIntAttr(gatherMaskDim)) {
        // If the gather mask dimension is a constant, we can use it directly.
        unsigned gatherMaskDimValue = gatherMaskDimIndex.value();
        offsetSize = std::min(offsetSize, gatherMaskDimValue);
        upperBound = arith::ConstantIndexOp::create(rewriter, loc, offsetSize)
                         .getResult();
      } else {
        // Use arith::MinSIOp to get the minimum value of gatherMaskDim
        // and offsetSize.
        auto gatherMaskDimVal = cast<Value>(gatherMaskDim);
        auto offsetSizeVal =
            arith::ConstantIndexOp::create(rewriter, loc, offsetSize);
        upperBound = arith::MinSIOp::create(rewriter, loc, gatherMaskDimVal,
                                            offsetSizeVal)
                         .getResult();
      }
    }
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto loop = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);

    // Create tensor from alloc and use it as the result to replace op.
    Value tensor = bufferization::ToTensorOp::create(
        rewriter, loc, op.getType(), alloc, true /* restrict */,
        true /* writable */);
    rewriter.replaceOp(op, tensor);

    // Build loop body.
    rewriter.setInsertionPointToStart(loop.getBody());

    // Load the offsetElt first.
    Value inductionVar = loop.getInductionVar();
    auto gatherOffsetElt = tensor::ExtractOp::create(
        rewriter, loc, gatherOffset, ValueRange{inductionVar});

    // reinterpret_cast to current row as memRefPtr[gatherOffsetElt].
    Value srcPtr = rewriteGatherScatterPtrElement(staticSizes, ptr, memRefPtr,
                                                  gatherOffsetElt.getResult(),
                                                  gatherDim, rewriter);
    unsigned rank = ptr.getSizes().size();
    // The subview should not apply an additional stride to the source.
    SmallVector<OpFoldResult> oneStrides(rank, OpFoldResult(step));
    // subview from srcPtr for mask.
    // With offsets[gatherDim] set to 0 since the offset already in
    // reinterpret_cast. With sizes[gatherDim] set to 1 since we are load one
    // row each time.
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      mixedDims[gatherDim] = sizes[gatherDim];
      sizes = mixedDims;
      // maskOffsets should be all zero, since srcPtr already has the offsets.
      SmallVector<OpFoldResult> maskOffsets(rank, OpFoldResult(lowerBound));
      // Use oneStrides for subview.
      auto dstSubViewType = memref::SubViewOp::inferResultType(
          cast<MemRefType>(srcPtr.getType()), maskOffsets, sizes, oneStrides);
      srcPtr = memref::SubViewOp::create(rewriter, loc,
                                         cast<MemRefType>(dstSubViewType),
                                         srcPtr, maskOffsets, sizes, oneStrides)
                   .getResult();
    }

    // alloc[inductionVar]
    SmallVector<OpFoldResult> allocOffsets(rank, OpFoldResult(lowerBound));
    allocOffsets[gatherDim] = inductionVar;
    auto dstAllocType = memref::SubViewOp::inferResultType(
        allocType, allocOffsets, sizes, oneStrides);
    auto dstSubview =
        memref::SubViewOp::create(rewriter, loc, cast<MemRefType>(dstAllocType),
                                  alloc, allocOffsets, sizes, oneStrides);
    // Copy srcPtr to alloc[inductionVar].
    memref::CopyOp::create(rewriter, loc, srcPtr, dstSubview);

    return success();
  }

public:
  LoadConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::LoadOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = op.getPtr();
    if (auto gatherScatterPtr =
            ptr.getDefiningOp<tts::MakeGatherScatterTensorPtrOp>()) {
      // Get the remapped base pointer (converted to memref)
      Value baseMemref = rewriter.getRemappedValue(gatherScatterPtr.getBase());
      return rewriteGather(gatherScatterPtr, op, baseMemref, rewriter);
    }

    if (op.hasMask()) {
      return rewriteMaskedLoad(op, adaptor, rewriter);
    } else {
      return rewriteStructuredLoad(op, adaptor, rewriter);
    }
  }
};

struct StoreConverter : public OpConversionPattern<tts::StoreOp> {
private:
  using OpConversionPattern<tts::StoreOp>::OpConversionPattern;

  static tensor::ExtractSliceOp
  getExtractSlice(int rank, ArrayRef<OpFoldResult> dims, Value source,
                  const Location loc, OpBuilder &b) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

    return tensor::ExtractSliceOp::create(b, loc, source, offsets, dims,
                                          strides);
  }

  LogicalResult rewriteScatter(tts::MakeGatherScatterTensorPtrOp ptr,
                               tts::StoreOp op, Value memRefPtr, Value stVal,
                               ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();

    Value gatherOffset = ptr.getGatherScatterOffset();
    // Cast gatherOffset to index.
    auto offsetShapedType = cast<ShapedType>(gatherOffset.getType());
    unsigned offsetSize = offsetShapedType.getShape()[0];
    auto indexOffsetTy = RankedTensorType::get(offsetShapedType.getShape(),
                                               rewriter.getIndexType());
    gatherOffset =
        arith::IndexCastOp::create(rewriter, loc, indexOffsetTy, gatherOffset)
            .getResult();

    int gatherDim = ptr.getGatherScatterDim();

    auto offsets = ptr.getMixedOffsets();
    auto strides = ptr.getMixedStrides();

    std::vector<int64_t> staticSizes = ptr.getSizes();
    staticSizes[gatherDim] = 1;
    SmallVector<Value> dynSizes; // sizes are always static
    auto sizes = mlir::getMixedValues(staticSizes, dynSizes, rewriter);

    // Create loop to iterate every offset in gatherOffset.
    auto lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upperBound =
        arith::ConstantIndexOp::create(rewriter, loc, offsetSize).getResult();
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      OpFoldResult gatherMaskDim = mixedDims[gatherDim];
      // If gatherMaskDim is a immediate, we can just update the offsetSize
      // to the value of gatherMaskDim.
      // Otherwise, we will need to compare the induction variable with
      // gatherMaskDim to guard the load.
      if (auto gatherMaskDimIndex = getIntAttr(gatherMaskDim)) {
        // If the gather mask dimension is a constant, we can use it directly.
        unsigned gatherMaskDimValue = gatherMaskDimIndex.value();
        offsetSize = std::min(offsetSize, gatherMaskDimValue);
        upperBound = arith::ConstantIndexOp::create(rewriter, loc, offsetSize)
                         .getResult();
      } else {
        // Use arith::MinSIOp to get the minimum value of gatherMaskDim
        // and offsetSize.
        auto gatherMaskDimVal = cast<Value>(gatherMaskDim);
        auto offsetSizeVal =
            arith::ConstantIndexOp::create(rewriter, loc, offsetSize);
        upperBound = arith::MinSIOp::create(rewriter, loc, gatherMaskDimVal,
                                            offsetSizeVal)
                         .getResult();
      }
    }
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto loop = scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);

    // Build loop body.
    rewriter.setInsertionPointToStart(loop.getBody());

    // Load the offsetElt first.
    Value inductionVar = loop.getInductionVar();

    auto gatherOffsetElt = tensor::ExtractOp::create(
        rewriter, loc, gatherOffset, ValueRange{inductionVar});

    // Create extract_slice stVal[inductionVar].
    unsigned rank = ptr.getSizes().size();
    SmallVector<OpFoldResult> stValOffsets(rank, OpFoldResult(lowerBound));
    stValOffsets[gatherDim] = inductionVar;

    // Use mixed mask dims as sizes with mixedDims[gatherDim] set to 1 when
    // hasMask.
    if (op.hasMask()) {
      SmallVector<OpFoldResult> mixedDims = op.getMixedMaskDims();
      mixedDims[gatherDim] = sizes[gatherDim];
      sizes = mixedDims;
    }
    // The subview should not apply an additional stride to the source.
    SmallVector<OpFoldResult> oneStrides(rank, OpFoldResult(step));
    auto slice = tensor::ExtractSliceOp::create(
        rewriter, loc, stVal, stValOffsets, sizes, oneStrides);

    // reinterpret_cast to current row as memRefPtr[gatherOffsetElt].
    Value dstPtr = rewriteGatherScatterPtrElement(staticSizes, ptr, memRefPtr,
                                                  gatherOffsetElt.getResult(),
                                                  gatherDim, rewriter);
    // subview from dstPtr for mask.
    // Set offsets[] to 0 since it gatherOffsetElt already in reinterpret_cast.
    if (op.hasMask()) {
      // maskOffsets should be all zero, since srcPtr already has the offsets.
      SmallVector<OpFoldResult> maskOffsets(rank, OpFoldResult(lowerBound));
      auto dstType = memref::SubViewOp::inferResultType(
          cast<MemRefType>(dstPtr.getType()), maskOffsets, sizes, oneStrides);

      dstPtr =
          memref::SubViewOp::create(rewriter, loc, cast<MemRefType>(dstType),
                                    dstPtr, maskOffsets, sizes, oneStrides)
              .getResult();
    }
    // store slice to dstPtr.
    auto storeOp = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, slice, dstPtr);
    storeOp.setWritable(true);

    rewriter.eraseOp(op);

    return success();
  }

public:
  StoreConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::StoreOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    if (auto gatherScatterPtr =
            op.getPtr().getDefiningOp<tts::MakeGatherScatterTensorPtrOp>()) {
      // Get the remapped base pointer (converted to memref)
      Value baseMemref = rewriter.getRemappedValue(gatherScatterPtr.getBase());
      return rewriteScatter(gatherScatterPtr, op, baseMemref,
                            adaptor.getValue(), rewriter);
    }

    auto ptr = adaptor.getPtr();
    Value storeValue = op.getValue();
    auto storeValueType = cast<RankedTensorType>(storeValue.getType());
    auto rank = storeValueType.getRank();

    // Handle element type mismatch due to tt.bitcast on pointer tensors.
    // When tt.bitcast changes the pointee type (e.g., !tt.ptr<i1> ->
    // !tt.ptr<i8>), the store value type may differ from the memref element
    // type. We need to convert the store value to match the memref's element
    // type.
    auto ptrMemRefType = cast<MemRefType>(ptr.getType());
    Type storeElemType = storeValueType.getElementType();
    Type ptrElemType = ptrMemRefType.getElementType();

    if (storeElemType != ptrElemType) {
      // Convert the store value to match the memref's element type
      auto storeElemBitWidth = storeElemType.getIntOrFloatBitWidth();
      auto ptrElemBitWidth = ptrElemType.getIntOrFloatBitWidth();

      Value convertedValue;
      auto newTensorType =
          RankedTensorType::get(storeValueType.getShape(), ptrElemType);

      if (isa<IntegerType>(storeElemType) && isa<IntegerType>(ptrElemType)) {
        if (storeElemBitWidth > ptrElemBitWidth) {
          // Truncate: e.g., i8 -> i1
          convertedValue =
              linalg::GenericOp::create(
                  rewriter, loc,
                  /*resultTypes=*/TypeRange{newTensorType},
                  /*inputs=*/ValueRange{storeValue},
                  /*outputs=*/
                  ValueRange{tensor::EmptyOp::create(
                      rewriter, loc, newTensorType.getShape(), ptrElemType)},
                  /*indexingMaps=*/
                  SmallVector<AffineMap>{rewriter.getMultiDimIdentityMap(rank),
                                         rewriter.getMultiDimIdentityMap(rank)},
                  /*iteratorTypes=*/
                  SmallVector<utils::IteratorType>(
                      rank, utils::IteratorType::parallel),
                  /*bodyBuilder=*/
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value truncated =
                        arith::TruncIOp::create(b, loc, ptrElemType, args[0]);
                    linalg::YieldOp::create(b, loc, truncated);
                  })
                  .getResult(0);
        } else {
          // Extend: e.g., i1 -> i8
          convertedValue =
              linalg::GenericOp::create(
                  rewriter, loc,
                  /*resultTypes=*/TypeRange{newTensorType},
                  /*inputs=*/ValueRange{storeValue},
                  /*outputs=*/
                  ValueRange{tensor::EmptyOp::create(
                      rewriter, loc, newTensorType.getShape(), ptrElemType)},
                  /*indexingMaps=*/
                  SmallVector<AffineMap>{rewriter.getMultiDimIdentityMap(rank),
                                         rewriter.getMultiDimIdentityMap(rank)},
                  /*iteratorTypes=*/
                  SmallVector<utils::IteratorType>(
                      rank, utils::IteratorType::parallel),
                  /*bodyBuilder=*/
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value extended =
                        arith::ExtUIOp::create(b, loc, ptrElemType, args[0]);
                    linalg::YieldOp::create(b, loc, extended);
                  })
                  .getResult(0);
        }
      } else if (isa<FloatType>(storeElemType) && isa<FloatType>(ptrElemType)) {
        if (storeElemBitWidth > ptrElemBitWidth) {
          // Truncate float
          convertedValue =
              linalg::GenericOp::create(
                  rewriter, loc,
                  /*resultTypes=*/TypeRange{newTensorType},
                  /*inputs=*/ValueRange{storeValue},
                  /*outputs=*/
                  ValueRange{tensor::EmptyOp::create(
                      rewriter, loc, newTensorType.getShape(), ptrElemType)},
                  /*indexingMaps=*/
                  SmallVector<AffineMap>{rewriter.getMultiDimIdentityMap(rank),
                                         rewriter.getMultiDimIdentityMap(rank)},
                  /*iteratorTypes=*/
                  SmallVector<utils::IteratorType>(
                      rank, utils::IteratorType::parallel),
                  /*bodyBuilder=*/
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value truncated =
                        arith::TruncFOp::create(b, loc, ptrElemType, args[0]);
                    linalg::YieldOp::create(b, loc, truncated);
                  })
                  .getResult(0);
        } else {
          // Extend float
          convertedValue =
              linalg::GenericOp::create(
                  rewriter, loc,
                  /*resultTypes=*/TypeRange{newTensorType},
                  /*inputs=*/ValueRange{storeValue},
                  /*outputs=*/
                  ValueRange{tensor::EmptyOp::create(
                      rewriter, loc, newTensorType.getShape(), ptrElemType)},
                  /*indexingMaps=*/
                  SmallVector<AffineMap>{rewriter.getMultiDimIdentityMap(rank),
                                         rewriter.getMultiDimIdentityMap(rank)},
                  /*iteratorTypes=*/
                  SmallVector<utils::IteratorType>(
                      rank, utils::IteratorType::parallel),
                  /*bodyBuilder=*/
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value extended =
                        arith::ExtFOp::create(b, loc, ptrElemType, args[0]);
                    linalg::YieldOp::create(b, loc, extended);
                  })
                  .getResult(0);
        }
      } else {
        // For other type combinations, use bitcast semantics
        // This is a fallback and may not be correct for all cases
        convertedValue = storeValue;
      }

      storeValue = convertedValue;
    }

    if (op.hasMask()) {
      auto mixedDims = op.getMixedMaskDims();

      auto srcSlice =
          getExtractSlice(rank, mixedDims, storeValue, loc, rewriter);
      auto dstSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);

      auto storeOp = bufferization::MaterializeInDestinationOp::create(
          rewriter, loc, srcSlice, dstSubview);
      storeOp.setWritable(true);
    } else if (!op.getBoundaryCheck().empty()) {
      SmallVector<OpFoldResult> sizes;
      if (auto castOp = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
        sizes = castOp.getMixedSizes();
      } else if (auto allocOp = ptr.getDefiningOp<memref::AllocOp>()) {
        auto memrefType = allocOp.getType();
        auto shape = memrefType.getShape();
        for (int64_t dim : shape) {
          sizes.push_back(rewriter.getIndexAttr(dim));
        }
      } else if (auto transposeOp = ptr.getDefiningOp<memref::TransposeOp>()) {
        auto memrefType = transposeOp.getType();
        auto shape = memrefType.getShape();
        for (int64_t dim : shape) {
          sizes.push_back(rewriter.getIndexAttr(dim));
        }
      } else if (auto subView = ptr.getDefiningOp<memref::SubViewOp>()) {
        for (OpFoldResult ofr : subView.getMixedSizes())
          sizes.push_back(ofr);
      } else if (auto CastOp = ptr.getDefiningOp<memref::CastOp>()) {
        // CastOp may cast from dynamic size to static size, we need to get
        // the actual dynamic sizes from the source operation
        auto source = CastOp.getSource();
        if (auto srcReinterpretCast =
                source.getDefiningOp<memref::ReinterpretCastOp>()) {
          sizes = srcReinterpretCast.getMixedSizes();
        } else {
          auto memrefType = CastOp.getType();
          auto shape = memrefType.getShape();
          for (int64_t dim : shape) {
            sizes.push_back(rewriter.getIndexAttr(dim));
          }
        }
      } else if (auto unrealizedCast =
                     ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
        // Handle unrealized_conversion_cast - get sizes from the source memref
        auto source = unrealizedCast.getInputs()[0];
        if (auto srcReinterpretCast =
                source.getDefiningOp<memref::ReinterpretCastOp>()) {
          sizes = srcReinterpretCast.getMixedSizes();
        } else {
          // Get sizes from the source memref type, using dim ops for dynamic
          // dims
          auto srcMemrefType = cast<MemRefType>(source.getType());
          auto shape = srcMemrefType.getShape();
          for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == ShapedType::kDynamic) {
              Value dimVal = memref::DimOp::create(rewriter, loc, source, i);
              sizes.push_back(dimVal);
            } else {
              sizes.push_back(rewriter.getIndexAttr(shape[i]));
            }
          }
        }
        // Use the source memref directly instead of the cast result
        ptr = source;
      } else {
        // Default: get sizes from the memref type
        auto memrefType = cast<MemRefType>(ptr.getType());
        auto shape = memrefType.getShape();
        for (size_t i = 0; i < shape.size(); ++i) {
          if (shape[i] == ShapedType::kDynamic) {
            // For dynamic dimensions, use memref.dim
            Value dimVal = memref::DimOp::create(rewriter, loc, ptr, i);
            sizes.push_back(dimVal);
          } else {
            sizes.push_back(rewriter.getIndexAttr(shape[i]));
          }
        }
      }
      auto srcSlice = getExtractSlice(rank, sizes, storeValue, loc, rewriter);
      auto dstSubview = getSubview(rank, sizes, ptr, loc, rewriter);
      auto storeOp = bufferization::MaterializeInDestinationOp::create(
          rewriter, loc, srcSlice, dstSubview);
      storeOp.setWritable(true);
    } else {
      auto storeOp = bufferization::MaterializeInDestinationOp::create(
          rewriter, loc, storeValue, ptr);
      storeOp.setWritable(true);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct XSMTAllocConverter : public OpConversionPattern<xsmt::AllocOp> {
private:
  using OpConversionPattern<xsmt::AllocOp>::OpConversionPattern;

public:
  XSMTAllocConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<xsmt::AllocOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(xsmt::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto shapeAttr = op.getShape();
    SmallVector<int64_t> shape;
    shape.reserve(shapeAttr.size());
    for (int32_t dim : shapeAttr) {
      shape.push_back(static_cast<int64_t>(dim));
    }

    Value allocResult = op.getResult();
    auto allocPtrType = cast<triton::PointerType>(allocResult.getType());
    Type originalPointeeType = allocPtrType.getPointeeType();

    Type elementType;
    if (auto tensorType = dyn_cast<RankedTensorType>(originalPointeeType)) {
      elementType = tensorType.getElementType();
    } else if (isa<FloatType>(originalPointeeType) ||
               isa<IntegerType>(originalPointeeType)) {
      elementType = originalPointeeType;
    } else {
      emitError(loc) << "Unsupported pointee type: " << originalPointeeType;
      return failure();
    }

    auto storageAttr = op->getAttr("storage");
    if (!storageAttr) {
      emitWarning(loc)
          << "'storage' attribute not found on xsmt.alloc, using default";
    }

    MemRefType memrefType = MemRefType::get(shape, elementType);

    auto allocOp = memref::AllocOp::create(rewriter, loc, memrefType,
                                           /*dynamicSizes=*/ValueRange{},
                                           /*symbolOperands=*/ValueRange{},
                                           /*alignment=*/nullptr);

    if (storageAttr) {
      allocOp->setAttr("storage", storageAttr);
    }

    rewriter.replaceOp(op, allocOp.getResult());
    return success();
  }
};

struct XSMTViewConverter : public OpConversionPattern<xsmt::ViewPtrOp> {
private:
  using OpConversionPattern<xsmt::ViewPtrOp>::OpConversionPattern;

public:
  XSMTViewConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<xsmt::ViewPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(xsmt::ViewPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value base = adaptor.getBase();
    auto baseType = dyn_cast<MemRefType>(base.getType());
    if (!baseType)
      return rewriter.notifyMatchFailure(
          op, "base is not a MemRefType after conversion");

    auto resultType = op.getResult().getType();
    auto ptrType = dyn_cast<triton::PointerType>(resultType);
    if (!ptrType)
      return rewriter.notifyMatchFailure(
          op, "Unsupported result type (not a triton ptr)");

    Type elementType;
    Type pointeeType = ptrType.getPointeeType();
    if (auto tensorType = dyn_cast<RankedTensorType>(pointeeType))
      elementType = tensorType.getElementType();
    else
      elementType = pointeeType;

    ValueRange offsets = adaptor.getOffsets();
    SmallVector<OpFoldResult> offsetValues;
    offsetValues.reserve(offsets.size());
    for (Value off : offsets) {
      if (!off.getType().isIndex())
        off = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                         off);
      offsetValues.push_back(off);
    }

    auto shapeAttr = op.getShape();
    auto microSizeAttr = op.getMicroSize();

    bool hasMicroSize = false;
    bool isSameMicroSize = false;
    ArrayRef<int32_t> preMicroSize;

    if (auto dense =
            op->getAttrOfType<mlir::DenseI32ArrayAttr>("pre_micro_size")) {
      preMicroSize = dense.asArrayRef();
    }

    if (!preMicroSize.empty()) {
      hasMicroSize = true;
      if (preMicroSize == microSizeAttr) {
        isSameMicroSize = true;
      }
    }

    if (!hasMicroSize) {
      SmallVector<int64_t> shapeDims;
      shapeDims.reserve(shapeAttr.size());
      for (int32_t d : shapeAttr)
        shapeDims.push_back(static_cast<int64_t>(d));

      SmallVector<int64_t> microSizeDims;
      microSizeDims.reserve(microSizeAttr.size());
      for (int32_t d : microSizeAttr)
        microSizeDims.push_back(static_cast<int64_t>(d));

      if (shapeDims.size() != microSizeDims.size())
        return rewriter.notifyMatchFailure(op, "shape rank != micro_size rank");

      SmallVector<OpFoldResult> sizeValues;
      sizeValues.reserve(shapeDims.size());
      for (int64_t dim : shapeDims)
        sizeValues.push_back(rewriter.getIndexAttr(dim));

      SmallVector<OpFoldResult> strideValues(shapeDims.size(),
                                             rewriter.getIndexAttr(1));

      Type inferredSubviewTy = memref::SubViewOp::inferResultType(
          baseType, offsetValues, sizeValues, strideValues);
      auto inferredSubviewMemRefTy = dyn_cast<MemRefType>(inferredSubviewTy);
      if (!inferredSubviewMemRefTy)
        return rewriter.notifyMatchFailure(
            op, "inferResultType did not return a MemRefType");

      auto subview = memref::SubViewOp::create(
          rewriter, loc, inferredSubviewMemRefTy, base, offsetValues,
          sizeValues, strideValues);

      SmallVector<int64_t> expandedShape;
      expandedShape.reserve(shapeDims.size() * 2);

      SmallVector<ReassociationIndices> reassociation;
      reassociation.reserve(shapeDims.size());

      for (size_t i = 0; i < shapeDims.size(); i++) {
        int64_t dim = shapeDims[i];
        int64_t micro = microSizeDims[i];
        if (micro <= 0)
          return rewriter.notifyMatchFailure(op, "micro_size must be > 0");
        if (dim % micro != 0)
          return rewriter.notifyMatchFailure(
              op, "shape dim is not divisible by micro_size");

        expandedShape.push_back(dim / micro);
        expandedShape.push_back(micro);

        reassociation.push_back(ReassociationIndices{
            static_cast<int64_t>(2 * i), static_cast<int64_t>(2 * i + 1)});
      }

      SmallVector<int64_t> subStrides;
      int64_t subOffset = 0;
      if (failed(inferredSubviewMemRefTy.getStridesAndOffset(subStrides,
                                                             subOffset))) {
        return rewriter.notifyMatchFailure(
            op, "failed to get strides/offset from subview memref");
      }
      if (subStrides.size() != shapeDims.size())
        return rewriter.notifyMatchFailure(op, "subview stride rank mismatch");

      SmallVector<int64_t> expandedStrides;
      expandedStrides.reserve(subStrides.size() * 2);

      for (size_t i = 0; i < subStrides.size(); i++) {
        int64_t s = subStrides[i];
        int64_t micro = microSizeDims[i];
        int64_t outerStride;
        if (s == ShapedType::kDynamic) {
          outerStride = ShapedType::kDynamic;
        } else {
          if (micro != 0 && (s > (std::numeric_limits<int64_t>::max() / micro)))
            return rewriter.notifyMatchFailure(
                op, "stride overflow when computing expanded strides");
          outerStride = s * micro;
        }

        expandedStrides.push_back(outerStride);
        expandedStrides.push_back(s);
      }

      auto *ctx = rewriter.getContext();
      auto expandedLayout =
          StridedLayoutAttr::get(ctx, subOffset, expandedStrides);
      auto expandedType =
          MemRefType::get(expandedShape, elementType, expandedLayout,
                          inferredSubviewMemRefTy.getMemorySpace());

      auto expandShape = memref::ExpandShapeOp::create(
          rewriter, loc, expandedType, subview.getResult(), reassociation);

      SmallVector<int64_t> permutation;
      permutation.reserve(expandedShape.size());
      if (expandedShape.size() == 4) {
        permutation = {0, 2, 1, 3};
      } else {
        for (int64_t i = 0, e = static_cast<int64_t>(expandedShape.size());
             i < e; i++)
          permutation.push_back(i);
      }

      SmallVector<AffineExpr> exprs;
      exprs.reserve(permutation.size());
      for (int64_t p : permutation)
        exprs.push_back(rewriter.getAffineDimExpr(p));
      auto permMap = AffineMap::get(static_cast<unsigned>(expandedShape.size()),
                                    /*symbolCount=*/0, exprs, ctx);
      auto permMapAttr = AffineMapAttr::get(permMap);

      SmallVector<int64_t> finalShape;
      finalShape.reserve(expandedShape.size());
      SmallVector<int64_t> finalStrides;
      finalStrides.reserve(expandedStrides.size());

      for (size_t i = 0; i < permutation.size(); i++) {
        int64_t p = permutation[i];
        finalShape.push_back(expandedShape[p]);
        finalStrides.push_back(expandedStrides[p]);
      }

      auto finalLayout = StridedLayoutAttr::get(ctx, subOffset, finalStrides);
      auto finalType = MemRefType::get(finalShape, elementType, finalLayout,
                                       expandedType.getMemorySpace());

      auto transpose = memref::TransposeOp::create(
          rewriter, loc, finalType, expandShape.getResult(), permMapAttr);

      rewriter.replaceOp(op, transpose.getResult());
      return success();
    } else if (hasMicroSize && isSameMicroSize) {
      SmallVector<OpFoldResult> sizes;

      ArrayRef<int32_t> preShape;
      if (auto shape =
              op->getAttrOfType<mlir::DenseI32ArrayAttr>("pre_shape")) {
        preShape = shape.asArrayRef();
      } else {
        llvm::errs() << "shape missing or not DenseI32ArrayAttr\n";
      }

      for (int64_t shape : preShape) {
        sizes.push_back(rewriter.getIndexAttr(shape));
      }
      Value tile0Value =
          arith::ConstantIndexOp::create(rewriter, loc, microSizeAttr[0]);
      Value tile1Value =
          arith::ConstantIndexOp::create(rewriter, loc, microSizeAttr[1]);

      Value shape0 =
          arith::ConstantIndexOp::create(rewriter, loc, shapeAttr[0]);
      Value shape1 =
          arith::ConstantIndexOp::create(rewriter, loc, shapeAttr[1]);
      Value size0 = createCeilDivUI(rewriter, loc, shape0, tile0Value);
      Value size1 = createCeilDivUI(rewriter, loc, shape1, tile1Value);

      Value offset0 = createCeilDivUI(rewriter, loc, offsets[0], tile0Value);
      Value offset1 = createCeilDivUI(rewriter, loc, offsets[1], tile1Value);

      Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value src = base;
      if (auto castOp = base.getDefiningOp<UnrealizedConversionCastOp>())
        src = castOp.getOperand(0);

      auto srcMemRefTy = dyn_cast<MemRefType>(src.getType());
      if (!srcMemRefTy)
        return rewriter.notifyMatchFailure(op, "src is not a MemRefType");

      SmallVector<OpFoldResult> offsetValues = {offset0, offset1, c0, c0};
      SmallVector<OpFoldResult> sizeValues = {size0, size1, tile0Value,
                                              tile1Value};
      SmallVector<OpFoldResult> strideValues = {c1, c1, c1, c1};

      auto inferredTy = memref::SubViewOp::inferResultType(
          srcMemRefTy, offsetValues, sizeValues, strideValues);

      auto subview =
          memref::SubViewOp::create(rewriter, loc, inferredTy, src,
                                    offsetValues, sizeValues, strideValues);

      rewriter.replaceOp(op, subview.getResult());
      return success();
    }
    op->emitRemark("StructuredToMemref: do not support diff micro size");
    return failure();
  }
};

struct ViewOpPtrPattern : public OpRewritePattern<xsmt::ViewPtrOp> {
  using OpRewritePattern<xsmt::ViewPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmt::ViewPtrOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("pre_micro_size") && op->hasAttr("pre_shape")) {
      return failure();
    }

    Value base = op.getBase();
    auto defView = base.getDefiningOp<xsmt::ViewPtrOp>();
    if (!defView) {
      return failure();
    }

    auto microSize = defView.getMicroSize();
    auto shape = defView.getShape();
    if (microSize.empty() || shape.empty()) {
      return failure();
    }

    llvm::SmallVector<int32_t, 8> ms32(microSize.begin(), microSize.end());
    llvm::SmallVector<int32_t, 8> sh32(shape.begin(), shape.end());

    auto ctx = rewriter.getContext();
    auto dense0 = DenseI32ArrayAttr::get(ctx, ms32);
    auto dense1 = DenseI32ArrayAttr::get(ctx, sh32);

    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr("pre_micro_size", dense0);
      op->setAttr("pre_shape", dense1);
    });

    return success();
  }
};

static bool allOffsetsAreConstZero(mlir::ValueRange offsets) {
  for (Value v : offsets) {
    APInt c;
    if (!matchPattern(v, m_ConstantInt(&c)))
      return false;
    if (!c.isZero())
      return false;
  }
  return true;
}

static FailureOr<DenseI32ArrayAttr>
computeOutShapeFromViewAttrs(xsmt::ViewPtrOp view) {
  DenseI32ArrayAttr shapeAttr = view.getShapeAttr();
  DenseI32ArrayAttr microAttr = view.getMicroSizeAttr();
  if (!shapeAttr || !microAttr)
    return failure();

  ArrayRef<int32_t> shape = shapeAttr.asArrayRef();
  ArrayRef<int32_t> micro = microAttr.asArrayRef();

  if (shape.size() != micro.size())
    return failure();

  SmallVector<int32_t, 8> outDims;
  outDims.reserve(shape.size() * 2);

  for (size_t i = 0; i < shape.size(); ++i) {
    int32_t s = shape[i];
    int32_t m = micro[i];
    if (m <= 0)
      return failure();
    if (s % m != 0)
      return failure();
    outDims.push_back(s / m);
  }
  for (int32_t m : micro)
    outDims.push_back(m);

  return DenseI32ArrayAttr::get(view.getContext(), outDims);
}

struct FoldAllocViewPtrToAlloc final
    : public OpRewritePattern<xsmt::ViewPtrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmt::ViewPtrOp view,
                                PatternRewriter &rewriter) const override {
    Value base = view.getBase();
    auto oldAlloc = base.getDefiningOp<xsmt::AllocOp>();
    if (!oldAlloc)
      return failure();

    if (!oldAlloc.getResult().hasOneUse())
      return failure();
    if (*oldAlloc.getResult().getUsers().begin() != view.getOperation())
      return failure();

    if (!allOffsetsAreConstZero(view.getOffsets()))
      return failure();
    Type outTy = view.getResult().getType();
    FailureOr<DenseI32ArrayAttr> newShapeAttr =
        computeOutShapeFromViewAttrs(view);
    if (failed(newShapeAttr))
      return failure();

    StringAttr storageAttr = oldAlloc.getStorageAttr();

    rewriter.setInsertionPoint(view);

    auto newAlloc = xsmt::AllocOp::create(rewriter, view.getLoc(), outTy,
                                          *newShapeAttr, storageAttr);

    {
      NamedAttrList extra(oldAlloc->getAttrs());
      extra.erase(StringAttr::get(rewriter.getContext(), "shape"));
      extra.erase(StringAttr::get(rewriter.getContext(), "storage"));
      for (auto it : extra)
        newAlloc->setAttr(it.getName(), it.getValue());
    }

    rewriter.replaceOp(view, newAlloc.getResult());
    rewriter.eraseOp(oldAlloc);

    return success();
  }
};

struct FuseTTSLoadAndViewToDescriptorLoadView
    : public mlir::OpRewritePattern<xsmt::ViewOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(xsmt::ViewOp view,
                  mlir::PatternRewriter &rewriter) const override {

    auto load = view.getBase().getDefiningOp<tts::LoadOp>();
    if (!load)
      return mlir::failure();

    if (!load->hasOneUse())
      return mlir::failure();
    if (*load->user_begin() != view.getOperation())
      return mlir::failure();

    mlir::Location loc = view.getLoc();
    mlir::Type i64Ty = rewriter.getI64Type();

    llvm::SmallVector<mlir::Value, 4> offsets64;
    offsets64.reserve(view.getOffsets().size());

    for (mlir::Value off : view.getOffsets()) {
      mlir::Type ty = off.getType();

      if (ty == i64Ty) {
        offsets64.push_back(off);
        continue;
      }

      if (ty.isInteger(32)) {
        offsets64.push_back(
            mlir::arith::ExtSIOp::create(rewriter, loc, i64Ty, off));
        continue;
      }

      if (ty.isIndex()) {
        offsets64.push_back(
            mlir::arith::IndexCastOp::create(rewriter, loc, i64Ty, off));
        continue;
      }

      return mlir::failure();
    }

    auto shape = view.getShape();
    auto micro = view.getMicroSize();

    auto fused = xsmt::DescriptorLoadViewOp::create(
        rewriter, loc,
        /*resultType=*/view.getResult().getType(),
        /*base=*/load.getPtr(),
        /*offsets=*/offsets64,
        /*shape=*/shape,
        /*micro_size=*/micro);

    rewriter.replaceOp(view, fused.getResult());
    rewriter.eraseOp(load);

    return mlir::success();
  }
};

struct AllocCopiesOpLowering final
    : public OpConversionPattern<xsmt::AllocCopiesOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xsmt::AllocCopiesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type converted = getTypeConverter()->convertType(op.getResult().getType());
    auto dstTy = dyn_cast<MemRefType>(converted);
    if (!dstTy)
      return rewriter.notifyMatchFailure(
          op, "alloc_copies result not convertible to MemRefType");

    MemRefType allocTy =
        MemRefType::get(dstTy.getShape(), dstTy.getElementType(),
                        /*layout=*/MemRefLayoutAttrInterface(),
                        /*memorySpace=*/dstTy.getMemorySpace());

    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < allocTy.getRank(); ++i) {
      if (allocTy.isDynamicDim(i)) {
        return rewriter.notifyMatchFailure(
            op,
            "dynamic dims not supported: alloc_copies has no size operands");
      }
    }

    Value alloc = memref::AllocOp::create(rewriter, loc, allocTy, dynSizes);

    Value result = alloc;
    if (allocTy != dstTy) {
      result = memref::CastOp::create(rewriter, loc, dstTy, alloc);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct BufferTensorViewOpLowering final
    : public OpConversionPattern<xsmt::BufferTensorViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xsmt::BufferTensorViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value buffer = adaptor.getBuffer();
    Value bufferIdx = adaptor.getBufferIdx();

    auto srcTy = dyn_cast<MemRefType>(buffer.getType());
    if (!srcTy)
      return rewriter.notifyMatchFailure(
          op, "buffer is not a MemRefType after conversion");
    if (srcTy.getRank() < 1)
      return rewriter.notifyMatchFailure(op, "buffer rank < 1");

    Type convertedResTy =
        getTypeConverter()->convertType(op.getResult().getType());
    auto dstTy = dyn_cast<MemRefType>(convertedResTy);
    if (!dstTy)
      return rewriter.notifyMatchFailure(
          op, "result not convertible to MemRefType");

    if (!isa<IndexType>(bufferIdx.getType()))
      bufferIdx = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), bufferIdx);

    int64_t rank = srcTy.getRank();
    SmallVector<OpFoldResult> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);

    offsets.push_back(bufferIdx);
    sizes.push_back(rewriter.getIndexAttr(1));
    strides.push_back(rewriter.getIndexAttr(1));

    for (int64_t i = 1; i < rank; ++i) {
      offsets.push_back(rewriter.getIndexAttr(0));
      strides.push_back(rewriter.getIndexAttr(1));

      if (srcTy.isDynamicDim(i)) {
        Value d = memref::DimOp::create(rewriter, loc, buffer, i);
        sizes.push_back(d);
      } else {
        sizes.push_back(rewriter.getIndexAttr(srcTy.getDimSize(i)));
      }
    }
    Value sub = memref::SubViewOp::create(rewriter, loc, dstTy, buffer, offsets,
                                          sizes, strides);

    rewriter.replaceOp(op, sub);
    return success();
  }
};

struct MBarrierCopiesOpLowering
    : public OpConversionPattern<mlir::xsmt::MBarrierCopiesOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::xsmt::MBarrierCopiesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type newTy = getTypeConverter()->convertType(op.getResult().getType());
    auto tensorTy = cast<RankedTensorType>(newTy);
    int64_t n = tensorTy.getShape()[0];

    auto i16 = rewriter.getI16Type();
    auto cstI16 = [&](int64_t v) -> Value {
      return arith::ConstantOp::create(rewriter, loc, i16,
                                       rewriter.getI16IntegerAttr((int16_t)v));
    };

    Value parity = cstI16(op.getFlag());
    Value arrCount = cstI16(op.getArriveCount());
    Value txCount = cstI16(op.getTransactionCount());
    Value exCount = cstI16(op.getExpectCount());

    SmallVector<Value> handles;
    handles.reserve(n);

    auto i64 = rewriter.getI64Type();
    for (int64_t i = 0; i < n; ++i) {
      Value h = mlir::xsmt_async::MBarrierAllocOp::create(
          rewriter, loc, i64, parity, arrCount, txCount, exCount);
      handles.push_back(h);
    }

    Value packed =
        tensor::FromElementsOp::create(rewriter, loc, tensorTy, handles);

    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct MBarrierSubviewOpLowering
    : public OpConversionPattern<mlir::xsmt::MBarrierSubviewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::xsmt::MBarrierSubviewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value barsTensor = adaptor.getMbarrier();
    Value idxVal = adaptor.getIndex();

    if (!idxVal.getType().isIndex()) {
      idxVal = arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), idxVal);
    }

    Value h = tensor::ExtractOp::create(rewriter, loc, barsTensor,
                                        ValueRange{idxVal});
    rewriter.replaceOp(op, h);
    return success();
  }
};

} // namespace

void mlir::triton::ViewOpPtrPatternConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ViewOpPtrPattern>(patterns.getContext());
  patterns.add<FoldAllocViewPtrToAlloc>(patterns.getContext());
  patterns.add<FuseTTSLoadAndViewToDescriptorLoadView>(patterns.getContext());
}

void mlir::triton::populateStructuredToMemrefConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<MakeTensorPtrConverter, XSMTAllocConverter, XSMTViewConverter,
               AllocCopiesOpLowering, BufferTensorViewOpLowering,
               MBarrierCopiesOpLowering, MBarrierSubviewOpLowering>(
      typeConverter, patterns.getContext());
  patterns.add<LoadConverter, StoreConverter>(typeConverter,
                                              patterns.getContext());
}
