//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/XSMT/IR/XSMTOps.cpp.inc"

namespace mlir {
namespace xsmt {

using namespace mlir;
using namespace mlir::triton;

static int64_t ceilDivI64(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

void ViewOp::build(OpBuilder &builder, OperationState &state, Value base,
                   ValueRange offsets, ArrayRef<int32_t> shape,
                   ArrayRef<int32_t> micro_size) {

  Type baseType = base.getType();

  RankedTensorType baseRankedType = cast<RankedTensorType>(baseType);
  Type elementType = baseRankedType.getElementType();
  bool allOnes = llvm::all_of(micro_size, [](int32_t v) { return v == 1; });
  bool allZeros = llvm::all_of(micro_size, [](int32_t v) { return v == 0; });

  SmallVector<int64_t> resultShape;
  std::vector<int32_t> actualMicroSize = micro_size;

  if (allOnes) {
    resultShape = {static_cast<int64_t>(shape[0]),
                   static_cast<int64_t>(shape[1])};
  } else if (allZeros) {
    auto baseShape = baseRankedType.getShape();
    if (baseShape.size() < 2) {
      llvm_unreachable(
          "Base tensor must have at least 2 dimensions for allZeros mode");
    }

    actualMicroSize = {static_cast<int32_t>(baseShape[baseShape.size() - 2]),
                       static_cast<int32_t>(baseShape[baseShape.size() - 1])};

    resultShape = {ceilDivI64(shape[0], actualMicroSize[0]),
                   ceilDivI64(shape[1], actualMicroSize[1]), actualMicroSize[0],
                   actualMicroSize[1]};
  } else {
    resultShape = {ceilDivI64(shape[0], micro_size[0]),
                   ceilDivI64(shape[1], micro_size[1]),
                   static_cast<int64_t>(micro_size[0]),
                   static_cast<int64_t>(micro_size[1])};
  }

  auto resultType = RankedTensorType::get(resultShape, elementType);

  build(builder, state, resultType, base, offsets,
        builder.getDenseI32ArrayAttr(shape),
        builder.getDenseI32ArrayAttr(actualMicroSize));
}

LogicalResult ViewOp::verify() {
  auto shapeAttr = getShape();
  if (getOffsets().size() != shapeAttr.size()) {
    return emitOpError("number of offsets (")
           << getOffsets().size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }
  auto microSizeAttr = getMicroSize();
  if (microSizeAttr.size() != shapeAttr.size()) {
    return emitOpError("micro_size dimensions (")
           << microSizeAttr.size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }

  return success();
}

void ViewPtrOp::build(OpBuilder &builder, OperationState &state, Value base,
                      ValueRange offsets, ArrayRef<int32_t> shape,
                      ArrayRef<int32_t> micro_size) {

  Type baseType = base.getType();
  auto ptrType = cast<PointerType>(baseType);
  RankedTensorType baseRankedType =
      cast<RankedTensorType>(ptrType.getPointeeType());
  Type elementType = baseRankedType.getElementType();
  unsigned addrSpace = ptrType.getAddressSpace();

  bool allOnes = llvm::all_of(micro_size, [](int32_t v) { return v == 1; });
  bool allZeros = llvm::all_of(micro_size, [](int32_t v) { return v == 0; });

  SmallVector<int64_t> resultShape;
  std::vector<int32_t> actualMicroSize = micro_size;

  if (allOnes) {
    resultShape = {static_cast<int64_t>(shape[0]),
                   static_cast<int64_t>(shape[1])};
  } else if (allZeros) {
    auto baseShape = baseRankedType.getShape();
    if (baseShape.size() < 2) {
      llvm_unreachable(
          "Base tensor must have at least 2 dimensions for allZeros mode");
    }
    actualMicroSize = {static_cast<int32_t>(baseShape[baseShape.size() - 2]),
                       static_cast<int32_t>(baseShape[baseShape.size() - 1])};
    resultShape = {ceilDivI64(shape[0], actualMicroSize[0]),
                   ceilDivI64(shape[1], actualMicroSize[1]), actualMicroSize[0],
                   actualMicroSize[1]};
  } else {
    resultShape = {ceilDivI64(shape[0], micro_size[0]),
                   ceilDivI64(shape[1], micro_size[1]),
                   static_cast<int64_t>(micro_size[0]),
                   static_cast<int64_t>(micro_size[1])};
  }

  auto resultRankedType = RankedTensorType::get(resultShape, elementType);
  Type resultType = PointerType::get(resultRankedType, addrSpace);
  build(builder, state, resultType, base, offsets,
        builder.getDenseI32ArrayAttr(shape),
        builder.getDenseI32ArrayAttr(actualMicroSize));
}

LogicalResult ViewPtrOp::verify() {
  auto shapeAttr = getShape();
  if (getOffsets().size() != shapeAttr.size()) {
    return emitOpError("number of offsets (")
           << getOffsets().size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }
  auto microSizeAttr = getMicroSize();
  if (microSizeAttr.size() != shapeAttr.size()) {
    return emitOpError("micro_size dimensions (")
           << microSizeAttr.size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }

  return success();
}

void DescriptorLoadViewOp::build(OpBuilder &builder, OperationState &state,
                                 Value base, ValueRange offsets,
                                 ArrayRef<int32_t> shape,
                                 ArrayRef<int32_t> micro_size) {
  auto pointerType = cast<mlir::triton::PointerType>(base.getType());
  auto pointeeType = pointerType.getPointeeType();

  Type elementType = pointeeType;
  while (auto tensorType = dyn_cast<RankedTensorType>(elementType)) {
    elementType = tensorType.getElementType();
  }

  if (isa<RankedTensorType>(elementType)) {
    llvm::report_fatal_error("Cannot determine scalar element type for tensor");
  }

  SmallVector<int64_t> resultShape;
  resultShape.push_back(ceilDivI64(shape[0], micro_size[0]));
  resultShape.push_back(ceilDivI64(shape[1], micro_size[1]));
  resultShape.push_back(micro_size[0]);
  resultShape.push_back(micro_size[1]);

  auto resultType = RankedTensorType::get(resultShape, elementType);

  return build(builder, state, resultType, base, offsets,
               builder.getDenseI32ArrayAttr(shape),
               builder.getDenseI32ArrayAttr(micro_size));
}

void BufferTensorViewOp::build(OpBuilder &builder, OperationState &state,
                               Value buffer, Value bufferIdx) {
  auto bufferType = cast<BufferType>(buffer.getType());
  auto shape = bufferType.getShape();

  llvm::SmallVector<int64_t> resultShape;
  if (bufferType.getCopies() == 0) {
    if (shape.size() == 1) {
      resultShape = {1};
    } else {
      resultShape.assign(shape.begin() + 1, shape.end());
    }
  } else {
    resultShape.assign(shape.begin() + 1, shape.end());
  }

  auto tensorType =
      RankedTensorType::get(resultShape, bufferType.getElementType());

  auto resultType = triton::PointerType::get(tensorType, 1);

  state.addTypes(resultType);
  state.addOperands({buffer, bufferIdx});
}

} // namespace xsmt
} // namespace mlir
