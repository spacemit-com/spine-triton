#include "include/triton-shared/Dialect/TLE/IR/TLEDialect.h"
#include "include/triton-shared/Dialect/TLE/IR/TLEOps.h"
#include "include/triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "include/triton-shared/Dialect/XSMT/IR/XSMTOps.h"
#include "include/triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "include/triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"
#include "ir.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <iostream>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ir;
using namespace mlir;
namespace xsmt = mlir::xsmt;
namespace xsmt_async = mlir::xsmt_async;
namespace tle = mlir::tle;

void init_triton_xsmt_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls
      ->def("create_annotation",
            [](TritonOpBuilder &self, Value &ptr, const std::string &attrKey,
               Attribute &attrVal) {
              auto annotationOp = self.create<xsmt::AnnotationOp>(ptr);
              annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                                    attrVal);
            })
      .def("create_descriptor_load",
           [](TritonOpBuilder &self, Value &base,
              std::vector<Value> &offsets) -> Value {
             auto AdvanceOp =
                 self.create<triton::AdvanceOp>(base.getType(), base, offsets);
             auto pointeeType = cast<mlir::triton::PointerType>(base.getType())
                                    .getPointeeType();
             auto resultType = dyn_cast<RankedTensorType>(pointeeType);
             int rank = resultType.getRank();
             std::vector<int32_t> boundary_check;
             for (int i = 0; i < rank; ++i) {
               boundary_check.push_back(i);
             }
             auto LoadOp = self.create<triton::LoadOp>(
                 AdvanceOp.getResult(), boundary_check, std::nullopt,
                 triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
                 false);
             return LoadOp;
           })
      .def("create_descriptor_load_to_destination",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
              Value &destination) {
             auto AdvanceOp =
                 self.create<triton::AdvanceOp>(base.getType(), base, offsets);
             auto pointeeType = cast<mlir::triton::PointerType>(base.getType())
                                    .getPointeeType();
             auto resultType = dyn_cast<RankedTensorType>(pointeeType);
             int rank = resultType.getRank();
             std::vector<int32_t> boundary_check;
             for (int i = 0; i < rank; ++i) {
               boundary_check.push_back(i);
             }
             auto LoadOp = self.create<triton::LoadOp>(
                 AdvanceOp.getResult(), boundary_check, std::nullopt,
                 triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
                 false);
             self.create<mlir::triton::StoreOp>(
                 destination, LoadOp, boundary_check,
                 triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
           })
      .def(
          "create_pack",
          [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
             std::vector<int32_t> &shape, std::vector<int32_t> &packed_size,
             std::optional<Value> destination) -> Value {
            if (destination.has_value())
              return self.create<xsmt::PackOp>(
                  base, offsets, destination.value(), shape, packed_size);
            return self.create<xsmt::PackOp>(base, offsets, shape, packed_size);
          },
          py::arg("base"), py::arg("offsets"), py::arg("shape"),
          py::arg("packed_size"), py::arg("destination") = py::none())
      .def(
          "create_unpack",
          [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
             std::vector<int32_t> &shape,
             std::optional<Value> destination) -> Value {
            if (destination.has_value())
              return self.create<xsmt::UnpackOp>(base, offsets,
                                                 destination.value(), shape);
            return self.create<xsmt::UnpackOp>(base, offsets, shape);
          },
          py::arg("base"), py::arg("offsets"), py::arg("shape"),
          py::arg("destination") = py::none())
      .def(
          "create_repack",
          [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
             std::vector<int32_t> &shape, std::vector<int32_t> &packed_size,
             std::optional<Value> destination) -> Value {
            if (destination.has_value())
              return self.create<xsmt::RepackOp>(
                  base, offsets, destination.value(), shape, packed_size);
            return self.create<xsmt::RepackOp>(base, offsets, shape,
                                               packed_size);
          },
          py::arg("base"), py::arg("offsets"), py::arg("shape"),
          py::arg("packed_size"), py::arg("destination") = py::none())
      .def("create_subview",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
              std::vector<int32_t> &shape) -> Value {
             return self.create<xsmt::SubviewOp>(base, offsets, shape);
           })
      .def("create_subview_pack",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
              std::vector<int32_t> &shape,
              std::vector<int32_t> &packed_size) -> Value {
             return self.create<xsmt::SubviewPackOp>(base, offsets, shape,
                                                     packed_size);
           })
      .def("create_alloc",
           [](TritonOpBuilder &self, std::vector<int32_t> &shape,
              mlir::Type type, std::string storage) -> Value {
             if (shape.empty()) {
               throw std::runtime_error("alloc shape cannot be empty");
             }
             auto op = self.create<xsmt::AllocOp>(type, shape, storage);
             return op;
           })
      .def("create_alloc_copies",
           [](TritonOpBuilder &self, std::vector<int64_t> &shape,
              mlir::Type elementType, std::string storage) -> mlir::Value {
             if (shape.empty())
               throw std::runtime_error("alloc_copies shape cannot be empty");
             auto op =
                 self.create<xsmt::AllocCopiesOp>(shape, elementType, storage);
             return op;
           })
      .def("create_mmt4d",
           [](TritonOpBuilder &self, Value &a, Value &b,
              std::optional<Value> c = std::nullopt) -> Value {
             auto aType = cast<RankedTensorType>(a.getType());
             auto bType = cast<RankedTensorType>(b.getType());

             assert(aType.getRank() == 4 && "A must be 4D packed tensor");
             assert(bType.getRank() == 4 && "B must be 4D packed tensor");

             auto aShape = aType.getShape();
             auto bShape = bType.getShape();

             SmallVector<int64_t> outputShape;

             if (aShape[1] == bShape[0] && aShape[3] == bShape[2]) {
               outputShape = {
                   aShape[0],
                   bShape[1],
                   aShape[2],
                   bShape[3],
               };
               auto resultType =
                   RankedTensorType::get(outputShape, aType.getElementType());

               auto perm = std::vector<int>{1, 0, 3, 2};
               auto transbOp = self.create<mlir::triton::TransOp>(b, perm);
               mlir::Value transbValue = transbOp->getResult(0);

               mlir::Value cValue;
               if (c.has_value()) {
                 cValue = *c;
               } else {
                 cValue = Value();
               }

               return self.create<xsmt::MMT4DOp>(resultType, a, transbValue,
                                                 cValue);
             } else if (aShape[1] == bShape[1] && aShape[3] == bShape[3]) {
               outputShape = {
                   aShape[0],
                   bShape[0],
                   aShape[2],
                   bShape[2],
               };
               auto resultType =
                   RankedTensorType::get(outputShape, aType.getElementType());

               mlir::Value cValue;
               if (c.has_value()) {
                 cValue = *c;
               } else {
                 cValue = Value();
               }

               return self.create<xsmt::MMT4DOp>(resultType, a, b, cValue);
             } else {
               throw std::runtime_error("Unsupported packing shapes");
             }
           })
      .def("create_mbarrier",
           [](TritonOpBuilder &self, Value &flag, Value &atc, Value &tc,
              Value &exp) -> Value {
             auto barrierType = self.getBuilder().getI64Type();
             return self.create<mlir::xsmt_async::MBarrierAllocOp>(
                 barrierType, flag, atc, tc, exp);
           })
      .def("create_barrier_arrive",
           [](TritonOpBuilder &self, Value &bar) {
             self.create<mlir::xsmt_async::MBarrierArriveOp>(bar);
           })
      .def("create_barrier_wait",
           [](TritonOpBuilder &self, Value &bar, Value &flag, Value &exp) {
             self.create<mlir::xsmt_async::MBarrierWaitOp>(bar, flag, exp);
           })
      .def("create_get_num_of_thread",
           [](TritonOpBuilder &self) { self.create<xsmt::GetThreadOp>(); })
      .def("create_global_mbarrier",
           [](TritonOpBuilder &self, Value &id) -> Value {
             auto barrierType = self.getBuilder().getI64Type();
             return self.create<xsmt::GlobalMBarrierInitOp>(barrierType, id);
           })
      .def("create_barrier_set_expect",
           [](TritonOpBuilder &self, Value &bar, Value &exp) {
             self.create<xsmt::BarrierSetEepectOp>(bar, exp);
           })
      .def("create_smt_buffer_type",
           [](TritonOpBuilder &self, std::vector<int64_t> shape,
              Type &elementType, int copies, std::string storageKind) -> Type {
             return xsmt::BufferType::get(shape, elementType, copies,
                                          storageKind);
           })
      .def("create_buffer_tensor_subview",
           [](TritonOpBuilder &self, Value buffer, Value bufferIdx) -> Value {
             return self.create<xsmt::BufferTensorViewOp>(buffer, bufferIdx);
           })
      .def("get_mbarrier_type",
           [](TritonOpBuilder &self, int copies) -> mlir::Type {
             auto *ctx = self.getBuilder().getContext();
             return mlir::xsmt::MBarrierType::get(ctx, copies);
           })
      .def("create_mbarrier_copies",
           [](TritonOpBuilder &self, int numCopies, int flag, int arriveCount,
              int transactionCount, int expectCount) -> mlir::Value {
             auto *ctx = self.getBuilder().getContext();

             auto resultTy = mlir::xsmt::MBarrierType::get(ctx, numCopies);

             auto op = self.create<mlir::xsmt::MBarrierCopiesOp>(
                 resultTy, numCopies, flag, arriveCount, transactionCount,
                 expectCount);

             return op.getResult();
           })
      .def("create_mbarrier_subview",
           [](TritonOpBuilder &self, mlir::Value mbarrierHandle,
              mlir::Value indexValue) -> mlir::Value {
             auto i64Type = self.getBuilder().getI64Type();
             auto op = self.create<mlir::xsmt::MBarrierSubviewOp>(
                 i64Type, mbarrierHandle, indexValue);

             return op.getResult();
           })
      .def("create_i64_constant",
           [](TritonOpBuilder &self, int64_t value) -> mlir::Value {
             auto i64Type = self.getBuilder().getI64Type();
             auto attr = self.getBuilder().getI64IntegerAttr(value);
             return self.create<mlir::arith::ConstantOp>(i64Type, attr)
                 .getResult();
           });
}

// ============================================================================
// TLE (Triton Language Extension) IR bindings
// ============================================================================
void init_triton_tle_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls
      ->def(
          "create_extract_tile",
          [](TritonOpBuilder &self, Value &input, Value &index,
             std::vector<int64_t> &tileShape) -> Value {
            auto op = self.create<tle::ExtractTileOp>(input, index, tileShape);
            return op.getResult();
          },
          py::arg("input"), py::arg("index"), py::arg("tileShape"),
          "Create extract_tile operation")
      .def(
          "create_insert_tile",
          [](TritonOpBuilder &self, Value &input, Value &tile,
             Value &index) -> Value {
            auto op = self.create<tle::InsertTileOp>(input, tile, index);
            return op.getResult();
          },
          py::arg("input"), py::arg("tile"), py::arg("index"),
          "Create insert_tile operation")
      .def(
          "create_tle_dsl_region",
          [](TritonOpBuilder &self, const std::string &fn_name,
             const std::string &raw_linalg, std::vector<Value> &inputs) {
            // Parse the raw-kernel MLIR text once, here at build time, and hang
            // the body in tle.dsl_region's real region — so the TTIR stays
            // readable (no escaped raw_linalg string attr). Custom ops (e.g.
            // vector_ext.matmul) parse in generic form thanks to the context's
            // allow-unregistered flag (set in load_dialects).
            auto &builder = self.getBuilder();
            mlir::MLIRContext *ctx = builder.getContext();
            mlir::ParserConfig config(ctx, /*verifyAfterParse=*/false);
            mlir::OwningOpRef<mlir::ModuleOp> rawMod =
                mlir::parseSourceString<mlir::ModuleOp>(raw_linalg, config);
            if (!rawMod)
              throw std::runtime_error(
                  "create_tle_dsl_region: failed to parse raw_linalg text");
            mlir::func::FuncOp rawFunc;
            rawMod->walk([&](mlir::func::FuncOp f) {
              if (!f.empty()) {
                rawFunc = f;
                return mlir::WalkResult::interrupt();
              }
              return mlir::WalkResult::advance();
            });
            if (!rawFunc)
              throw std::runtime_error(
                  "create_tle_dsl_region: no func.func in raw_linalg");

            auto fnAttr = builder.getStringAttr(fn_name);
            SmallVector<Value> operands(inputs.begin(), inputs.end());
            auto op = self.create<tle::DSLRegionOp>(operands, fnAttr);

            // Build the region block with the raw fn's arg types, then clone the
            // fn body into it (func.return stays — lowering turns it into
            // spine_ext.return later).
            mlir::Region &body = op.getBody();
            mlir::Block *block = new mlir::Block();
            body.push_back(block);
            for (mlir::Type paramTy : rawFunc.getArgumentTypes())
              block->addArgument(paramTy, op.getLoc());
            mlir::IRMapping mapping;
            for (auto [fArg, bArg] :
                 llvm::zip(rawFunc.getArguments(), block->getArguments()))
              mapping.map(fArg, bArg);
            mlir::OpBuilder bodyBuilder(block, block->end());
            for (mlir::Operation &inner : rawFunc.getBody().front()) {
              // func.return can't live under tle.dsl_region (its verifier wants
              // parent func.func). Emit generic spine_ext.return instead — the
              // TLEToLinalg pattern expects that terminator anyway.
              if (mlir::isa<mlir::func::ReturnOp>(inner)) {
                mlir::OperationState retState(inner.getLoc(),
                                              "spine_ext.return");
                bodyBuilder.create(retState);
              } else {
                bodyBuilder.clone(inner, mapping);
              }
            }
          },
          py::arg("fn_name"), py::arg("raw_linalg"), py::arg("inputs"),
          "Create tle.dsl_region — spine_raw.call() TTIR op")
      .def(
          "create_tle_dsl_region_direct",
          [](TritonOpBuilder &self, const std::string &fn_name,
             std::vector<Value> &inputs, std::vector<Type> &paramTypes,
             py::function bodyBuilder) {
            // Builder-direct counterpart of create_tle_dsl_region: no MLIR text
            // round trip. codegen.py builds the raw fn body straight into
            // dsl_region's region via bodyBuilder, using the same builder
            // instance — no parseSourceString, no raw_linalg string.
            auto &builder = self.getBuilder();
            auto fnAttr = builder.getStringAttr(fn_name);
            SmallVector<Value> operands(inputs.begin(), inputs.end());
            auto op = self.create<tle::DSLRegionOp>(operands, fnAttr);

            mlir::Region &body = op.getBody();
            mlir::Block *block = new mlir::Block();
            body.push_back(block);
            for (mlir::Type paramTy : paramTypes)
              block->addArgument(paramTy, op.getLoc());

            mlir::OpBuilder::InsertionGuard guard(builder);
            self.setInsertionPointToStart(*block);
            std::vector<Value> blockArgs(block->getArguments().begin(),
                                          block->getArguments().end());
            bodyBuilder(std::ref(self), blockArgs);

            // bodyBuilder emits the raw fn's ops but not the terminator (there's
            // no func.return here to translate) — always close with
            // spine_ext.return, matching the text path's substitution.
            mlir::OperationState retState(op.getLoc(), "spine_ext.return");
            builder.create(retState);
          },
          py::arg("fn_name"), py::arg("inputs"), py::arg("param_types"),
          py::arg("body_builder"),
          "Create tle.dsl_region by invoking body_builder(builder, block_args) "
          "directly — no MLIR text parse round trip");
}

// ============================================================================
// Spine Raw IR Builder Bindings
// 为 spine_raw codegen 提供标准的 MLIR builder API，替代字符串拼接
// ============================================================================

// Helper: parse MLIR type string
static Type parseTypeString(OpBuilder &builder, const std::string &typeStr) {
  MLIRContext *ctx = builder.getContext();
  return mlir::parseType(typeStr, ctx);
}

void init_triton_spine_raw_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();

  // ========================================================================
  // Type utilities
  // ========================================================================
  builder_cls
      ->def("get_index_type",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getIndexType();
           })
      .def("get_i32_type",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI32Type();
           })
      .def("get_i64_type",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getI64Type();
           })
      .def("get_f16_type",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getF16Type();
           })
      .def("get_f32_type",
           [](TritonOpBuilder &self) -> Type {
             return self.getBuilder().getF32Type();
           })
      .def("parse_type",
           [](TritonOpBuilder &self, const std::string &typeStr) -> Type {
             return parseTypeString(self.getBuilder(), typeStr);
           },
           py::arg("type_str"),
           "Parse MLIR type string (e.g., 'vector<32xf32>', 'memref<?xf16>')")

  // ========================================================================
  // Arith dialect - 常量
  // ========================================================================
      .def("create_arith_constant_index",
           [](TritonOpBuilder &self, int64_t value) -> Value {
             auto type = self.getBuilder().getIndexType();
             auto attr = self.getBuilder().getIntegerAttr(type, value);
             return self.create<arith::ConstantOp>(type, attr).getResult();
           },
           py::arg("value"))
      .def("create_arith_constant_int",
           [](TritonOpBuilder &self, int64_t value, Type intType) -> Value {
             auto attr = self.getBuilder().getIntegerAttr(intType, value);
             return self.create<arith::ConstantOp>(intType, attr).getResult();
           },
           py::arg("value"), py::arg("int_type"))
      .def("create_arith_constant_float",
           [](TritonOpBuilder &self, double value, Type floatType) -> Value {
             auto attr = self.getBuilder().getFloatAttr(floatType, value);
             return self.create<arith::ConstantOp>(floatType, attr).getResult();
           },
           py::arg("value"), py::arg("float_type"))

  // ========================================================================
  // Arith dialect - 整数运算
  // ========================================================================
      .def("create_arith_addi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::AddIOp>(lhs, rhs).getResult();
           },
           py::arg("lhs"), py::arg("rhs"))
      .def("create_arith_subi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::SubIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_muli",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MulIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_divsi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::DivSIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_divui",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::DivUIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_remsi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::RemSIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_minsi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MinSIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_maxsi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MaxSIOp>(lhs, rhs).getResult();
           })

  // ========================================================================
  // Arith dialect - 浮点运算
  // ========================================================================
      .def("create_arith_addf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::AddFOp>(lhs, rhs).getResult();
           })
      .def("create_arith_subf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::SubFOp>(lhs, rhs).getResult();
           })
      .def("create_arith_mulf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MulFOp>(lhs, rhs).getResult();
           })
      .def("create_arith_divf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::DivFOp>(lhs, rhs).getResult();
           })
      .def("create_arith_negf",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<arith::NegFOp>(operand).getResult();
           })
      .def("create_arith_minimumf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MinimumFOp>(lhs, rhs).getResult();
           })
      .def("create_arith_maximumf",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::MaximumFOp>(lhs, rhs).getResult();
           })

  // ========================================================================
  // Arith dialect - 类型转换
  // ========================================================================
      .def("create_arith_extf",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::ExtFOp>(targetType, input).getResult();
           },
           py::arg("input"), py::arg("target_type"))
      .def("create_arith_truncf",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::TruncFOp>(targetType, input).getResult();
           })
      .def("create_arith_extsi",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::ExtSIOp>(targetType, input).getResult();
           })
      .def("create_arith_trunci",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::TruncIOp>(targetType, input).getResult();
           })
      .def("create_arith_sitofp",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::SIToFPOp>(targetType, input).getResult();
           })
      .def("create_arith_fptosi",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::FPToSIOp>(targetType, input).getResult();
           })
      .def("create_arith_index_cast",
           [](TritonOpBuilder &self, Value input, Type targetType) -> Value {
             return self.create<arith::IndexCastOp>(targetType, input).getResult();
           })

  // ========================================================================
  // Arith dialect - 比较
  // ========================================================================
      .def("create_arith_cmpi",
           [](TritonOpBuilder &self, const std::string &predicate,
              Value lhs, Value rhs) -> Value {
             arith::CmpIPredicate pred;
             if (predicate == "eq") pred = arith::CmpIPredicate::eq;
             else if (predicate == "ne") pred = arith::CmpIPredicate::ne;
             else if (predicate == "slt") pred = arith::CmpIPredicate::slt;
             else if (predicate == "sle") pred = arith::CmpIPredicate::sle;
             else if (predicate == "sgt") pred = arith::CmpIPredicate::sgt;
             else if (predicate == "sge") pred = arith::CmpIPredicate::sge;
             else throw std::runtime_error("Unknown cmpi predicate: " + predicate);
             return self.create<arith::CmpIOp>(pred, lhs, rhs).getResult();
           },
           py::arg("predicate"), py::arg("lhs"), py::arg("rhs"))
      .def("create_arith_cmpf",
           [](TritonOpBuilder &self, const std::string &predicate,
              Value lhs, Value rhs) -> Value {
             arith::CmpFPredicate pred;
             if (predicate == "oeq") pred = arith::CmpFPredicate::OEQ;
             else if (predicate == "one") pred = arith::CmpFPredicate::ONE;
             else if (predicate == "olt") pred = arith::CmpFPredicate::OLT;
             else if (predicate == "ole") pred = arith::CmpFPredicate::OLE;
             else if (predicate == "ogt") pred = arith::CmpFPredicate::OGT;
             else if (predicate == "oge") pred = arith::CmpFPredicate::OGE;
             else throw std::runtime_error("Unknown cmpf predicate: " + predicate);
             return self.create<arith::CmpFOp>(pred, lhs, rhs).getResult();
           })
      .def("create_arith_select",
           [](TritonOpBuilder &self, Value condition, Value trueValue,
              Value falseValue) -> Value {
             return self.create<arith::SelectOp>(condition, trueValue, falseValue)
                 .getResult();
           })

  // ========================================================================
  // Arith dialect - 位运算
  // ========================================================================
      .def("create_arith_andi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::AndIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_ori",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::OrIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_xori",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::XOrIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_shli",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::ShLIOp>(lhs, rhs).getResult();
           })
      .def("create_arith_shrsi",
           [](TritonOpBuilder &self, Value lhs, Value rhs) -> Value {
             return self.create<arith::ShRSIOp>(lhs, rhs).getResult();
           })

  // ========================================================================
  // Math dialect
  // ========================================================================
      .def("create_math_fma",
           [](TritonOpBuilder &self, Value a, Value b, Value c) -> Value {
             return self.create<math::FmaOp>(a, b, c).getResult();
           },
           py::arg("a"), py::arg("b"), py::arg("c"))
      .def("create_math_sqrt",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::SqrtOp>(operand).getResult();
           })
      .def("create_math_rsqrt",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::RsqrtOp>(operand).getResult();
           })
      .def("create_math_exp",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::ExpOp>(operand).getResult();
           })
      .def("create_math_exp2",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::Exp2Op>(operand).getResult();
           })
      .def("create_math_log",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::LogOp>(operand).getResult();
           })
      .def("create_math_log2",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::Log2Op>(operand).getResult();
           })
      .def("create_math_absf",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::AbsFOp>(operand).getResult();
           })
      .def("create_math_absi",
           [](TritonOpBuilder &self, Value operand) -> Value {
             return self.create<math::AbsIOp>(operand).getResult();
           })

  // ========================================================================
  // Vector dialect - 基础操作
  // ========================================================================
      .def("create_vector_splat",
           [](TritonOpBuilder &self, Value input, Type vectorType) -> Value {
             // splat = broadcast a scalar to all lanes (SplatOp 已废弃, 用 BroadcastOp)
             return self.create<vector::BroadcastOp>(vectorType, input).getResult();
           },
           py::arg("input"), py::arg("vector_type"))
      .def("create_vector_broadcast",
           [](TritonOpBuilder &self, Value source, Type destType) -> Value {
             return self.create<vector::BroadcastOp>(destType, source).getResult();
           },
           py::arg("source"), py::arg("dest_type"))
      .def("create_vector_shape_cast",
           [](TritonOpBuilder &self, Value source, Type resultType) -> Value {
             return self.create<vector::ShapeCastOp>(resultType, source).getResult();
           })
      .def("create_vector_step",
           [](TritonOpBuilder &self, Type resultType) -> Value {
             // vector.step : vector<Nxindex> → [0, 1, .., N-1] (iota).
             // Used for index-tracking reductions (argmax/argmin).
             return self.create<vector::StepOp>(cast<VectorType>(resultType));
           },
           py::arg("result_type"))

  // ========================================================================
  // Vector dialect - Load/Store
  // ========================================================================
      .def("create_vector_load",
           [](TritonOpBuilder &self, Type vectorType, Value base,
              std::vector<Value> indices) -> Value {
             return self.create<vector::LoadOp>(vectorType, base, indices).getResult();
           },
           py::arg("vector_type"), py::arg("base"), py::arg("indices"))
      .def("create_vector_store",
           [](TritonOpBuilder &self, Value valueToStore, Value base,
              std::vector<Value> indices) {
             self.create<vector::StoreOp>(valueToStore, base, indices);
           })
      .def("create_vector_transfer_read",
           [](TritonOpBuilder &self, Type vectorType, Value source,
              std::vector<Value> indices, Value padding,
              std::optional<std::vector<bool>> inBounds = std::nullopt) -> Value {
             SmallVector<bool> inBoundsVec;
             if (inBounds.has_value()) {
               inBoundsVec = SmallVector<bool>(inBounds->begin(), inBounds->end());
             } else {
               // 默认全部 in_bounds=true
               auto vecTy = cast<VectorType>(vectorType);
               inBoundsVec = SmallVector<bool>(vecTy.getRank(), true);
             }
             return self.create<vector::TransferReadOp>(
                 cast<VectorType>(vectorType), source, indices, padding,
                 ArrayRef<bool>(inBoundsVec)).getResult();
           },
           py::arg("vector_type"), py::arg("source"), py::arg("indices"),
           py::arg("padding"), py::arg("in_bounds") = py::none())
      .def("create_vector_transfer_write",
           [](TritonOpBuilder &self, Value vector, Value dest,
              std::vector<Value> indices,
              std::optional<std::vector<bool>> inBounds = std::nullopt) {
             SmallVector<bool> inBoundsVec;
             if (inBounds.has_value()) {
               inBoundsVec = SmallVector<bool>(inBounds->begin(), inBounds->end());
             } else {
               auto vecTy = cast<VectorType>(vector.getType());
               inBoundsVec = SmallVector<bool>(vecTy.getRank(), true);
             }
             self.create<vector::TransferWriteOp>(
                 vector, dest, indices, ArrayRef<bool>(inBoundsVec));
           },
           py::arg("vector"), py::arg("dest"), py::arg("indices"),
           py::arg("in_bounds") = py::none())

  // ========================================================================
  // Vector dialect - Reduction
  // ========================================================================
      .def("create_vector_reduction",
           [](TritonOpBuilder &self, const std::string &kind, Value source) -> Value {
             vector::CombiningKind combKind;
             if (kind == "add") combKind = vector::CombiningKind::ADD;
             else if (kind == "mul") combKind = vector::CombiningKind::MUL;
             else if (kind == "minf") combKind = vector::CombiningKind::MINIMUMF;
             else if (kind == "maxf") combKind = vector::CombiningKind::MAXIMUMF;
             else throw std::runtime_error("Unknown reduction kind: " + kind);
             return self.create<vector::ReductionOp>(combKind, source).getResult();
           },
           py::arg("kind"), py::arg("source"))

  // ========================================================================
  // SCF dialect - 控制流
  // ========================================================================
      .def("create_scf_for",
           [](TritonOpBuilder &self, Value lb, Value ub, Value step,
              std::vector<Value> iterArgs,
              py::function bodyBuilder) -> std::vector<Value> {
             // 创建 scf.for
             ValueRange iterArgsRange(iterArgs);
             auto forOp = self.create<scf::ForOp>(lb, ub, step, iterArgsRange);

             // 设置 body builder 的插入点
             OpBuilder::InsertionGuard guard(self.getBuilder());
             Block *bodyBlock = forOp.getBody();

             // ForOp::build 在 initArgs 为空且无 bodyBuilder 时会自动 ensureTerminator
             // 塞一个空 scf.yield;若不清除,后面我们再 create<YieldOp> 会得到两个 yield
             // (触发 'scf.yield must be the last operation')。统一由本函数负责 yield,
             // 先擦掉自动终结符。
             if (!bodyBlock->empty() &&
                 bodyBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
               bodyBlock->back().erase();

             self.getBuilder().setInsertionPointToEnd(bodyBlock);

             // 调用 Python 传入的 body builder
             // Python 侧需要返回 yield 的值列表
             Value iv = forOp.getInductionVar();
             std::vector<Value> regionIterArgs(
                 bodyBlock->getArguments().begin() + 1,
                 bodyBlock->getArguments().end());

             py::object yieldValsObj = bodyBuilder(std::ref(self), iv, regionIterArgs);
             std::vector<Value> yieldVals = yieldValsObj.cast<std::vector<Value>>();

             // 创建 scf.yield(本函数唯一 yield 来源)
             self.getBuilder().setInsertionPointToEnd(bodyBlock);
             self.create<scf::YieldOp>(yieldVals);

             // 返回 for 的结果
             return std::vector<Value>(forOp.getResults().begin(),
                                      forOp.getResults().end());
           },
           py::arg("lb"), py::arg("ub"), py::arg("step"),
           py::arg("iter_args"), py::arg("body_builder"))
      .def("create_scf_yield",
           [](TritonOpBuilder &self, std::vector<Value> results) {
             self.create<scf::YieldOp>(results);
           })

  // ========================================================================
  // MemRef dialect
  // ========================================================================
      .def("create_memref_alloc",
           [](TritonOpBuilder &self, Type memrefType,
              std::optional<std::vector<Value>> dynamicSizes = std::nullopt,
              std::optional<int64_t> alignment = std::nullopt) -> Value {
             SmallVector<Value> dynSizes;
             if (dynamicSizes.has_value()) {
               dynSizes = SmallVector<Value>(dynamicSizes->begin(), dynamicSizes->end());
             }
             // 必须 cast<MemRefType>:参数是 Type,若直接传给 create<AllocOp> 会匹配
             // 泛型 build(TypeRange, ValueRange, attrs) 重载,不设 operandSegmentSizes
             // → 'operand count does not match total size in operandSegmentSizes'。
             // cast 后命中 AllocOp::build(MemRefType, ValueRange dynamicSizes, ...)。
             auto op = self.create<memref::AllocOp>(
                 cast<MemRefType>(memrefType), dynSizes);
             if (alignment.has_value()) {
               op->setAttr("alignment",
                          self.getBuilder().getI64IntegerAttr(alignment.value()));
             }
             return op.getResult();
           },
           py::arg("memref_type"), py::arg("dynamic_sizes") = py::none(),
           py::arg("alignment") = py::none())
      .def("create_memref_load",
           [](TritonOpBuilder &self, Value memref, std::vector<Value> indices) -> Value {
             return self.create<memref::LoadOp>(memref, indices).getResult();
           })
      .def("create_memref_store",
           [](TritonOpBuilder &self, Value value, Value memref,
              std::vector<Value> indices) {
             self.create<memref::StoreOp>(value, memref, indices);
           })
      .def("create_memref_reinterpret_cast",
           [](TritonOpBuilder &self, Type resultType, Value source,
              std::vector<Value> offsets, std::vector<Value> sizes,
              std::vector<Value> strides) -> Value {
             // ReinterpretCastOp 需要 OpFoldResult（Value 或 静态 int）。
             // 全部动态时，static 数组用 ShapedType::kDynamic 占位。
             auto toOFR = [](const std::vector<Value> &vals)
                 -> SmallVector<OpFoldResult> {
               SmallVector<OpFoldResult> result;
               for (Value v : vals)
                 result.push_back(v);
               return result;
             };
             return self.create<memref::ReinterpretCastOp>(
                 cast<MemRefType>(resultType), source,
                 toOFR(offsets)[0], toOFR(sizes), toOFR(strides)).getResult();
           },
           py::arg("result_type"), py::arg("source"), py::arg("offsets"),
           py::arg("sizes"), py::arg("strides"))
      .def("create_memref_reinterpret_cast_mixed",
           [](TritonOpBuilder &self, Type resultType, Value source,
              py::list offsets, py::list sizes, py::list strides) -> Value {
             // 混合 static/dynamic 版本:py::list 每个元素是 int(静态,进 static_*
             // 数组)或 Value(动态,static 位填 kDynamic)。用于结果类型含静态维
             // (如 memref<?x64xf16>)时,static_sizes 必须与结果类型逐维一致,
             // 否则报 'expected result type with size = dynamic instead of N'。
             auto &b = self.getBuilder();
             auto toOFR = [&b](py::list items) -> SmallVector<OpFoldResult> {
               SmallVector<OpFoldResult> result;
               for (py::handle it : items) {
                 if (py::isinstance<py::int_>(it))
                   result.push_back(b.getIndexAttr(it.cast<int64_t>()));
                 else
                   result.push_back(it.cast<Value>());
               }
               return result;
             };
             return self.create<memref::ReinterpretCastOp>(
                 cast<MemRefType>(resultType), source,
                 toOFR(offsets)[0], toOFR(sizes), toOFR(strides)).getResult();
           },
           py::arg("result_type"), py::arg("source"), py::arg("offsets"),
           py::arg("sizes"), py::arg("strides"))
      .def("create_memref_collapse_shape",
           [](TritonOpBuilder &self, Value src,
              const std::vector<std::vector<int64_t>> &reassociation) -> Value {
             SmallVector<ReassociationIndices> reassoc;
             for (const auto &group : reassociation) {
               reassoc.push_back(ReassociationIndices(group.begin(), group.end()));
             }
             return self.create<memref::CollapseShapeOp>(src, reassoc).getResult();
           },
           py::arg("src"), py::arg("reassociation"))
      .def("create_memref_cast",
           [](TritonOpBuilder &self, Type targetType, Value source) -> Value {
             return self.create<memref::CastOp>(targetType, source).getResult();
           })

  // ========================================================================
  // Tensor dialect (for bufferization path)
  // ========================================================================
      .def("create_tensor_empty",
           [](TritonOpBuilder &self, Type tensorType,
              std::optional<std::vector<Value>> dynamicSizes = std::nullopt) -> Value {
             SmallVector<Value> dynSizes;
             if (dynamicSizes.has_value()) {
               dynSizes = SmallVector<Value>(dynamicSizes->begin(), dynamicSizes->end());
             }
             return self.create<tensor::EmptyOp>(tensorType, dynSizes).getResult();
           },
           py::arg("tensor_type"), py::arg("dynamic_sizes") = py::none())
      .def("create_tensor_insert_slice",
           [](TritonOpBuilder &self, Value source, Value dest,
              std::vector<Value> offsets, std::vector<Value> sizes,
              std::vector<Value> strides) -> Value {
             return self.create<tensor::InsertSliceOp>(
                 source, dest, offsets, sizes, strides).getResult();
           })
      .def("create_tensor_insert_slice_mixed",
           [](TritonOpBuilder &self, Value source, Value dest,
              py::list offsets, py::list sizes, py::list strides) -> Value {
             // 混合 static/dynamic:int→static OFR, Value→dynamic。source 含静态维
             // (如 tensor<?x64xf16>)时,insert_slice static_sizes 须逐维匹配 source,
             // 否则报 'expected type to be tensor<?x?xf16>' rank/size mismatch。
             auto &b = self.getBuilder();
             auto toOFR = [&b](py::list items) -> SmallVector<OpFoldResult> {
               SmallVector<OpFoldResult> result;
               for (py::handle it : items) {
                 if (py::isinstance<py::int_>(it))
                   result.push_back(b.getIndexAttr(it.cast<int64_t>()));
                 else
                   result.push_back(it.cast<Value>());
               }
               return result;
             };
             return self.create<tensor::InsertSliceOp>(
                 source, dest, toOFR(offsets), toOFR(sizes),
                 toOFR(strides)).getResult();
           })
      .def("create_tensor_collapse_shape",
           [](TritonOpBuilder &self, Value src,
              const std::vector<std::vector<int64_t>> &reassociation) -> Value {
             SmallVector<ReassociationIndices> reassoc;
             for (const auto &group : reassociation) {
               reassoc.push_back(ReassociationIndices(group.begin(), group.end()));
             }
             return self.create<tensor::CollapseShapeOp>(src, reassoc).getResult();
           })

  // ========================================================================
  // Linalg dialect
  // ========================================================================
      .def("create_linalg_fill",
           [](TritonOpBuilder &self, Value value, Value output) -> Value {
             return self.create<linalg::FillOp>(value, output).getResult(0);
           })
      .def("create_linalg_pack",
           [](TritonOpBuilder &self, Value source, Value dest, Value paddingValue,
              const std::vector<int64_t> &innerTiles,
              const std::vector<int64_t> &outerDimsPerm,
              const std::vector<int64_t> &innerDimsPos) -> Value {
             // 根据 MLIR 签名: create(builder, loc, type, source, dest, padding_value,
             //                        outer_dims_perm, inner_dims_pos, inner_tiles, static_inner_tiles)
             auto &builder = self.getBuilder();
             SmallVector<int64_t> staticInnerTiles(innerTiles.begin(), innerTiles.end());
             SmallVector<Value> innerTilesValues;  // 空的动态 tiles

             return self.create<linalg::PackOp>(
                 dest.getType(), source, dest, paddingValue,
                 outerDimsPerm, innerDimsPos,
                 innerTilesValues, staticInnerTiles).getResult();
           },
           py::arg("source"), py::arg("dest"), py::arg("padding_value"),
           py::arg("inner_tiles"), py::arg("outer_dims_perm"),
           py::arg("inner_dims_pos"))

  // ========================================================================
  // Bufferization dialect
  // ========================================================================
      .def("create_bufferization_to_tensor",
           [](TritonOpBuilder &self, Value memref, Type tensorType) -> Value {
             auto op = self.create<bufferization::ToTensorOp>(tensorType, memref);
             // Match text path "bufferization.to_tensor ... restrict"
             op->setAttr("restrict", self.getBuilder().getUnitAttr());
             return op.getResult();
           })
      .def("create_bufferization_to_memref",
           [](TritonOpBuilder &self, Value tensor, Type memrefType) -> Value {
             // ToMemrefOp 已改名 ToBufferOp（新版 MLIR），需显式给出 memref 结果类型
             return self.create<bufferization::ToBufferOp>(memrefType, tensor)
                 .getResult();
           })

  // ========================================================================
  // Generic (unregistered) op builder — for ops in dialects that are never
  // registered/loaded (e.g. vector_ext.*), which parse only in generic form
  // under the context's allowUnregisteredDialects flag (see load_dialects).
  // int_attrs covers the only attr shape spine_raw's generic ops need today
  // (vector_ext.cross_batch_matmul's m/n/k, vector_ext.group_interleave's
  // groupLen — all i64 integer attrs).
  // ========================================================================
      .def("create_generic_op",
           [](TritonOpBuilder &self, const std::string &opName,
              std::vector<Value> operands,
              std::map<std::string, int64_t> intAttrs,
              std::vector<Type> resultTypes) -> std::vector<Value> {
             auto &builder = self.getBuilder();
             mlir::OperationState state(self.getLastLoc(), opName);
             state.addOperands(operands);
             state.addTypes(resultTypes);
             for (auto &[name, val] : intAttrs)
               state.addAttribute(name, builder.getI64IntegerAttr(val));
             mlir::Operation *op = builder.create(state);
             return std::vector<Value>(op->getResults().begin(),
                                       op->getResults().end());
           },
           py::arg("op_name"), py::arg("operands"), py::arg("int_attrs"),
           py::arg("result_types"),
           "Create an unregistered/generic-form op (e.g. vector_ext.*)");
}

void init_triton_spine_triton(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::xsmt::XSMTDialect, mlir::xsmt_async::XSMTAsyncDialect,
                    tensor::TensorDialect, mlir::triton::proton::ProtonDialect,
                    mlir::tle::TLEDialect,
                    // Payload dialects for tle.dsl_region's real region body,
                    // so the raw-kernel MLIR text parses in-process (custom ops
                    // like vector_ext.matmul stay generic via the context's
                    // allow-unregistered flag).
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
                    mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                    mlir::math::MathDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::ptr::PtrDialect>();
    // Registering func dialect above makes TTIR's InlinerPass query func's
    // DialectInlinerInterface; that interface lives in a separate extension
    // that must be registered explicitly, or the inliner aborts with
    // "interface promised by dialect 'func' but never implemented".
    mlir::func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);
    context.allowUnregisteredDialects();
    context.loadAllAvailableDialects();
  });

  init_triton_xsmt_ir(m.def_submodule("xsmt_ir"));
  init_triton_tle_ir(m.def_submodule("tle_ir"));
  init_triton_spine_raw_ir(m.def_submodule("spine_raw_ir"));
}
