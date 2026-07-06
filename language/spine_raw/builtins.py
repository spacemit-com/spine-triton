# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""spine_raw built-in functions.

These are Python-side marker objects. When called inside a @spine_raw function
body, they get translated to MLIR ops by SpineMLIRCodeGenerator.

At Python runtime (outside codegen), they raise NotImplementedError so
accidental direct calls are caught early.
"""
from __future__ import annotations


class _SpineRawBuiltin:

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"spine_raw.{self._name}() must only be called inside a "
                                  f"@spine_raw function body (used by SpineMLIRCodeGenerator)")

    def __repr__(self):
        return f"spine_raw.{self._name}"


class _SpineRawRange:
    """Marker for spine_raw.range(...) — translated to scf.for bounds.

    Accepts range(stop) or range(start, stop, step) like the Python builtin;
    only ever evaluated by SpineMLIRCodeGenerator (raises if called directly).
    """

    def __call__(self, *args):
        raise NotImplementedError("spine_raw.range() must only be used in a @spine_raw function body")

    def __repr__(self):
        return "spine_raw.range"


# Public built-in objects (batch_macc / vfwmacc mv path)
batch_macc = _SpineRawBuiltin("batch_macc")  # batch_macc(lhs_memref, rhs_vec, acc_vec) → vector_ext.batch_macc
view_2d = _SpineRawBuiltin("view_2d")  # view_2d(ptr, rows, cols, dtype) → 2D strided memref view
load_2d = _SpineRawBuiltin("load_2d")  # load_2d(ptr, rows, cols, dtype) → vector<rows x cols>
alloc_tcm_2d = _SpineRawBuiltin(
    "alloc_tcm_2d")  # alloc_tcm_2d(K, NB, dtype) → memref<K×NB> via memref.alloc (→ spine_thread_malloc/TCM)
pack_2d_t_into = _SpineRawBuiltin(
    "pack_2d_t_into")  # pack_2d_t_into(buf, ptr, row_base, K, NB, M, dtype) → packs A^T into existing buf (no alloca)
proton_mark = _SpineRawBuiltin("proton_mark")  # proton_mark(name, is_start) → rdtime + func.call @proton_record
splat_2d = _SpineRawBuiltin("splat_2d")  # splat_2d(val, rows, cols, dtype) → vector<rows x cols>
store_2d_at = _SpineRawBuiltin("store_2d_at")  # store_2d_at(ptr, elem_off, rows, cols, vec)
range = _SpineRawRange()  # range(n) / range(start, stop, step) → scf.for bounds

# ---------------------------------------------------------------------------
# svector-level markers (feishu 3.3 mv 示例). Fixed-VL eDSL that maps document
# names to already-verified vector/arith/memref primitives.
#   vconfig  : record active VL/SEW, return fixed VL constant (constexpr int)
#   vzero    : vector.broadcast 0.0 -> vector<VL x dtype>
#   vload    : transfer_read a VL-length vector (1D idx, or 2D idx + row stride)
#   vmacc    : widening multiply-accumulate  acc += extf(x) * extf(y)
#   vreduce_sum : vector.reduction <add> -> scalar
#   vstore   : store a scalar to memref[idx]
#   alloc    : memref.alloc N-D scratch (写法3 packed_B)
#   pack     : pack a B row-block into the packed_B scratch layout (写法3)
#   vpack    : vpack(a, b, group_len) → vector_ext.interleave → smt.vpack.vv (硬件 cube pack)
#   vmadot   : matrix-unit dot (写法4) -> vector_ext.matmul, 直接产出宽结果
# ---------------------------------------------------------------------------
vconfig = _SpineRawBuiltin("vconfig")  # vconfig(avl, sew_bytes) → fixed VL
vzero = _SpineRawBuiltin("vzero")  # vzero(dtype) → vector<VL x dtype> zeros
vload = _SpineRawBuiltin("vload")  # vload(ptr, idx_tuple[, stride]) → vector<VL x dtype>
vmacc = _SpineRawBuiltin("vmacc")  # vmacc(acc, x, y) → widening fma accumulate
vreduce_sum = _SpineRawBuiltin("vreduce_sum")  # vreduce_sum(vec) → scalar
vstore = _SpineRawBuiltin("vstore")  # vstore(ptr, idx_tuple, scalar | vec) → memref.store / transfer_write
alloc = _SpineRawBuiltin("alloc")  # alloc(shape_tuple, dtype) → memref.alloc
pack = _SpineRawBuiltin("pack")  # pack(src, src_idx, dst, dst_shape) → pack rows (写法3)
vpack = _SpineRawBuiltin("vpack")  # vpack(a, b, group_len) → vector_ext.interleave → smt.vpack.vv (硬件 cube pack)
vmadot = _SpineRawBuiltin("vmadot")  # vmadot(acc, x, y) → "vector_ext.matmul" (矩阵单元, 写法4)

# ---------------------------------------------------------------------------
# Document-facing sugar: dtype names and the `mem` / `index` / `raw_kernel`
# helpers so a kernel can be written close to the feishu 3.3 surface syntax.
# dtype constants are plain MLIR element-type strings.
# ---------------------------------------------------------------------------
f16 = "f16"
f32 = "f32"
bf16 = "bf16"
