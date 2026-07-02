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
        raise NotImplementedError(
            f"spine_raw.{self._name}() must only be called inside a "
            f"@spine_raw function body (used by SpineMLIRCodeGenerator)"
        )

    def __repr__(self):
        return f"spine_raw.{self._name}"


class _SpineRawRange:
    """Marker for spine_raw.range(...) — translated to scf.for bounds.

    Accepts range(stop) or range(start, stop, step) like the Python builtin;
    only ever evaluated by SpineMLIRCodeGenerator (raises if called directly).
    """
    def __call__(self, *args):
        raise NotImplementedError(
            "spine_raw.range() must only be used in a @spine_raw function body"
        )

    def __repr__(self):
        return "spine_raw.range"


# Public built-in objects
splat    = _SpineRawBuiltin("splat")     # splat(val, shape=[N]) → vector.splat
load_vec = _SpineRawBuiltin("load_vec")  # load_vec(ptr, idx, N, dtype) → vector.load
store_vec = _SpineRawBuiltin("store_vec")# store_vec(ptr, idx, vec) → vector.store
store_scalar = _SpineRawBuiltin("store_scalar")  # store_scalar(ptr, idx, val) → memref.store
fma      = _SpineRawBuiltin("fma")       # fma(a, b, c) → vector.fma
extf     = _SpineRawBuiltin("extf")      # extf(v, dtype) → arith.extf
reduce_add = _SpineRawBuiltin("reduce_add")  # reduce_add(vec) → vector.reduction <add>
matmul   = _SpineRawBuiltin("matmul")    # matmul(lhs, rhs, acc, m, n, k) → vector_ext.matmul
load_tile = _SpineRawBuiltin("load_tile")  # load_tile(ptr, row_base, row_stride, M, K, dtype) → MxK row-major vector<(M*K)>
pad_vec  = _SpineRawBuiltin("pad_vec")   # pad_vec(vec, total) → place vec at front of vector<total>, rest zero
extract_elem = _SpineRawBuiltin("extract_elem")  # extract_elem(vec, idx) → vector.extract
batch_macc = _SpineRawBuiltin("batch_macc")  # batch_macc(lhs_memref, rhs_vec, acc_vec) → vector_ext.batch_macc
view_2d  = _SpineRawBuiltin("view_2d")   # view_2d(ptr, rows, cols, dtype) → 2D strided memref view
load_2d  = _SpineRawBuiltin("load_2d")   # load_2d(ptr, rows, cols, dtype) → vector<rows x cols>
load_2d_at = _SpineRawBuiltin("load_2d_at")  # load_2d_at(ptr, elem_off, rows, cols, dtype)
load_2d_t = _SpineRawBuiltin("load_2d_t")  # load_2d_t(ptr, row_base, K, NB, M, dtype) → transposed vector<KxNB>
pack_2d_t = _SpineRawBuiltin("pack_2d_t")  # pack_2d_t(ptr, row_base, K, NB, M, dtype) → same but reads A row-major (unit inner stride) via alloca+linalg.generic transpose
alloc_tcm_2d = _SpineRawBuiltin("alloc_tcm_2d")  # alloc_tcm_2d(K, NB, dtype) → memref<K×NB> via memref.alloc (→ spine_thread_malloc/TCM)
pack_2d_t_into = _SpineRawBuiltin("pack_2d_t_into")  # pack_2d_t_into(buf, ptr, row_base, K, NB, M, dtype) → packs A^T into existing buf (no alloca)
free_tcm = _SpineRawBuiltin("free_tcm")  # free_tcm(buf) → memref.dealloc (→ spine_thread_free)
proton_mark = _SpineRawBuiltin("proton_mark")  # proton_mark(name, is_start) → rdtime + func.call @proton_record
splat_2d = _SpineRawBuiltin("splat_2d")  # splat_2d(val, rows, cols, dtype) → vector<rows x cols>
store_2d = _SpineRawBuiltin("store_2d")  # store_2d(ptr, rows, cols, vec) → 2D transfer_write
store_2d_at = _SpineRawBuiltin("store_2d_at")  # store_2d_at(ptr, elem_off, rows, cols, vec)
range    = _SpineRawRange()              # range(n) / range(start, stop, step) → scf.for bounds

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
#   vpack    : pack a B row-block into the packed_B scratch layout (写法3)
# ---------------------------------------------------------------------------
vconfig     = _SpineRawBuiltin("vconfig")      # vconfig(avl, sew_bytes) → fixed VL
vzero       = _SpineRawBuiltin("vzero")        # vzero(dtype) → vector<VL x dtype> zeros
vload       = _SpineRawBuiltin("vload")        # vload(ptr, idx_tuple[, stride]) → vector<VL x dtype>
vmacc       = _SpineRawBuiltin("vmacc")        # vmacc(acc, x, y) → widening fma accumulate
vreduce_sum = _SpineRawBuiltin("vreduce_sum")  # vreduce_sum(vec) → scalar
vstore      = _SpineRawBuiltin("vstore")       # vstore(ptr, idx_tuple, scalar) → memref.store
alloc       = _SpineRawBuiltin("alloc")        # alloc(shape_tuple, dtype) → memref.alloc
vpack       = _SpineRawBuiltin("vpack")        # vpack(src, src_idx, dst, dst_shape) → pack rows

# ---------------------------------------------------------------------------
# Document-facing sugar: dtype names and the `mem` / `index` / `raw_kernel`
# helpers so a kernel can be written close to the feishu 3.3 surface syntax.
# dtype constants are plain MLIR element-type strings.
# ---------------------------------------------------------------------------
f16 = "f16"
f32 = "f32"
bf16 = "bf16"

