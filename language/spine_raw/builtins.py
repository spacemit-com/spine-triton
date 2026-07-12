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


# Public built-in objects
proton_mark = _SpineRawBuiltin("proton_mark")  # proton_mark(name, is_start) → rdtime + func.call @proton_record
range = _SpineRawRange()  # range(n) / range(start, stop, step) → scf.for bounds

# ---------------------------------------------------------------------------
# svector-level markers (feishu 3.3 mv 示例). Fixed-VL eDSL that maps document
# names to already-verified vector/arith/memref primitives.
#   vconfig  : vconfig(avl, lmul) → VLMAX const (SEW from dtype); avl deferred
#   vzero    : vector.broadcast 0.0 -> vector<VL x dtype>
#   vload    : transfer_read a VL-length vector (1D idx, or 2D idx + row stride)
#   vmacc    : widening multiply-accumulate  acc += extf(x) * extf(y)
#   vreduce_sum : vector.reduction <add> -> scalar
#   vstore   : store a scalar to memref[idx]
#   alloc    : memref.alloc N-D scratch (写法3 packed_B)
#   pack     : pack a B row-block into the packed_B scratch layout (写法3)
#   vpack    : vpack(v, group_len) → vector_ext.group_interleave → 多条 smt.vpack.vv (cube 交织)
#   vmadot   : vmadot(acc, x, y) → vector_ext.cross_batch_matmul → 多条 smt.vfwmadot (批量 cube 叉乘)
#   vshape   : vshape(v, shape) → vector.shape_cast (reshape)
#   vbroadcast: vbroadcast(v, n) → vector.broadcast (广播维)
# ---------------------------------------------------------------------------
vconfig = _SpineRawBuiltin("vconfig")  # vconfig(avl, lmul) → VL = min(avl, VLMAX), VLMAX = lmul × VLEN / SEW (SPEC §6.1)
vzero = _SpineRawBuiltin("vzero")  # vzero(dtype) → vector<VL x dtype> zeros
vload = _SpineRawBuiltin("vload")  # vload(ptr, idx_tuple[, stride]) → vector<VL x dtype>
vmacc = _SpineRawBuiltin("vmacc")  # vmacc(acc, x, y) → widening fma accumulate
vreduce_sum = _SpineRawBuiltin("vreduce_sum")  # vreduce_sum(vec) → scalar
vstore = _SpineRawBuiltin("vstore")  # vstore(ptr, idx_tuple, scalar | vec) → memref.store / transfer_write
alloc = _SpineRawBuiltin("alloc")  # alloc(shape_tuple, dtype) → memref.alloc
pack = _SpineRawBuiltin("pack")  # pack(src, src_idx, dst, dst_shape) → pack rows (写法3)
vpack = _SpineRawBuiltin("vpack")  # vpack(v, group_len) → vector_ext.group_interleave → 多条 smt.vpack.vv (cube 交织, vector<b×N>→<(b/2)×2N>)
vmadot = _SpineRawBuiltin("vmadot")  # vmadot(acc, x, y) → vector_ext.cross_batch_matmul → 多条 smt.vfwmadot (批量 cube 叉乘)
vshape = _SpineRawBuiltin("vshape")  # vshape(v, shape) → vector.shape_cast (同 numel reshape)
vbroadcast = _SpineRawBuiltin("vbroadcast")  # vbroadcast(v, n) → vector.broadcast: vector<64> → vector<n×64> (广播维)
# §6.4 逐元素具名函数(算术运算符直接用 Python 操作符, 无需 marker)
vmin = _SpineRawBuiltin("vmin")  # vmin(a, b) → 逐元素 min → arith.minimumf / minsi
vmax = _SpineRawBuiltin("vmax")  # vmax(a, b) → 逐元素 max → arith.maximumf / maxsi
sqrt = _SpineRawBuiltin("sqrt")  # sqrt(a) → √a → math.sqrt
rsqrt = _SpineRawBuiltin("rsqrt")  # rsqrt(a) → 1/√a → math.rsqrt
abs = _SpineRawBuiltin("abs")  # abs(a) → |a| → math.absf / absi  # noqa: A001 (shadows builtin intentionally)
cast = _SpineRawBuiltin("cast")  # cast(a, dtype) → 类型转换 → arith.extf/truncf/sitofp/fptosi/extsi/trunci
select = _SpineRawBuiltin("select")  # select(m, a, b) → a if m else b → arith.select (§6.8 回退)

# ---------------------------------------------------------------------------
# Document-facing sugar: dtype names and the `mem` / `index` / `raw_kernel`
# helpers so a kernel can be written close to the feishu 3.3 surface syntax.
# dtype constants are plain MLIR element-type strings.
# ---------------------------------------------------------------------------
f16 = "f16"
f32 = "f32"
bf16 = "bf16"
