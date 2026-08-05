"""3-stage mv: svector pre-scale -> call_intrinsic vfwmacc -> svector post-scale.

Mirrors test_mixed_syntax_three_layer.py's multi-stage pattern, applied to mv:

  stage 1  pre_scale_sv   : vec_s[k] = vec[k] * alpha   —— svector helpers
                            (tle.vload / tle.cast / tle.vstore + BinOp)
  stage 2  mv_vfwmacc_ci  : scores = Mat @ vec_s        —— tle.call_intrinsic
                            only. It's the ONE existing tle semantic: the
                            emitter spells llvm.load / llvm.vector.reduce.fadd /
                            llvm.store as NATIVE llvm ops, so ONLY
                            llvm.riscv.vfwmacc becomes a real llvm.call_intrinsic
                            in the IR (it alone has no native MLIR equivalent).
  stage 3  post_scale_sv  : out[n] = scores[n] * beta   —— svector helpers

WHY 3 STAGES: per design request, vfwmacc (the only op svector can't emit
directly — tle.vmacc lowers to vfwcvt+vfmul+vfadd, 3 ops) is isolated in its
own call_intrinsic kernel (sibling llvm.func, bypasses BufferDeallocation).
Stages 1/3 use svector helpers via @tle.raw_kernel (dsl_region inlined into
host func.func). The mixed-mode bridge (commit 3825395) injects the
sibling llvm.func + host llvm.call at the ll.mlir layer, so all 3 stages
compose in ONE launch.

Constraints: K % 64 == 0 (stage 1/2 full tiles), N % 8 == 0 (stage 2 8-row).
Run under pytest — `python file.py` re-triggers the do_not_specialize host
recompile quirk (see test_mixed_syntax_three_layer.py docstring).
"""
import time
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32

_ALPHA = 1.5    # pre-scale  factor (module-level → svector constexpr_float)
_BETA = 0.5     # post-scale factor


# ── stage 1: svector pre-scale  vec_s = vec * alpha ─────────────────────────
# f16 vec → cast f32 → mul alpha (broadcast) → cast f16 → store. K % 64 == 0.
@tle.raw_kernel
def pre_scale_svector(vec: tle.mem(f16), vec_s: tle.mem(f16, out=True), K: tle.index):
    nvl = tle.vconfig(-1, 1)
    Kfloor = (K // nvl) * nvl
    for ki in tle.range(0, Kfloor, nvl):
        v = tle.vload(vec, ki)                    # vector<64xf16>
        v_f = tle.cast(v, f32)                    # vector<64xf32>  (arith.extf)
        v_s = v_f * _ALPHA                        # vector<64xf32>  (arith.mulf, scalar bcast)
        v_s_h = tle.cast(v_s, f16)                # vector<64xf16>  (arith.truncf)
        tle.vstore(vec_s, ki, v_s_h)
    # tail loop: distinct names — reusing v/v_f/v_s/v_s_h would make the codegen
    # treat them as cross-loop iter_args referencing SSA from the closed main loop.
    for ki in tle.range(Kfloor, K, nvl):
        nvl = tle.vconfig(K - ki, 1)
        tv = tle.vload(vec, ki)
        tv_f = tle.cast(tv, f32)
        tv_s = tv_f * _ALPHA
        tv_s_h = tle.cast(tv_s, f16)
        tle.vstore(vec_s, ki, tv_s_h)


# ── stage 2: 8-row vfwmacc mv  scores = Mat @ vec_s ─────────────────────────
# Pure llvm-direct (sibling llvm.func, bypasses spine-opt bufferization). Every op
# is written with the SAME existing tle.call_intrinsic — no new eDSL primitives.
# The emitter renders llvm.load / llvm.vector.reduce.fadd / llvm.store as NATIVE
# llvm ops (→ vle / vfredusum / vse), so ONLY llvm.riscv.vfwmacc (6-arg policy
# form) stays a real llvm.call_intrinsic in the IR. 8-row unroll.
@tle.raw_kernel
def mv_vfwmacc_call_intrinsic(B: tle.mem(f16), A: tle.mem(f16),
                              C: tle.mem(f32, out=True),
                              K: tle.index, N: tle.index):
    vl = tle.llvm_const(64, "i64")
    zero = tle.llvm_const(0, "i64")
    eight = tle.llvm_const(8, "i64")
    zero_acc = tle.llvm_const("0.000000e+00", "vector<[4]xf32>")
    zero_f = tle.llvm_const("0.000000e+00", "f32")
    cbase = tle.llvm_base_ptr(C)
    abase = tle.llvm_base_ptr(A)
    bbase = tle.llvm_base_ptr(B)

    for ni in tle.range(zero, N, eight):
        acc0 = zero_acc
        acc1 = zero_acc
        acc2 = zero_acc
        acc3 = zero_acc
        acc4 = zero_acc
        acc5 = zero_acc
        acc6 = zero_acc
        acc7 = zero_acc
        for ki in tle.range(zero, K, vl):
            ga = tle.llvm_gep(abase, ki, "f16")
            va = tle.call_intrinsic("llvm.load", [ga], result_type="vector<[4]xf16>")
            off0 = ni * K + ki
            gb0 = tle.llvm_gep(bbase, off0, "f16")
            gb1 = tle.llvm_gep(bbase, (ni + 1) * K + ki, "f16")
            gb2 = tle.llvm_gep(bbase, (ni + 2) * K + ki, "f16")
            gb3 = tle.llvm_gep(bbase, (ni + 3) * K + ki, "f16")
            gb4 = tle.llvm_gep(bbase, (ni + 4) * K + ki, "f16")
            gb5 = tle.llvm_gep(bbase, (ni + 5) * K + ki, "f16")
            gb6 = tle.llvm_gep(bbase, (ni + 6) * K + ki, "f16")
            gb7 = tle.llvm_gep(bbase, (ni + 7) * K + ki, "f16")
            vb0 = tle.call_intrinsic("llvm.load", [gb0], result_type="vector<[4]xf16>")
            vb1 = tle.call_intrinsic("llvm.load", [gb1], result_type="vector<[4]xf16>")
            vb2 = tle.call_intrinsic("llvm.load", [gb2], result_type="vector<[4]xf16>")
            vb3 = tle.call_intrinsic("llvm.load", [gb3], result_type="vector<[4]xf16>")
            vb4 = tle.call_intrinsic("llvm.load", [gb4], result_type="vector<[4]xf16>")
            vb5 = tle.call_intrinsic("llvm.load", [gb5], result_type="vector<[4]xf16>")
            vb6 = tle.call_intrinsic("llvm.load", [gb6], result_type="vector<[4]xf16>")
            vb7 = tle.call_intrinsic("llvm.load", [gb7], result_type="vector<[4]xf16>")
            acc0 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc0, va, vb0, zero, vl, zero], result_type="vector<[4]xf32>")
            acc1 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc1, va, vb1, zero, vl, zero], result_type="vector<[4]xf32>")
            acc2 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc2, va, vb2, zero, vl, zero], result_type="vector<[4]xf32>")
            acc3 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc3, va, vb3, zero, vl, zero], result_type="vector<[4]xf32>")
            acc4 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc4, va, vb4, zero, vl, zero], result_type="vector<[4]xf32>")
            acc5 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc5, va, vb5, zero, vl, zero], result_type="vector<[4]xf32>")
            acc6 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc6, va, vb6, zero, vl, zero], result_type="vector<[4]xf32>")
            acc7 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc7, va, vb7, zero, vl, zero], result_type="vector<[4]xf32>")
        s0 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc0], result_type="f32")
        s1 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc1], result_type="f32")
        s2 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc2], result_type="f32")
        s3 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc3], result_type="f32")
        s4 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc4], result_type="f32")
        s5 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc5], result_type="f32")
        s6 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc6], result_type="f32")
        s7 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc7], result_type="f32")
        tle.call_intrinsic("llvm.store", [s0, tle.llvm_gep(cbase, ni, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s1, tle.llvm_gep(cbase, ni + 1, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s2, tle.llvm_gep(cbase, ni + 2, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s3, tle.llvm_gep(cbase, ni + 3, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s4, tle.llvm_gep(cbase, ni + 4, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s5, tle.llvm_gep(cbase, ni + 5, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s6, tle.llvm_gep(cbase, ni + 6, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s7, tle.llvm_gep(cbase, ni + 7, "f32")], result_type="()")



# ── stage 3(v2): call_intrinsic post-scale  out = scores * beta ─────────────
# Sibling llvm.func (LLVM-direct). vle f32 → llvm.fmul (beta splat) → vse.
# Fixed VL=8, N % 8 == 0. Bridge #2 in the fused single-launch host.
@tle.raw_kernel
def post_scale_call_intrinsic(scores: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    vl = tle.llvm_const(8, "i64")
    zero = tle.llvm_const(0, "i64")
    beta = tle.llvm_const("5.000000e-01", "vector<[8]xf32>")
    for i in tle.range(zero, N, vl):
        p = tle.llvm_poison("vector<[8]xf32>")
        gs = tle.llvm_gep(tle.llvm_base_ptr(scores), i, "f32")
        v = tle.call_intrinsic("llvm.riscv.vle", [p, gs, vl], result_type="vector<[8]xf32>")
        r = tle.call_intrinsic("llvm.fmul", [v, beta], result_type="vector<[8]xf32>")
        go = tle.llvm_gep(tle.llvm_base_ptr(out), i, "f32")
        tle.call_intrinsic("llvm.riscv.vse", [r, go, vl], result_type="()")


# ── stage 3 (svector): out = scores * beta ──────────────────────────────────
# f32 svector post-scale, placed AFTER the vfwmacc bridge in the fused host.
# This is the case the positional-anchor mechanism unlocks: pre-anchor, all
# bridges were forced before llvm.return, so a svector stage could never follow
# a bridge (it would read `scores` before the bridge wrote it). vconfig derives
# SEW from the f32 dtype, so VL here is half the f16 stage-1 VL.
@tle.raw_kernel
def post_scale_svector(scores: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    for ni in tle.range(0, Nfloor, nvl):
        v = tle.vload(scores, ni, dtype=f32)      # vector<VL x f32>
        v_s = v * _BETA                           # arith.mulf, scalar bcast
        tle.vstore(out, ni, v_s)
    for ni in tle.range(Nfloor, N, nvl):          # tail — distinct SSA names
        nvl = tle.vconfig(N - ni, 1)
        tv = tle.vload(scores, ni, dtype=f32)
        tv_s = tv * _BETA
        tle.vstore(out, ni, tv_s)


# ── single-launch fused host: svector → call_intrinsic → call_intrinsic ─────
# pre_scale_svector        → dsl_region inline (runs first in host body)
# mv_vfwmacc_call_intrinsic → bridge #1 (injected before llvm.return)
# post_scale_call_intrinsic → bridge #2 (injected after #1, before llvm.return)
# All three stages execute in ONE launch. Data flow:
#   vec → vec_s (inline) → scores (bridge #1) → out (bridge #2)
@triton.jit(do_not_specialize=["K", "N"])
def _mv_fused_host(Mat, vec, vec_s, scores, out, K, N):
    _sr_call(pre_scale_svector, outputs=[], inputs=[vec, vec_s, K])
    _sr_call(mv_vfwmacc_call_intrinsic, outputs=[], inputs=[Mat, vec_s, scores, K, N])
    _sr_call(post_scale_call_intrinsic, outputs=[], inputs=[scores, out, N])


# ── PARALLEL fused: program_id-partitioned call_intrinsic (grid=(N//8,)) ─────
# The single-launch fused host above runs grid=(1,) — one program does all N
# rows serially, which loses to multi-core svector style2 on large shapes.
# These variants partition the N rows across programs: each program handles ONE
# 8-row tile at row_base = program_id(0)*8. The sibling ABI now carries the 6
# trailing grid args (llvm_direct_text.emit_llvm_func_for_inline) and the bridge
# forwards the host's grid args (compiler._inject_mixed_llvm_llmlir), so
# tle.program_id(0) resolves inside the sibling llvm.func. Launch grid=(N//8,).
#   stage 1 pre_scale_svector : runs on every program, redundantly writes the
#     full vec_s (deterministic identical stores → benign; K-work ≪ N·K).
#   stage 2 mv_vfwmacc_ci_parallel : this program's 8 rows only.
#   stage 3 post_scale_ci_parallel : this program's 8 scores only.
# Data flow is per-program-local (each touches only its own rows) → race-free.
# BLK is a RUNTIME scalar (bridged as i64), swept for granularity tuning — each
# program handles BLK rows via an inner loop of 8-row sub-tiles. grid=(N//BLK,).
# Matching style2's block granularity is what closes the perf gap (BLOCK=8 alone
# spawns too many tiny programs; dispatch overhead dominates on large shapes).
# BLK % 8 == 0, N % BLK == 0.
@tle.raw_kernel
def mv_vfwmacc_ci_parallel(B: tle.mem(f16), A: tle.mem(f16),
                           C: tle.mem(f32, out=True),
                           K: tle.index, N: tle.index, BLK: tle.index):
    vl = tle.llvm_const(64, "i64")
    zero = tle.llvm_const(0, "i64")
    zero_acc = tle.llvm_const("0.000000e+00", "vector<[4]xf32>")
    zero_f = tle.llvm_const("0.000000e+00", "f32")
    cbase = tle.llvm_base_ptr(C)
    abase = tle.llvm_base_ptr(A)
    bbase = tle.llvm_base_ptr(B)

    row_base = tle.program_id(0) * BLK    # this program's BLK-row tile base
    row_end = row_base + BLK
    for ni in tle.range(row_base, row_end, 8):
        acc0 = zero_acc
        acc1 = zero_acc
        acc2 = zero_acc
        acc3 = zero_acc
        acc4 = zero_acc
        acc5 = zero_acc
        acc6 = zero_acc
        acc7 = zero_acc
        for ki in tle.range(zero, K, vl):
            ga = tle.llvm_gep(abase, ki, "f16")
            va = tle.call_intrinsic("llvm.load", [ga], result_type="vector<[4]xf16>")
            gb0 = tle.llvm_gep(bbase, ni * K + ki, "f16")
            gb1 = tle.llvm_gep(bbase, (ni + 1) * K + ki, "f16")
            gb2 = tle.llvm_gep(bbase, (ni + 2) * K + ki, "f16")
            gb3 = tle.llvm_gep(bbase, (ni + 3) * K + ki, "f16")
            gb4 = tle.llvm_gep(bbase, (ni + 4) * K + ki, "f16")
            gb5 = tle.llvm_gep(bbase, (ni + 5) * K + ki, "f16")
            gb6 = tle.llvm_gep(bbase, (ni + 6) * K + ki, "f16")
            gb7 = tle.llvm_gep(bbase, (ni + 7) * K + ki, "f16")
            vb0 = tle.call_intrinsic("llvm.load", [gb0], result_type="vector<[4]xf16>")
            vb1 = tle.call_intrinsic("llvm.load", [gb1], result_type="vector<[4]xf16>")
            vb2 = tle.call_intrinsic("llvm.load", [gb2], result_type="vector<[4]xf16>")
            vb3 = tle.call_intrinsic("llvm.load", [gb3], result_type="vector<[4]xf16>")
            vb4 = tle.call_intrinsic("llvm.load", [gb4], result_type="vector<[4]xf16>")
            vb5 = tle.call_intrinsic("llvm.load", [gb5], result_type="vector<[4]xf16>")
            vb6 = tle.call_intrinsic("llvm.load", [gb6], result_type="vector<[4]xf16>")
            vb7 = tle.call_intrinsic("llvm.load", [gb7], result_type="vector<[4]xf16>")
            acc0 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc0, va, vb0, zero, vl, zero], result_type="vector<[4]xf32>")
            acc1 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc1, va, vb1, zero, vl, zero], result_type="vector<[4]xf32>")
            acc2 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc2, va, vb2, zero, vl, zero], result_type="vector<[4]xf32>")
            acc3 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc3, va, vb3, zero, vl, zero], result_type="vector<[4]xf32>")
            acc4 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc4, va, vb4, zero, vl, zero], result_type="vector<[4]xf32>")
            acc5 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc5, va, vb5, zero, vl, zero], result_type="vector<[4]xf32>")
            acc6 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc6, va, vb6, zero, vl, zero], result_type="vector<[4]xf32>")
            acc7 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc7, va, vb7, zero, vl, zero], result_type="vector<[4]xf32>")
        s0 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc0], result_type="f32")
        s1 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc1], result_type="f32")
        s2 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc2], result_type="f32")
        s3 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc3], result_type="f32")
        s4 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc4], result_type="f32")
        s5 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc5], result_type="f32")
        s6 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc6], result_type="f32")
        s7 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc7], result_type="f32")
        tle.call_intrinsic("llvm.store", [s0, tle.llvm_gep(cbase, ni, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s1, tle.llvm_gep(cbase, ni + 1, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s2, tle.llvm_gep(cbase, ni + 2, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s3, tle.llvm_gep(cbase, ni + 3, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s4, tle.llvm_gep(cbase, ni + 4, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s5, tle.llvm_gep(cbase, ni + 5, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s6, tle.llvm_gep(cbase, ni + 6, "f32")], result_type="()")
        tle.call_intrinsic("llvm.store", [s7, tle.llvm_gep(cbase, ni + 7, "f32")], result_type="()")


@tle.raw_kernel
def post_scale_ci_parallel(scores: tle.mem(f32), out: tle.mem(f32, out=True),
                           N: tle.index, BLK: tle.index):
    vl = tle.llvm_const(8, "i64")
    beta = tle.llvm_const("5.000000e-01", "vector<[8]xf32>")
    base = tle.program_id(0) * BLK
    bend = base + BLK
    for i in tle.range(base, bend, vl):
        p = tle.llvm_poison("vector<[8]xf32>")
        gs = tle.llvm_gep(tle.llvm_base_ptr(scores), i, "f32")
        v = tle.call_intrinsic("llvm.riscv.vle", [p, gs, vl], result_type="vector<[8]xf32>")
        r = tle.call_intrinsic("llvm.fmul", [v, beta], result_type="vector<[8]xf32>")
        go = tle.llvm_gep(tle.llvm_base_ptr(out), i, "f32")
        tle.call_intrinsic("llvm.riscv.vse", [r, go, vl], result_type="()")


# grid=(N//BLK,): one BLK-row tile per program → multi-core parallel.
@triton.jit(do_not_specialize=["K", "N", "BLK"])
def _mv_fused_host_par(Mat, vec, vec_s, scores, out, K, N, BLK):
    _sr_call(pre_scale_svector, outputs=[], inputs=[vec, vec_s, K])
    _sr_call(mv_vfwmacc_ci_parallel, outputs=[], inputs=[Mat, vec_s, scores, K, N, BLK])
    _sr_call(post_scale_ci_parallel, outputs=[], inputs=[scores, out, N, BLK])


def _run_fused_par_correctness(N, K, BLK=8):
    assert K % 64 == 0 and N % 8 == 0, "fused_par: K%64==0, N%8==0"
    assert BLK % 8 == 0 and N % BLK == 0, "fused_par: BLK%8==0, N%BLK==0"
    Np = ((N + 7) // 8) * 8
    torch.manual_seed(0)
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)

    _mv_fused_host_par[(N // BLK,)](
        Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N, BLK)

    got = out[:N]
    ref = torch.mv(Mat.float(), (vec.float() * _ALPHA).half().float()) * _BETA
    max_diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"
    return max_diff


# ---------------------------------------------------------------------------
# svector style2 baseline (for perf comparison) — copied from test_raw_mv_mixed
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mv_block_style2(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True),
                    K: tle.index, row_base: tle.index, row_end: tle.index):
    nvl = tle.vconfig(-1, 1)
    Kfloor = (K // nvl) * nvl
    for ni in tle.range(row_base, row_end, 4):
        acc0 = tle.vzero(f32)
        acc1 = tle.vzero(f32)
        acc2 = tle.vzero(f32)
        acc3 = tle.vzero(f32)
        for ki in tle.range(0, Kfloor, nvl):
            va = tle.vload(A, ki)
            vb0 = tle.vload(B, ni * K + ki)
            vb1 = tle.vload(B, (ni + 1) * K + ki)
            vb2 = tle.vload(B, (ni + 2) * K + ki)
            vb3 = tle.vload(B, (ni + 3) * K + ki)
            acc0 = tle.vmacc(acc0, vb0, va)
            acc1 = tle.vmacc(acc1, vb1, va)
            acc2 = tle.vmacc(acc2, vb2, va)
            acc3 = tle.vmacc(acc3, vb3, va)
        for ki in tle.range(Kfloor, K, nvl):
            nvl = tle.vconfig(K - ki, 1)
            ta = tle.vload(A, ki)
            tb0 = tle.vload(B, ni * K + ki)
            tb1 = tle.vload(B, (ni + 1) * K + ki)
            tb2 = tle.vload(B, (ni + 2) * K + ki)
            tb3 = tle.vload(B, (ni + 3) * K + ki)
            acc0 = tle.vmacc(acc0, tb0, ta)
            acc1 = tle.vmacc(acc1, tb1, ta)
            acc2 = tle.vmacc(acc2, tb2, ta)
            acc3 = tle.vmacc(acc3, tb3, ta)
        tle.sstore(C, ni, tle.vreduce_sum(acc0))
        tle.sstore(C, ni + 1, tle.vreduce_sum(acc1))
        tle.sstore(C, ni + 2, tle.vreduce_sum(acc2))
        tle.sstore(C, ni + 3, tle.vreduce_sum(acc3))


@triton.jit(do_not_specialize=["K", "N"])
def _mv_sv_host_style2(B, A, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)
    _sr_call(mv_block_style2, outputs=[], inputs=[B, A, C, K, row_base, row_end])


# ---------------------------------------------------------------------------
# Correctness + perf
# ---------------------------------------------------------------------------
_SHAPES = [(8, 64), (16, 128), (32, 256), (64, 512), (128, 256),
           (64, 64), (32, 128), (16, 64)]



def _run_fused_correctness(N, K):
    assert K % 64 == 0 and N % 8 == 0, "fused: K%64==0, N%8==0"
    Np = ((N + 7) // 8) * 8
    torch.manual_seed(0)
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)

    _mv_fused_host[(1,)](
        Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N)

    got = out[:N]
    ref = torch.mv(Mat.float(), (vec.float() * _ALPHA).half().float()) * _BETA
    max_diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"
    return max_diff


# ── fused host with svector stage 3 AFTER the bridge (anchor feature) ────────
# pre_scale_svector         → dsl_region inline (svector, before the bridge)
# mv_vfwmacc_call_intrinsic → bridge (injected AT its positional anchor)
# post_scale_svector        → dsl_region inline (svector, AFTER the bridge)
# Impossible before positional anchors: the bridge used to be forced to
# llvm.return, so no svector could read `scores` after it. Now the bridge lands
# at its anchor and the stage-3 svector body follows it in program order.
@triton.jit(do_not_specialize=["K", "N"])
def _mv_fused_host_sv3(Mat, vec, vec_s, scores, out, K, N):
    _sr_call(pre_scale_svector, outputs=[], inputs=[vec, vec_s, K])
    _sr_call(mv_vfwmacc_call_intrinsic, outputs=[], inputs=[Mat, vec_s, scores, K, N])
    _sr_call(post_scale_svector, outputs=[], inputs=[scores, out, N])


def _run_fused_sv3_correctness(N, K):
    assert K % 64 == 0 and N % 8 == 0, "fused_sv3: K%64==0, N%8==0"
    Np = ((N + 7) // 8) * 8
    torch.manual_seed(0)
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)

    _mv_fused_host_sv3[(1,)](
        Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N)

    got = out[:N]
    ref = torch.mv(Mat.float(), (vec.float() * _ALPHA).half().float()) * _BETA
    max_diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"
    return max_diff


def _measure_fused(N, K, iters=50, warmup=5):
    Np = ((N + 7) // 8) * 8
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)
    for _ in range(warmup):
        _mv_fused_host[(1,)](
            Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N)
    t0 = time.perf_counter()
    for _ in range(iters):
        _mv_fused_host[(1,)](
            Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def _measure_fused_sv3(N, K, iters=50, warmup=5):
    """Measure fused_sv3: svector pre + vfwmacc bridge + svector post."""
    Np = ((N + 7) // 8) * 8
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)
    for _ in range(warmup):
        _mv_fused_host_sv3[(1,)](
            Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N)
    t0 = time.perf_counter()
    for _ in range(iters):
        _mv_fused_host_sv3[(1,)](
            Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def _measure_fused_par(N, K, BLK=8, iters=50, warmup=5):
    assert BLK % 8 == 0 and N % BLK == 0, "fused_par: BLK%8==0, N%BLK==0"
    Np = ((N + 7) // 8) * 8
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)
    grid = (N // BLK,)
    for _ in range(warmup):
        _mv_fused_host_par[grid](
            Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N, BLK)
    t0 = time.perf_counter()
    for _ in range(iters):
        _mv_fused_host_par[grid](
            Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N, BLK)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def _measure_svector(N, K, BLOCK=4, iters=50, warmup=5):
    Np = ((N + BLOCK - 1) // BLOCK) * BLOCK
    B = torch.randn(N, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    C = torch.empty(Np, dtype=torch.float32)
    grid = (Np // BLOCK,)
    for _ in range(warmup):
        _mv_sv_host_style2[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N, BLOCK=BLOCK)
    t0 = time.perf_counter()
    for _ in range(iters):
        _mv_sv_host_style2[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N, BLOCK=BLOCK)
    t1 = time.perf_counter()
    return (t1 - t0) / iters



@pytest.mark.parametrize("N, K", _SHAPES)
def test_mv_fused_single_launch_correctness(N, K):
    _run_fused_correctness(N, K)


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mv_fused_parallel_correctness(N, K):
    _run_fused_par_correctness(N, K)


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mv_fused_svector_after_bridge_correctness(N, K):
    """svector stage 3 placed AFTER the vfwmacc bridge (positional anchor)."""
    _run_fused_sv3_correctness(N, K)


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mv_fused_perf_vs_svector(N, K):
    """Single-launch fused variants vs pure svector style2.

    fused       : grid=(1,)       — one program handles all N rows serially
    fused_par   : grid=(N//BLK,) — one BLK-row tile per program, multi-core

    Both fused variants do EXTRA work (pre/post scale) the svector baseline
    doesn't. fused_par recovers throughput at larger shapes via multi-core.
    """
    BLK = 8
    t_sv = _measure_svector(N, K, BLOCK=4)
    t_f  = _measure_fused(N, K)
    t_fp = _measure_fused_par(N, K, BLK=BLK)
    gf_sv = 2.0 * N * K / t_sv / 1e9
    gf_f  = 2.0 * N * K / t_f  / 1e9
    gf_fp = 2.0 * N * K / t_fp / 1e9
    print(f"N={N:4d} K={K:4d}  svector={t_sv*1e6:8.1f}us ({gf_sv:.2f}GF)  "
          f"fused={t_f*1e6:8.1f}us ({gf_f:.2f}GF)  "
          f"fused_par[BLK={BLK}]={t_fp*1e6:8.1f}us ({gf_fp:.2f}GF)")


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mv_fused_sv3_perf_vs_ci(N, K):
    """Compare post_scale implementations: svector vs call_intrinsic.

    fused (ci post)  : svector pre + bridge mv + call_intrinsic post
    fused_sv3 (sv post) : svector pre + bridge mv + svector post

    Both use the same pre_scale and mv kernels, only differ in post_scale.
    """
    t_ci = _measure_fused(N, K)
    t_sv = _measure_fused_sv3(N, K)
    gf_ci = 2.0 * N * K / t_ci / 1e9
    gf_sv = 2.0 * N * K / t_sv / 1e9
    ratio = t_ci / t_sv
    print(f"N={N:4d} K={K:4d}  ci_post={t_ci*1e6:8.1f}us ({gf_ci:.2f}GF)  "
          f"sv_post={t_sv*1e6:8.1f}us ({gf_sv:.2f}GF)  ratio={ratio:.3f}x")


if __name__ == "__main__":
    print("=== single-launch fused mv: svector pre + vfwmacc + ci post ===")
    print("=== correctness: fused grid=(1,) ===")
    for N, K in _SHAPES:
        try:
            md = _run_fused_correctness(N, K)
            print(f"  N={N:4d} K={K:4d}  max_diff={md:.4e}  PASS")
        except Exception as e:
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
    print("=== correctness: fused_par grid=(N//BLK,) ===")
    for N, K in _SHAPES:
        try:
            md = _run_fused_par_correctness(N, K)
            print(f"  N={N:4d} K={K:4d}  max_diff={md:.4e}  PASS")
        except Exception as e:
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
    print("=== correctness: fused sv3 (svector AFTER bridge) grid=(1,) ===")
    for N, K in _SHAPES:
        try:
            md = _run_fused_sv3_correctness(N, K)
            print(f"  N={N:4d} K={K:4d}  max_diff={md:.4e}  PASS")
        except Exception as e:
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
    print("=== perf vs svector style2 ===")
    BLK = 8
    for N, K in _SHAPES:
        try:
            t_sv = _measure_svector(N, K, BLOCK=4)
            t_f  = _measure_fused(N, K)
            t_fp = _measure_fused_par(N, K, BLK=BLK)
            gf_sv = 2.0 * N * K / t_sv / 1e9
            gf_f  = 2.0 * N * K / t_f  / 1e9
            gf_fp = 2.0 * N * K / t_fp / 1e9
            print(f"  N={N:4d} K={K:4d}  sv={t_sv*1e6:8.1f}us ({gf_sv:.2f}GF)  "
                  f"fused={t_f*1e6:8.1f}us ({gf_f:.2f}GF)  "
                  f"fused_par={t_fp*1e6:8.1f}us ({gf_fp:.2f}GF)")
        except Exception as e:
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
    print("=== perf: sv3 (svector post) vs ci (call_intrinsic post) ===")
    for N, K in _SHAPES:
        try:
            t_ci = _measure_fused(N, K)
            t_sv = _measure_fused_sv3(N, K)
            gf_ci = 2.0 * N * K / t_ci / 1e9
            gf_sv = 2.0 * N * K / t_sv / 1e9
            ratio = t_ci / t_sv
            print(f"  N={N:4d} K={K:4d}  ci_post={t_ci*1e6:8.1f}us ({gf_ci:.2f}GF)  "
                  f"sv_post={t_sv*1e6:8.1f}us ({gf_sv:.2f}GF)  ratio={ratio:.3f}x")
        except Exception as e:
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
