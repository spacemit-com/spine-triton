"""3-stage mv: svector pre-scale -> call_intrinsic vfwmacc -> svector post-scale.

Mirrors test_mixed_syntax_three_layer.py's multi-stage pattern, applied to mv:

  stage 1  pre_scale_sv   : vec_s[k] = vec[k] * alpha   —— svector helpers
                            (tle.vload / tle.cast / tle.vstore + BinOp)
  stage 2  mv_vfwmacc_ci  : scores = Mat @ vec_s        —— call_intrinsic
                            (llvm.riscv.vle + llvm.riscv.vfwmacc + llvm.store)
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


# ── stage 2: call_intrinsic 8-row vfwmacc mv  scores = Mat @ vec_s ──────────
# Pure llvm-direct (sibling llvm.func). 8-row unroll + vfwmacc.vv (6-arg policy
# form). Reuses the proven design from commit e7f38fbed (test_raw_mv_mixed.py).
@tle.raw_kernel
def mv_vfwmacc_call_intrinsic(B: tle.mem(f16), A: tle.mem(f16),
                              C: tle.mem(f32, out=True),
                              K: tle.index, N: tle.index):
    vl = tle.llvm_const(64, "i64")
    zero = tle.llvm_const(0, "i64")
    eight = tle.llvm_const(8, "i64")
    zero_acc = tle.llvm_const("0.000000e+00", "vector<[4]xf32>")
    zero_f = tle.llvm_const("0.000000e+00", "f32")
    pt = tle.llvm_poison("vector<[4]xf16>")
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
            va = tle.call_intrinsic("llvm.riscv.vle", [pt, ga, vl],
                                    result_type="vector<[4]xf16>")
            off0 = ni * K + ki
            gb0 = tle.llvm_gep(bbase, off0, "f16")
            gb1 = tle.llvm_gep(bbase, (ni + 1) * K + ki, "f16")
            gb2 = tle.llvm_gep(bbase, (ni + 2) * K + ki, "f16")
            gb3 = tle.llvm_gep(bbase, (ni + 3) * K + ki, "f16")
            gb4 = tle.llvm_gep(bbase, (ni + 4) * K + ki, "f16")
            gb5 = tle.llvm_gep(bbase, (ni + 5) * K + ki, "f16")
            gb6 = tle.llvm_gep(bbase, (ni + 6) * K + ki, "f16")
            gb7 = tle.llvm_gep(bbase, (ni + 7) * K + ki, "f16")
            vb0 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb0, vl], result_type="vector<[4]xf16>")
            vb1 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb1, vl], result_type="vector<[4]xf16>")
            vb2 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb2, vl], result_type="vector<[4]xf16>")
            vb3 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb3, vl], result_type="vector<[4]xf16>")
            vb4 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb4, vl], result_type="vector<[4]xf16>")
            vb5 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb5, vl], result_type="vector<[4]xf16>")
            vb6 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb6, vl], result_type="vector<[4]xf16>")
            vb7 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb7, vl], result_type="vector<[4]xf16>")
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


# ── stage 3: svector post-scale  out = scores * beta ────────────────────────
# f32 scores → mul beta (broadcast) → store. N arbitrary (tail-aware vconfig).
@tle.raw_kernel
def post_scale_svector(scores: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    for ni in tle.range(0, Nfloor, nvl):
        v = tle.vload(scores, ni, dtype=f32)     # vector<64xf32>
        v_s = v * _BETA                          # vector<64xf32>  (arith.mulf, scalar bcast)
        tle.vstore(out, ni, v_s)
    # tail loop: distinct names (same reason as pre_scale_svector)
    for ni in tle.range(Nfloor, N, nvl):
        nvl = tle.vconfig(N - ni, 1)
        tv = tle.vload(scores, ni, dtype=f32)
        tv_s = tv * _BETA
        tle.vstore(out, ni, tv_s)


# ── hosts: 2 launches (sibling call_intrinsic must be LAST in its launch) ──
# The mixed-mode bridge (_inject_mixed_llvm_llmlir) injects all llvm.call
# bridges right before the host's first llvm.return — so the sibling
# llvm.func effectively runs LAST in its launch host. A svector dsl_region
# AFTER the sibling (single-launch 3-stage) hits a dominance error because
# the bridge writes through `scores` after stage 3 already read it.
# Splitting into 2 launches keeps the sibling last in launch 1 and gives
# stage 3 its own launch where it's the only op.
@triton.jit(do_not_specialize=["K", "N"])
def _mv_pre_and_gemm_host(Mat, vec, vec_s, scores, K, N):
    _sr_call(pre_scale_svector, outputs=[], inputs=[vec, vec_s, K])
    _sr_call(mv_vfwmacc_call_intrinsic, outputs=[], inputs=[Mat, vec_s, scores, K, N])


@triton.jit(do_not_specialize=["N"])
def _mv_post_scale_host(scores, out, N):
    _sr_call(post_scale_svector, outputs=[], inputs=[scores, out, N])


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


def _run_correctness(N, K):
    assert K % 64 == 0 and N % 8 == 0, "stage 1/2 need K%64==0, N%8==0"
    Np = ((N + 7) // 8) * 8
    torch.manual_seed(0)
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)

    _mv_pre_and_gemm_host[(1,)](
        Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, K, N)
    _mv_post_scale_host[(1,)](scores, out, N)

    got = out[:N]
    ref = torch.mv(Mat.float(), (vec.float() * _ALPHA).half().float()) * _BETA
    max_diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"
    return max_diff


def _measure_three_stage(N, K, iters=50, warmup=5):
    Np = ((N + 7) // 8) * 8
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)
    for _ in range(warmup):
        _mv_pre_and_gemm_host[(1,)](
            Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, K, N)
        _mv_post_scale_host[(1,)](scores, out, N)
    t0 = time.perf_counter()
    for _ in range(iters):
        _mv_pre_and_gemm_host[(1,)](
            Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, K, N)
        _mv_post_scale_host[(1,)](scores, out, N)
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
def test_mv_three_stage_correctness(N, K):
    _run_correctness(N, K)


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mv_three_stage_perf_vs_svector(N, K):
    """3-stage (svector+call_intrinsic+svector) vs pure svector style2.

    The 3-stage does EXTRA work (pre/post scale) the svector baseline doesn't,
    so absolute time is higher — this is a report-only comparison showing the
    overhead of the additional svector stages around the vfwmacc gemv. The
    vfwmacc.vv in stage 2's assembly (verified separately) is the architectural
    point: only vfwmacc uses call_intrinsic, everything else uses svector.
    """
    t_sv = _measure_svector(N, K, BLOCK=4)
    t_ts = _measure_three_stage(N, K)
    gf_sv = 2.0 * N * K / t_sv / 1e9
    gf_ts = 2.0 * N * K / t_ts / 1e9
    overhead_us = (t_ts - t_sv) * 1e6
    print(f"N={N:4d} K={K:4d}  svector={t_sv*1e6:8.1f}us ({gf_sv:.2f}GF)  "
          f"three_stage={t_ts*1e6:8.1f}us ({gf_ts:.2f}GF)  "
          f"overhead={overhead_us:6.1f}us")


if __name__ == "__main__":
    print("=== 3-stage mv: svector -> call_intrinsic vfwmacc -> svector ===")
    print("=== correctness ===")
    for N, K in _SHAPES:
        try:
            md = _run_correctness(N, K)
            print(f"  N={N:4d} K={K:4d}  max_diff={md:.4e}  PASS")
        except Exception as e:
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
    print("=== perf ===")
    for N, K in _SHAPES:
        try:
            t_sv = _measure_svector(N, K, BLOCK=4)
            t_ts = _measure_three_stage(N, K)
            gf_sv = 2.0 * N * K / t_sv / 1e9
            gf_ts = 2.0 * N * K / t_ts / 1e9
            print(f"  N={N:4d} K={K:4d}  sv={t_sv*1e6:8.1f}us ({gf_sv:.2f}GF)  "
                  f"ts={t_ts*1e6:8.1f}us ({gf_ts:.2f}GF)  overhead={(t_ts-t_sv)*1e6:6.1f}us")
        except Exception as e:
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
