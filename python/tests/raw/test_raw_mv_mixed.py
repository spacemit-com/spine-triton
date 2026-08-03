"""Mixed tle.raw + tle.call_intrinsic mv kernel.

Goal: C = B @ A   with  B: [N, K] f16 row-major,  A: [K] f16,  C: [N] f32.

mixed_call_intrinsic: 8-row unrolled, hand-emitted via call_intrinsic:
  - llvm.riscv.vle16      for f16 vector loads (A cached once per K-tile, 4 B rows)
  - llvm.riscv.vfwmacc.vv for widening fma (f16 * f16 -> f32 accumulate)
  - tle.vreduce_sum       for vector->scalar reduction (svector helper, mixed)
  - tle.sstore            for scalar store (svector helper, mixed)

vs svector style2 (4-row unroll, tle.vload + tle.vmacc): 2x row-unroll gives
more independent vfwmacc per K-tile -> better in-order pipeline utilization.

Constraint: K % 64 == 0, N % 8 == 0 (full tiles, no tail handling).
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

# K3 vlen=1024, f16 SEW=16 -> VLMAX=64 at lmul=1. Scalable vector<[4]xf16>
# holds 4*vscale=64 elements (1 VREG). f32 acc at the same VL=64 needs LMUL=2
# (64 f32 = 2 VREGs) -> vector<[8]xf32> (8*vscale; vscale for nxv8f32 = 1024/256 = 4
# -> 8*4 = 32... still 1 VREG). LMUL encoding in scalable types is implicit: the
# LLVM intrinsic name llvm.riscv.vfwmacc.nxv8f32.nxv4f16.nxv4f16 with VL=64 asks
# the backend to use 64 f16 inputs -> 64 f32 outputs in 2 VREGs.
# NOTE: literals must be inlined (codegen reads ast.Constant.value, not ast.Name).


@tle.raw_kernel
def mv_mixed_call_intrinsic(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True),
                            K: tle.index, N: tle.index):
    """8-row unrolled mv via call_intrinsic. Assembly shows vfwmacc.vv.

    Single program processes all N rows in 8-row tiles (grid=(1,)).
    Multi-program would need program_id support in sibling ABI (not yet wired).
    """
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
            # Load A[ki:ki+64] once per K-tile (cached across 8 rows)
            ga = tle.llvm_gep(abase, ki, "f16")
            va = tle.call_intrinsic("llvm.riscv.vle", [pt, ga, vl], result_type="vector<[4]xf16>")

            # 8 B rows: B[ni+i, ki:ki+64]
            off0 = ni * K + ki
            off1 = (ni + 1) * K + ki
            off2 = (ni + 2) * K + ki
            off3 = (ni + 3) * K + ki
            off4 = (ni + 4) * K + ki
            off5 = (ni + 5) * K + ki
            off6 = (ni + 6) * K + ki
            off7 = (ni + 7) * K + ki
            gb0 = tle.llvm_gep(bbase, off0, "f16")
            gb1 = tle.llvm_gep(bbase, off1, "f16")
            gb2 = tle.llvm_gep(bbase, off2, "f16")
            gb3 = tle.llvm_gep(bbase, off3, "f16")
            gb4 = tle.llvm_gep(bbase, off4, "f16")
            gb5 = tle.llvm_gep(bbase, off5, "f16")
            gb6 = tle.llvm_gep(bbase, off6, "f16")
            gb7 = tle.llvm_gep(bbase, off7, "f16")
            vb0 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb0, vl], result_type="vector<[4]xf16>")
            vb1 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb1, vl], result_type="vector<[4]xf16>")
            vb2 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb2, vl], result_type="vector<[4]xf16>")
            vb3 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb3, vl], result_type="vector<[4]xf16>")
            vb4 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb4, vl], result_type="vector<[4]xf16>")
            vb5 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb5, vl], result_type="vector<[4]xf16>")
            vb6 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb6, vl], result_type="vector<[4]xf16>")
            vb7 = tle.call_intrinsic("llvm.riscv.vle", [pt, gb7, vl], result_type="vector<[4]xf16>")

            # 8 widening fma (vfwmacc.vv, 6-arg policy form): acc += va * vb (f16*f16 -> f32)
            acc0 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc0, va, vb0, zero, vl, zero], result_type="vector<[4]xf32>")
            acc1 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc1, va, vb1, zero, vl, zero], result_type="vector<[4]xf32>")
            acc2 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc2, va, vb2, zero, vl, zero], result_type="vector<[4]xf32>")
            acc3 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc3, va, vb3, zero, vl, zero], result_type="vector<[4]xf32>")
            acc4 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc4, va, vb4, zero, vl, zero], result_type="vector<[4]xf32>")
            acc5 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc5, va, vb5, zero, vl, zero], result_type="vector<[4]xf32>")
            acc6 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc6, va, vb6, zero, vl, zero], result_type="vector<[4]xf32>")
            acc7 = tle.call_intrinsic("llvm.riscv.vfwmacc", [acc7, va, vb7, zero, vl, zero], result_type="vector<[4]xf32>")

        # Reduce vector -> scalar (llvm.vector.reduce.fadd) + scalar store (llvm.store)
        s0 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc0], result_type="f32")
        s1 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc1], result_type="f32")
        s2 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc2], result_type="f32")
        s3 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc3], result_type="f32")
        s4 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc4], result_type="f32")
        s5 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc5], result_type="f32")
        s6 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc6], result_type="f32")
        s7 = tle.call_intrinsic("llvm.vector.reduce.fadd", [zero_f, acc7], result_type="f32")
        gc0 = tle.llvm_gep(cbase, ni, "f32")
        gc1 = tle.llvm_gep(cbase, ni + 1, "f32")
        gc2 = tle.llvm_gep(cbase, ni + 2, "f32")
        gc3 = tle.llvm_gep(cbase, ni + 3, "f32")
        gc4 = tle.llvm_gep(cbase, ni + 4, "f32")
        gc5 = tle.llvm_gep(cbase, ni + 5, "f32")
        gc6 = tle.llvm_gep(cbase, ni + 6, "f32")
        gc7 = tle.llvm_gep(cbase, ni + 7, "f32")
        tle.call_intrinsic("llvm.store", [s0, gc0], result_type="()")
        tle.call_intrinsic("llvm.store", [s1, gc1], result_type="()")
        tle.call_intrinsic("llvm.store", [s2, gc2], result_type="()")
        tle.call_intrinsic("llvm.store", [s3, gc3], result_type="()")
        tle.call_intrinsic("llvm.store", [s4, gc4], result_type="()")
        tle.call_intrinsic("llvm.store", [s5, gc5], result_type="()")
        tle.call_intrinsic("llvm.store", [s6, gc6], result_type="()")
        tle.call_intrinsic("llvm.store", [s7, gc7], result_type="()")


@triton.jit(do_not_specialize=["K", "N"])
def _mv_mixed_host(B, A, C, K, N):
    _sr_call(mv_mixed_call_intrinsic, outputs=[], inputs=[B, A, C, K, N])


# ---------------------------------------------------------------------------
# svector style2 baseline (copied for direct perf comparison)
# ---------------------------------------------------------------------------
@tle.raw_kernel
def mv_block_style2(B: tle.mem(f16), A: tle.mem(f16), C: tle.mem(f32, out=True), K: tle.index, row_base: tle.index,
                    row_end: tle.index):
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
def _run_correctness(host, N, K, BLOCK=8, mixed=False):
    Np = ((N + BLOCK - 1) // BLOCK) * BLOCK
    B = torch.randn(N, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    C = torch.empty(Np, dtype=torch.float32)
    if mixed:
        grid = (1,)
        host[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N)
    else:
        grid = (Np // BLOCK,)
        host[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N, BLOCK=BLOCK)
    got = C[:N]
    ref = torch.mv(B.float(), A.float())
    max_diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), \
        f"N={N} K={K} max_diff={max_diff:.4e}"
    return max_diff


def _measure(host, N, K, BLOCK, iters=50, warmup=5, mixed=False):
    Np = ((N + BLOCK - 1) // BLOCK) * BLOCK
    B = torch.randn(N, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    C = torch.empty(Np, dtype=torch.float32)
    if mixed:
        grid = (1,)
    else:
        grid = (Np // BLOCK,)
    # warmup
    for _ in range(warmup):
        if mixed:
            host[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N)
        else:
            host[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N, BLOCK=BLOCK)
    # timed
    t0 = time.perf_counter()
    for _ in range(iters):
        if mixed:
            host[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N)
        else:
            host[grid](B.contiguous().reshape(-1), A.contiguous(), C, K, N, BLOCK=BLOCK)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


# Mixed version requires N%8==0, K%64==0 (8-row unroll, VL=64 full tiles)
_SHAPES = [(8, 64), (16, 128), (32, 256), (64, 512), (128, 256), (64, 64), (32, 128), (16, 64)]


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mv_mixed_correctness(N, K):
    _run_correctness(_mv_mixed_host, N, K, BLOCK=8, mixed=True)


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mv_mixed_perf_vs_svector(N, K):
    """Assert mixed call_intrinsic is faster than svector style2."""
    # svector style2 uses BLOCK=4 (its design)
    t_sv = _measure(_mv_sv_host_style2, N, K, BLOCK=4)
    # mixed uses BLOCK=8 (rows per program)
    t_mx = _measure(_mv_mixed_host, N, K, BLOCK=8, mixed=True)
    gflops_sv = 2.0 * N * K / t_sv / 1e9
    gflops_mx = 2.0 * N * K / t_mx / 1e9
    print(f"N={N:4d} K={K:4d}  svector={t_sv*1e6:8.1f}us ({gflops_sv:.2f} GFLOPS)  "
          f"mixed={t_mx*1e6:8.1f}us ({gflops_mx:.2f} GFLOPS)  "
          f"speedup={t_sv/t_mx:.2f}x")
    assert t_mx <= t_sv, f"mixed slower than svector: mixed={t_mx*1e6}us sv={t_sv*1e6}us"


if __name__ == "__main__":
    print("=== correctness ===")
    for N, K in _SHAPES:
        try:
            md = _run_correctness(_mv_mixed_host, N, K, BLOCK=8, mixed=True)
            print(f"  N={N:4d} K={K:4d}  max_diff={md:.4e}  PASS")
        except Exception as e:
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
    print("=== perf ===")
    for N, K in _SHAPES:
        try:
            t_sv = _measure(_mv_sv_host_style2, N, K, BLOCK=4)
            t_mx = _measure(_mv_mixed_host, N, K, BLOCK=8, mixed=True)
            gf_sv = 2.0 * N * K / t_sv / 1e9
            gf_mx = 2.0 * N * K / t_mx / 1e9
            print(f"  N={N:4d} K={K:4d}  sv={t_sv*1e6:8.1f}us ({gf_sv:.2f}GF)  "
                  f"mx={t_mx*1e6:8.1f}us ({gf_mx:.2f}GF)  speedup={t_sv/t_mx:.2f}x")
        except Exception as e:
            print(f"  N={N:4d} K={K:4d}  FAIL: {type(e).__name__}: {str(e)[:200]}")
