"""Mixed-syntax composition — single stage (PLAN_mixed_syntax_composition.md).

Demonstrates the two syntax layers that DO compose today via host-level
orchestration:

  • 语法层级 1 (@triton.jit host): tl.program_id + 算术 + min 做工作分配
  • 语法层级 2 (@tle.raw_kernel, dsl_region path): vload/vmacc/vreduce_sum/sstore

The host computes each program's row range with *Triton* arithmetic and passes
BASE pointers + scalar dims/bounds into one spine_raw sub-kernel. The sub-kernel
is inlined as a `tle.dsl_region` into the host body (call_registry.py:99-109),
so this is a genuine compile-time composition, not a runtime call.

Compute: C = Mat @ vec   (Mat: [N,K] f32 row-major, vec: [K] f32, C: [N] f32).

Per the SPMD contract (AGENT.md §8.1) the per-row work split is done in the
host and handed to the kernel as scalar bounds (row_base/row_end) — NOT as a
pre-offset pointer. This is the working half of the PLAN.
"""
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


# ── 语法层级 2: spine_raw 子 kernel (dsl_region 路径) ──────────────────────
# Mat/vec are f16, accumulator f32: tle.vmacc IS the widening vfwmacc
# (f16×f16→f32), the K3-proven idiom (test_raw_mv_svector.py). Feeding f32 into
# vmacc builds a 2048-bit vector<64xf32> fma that mis-tiles for K>VL — so the
# matmul inputs stay f16 and only the reduction result is f32.
@tle.raw_kernel
def gemv_rows(Mat: tle.mem(f16), vec: tle.mem(f16), C: tle.mem(f32, out=True),
              K: tle.index, row_base: tle.index, row_end: tle.index):
    """Compute C[n] = sum_k Mat[n*K + k] * vec[k] for n in [row_base, row_end)."""
    nvl = tle.vconfig(-1, 1)             # f16 lmul=1 → VLMAX=64
    Kfloor = (K // nvl) * nvl            # full-tile K span
    for n in tle.range(row_base, row_end, 1):
        acc = tle.vzero(f32)
        for ki in tle.range(0, Kfloor, nvl):        # main loop: full tiles, fast path
            vm = tle.vload(Mat, n * K + ki)          # f16 (vload default)
            vv = tle.vload(vec, ki)
            acc = tle.vmacc(acc, vm, vv)             # widening f16×f16→f32
        for ki in tle.range(Kfloor, K, nvl):        # tail: 0/1 iters, narrow → fill-0
            nvl = tle.vconfig(K - ki, 1)
            tm = tle.vload(Mat, n * K + ki)          # distinct temp names (tail iter_arg rule)
            tv = tle.vload(vec, ki)
            acc = tle.vmacc(acc, tm, tv)
        tle.sstore(C, n, tle.vreduce_sum(acc))


# ── 语法层级 1: Triton host — 用 Triton 语法做工作分配 ──────────────────────
@triton.jit(do_not_specialize=["K", "N"])
def gemv_host(Mat, vec, C, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)   # 末 program 不越界 N
    _sr_call(gemv_rows, outputs=[], inputs=[Mat, vec, C, K, row_base, row_end])


def _run(N, K, BLOCK=4):
    torch.manual_seed(0)
    # 末 program 的 row_end 已被 min 限到 N,但内层按 row 无条件 sstore(C, n),
    # n 严格 < row_end ≤ N,故不写 phantom 行 → C 分配 N 即可。
    Mat = torch.randn(N, K, dtype=torch.float16)   # f16 inputs (widening vmacc)
    vec = torch.randn(K, dtype=torch.float16)
    C = torch.zeros(N, dtype=torch.float32)         # f32 accumulator/output
    grid = ((N + BLOCK - 1) // BLOCK, )
    gemv_host[grid](Mat.contiguous().reshape(-1), vec.contiguous(), C, K, N, BLOCK=BLOCK)
    # golden in f32 from the SAME f16-rounded inputs the kernel reads
    ref = torch.mv(Mat.float(), vec.float())
    max_diff = (C - ref).abs().max().item()
    assert torch.allclose(C, ref, rtol=1e-2, atol=1e-1), \
        f"N={N} K={K} BLOCK={BLOCK} max_diff={max_diff:.4e}"
    return max_diff


_SHAPES = [(4, 64), (8, 128), (16, 256), (7, 65), (13, 100)]


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mixed_single_stage(N, K):
    _run(N, K)


if __name__ == "__main__":
    print("=== Mixed-syntax single stage: Triton host + spine_raw GEMV ===")
    all_ok = True
    for N, K in _SHAPES:
        try:
            md = _run(N, K)
            print(f"PASS  N={N:3d} K={K:3d}  max_diff={md:.4e}")
        except Exception as e:
            all_ok = False
            print(f"FAIL  N={N:3d} K={K:3d}  {type(e).__name__}: {str(e)[:80]}")
    print("ALL_PASS" if all_ok else "HAS_FAILURES")
