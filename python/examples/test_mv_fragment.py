"""Matrix-vector multiply (MV) with smt.alloc(scope="fragment") scratch.

Reference: spine-FlagGems/src/flag_gems/ops/mv.py
Style ref: spine-triton/python/perf/test_softmax_fragment.py

The fragment buffer holds a 2D accumulator [BLOCK_M, BLOCK_N] in
register-backed storage.  Each program handles BLOCK_M rows and iterates
over N in BLOCK_N-sized chunks, doing element-wise fma (no per-iteration
reduce).  After the loop, a single tl.sum(axis=1) produces the final
[BLOCK_M] result — matching the FlagGems mv.py pattern for best
vectorization on RISC-V V extension hardware.

Usage:
    python python/examples/test_mv_fragment.py
"""

import time
import torch

import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

# ═══════════════════════════════════════════════════════════════════════════
# Baseline: plain block_ptr MV (no fragment)
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def mv_baseline_kernel(
    mat_ptr,
    vec_ptr,
    out_ptr,
    M,
    N,
    stride_mm,
    stride_mn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    element_ty = mat_ptr.type.element_ty
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for n_off in range(0, N, BLOCK_N):
        mat_blk = tl.make_block_ptr(
            base=mat_ptr,
            shape=[M, N],
            strides=[stride_mm, stride_mn],
            offsets=[pid * BLOCK_M, n_off],
            block_shape=[BLOCK_M, BLOCK_N],
            order=[1, 0],
        )
        vec_blk = tl.make_block_ptr(
            base=vec_ptr,
            shape=[N],
            strides=[1],
            offsets=[n_off],
            block_shape=[BLOCK_N],
            order=[0],
        )
        mat = tl.load(mat_blk, boundary_check=(0, 1)).to(tl.float32)
        vec = tl.load(vec_blk, boundary_check=(0, )).to(tl.float32)
        acc += mat * vec[None, :]

    result = tl.sum(acc, axis=1)
    out_blk = tl.make_block_ptr(
        base=out_ptr,
        shape=[M],
        strides=[1],
        offsets=[pid * BLOCK_M],
        block_shape=[BLOCK_M],
        order=[0],
    )
    tl.store(out_blk, result.to(element_ty), boundary_check=(0, ))


# ═══════════════════════════════════════════════════════════════════════════
# Fragment version: accumulator lives in fragment-scoped scratch
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def mv_fragment_kernel(
    mat_ptr,
    vec_ptr,
    out_ptr,
    M,
    N,
    stride_mm,
    stride_mn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """out[m] = sum_n( mat[m, n] * vec[n] )

    The fragment scratch [BLOCK_M, BLOCK_N] holds the 2D accumulator in
    register-backed storage.  Each BLOCK_N chunk does element-wise fma
    (no per-iteration reduce).  After the loop, tl.sum(axis=1) produces
    the final [BLOCK_M] result.
    """
    pid = tl.program_id(0)
    element_ty = mat_ptr.type.element_ty

    # Fragment-scoped 2D accumulator: memref<BLOCK_M x BLOCK_N x f32, 3>
    acc_frag = smt.alloc(shape=[BLOCK_M, BLOCK_N], type=tl.float32, scope="fragment")

    # Zero-init fragment
    tl.store(acc_frag, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    for n_off in range(0, N, BLOCK_N):
        mat_blk = tl.make_block_ptr(
            base=mat_ptr,
            shape=[M, N],
            strides=[stride_mm, stride_mn],
            offsets=[pid * BLOCK_M, n_off],
            block_shape=[BLOCK_M, BLOCK_N],
            order=[1, 0],
        )
        vec_blk = tl.make_block_ptr(
            base=vec_ptr,
            shape=[N],
            strides=[1],
            offsets=[n_off],
            block_shape=[BLOCK_N],
            order=[0],
        )
        mat = tl.load(mat_blk, boundary_check=(0, 1)).to(tl.float32)
        vec = tl.load(vec_blk, boundary_check=(0, )).to(tl.float32)

        # Accumulate element-wise fma in fragment (no per-iter reduce)
        acc_prev = tl.load(acc_frag)
        tl.store(acc_frag, acc_prev + mat * vec[None, :])

    # Single reduce after loop
    result = tl.sum(tl.load(acc_frag), axis=1).to(element_ty)
    out_blk = tl.make_block_ptr(
        base=out_ptr,
        shape=[M],
        strides=[1],
        offsets=[pid * BLOCK_M],
        block_shape=[BLOCK_M],
        order=[0],
    )
    tl.store(out_blk, result, boundary_check=(0, ))


# ═══════════════════════════════════════════════════════════════════════════
# Host helpers
# ═══════════════════════════════════════════════════════════════════════════


def _launch(kernel, mat, vec, out, block_m, block_n):
    M, N = mat.shape
    grid = (triton.cdiv(M, block_m), )
    kernel[grid](
        mat,
        vec,
        out,
        M,
        N,
        mat.stride(0),
        mat.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )


def mv_baseline(mat, vec, block_m=16, block_n=4):
    out = torch.empty(mat.shape[0], dtype=mat.dtype)
    _launch(mv_baseline_kernel, mat, vec, out, block_m, block_n)
    return out


def mv_fragment(mat, vec, block_m=16, block_n=4):
    out = torch.empty(mat.shape[0], dtype=mat.dtype)
    _launch(mv_fragment_kernel, mat, vec, out, block_m, block_n)
    return out


def _best_of_repeats(fn, warmup=5, iters=100, repeats=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.time()
        for _ in range(iters):
            fn()
        best = min(best, (time.time() - t0) / iters * 1000)
    return best


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)

    shapes = [(128, 64), (256, 128), (512, 256), (1024, 512), (1024, 1024)]
    dtypes = [torch.float32]

    # ── correctness ──────────────────────────────────────────────────────
    print("=" * 72)
    print("Correctness: baseline vs fragment vs torch.mv")
    print("=" * 72)
    for dt in dtypes:
        for (M, N) in shapes:
            mat = torch.randn(M, N, dtype=dt)
            vec = torch.randn(N, dtype=dt)
            y_base = mv_fragment(mat, vec, block_m=32, block_n=8)
            y_frag = mv_baseline(mat, vec, block_m=32, block_n=8)
            y_ref = torch.mv(mat, vec)

            ok_b = torch.allclose(y_base, y_ref, atol=1e-2, rtol=1e-2)
            ok_f = torch.allclose(y_frag, y_ref, atol=1e-2, rtol=1e-2)
            diff_b = (y_base - y_ref).abs().max().item()
            diff_f = (y_frag - y_ref).abs().max().item()
            tag_b = "✅" if ok_b else f"❌ {diff_b:.2e}"
            tag_f = "✅" if ok_f else f"❌ {diff_f:.2e}"
            print(f"  {dt}  ({M:>4},{N:>4})  baseline={tag_b}  fragment={tag_f}")

    # ── perf comparison ──────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("Performance: baseline vs fragment (BLOCK_M=16, BLOCK_N=4)")
    print("=" * 72)
    for dt in dtypes:
        print(f"\n  dtype={dt}")
        print(f"  {'shape':>12}  {'baseline':>12}  {'fragment':>12}  {'speedup':>8}")
        print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
        for (M, N) in shapes:
            mat = torch.randn(M, N, dtype=dt)
            vec = torch.randn(N, dtype=dt)
            ms_b = _best_of_repeats(lambda: mv_baseline(mat, vec, block_m=16, block_n=4))
            ms_f = _best_of_repeats(lambda: mv_fragment(mat, vec, block_m=16, block_n=4))
            speedup = ms_b / ms_f if ms_f > 0 else float("inf")
            print(f"  ({M:>4},{N:>4})  {ms_b:>9.3f} ms  {ms_f:>9.3f} ms  {speedup:>7.2f}x")
