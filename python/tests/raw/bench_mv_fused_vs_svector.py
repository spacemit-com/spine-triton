"""Honest perf comparison: single-launch fused mv (grid=1) vs svector style2/style3.

Goal (per user): _mv_fused_host must be >= style2/style3 in perf.

CRITICAL fairness rules (from mv_perf_block_tuning lessons):
  - svector style2/style3 run grid=(Np//BLOCK,) → MULTI-CORE parallel. Their perf
    depends heavily on BLOCK; a fixed BLOCK=4 understates them ("single-wave"
    pollution). So we SWEEP BLOCK per shape and take the BEST (fastest) time.
  - _mv_fused_host runs grid=(1,) → SINGLE program (sibling ABI has no program_id,
    see test_raw_mv_mixed.py:44-45). This is the architectural ceiling under test.

Reports per shape: fused us, best style2 us (+BLOCK), best style3 us (+BLOCK),
and speedup fused_vs_style2 / fused_vs_style3 (>1.0 means fused is faster).
"""
import time
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

# fused single-launch host (stage1 svector → stage2/3 call_intrinsic bridges)
from test_raw_mv_three_stage import _mv_fused_host, _mv_fused_host_par
# svector baselines (multi-core parallel, program_id-strided)
from test_raw_mv_svector import _mv_sv_host_style2, _mv_sv_host_style3

_SHAPES = [(8, 64), (64, 512), (128, 256),
           # large shapes: compute should dominate the ~125us launch-overhead floor
           (256, 1024), (512, 1024), (1024, 1024), (512, 2048), (1024, 4096)]
_BLOCKS = [4, 8, 16, 32, 64, 128]  # swept per shape; must be multiple of 4 (inner 4-row group)


def _measure_fused(N, K, iters=50, warmup=5):
    Np = ((N + 7) // 8) * 8
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)
    args = (Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N)
    for _ in range(warmup):
        _mv_fused_host[(1, )](*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        _mv_fused_host[(1, )](*args)
    return (time.perf_counter() - t0) / iters


def _measure_fused_par(N, K, BLK, iters=50, warmup=5):
    # grid=(N//BLK,): program_id-partitioned multi-core parallel fused host.
    # Each program handles BLK rows (inner loop of 8-row sub-tiles). BLK is a
    # runtime i64 bridged into the sibling; sweeping it matches style2's block
    # granularity (mv_perf_block_tuning lesson — fixed small BLK oversubscribes).
    Np = ((N + BLK - 1) // BLK) * BLK
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(Np, dtype=torch.float32)
    out = torch.zeros(Np, dtype=torch.float32)
    args = (Mat.contiguous().reshape(-1), vec.contiguous(), vec_s, scores, out, K, N, BLK)
    grid = (Np // BLK, )
    for _ in range(warmup):
        _mv_fused_host_par[grid](*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        _mv_fused_host_par[grid](*args)
    return (time.perf_counter() - t0) / iters


def _best_fused_par(N, K):
    # Sweep BLK like _best_svector; BLK must be %8==0 and divide Np (=N rounded
    # up to BLK). Restrict to multiples of 8 (inner 8-row group). Take fastest.
    best_t, best_b = float("inf"), None
    for b in _BLOCKS:
        if b % 8 != 0 or N % b != 0:
            continue
        try:
            t = _measure_fused_par(N, K, b)
            if t < best_t:
                best_t, best_b = t, b
        except Exception as e:
            print(f"    [fused BLK={b} skip: {type(e).__name__}: {str(e)[:80]}]")
    return best_t, best_b


def _measure_svector(host, N, K, BLOCK, iters=50, warmup=5):
    Np = ((N + BLOCK - 1) // BLOCK) * BLOCK
    B = torch.randn(N, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    C = torch.empty(Np, dtype=torch.float32)
    grid = (Np // BLOCK, )
    args = (B.contiguous().reshape(-1), A.contiguous(), C, K, N)
    for _ in range(warmup):
        host[grid](*args, BLOCK=BLOCK)
    t0 = time.perf_counter()
    for _ in range(iters):
        host[grid](*args, BLOCK=BLOCK)
    return (time.perf_counter() - t0) / iters


def _best_svector(host, N, K):
    best_t, best_b = float("inf"), None
    for b in _BLOCKS:
        try:
            t = _measure_svector(host, N, K, b)
            if t < best_t:
                best_t, best_b = t, b
        except Exception as e:
            print(f"    [BLOCK={b} skip: {type(e).__name__}: {str(e)[:80]}]")
    return best_t, best_b


if __name__ == "__main__":
    print("=== fused_par (BLK-swept best) vs svector style2/style3 (BLOCK-swept best) ===")
    print("(also shows fused grid=1 baseline for reference)")
    print(
        f"{'N':>4} {'K':>5} | {'par_us':>8} {'b':>3} | {'g1_us':>8} | {'sv2_us':>8} {'b':>3} | {'sv3_us':>8} {'b':>3} "
        f"| {'par/sv2':>8} {'par/sv3':>8}")
    for N, K in _SHAPES:
        try:
            tp, bp = _best_fused_par(N, K)
            tf = _measure_fused(N, K)
            t2, b2 = _best_svector(_mv_sv_host_style2, N, K)
            t3, b3 = _best_svector(_mv_sv_host_style3, N, K)
            # speedup >1.0 means parallel fused faster than svector
            sp2 = t2 / tp if tp > 0 else 0.0
            sp3 = t3 / tp if tp > 0 else 0.0
            print(
                f"{N:>4} {K:>5} | {tp*1e6:8.1f} {str(bp):>3} | {tf*1e6:8.1f} | {t2*1e6:8.1f} {b2:>3} | {t3*1e6:8.1f} {b3:>3} "
                f"| {sp2:8.2f} {sp3:8.2f}")
        except Exception as e:
            print(f"{N:>4} {K:>5} | FAIL: {type(e).__name__}: {str(e)[:120]}")
    print("par/sv >1.0 = parallel fused faster than svector; <1.0 = svector faster")
