"""mv bandwidth-utilization ceiling on K3: sweep BLOCK × shape, report best.

Roofline references:
  - multi-thread peak: torch STREAM COPY (uses all cores)
  - single-thread peak: spine_raw copy kernel grid=(1,)
mv effective bytes = B[M,K]*2(f16) + A[K]*2(f16) + C[M]*4(f32).
BW = bytes / median_time. util = BW / peak.
Best util per shape = max over BLOCK sweep (dispatch/parallelism tradeoff).
"""
import os, sys, time
import numpy as np
import torch
import triton, triton.language as tl
from importlib.machinery import SourceFileLoader
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle  # noqa: F401

_TESTS = os.path.dirname(__file__)
sv = SourceFileLoader("mv_sv", os.path.join(_TESTS, "test_raw_mv_svector.py")).load_module()

WARMUP, REPS = 10, 30


def bench(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(ts))


# ── peak BW references ───────────────────────────────────────────────────────
def peak_multithread():
    N = 16 << 20
    a = torch.randn(N, dtype=torch.float32)
    b = torch.empty_like(a)
    us = bench(lambda: b.copy_(a))
    return (2 * N * 4) / (us * 1e-6) / 1e9  # read+write


def main():
    mt = peak_multithread()
    print(f"multi-thread peak (torch copy): {mt:.1f} GB/s")
    print()
    # BLOCK must divide the row count; mv_svector uses BLOCK-row program.
    BLOCKS = [4, 8, 16, 32, 64, 128, 256]
    SHAPES = [(1024, 512), (1024, 1024), (2048, 1024), (2048, 2048), (4096, 2048)]
    print(f"{'M':>6}{'K':>6}{'bestBLK':>8}{'bestUS':>9}{'BW_GBs':>9}{'util%':>7}")
    print("-" * 50)
    best_overall = 0.0
    for M, K in SHAPES:
        B = torch.randn(M, K, dtype=torch.float16).contiguous().reshape(-1)
        A = torch.randn(K, dtype=torch.float16).contiguous()
        Cbuf = torch.zeros(M, dtype=torch.float32)
        ref = (B.reshape(M, K).float() @ A.float())
        nbytes = M * K * 2 + K * 2 + M * 4
        best_us, best_blk, ok_flag = 1e18, 0, False
        for BLOCK in BLOCKS:
            if M % BLOCK != 0:
                continue
            grid = (M // BLOCK,)

            def run(g=grid, bl=BLOCK):
                sv._mv_sv_host_style2[g](B, A, Cbuf, K, M, bl)
            try:
                us = bench(run)
            except Exception:
                continue
            ok = (Cbuf.float() - ref).abs().max().item() < 5e-2
            if us < best_us:
                best_us, best_blk, ok_flag = us, BLOCK, ok
        bw = nbytes / (best_us * 1e-6) / 1e9
        util = bw / mt * 100
        best_overall = max(best_overall, util)
        print(f"{M:>6}{K:>6}{best_blk:>8}{best_us:>9.1f}{bw:>9.2f}{util:>6.1f}%  ok={ok_flag}")
    print("-" * 50)
    print(f"best mv BW utilization (vs multi-thread peak): {best_overall:.1f}%")


if __name__ == "__main__":
    main()
