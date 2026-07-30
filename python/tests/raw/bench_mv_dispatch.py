"""Dispatch-overhead experiment: same total work, vary grid_size via BLOCK.

Reuses the proven mv_block_style2 kernel from test_raw_mv_svector.py (loaded
the same way bench_mv_bw.py does, so f16 is handled inside that module).
The host _mv_sv_host_style2 takes BLOCK as a runtime arg and slices rows by
program_id, so calling it with a larger BLOCK shrinks grid = M//BLOCK -> fewer
cpu_utils.launch C-calls. Total vector work is identical; only dispatch count
changes. That isolates dispatch overhead from vector compute.
"""
import os, time
import numpy as np
import torch
import triton, triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle  # noqa: F401
from importlib.machinery import SourceFileLoader

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

def main():
    M, K = 1024, 512
    torch.manual_seed(0)
    B = torch.randn(M, K, dtype=torch.float16)
    A = torch.randn(K, dtype=torch.float16)
    ref = (B.float() @ A.float())

    print("=" * 64)
    print(f"Dispatch experiment: mv M={M} K={K}, SAME work, vary BLOCK->grid")
    print("=" * 64)
    print(f"{'BLOCK':>6} {'grid':>6} {'us':>10} {'vs BLOCK=4':>12} {'ok':>5}")
    print("-" * 64)
    base = None
    for BLOCK in (4, 8, 16, 64, 256, 1024):
        if M % BLOCK != 0:
            continue
        Bf = B.contiguous().reshape(-1)
        C = torch.zeros(M, dtype=torch.float32)
        grid = (M // BLOCK,)
        def run(Bf=Bf, A=A, C=C, g=grid, k=K, m=M, bl=BLOCK):
            sv._mv_sv_host_style2[g](Bf, A.contiguous(), C, k, m, bl)
        us = bench(run)
        ok = (C - ref).abs().max().item() < 5e-2
        if base is None:
            base = us
        print(f"{BLOCK:>6} {grid[0]:>6} {us:>10.1f} {base/us:>11.2f}x {str(ok):>5}")
    print("-" * 64)
    print("If dispatch dominates: fewer programs (bigger BLOCK) -> faster,")
    print("even though total vector work is unchanged.")

if __name__ == "__main__":
    main()
