"""K3 single-thread memory bandwidth via spine_raw copy kernel + mv utilization."""
import os, sys, time
import numpy as np
import torch
import triton, triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f32 = tle.f32

# ── single-thread STREAM COPY via spine_raw ──────────────────────────────────
@tle.raw_kernel
def copy_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nf = (N // nvl) * nvl
    for i in tle.range(0, Nf, nvl):
        tle.vstore(out, i, tle.vload(X, i, dtype=f32))
    for i in tle.range(Nf, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tle.vstore(out, i, tle.vload(X, i, dtype=f32))

@triton.jit
def copy_host(X, out, N):
    _sr_call(copy_kernel, outputs=[], inputs=[X, out, N])


def bench(fn, reps=30):
    for _ in range(10): fn()
    ts = [0.0] * reps
    for k in range(reps):
        t0 = time.perf_counter(); fn(); ts[k] = (time.perf_counter()-t0)*1e6
    return float(np.median(ts))


def main():
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'

    print("=== K3 single-thread peak bandwidth (spine_raw STREAM COPY) ===")
    peak_bw = 0.0
    for N in [1*1024*1024, 4*1024*1024, 16*1024*1024, 32*1024*1024]:
        X = torch.randn(N, dtype=torch.float32)
        out = torch.zeros(N, dtype=torch.float32)
        us = bench(lambda: copy_host[(1,)](X, out, N))
        bw = (2*N*4/1e9) / (us*1e-6)
        peak_bw = max(peak_bw, bw)
        print(f"  N={N//1024//1024:3d}M  {us:7.0f} us  {bw:.2f} GB/s")

    print(f"\n→ single-thread peak BW = {peak_bw:.2f} GB/s\n")

    # ── mv bandwidth utilization ──────────────────────────────────────────────
    from importlib.machinery import SourceFileLoader
    sv = SourceFileLoader("mv_sv",
         os.path.join(os.path.dirname(__file__), "test_raw_mv_svector.py")).load_module()

    SHAPES = [
        (256, 256),(512, 256),(1024, 256),
        (256, 512),(512, 512),(1024, 512),
        (256,1024),(512,1024),(1024,1024),
        (2048,1024),(4096,1024),(2048,2048),
    ]
    BLOCK = 4

    print("=== mv svector style2 bandwidth utilization ===")
    print(f"{'M':>6} {'K':>6} {'us':>8} {'data_MB':>8} {'BW_GBs':>8} {'util%':>7}")
    print("-"*55)

    utils = []
    for M, K in SHAPES:
        B_t = torch.randn(M, K, dtype=torch.float16)
        A_t = torch.randn(K,    dtype=torch.float16)
        C_t = torch.zeros(M,    dtype=torch.float32)
        grid = (M // BLOCK,)
        def run(B=B_t, A=A_t, C=C_t, g=grid, k=K, m=M, bl=BLOCK):
            sv._mv_sv_host_style2[g](B.contiguous().reshape(-1), A.contiguous(), C, k, m, bl)
        us = bench(run)
        data = (2*M*K + 2*K + 4*M) / 1e6   # MB
        bw   = data / 1e3 / (us*1e-6)
        util = bw / peak_bw * 100
        utils.append(util)
        print(f"{M:>6} {K:>6} {us:>8.1f} {data:>8.2f} {bw:>8.2f} {util:>6.1f}%")

    print("-"*55)
    print(f"mean util: {np.mean(utils):.1f}%   max: {np.max(utils):.1f}%")
    print()
    print("Interpretation:")
    print(f"  single-thread peak BW = {peak_bw:.2f} GB/s")
    print(f"  mv arithmetic intensity = O(1) flops/byte → memory-bound regime")
    if np.max(utils) < 50:
        print(f"  ⚠ {np.max(utils):.0f}% peak — kernel launch overhead dominates at small shapes")
        print(f"    (larger M/K pushes utilization higher; 256x256 is tiny)")
    else:
        print(f"  ✅ {np.max(utils):.0f}% peak — good bandwidth utilization")

if __name__ == "__main__":
    main()
