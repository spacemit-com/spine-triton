"""mv perf sweep: spine_raw's three mv writings vs FlagGems native, f16.

  style2  — pure svector (vmacc + vreduce_sum, no packing)
  style3  — svector + pre-pack B (alloc + pack)
  cbm     — matrix engine (vpack/spread/vmadot cross_batch_matmul)
  flaggems — native FlagGems mv_kernel (tl.load/mul/sum), if importable

Common shape constraints so all run: N(=M) % 16 == 0, K % 64 == 0
(svector: N%4 & K%64; cbm: M%16 & K%8; flaggems: any). Each impl:
WARMUP warmup + REPS iters, median us; speedup vs FlagGems; checked vs torch.mv.

Run on K3: PYTHONPATH -> the riscv64 build, GEMS_VENDOR=spacemit for FlagGems,
LD_LIBRARY_PATH -> the spine TCM runtime. See language notes for details.
"""
import os
import sys
import time
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle  # noqa: F401  (registers the backend)

WARMUP, REPS = 20, 100

# Reuse the mv kernels from the raw tests (repo-relative, no absolute paths).
_TESTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tests"))
sv = SourceFileLoader("mv_sv", os.path.join(_TESTS, "test_raw_mv_svector.py")).load_module()
cbm = SourceFileLoader("mv_cbm", os.path.join(_TESTS, "test_raw_mv_cbm.py")).load_module()

# FlagGems lives in a separate repo; make it optional so this runs standalone.
try:
    from flag_gems.ops.mv import mv as fg_mv
    _HAVE_FG = True
except Exception as e:  # noqa: BLE001
    _HAVE_FG = False
    print(f"[perf_mv] FlagGems not importable ({e}); skipping the flaggems column.")

# (N=M, K): N%16==0 (svector N%4 & cbm M%16), K%64==0 (svector K%64 & cbm K%8)
SHAPES = [(64, 64), (128, 64), (256, 64), (512, 64), (1024, 64),
          (64, 128), (128, 128), (256, 128), (512, 128),
          (128, 256), (256, 256), (512, 512)]


def bench(fn, ref, check):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    md = (check().float() - ref).abs().max().item()
    return float(np.median(ts)), md < 5e-2


def main():
    print(f"\nmv perf sweep, f16 (median of {REPS} iters, {WARMUP} warmup)")
    hdr = f"{'N x K':>10} | {'style2':>9} {'style3':>9} {'cbm':>9}"
    if _HAVE_FG:
        hdr += f" {'flaggems':>9} | {'s2/fg':>6} {'s3/fg':>6} {'cbm/fg':>7}"
    hdr += f" | {'all_ok':>6}"
    print(hdr)
    print("-" * len(hdr))

    for N, K in SHAPES:
        torch.manual_seed(0)
        Blog = torch.randn(N, K, dtype=torch.float16)
        Alog = torch.randn(K, dtype=torch.float16)
        ref = torch.mv(Blog.float(), Alog.float())
        B, A = Blog.contiguous(), Alog.contiguous()

        Cs2 = torch.empty(N, dtype=torch.float32)
        t_s2, ok2 = bench(lambda: sv._mv_sv_host_style2[(N // 4,)](B, A, Cs2, K, N, BLOCK=4), ref, lambda: Cs2)
        Cs3 = torch.empty(N, dtype=torch.float32)
        t_s3, ok3 = bench(lambda: sv._mv_sv_host_style3[(N // 4,)](B, A, Cs3, K, N, BLOCK=4), ref, lambda: Cs3)
        Ccbm = torch.zeros(N, cbm.Npad, dtype=torch.float16)
        ch = cbm.make_mv(N, K)
        t_cb, okc = bench(lambda: ch[(N // cbm.MB,)](B, A, Ccbm, BLOCK=cbm.MB), ref, lambda: Ccbm[:, 0])

        line = f"{N:>4}x{K:<4} | {t_s2:9.1f} {t_s3:9.1f} {t_cb:9.1f}"
        all_ok = ok2 and ok3 and okc
        if _HAVE_FG:
            of = [None]
            t_fg, okf = bench(lambda: of.__setitem__(0, fg_mv(B, A)), ref, lambda: of[0])
            line += (f" {t_fg:9.1f} | {t_fg / t_s2:6.2f} {t_fg / t_s3:6.2f} {t_fg / t_cb:7.2f}")
            all_ok = all_ok and okf
        line += f" | {str(all_ok):>6}"
        print(line)


if __name__ == "__main__":
    sys.exit(main())
