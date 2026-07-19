"""reduce-family extended perf: group_norm 2D sweep + cumsum vec vs scalar.

Extends perf_reduce.py results with:
  group_norm: spine_raw vs FlagGems, sweep over (G, C) shapes
  cumsum_vec: 3-phase vectorized vs sequential scalar, large N
"""
import os, sys, time
from importlib.machinery import SourceFileLoader
import numpy as np
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle  # noqa

_TESTS = os.path.dirname(__file__)
_gn = SourceFileLoader("gn_mod", os.path.join(_TESTS, "test_raw_group_norm.py")).load_module()
_cs = SourceFileLoader("csv_mod", os.path.join(_TESTS, "test_raw_cumsum_vec.py")).load_module()
_cs1 = SourceFileLoader("cs1_mod", os.path.join(_TESTS, "test_raw_cumsum.py")).load_module()

try:
    import flag_gems
    _HAVE_FG = True
except Exception as e:
    _HAVE_FG = False
    print(f"[perf_ext] FlagGems not importable: {e}")

WARMUP, REPS = 15, 80
EPS = 1e-5


def bench(fn):
    for _ in range(WARMUP): fn()
    ts = [0.0] * REPS
    for k in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts[k] = (time.perf_counter() - t0) * 1e6
    return float(np.median(ts))


# ──────────────────────────────────────────────────────────────────
# group_norm: spine_raw grid=(G,) vs FlagGems
# ──────────────────────────────────────────────────────────────────
def bench_group_norm():
    print("\n=== group_norm (f16 in, 2D) ===")
    print(f"{'G':>5} {'C':>6} {'raw_us':>10} {'fg_us':>10} {'speedup':>9} {'ok':>4}")
    shapes = [(4,64),(8,64),(16,64),(32,64),(4,128),(8,128),(16,256),(32,256)]
    for G, C in shapes:
        torch.manual_seed(0)
        X = torch.randn(G * C, dtype=torch.float16)
        out = torch.zeros(G * C, dtype=torch.float32)
        def raw():
            _gn.group_norm_host[(G,)](X, out, G, C)
        raw_us = bench(raw)
        # reference
        xf = X.float().reshape(G, C)
        m = xf.mean(1, keepdim=True); v = ((xf-m)**2).mean(1, keepdim=True)
        ref = ((xf-m)/torch.sqrt(v+EPS)).reshape(-1)
        ok = (out-ref).abs().max().item() < 1e-2
        # FlagGems
        if _HAVE_FG:
            try:
                Xt = X.reshape(1, G, C)
                w = torch.ones(C, dtype=torch.float16)
                b2 = torch.zeros(C, dtype=torch.float16)
                def fg(): flag_gems.group_norm(Xt, G, w, b2, EPS)
                fg_us = bench(fg)
                sp = f"{fg_us/raw_us:.2f}x"
            except Exception as e2:
                fg_us, sp = float("nan"), f"?({str(e2)[:10]})"
        else:
            fg_us, sp = float("nan"), "-"
        print(f"{G:>5} {C:>6} {raw_us:>10.1f} {fg_us:>10.1f} {sp:>9} {str(ok):>4}")


# ──────────────────────────────────────────────────────────────────
# cumsum: vectorized 3-phase vs sequential scalar
# ──────────────────────────────────────────────────────────────────
def bench_cumsum():
    print("\n=== cumsum vec vs scalar (f32) ===")
    print(f"{'N':>7} {'vec_us':>10} {'scl_us':>10} {'speedup':>9} {'ok':>4}")
    for N in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        torch.manual_seed(0)
        X = torch.randn(N, dtype=torch.float32)
        out_v = torch.zeros(N, dtype=torch.float32)
        out_s = torch.zeros(N, dtype=torch.float32)
        def run_vec(): return _cs.cumsum_vectorized(X)
        def run_scl():
            _cs1.cumsum_1d_host[(1,)](X, out_s, N)
        vec_us = bench(run_vec)
        scl_us = bench(run_scl)
        ref = torch.cumsum(X, 0)
        res_v = run_vec()
        ok = (res_v - ref).abs().max().item() < 1e-4
        sp = f"{scl_us/vec_us:.2f}x"
        print(f"{N:>7} {vec_us:>10.1f} {scl_us:>10.1f} {sp:>9} {str(ok):>4}")


if __name__ == "__main__":
    bench_group_norm()
    bench_cumsum()
