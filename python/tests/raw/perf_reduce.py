"""reduce-family perf: spine_raw single-pass streaming vs FlagGems, f32.

Validates PLAN_reduce_gap.md's core claim — that spine_raw's single-pass
streaming reduce (register-resident accumulator, one memory sweep) beats
FlagGems' multi-pass / discrete-autotune kernels for reduce-shaped ops.

Ops: rms_norm, layernorm, softmax. 1D vectors (single row).
Each impl: WARMUP warmup + REPS iters, median us; speedup vs FlagGems;
correctness checked against torch reference.

Run on K3:
  PYTHONPATH -> worktree build-riscv64 + FlagGems src + triton site-packages
  GEMS_VENDOR=spacemit
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
import triton.language.extra.spine_raw as tle  # noqa: F401  (registers backend)

WARMUP, REPS = 20, 100
_TESTS = os.path.dirname(__file__)

# Reuse the raw kernels from the test files (no absolute paths).
_rms = SourceFileLoader("rms_mod", os.path.join(_TESTS, "test_raw_mean_rmsnorm.py")).load_module()
_ln = SourceFileLoader("ln_mod", os.path.join(_TESTS, "test_raw_layernorm.py")).load_module()
_sm = SourceFileLoader("sm_mod", os.path.join(_TESTS, "test_raw_softmax.py")).load_module()

try:
    import flag_gems
    _HAVE_FG = True
except Exception as e:  # noqa: BLE001
    _HAVE_FG = False
    print(f"[perf_reduce] FlagGems not importable ({e}); FlagGems column skipped.")

SHAPES = [128, 256, 512, 1024, 2048, 4096, 8192]
EPS = 1e-5


def bench(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(ts))


def run_op(name, raw_fn, fg_fn, ref_fn, make_out, in_dtype=torch.float32):
    print(f"\n=== {name} (in={in_dtype}, 1D) ===")
    hdr = f"{'N':>7} {'raw_us':>10} {'fg_us':>10} {'speedup':>9} {'ok':>4}"
    print(hdr)
    for N in SHAPES:
        torch.manual_seed(0)
        X = torch.randn(N, dtype=in_dtype)
        out = make_out(N)
        raw_ms = bench(lambda: raw_fn(X, out, N))
        ref = ref_fn(X)
        ok = (out.float() - ref).abs().max().item() < 1e-2
        if _HAVE_FG and fg_fn is not None:
            try:
                fg_ms = bench(lambda: fg_fn(X))
                sp = f"{fg_ms / raw_ms:.2f}x"
            except Exception as e:  # noqa: BLE001
                fg_ms, sp = float("nan"), f"err:{str(e)[:12]}"
        else:
            fg_ms, sp = float("nan"), "-"
        print(f"{N:>7} {raw_ms:>10.1f} {fg_ms:>10.1f} {sp:>9} {str(ok):>4}")


def main():
    # rms_norm
    def rms_raw(X, out, N):
        _rms.rms_norm_1d_host[(1,)](X, out, N)
    def rms_ref(X):
        xf = X.float(); return xf / torch.sqrt((xf * xf).mean())
    def rms_fg(X):
        w = torch.ones_like(X)
        return flag_gems.rms_norm(X.unsqueeze(0), [X.shape[0]], w, EPS)
    run_op("rms_norm", rms_raw, rms_fg if _HAVE_FG else None, rms_ref, lambda N: torch.zeros(N, dtype=torch.float32), in_dtype=torch.float16)

    # layernorm
    def ln_raw(X, out, N):
        _ln.layernorm_1d_host[(1,)](X, out, N)
    def ln_ref(X):
        xf = X.float(); m = xf.mean(); v = ((xf - m) ** 2).mean()
        return (xf - m) / torch.sqrt(v + _ln.EPS)
    def ln_fg(X):
        w = torch.ones_like(X); b = torch.zeros_like(X)
        return flag_gems.layer_norm(X.unsqueeze(0), [X.shape[0]], w, b, _ln.EPS)
    run_op("layernorm", ln_raw, ln_fg if _HAVE_FG else None, ln_ref, lambda N: torch.zeros(N, dtype=torch.float32), in_dtype=torch.float16)

    # softmax
    def sm_raw(X, out, N):
        _sm.softmax_1d_host[(1,)](X, out, N)
    def sm_ref(X):
        return torch.softmax(X.float(), dim=0)
    def sm_fg(X):
        return flag_gems.softmax(X.unsqueeze(0), dim=-1)
    run_op("softmax", sm_raw, sm_fg if _HAVE_FG else None, sm_ref, lambda N: torch.zeros(N, dtype=torch.float32))


if __name__ == "__main__":
    main()
