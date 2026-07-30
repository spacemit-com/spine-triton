"""spine_raw log_softmax — numerically stable log(softmax(x)).

Kernel: out[i] = log(exp(x[i] - max(x)) / sum(exp(x - max(x))))
             = (x[i] - max(x)) - log(sum(exp(x - max(x))))

Uses: vreduce_max + vexp + vreduce_sum + vlog (all L1 primitives).
Fused: avoids materializing softmax output and then re-reading it for log.
"""
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f32 = tle.f32


@tle.raw_kernel
def log_softmax_1d_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl

    # ── 趟1: max(x) ───────────────────────────────────────────────────────
    acc_max = tle.vload(X, 0, dtype=f32)
    for i in tle.range(0, Nfloor, nvl):
        va = tle.vload(X, i, dtype=f32)
        acc_max = tle.vmax(acc_max, va)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t1 = tle.vconfig(N - i, 1)
        ta = tle.vload(X, i, dtype=f32)
        acc_max = tle.vmax(acc_max, ta)
    xmax = tle.vreduce_max(acc_max)

    # ── 趟2: sum(exp(x - max)) ────────────────────────────────────────────
    acc_sum = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vb = tle.vload(X, i, dtype=f32)
        acc_sum = acc_sum + tle.vexp(vb - xmax)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t2 = tle.vconfig(N - i, 1)
        tb = tle.vload(X, i, dtype=f32, fill=-1e38)
        acc_sum = acc_sum + tle.vexp(tb - xmax)
    denom = tle.vreduce_sum(acc_sum)

    # ── 趟3: (x - max) - log(denom) ───────────────────────────────────────
    # log_softmax[i] = log(exp(x[i]-max)/denom) = (x[i]-max) - log(denom)
    log_denom = tle.vlog(denom)
    for i in tle.range(0, Nfloor, nvl):
        vc = tle.vload(X, i, dtype=f32)
        tle.vstore(out, i, (vc - xmax) - log_denom)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t3 = tle.vconfig(N - i, 1)
        tc = tle.vload(X, i, dtype=f32)
        tle.vstore(out, i, (tc - xmax) - log_denom)


@triton.jit
def log_softmax_1d_host(X, out, N):
    _sr_call(log_softmax_1d_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [64, 128, 256])
def test_log_softmax_1d(N):
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    log_softmax_1d_host[(1,)](X, out, N)
    ref = torch.log_softmax(X, dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("N", [100, 200])
def test_log_softmax_1d_arb(N):
    torch.manual_seed(7)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    log_softmax_1d_host[(1,)](X, out, N)
    ref = torch.log_softmax(X, dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
