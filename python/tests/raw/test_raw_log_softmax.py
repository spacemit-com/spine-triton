"""smt_rvisa log_softmax — numerically stable log(softmax(x)).

Kernel: out[i] = log(exp(x[i] - max(x)) / sum(exp(x - max(x))))
             = (x[i] - max(x)) - log(sum(exp(x - max(x))))

Uses: vreduce_max + vexp + vreduce_sum + vlog (all L1 primitives).
Fused: avoids materializing softmax output and then re-reading it for log.
"""
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.smt_rvisa as sr
from triton.language.extra.smt_rvisa import call as _sr_call

f32 = sr.f32


@sr.raw_kernel
def log_softmax_1d_kernel(X: sr.mem(f32), out: sr.mem(f32, out=True), N: sr.index):
    nvl = sr.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl

    # ── 趟1: max(x) ───────────────────────────────────────────────────────
    acc_max = sr.vload(X, 0, dtype=f32)
    for i in sr.range(0, Nfloor, nvl):
        va = sr.vload(X, i, dtype=f32)
        acc_max = sr.vmax(acc_max, va)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        ta = sr.vload(X, i, dtype=f32)
        acc_max = sr.vmax(acc_max, ta)
    xmax = sr.vreduce_max(acc_max)

    # ── 趟2: sum(exp(x - max)) ────────────────────────────────────────────
    acc_sum = sr.vzero(f32)
    for i in sr.range(0, Nfloor, nvl):
        vb = sr.vload(X, i, dtype=f32)
        acc_sum = acc_sum + sr.vexp(vb - xmax)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tb = sr.vload(X, i, dtype=f32, fill=-1e38)
        acc_sum = acc_sum + sr.vexp(tb - xmax)
    denom = sr.vreduce_sum(acc_sum)

    # ── 趟3: (x - max) - log(denom) ───────────────────────────────────────
    # log_softmax[i] = log(exp(x[i]-max)/denom) = (x[i]-max) - log(denom)
    log_denom = sr.vlog(denom)
    for i in sr.range(0, Nfloor, nvl):
        vc = sr.vload(X, i, dtype=f32)
        sr.vstore(out, i, (vc - xmax) - log_denom)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tc = sr.vload(X, i, dtype=f32)
        sr.vstore(out, i, (tc - xmax) - log_denom)


@triton.jit
def log_softmax_1d_host(X, out, N):
    _sr_call(log_softmax_1d_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [64, 128, 256])
def test_log_softmax_1d(N):
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    log_softmax_1d_host[(1, )](X, out, N)
    ref = torch.log_softmax(X, dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("N", [100, 200])
def test_log_softmax_1d_arb(N):
    torch.manual_seed(7)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    log_softmax_1d_host[(1, )](X, out, N)
    ref = torch.log_softmax(X, dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
