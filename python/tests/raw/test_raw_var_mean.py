"""spine_raw var_mean — single-pass variance + mean via E[x²]-mean².

Accumulates sum(x) and sum(x²) simultaneously in one sweep, then:
  mean = sum(x) / N
  var  = sum(x²)/N - mean²

Returns both scalars in a single kernel launch — 1 memory sweep vs
2 separate passes (mean-then-variance). Validates the E[x²]-mean² formula
introduced in the batch_norm/group_norm fix (correct for fill-0 padded lanes).
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
EPS = 1e-5


@tle.raw_kernel
def var_mean_1d_kernel(
    X: tle.mem(f32),
    var_out: tle.mem(f32, out=True),
    mean_out: tle.mem(f32, out=True),
    N: tle.index
):
    """Single-pass: accumulate sum(x) and sum(x²) simultaneously."""
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl

    acc_sum = tle.vzero(f32)   # accumulates x  → for mean
    acc_sq = tle.vzero(f32)    # accumulates x² → for E[x²]

    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc_sum = acc_sum + vx
        acc_sq  = acc_sq  + vx * vx

    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)        # fill=0 → 0²=0 ✓
        acc_sum = acc_sum + tx
        acc_sq  = acc_sq  + tx * tx

    mean = tle.vreduce_sum(acc_sum) / N        # f32 scalar
    ex2  = tle.vreduce_sum(acc_sq)  / N        # E[x²]
    var  = ex2 - mean * mean                   # Var(x) = E[x²] - mean²

    tle.vstore(var_out,  0, var)
    tle.vstore(mean_out, 0, mean)


@triton.jit
def var_mean_1d_host(X, var_out, mean_out, N):
    _sr_call(var_mean_1d_kernel, outputs=[], inputs=[X, var_out, mean_out, N])


@pytest.mark.parametrize("N", [64, 128, 256, 100, 257])
def test_var_mean_1d(N):
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float32)
    var_out  = torch.zeros(1, dtype=torch.float32)
    mean_out = torch.zeros(1, dtype=torch.float32)
    var_mean_1d_host[(1,)](X, var_out, mean_out, N)

    ref_mean = X.mean()
    ref_var  = X.var(unbiased=False)   # population variance (divide by N)
    torch.testing.assert_close(mean_out[0], ref_mean, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(var_out[0],  ref_var,  rtol=1e-4, atol=1e-4)
