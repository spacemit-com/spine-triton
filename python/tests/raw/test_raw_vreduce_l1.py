"""spine_raw L1 reduce primitives: vreduce_max / vreduce_min / vreduce_mul.

These were almost free — create_vector_reduction already supported maxf/minf/mul;
only codegen marker+handler was missing.
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


# ---------------------------------------------------------------------------
# amax_1d
# ---------------------------------------------------------------------------
@tle.raw_kernel
def amax_1d_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vload(X, 0, dtype=f32)   # seed with first tile
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc = tle.vmax(acc, vx)         # element-wise max across tiles
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        acc = tle.vmax(acc, tx)
    tle.vstore(out, 0, tle.vreduce_max(acc))   # L1: horizontal max


@triton.jit
def amax_1d_host(X, out, N):
    _sr_call(amax_1d_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# amin_1d
# ---------------------------------------------------------------------------
@tle.raw_kernel
def amin_1d_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vload(X, 0, dtype=f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc = tle.vmin(acc, vx)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        acc = tle.vmin(acc, tx)
    tle.vstore(out, 0, tle.vreduce_min(acc))   # L1: horizontal min


@triton.jit
def amin_1d_host(X, out, N):
    _sr_call(amin_1d_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# prod_1d (scalar product of all elements)
# ---------------------------------------------------------------------------
@tle.raw_kernel
def prod_1d_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    # Initialise accumulator to 1.0 (identity for mul)
    acc = tle.vzero(f32) + 1.0          # vzero + 1.0 broadcast
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc = acc * vx
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        acc = acc * tx
    tle.vstore(out, 0, tle.vreduce_mul(acc))   # L1: horizontal product


@triton.jit
def prod_1d_host(X, out, N):
    _sr_call(prod_1d_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [64, 128, 100])
def test_amax_1d(N):
    torch.manual_seed(1)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    amax_1d_host[(1,)](X, out, N)
    torch.testing.assert_close(out[0], X.max(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N", [64, 128, 100])
def test_amin_1d(N):
    torch.manual_seed(2)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    amin_1d_host[(1,)](X, out, N)
    torch.testing.assert_close(out[0], X.min(), rtol=1e-5, atol=1e-5)


@pytest.mark.xfail(reason="vreduce_mul lowers to vector.reduction<mul>; K3 llc has no "
                          "hardware vfredmul and the scalar expansion path crashes "
                          "('getOrderedReduction' assertion) — same gap as x86. "
                          "Work-around: compute product via a sequential scalar loop.",
                   strict=True)
@pytest.mark.parametrize("N", [64])
def test_prod_1d(N):
    # Small values to avoid overflow in 64-element product
    X = torch.full((N,), 1.01, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    prod_1d_host[(1,)](X, out, N)
    ref = X.prod()
    torch.testing.assert_close(out[0], ref, rtol=1e-2, atol=1e-3)
