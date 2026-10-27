"""smt_rvisa L1 reduce primitives: vreduce_max / vreduce_min / vreduce_mul.

These were almost free — create_vector_reduction already supported maxf/minf/mul;
only codegen marker+handler was missing.
"""
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.smt_rvisa as sr
from triton.language.extra.smt_rvisa import call as _sr_call

f32 = sr.f32


# ---------------------------------------------------------------------------
# amax_1d
# ---------------------------------------------------------------------------
@sr.raw_kernel
def amax_1d_kernel(X: sr.mem(f32), out: sr.mem(f32, out=True), N: sr.index):
    nvl = sr.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = sr.vload(X, 0, dtype=f32)  # seed with first tile
    for i in sr.range(0, Nfloor, nvl):
        vx = sr.vload(X, i, dtype=f32)
        acc = sr.vmax(acc, vx)  # element-wise max across tiles
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tx = sr.vload(X, i, dtype=f32)
        acc = sr.vmax(acc, tx)
    sr.sstore(out, 0, sr.vreduce_max(acc))  # L1: horizontal max


@triton.jit
def amax_1d_host(X, out, N):
    _sr_call(amax_1d_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# amin_1d
# ---------------------------------------------------------------------------
@sr.raw_kernel
def amin_1d_kernel(X: sr.mem(f32), out: sr.mem(f32, out=True), N: sr.index):
    nvl = sr.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = sr.vload(X, 0, dtype=f32)
    for i in sr.range(0, Nfloor, nvl):
        vx = sr.vload(X, i, dtype=f32)
        acc = sr.vmin(acc, vx)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tx = sr.vload(X, i, dtype=f32)
        acc = sr.vmin(acc, tx)
    sr.sstore(out, 0, sr.vreduce_min(acc))  # L1: horizontal min


@triton.jit
def amin_1d_host(X, out, N):
    _sr_call(amin_1d_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# prod_1d (scalar product of all elements)
# ---------------------------------------------------------------------------
@sr.raw_kernel
def prod_1d_kernel(X: sr.mem(f32), out: sr.mem(f32, out=True), N: sr.index):
    nvl = sr.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    # Initialise accumulator to 1.0 (identity for mul)
    acc = sr.vzero(f32) + 1.0  # vzero + 1.0 broadcast
    for i in sr.range(0, Nfloor, nvl):
        vx = sr.vload(X, i, dtype=f32)
        acc = acc * vx
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tx = sr.vload(X, i, dtype=f32)
        acc = acc * tx
    sr.sstore(out, 0, sr.vreduce_mul(acc))  # L1: horizontal product


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
    amax_1d_host[(1, )](X, out, N)
    torch.testing.assert_close(out[0], X.max(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N", [64, 128, 100])
def test_amin_1d(N):
    torch.manual_seed(2)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    amin_1d_host[(1, )](X, out, N)
    torch.testing.assert_close(out[0], X.min(), rtol=1e-5, atol=1e-5)


@pytest.mark.xfail(
    reason="vreduce_mul lowers to vector.reduction<mul>; K3 llc has no "
    "hardware vfredmul and the scalar expansion path crashes "
    "('getOrderedReduction' assertion) — same gap as x86. "
    "Work-around: compute product via a sequential scalar loop.", strict=True)
@pytest.mark.parametrize("N", [64])
def test_prod_1d(N):
    # Small values to avoid overflow in 64-element product
    X = torch.full((N, ), 1.01, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    prod_1d_host[(1, )](X, out, N)
    ref = X.prod()
    torch.testing.assert_close(out[0], ref, rtol=1e-2, atol=1e-3)
