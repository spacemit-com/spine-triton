"""spine_raw vector_norm — L2 / L1 / L∞ norms and L2-normalize.

  l2_norm(x)   = sqrt(sum(x²))
  l1_norm(x)   = sum(|x|)
  linf_norm(x) = max(|x|)
  normalize(x) = x / l2_norm(x)

All reuse existing primitives (vreduce_sum/max, abs, sqrt, rsqrt, sload).
No new codegen — pure kernel composition.
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
# l2_norm : sqrt(sum(x²))
# ---------------------------------------------------------------------------
@tle.raw_kernel
def l2_norm_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc = acc + vx * vx
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        acc = acc + tx * tx
    tle.sstore(out, 0, tle.sqrt(tle.vreduce_sum(acc)))


@triton.jit
def l2_norm_host(X, out, N):
    _sr_call(l2_norm_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# l1_norm : sum(|x|)
# ---------------------------------------------------------------------------
@tle.raw_kernel
def l1_norm_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc = acc + tle.abs(vx)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        acc = acc + tle.abs(tx)
    tle.sstore(out, 0, tle.vreduce_sum(acc))


@triton.jit
def l1_norm_host(X, out, N):
    _sr_call(l1_norm_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# linf_norm : max(|x|)
# ---------------------------------------------------------------------------
@tle.raw_kernel
def linf_norm_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.abs(tle.vload(X, 0, dtype=f32))
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc = tle.vmax(acc, tle.abs(vx))
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        acc = tle.vmax(acc, tle.abs(tx))
    tle.sstore(out, 0, tle.vreduce_max(acc))


@triton.jit
def linf_norm_host(X, out, N):
    _sr_call(linf_norm_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# normalize : x / l2_norm(x)
# ---------------------------------------------------------------------------
@tle.raw_kernel
def normalize_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    acc = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc = acc + vx * vx
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        acc = acc + tx * tx
    inv = tle.rsqrt(tle.vreduce_sum(acc))   # 1 / sqrt(sum(x²))
    for i in tle.range(0, Nfloor, nvl):
        nx = tle.vload(X, i, dtype=f32)
        tle.vstore(out, i, nx * inv)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t2 = tle.vconfig(N - i, 1)
        mx = tle.vload(X, i, dtype=f32)
        tle.vstore(out, i, mx * inv)


@triton.jit
def normalize_host(X, out, N):
    _sr_call(normalize_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [64, 128, 100, 257])
def test_l2_norm(N):
    torch.manual_seed(1)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    l2_norm_host[(1,)](X, out, N)
    torch.testing.assert_close(out[0], torch.linalg.vector_norm(X, ord=2), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("N", [64, 128, 100])
def test_l1_norm(N):
    torch.manual_seed(2)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    l1_norm_host[(1,)](X, out, N)
    torch.testing.assert_close(out[0], torch.linalg.vector_norm(X, ord=1), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("N", [64, 128, 100])
def test_linf_norm(N):
    torch.manual_seed(3)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    linf_norm_host[(1,)](X, out, N)
    torch.testing.assert_close(out[0], torch.linalg.vector_norm(X, ord=float("inf")), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N", [64, 128, 100, 200])
def test_normalize(N):
    torch.manual_seed(4)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    normalize_host[(1,)](X, out, N)
    ref = X / torch.linalg.vector_norm(X, ord=2)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
