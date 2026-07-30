"""spine_raw activations — relu / sigmoid / gelu from existing primitives.

relu    : vmax(x, 0)
sigmoid : 1 / (1 + exp(-x))
gelu    : 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x³)))
         where tanh(y) = (exp(2y)-1)/(exp(2y)+1) — composable from vexp

No new C++ bindings needed.
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
_SQRT_2_PI = 0.7978845608028654  # sqrt(2/pi)
_GELU_COEF = 0.044715


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------
@tle.raw_kernel
def relu_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    zero = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        tle.vstore(out, i, tle.vmax(vx, zero))
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        tle.vstore(out, i, tle.vmax(tx, zero))

@triton.jit
def relu_host(X, out, N):
    _sr_call(relu_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------
@tle.raw_kernel
def sigmoid_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        tle.vstore(out, i, 1.0 / (1.0 + tle.vexp(-vx)))
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        tle.vstore(out, i, 1.0 / (1.0 + tle.vexp(-tx)))

@triton.jit
def sigmoid_host(X, out, N):
    _sr_call(sigmoid_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# gelu — tanh approximation (Hendrycks & Gimpel 2016 / PyTorch default)
# ---------------------------------------------------------------------------
@tle.raw_kernel
def gelu_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        inner = _SQRT_2_PI * (vx + _GELU_COEF * vx * vx * vx)
        e2 = tle.vexp(inner + inner)          # exp(2 * inner) for tanh
        tanh_v = (e2 - 1.0) / (e2 + 1.0)     # tanh via exp
        tle.vstore(out, i, 0.5 * vx * (1.0 + tanh_v))
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        inner2 = _SQRT_2_PI * (tx + _GELU_COEF * tx * tx * tx)
        e22 = tle.vexp(inner2 + inner2)
        tanh2 = (e22 - 1.0) / (e22 + 1.0)
        tle.vstore(out, i, 0.5 * tx * (1.0 + tanh2))

@triton.jit
def gelu_host(X, out, N):
    _sr_call(gelu_kernel, outputs=[], inputs=[X, out, N])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [64, 128, 100, 513])
def test_relu(N):
    torch.manual_seed(1)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    relu_host[(1,)](X, out, N)
    torch.testing.assert_close(out, torch.relu(X), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N", [64, 128, 100, 513])
def test_sigmoid(N):
    torch.manual_seed(2)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    sigmoid_host[(1,)](X, out, N)
    torch.testing.assert_close(out, torch.sigmoid(X), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N", [64, 128, 100, 513])
def test_gelu(N):
    torch.manual_seed(3)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    gelu_host[(1,)](X, out, N)
    ref = torch.nn.functional.gelu(X, approximate='tanh')
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
