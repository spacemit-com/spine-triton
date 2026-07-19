"""spine_raw silu/swish — x * sigmoid(x), composable from existing primitives.

silu(x) = x * (1 / (1 + exp(-x)))
         = x * sigmoid(x)

Used in modern LLM activations (LLaMA, Mistral use SwiGLU = silu * linear).
Demonstrates that non-trivial activation functions compose from vexp + scalar
arithmetic without new primitives.
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
def silu_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        sig = 1.0 / (1.0 + tle.vexp(-vx))   # sigmoid(x)
        tle.vstore(out, i, vx * sig)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32)
        sig2 = 1.0 / (1.0 + tle.vexp(-tx))
        tle.vstore(out, i, tx * sig2)


@triton.jit
def silu_host(X, out, N):
    _sr_call(silu_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [64, 128, 256, 100, 513])
def test_silu(N):
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    silu_host[(1,)](X, out, N)
    ref = torch.nn.functional.silu(X)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
