"""spine_raw cumsum (L4 scan) — sequential scalar prefix sum.

cumsum[i] = sum(x[0..i]). Unlike the reduce family ("reduce a vector to a
scalar"), scan emits one output per position. The simplest correct form is a
scalar scf.for with a running f32 accumulator carried as an iter_arg:

    acc = 0
    for i in range(N):
        acc = acc + x[i]     # scalar load, scalar add
        out[i] = acc         # scalar store

This is O(N) sequential (no vector parallelism), but proves the scan capability
end to end using memref.load/store + scf.for scalar iter_args — all inside the
scalable-lowering whitelist. A vectorized block-scan + block fan-out is future
work.
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
def cumsum_1d_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)                 # sets VL (vzero needs it)
    acc = tle.vreduce_sum(tle.vzero(f32))   # 0.0 as an f32 scalar (scan seed)
    for i in tle.range(0, N, 1):
        xi = tle.vscalar(X, i, dtype=f32)
        acc = acc + xi
        tle.vstore(out, i, acc)


@triton.jit
def cumsum_1d_host(X, out, N):
    _sr_call(cumsum_1d_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [16, 64, 100, 257])
def test_cumsum_1d(N):
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)
    cumsum_1d_host[(1,)](X, out, N)
    ref = torch.cumsum(X, dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
