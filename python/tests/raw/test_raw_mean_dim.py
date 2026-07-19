"""spine_raw mean_dim — 2D reduce along dim=1 giving per-row means.

torch.mean(x, dim=1) → values[M]  for input [M, N].
Same grid=(M,) pattern as max_dim/sum_2d; each program handles one row.
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
def mean_dim1_kernel(
    X: tle.mem(f32), out: tle.mem(f32, out=True),
    M: tle.index, N: tle.index, row: tle.index
):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    base = row * N
    acc = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, base + i, dtype=f32)
        acc = acc + vx
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, base + i, dtype=f32)
        acc = acc + tx
    tle.vstore(out, row, tle.vreduce_sum(acc) / N)


@triton.jit
def mean_dim1_host(X, out, M, N):
    row = tl.program_id(0)
    if row < M:
        _sr_call(mean_dim1_kernel, outputs=[], inputs=[X, out, M, N, row])


@pytest.mark.parametrize("M,N", [(4, 64), (8, 128), (3, 100), (16, 200)])
def test_mean_dim1(M, N):
    torch.manual_seed(42)
    X = torch.randn(M, N, dtype=torch.float32)
    out = torch.zeros(M, dtype=torch.float32)
    mean_dim1_host[(M,)](X.contiguous().reshape(-1), out, M, N)
    ref = X.mean(dim=1)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
