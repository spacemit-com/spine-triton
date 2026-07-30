"""spine_raw group_norm — per-group layernorm over a [G, C] layout.

group_norm normalizes each group independently: for input reshaped to
[num_groups, group_size], each group g gets (x - mean_g) / sqrt(var_g + eps).
This is layernorm applied per-row, driven by grid=(G,) with row = program_id.

Validates that the layernorm 3-pass reduce composes onto a 2D grid the same
way max_dim did.
"""
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32
EPS = 1e-5


@tle.raw_kernel
def group_norm_kernel(
    X: tle.mem(f16), out: tle.mem(f32, out=True),
    G: tle.index, C: tle.index, row: tle.index
):
    """Normalize one group (row) of C elements: (x - mean) / sqrt(var + eps)."""
    nvl = tle.vconfig(-1, 1)
    Cfloor = (C // nvl) * nvl
    base = row * C

    # 趟1: sum(x)
    acc1 = tle.vzero(f32)
    for i in tle.range(0, Cfloor, nvl):
        va = tle.cast(tle.vload(X, base + i), f32)
        acc1 = acc1 + va
    for i in tle.range(Cfloor, C, nvl):
        nvl_t1 = tle.vconfig(C - i, 1)
        ta = tle.cast(tle.vload(X, base + i), f32)
        acc1 = acc1 + ta
    mean = tle.vreduce_sum(acc1) / C

    # 趟2: sum(x²) — use E[x²]-mean² to compute variance.
    # Avoids (0-mean)²=mean² inflation from fill-0 padded lanes (0²=0 contributes nothing).
    acc2 = tle.vzero(f32)
    for i in tle.range(0, Cfloor, nvl):
        vb = tle.cast(tle.vload(X, base + i), f32)
        acc2 = acc2 + vb * vb
    for i in tle.range(Cfloor, C, nvl):
        nvl_t2 = tle.vconfig(C - i, 1)
        tb = tle.cast(tle.vload(X, base + i), f32)   # fill=0: 0²=0, no inflation
        acc2 = acc2 + tb * tb
    var = tle.vreduce_sum(acc2) / C - mean * mean     # E[x²] - mean² = Var(x)
    scale = tle.rsqrt(var + EPS)

    # 趟3: (x - mean) * scale
    for i in tle.range(0, Cfloor, nvl):
        nx = tle.cast(tle.vload(X, base + i), f32)
        tle.vstore(out, base + i, (nx - mean) * scale)
    for i in tle.range(Cfloor, C, nvl):
        nvl_t3 = tle.vconfig(C - i, 1)
        mx = tle.cast(tle.vload(X, base + i), f32)
        tle.vstore(out, base + i, (mx - mean) * scale)


@triton.jit
def group_norm_host(X, out, G, C):
    row = tl.program_id(0)
    if row < G:
        _sr_call(group_norm_kernel, outputs=[], inputs=[X, out, G, C, row])


def _ref_group_norm(X: torch.Tensor, G: int, C: int) -> torch.Tensor:
    xf = X.float().reshape(G, C)
    mean = xf.mean(dim=1, keepdim=True)
    var = ((xf - mean) ** 2).mean(dim=1, keepdim=True)
    return ((xf - mean) / torch.sqrt(var + EPS)).reshape(-1)


@pytest.mark.parametrize("G,C", [(4, 64), (8, 128), (3, 100), (16, 256), (2, 200)])
def test_group_norm(G, C):
    torch.manual_seed(42)
    X = torch.randn(G * C, dtype=torch.float16)
    out = torch.zeros(G * C, dtype=torch.float32)
    group_norm_host[(G,)](X, out, G, C)
    ref = _ref_group_norm(X, G, C)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
