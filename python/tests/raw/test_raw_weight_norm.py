"""spine_raw weight_norm — per-row L2 normalize a weight matrix.

For W [C_out, C_in]:
  g[i]       = ||W[i,:]||_2         (L2 norm per output filter)
  W_norm[i,] = W[i,:] / g[i]        (normalized weight)

Returns both W_norm and g.  Uses normalize() pattern with dual output.
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
def weight_norm_kernel(
    W: tle.mem(f32),
    W_norm: tle.mem(f32, out=True),
    g_out: tle.mem(f32, out=True),
    C_out: tle.index, C_in: tle.index, row: tle.index
):
    """One program per output filter (row). Computes g and W_norm for that row."""
    nvl = tle.vconfig(-1, 1)
    Nfloor = (C_in // nvl) * nvl
    base = row * C_in

    # Accumulate sum(w²)
    acc_sq = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vw = tle.vload(W, base + i, dtype=f32)
        acc_sq = acc_sq + vw * vw
    for i in tle.range(Nfloor, C_in, nvl):
        nvl_t = tle.vconfig(C_in - i, 1)
        tw = tle.vload(W, base + i, dtype=f32)
        acc_sq = acc_sq + tw * tw

    g = tle.sqrt(tle.vreduce_sum(acc_sq))   # L2 norm (scalar)
    inv_g = tle.rsqrt(tle.vreduce_sum(acc_sq))   # 1/g

    tle.sstore(g_out, row, g)

    # Normalize and write W_norm
    for i in tle.range(0, Nfloor, nvl):
        vw2 = tle.vload(W, base + i, dtype=f32)
        tle.vstore(W_norm, base + i, vw2 * inv_g)
    for i in tle.range(Nfloor, C_in, nvl):
        nvl_t2 = tle.vconfig(C_in - i, 1)
        tw2 = tle.vload(W, base + i, dtype=f32)
        tle.vstore(W_norm, base + i, tw2 * inv_g)


@triton.jit
def weight_norm_host(W, W_norm, g_out, C_out, C_in):
    row = tl.program_id(0)
    if row < C_out:
        _sr_call(weight_norm_kernel, outputs=[], inputs=[W, W_norm, g_out, C_out, C_in, row])


@pytest.mark.parametrize("C_out,C_in", [(4, 64), (8, 128), (3, 100), (16, 256)])
def test_weight_norm(C_out, C_in):
    torch.manual_seed(42)
    W = torch.randn(C_out, C_in, dtype=torch.float32)
    W_norm = torch.zeros(C_out, C_in, dtype=torch.float32)
    g_out  = torch.zeros(C_out, dtype=torch.float32)
    weight_norm_host[(C_out,)](W.reshape(-1), W_norm.reshape(-1), g_out, C_out, C_in)

    ref_g     = W.norm(dim=1, p=2)          # per-row L2 norm
    ref_wnorm = W / ref_g.unsqueeze(1)      # per-row normalize
    torch.testing.assert_close(g_out, ref_g,     rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(W_norm, ref_wnorm, rtol=1e-4, atol=1e-4)
