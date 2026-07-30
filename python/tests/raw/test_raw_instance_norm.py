"""spine_raw instance_norm — per-(sample,channel) normalization over spatial dim.

For input [N, C, L]:
  out[n, c, :] = (x[n,c,:] - mean_{n,c}) / sqrt(var_{n,c} + eps)

Normalize each (n, c) slice independently over L spatial elements.
Structurally G = N*C groups of size L — directly reuses group_norm kernel.
Grid = (N*C,), each program is one (n, c) pair.

This is the last norm family member from PLAN_reduce_gap.md L0 list.
"""
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
from importlib.machinery import SourceFileLoader
import os

import triton.language.extra.spine_raw as tle  # noqa: F401
from triton.language.extra.spine_raw import call as _sr_call

_gn = SourceFileLoader(
    "gn_mod", os.path.join(os.path.dirname(__file__), "test_raw_group_norm.py")
).load_module()

group_norm_host = _gn.group_norm_host
EPS = 1e-5


def instance_norm(X: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """instance_norm via group_norm reuse.

    X: [N, C, L] float16.
    G = N*C groups, each of size L.  group_norm_host normalizes each group.
    """
    N, C, L = X.shape
    G = N * C
    X_flat = X.reshape(G, L).contiguous()          # [G, L], each row = one (n,c) slice
    out_flat = torch.zeros(G, L, dtype=torch.float32)
    group_norm_host[(G,)](X_flat.reshape(-1), out_flat.reshape(-1), G, L)
    return out_flat.reshape(N, C, L)


def _ref_instance_norm(X: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    N, C, L = X.shape
    xf = X.float()
    mean = xf.mean(dim=2, keepdim=True)
    var  = ((xf - mean) ** 2).mean(dim=2, keepdim=True)
    return (xf - mean) / torch.sqrt(var + eps)


@pytest.mark.parametrize("N,C,L", [
    (2, 4, 64),
    (4, 2, 128),
    (2, 3, 100),
    (8, 4, 256),
    (3, 2, 200),
])
def test_instance_norm(N, C, L):
    torch.manual_seed(42)
    X = torch.randn(N, C, L, dtype=torch.float16)
    out = instance_norm(X)
    ref = _ref_instance_norm(X)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
