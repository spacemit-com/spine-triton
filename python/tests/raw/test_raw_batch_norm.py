"""spine_raw batch_norm — normalize over the batch (N) dimension per channel.

For input [N, C]:
  mean[c] = sum(x[:, c]) / N
  var[c]  = sum((x[:, c] - mean[c])^2) / N
  out[n,c] = (x[n,c] - mean[c]) / sqrt(var[c] + eps)

Path A (efficient): transpose [N,C] → [C,N] on the host (one .t().contiguous()
copy) then reuse the group_norm 3-pass reduce, grid=(C,), each program
normalizes one channel over N elements.  No codegen changes.

This is a pure composition test: demonstrates that the 2D-grid layernorm
pattern composes cleanly onto batch_norm via a host-side layout swap.
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

f16 = tle.f16
f32 = tle.f32
EPS = 1e-5

# Reuse the group_norm kernel directly — it normalizes each "group" (row) of C
# elements. By transposing [N,C]→[C,N] we make each channel a contiguous row.
_gn = SourceFileLoader(
    "gn_mod", os.path.join(os.path.dirname(__file__), "test_raw_group_norm.py")
).load_module()

group_norm_host = _gn.group_norm_host     # @triton.jit wrapper
group_norm_kernel = _gn.group_norm_kernel  # @tle.raw_kernel


def batch_norm(X: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """batch_norm via transpose + group_norm.

    X: [N, C] float16.
    Returns: [N, C] float32 normalized values.
    """
    N, C = X.shape
    # Transpose [N, C] → [C, N] so each channel is a contiguous row
    X_t = X.t().contiguous()                 # [C, N], row = one channel's batch
    out_t = torch.zeros(C, N, dtype=torch.float32)
    # grid=(C,): each program normalizes one channel (row of length N)
    group_norm_host[(C,)](X_t.reshape(-1), out_t.reshape(-1), C, N)
    # Transpose back [C, N] → [N, C]
    return out_t.t().contiguous()


def _ref_batch_norm(X: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    xf = X.float()
    mean = xf.mean(dim=0, keepdim=True)      # [1, C]
    var  = ((xf - mean) ** 2).mean(dim=0, keepdim=True)
    return (xf - mean) / torch.sqrt(var + eps)


@pytest.mark.parametrize("N,C", [
    (4, 64), (8, 128), (16, 64), (3, 100), (8, 200)
])
def test_batch_norm(N, C):
    torch.manual_seed(42)
    X = torch.randn(N, C, dtype=torch.float16)
    out = batch_norm(X)
    ref = _ref_batch_norm(X)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
