"""spine_raw cross_entropy — fused negative log-likelihood.

kernel: loss = -log_softmax[target]
             = log(sum(exp(x - max))) + max - x[target]

Uses: vreduce_max + vexp + vreduce_sum + vlog + vscalar (all available primitives).
Single scalar output — no output vector materialization.
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
def cross_entropy_1d_kernel(
    X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index, target: tle.index
):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl

    # ── 趟1: max(x) ───────────────────────────────────────────────────────
    acc_max = tle.vload(X, 0, dtype=f32)
    for i in tle.range(0, Nfloor, nvl):
        va = tle.vload(X, i, dtype=f32)
        acc_max = tle.vmax(acc_max, va)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t1 = tle.vconfig(N - i, 1)
        ta = tle.vload(X, i, dtype=f32)
        acc_max = tle.vmax(acc_max, ta)
    xmax = tle.vreduce_max(acc_max)

    # ── 趟2: sum(exp(x - max)) ────────────────────────────────────────────
    acc_sum = tle.vzero(f32)
    for i in tle.range(0, Nfloor, nvl):
        vb = tle.vload(X, i, dtype=f32)
        acc_sum = acc_sum + tle.vexp(vb - xmax)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t2 = tle.vconfig(N - i, 1)
        tb = tle.vload(X, i, dtype=f32, fill=-1e38)
        acc_sum = acc_sum + tle.vexp(tb - xmax)
    denom = tle.vreduce_sum(acc_sum)

    # ── 单元素提取 + 计算 loss ─────────────────────────────────────────────
    xt = tle.vscalar(X, target, dtype=f32)   # vscalar: scalar load at dynamic idx
    loss = tle.vlog(denom) + xmax - xt       # = -log_softmax[target]
    tle.vstore(out, 0, loss)


@triton.jit
def cross_entropy_1d_host(X, out, N, target):
    _sr_call(cross_entropy_1d_kernel, outputs=[], inputs=[X, out, N, target])


def _ref_cross_entropy(logits: torch.Tensor, target: int) -> float:
    return torch.nn.functional.cross_entropy(
        logits.unsqueeze(0), torch.tensor([target])
    ).item()


@pytest.mark.parametrize("N,target", [
    (64, 0), (64, 32), (64, 63),
    (128, 10), (256, 100),
])
def test_cross_entropy_1d(N, target):
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    cross_entropy_1d_host[(1,)](X, out, N, target)
    ref = _ref_cross_entropy(X, target)
    torch.testing.assert_close(out[0].item(), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N,target", [(100, 50), (200, 0)])
def test_cross_entropy_1d_arb(N, target):
    torch.manual_seed(7)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    cross_entropy_1d_host[(1,)](X, out, N, target)
    ref = _ref_cross_entropy(X, target)
    torch.testing.assert_close(out[0].item(), ref, rtol=1e-4, atol=1e-4)
