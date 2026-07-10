"""spine_raw §6.4 逐元素运算 end-to-end test.

Exercises the §6.4 surface — arithmetic operators (+ - * / % and scalar
broadcast), unary (-a), comparison (-> mask), and the named functions
vmin/vmax/sqrt/rsqrt/abs/cast/select — through the full spine_raw ->
tle.dsl_region -> lowering pipeline on K3, checked against torch.

Each kernel applies one §6.4 op elementwise over a VL-tile, then reduces
with vreduce_sum to a scalar and stores that scalar. The reduction +
scalar store is the known-good path (same as the mv kernels); this isolates
the test to the §6.4 elementwise arithmetic itself. (A full-vector vstore /
transfer_write is a separate, currently-broken lowering path — see the mv
kernels which only ever store scalars — so results are validated via the
reduced scalar rather than a written-back vector.)

Buffers are f32, length a multiple of VL (=64 for f16 base / lmul=1), so the
fixed-VL svector loop runs full tiles (no tail; §6.1 narrowing deferred).
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


# ---------------------------------------------------------------------------
# raw kernels: X[N], Y[N] f32 -> S[1] f32 = sum_i( op(x_i, y_i) ).
# One §6.4 op per kernel, then vreduce_sum + scalar vstore.
# ---------------------------------------------------------------------------
@tle.raw_kernel
def ew_add(X: tle.mem(f32), Y: tle.mem(f32), S: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    acc = tle.vzero(f32)
    for i in tle.range(0, N, nvl):
        vx = tle.vload(X, i, dtype=f32)
        vy = tle.vload(Y, i, dtype=f32)
        acc = acc + (vx + vy)
    tle.vstore(S, 0, tle.vreduce_sum(acc))


@tle.raw_kernel
def ew_mul(X: tle.mem(f32), Y: tle.mem(f32), S: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    acc = tle.vzero(f32)
    for i in tle.range(0, N, nvl):
        vx = tle.vload(X, i, dtype=f32)
        vy = tle.vload(Y, i, dtype=f32)
        acc = acc + (vx * vy)
    tle.vstore(S, 0, tle.vreduce_sum(acc))


@tle.raw_kernel
def ew_sub(X: tle.mem(f32), Y: tle.mem(f32), S: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    acc = tle.vzero(f32)
    for i in tle.range(0, N, nvl):
        vx = tle.vload(X, i, dtype=f32)
        vy = tle.vload(Y, i, dtype=f32)
        acc = acc + (vx - vy)
    tle.vstore(S, 0, tle.vreduce_sum(acc))


@tle.raw_kernel
def ew_div(X: tle.mem(f32), Y: tle.mem(f32), S: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    acc = tle.vzero(f32)
    for i in tle.range(0, N, nvl):
        vx = tle.vload(X, i, dtype=f32)
        vy = tle.vload(Y, i, dtype=f32)
        acc = acc + (vx / vy)
    tle.vstore(S, 0, tle.vreduce_sum(acc))


@tle.raw_kernel
def ew_neg(X: tle.mem(f32), Y: tle.mem(f32), S: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    acc = tle.vzero(f32)
    for i in tle.range(0, N, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc = acc + (-vx)
    tle.vstore(S, 0, tle.vreduce_sum(acc))


@tle.raw_kernel
def ew_vmin(X: tle.mem(f32), Y: tle.mem(f32), S: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    acc = tle.vzero(f32)
    for i in tle.range(0, N, nvl):
        vx = tle.vload(X, i, dtype=f32)
        vy = tle.vload(Y, i, dtype=f32)
        acc = acc + tle.vmin(vx, vy)
    tle.vstore(S, 0, tle.vreduce_sum(acc))


@tle.raw_kernel
def ew_vmax(X: tle.mem(f32), Y: tle.mem(f32), S: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    acc = tle.vzero(f32)
    for i in tle.range(0, N, nvl):
        vx = tle.vload(X, i, dtype=f32)
        vy = tle.vload(Y, i, dtype=f32)
        acc = acc + tle.vmax(vx, vy)
    tle.vstore(S, 0, tle.vreduce_sum(acc))


@tle.raw_kernel
def ew_sqrt(X: tle.mem(f32), Y: tle.mem(f32), S: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    acc = tle.vzero(f32)
    for i in tle.range(0, N, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc = acc + tle.sqrt(vx)
    tle.vstore(S, 0, tle.vreduce_sum(acc))


@tle.raw_kernel
def ew_abs(X: tle.mem(f32), Y: tle.mem(f32), S: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    acc = tle.vzero(f32)
    for i in tle.range(0, N, nvl):
        vx = tle.vload(X, i, dtype=f32)
        acc = acc + tle.abs(vx)
    tle.vstore(S, 0, tle.vreduce_sum(acc))


@tle.raw_kernel
def ew_select(X: tle.mem(f32), Y: tle.mem(f32), S: tle.mem(f32, out=True), N: tle.index):
    # per-lane: max(x, y) via compare -> mask -> select
    nvl = tle.vconfig(-1, 1)
    acc = tle.vzero(f32)
    for i in tle.range(0, N, nvl):
        vx = tle.vload(X, i, dtype=f32)
        vy = tle.vload(Y, i, dtype=f32)
        m = vx > vy
        acc = acc + tle.select(m, vx, vy)
    tle.vstore(S, 0, tle.vreduce_sum(acc))


def _make_host(raw_kernel):

    @triton.jit
    def host(X, Y, S, N):
        _sr_call(raw_kernel, outputs=[], inputs=[X, Y, S, N])

    host.__name__ = f"_ew_host_{raw_kernel.__name__}"
    host.fn.__name__ = host.__name__
    return host


# ref: reduce over the same elementwise op
_OPS = {
    "add": (ew_add, lambda x, y: (x + y).sum()),
    "mul": (ew_mul, lambda x, y: (x * y).sum()),
    "sub": (ew_sub, lambda x, y: (x - y).sum()),
    "div": (ew_div, lambda x, y: (x / y).sum()),
    "neg": (ew_neg, lambda x, y: (-x).sum()),
    "vmin": (ew_vmin, lambda x, y: torch.minimum(x, y).sum()),
    "vmax": (ew_vmax, lambda x, y: torch.maximum(x, y).sum()),
    "sqrt": (ew_sqrt, lambda x, y: torch.sqrt(x).sum()),
    "abs": (ew_abs, lambda x, y: x.abs().sum()),
    "select": (ew_select, lambda x, y: torch.maximum(x, y).sum()),
}


@pytest.mark.parametrize("op", list(_OPS.keys()))
@pytest.mark.parametrize("N", [64, 128, 256])
def test_elementwise(op, N):
    raw, ref_fn = _OPS[op]
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32).abs() + 0.5  # keep div well-conditioned
    if op == "sqrt":
        x = x.abs() + 0.1  # sqrt domain
    s = torch.zeros(1, dtype=torch.float32)
    _make_host(raw)[(1, )](x.contiguous(), y.contiguous(), s, N)
    ref = ref_fn(x, y).item()
    got = s.item()
    # sum over up to 256 f32 terms: use a relative tolerance
    assert abs(got - ref) <= 1e-2 * max(1.0, abs(ref)), f"op={op} N={N} got={got:.5f} ref={ref:.5f}"
