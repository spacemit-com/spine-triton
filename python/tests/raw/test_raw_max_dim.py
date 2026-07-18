"""spine_raw max_dim / min_dim — 2D reduce along dim=1 with value + index outputs.

torch.max(x, dim=1) → (values[M], indices[M]). Each program handles one row,
reusing the argmax select-based index tracking. grid=(M,), row via program_id.
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
INF_IDX = 1.0e30


@tle.raw_kernel
def max_dim1_kernel(
    X: tle.mem(f32), vals: tle.mem(f32, out=True), idxs: tle.mem(f32, out=True),
    M: tle.index, N: tle.index, row: tle.index
):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    base = row * N
    lane = tle.viota()

    best_val = tle.vload(X, base, dtype=f32)
    best_idx = lane
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, base + i, dtype=f32)
        idx = lane + tle.cast(i, f32)
        gt = vx > best_val
        best_val = tle.select(gt, vx, best_val)
        best_idx = tle.select(gt, idx, best_idx)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, base + i, dtype=f32, fill=-1e38)
        tidx = lane + tle.cast(i, f32)
        gt2 = tx > best_val
        best_val = tle.select(gt2, tx, best_val)
        best_idx = tle.select(gt2, tidx, best_idx)

    gmax = tle.vreduce_max(best_val)
    is_max = best_val >= gmax
    big = tle.vzero(f32) + INF_IDX
    masked = tle.select(is_max, best_idx, big)
    argmax = tle.vreduce_min(masked)
    tle.vstore(vals, row, gmax)
    tle.vstore(idxs, row, argmax)


@triton.jit
def max_dim1_host(X, vals, idxs, M, N):
    row = tl.program_id(0)
    if row < M:
        _sr_call(max_dim1_kernel, outputs=[], inputs=[X, vals, idxs, M, N, row])


@pytest.mark.parametrize("M,N", [(4, 64), (8, 128), (3, 100), (16, 200)])
def test_max_dim1(M, N):
    torch.manual_seed(42)
    X = torch.randn(M, N, dtype=torch.float32)
    vals = torch.zeros(M, dtype=torch.float32)
    idxs = torch.zeros(M, dtype=torch.float32)
    max_dim1_host[(M,)](X.contiguous().reshape(-1), vals, idxs, M, N)
    ref_v, ref_i = torch.max(X, dim=1)
    torch.testing.assert_close(vals, ref_v, rtol=1e-5, atol=1e-5)
    got_i = idxs.round().to(torch.int64)
    assert torch.equal(got_i, ref_i), f"idx mismatch: got {got_i}, want {ref_i}"


# ---------------------------------------------------------------------------
# min_dim
# ---------------------------------------------------------------------------
@tle.raw_kernel
def min_dim1_kernel(
    X: tle.mem(f32), vals: tle.mem(f32, out=True), idxs: tle.mem(f32, out=True),
    M: tle.index, N: tle.index, row: tle.index
):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    base = row * N
    lane = tle.viota()

    best_val = tle.vload(X, base, dtype=f32)
    best_idx = lane
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, base + i, dtype=f32)
        idx = lane + tle.cast(i, f32)
        lt = vx < best_val
        best_val = tle.select(lt, vx, best_val)
        best_idx = tle.select(lt, idx, best_idx)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, base + i, dtype=f32, fill=1e38)
        tidx = lane + tle.cast(i, f32)
        lt2 = tx < best_val
        best_val = tle.select(lt2, tx, best_val)
        best_idx = tle.select(lt2, tidx, best_idx)

    gmin = tle.vreduce_min(best_val)
    is_min = best_val <= gmin
    big = tle.vzero(f32) + INF_IDX
    masked = tle.select(is_min, best_idx, big)
    argmin = tle.vreduce_min(masked)
    tle.vstore(vals, row, gmin)
    tle.vstore(idxs, row, argmin)


@triton.jit
def min_dim1_host(X, vals, idxs, M, N):
    row = tl.program_id(0)
    if row < M:
        _sr_call(min_dim1_kernel, outputs=[], inputs=[X, vals, idxs, M, N, row])


@pytest.mark.parametrize("M,N", [(4, 64), (8, 128), (3, 100), (16, 200)])
def test_min_dim1(M, N):
    torch.manual_seed(43)
    X = torch.randn(M, N, dtype=torch.float32)
    vals = torch.zeros(M, dtype=torch.float32)
    idxs = torch.zeros(M, dtype=torch.float32)
    min_dim1_host[(M,)](X.contiguous().reshape(-1), vals, idxs, M, N)
    ref_v, ref_i = torch.min(X, dim=1)
    torch.testing.assert_close(vals, ref_v, rtol=1e-5, atol=1e-5)
    got_i = idxs.round().to(torch.int64)
    assert torch.equal(got_i, ref_i), f"idx mismatch: got {got_i}, want {ref_i}"
