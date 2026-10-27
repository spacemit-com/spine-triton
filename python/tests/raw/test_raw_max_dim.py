"""smt_rvisa max_dim / min_dim — 2D reduce along dim=1 with value + index outputs.

torch.max(x, dim=1) → (values[M], indices[M]). Each program handles one row,
reusing the argmax select-based index tracking. grid=(M,), row via program_id.
"""
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.smt_rvisa as sr
from triton.language.extra.smt_rvisa import call as _sr_call

f32 = sr.f32
INF_IDX = 1.0e30


@sr.raw_kernel
def max_dim1_kernel(X: sr.mem(f32), vals: sr.mem(f32, out=True), idxs: sr.mem(f32, out=True), M: sr.index, N: sr.index,
                    row: sr.index):
    nvl = sr.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    base = row * N
    lane = sr.viota()

    best_val = sr.vload(X, base, dtype=f32)
    best_idx = lane
    for i in sr.range(0, Nfloor, nvl):
        vx = sr.vload(X, base + i, dtype=f32)
        idx = lane + sr.cast(i, f32)
        gt = vx > best_val
        best_val = sr.select(gt, vx, best_val)
        best_idx = sr.select(gt, idx, best_idx)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tx = sr.vload(X, base + i, dtype=f32, fill=-1e38)
        tidx = lane + sr.cast(i, f32)
        gt2 = tx > best_val
        best_val = sr.select(gt2, tx, best_val)
        best_idx = sr.select(gt2, tidx, best_idx)

    gmax = sr.vreduce_max(best_val)
    is_max = best_val >= gmax
    big = sr.vzero(f32) + INF_IDX
    masked = sr.select(is_max, best_idx, big)
    argmax = sr.vreduce_min(masked)
    sr.sstore(vals, row, gmax)
    sr.sstore(idxs, row, argmax)


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
    max_dim1_host[(M, )](X.contiguous().reshape(-1), vals, idxs, M, N)
    ref_v, ref_i = torch.max(X, dim=1)
    torch.testing.assert_close(vals, ref_v, rtol=1e-5, atol=1e-5)
    got_i = idxs.round().to(torch.int64)
    assert torch.equal(got_i, ref_i), f"idx mismatch: got {got_i}, want {ref_i}"


# ---------------------------------------------------------------------------
# min_dim
# ---------------------------------------------------------------------------
@sr.raw_kernel
def min_dim1_kernel(X: sr.mem(f32), vals: sr.mem(f32, out=True), idxs: sr.mem(f32, out=True), M: sr.index, N: sr.index,
                    row: sr.index):
    nvl = sr.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    base = row * N
    lane = sr.viota()

    best_val = sr.vload(X, base, dtype=f32)
    best_idx = lane
    for i in sr.range(0, Nfloor, nvl):
        vx = sr.vload(X, base + i, dtype=f32)
        idx = lane + sr.cast(i, f32)
        lt = vx < best_val
        best_val = sr.select(lt, vx, best_val)
        best_idx = sr.select(lt, idx, best_idx)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tx = sr.vload(X, base + i, dtype=f32, fill=1e38)
        tidx = lane + sr.cast(i, f32)
        lt2 = tx < best_val
        best_val = sr.select(lt2, tx, best_val)
        best_idx = sr.select(lt2, tidx, best_idx)

    gmin = sr.vreduce_min(best_val)
    is_min = best_val <= gmin
    big = sr.vzero(f32) + INF_IDX
    masked = sr.select(is_min, best_idx, big)
    argmin = sr.vreduce_min(masked)
    sr.sstore(vals, row, gmin)
    sr.sstore(idxs, row, argmin)


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
    min_dim1_host[(M, )](X.contiguous().reshape(-1), vals, idxs, M, N)
    ref_v, ref_i = torch.min(X, dim=1)
    torch.testing.assert_close(vals, ref_v, rtol=1e-5, atol=1e-5)
    got_i = idxs.round().to(torch.int64)
    assert torch.equal(got_i, ref_i), f"idx mismatch: got {got_i}, want {ref_i}"
