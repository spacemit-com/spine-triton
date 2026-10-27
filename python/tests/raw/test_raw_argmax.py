"""smt_rvisa argmax / argmin — index-tracking reduction via viota + select.

Strategy (single VL-wide lane accumulator):
  1. Reduce element-wise: best_val[lane] = max over tiles at that lane,
     best_idx[lane] = global element index that produced it.
  2. gmax = vreduce_max(best_val)  — the global max value.
  3. Build mask (best_val == gmax); where true keep best_idx else +INF_IDX;
     argmax = vreduce_min(masked_idx)  — smallest index achieving the max
     (matches torch.argmax tie-break: first occurrence).

Requires viota (vector.step) + arith.select + cmpf + integer index min-reduce.
Since vreduce_min is float-only in the current binding, the final index
min-reduce is done by casting indices to f32.
"""
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.smt_rvisa as sr
from triton.language.extra.smt_rvisa import call as _sr_call

f32 = sr.f32
i32 = "i32"

# large sentinel for "not the max" lanes in the index min-reduce
INF_IDX = 1.0e30


@sr.raw_kernel
def argmax_1d_kernel(X: sr.mem(f32), out: sr.mem(f32, out=True), N: sr.index):
    nvl = sr.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    lane = sr.viota()  # [0,1,..,VL-1] as f32

    best_val = sr.vload(X, 0, dtype=f32)
    best_idx = lane  # indices 0..VL-1 for first tile
    for i in sr.range(0, Nfloor, nvl):
        vx = sr.vload(X, i, dtype=f32)
        idx = lane + sr.cast(i, f32)  # global element indices this tile
        gt = vx > best_val  # mask: new value strictly greater
        best_val = sr.select(gt, vx, best_val)
        best_idx = sr.select(gt, idx, best_idx)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tx = sr.vload(X, i, dtype=f32, fill=-1e38)
        tidx = lane + sr.cast(i, f32)
        gt2 = tx > best_val
        best_val = sr.select(gt2, tx, best_val)
        best_idx = sr.select(gt2, tidx, best_idx)

    gmax = sr.vreduce_max(best_val)  # global max value (scalar)
    is_max = best_val >= gmax  # lanes achieving the max
    big = sr.vzero(f32) + INF_IDX
    masked = sr.select(is_max, best_idx, big)  # keep idx where max, else +INF
    argmax = sr.vreduce_min(masked)  # smallest index with max value
    sr.sstore(out, 0, argmax)


@triton.jit
def argmax_1d_host(X, out, N):
    _sr_call(argmax_1d_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [64, 128, 256, 100, 200])
def test_argmax_1d(N):
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    argmax_1d_host[(1, )](X, out, N)
    ref = int(torch.argmax(X))
    assert int(round(out[0].item())) == ref, f"got {out[0].item()}, want {ref}"


# ---------------------------------------------------------------------------
# argmin — mirror of argmax (vmin + strict-less mask; tie-break = first index)
# ---------------------------------------------------------------------------
@sr.raw_kernel
def argmin_1d_kernel(X: sr.mem(f32), out: sr.mem(f32, out=True), N: sr.index):
    nvl = sr.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    lane = sr.viota()

    best_val = sr.vload(X, 0, dtype=f32)
    best_idx = lane
    for i in sr.range(0, Nfloor, nvl):
        vx = sr.vload(X, i, dtype=f32)
        idx = lane + sr.cast(i, f32)
        lt = vx < best_val
        best_val = sr.select(lt, vx, best_val)
        best_idx = sr.select(lt, idx, best_idx)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tx = sr.vload(X, i, dtype=f32, fill=1e38)  # pad lanes: +INF so never the min
        tidx = lane + sr.cast(i, f32)
        lt2 = tx < best_val
        best_val = sr.select(lt2, tx, best_val)
        best_idx = sr.select(lt2, tidx, best_idx)

    gmin = sr.vreduce_min(best_val)
    is_min = best_val <= gmin
    big = sr.vzero(f32) + INF_IDX
    masked = sr.select(is_min, best_idx, big)
    argmin = sr.vreduce_min(masked)
    sr.sstore(out, 0, argmin)


@triton.jit
def argmin_1d_host(X, out, N):
    _sr_call(argmin_1d_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [64, 128, 256, 100, 200])
def test_argmin_1d(N):
    torch.manual_seed(43)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    argmin_1d_host[(1, )](X, out, N)
    ref = int(torch.argmin(X))
    assert int(round(out[0].item())) == ref, f"got {out[0].item()}, want {ref}"
