"""spine_raw argmax / argmin — index-tracking reduction via viota + select.

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
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f32 = tle.f32
i32 = "i32"

# large sentinel for "not the max" lanes in the index min-reduce
INF_IDX = 1.0e30


@tle.raw_kernel
def argmax_1d_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    lane = tle.viota()                          # [0,1,..,VL-1] as f32

    best_val = tle.vload(X, 0, dtype=f32)
    best_idx = lane                             # indices 0..VL-1 for first tile
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        idx = lane + tle.cast(i, f32)           # global element indices this tile
        gt = vx > best_val                      # mask: new value strictly greater
        best_val = tle.select(gt, vx, best_val)
        best_idx = tle.select(gt, idx, best_idx)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32, fill=-1e38)
        tidx = lane + tle.cast(i, f32)
        gt2 = tx > best_val
        best_val = tle.select(gt2, tx, best_val)
        best_idx = tle.select(gt2, tidx, best_idx)

    gmax = tle.vreduce_max(best_val)            # global max value (scalar)
    is_max = best_val >= gmax                   # lanes achieving the max
    big = tle.vzero(f32) + INF_IDX
    masked = tle.select(is_max, best_idx, big)  # keep idx where max, else +INF
    argmax = tle.vreduce_min(masked)            # smallest index with max value
    tle.vstore(out, 0, argmax)


@triton.jit
def argmax_1d_host(X, out, N):
    _sr_call(argmax_1d_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [64, 128, 256, 100, 200])
def test_argmax_1d(N):
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    argmax_1d_host[(1,)](X, out, N)
    ref = int(torch.argmax(X))
    assert int(round(out[0].item())) == ref, f"got {out[0].item()}, want {ref}"


# ---------------------------------------------------------------------------
# argmin — mirror of argmax (vmin + strict-less mask; tie-break = first index)
# ---------------------------------------------------------------------------
@tle.raw_kernel
def argmin_1d_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl
    lane = tle.viota()

    best_val = tle.vload(X, 0, dtype=f32)
    best_idx = lane
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i, dtype=f32)
        idx = lane + tle.cast(i, f32)
        lt = vx < best_val
        best_val = tle.select(lt, vx, best_val)
        best_idx = tle.select(lt, idx, best_idx)
    for i in tle.range(Nfloor, N, nvl):
        nvl_t = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i, dtype=f32, fill=1e38)   # pad lanes: +INF so never the min
        tidx = lane + tle.cast(i, f32)
        lt2 = tx < best_val
        best_val = tle.select(lt2, tx, best_val)
        best_idx = tle.select(lt2, tidx, best_idx)

    gmin = tle.vreduce_min(best_val)
    is_min = best_val <= gmin
    big = tle.vzero(f32) + INF_IDX
    masked = tle.select(is_min, best_idx, big)
    argmin = tle.vreduce_min(masked)
    tle.vstore(out, 0, argmin)


@triton.jit
def argmin_1d_host(X, out, N):
    _sr_call(argmin_1d_kernel, outputs=[], inputs=[X, out, N])


@pytest.mark.parametrize("N", [64, 128, 256, 100, 200])
def test_argmin_1d(N):
    torch.manual_seed(43)
    X = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)
    argmin_1d_host[(1,)](X, out, N)
    ref = int(torch.argmin(X))
    assert int(round(out[0].item())) == ref, f"got {out[0].item()}, want {ref}"
