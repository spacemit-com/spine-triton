"""spine_raw cumsum vectorized — 3-phase block-scan, O(N) with grid parallelism.

Phase 1 (grid=(P,)): each program reduce-sums VL=64 elements → block_sums[p]
Phase 2 (grid=(1,)): scalar exclusive-prefix over P block_sums → offsets[p]
Phase 3 (grid=(P,)): per-block scalar inner loop (VL steps) + add offset

Phase 1 and 3 run as P concurrent programs, giving parallel speedup on
multi-program dispatch. Phase 2 is tiny (P = N//VL, e.g. 128 steps for N=8192).

Contrast with cumsum_1d (sequential scalar): that does N sequential steps with
no grid parallelism. The vectorized version has the same total work but exposes
P-way parallelism.

For N not a multiple of VL, tail elements (< VL) are handled by Phase 3's last
program (guard on program count) or can fall back to the sequential kernel.
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

# Phase 1 — reduce VL elements per program → block_sums
@tle.raw_kernel
def block_sum_kernel(X: tle.mem(f32), block_sums: tle.mem(f32, out=True),
                     N: tle.index, p: tle.index):
    nvl = tle.vconfig(-1, 1)
    base = p * nvl
    vx = tle.vload(X, base, dtype=f32)
    tle.vstore(block_sums, p, tle.vreduce_sum(vx))

@triton.jit
def block_sum_host(X, block_sums, N, P):
    p = tl.program_id(0)
    if p < P:
        _sr_call(block_sum_kernel, outputs=[], inputs=[X, block_sums, N, p])


# Phase 2 — exclusive prefix over block_sums (single program, P small)
@tle.raw_kernel
def prefix_offset_kernel(block_sums: tle.mem(f32), offsets: tle.mem(f32, out=True),
                         P: tle.index):
    nvl = tle.vconfig(-1, 1)
    acc = tle.vreduce_sum(tle.vzero(f32))   # 0.0 — exclusive: offsets[p] = sum(0..p-1)
    for i in tle.range(0, P, 1):
        tle.vstore(offsets, i, acc)          # write BEFORE adding
        s = tle.vscalar(block_sums, i, dtype=f32)
        acc = acc + s

@triton.jit
def prefix_offset_host(block_sums, offsets, P):
    _sr_call(prefix_offset_kernel, outputs=[], inputs=[block_sums, offsets, P])


# Phase 3 — local prefix (VL-step scalar loop) + add exclusive offset per block
@tle.raw_kernel
def apply_prefix_kernel(X: tle.mem(f32), out: tle.mem(f32, out=True),
                        offsets: tle.mem(f32),
                        N: tle.index, p: tle.index):
    nvl = tle.vconfig(-1, 1)
    base = p * nvl
    offset = tle.vscalar(offsets, p, dtype=f32)
    acc = tle.vreduce_sum(tle.vzero(f32))   # 0.0 scalar
    for j in tle.range(0, nvl, 1):
        xi = tle.vscalar(X, base + j, dtype=f32)
        acc = acc + xi
        tle.vstore(out, base + j, acc + offset)

@triton.jit
def apply_prefix_host(X, out, offsets, N, P):
    p = tl.program_id(0)
    if p < P:
        _sr_call(apply_prefix_kernel, outputs=[], inputs=[X, out, offsets, N, p])


def cumsum_vectorized(X: torch.Tensor) -> torch.Tensor:
    """3-phase block-scan cumsum. VL=64 (K3 f32 VLMAX). Tail handled sequentially."""
    N = X.numel()
    VL = 64
    P = N // VL
    Nfloor = P * VL
    out = torch.zeros(N, dtype=torch.float32)

    if P > 0:
        bs = torch.zeros(P, dtype=torch.float32)
        offs = torch.zeros(P, dtype=torch.float32)
        Xf = X[:Nfloor].contiguous().reshape(-1)
        block_sum_host[(P,)](Xf, bs, Nfloor, P)
        prefix_offset_host[(1,)](bs, offs, P)
        apply_prefix_host[(P,)](Xf, out[:Nfloor], offs, Nfloor, P)

    # Tail (< VL elements): scalar sequential with running offset
    if Nfloor < N:
        last_val = out[Nfloor - 1].item() if Nfloor > 0 else 0.0
        tail = X[Nfloor:].float()
        out[Nfloor:] = torch.cumsum(tail, dim=0) + last_val

    return out


@pytest.mark.parametrize("N", [64, 128, 512, 1024, 4096, 8192])
def test_cumsum_vec_aligned(N):
    """VL-aligned shapes: full 3-phase pipeline."""
    torch.manual_seed(42)
    X = torch.randn(N, dtype=torch.float32)
    out = cumsum_vectorized(X)
    ref = torch.cumsum(X, dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("N", [100, 200, 513])
def test_cumsum_vec_arb(N):
    """Non-aligned shapes: 3-phase prefix + scalar tail."""
    torch.manual_seed(7)
    X = torch.randn(N, dtype=torch.float32)
    out = cumsum_vectorized(X)
    ref = torch.cumsum(X, dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
