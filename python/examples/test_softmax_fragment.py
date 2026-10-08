"""Softmax with smt.alloc(scope="fragment") as scratch buffer.

Demonstrates a real use case for fragment-scoped allocation:

  Baseline 3-pass softmax:
    Pass 1: find row_max
    Pass 2: exp(x - max) → store to OUTPUT, accumulate denom
    Pass 3: reload from OUTPUT, multiply inv_denom, store back

  Fragment version (multi-chunk capable):
    Pass 1: find row_max
    Pass 2: exp(x - max) → store to FRAGMENT SCRATCH, accumulate denom,
            then flush fragment to output
    Pass 3: load from output into FRAGMENT, normalize in-register, store back

The fragment buffer is COL_SIZE-sized and reused per chunk.  Within each
chunk the exp / normalize computation happens on register-backed storage
instead of directly on output memory.

Usage:
    python python/perf/test_softmax_fragment.py
"""

import time
import torch

import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.language.extra.cpu import libdevice as tl_extra_shim
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

# ═══════════════════════════════════════════════════════════════════════════
# Baseline: standard 3-pass softmax
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def softmax_baseline_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    ROW_SIZE: tl.constexpr,
    COL_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0) * ROW_SIZE
    element_ty = input_ptr.type.element_ty

    for row_idx in range(row_start, row_start + ROW_SIZE):
        if row_idx < n_rows:
            # Pass 1: row max
            row_max = tl.full((COL_SIZE, ), -float("inf"), dtype=element_ty)
            for col_off in range(0, n_cols, COL_SIZE):
                in_blk = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols, ),
                    strides=(1, ),
                    offsets=(col_off, ),
                    block_shape=(COL_SIZE, ),
                    order=(0, ),
                )
                row_max = tl.maximum(tl.load(in_blk, boundary_check=(0, ), padding_option="neg_inf"), row_max)
            row_max_s = tl.max(row_max, axis=0).to(element_ty)

            # Pass 2: exp → store to OUTPUT, accumulate denom
            denom = tl.zeros((1, ), dtype=element_ty)
            for col_off in range(0, n_cols, COL_SIZE):
                in_blk = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols, ),
                    strides=(1, ),
                    offsets=(col_off, ),
                    block_shape=(COL_SIZE, ),
                    order=(0, ),
                )
                out_blk = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols, ),
                    strides=(1, ),
                    offsets=(col_off, ),
                    block_shape=(COL_SIZE, ),
                    order=(0, ),
                )
                exp_v = tl_extra_shim.exp(tl.load(in_blk, boundary_check=(0, ), padding_option="neg_inf") -
                                          row_max_s).to(element_ty)
                denom += tl.sum(exp_v, axis=0)
                tl.store(out_blk, exp_v, boundary_check=(0, ))

            # Pass 3: RE-LOAD from output memory, normalize, store back
            inv_d = tl.full((1, ), 1, dtype=element_ty) / denom
            for col_off in range(0, n_cols, COL_SIZE):
                out_blk = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols, ),
                    strides=(1, ),
                    offsets=(col_off, ),
                    block_shape=(COL_SIZE, ),
                    order=(0, ),
                )
                tl.store(out_blk, tl.load(out_blk, boundary_check=(0, )) * inv_d, boundary_check=(0, ))


# ═══════════════════════════════════════════════════════════════════════════
# Fragment version: exp scratch lives in register-backed fragment buffer
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def softmax_fragment_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    ROW_SIZE: tl.constexpr,
    COL_SIZE: tl.constexpr,
):
    """Multi-chunk softmax using fragment scratch.

    Supports arbitrary n_cols (not limited to n_cols <= COL_SIZE).
    The fragment buffer (COL_SIZE elements) is reused per chunk:
      - Pass 2: exp values are computed into fragment, then flushed to output
      - Pass 3: output is loaded into fragment, normalized in-register, stored back

    When n_cols <= COL_SIZE (single chunk), pass 3 reads directly from
    fragment without the output round-trip — same as the original version.
    """
    row_start = tl.program_id(0) * ROW_SIZE
    element_ty = input_ptr.type.element_ty

    for row_idx in range(row_start, row_start + ROW_SIZE):
        if row_idx < n_rows:
            # Fragment-scoped scratch: backend sees memref<COL_SIZE x ty, 3>
            scratch_blk = smt.alloc(shape=[COL_SIZE], type=element_ty, scope="fragment")

            # Pass 1: row max (loop over all chunks)
            row_max = tl.full((COL_SIZE, ), -float("inf"), dtype=element_ty)
            for col_off in range(0, n_cols, COL_SIZE):
                in_blk = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols, ),
                    strides=(1, ),
                    offsets=(col_off, ),
                    block_shape=(COL_SIZE, ),
                    order=(0, ),
                )
                row_max = tl.maximum(tl.load(in_blk, boundary_check=(0, ), padding_option="neg_inf"), row_max)
            row_max_s = tl.max(row_max, axis=0).to(element_ty)

            # Pass 2: exp → fragment scratch → flush to output, accumulate denom
            denom = tl.zeros((1, ), dtype=element_ty)
            for col_off in range(0, n_cols, COL_SIZE):
                in_blk = tl.make_block_ptr(
                    base=input_ptr + row_idx * input_row_stride,
                    shape=(n_cols, ),
                    strides=(1, ),
                    offsets=(col_off, ),
                    block_shape=(COL_SIZE, ),
                    order=(0, ),
                )
                exp_v = tl_extra_shim.exp(tl.load(in_blk, boundary_check=(0, ), padding_option="neg_inf") -
                                          row_max_s).to(element_ty)
                denom += tl.sum(exp_v, axis=0)

                # Store exp to fragment scratch
                tl.store(scratch_blk, exp_v)

                # Flush fragment to output
                out_blk = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols, ),
                    strides=(1, ),
                    offsets=(col_off, ),
                    block_shape=(COL_SIZE, ),
                    order=(0, ),
                )
                tl.store(out_blk, tl.load(scratch_blk), boundary_check=(0, ))

            # Pass 3: load from output into fragment, normalize, store back
            inv_d = tl.full((1, ), 1, dtype=element_ty) / denom
            for col_off in range(0, n_cols, COL_SIZE):
                out_blk = tl.make_block_ptr(
                    base=output_ptr + row_idx * output_row_stride,
                    shape=(n_cols, ),
                    strides=(1, ),
                    offsets=(col_off, ),
                    block_shape=(COL_SIZE, ),
                    order=(0, ),
                )
                # Load into fragment, normalize in-register, store back
                tl.store(scratch_blk, tl.load(out_blk, boundary_check=(0, )))
                result = tl.load(scratch_blk) * inv_d
                tl.store(out_blk, result, boundary_check=(0, ))


# ═══════════════════════════════════════════════════════════════════════════
# Host helpers
# ═══════════════════════════════════════════════════════════════════════════


def _launch(kernel, x, y, row_size, col_size):
    n_rows, n_cols = x.shape
    grid = (triton.cdiv(n_rows, row_size), )
    kernel[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        ROW_SIZE=row_size,
        COL_SIZE=col_size,
    )


def softmax_baseline(x, row_size=1, col_size=32):
    y = torch.empty_like(x)
    _launch(softmax_baseline_kernel, x, y, row_size, col_size)
    return y


def softmax_fragment(x, row_size=1, col_size=32):
    y = torch.empty_like(x)
    _launch(softmax_fragment_kernel, x, y, row_size, col_size)
    return y


def _best_of_repeats(fn, warmup=5, iters=100, repeats=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.time()
        for _ in range(iters):
            fn()
        best = min(best, (time.time() - t0) / iters * 1000)
    return best


if __name__ == "__main__":
    torch.manual_seed(0)

    # Fragment version now supports n_cols > COL_SIZE (multi-chunk)
    row_size = 4
    col_size = 64
    shapes = [
        # single-chunk: n_cols <= COL_SIZE
        (1024, 64),
        (512, 64),
        # multi-chunk: n_cols > COL_SIZE
        (1024, 128),
        (1024, 256),
        (512, 128),
        (512, 256),
    ]
    dtypes = [torch.float32]

    # ── correctness ──────────────────────────────────────────────────────
    print("=" * 72)
    print("Correctness: baseline vs fragment vs torch")
    print("=" * 72)
    for dt in dtypes:
        for (M, N) in shapes:
            x = torch.randn(M, N, dtype=dt)
            y_base = softmax_fragment(x, row_size=row_size, col_size=col_size)
            y_frag = softmax_baseline(x, row_size=row_size, col_size=col_size)
            y_ref = torch.softmax(x, dim=1)

            ok_base = torch.allclose(y_base, y_ref, atol=1e-2, rtol=1e-2)
            ok_frag = torch.allclose(y_frag, y_ref, atol=1e-2, rtol=1e-2)
            diff_b = (y_base - y_ref).abs().max().item()
            diff_f = (y_frag - y_ref).abs().max().item()

            tag_b = "✅" if ok_base else f"❌ {diff_b:.2e}"
            tag_f = "✅" if ok_frag else f"❌ {diff_f:.2e}"
            print(f"  {dt}  ({M:>4},{N:>4})  baseline={tag_b}  fragment={tag_f}")

    # ── perf comparison ──────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("Performance: baseline vs fragment (scope='fragment')")
    print("=" * 72)
    for dt in dtypes:
        print(f"\n  dtype={dt}")
        print(f"  {'shape':>12}  {'baseline':>12}  {'fragment':>12}  {'speedup':>8}")
        print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
        for (M, N) in shapes:
            x = torch.randn(M, N, dtype=dt)
            ms_b = _best_of_repeats(lambda: softmax_baseline(x, row_size=row_size, col_size=col_size))
            ms_f = _best_of_repeats(lambda: softmax_fragment(x, row_size=row_size, col_size=col_size))
            speedup = ms_b / ms_f if ms_f > 0 else float("inf")
            print(f"  ({M:>4},{N:>4})  {ms_b:>9.3f} ms  {ms_f:>9.3f} ms  {speedup:>7.2f}x")
