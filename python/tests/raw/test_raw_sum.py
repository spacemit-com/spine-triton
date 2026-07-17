"""spine_raw sum 算子实现 - L0 真正零缺口

sum 是唯一不需要标量算术的 reduce 算子：
- 只需要 vreduce_sum（已存在）
- 无需除以 N
- 无需其他原语

验证 spine_raw 的基本 reduce 能力。
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
# sum_1d: 对 1D 张量求和
# ---------------------------------------------------------------------------
@tle.raw_kernel
def sum_1d_kernel(
    X: tle.mem(f16),
    out: tle.mem(f32, out=True),
    N: tle.index
):
    """1D sum: 单 kernel 处理整个向量。"""
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl

    acc = tle.vzero(f32)

    # Main loop: full tiles
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, i)
        vx_f32 = tle.cast(vx, f32)
        acc = acc + vx_f32

    # Tail loop: partial tile (use different variable names)
    for i in tle.range(Nfloor, N, nvl):
        nvl_tail = tle.vconfig(N - i, 1)
        tx = tle.vload(X, i)
        tx_f32 = tle.cast(tx, f32)
        acc = acc + tx_f32

    # Reduce and store
    tle.vstore(out, 0, tle.vreduce_sum(acc))


@triton.jit  # Remove do_not_specialize to allow different N values
def sum_1d_host(X, out, N):
    """Host wrapper for 1D sum."""
    _sr_call(sum_1d_kernel, outputs=[], inputs=[X, out, N])


def sum_1d_raw(X: torch.Tensor) -> torch.Tensor:
    """1D sum using spine_raw."""
    assert X.ndim == 1
    assert X.dtype == torch.float16

    N = X.shape[0]

    # Always create a fresh output tensor for each call
    out = torch.empty(1, dtype=torch.float32)

    sum_1d_host[(1,)](X.contiguous(), out, N)

    return out[0]


# ---------------------------------------------------------------------------
# sum_2d: 对 2D 张量的某个维度求和
# ---------------------------------------------------------------------------
@tle.raw_kernel
def sum_2d_dim1_kernel(
    X: tle.mem(f16),
    out: tle.mem(f32, out=True),
    M: tle.index,
    N: tle.index,
    row_idx: tle.index
):
    """2D sum along dim=1: 每行独立求和，输出 [M]。"""
    nvl = tle.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl

    acc = tle.vzero(f32)

    # Main loop
    for i in tle.range(0, Nfloor, nvl):
        vx = tle.vload(X, row_idx * N + i)
        vx_f32 = tle.cast(vx, f32)
        acc = acc + vx_f32

    # Tail loop (use different variable names)
    for i in tle.range(Nfloor, N, nvl):
        nvl_tail = tle.vconfig(N - i, 1)
        tx = tle.vload(X, row_idx * N + i)
        tx_f32 = tle.cast(tx, f32)
        acc = acc + tx_f32

    tle.vstore(out, row_idx, tle.vreduce_sum(acc))


@triton.jit  # Remove do_not_specialize to allow different M, N values
def sum_2d_dim1_host(X, out, M, N):
    """Host wrapper for 2D sum along dim=1."""
    row_idx = tl.program_id(0)
    if row_idx < M:
        _sr_call(sum_2d_dim1_kernel, outputs=[], inputs=[X, out, M, N, row_idx])


def sum_2d_raw(X: torch.Tensor, dim: int) -> torch.Tensor:
    """2D sum along specified dimension using spine_raw."""
    assert X.ndim == 2
    assert X.dtype == torch.float16
    assert dim in [0, 1]

    if dim == 1:
        # Sum along columns: [M, N] -> [M]
        M, N = X.shape
        out = torch.empty(M, dtype=torch.float32)
        sum_2d_dim1_host[(M,)](X.contiguous().reshape(-1), out, M, N)
        return out
    else:
        # Sum along rows: [M, N] -> [N]
        # Transpose then sum along dim=1
        return sum_2d_raw(X.t().contiguous(), dim=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _test_sum_1d(N):
    """Test 1D sum."""
    # Create fresh input for each test
    X = torch.randn(N, dtype=torch.float16)

    # Reference
    ref = X.float().sum()

    # spine_raw - this creates a new output tensor inside
    got = sum_1d_raw(X)

    diff = abs(got.item() - ref.item())
    print(f"sum_1d N={N:5d}: ref={ref:.6f} got={got:.6f} diff={diff:.2e}")

    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), f"diff={diff}"


def _test_sum_2d(M, N, dim):
    """Test 2D sum."""
    # Create fresh input for each test
    X = torch.randn(M, N, dtype=torch.float16)

    # Reference
    ref = X.float().sum(dim=dim)

    # spine_raw
    got = sum_2d_raw(X, dim=dim)

    max_diff = (got - ref).abs().max().item()
    mean_diff = (got - ref).abs().mean().item()

    print(f"sum_2d M={M:4d} N={N:4d} dim={dim}: max_diff={max_diff:.2e} mean_diff={mean_diff:.2e}")

    assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), f"max_diff={max_diff}"


# Test shapes
_SHAPES_1D = [64, 128, 256, 512, 100, 130, 200]
_SHAPES_2D = [(4, 64), (16, 128), (32, 256), (8, 100), (16, 130)]


@pytest.mark.parametrize("N", _SHAPES_1D)
def test_sum_1d(N):
    """Test 1D sum with various sizes."""
    _test_sum_1d(N)


@pytest.mark.parametrize("M, N", _SHAPES_2D)
@pytest.mark.parametrize("dim", [0, 1])
def test_sum_2d(M, N, dim):
    """Test 2D sum along different dimensions."""
    _test_sum_2d(M, N, dim)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing 1D sum (L0 - zero gaps)")
    print("=" * 60)
    for N in _SHAPES_1D[:3]:
        _test_sum_1d(N)

    print("\n" + "=" * 60)
    print("Testing 2D sum")
    print("=" * 60)
    for M, N in _SHAPES_2D[:3]:
        for dim in [0, 1]:
            _test_sum_2d(M, N, dim)

    print("\n✅ All tests passed!")
