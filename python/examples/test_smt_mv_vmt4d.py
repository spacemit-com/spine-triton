"""Matrix-vector product (M=1 matmul) via the smt.vmt4d engine path.

smt.vmt4d(a, b, init) lowers to an M=1 linalg.matmul which spine-mlir's
DirectMVPattern (PackMatmul pass, main >= cd8e72ff) rewrites to
spestruct.vmt4d -> llvm.riscv.smt.vfwmadot, using the matrix engine for
f16 inputs.

Constraints (verified on k3, see .claude/rules-archive/09-vmt4d-mv.md):
- b is either a contiguous [K, N] pre-transposed matrix (strided=False) or
  a zero-copy strided [K, N] view of the row-major [N, K] W (strided=True).
  The strided view needs spine-mlir with the MV strided-RHS pack fix
  (MR 68, 6f472365); it measured ~2x faster than the pre-transposed layout
  on large shapes and skips the one-time W.t().contiguous() at weight load.
- f16 inputs with an f32 accumulator (acc_dtype=float32) are supported via
  engine widening; this needs spine-mlir with the VMT4D init-pad dtype fix
  (MR 68). f32 accumulation gives ~1e-6 rel error (vs ~1e-3 for f16 acc)
  and measured no slower than f16 acc.
- K must be a multiple of BK and N a multiple of BN (no tail masking).
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def mv_vmt4d_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    STRIDED_B: tl.constexpr,
):
    pid = tl.program_id(0)
    n0 = pid * BN
    r = tl.arange(0, 1)
    ks = tl.arange(0, BK)
    ns = n0 + tl.arange(0, BN)
    acc = tl.zeros((1, BN), dtype=ACC_DTYPE)
    for k0 in range(0, K, BK):
        a = tl.load(x_ptr + r[:, None] * K + (k0 + ks)[None, :])  # [1, BK]
        if STRIDED_B:
            # W is [N, K] row-major, used zero-copy: b[k, n] = W[n, k]
            b = tl.load(w_ptr + ns[None, :] * K + (k0 + ks)[:, None])  # [BK, BN]
        else:
            # W_t is [K, N] contiguous (prepared once on the host)
            b = tl.load(w_ptr + (k0 + ks)[:, None] * N + ns[None, :])  # [BK, BN]
        acc = smt.vmt4d(a, b, acc)  # [1, BN]
    tl.store(y_ptr + r[:, None] * N + ns[None, :], acc)


def triton_mv(x, w, BK=128, BN=256, acc_dtype=torch.float32, strided=True):
    # x: [K], w: [N, K] row-major weight.
    # strided=True (recommended): feed w directly as a zero-copy [K, N] view
    # -- faster on large shapes and skips the transpose at weight load.
    # strided=False: use the pre-transposed w.t() layout (deployment hoists
    # the one-time transpose to weight load).
    K = x.numel()
    N, K_w = w.shape
    assert K == K_w, "incompatible dimensions"
    assert K % BK == 0 and N % BN == 0, "K and N must be tile-aligned"
    assert w.dtype == torch.float16 and x.dtype == torch.float16, "vmt4d path takes f16 inputs"

    mat = w if strided else w.t().contiguous()  # [K, N] view or copy
    acc_tl = tl.float32 if acc_dtype == torch.float32 else tl.float16
    y = torch.empty((1, N), dtype=acc_dtype)
    mv_vmt4d_kernel[(N // BN, )](x.view(1, K), mat, y, K=K, N=N, BK=BK, BN=BN, ACC_DTYPE=acc_tl, STRIDED_B=strided)
    return y


def _run_correctness(K: int, N: int, dtype=torch.float16, device: str = "cpu"):
    x = torch.randn(K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    # f64 reference: an f16 torch.mv reference carries ~3e-4 of its own
    # error and would mask the ~1e-6 of the f32-acc kernel.
    y_ref = torch.mv(w.double(), x.double())

    # f16-acc output is f16-quantized: compare with a relative gate (ULP of
    # |ref| ~ 60 is already 0.0625). f32-acc output is ~1e-6 rel.
    cases = [
        ("pre f16-acc", torch.float16, False, 2e-2),
        ("pre f32-acc", torch.float32, False, 1e-3),
        ("str f32-acc", torch.float32, True, 1e-3),
    ]
    for tag, acc_dtype, strided, gate in cases:
        y = triton_mv(x, w, acc_dtype=acc_dtype, strided=strided)
        rel = (y.float() - y_ref.float()).abs().max().item() / y_ref.abs().max().item()
        assert rel < gate, f"{tag}: rel err {rel:.2e} exceeds {gate:g}"
        print(f"PASS {tag} K={K} N={N} rel={rel:.2e}")


# -----------------------------
# Performance benchmark (GFLOPS)
# -----------------------------
def benchmark_gflops(
    K: int = 4096,
    N: int = 4096,
    iters: int = 100,
    warmup: int = 10,
    dtype=torch.float16,
    device: str = "cpu",
):
    import time

    x = torch.randn(K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    y = torch.empty((1, N), dtype=torch.float32)

    def launch():
        mv_vmt4d_kernel[(N // 256, )](x.view(1, K), w, y, K=K, N=N, BK=128, BN=256, ACC_DTYPE=tl.float32,
                                      STRIDED_B=True)

    for _ in range(warmup):
        launch()
    t0 = time.perf_counter()
    for _ in range(iters):
        launch()
    t1 = time.perf_counter()

    avg_s = (t1 - t0) / iters
    ops = 2.0 * float(K) * float(N)
    gflops = ops / avg_s / 1e9

    print(f"[PERF] K={K}, N={N}, iters={iters}, "
          f"avg={avg_s * 1e3:.3f} ms, GFLOPS={gflops:.3f}")


# Run perf by env var:
#   RUN_PERF=1 python test_smt_mv_vmt4d.py
if __name__ == "__main__":
    import os

    for (K, N) in [(2048, 2048), (4096, 4096)]:
        _run_correctness(K, N)

    if os.getenv("RUN_PERF", "0") == "1":
        benchmark_gflops(K=4096, N=4096)
