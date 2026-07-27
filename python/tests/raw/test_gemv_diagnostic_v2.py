"""Simplified diagnostic test - copy of working test_sequential.py logic."""
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32

@tle.raw_kernel
def gemv_spine_raw(Mat: tle.mem(f16), vec_s: tle.mem(f16), scores: tle.mem(f32, out=True),
                   K: tle.index, row_base: tle.index, row_end: tle.index):
    nvl = tle.vconfig(-1, 1)
    Kfloor = (K // nvl) * nvl
    for n in tle.range(row_base, row_end, 1):
        acc = tle.vzero(f32)
        for ki in tle.range(0, Kfloor, nvl):
            vm = tle.vload(Mat, n * K + ki)
            vv = tle.vload(vec_s, ki)
            acc = tle.vmacc(acc, vm, vv)
        for ki in tle.range(Kfloor, K, nvl):
            nvl = tle.vconfig(K - ki, 1)
            tm = tle.vload(Mat, n * K + ki)
            tv = tle.vload(vec_s, ki)
            acc = tle.vmacc(acc, tm, tv)
        tle.sstore(scores, n, tle.vreduce_sum(acc))

@triton.jit(do_not_specialize=["K", "N"])
def gemv_host(Mat, vec_s, scores, K, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)
    _sr_call(gemv_spine_raw, outputs=[], inputs=[Mat, vec_s, scores, K, row_base, row_end])

def test_shape(N, K):
    torch.manual_seed(0)
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec_s = torch.randn(K, dtype=torch.float16)
    scores = torch.zeros(N, dtype=torch.float32)

    gemv_host[(1,)](Mat.contiguous().reshape(-1), vec_s, scores, K, N, BLOCK=N)

    ref = torch.mv(Mat.float(), vec_s.float())
    max_diff = (scores - ref).abs().max().item()

    result = "PASS" if max_diff < 1e-1 else "FAIL"
    print(f"N={N} K={K}: max_diff={max_diff:.4e} {result}")
    return result == "PASS"

if __name__ == "__main__":
    print("=== Test (8, 64) ===")
    r1 = test_shape(8, 64)

    print("\n=== Test (16, 128) ===")
    r2 = test_shape(16, 128)

    print("\n=== Test (8, 65) ===")
    r3 = test_shape(8, 65)

    print("\n" + ("ALL_PASS" if (r1 and r2 and r3) else "HAS_FAILURES"))
