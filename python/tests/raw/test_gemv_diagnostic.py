"""Simplified diagnostic test for mixed-syntax three-layer."""
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32

# Test only stage 2 (gemv_spine_raw) to isolate the issue
@tle.raw_kernel
def gemv_spine_raw(Mat: tle.mem(f16), vec_s: tle.mem(f16), scores: tle.mem(f32, out=True),
                   K: tle.index, N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Kfloor = (K // nvl) * nvl
    for n in tle.range(0, N, 1):
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
def gemv_host(Mat, vec_s, scores, K, N):
    _sr_call(gemv_spine_raw, outputs=[], inputs=[Mat, vec_s, scores, K, N])

def test_gemv_only(N, K):
    torch.manual_seed(0)
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec_s = torch.randn(K, dtype=torch.float16)
    scores = torch.zeros(N, dtype=torch.float32)

    gemv_host[(1,)](Mat.contiguous().reshape(-1), vec_s, scores, K, N)

    ref = torch.mv(Mat.float(), vec_s.float())
    max_diff = (scores - ref).abs().max().item()

    # Debug: print first few values
    print(f"N={N} K={K}")
    print(f"  scores[:4] = {scores[:4].tolist()}")
    print(f"  ref[:4]    = {ref[:4].tolist()}")
    print(f"  max_diff   = {max_diff:.4e}")

    passed = torch.allclose(scores, ref, rtol=1e-2, atol=1e-1)
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed

if __name__ == "__main__":
    shapes = [(8, 64), (16, 128), (8, 65)]
    all_pass = True
    for N, K in shapes:
        if not test_gemv_only(N, K):
            all_pass = False
        print()

    print("ALL_PASS" if all_pass else "HAS_FAILURES")
