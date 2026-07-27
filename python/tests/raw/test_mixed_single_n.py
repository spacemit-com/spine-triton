"""Test mixed-syntax three layers with fixed N to avoid specialization issue."""
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32
_BETA = 0.5

@triton.jit
def pre_scale_tl(vec_ptr, vec_s_ptr, alpha, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(vec_ptr + offs, mask=mask, other=0.0)
    y = (x.to(tl.float32) * alpha).to(tl.float16)
    tl.store(vec_s_ptr + offs, y, mask=mask)

@tle.raw_kernel
def gemv_spine_raw(Mat: tle.mem(f16), vec_s: tle.mem(f16), scores: tle.mem(f32, out=True),
                   K: tle.index, N: tle.index):
    nvl = tle.vconfig(-1, 1)
    Kfloor = (K // nvl) * nvl
    for n in tle.range(0, N, 1):
        acc = tle.vzero(f32)
        # Main loop: full vectors
        for ki in tle.range(0, Kfloor, nvl):
            vm = tle.vload(Mat, n * K + ki)
            vv = tle.vload(vec_s, ki)
            acc = tle.vmacc(acc, vm, vv)
        # Tail: single fixed iteration, vconfig handles actual length
        # When Kfloor==K, this loads/processes 0 elements (vconfig(0, 1))
        tail_vl = tle.vconfig(K - Kfloor, 1)
        tm = tle.vload(Mat, n * K + Kfloor)
        tv = tle.vload(vec_s, Kfloor)
        acc = tle.vmacc(acc, tm, tv)
        tle.sstore(scores, n, tle.vreduce_sum(acc))

@triton.jit
def gemv_host(Mat, vec_s, scores, K, N):
    _sr_call(gemv_spine_raw, outputs=[], inputs=[Mat, vec_s, scores, K, N])

@tle.raw_kernel
def post_scale_llvm(scores: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    vl = tle.llvm_const(8, "i64")
    zero = tle.llvm_const(0, "i64")
    beta = tle.llvm_const("5.000000e-01", "vector<[8]xf32>")
    for i in tle.range(zero, N, vl):
        p = tle.llvm_poison("vector<[8]xf32>")
        gs = tle.llvm_gep(tle.llvm_base_ptr(scores), i, "f32")
        v = tle.call_intrinsic("llvm.riscv.vle", [p, gs, vl], result_type="vector<[8]xf32>")
        r = tle.call_intrinsic("llvm.fmul", [v, beta], result_type="vector<[8]xf32>")
        go = tle.llvm_gep(tle.llvm_base_ptr(out), i, "f32")
        tle.call_intrinsic("llvm.riscv.vse", [r, go, vl], result_type="()")

@triton.jit
def post_scale_host(scores, out, N):
    _sr_call(post_scale_llvm, outputs=[], inputs=[scores, out, N])

def test_single_n(N, K, alpha=1.5, BLOCK=64):
    """Test with single N value to avoid specialization cache collision."""
    assert N % 8 == 0
    torch.manual_seed(0)
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)
    scores = torch.zeros(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)

    grid1 = ((K + BLOCK - 1) // BLOCK,)
    pre_scale_tl[grid1](vec.contiguous(), vec_s, alpha, K, BLOCK=BLOCK)
    gemv_host[(1,)](Mat.contiguous().reshape(-1), vec_s, scores, K, N)
    post_scale_host[(1,)](scores, out, N)

    ref = torch.mv(Mat.float(), (vec.float() * alpha).half().float()) * _BETA
    max_diff = (out - ref).abs().max().item()

    print(f"N={N} K={K}: max_diff={max_diff:.4e} {'PASS' if max_diff < 1e-1 else 'FAIL'}")
    return max_diff < 1e-1

if __name__ == "__main__":
    # Test each N separately to avoid cache collision
    all_pass = True
    for N, K in [(8, 64), (8, 128), (8, 100)]:
        if not test_single_n(N, K):
            all_pass = False

    print("ALL_PASS" if all_pass else "HAS_FAILURES")
