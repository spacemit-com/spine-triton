"""Test post_scale_llvm (call_intrinsic stage) in isolation."""
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f32 = tle.f32
_BETA = 0.5

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

def test_post_scale_only(N):
    assert N % 8 == 0
    torch.manual_seed(0)
    scores = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(N, dtype=torch.float32)

    post_scale_host[(1,)](scores, out, N)

    ref = scores * _BETA
    max_diff = (out - ref).abs().max().item()

    print(f"N={N}")
    print(f"  out[:4] = {out[:4].tolist()}")
    print(f"  ref[:4] = {ref[:4].tolist()}")
    print(f"  max_diff = {max_diff:.4e}")

    passed = torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed

if __name__ == "__main__":
    shapes = [8, 16, 32, 64]
    all_pass = True
    for N in shapes:
        if not test_post_scale_only(N):
            all_pass = False
        print()

    print("ALL_PASS" if all_pass else "HAS_FAILURES")
