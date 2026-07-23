"""LLVM-direct end-to-end on K3: matrix-vector multiply with K-loop.

Validates: for-loop, iter-arg accumulator, llvm_size, llvm_fadd/fmul, llvm_gep.
Compute: C[i] = sum_k A[i*K + k] * B[k]  (simplified: single row, K tiles)
"""
import os
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f32 = "f32"


@tle.raw_kernel
def llvm_direct_mv_k3(A: tle.mem(f32), B: tle.mem(f32), C: tle.mem(f32, out=True), K: tle.index):
    """LLVM-direct MV with K-loop: C = sum_k A[k] * B[k] (element-wise, then reduce).

    Simplified: treat A/B as 1D vectors of length K, accumulate into vector C.
    Real MV would tile across rows, but this validates loop+accumulator.
    """
    vl = tle.llvm_const(8, "i64")
    acc = tle.llvm_const("0.000000e+00", "vector<[8]xf32>")
    zero = tle.llvm_const(0, "i64")

    # Loop bound comes from the scalar K param (driver ABI passes rank-0 memref
    # descriptors with no shape, so llvm_size is unavailable in llvm-direct).
    for k in tle.range(zero, K, vl):
        pa = tle.llvm_poison("vector<[8]xf32>")
        pb = tle.llvm_poison("vector<[8]xf32>")
        ga = tle.llvm_gep(tle.llvm_base_ptr(A), k, "f32")
        gb = tle.llvm_gep(tle.llvm_base_ptr(B), k, "f32")
        va = tle.call_intrinsic("llvm.riscv.vle", [pa, ga, vl], result_type="vector<[8]xf32>")
        vb = tle.call_intrinsic("llvm.riscv.vle", [pb, gb, vl], result_type="vector<[8]xf32>")
        prod = tle.llvm_fmul(va, vb)
        acc = tle.llvm_fadd(acc, prod)

    # Store accumulated vector (simplified: no reduction to scalar)
    gc = tle.llvm_base_ptr(C)
    tle.call_intrinsic("llvm.riscv.vse", [acc, gc, vl], result_type="()")


@triton.jit
def llvm_direct_mv_k3_host(A, B, C, K):
    _sr_call(llvm_direct_mv_k3, outputs=[], inputs=[A, B, C, K])


def main():
    K = 64
    A = torch.arange(K, dtype=torch.float32)
    B = torch.ones(K, dtype=torch.float32)
    C = torch.zeros(8, dtype=torch.float32)
    C_ref = torch.zeros(8, dtype=torch.float32)

    # Reference: C[i] = sum of A[i::8] * B[i::8] for each lane i in [0,8)
    for i in range(8):
        C_ref[i] = (A[i::8] * B[i::8]).sum()

    print(f"=== LLVM-direct MV with K-loop (K={K}) ===")
    try:
        llvm_direct_mv_k3_host[(1,)](A, B, C, K)
        print("COMPILED OK")
        print("C (first 8):", C[:8])
        print("C_ref:      ", C_ref[:8])
        err = (C - C_ref).abs().max().item()
        print(f"max_err: {err:.6e}")
        if err < 1e-3:
            print("PASS: numerical correct")
        else:
            print(f"FAIL: err {err} >= 1e-3")
    except Exception as e:
        import traceback
        print("COMPILE/RUN FAILED:", type(e).__name__)
        traceback.print_exc()


if __name__ == "__main__":
    main()
