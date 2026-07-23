"""LLVM-direct probe: full call_intrinsic LLVM-dialect kernel. Dump TTIR to inspect
structure (does tle.dsl_region carry the LLVM ops correctly?)."""
import os, torch, triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16


@tle.raw_kernel
def llvm_direct_copy(X: tle.mem(f16), out: tle.mem(f16, out=True), N: tle.index):
    vl = tle.llvm_const(8, "i64")
    pt = tle.llvm_poison("vector<[8]xf16>")
    bx = tle.llvm_base_ptr(X)
    bo = tle.llvm_base_ptr(out)
    v = tle.call_intrinsic("llvm.riscv.vle", [pt, bx, vl], result_type="vector<[8]xf16>")
    tle.call_intrinsic("llvm.riscv.vse", [v, bo, vl], result_type="()")


@triton.jit
def llvm_direct_copy_host(X, out, N):
    _sr_call(llvm_direct_copy, outputs=[], inputs=[X, out, N])


if __name__ == "__main__":
    X = torch.arange(8, dtype=torch.float16)
    out = torch.zeros(8, dtype=torch.float16)
    try:
        llvm_direct_copy_host[(1,)](X, out, 8)
        print("COMPILED OK")
        print("out:", out)
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e)[:2000])
