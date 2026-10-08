"""smt.alloc scope → #xsmt.memory_space lowering smoke test.

For each of the four frontend scopes (global / tcm / l2 / fragment), allocate
a buffer with smt.alloc, write a constant into it (so the alloc is not DCE'd),
read it back, reduce-sum, and write to output. Dumps IR via
SPINE_TRITON_DUMP_PATH so you can inspect linalg.mlir and ll.mlir.

Expected mapping (frontend scope → IR #xsmt.memory_space<...>):
    "global"   → #xsmt.memory_space<"global">
    "tcm"      → #xsmt.memory_space<"thread_local">
    "l2"       → #xsmt.memory_space<"cluster_shared_l2">
    "fragment" → #xsmt.memory_space<"fragment">

Each scope uses a distinct kernel name so IR dumps do not overwrite each
other. Set SPINE_TRITON_DUMP_PATH to inspect:
    ls $SPINE_TRITON_DUMP_PATH/
    grep -n "memory_space" $SPINE_TRITON_DUMP_PATH/alloc_global_*linalg.mlir
    grep -n "memory_space" $SPINE_TRITON_DUMP_PATH/alloc_global_*ll.mlir

Usage:
    export SPINE_TRITON_DUMP_PATH=/tmp/spine_dump
    mkdir -p $SPINE_TRITON_DUMP_PATH
    python python/examples/test_alloc_scopes.py
"""

import os

import torch

import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


# One kernel per scope. SCOPE is a tl.constexpr so it survives into the IR
# builder call (smt.alloc reads scope.value if it's a constexpr).
# Each kernel has a distinct name so IR dumps do not collide.
@triton.jit
def alloc_global_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    buf = smt.alloc([BLOCK], type=tl.float32, scope="global")
    offs = tl.arange(0, BLOCK)
    val = offs.to(tl.float32) + pid + 1.0
    tl.store(buf, val)
    loaded = tl.load(buf)
    tl.store(out_ptr + pid, tl.sum(loaded))


@triton.jit
def alloc_tcm_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    buf = smt.alloc([BLOCK], type=tl.float32, scope="tcm")
    offs = tl.arange(0, BLOCK)
    val = offs.to(tl.float32) + pid + 1.0
    tl.store(buf, val)
    loaded = tl.load(buf)
    tl.store(out_ptr + pid, tl.sum(loaded))


@triton.jit
def alloc_l2_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    buf = smt.alloc([BLOCK], type=tl.float32, scope="l2")
    offs = tl.arange(0, BLOCK)
    val = offs.to(tl.float32) + pid + 1.0
    tl.store(buf, val)
    loaded = tl.load(buf)
    tl.store(out_ptr + pid, tl.sum(loaded))


@triton.jit
def alloc_fragment_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    buf = smt.alloc([BLOCK], type=tl.float32, scope="fragment")
    offs = tl.arange(0, BLOCK)
    val = offs.to(tl.float32) + pid + 1.0
    tl.store(buf, val)
    loaded = tl.load(buf)
    tl.store(out_ptr + pid, tl.sum(loaded))


SCOPE_TO_KERNEL = {
    "global": alloc_global_kernel,
    "tcm": alloc_tcm_kernel,
    "l2": alloc_l2_kernel,
    "fragment": alloc_fragment_kernel,
}


def run_one_scope(scope: str, BLOCK: int = 32, N: int = 4):
    out = torch.empty((N, ), device="cpu", dtype=torch.float32)
    kernel = SCOPE_TO_KERNEL[scope]
    kernel[(N, )](out, BLOCK=BLOCK)
    return out


def main():
    dump = os.environ.get("SPINE_TRITON_DUMP_PATH", "")
    if not dump:
        print("WARNING: SPINE_TRITON_DUMP_PATH not set; no IR will be dumped.")
        print("         Run with: export SPINE_TRITON_DUMP_PATH=/tmp/spine_dump")
    else:
        os.makedirs(dump, exist_ok=True)
        print(f"IR will be dumped to {dump}")

    for scope in ["global", "tcm", "l2", "fragment"]:
        print(f"\n=== scope={scope!r} ===")
        try:
            out = run_one_scope(scope)
            print(f"  output: {out.tolist()}")
            print("  PASS (kernel ran)")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")

    if dump:
        print("\nNow inspect the dumped IR:")
        print(f"  ls {dump}/")
        print(f"  grep -n 'memory_space' {dump}/alloc_*linalg.mlir")
        print(f"  grep -n 'memory_space' {dump}/alloc_*ll.mlir")


if __name__ == "__main__":
    main()
