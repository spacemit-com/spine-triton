"""Mode-1 text-emitter end-to-end (x86, no K3, no rebuild):
   @spine_raw copy kernel -> llvm.func module text
     -> spine-opt --spine-triton-e2e-pipeline  (x86)
     -> mlir-translate --mlir-to-llvmir        (x86)
     -> llc --march=riscv64 ...                 -> assert `T <name>` symbol.

Proves the emitter produces a module the full backend accepts down to a
riscv64 object, without touching libtriton.so or the K3 board.
"""
import os
import subprocess
import sys

# import emitter straight from the source tree copy
_SR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "language")
sys.path.insert(0, os.path.abspath(_SR))
from spine_raw.mode1_text import emit_mode1_module          # noqa: E402
from spine_raw import types as _t                            # noqa: E402

f16 = "f16"
In = _t.In
mem = _t.mem
index = _t.index


# ---- a minimal mode-1 copy kernel written with llvm_* primitives ----
def mode1_copy(X: mem(f16), out: mem(f16, out=True), N: index):
    vl = None      # placeholders so python doesn't choke; real values via primitives
    # NOTE: body is walked as AST, not executed.


_KSRC = '''
def mode1_copy(X, out, N):
    vl = tle.llvm_const(8, "i64")
    pt = tle.llvm_poison("vector<[8]xf16>")
    bx = tle.llvm_base_ptr(X)
    bo = tle.llvm_base_ptr(out)
    v = tle.call_intrinsic("llvm.riscv.vle", [pt, bx, vl], result_type="vector<[8]xf16>")
    tle.call_intrinsic("llvm.riscv.vse", [v, bo, vl], result_type="()")
'''

# ---- mv with a K loop: accumulate vfmacc over K-tiles, store one f32 tile ----
# single-program, single-row-tile: acc = sum_k vle(A+k*VL) fma vle(B+k*VL); store acc
_KSRC_MV = '''
def mode1_mv(A, B, C, K):
    vl = tle.llvm_const(8, "i64")
    acc = tle.llvm_const("0.000000e+00", "vector<[8]xf32>")
    zero = tle.llvm_const(0, "i64")
    for k in tle.range(zero, K, vl):
        pa = tle.llvm_poison("vector<[8]xf32>")
        pb = tle.llvm_poison("vector<[8]xf32>")
        ga = tle.llvm_gep(tle.llvm_base_ptr(A), k, "f32")
        gb = tle.llvm_gep(tle.llvm_base_ptr(B), k, "f32")
        va = tle.call_intrinsic("llvm.riscv.vle", [pa, ga, vl], result_type="vector<[8]xf32>")
        vb = tle.call_intrinsic("llvm.riscv.vle", [pb, gb, vl], result_type="vector<[8]xf32>")
        prod = tle.llvm_fmul(va, vb)
        acc = tle.llvm_fadd(acc, prod)
    bc = tle.llvm_base_ptr(C)
    tle.call_intrinsic("llvm.riscv.vse", [acc, bc, vl], result_type="()")
'''

# ---- dot-product with loop: sum_k A[k]*B[k], single scalar result ----
_KSRC_DOT = '''
def mode1_dot(A, B, C, N):
    vl = tle.llvm_const(8, "i64")
    acc = tle.llvm_const("0.000000e+00", "vector<[8]xf32>")
    zero = tle.llvm_const(0, "i64")
    for k in tle.range(zero, N, vl):
        pa = tle.llvm_poison("vector<[8]xf32>")
        pb = tle.llvm_poison("vector<[8]xf32>")
        ga = tle.llvm_gep(tle.llvm_base_ptr(A), k, "f32")
        gb = tle.llvm_gep(tle.llvm_base_ptr(B), k, "f32")
        va = tle.call_intrinsic("llvm.riscv.vle", [pa, ga, vl], result_type="vector<[8]xf32>")
        vb = tle.call_intrinsic("llvm.riscv.vle", [pb, gb, vl], result_type="vector<[8]xf32>")
        prod = tle.llvm_fmul(va, vb)
        acc = tle.llvm_fadd(acc, prod)
    # reduce acc to scalar (简化:只写 acc[0],实际应 vredsum)
    gc = tle.llvm_base_ptr(C)
    tle.call_intrinsic("llvm.riscv.vse", [acc, gc, vl], result_type="()")
'''

_BIN = "/home/share/nfs_share/zuoweixia/.worktrees/tmr3jn4lr/build-x86_64/triton/backends/spine_triton/bin"
_MATTR = "64bit,a,b,c,d,f,i,m,v,zfh,zvfh,zicbop,zicbom,zicboz,xsmtvdotii"


def _build_fn(name, src, anns):
    """Attach signature annotations + AST source to a real function object."""
    import ast
    tree = ast.parse(src)
    code = compile(tree, f"<{name}>", "exec")
    g = {}
    exec(code, g)
    fn = g[name]
    fn.__annotations__ = anns
    fn._mode1_src = src
    return fn


def _emit(fn):
    import inspect
    _orig = inspect.getsource
    inspect.getsource = lambda f: fn._mode1_src if f is fn else _orig(f)
    try:
        return emit_mode1_module(fn)
    finally:
        inspect.getsource = _orig


def _check(name, src, anns):
    print(f"\n########## {name} ##########")
    fn = _build_fn(name, src, anns)
    mod = _emit(fn)
    print("=== emitted module ===")
    print(mod)

    import tempfile
    d = tempfile.mkdtemp(prefix="mode1_")
    inp = os.path.join(d, "in.mlir")
    o1 = os.path.join(d, "out.mlir")
    o2 = os.path.join(d, "out.ll")
    o3 = os.path.join(d, "out.o")
    open(inp, "w").write(mod)

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr

    rc, _, err = run([f"{_BIN}/spine-opt", inp,
                      '--spine-triton-e2e-pipeline=enable-always-tls=1 enable-fuse-group=false',
                      "-o", o1])
    print("spine-opt rc", rc, err[-800:] if rc else "")
    assert rc == 0, "spine-opt failed"

    rc, _, err = run([f"{_BIN}/mlir-translate", o1, "--mlir-to-llvmir", "-o", o2])
    print("translate rc", rc, err[-800:] if rc else "")
    assert rc == 0, "mlir-translate failed"

    rc, _, err = run([f"{_BIN}/llc", "-O3", "--float-abi=hard", "--relocation-model=pic",
                      "--march=riscv64", "--mattr=" + _MATTR, o2, "-filetype=obj", "-o", o3])
    print("llc rc", rc, err[-800:] if rc else "")
    assert rc == 0, "llc riscv64 failed"

    nm = f"{_BIN}/llvm-nm" if os.path.exists(f"{_BIN}/llvm-nm") else "nm"
    rc, out, _ = run([nm, o3])
    print("symbols:\n", out)
    assert f" T {name}" in out, "kernel symbol not exported"
    print(f"PASS: {name} -> riscv64 .o with exported symbol")


def main():
    _check("mode1_copy", _KSRC,
           {"X": mem(f16), "out": mem(f16, out=True), "N": index})
    _check("mode1_mv", _KSRC_MV,
           {"A": mem("f32"), "B": mem("f32"), "C": mem("f32", out=True), "K": index})
    _check("mode1_dot", _KSRC_DOT,
           {"A": mem("f32"), "B": mem("f32"), "C": mem("f32", out=True), "N": index})
    print("\nALL PASS (3 kernels: copy/mv/dot)")


if __name__ == "__main__":
    main()
