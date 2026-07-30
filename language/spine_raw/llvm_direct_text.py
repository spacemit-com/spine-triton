# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""LLVM-direct text emitter: @spine_raw AST → top-level `llvm.func` module TEXT.

Unlike SpineMLIRBuilderCodegen (which emits ops via the C++ builder API into a
`tle.dsl_region` that later inlines into a *func.func* — where LLVM ops trip
BufferDeallocation's "unknown memory side effects"), this backend produces a
standalone module whose body is a single top-level `llvm.func`. Pure llvm.func
is a no-op for BufferDeallocation/ConvertToScalableVector, so it bypasses
spine-opt entirely and feeds straight into mlir-translate → llc(riscv64).

ABI (matches build/.../driver.py `_launch`): each memref param is passed as
`(i64 rank, !llvm.ptr descriptor)`; scalars by value; then 6 trailing i32 =
gridX,gridY,gridZ, progX,progY,progZ. Single-program mode ignores prog*.

Emits TEXT only — no C++ rebuild needed; testable end-to-end on x86 up to the
llc riscv64 object.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from .codegen import _parse_signature

# Descriptor struct for a memref arg. The driver (backend/driver.py `_launch`)
# passes each memref as (int64_t rank=0, void* &ptr_arg) where ptr_arg is a
# rank-0 StridedMemRefType<char,0> = {allocated_ptr, aligned_ptr, offset}.
# There are NO sizes/strides arrays (rank 0), so the data pointer is field [1]
# (aligned) and llvm_size is unavailable in this ABI.
_DESC = "!llvm.struct<(ptr, ptr, i64)>"


class LLVMDirectTextCodegen:
    """Walk a @spine_raw fn and emit a top-level `llvm.func` module as text."""

    def __init__(self, sibling_abi: bool = False) -> None:
        self._ssa = 0            # %0, %1, ... counter
        self._blk = 0            # ^bb0, ^bb1, ... counter
        self._lines: list[str] = []
        self._env: dict[str, str] = {}   # py var -> SSA name (e.g. "%3")
        self._types: dict[str, str] = {}  # SSA name -> mlir type
        self._desc_cache: dict[str, str] = {}  # pyname -> loaded-descriptor SSA
        self._arch = '0xA064'
        self._num_threads = 4
        # sibling_abi=True: called from a func.func sibling (mixed mode). Each
        # memref param arrives as a single i64 (the aligned data pointer, cast
        # from index by the host bridge), recovered via llvm.inttoptr — NOT the
        # driver's (i64 rank, !llvm.ptr descriptor) pair. No 6 trailing grid args.
        # Proven by test_manual_mixed_ir.py.
        self._sibling_abi = sibling_abi
        self._ptr_i64: dict[str, str] = {}  # pyname -> i64 arg holding data ptr

    # --- SSA / emit helpers ---
    def _fresh(self) -> str:
        s = f"%{self._ssa}"
        self._ssa += 1
        return s

    def _fresh_blk(self) -> str:
        s = f"^bb{self._blk}"
        self._blk += 1
        return s

    def _emit(self, line: str) -> None:
        self._lines.append("    " + line)

    def _emit_label(self, line: str) -> None:
        self._lines.append("  " + line)  # block labels at region indent

    def _def(self, rhs: str, typ: str) -> str:
        """Emit `%k = rhs` and record the result type; return the SSA name."""
        s = self._fresh()
        self._emit(f"{s} = {rhs}")
        self._types[s] = typ
        return s

    @staticmethod
    def _mem_elem(mlir_type: str) -> str:
        import re
        m = re.search(r'memref<\*x([a-z0-9]+)', mlir_type)
        if not m:
            raise ValueError(f"llvm-direct: bad memref type {mlir_type!r}")
        return m.group(1)

    # ------------------------------------------------------------------
    # Public entry: @spine_raw fn -> module text with a top-level llvm.func
    # ------------------------------------------------------------------
    def emit_module(self, fn) -> str:
        params = _parse_signature(fn)
        self._params = params  # Store for program_id computation
        src = textwrap.dedent(inspect.getsource(fn))
        func_node = next(n for n in ast.walk(ast.parse(src))
                         if isinstance(n, ast.FunctionDef))

        # --- signature: memref -> (i64 rank, !llvm.ptr); scalar -> i64;
        #     then 6 trailing i32 (gridX/Y/Z, progX/Y/Z, per driver ABI) ---
        self._mem_ptr: dict[str, str] = {}   # pyname -> !llvm.ptr arg holding descriptor addr
        self._mem_dtype: dict[str, str] = {}  # pyname -> element dtype (f16/f32)
        sig: list[str] = []
        ai = 0
        for pname, ann in params:
            if ann.mlir_type.startswith("memref"):
                sig.append(f"%arg{ai}: i64")
                sig.append(f"%arg{ai+1}: !llvm.ptr")
                self._mem_ptr[pname] = f"%arg{ai+1}"
                self._mem_dtype[pname] = self._mem_elem(ann.mlir_type)
                ai += 2
            else:  # scalar (index) passed by value as i64
                a = f"%arg{ai}"
                sig.append(f"{a}: i64")
                self._env[pname] = a
                self._types[a] = "i64"
                ai += 1
        for _ in range(6):  # gridX,gridY,gridZ, progX,progY,progZ
            sig.append(f"%arg{ai}: i32")
            ai += 1

        # --- body ---
        for stmt in func_node.body:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # docstring
            self._gen_stmt(stmt)
        self._emit("llvm.return")

        hdr = (f'module attributes {{dlti.target_system_spec = '
               f'#dlti.target_system_spec<"CPU" = #dlti.target_device_spec<'
               f'"arch_id" = "{self._arch}", "num_threads" = {self._num_threads} : i32>>, '
               f'tt.force_vector_interleave = 2 : i32}} {{')
        body = "\n".join(self._lines)
        return (f"{hdr}\n  llvm.func @{fn.__name__}({', '.join(sig)}) {{\n"
                f"{body}\n  }}\n}}\n")

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------
    def _gen_stmt(self, node) -> None:
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            ssa, typ = self._gen_expr(node.value)
            self._env[node.targets[0].id] = ssa
            self._types[ssa] = typ
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self._gen_call(node.value)  # void call_intrinsic (vse etc.)
        elif isinstance(node, ast.For):
            self._gen_for(node)
        elif isinstance(node, (ast.Return, ast.Pass)):
            pass
        else:
            raise NotImplementedError(f"llvm-direct: unsupported stmt {ast.dump(node)}")

    def _gen_for(self, node: ast.For) -> None:
        """`for k in tle.range(lb[, ub, step]):` → llvm.br/cond_br loop.

        Vars reassigned in the body become iter-args (carried across the
        back-edge), mirroring host_ll.mlir's ^bb1(iv, acc...) header. Loop is
        [lb, ub) step. iv and iter-args are i64/typed block args.
        """
        assert isinstance(node.target, ast.Name)
        iv_name = node.target.id
        rargs = node.iter.args
        if len(rargs) == 1:
            lb = self._const_i64(0); ub, _ = self._gen_expr(rargs[0]); step = self._const_i64(1)
        else:
            lb, _ = self._gen_expr(rargs[0]); ub, _ = self._gen_expr(rargs[1]); step, _ = self._gen_expr(rargs[2])

        # iter-args: vars defined before the loop and reassigned inside it
        outer = set(self._env)
        reassigned = []
        for s in node.body:
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    if isinstance(t, ast.Name) and t.id in outer and t.id not in reassigned:
                        reassigned.append(t.id)
        ia_init = [(v, self._env[v], self._types[self._env[v]]) for v in reassigned]

        hdr, body, exit_ = self._fresh_blk(), self._fresh_blk(), self._fresh_blk()
        # entry -> header with initial values
        init_vals = ", ".join([lb] + [d[1] for d in ia_init])
        init_tys = ", ".join(["i64"] + [d[2] for d in ia_init])
        self._emit(f"llvm.br {hdr}({init_vals} : {init_tys})")

        # header block: bind iv + iter-arg block args, test, cond_br
        iv_ssa = self._fresh(); self._types[iv_ssa] = "i64"
        ia_hdr = [(v, self._fresh(), ty) for v, (_, _, ty) in zip(reassigned, ia_init)]
        for v, s, ty in ia_hdr:
            self._env[v] = s; self._types[s] = ty
        self._env[iv_name] = iv_ssa
        hargs = ", ".join([f"{iv_ssa}: i64"] + [f"{s}: {ty}" for _, s, ty in ia_hdr])
        self._emit_label(f"{hdr}({hargs}):")
        cond = self._def(f"llvm.icmp \"slt\" {iv_ssa}, {ub} : i64", "i1")
        self._emit(f"llvm.cond_br {cond}, {body}, {exit_}")

        # body block
        self._emit_label(f"{body}:")
        for s in node.body:
            self._gen_stmt(s)
        nxt = self._def(f"llvm.add {iv_ssa}, {step} : i64", "i64")
        back_vals = ", ".join([nxt] + [self._env[v] for v in reassigned])
        self._emit(f"llvm.br {hdr}({back_vals} : {init_tys})")

        # exit block: iter-args live on as their header block-arg values
        self._emit_label(f"{exit_}:")
        for v, s, ty in ia_hdr:
            self._env[v] = s; self._types[s] = ty

    # ------------------------------------------------------------------
    # Expressions -> (ssa_name, mlir_type)
    # ------------------------------------------------------------------
    def _gen_expr(self, node):
        if isinstance(node, ast.Call):
            return self._gen_call(node)
        if isinstance(node, ast.Name):
            s = self._env[node.id]
            return s, self._types.get(s, "i64")
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return self._const_i64(node.value), "i64"
        if isinstance(node, ast.BinOp):
            lv, _ = self._gen_expr(node.left)
            rv, _ = self._gen_expr(node.right)
            opn = {ast.Mult: "mul", ast.Add: "add", ast.Sub: "sub"}.get(type(node.op))
            if opn is None:
                raise NotImplementedError(f"llvm-direct: unsupported binop {type(node.op).__name__}")
            return self._def(f"llvm.{opn} {lv}, {rv} : i64", "i64"), "i64"
        raise NotImplementedError(f"llvm-direct: unsupported expr {ast.dump(node)}")

    def _const_i64(self, n: int) -> str:
        return self._def(f"llvm.mlir.constant({n} : i64) : i64", "i64")

    def _arg(self, node):
        """Resolve a call arg node to (ssa, type). Bare string literal -> passthrough."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value, None
        return self._gen_expr(node)

    def _gen_call(self, node: ast.Call):
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        h = getattr(self, f"_p_{fname}", None)
        if h is None:
            raise NotImplementedError(f"llvm-direct: unsupported primitive {fname!r}")
        return h(node)

    # ---- primitive handlers ------------------------------------------
    def _p_llvm_const(self, node):
        val = node.args[0].value
        typ = node.args[1].value
        if typ.startswith("vector<"):  # dense splat
            rhs = f"llvm.mlir.constant(dense<{val}> : {typ}) : {typ}"
        else:
            rhs = f"llvm.mlir.constant({val} : {typ}) : {typ}"
        return self._def(rhs, typ), typ

    def _p_llvm_poison(self, node):
        typ = node.args[0].value
        return self._def(f"llvm.mlir.poison : {typ}", typ), typ

    def _p_llvm_base_ptr(self, node):
        pname = node.args[0].id
        if self._sibling_abi:
            # Sibling ABI: memref arrives as a single i64 (aligned data ptr).
            # Recover the pointer via inttoptr, once, cached.
            base = self._desc_cache.get(pname)
            if base is None:
                i64_arg = self._ptr_i64[pname]
                base = self._def(f"llvm.inttoptr {i64_arg} : i64 to !llvm.ptr", "!llvm.ptr")
                self._desc_cache[pname] = base
            return base, "!llvm.ptr"
        desc = self._desc_cache.get(pname)
        if desc is None:
            ptr_arg = self._mem_ptr[pname]
            desc = self._def(f"llvm.load {ptr_arg} : !llvm.ptr -> {_DESC}", _DESC)
            self._desc_cache[pname] = desc
        base = self._def(f"llvm.extractvalue {desc}[1] : {_DESC}", "!llvm.ptr")
        return base, "!llvm.ptr"

    def _p_llvm_gep(self, node):
        base, _ = self._gen_expr(node.args[0])
        off, _ = self._gen_expr(node.args[1])
        elem = node.args[2].value if len(node.args) > 2 else "f16"
        rhs = f"llvm.getelementptr {base}[{off}] : (!llvm.ptr, i64) -> !llvm.ptr, {elem}"
        return self._def(rhs, "!llvm.ptr"), "!llvm.ptr"

    def _p_llvm_size(self, node):
        """llvm_size(mem[, dim]) — UNSUPPORTED in the llvm-direct driver ABI.

        The driver passes memrefs as rank-0 StridedMemRefType descriptors
        {allocated, aligned, offset} with no sizes/strides. There is no dim
        size to read. Pass shape info as an explicit scalar kernel parameter
        (e.g. K: tle.index) and use that for loop bounds instead.
        """
        raise NotImplementedError(
            "llvm-direct: llvm_size is unavailable — the driver ABI passes rank-0 "
            "memref descriptors with no shape. Pass sizes as scalar params "
            "(e.g. K: tle.index) and use them for loop bounds.")

    def _p_program_id(self, node):
        """program_id(axis) -> i64 index of current program in the grid.

        The driver ABI passes 6 trailing i32 args: gridX/Y/Z, progX/Y/Z.
        program_id(0) -> progX, program_id(1) -> progY, program_id(2) -> progZ.
        These are at indices [N, N+1, ..., N+5] where N = len(user params).
        Grid args start at arg index N+3 (after gridX/Y/Z).
        """
        axis = node.args[0].value
        if axis not in (0, 1, 2):
            raise ValueError(f"program_id axis must be 0, 1, or 2; got {axis}")

        # Count user params to find where grid/prog args start
        # Each memref param takes 2 args (i64 rank, !llvm.ptr), scalar takes 1 (i64)
        n_user_args = sum(2 if p[1].mlir_type.startswith("memref") else 1
                          for p in self._params)
        # Trailing 6 args: gridX/Y/Z (indices n_user_args+0/1/2), progX/Y/Z (indices n_user_args+3/4/5)
        prog_arg_idx = n_user_args + 3 + axis  # progX at +3, progY at +4, progZ at +5

        prog_ssa = f"%arg{prog_arg_idx}"
        # Driver passes as i32, need to extend to i64 for arithmetic
        pid_i64 = self._def(f"llvm.sext {prog_ssa} : i32 to i64", "i64")
        return pid_i64, "i64"

    def _p_call_intrinsic(self, node):
        intrin = node.args[0].value
        elts = node.args[1].elts
        rt = "()"
        for kw in node.keywords:
            if kw.arg == "result_type":
                rt = kw.value.value
        ops, tys = [], []
        for e in elts:
            s, t = self._arg(e)
            ops.append(s)
            tys.append(t if t is not None else self._types.get(s, "i64"))

        # Detect: plain LLVM op (llvm.fadd) vs intrinsic (llvm.riscv.vle / llvm.sadd.with.overflow)
        # Heuristic: if name contains '.' after 'llvm', it's an intrinsic; otherwise plain op.
        # Plain ops: llvm.fadd, llvm.fmul, llvm.getelementptr (emitted directly)
        # Intrinsics: llvm.riscv.vle, llvm.sadd.with.overflow (wrapped in llvm.call_intrinsic)
        is_intrinsic = '.' in intrin[5:] if intrin.startswith("llvm.") else False

        if is_intrinsic:
            sig = f"({', '.join(tys)}) -> {rt}"
            call = f'llvm.call_intrinsic "{intrin}"({", ".join(ops)}) : {sig}'
        else:
            # Plain LLVM op: emit directly as "llvm.fadd %0, %1 : vector<[8]xf32>"
            # For binary ops, signature is just the result type (operands already typed)
            call = f'{intrin} {", ".join(ops)} : {rt}'

        if rt == "()":
            self._emit(call)
            return None, "()"
        return self._def(call, rt), rt


def emit_llvm_func_for_inline(fn) -> tuple[str, list[str]]:
    """Emit an llvm.func that can be called from a host func.func.

    Unlike emit_module (which wraps the llvm.func in a standalone module),
    this returns just the function text to be appended as a sibling in a
    mixed-mode module.

    Returns:
        (func_text, param_types) where:
        - func_text is the complete llvm.func definition (no module wrapper)
        - param_types is a list of MLIR type strings for the call site
          Format: ["i64", "!llvm.ptr", "i64", ...] (memref→i64+ptr, scalar→i64)
    """
    codegen = LLVMDirectTextCodegen(sibling_abi=True)
    params = _parse_signature(fn)
    codegen._params = params
    src = textwrap.dedent(inspect.getsource(fn))
    func_node = next(n for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.FunctionDef))

    # Sibling ABI (called from func.func, see test_manual_mixed_ir.py):
    #   memref param → single i64 (aligned data ptr, cast from index by host)
    #   scalar param → single i64
    # No 6 trailing grid args. base_ptr recovered via llvm.inttoptr inside body.
    codegen._mem_ptr: dict[str, str] = {}
    codegen._mem_dtype: dict[str, str] = {}
    sig: list[str] = []
    param_types: list[str] = []
    ai = 0

    for pname, ann in params:
        a = f"%arg{ai}"
        sig.append(f"{a}: i64")
        param_types.append("i64")
        if ann.mlir_type.startswith("memref"):
            codegen._ptr_i64[pname] = a
            codegen._mem_dtype[pname] = codegen._mem_elem(ann.mlir_type)
        else:  # scalar
            codegen._env[pname] = a
            codegen._types[a] = "i64"
        ai += 1

    # Generate body
    for stmt in func_node.body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        codegen._gen_stmt(stmt)
    codegen._emit("llvm.return")

    # Return just the llvm.func text (no module wrapper)
    body = "\n".join(codegen._lines)
    func_text = f"  llvm.func @{fn.__name__}({', '.join(sig)}) {{\n{body}\n  }}"

    return func_text, param_types


def emit_llvm_direct_module(fn) -> str:
    """Convenience: build a fresh codegen and return the module text."""
    return LLVMDirectTextCodegen().emit_module(fn)
