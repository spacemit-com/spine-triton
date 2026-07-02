# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""SpineMLIRCodeGenerator — translates @spine_raw Python functions to Linalg MLIR.

Phase 1: AST visitor for the spine_raw eDSL subset.
Supports: scf.for with iter_args, vector/arith/memref ops via spine_raw builtins.
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap
from typing import Callable

from .types import _TypedAnnotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_signature(fn: Callable) -> list[tuple[str, _TypedAnnotation]]:
    sig = inspect.signature(fn)
    result = []
    for pname, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            raise ValueError(
                f"Parameter '{pname}' of @spine_raw function '{fn.__name__}' "
                f"must have an In[...] or InOut[...] annotation."
            )
        if not isinstance(ann, _TypedAnnotation):
            raise ValueError(
                f"Parameter '{pname}' annotation must be In[...] or InOut[...], got {ann!r}"
            )
        result.append((pname, ann))
    return result


def _find_reassigned(body: list, outer_vars: set) -> set:
    """Variables in outer_vars that are assigned inside body (direct stmts only)."""
    found = set()
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id in outer_vars:
                    found.add(t.id)
    return found


def _eval_list_literal(node) -> list:
    if isinstance(node, ast.List):
        return [ast.literal_eval(e) for e in node.elts]
    return [ast.literal_eval(node)]


def _vec_n(mlir_type: str) -> int:
    m = re.match(r'vector<(\d+)x', mlir_type)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract size from {mlir_type!r}")


def _vec_elem(mlir_type: str) -> str:
    m = re.match(r'vector<\d+x(.+)>', mlir_type)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract elem type from {mlir_type!r}")


def _memref_elem(mlir_type: str) -> str:
    """Element dtype of a plain ranked memref, e.g. memref<1x?x4x64xf16> -> f16."""
    m = re.findall(r'x(bf16|f16|f32|f64|i8|i16|i32|i64)', mlir_type)
    if m:
        return m[-1]
    raise ValueError(f"Cannot extract elem type from {mlir_type!r}")


_SPINE_RAW_BUILTIN_NAMES = {"splat", "load_vec", "store_vec", "fma", "extf",
                            "reduce_add", "matmul", "load_tile", "pad_vec",
                            "extract_elem", "batch_macc", "view_2d", "load_2d",
                            "load_2d_at", "load_2d_t", "pack_2d_t",
                            "alloc_tcm_2d", "pack_2d_t_into", "free_tcm",
                            "splat_2d", "store_2d", "store_2d_at", "range", "proton_mark",
                            "vconfig", "vzero", "vload", "vmacc", "vreduce_sum",
                            "vstore", "alloc", "vpack"}


def _vl_from_sew(sew_bytes: int) -> int:
    """Fixed VL (element count) for a K3 scalable register at the given SEW.

    K3 vlen = 1024 bits = 128 bytes, so VL = 128 / sew_bytes:
      sew=2 (f16) -> 64, sew=1 (i8) -> 128, sew=4 (f32) -> 32.
    """
    return 128 // sew_bytes


def _is_spine_raw_attr(node, attr: str, aliases: set | None = None) -> bool:
    """Check if node is <alias>.<attr> where alias is a spine_raw module import."""
    if not (isinstance(node, ast.Attribute) and node.attr == attr):
        return False
    if not isinstance(node.value, ast.Name):
        return False
    if aliases is not None:
        return node.value.id in aliases
    # fallback: accept any name when aliases not provided
    return True


_DTYPE_NAMES = {"f16", "f32", "bf16", "f64", "i8", "i16", "i32"}


def _resolve_dtype(node, default: str = "f16") -> str:
    """Resolve a dtype arg written as a string literal ("f16") or a bare name
    (f16 / f32 / bf16, the module-level dtype constants) to its MLIR string."""
    if node is None:
        return default
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in _DTYPE_NAMES:
        return node.id
    # last resort: literal_eval (raises for anything unexpected)
    return ast.literal_eval(node)




# ---------------------------------------------------------------------------
# SpineMLIRCodeGenerator
# ---------------------------------------------------------------------------

class SpineMLIRCodeGenerator(ast.NodeVisitor):
    """Translate a @spine_raw Python function to a func.func MLIR string.

    Supported subset:
      - Function parameters with In[...] / InOut[...] annotations
      - spine_raw.splat / load_vec / store_vec / fma / extf builtins
      - for VAR in spine_raw.range(N): with automatic iter_arg detection
      - BinOp + / * / - on index types → arith.addi / muli / subi
      - Integer / float constants
    """

    def __init__(self):
        self._env: dict[str, tuple[str, str]] = {}
        self._preamble: list[str] = []     # constant defs, always at indent=2
        self._lines: list[str] = []        # body ops
        self._indent: int = 2
        self._counter: int = 0
        self._defined_ssas: set[str] = set()
        self._const_ints: dict[int, str] = {}
        self._const_floats: dict[tuple, str] = {}
        self._all_iter_arg_names: set[str] = set()
        self._loop_iter_args: set[str] = set()
        # svector eDSL: compile-time VL tracking (feishu 3.3). vconfig-assigned
        # names are constexpr ints, never SSA/iter_args; _active_vl is the VL
        # used by vzero/vload/vmacc (fixed-length, no dynamic vsetvl this round).
        self._constexpr_ints: dict[str, int] = {}
        self._active_vl: int | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self, fn: Callable) -> str:
        """Return a bare func.func @name(...) { ... } MLIR string for fn.

        Translates @spine_raw Python function to MLIR via AST visitor.
        """
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if not func_nodes:
            raise ValueError(f"No function definition found in {fn.__name__!r}")

        # Detect all aliases for the spine_raw module in the function's globals
        try:
            import spine_raw as _sr_mod
        except ModuleNotFoundError:
            # In build environment, spine_raw is at triton.language.extra.spine_raw
            try:
                from triton.language.extra import spine_raw as _sr_mod
            except (ModuleNotFoundError, ImportError):
                _sr_mod = None

        self._aliases: set[str] = set()
        if _sr_mod is not None:
            for k, v in (fn.__globals__ or {}).items():
                if v is _sr_mod:
                    self._aliases.add(k)
        if not self._aliases:
            self._aliases = {"spine_raw", "sr"}  # sensible fallback

        return self._gen_func(func_nodes[0], fn)

    # ------------------------------------------------------------------
    # SSA allocation helpers
    # ------------------------------------------------------------------

    def _alloc_ssa(self, hint: str) -> str:
        """Allocate a unique SSA name, deduplicating with counter suffix."""
        if hint not in self._defined_ssas:
            self._defined_ssas.add(hint)
            return f"%{hint}"
        self._counter += 1
        candidate = f"{hint}_{self._counter}"
        while candidate in self._defined_ssas:
            self._counter += 1
            candidate = f"{hint}_{self._counter}"
        self._defined_ssas.add(candidate)
        return f"%{candidate}"

    def _emit(self, line: str):
        self._lines.append(" " * self._indent + line)

    def _const_int(self, n: int) -> str:
        if n not in self._const_ints:
            name = f"c{abs(n)}" + ("" if n >= 0 else "_neg")
            ssa = self._alloc_ssa(name)
            self._const_ints[n] = ssa
            self._preamble.append(f"  {ssa} = arith.constant {n} : index")
        return self._const_ints[n]

    def _const_float(self, v: float, ftype: str = "f32") -> str:
        key = (v, ftype)
        if key not in self._const_floats:
            if v == 0.0:
                hint = f"zero_{ftype}"
                lit = "0.000000e+00"
            else:
                hint = f"cf_{ftype}"
                lit = f"{v:e}"
            ssa = self._alloc_ssa(hint)
            self._const_floats[key] = ssa
            self._preamble.append(f"  {ssa} = arith.constant {lit} : {ftype}")
        return self._const_floats[key]

    def _bind(self, name: str, ssa: str, typ: str):
        self._env[name] = (ssa, typ)

    def _get(self, name: str) -> tuple[str, str]:
        if name not in self._env:
            raise ValueError(f"Undefined variable: {name!r}")
        return self._env[name]

    # ------------------------------------------------------------------
    # Function-level generation
    # ------------------------------------------------------------------

    def _gen_func(self, node: ast.FunctionDef, fn: Callable) -> str:
        # Reset state
        self.__init__()

        params = _parse_signature(fn)
        fname = node.name

        # Bind parameters
        for pname, ann in params:
            self._defined_ssas.add(pname)
            self._env[pname] = (f"%{pname}", ann.mlir_type)

        # Pre-scan: find all variables that will become iter_args in for loops
        defined_so_far: set[str] = set(self._env.keys())
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        defined_so_far.add(t.id)
            elif isinstance(stmt, ast.For):
                self._all_iter_arg_names |= _find_reassigned(stmt.body, defined_so_far)

        # Build signature
        sig_parts = [f"    %{pname} : {ann.mlir_type}" for pname, ann in params]
        header = f"func.func @{fname}(\n" + ",\n".join(sig_parts) + "\n) {"

        # Generate body statements
        for stmt in node.body:
            if isinstance(stmt, ast.Pass):
                continue
            # Skip decorator / docstring expressions
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            self._gen_stmt(stmt)

        self._emit("return")

        body = "\n".join(self._preamble + self._lines)
        return f"{header}\n{body}\n}}"

    # ------------------------------------------------------------------
    # Statement generation
    # ------------------------------------------------------------------

    def _gen_stmt(self, node):
        if isinstance(node, ast.Assign):
            self._gen_assign(node)
        elif isinstance(node, ast.For):
            self._gen_for(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self._gen_call_stmt(node.value)
        elif isinstance(node, ast.Return):
            pass  # handled by caller
        elif isinstance(node, ast.Pass):
            pass
        else:
            raise NotImplementedError(f"Unsupported statement: {ast.dump(node)}")

    def _gen_assign(self, node: ast.Assign):
        assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        target = node.targets[0].id

        # vconfig(avl, sew) → compile-time VL constant, tracked separately from
        # SSA env so it never becomes an scf.for iter_arg. Records the active VL
        # used by subsequent vzero/vload/vmacc calls.
        if isinstance(node.value, ast.Call) and \
                _is_spine_raw_attr(node.value.func, "vconfig", self._aliases):
            self._gen_vconfig_assign(target, node.value)
            return

        in_loop = target in self._loop_iter_args
        is_future_iter = target in self._all_iter_arg_names

        if in_loop:
            hint = f"{target}_upd"
        elif is_future_iter:
            hint = f"{target}_init"
        else:
            hint = target

        ssa, typ = self._gen_expr(node.value, hint=hint)
        self._bind(target, ssa, typ)

    def _gen_for(self, node: ast.For):
        assert isinstance(node.target, ast.Name), "for target must be a simple name"
        loop_var = node.target.id

        assert _is_spine_raw_attr(node.iter.func, "range", self._aliases), \
            "for loop iter must be spine_raw.range(...)"
        rargs = node.iter.args
        assert len(rargs) in (1, 3), \
            "spine_raw.range takes range(stop) or range(start, stop, step)"
        if len(rargs) == 1:
            lb_ssa = self._const_int(0)
            ub_ssa, _ = self._gen_expr(rargs[0])
            step_ssa = self._const_int(1)
        else:
            lb_ssa, _ = self._gen_expr(rargs[0])
            ub_ssa, _ = self._gen_expr(rargs[1])
            step_ssa, _ = self._gen_expr(rargs[2])

        outer_vars = set(self._env.keys())
        reassigned = _find_reassigned(node.body, outer_vars)
        iter_args = sorted(reassigned)  # deterministic order

        # Gather init SSAs
        ia_data = [(v, *self._get(v)) for v in iter_args]  # (name, ssa, type)

        # Allocate result SSAs (use the Python var name directly)
        result_ssas = {v: self._alloc_ssa(v) for v in iter_args}

        # Allocate loop variable SSA
        loop_ssa = self._alloc_ssa(loop_var)
        self._bind(loop_var, loop_ssa, "index")

        # Emit scf.for header
        if ia_data:
            res_part = ", ".join(result_ssas[v] for v in iter_args)
            ia_part = ", ".join(f"%{v}_in = {init_ssa}" for v, init_ssa, _ in ia_data)
            types_part = ", ".join(typ for _, _, typ in ia_data)
            for_line = (
                f"{res_part} = scf.for {loop_ssa} = {lb_ssa} to {ub_ssa} step {step_ssa}"
                f" iter_args({ia_part}) -> ({types_part}) {{"
            )
        else:
            for_line = f"scf.for {loop_ssa} = {lb_ssa} to {ub_ssa} step {step_ssa} {{"

        self._emit(for_line)

        # Inside loop: rebind iter_args to _in SSAs
        old_env = {}
        for v, _, typ in ia_data:
            old_env[v] = self._env[v]
            in_name = f"{v}_in"
            self._defined_ssas.add(in_name)
            self._bind(v, f"%{in_name}", typ)

        prev_loop_iter_args = self._loop_iter_args
        self._loop_iter_args = set(iter_args)

        self._indent += 2
        for stmt in node.body:
            self._gen_stmt(stmt)

        # Emit scf.yield
        if ia_data:
            yield_ssas = [self._get(v)[0] for v in iter_args]
            yield_types = [self._get(v)[1] for v in iter_args]
            self._emit(
                f"scf.yield {', '.join(yield_ssas)} : {', '.join(yield_types)}"
            )

        self._indent -= 2
        self._emit("}")

        self._loop_iter_args = prev_loop_iter_args

        # After loop: rebind iter_arg names to result SSAs
        for v, _, typ in ia_data:
            self._bind(v, result_ssas[v], typ)

    def _gen_call_stmt(self, node: ast.Call):
        if _is_spine_raw_attr(node.func, "store_vec", self._aliases):
            self._gen_store_vec(node)
        elif _is_spine_raw_attr(node.func, "store_scalar", self._aliases):
            self._gen_store_scalar(node)
        elif _is_spine_raw_attr(node.func, "store_2d", self._aliases):
            self._gen_store_2d(node)
        elif _is_spine_raw_attr(node.func, "store_2d_at", self._aliases):
            self._gen_store_2d_at(node)
        elif _is_spine_raw_attr(node.func, "pack_2d_t_into", self._aliases):
            self._gen_pack_2d_t_into(node)
        elif _is_spine_raw_attr(node.func, "free_tcm", self._aliases):
            self._gen_free_tcm(node)
        elif _is_spine_raw_attr(node.func, "proton_mark", self._aliases):
            self._gen_proton_mark(node)
        elif _is_spine_raw_attr(node.func, "vstore", self._aliases):
            self._gen_vstore(node)
        elif _is_spine_raw_attr(node.func, "vpack", self._aliases):
            self._gen_vpack(node)
        else:
            raise NotImplementedError(
                f"Unsupported call statement: {ast.dump(node.func)}"
            )

    # ------------------------------------------------------------------
    # Expression generation
    # ------------------------------------------------------------------

    def _gen_expr(self, node, hint: str = "") -> tuple[str, str]:
        if isinstance(node, ast.Name):
            # constexpr int (e.g. nvl from vconfig) materializes as an index const
            if node.id in self._constexpr_ints:
                return self._const_int(self._constexpr_ints[node.id]), "index"
            return self._get(node.id)
        if isinstance(node, ast.Constant):
            return self._gen_literal(node)
        if isinstance(node, ast.BinOp):
            return self._gen_binop(node, hint)
        if isinstance(node, ast.Call):
            return self._gen_call_expr(node, hint)
        raise NotImplementedError(f"Unsupported expr: {ast.dump(node)}")

    def _gen_literal(self, node: ast.Constant) -> tuple[str, str]:
        v = node.value
        if isinstance(v, int):
            return self._const_int(v), "index"
        if isinstance(v, float):
            return self._const_float(v), "f32"
        raise NotImplementedError(f"Unsupported literal: {v!r}")

    def _gen_binop(self, node: ast.BinOp, hint: str) -> tuple[str, str]:
        lssa, ltype = self._gen_expr(node.left)
        rssa, rtype = self._gen_expr(node.right)
        op = type(node.op)
        result = self._alloc_ssa(hint or "t")

        if ltype == "index" and rtype == "index":
            opname = {ast.Add: "addi", ast.Mult: "muli", ast.Sub: "subi",
                      ast.FloorDiv: "divui"}.get(op)
            if opname is None:
                raise NotImplementedError(f"BinOp {op.__name__} not supported for index")
            self._emit(f"{result} = arith.{opname} {lssa}, {rssa} : index")
            return result, "index"

        if ltype == rtype and ltype.startswith("vector<"):
            opname = {ast.Add: "addf", ast.Mult: "mulf", ast.Sub: "subf"}.get(op)
            if opname is None:
                raise NotImplementedError(f"BinOp {op.__name__} not supported for {ltype}")
            self._emit(f"{result} = arith.{opname} {lssa}, {rssa} : {ltype}")
            return result, ltype

        raise NotImplementedError(
            f"BinOp between {ltype!r} and {rtype!r} not supported"
        )

    def _gen_call_expr(self, node: ast.Call, hint: str) -> tuple[str, str]:
        if _is_spine_raw_attr(node.func, "splat", self._aliases):
            return self._gen_splat(node, hint)
        if _is_spine_raw_attr(node.func, "load_vec", self._aliases):
            return self._gen_load_vec(node, hint)
        if _is_spine_raw_attr(node.func, "extf", self._aliases):
            return self._gen_extf(node, hint)
        if _is_spine_raw_attr(node.func, "fma", self._aliases):
            return self._gen_fma(node, hint)
        if _is_spine_raw_attr(node.func, "reduce_add", self._aliases):
            return self._gen_reduce_add(node, hint)
        if _is_spine_raw_attr(node.func, "matmul", self._aliases):
            return self._gen_matmul(node, hint)
        if _is_spine_raw_attr(node.func, "load_tile", self._aliases):
            return self._gen_load_tile(node, hint)
        if _is_spine_raw_attr(node.func, "pad_vec", self._aliases):
            return self._gen_pad_vec(node, hint)
        if _is_spine_raw_attr(node.func, "extract_elem", self._aliases):
            return self._gen_extract_elem(node, hint)
        if _is_spine_raw_attr(node.func, "batch_macc", self._aliases):
            return self._gen_batch_macc(node, hint)
        if _is_spine_raw_attr(node.func, "view_2d", self._aliases):
            return self._gen_view_2d(node, hint)
        if _is_spine_raw_attr(node.func, "load_2d", self._aliases):
            return self._gen_load_2d(node, hint)
        if _is_spine_raw_attr(node.func, "load_2d_at", self._aliases):
            return self._gen_load_2d_at(node, hint)
        if _is_spine_raw_attr(node.func, "load_2d_t", self._aliases):
            return self._gen_load_2d_t(node, hint)
        if _is_spine_raw_attr(node.func, "pack_2d_t", self._aliases):
            return self._gen_pack_2d_t(node, hint)
        if _is_spine_raw_attr(node.func, "alloc_tcm_2d", self._aliases):
            return self._gen_alloc_tcm_2d(node, hint)
        if _is_spine_raw_attr(node.func, "splat_2d", self._aliases):
            return self._gen_splat_2d(node, hint)
        if _is_spine_raw_attr(node.func, "vzero", self._aliases):
            return self._gen_vzero(node, hint)
        if _is_spine_raw_attr(node.func, "vload", self._aliases):
            return self._gen_vload(node, hint)
        if _is_spine_raw_attr(node.func, "vmacc", self._aliases):
            return self._gen_vmacc(node, hint)
        if _is_spine_raw_attr(node.func, "vreduce_sum", self._aliases):
            return self._gen_vreduce_sum(node, hint)
        if _is_spine_raw_attr(node.func, "alloc", self._aliases):
            return self._gen_alloc(node, hint)
        raise NotImplementedError(f"Unsupported call: {ast.dump(node.func)}")

    # ------------------------------------------------------------------
    # spine_raw builtin implementations
    # ------------------------------------------------------------------

    def _gen_splat(self, node: ast.Call, hint: str) -> tuple[str, str]:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        val_node = node.args[0] if node.args else kwargs["val"]
        shape_node = kwargs["shape"]
        shape = _eval_list_literal(shape_node)
        assert len(shape) == 1, "spine_raw.splat only supports 1D shape"
        N = shape[0]

        val_ssa, _ = self._gen_expr(val_node)
        vec_type = f"vector<{N}xf32>"
        result = self._alloc_ssa(hint or "splat")
        # Use vector.broadcast instead of vector.splat for compatibility
        self._emit(f"{result} = vector.broadcast {val_ssa} : f32 to {vec_type}")
        return result, vec_type

    def _gen_load_vec(self, node: ast.Call, hint: str) -> tuple[str, str]:
        args = node.args
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_node = args[0]
        idx_node = args[1]
        N = ast.literal_eval(args[2]) if len(args) > 2 else ast.literal_eval(kwargs["N"])
        dtype_node = kwargs.get("dtype") or (args[3] if len(args) > 3 else None)
        dtype = ast.literal_eval(dtype_node) if dtype_node else "f32"
        # in_bounds=False allows out-of-bounds reads (returns pad value); default True
        in_bounds_node = kwargs.get("in_bounds")
        in_bounds = ast.literal_eval(in_bounds_node) if in_bounds_node else True
        in_bounds_str = "true" if in_bounds else "false"

        ptr_ssa, ptr_type = self._gen_expr(ptr_node)
        idx_ssa, _ = self._gen_expr(idx_node)

        # vector.load requires ranked memref; if ptr_type is unranked (memref<*x...>),
        # cast to memref<?x...> first
        load_ptr_ssa = ptr_ssa
        load_ptr_type = ptr_type
        if ptr_type.startswith("memref<*x"):
            # Extract element type and address space from memref<*xTYPE, #space>
            # e.g., "memref<*xf32, #ptr.generic_space>" -> "memref<?xf32, #ptr.generic_space>"
            ranked_type = ptr_type.replace("memref<*x", "memref<?x", 1)
            cast_ssa = self._alloc_ssa("ranked")
            self._emit(f"{cast_ssa} = memref.cast {ptr_ssa} : {ptr_type} to {ranked_type}")
            load_ptr_ssa = cast_ssa
            load_ptr_type = ranked_type

        vec_type = f"vector<{N}x{dtype}>"
        result = self._alloc_ssa(hint or "vec")
        pad_ssa = self._const_float(0.0, dtype)
        self._emit(
            f"{result} = vector.transfer_read {load_ptr_ssa}[{idx_ssa}], {pad_ssa}"
            f" {{in_bounds = [{in_bounds_str}]}} : {load_ptr_type}, {vec_type}"
        )
        return result, vec_type

    def _gen_extf(self, node: ast.Call, hint: str) -> tuple[str, str]:
        v_node = node.args[0]
        dst_dtype = ast.literal_eval(node.args[1]) if len(node.args) > 1 else "f32"
        v_ssa, v_type = self._gen_expr(v_node)
        N = _vec_n(v_type)
        dst_type = f"vector<{N}x{dst_dtype}>"
        result = self._alloc_ssa(hint or "extf")
        self._emit(f"{result} = arith.extf {v_ssa} : {v_type} to {dst_type}")
        return result, dst_type

    def _gen_fma(self, node: ast.Call, hint: str) -> tuple[str, str]:
        a_ssa, a_type = self._gen_expr(node.args[0])
        b_ssa, _      = self._gen_expr(node.args[1])
        c_ssa, c_type = self._gen_expr(node.args[2])
        assert a_type == c_type, f"fma: a and acc types must match: {a_type} vs {c_type}"
        result = self._alloc_ssa(hint or "fma")
        self._emit(f"{result} = math.fma {a_ssa}, {b_ssa}, {c_ssa} : {a_type}")
        return result, a_type

    def _gen_reduce_add(self, node: ast.Call, hint: str) -> tuple[str, str]:
        v_ssa, v_type = self._gen_expr(node.args[0])
        # vector<NxT> -> T via horizontal add reduction
        elem = v_type[v_type.index("x") + 1 : v_type.rindex(">")] if "x" in v_type else "f32"
        result = self._alloc_ssa(hint or "rsum")
        self._emit(f"{result} = vector.reduction <add>, {v_ssa} : {v_type} into {elem}")
        return result, elem

    def _gen_matmul(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # matmul(lhs, rhs, acc, m, n, k) -> vector_ext.matmul
        #   out[m,n] = acc[m,n] + sum_k lhs[m,k] * rhs[n,k]
        #   lhs : vector<(m*k)x f16>  (row-major m x k)
        #   rhs : vector<(n*k)x f16>  (row-major n x k)
        #   acc/out : vector<(m*n)x f32>
        kwargs = {kw.arg: kw.value for kw in node.keywords}

        def _pick(i, name):
            if i < len(node.args):
                return node.args[i]
            return kwargs[name]

        lhs_ssa, lhs_type = self._gen_expr(_pick(0, "lhs"))
        rhs_ssa, _        = self._gen_expr(_pick(1, "rhs"))
        acc_ssa, acc_type = self._gen_expr(_pick(2, "acc"))
        m = ast.literal_eval(_pick(3, "m"))
        n = ast.literal_eval(_pick(4, "n"))
        k = ast.literal_eval(_pick(5, "k"))

        rhs_type = f"vector<{n * k}x{_vec_elem(lhs_type)}>"
        result = self._alloc_ssa(hint or "mm")
        # Use generic op form so the text parses even when vector_ext dialect
        # is not registered in the consuming tool (spine-triton-opt). The
        # ConvertToScalableVector pass in spine-mlir-k3 will then lower it.
        self._emit(
            f'{result} = "vector_ext.matmul"({lhs_ssa}, {rhs_ssa}, {acc_ssa})'
            f' <{{m = {m} : i64, n = {n} : i64, k = {k} : i64}}>'
            f' : ({lhs_type}, {rhs_type}, {acc_type}) -> {acc_type}'
        )
        return result, acc_type

    def _ranked_cast(self, ptr_ssa: str, ptr_type: str) -> tuple[str, str]:
        """Cast memref<*xT> to memref<?xT>; return (ssa, type)."""
        if ptr_type.startswith("memref<*x"):
            ranked_type = ptr_type.replace("memref<*x", "memref<?x", 1)
            cast_ssa = self._alloc_ssa("ranked")
            self._emit(f"{cast_ssa} = memref.cast {ptr_ssa} : {ptr_type} to {ranked_type}")
            return cast_ssa, ranked_type
        return ptr_ssa, ptr_type

    def _gen_load_tile(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # load_tile(ptr, row_base, col_base, row_stride, M, K, dtype)
        #   Loads an M×K tile from row-major memory using a 2D transfer_read
        #   then shape_casts to vector<M*K x dtype>.
        #   Avoids sub-vscale vector sizes by reading the whole tile at once.
        kwargs = {kw.arg: kw.value for kw in node.keywords}

        def _pick(i, name):
            return node.args[i] if i < len(node.args) else kwargs[name]

        ptr_node = _pick(0, "ptr")
        row_base_node = _pick(1, "row_base")
        col_base_node = _pick(2, "col_base")
        row_stride = ast.literal_eval(_pick(3, "row_stride"))
        M = ast.literal_eval(_pick(4, "M"))
        K = ast.literal_eval(_pick(5, "K"))
        dt_node = node.args[6] if len(node.args) > 6 else kwargs.get("dtype")
        dtype = ast.literal_eval(dt_node) if dt_node else "f16"

        ptr_ssa, ptr_type = self._gen_expr(ptr_node)
        rb_ssa, _ = self._gen_expr(row_base_node)
        cb_ssa, _ = self._gen_expr(col_base_node)

        # Reinterpret 1D memref as 2D with explicit strides
        ranked1d_type = ptr_type.replace("memref<*x", "memref<?x", 1)
        ranked2d_type = f"memref<?x{row_stride}x{dtype}, strided<[{row_stride}, 1], offset: ?>>"
        addr_space = ""
        if "#ptr.generic_space" in ptr_type:
            ranked2d_type = f"memref<?x{row_stride}x{dtype}, strided<[{row_stride}, 1], offset: ?>, #ptr.generic_space>"

        c0 = self._const_int(0)
        ranked1d = self._alloc_ssa("ranked")
        self._emit(f"{ranked1d} = memref.cast {ptr_ssa} : {ptr_type} to {ranked1d_type}")
        view2d = self._alloc_ssa("view2d")
        size1d = self._alloc_ssa("sz1d")
        self._emit(f"{size1d} = memref.dim {ranked1d}, {c0} : {ranked1d_type}")
        self._emit(
            f"{view2d} = memref.reinterpret_cast {ranked1d} to "
            f"offset: [0], sizes: [{size1d}, {row_stride}], "
            f"strides: [{row_stride}, 1]"
            f" : {ranked1d_type} to {ranked2d_type}"
        )

        tile2d_type = f"vector<{M}x{K}x{dtype}>"
        flat_type = f"vector<{M * K}x{dtype}>"
        pad_ssa = self._const_float(0.0, dtype)
        result2d = self._alloc_ssa(hint or "tile2d")
        self._emit(
            f"{result2d} = vector.transfer_read {view2d}[{rb_ssa}, {cb_ssa}], {pad_ssa}"
            f" {{in_bounds = [true, true]}} : {ranked2d_type}, {tile2d_type}"
        )
        result = self._alloc_ssa(hint or "tile")
        self._emit(f"{result} = vector.shape_cast {result2d} : {tile2d_type} to {flat_type}")
        return result, flat_type

    def _gen_pad_vec(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # pad_vec(vec, total): place vec<Nx dtype> at offset 0 of vector<total x dtype>,
        # remaining elements zero. Used to build matmul rhs (B in row 0).
        vec_ssa, vec_type = self._gen_expr(node.args[0])
        total = ast.literal_eval(node.args[1])
        dtype = _vec_elem(vec_type)
        full_type = f"vector<{total}x{dtype}>"
        zero_f = self._const_float(0.0, dtype)
        base = self._alloc_ssa(hint or "pad")
        self._emit(f"{base} = vector.broadcast {zero_f} : {dtype} to {full_type}")
        result = self._alloc_ssa(hint or "pad")
        self._emit(
            f"{result} = vector.insert_strided_slice {vec_ssa}, {base}"
            f" {{offsets = [0], strides = [1]}} : {vec_type} into {full_type}"
        )
        return result, full_type

    def _gen_extract_elem(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # extract_elem(vec, idx): extract a single scalar element.
        #   idx may be a Python int literal (static) or an index SSA value
        #   (dynamic). Static -> vector.extract; dynamic -> vector.extractelement.
        vec_ssa, vec_type = self._gen_expr(node.args[0])
        dtype = _vec_elem(vec_type)
        result = self._alloc_ssa(hint or "elem")
        idx_node = node.args[1]
        if isinstance(idx_node, ast.Constant) and isinstance(idx_node.value, int):
            self._emit(
                f"{result} = vector.extract {vec_ssa}[{idx_node.value}]"
                f" : {dtype} from {vec_type}"
            )
        else:
            idx_ssa, idx_type = self._gen_expr(idx_node)
            if idx_type != "index":
                raise NotImplementedError(
                    f"extract_elem dynamic index must be index, got {idx_type}")
            self._emit(
                f"{result} = vector.extractelement {vec_ssa}[{idx_ssa} : index]"
                f" : {vec_type}"
            )
        return result, dtype

    # ------------------------------------------------------------------
    # 2D helpers for batch_macc (vfwmacc) path
    # ------------------------------------------------------------------
    def _strided_2d_type(self, rows, cols, dtype, space):
        # memref<rows x cols x dtype, strided<[cols, 1]>, space>
        sp = f", {space}" if space else ""
        return f"memref<{rows}x{cols}x{dtype}, strided<[{cols}, 1]>{sp}>"

    def _space_of(self, ptr_type):
        m = re.search(r',\s*(#[\w.]+)>', ptr_type)
        return m.group(1) if m else ""

    def _gen_view_2d(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # view_2d(ptr, rows, cols, dtype, off=0) -> 2D strided memref view (batch_macc lhs)
        #   off: optional dynamic element offset (e.g. B's K-block base).
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_ssa, ptr_type = self._gen_expr(node.args[0])
        rows = ast.literal_eval(node.args[1])
        cols = ast.literal_eval(node.args[2])
        dtype = ast.literal_eval(node.args[3]) if len(node.args) > 3 else "f16"
        off_node = node.args[4] if len(node.args) > 4 else kwargs.get("off")
        space = self._space_of(ptr_type)
        ranked = ptr_type.replace("memref<*x", "memref<?x", 1) if ptr_type.startswith("memref<*x") else ptr_type
        if ptr_type.startswith("memref<*x"):
            rcast = self._alloc_ssa("ranked")
            self._emit(f"{rcast} = memref.cast {ptr_ssa} : {ptr_type} to {ranked}")
            ptr_ssa = rcast
        if off_node is not None:
            off_ssa, _ = self._gen_expr(off_node)
            sp = f", {space}" if space else ""
            out_type = f"memref<{rows}x{cols}x{dtype}, strided<[{cols}, 1], offset: ?>{sp}>"
            off_part = f"[{off_ssa}]"
        else:
            out_type = self._strided_2d_type(rows, cols, dtype, space)
            off_part = "[0]"
        result = self._alloc_ssa(hint or "view2d")
        self._emit(
            f"{result} = memref.reinterpret_cast {ptr_ssa} to "
            f"offset: {off_part}, sizes: [{rows}, {cols}], strides: [{cols}, 1]"
            f" : {ranked} to {out_type}"
        )
        return result, out_type

    def _gen_load_2d(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # load_2d(ptr, rows, cols, dtype) -> vector<rows x cols x dtype>
        ptr_ssa, ptr_type = self._gen_expr(node.args[0])
        rows = ast.literal_eval(node.args[1])
        cols = ast.literal_eval(node.args[2])
        dtype = ast.literal_eval(node.args[3]) if len(node.args) > 3 else "f16"
        space = self._space_of(ptr_type)
        ranked = ptr_type.replace("memref<*x", "memref<?x", 1) if ptr_type.startswith("memref<*x") else ptr_type
        if ptr_type.startswith("memref<*x"):
            rcast = self._alloc_ssa("ranked")
            self._emit(f"{rcast} = memref.cast {ptr_ssa} : {ptr_type} to {ranked}")
            ptr_ssa = rcast
        mtype = self._strided_2d_type(rows, cols, dtype, space)
        view = self._alloc_ssa("view2d")
        self._emit(
            f"{view} = memref.reinterpret_cast {ptr_ssa} to "
            f"offset: [0], sizes: [{rows}, {cols}], strides: [{cols}, 1]"
            f" : {ranked} to {mtype}"
        )
        c0 = self._const_int(0)
        pad = self._const_float(0.0, dtype)
        vtype = f"vector<{rows}x{cols}x{dtype}>"
        result = self._alloc_ssa(hint or "ld2d")
        self._emit(
            f"{result} = vector.transfer_read {view}[{c0}, {c0}], {pad}"
            f" {{in_bounds = [true, true]}} : {mtype}, {vtype}"
        )
        return result, vtype

    def _gen_load_2d_at(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # load_2d_at(ptr, elem_off, rows, cols, dtype) -> vector<rows x cols>
        #   reinterpret a rows×cols block starting at dynamic element offset.
        ptr_ssa, ptr_type = self._gen_expr(node.args[0])
        off_ssa, _ = self._gen_expr(node.args[1])
        rows = ast.literal_eval(node.args[2])
        cols = ast.literal_eval(node.args[3])
        dtype = ast.literal_eval(node.args[4]) if len(node.args) > 4 else "f16"
        space = self._space_of(ptr_type)
        ranked = ptr_type.replace("memref<*x", "memref<?x", 1) if ptr_type.startswith("memref<*x") else ptr_type
        if ptr_type.startswith("memref<*x"):
            rcast = self._alloc_ssa("ranked")
            self._emit(f"{rcast} = memref.cast {ptr_ssa} : {ptr_type} to {ranked}")
            ptr_ssa = rcast
        # block element offset = elem_off * rows  (block size = rows*cols, elem_off = pid*cols)
        rows_c = self._const_int(rows)
        boff = self._alloc_ssa("boff")
        self._emit(f"{boff} = arith.muli {off_ssa}, {rows_c} : index")
        sp = f", {space}" if space else ""
        mtype = f"memref<{rows}x{cols}x{dtype}, strided<[{cols}, 1], offset: ?>{sp}>"
        view = self._alloc_ssa("view2d")
        self._emit(
            f"{view} = memref.reinterpret_cast {ptr_ssa} to "
            f"offset: [{boff}], sizes: [{rows}, {cols}], strides: [{cols}, 1]"
            f" : {ranked} to {mtype}"
        )
        c0 = self._const_int(0)
        pad = self._const_float(0.0, dtype)
        vtype = f"vector<{rows}x{cols}x{dtype}>"
        result = self._alloc_ssa(hint or "ld2d")
        self._emit(
            f"{result} = vector.transfer_read {view}[{c0}, {c0}], {pad}"
            f" {{in_bounds = [true, true]}} : {mtype}, {vtype}"
        )
        return result, vtype

    def _gen_load_2d_t(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # load_2d_t(ptr, row_base, K, NB, M, dtype, col_off=0) -> vector<K x NB>
        #   Transposed read from row-major A[N,M]: result[k,c] = A[row_base+c, col_off+k].
        #   View: offset=row_base*M + col_off, sizes=[K,NB], strides=[1, M]
        #   M (column stride) may be a Python int literal (static stride) OR an
        #   index SSA expression (dynamic stride: strided<[1, ?]>). Dynamic M lets
        #   the same kernel handle any M via a K-block loop (block_M = K = 32).
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_ssa, ptr_type = self._gen_expr(node.args[0])
        rb_ssa, _ = self._gen_expr(node.args[1])
        K = ast.literal_eval(node.args[2])
        NB = ast.literal_eval(node.args[3])
        m_node = node.args[4]
        dtype = ast.literal_eval(node.args[5]) if len(node.args) > 5 else "f16"
        col_off_node = node.args[6] if len(node.args) > 6 else kwargs.get("col_off")
        space = self._space_of(ptr_type)
        ranked = ptr_type.replace("memref<*x", "memref<?x", 1) if ptr_type.startswith("memref<*x") else ptr_type
        if ptr_type.startswith("memref<*x"):
            rcast = self._alloc_ssa("ranked")
            self._emit(f"{rcast} = memref.cast {ptr_ssa} : {ptr_type} to {ranked}")
            ptr_ssa = rcast
        # M: static int literal -> constant stride; else dynamic SSA -> ? stride
        m_static = (isinstance(m_node, ast.Constant) and isinstance(m_node.value, int))
        if m_static:
            M = m_node.value
            m_ssa = self._const_int(M)
            col_stride = str(M)
        else:
            m_ssa, _ = self._gen_expr(m_node)
            col_stride = "?"
        boff = self._alloc_ssa("boff")
        self._emit(f"{boff} = arith.muli {rb_ssa}, {m_ssa} : index")
        if col_off_node is not None:
            co_ssa, _ = self._gen_expr(col_off_node)
            boff2 = self._alloc_ssa("boff")
            self._emit(f"{boff2} = arith.addi {boff}, {co_ssa} : index")
            boff = boff2
        sp = f", {space}" if space else ""
        mtype = f"memref<{K}x{NB}x{dtype}, strided<[1, {col_stride}], offset: ?>{sp}>"
        # reinterpret_cast strides operand: dynamic stride passes the SSA value
        stride_op = str(M) if m_static else m_ssa
        view = self._alloc_ssa("viewT")
        self._emit(
            f"{view} = memref.reinterpret_cast {ptr_ssa} to "
            f"offset: [{boff}], sizes: [{K}, {NB}], strides: [1, {stride_op}]"
            f" : {ranked} to {mtype}"
        )
        c0 = self._const_int(0)
        pad = self._const_float(0.0, dtype)
        vtype = f"vector<{K}x{NB}x{dtype}>"
        result = self._alloc_ssa(hint or "ldT")
        self._emit(
            f"{result} = vector.transfer_read {view}[{c0}, {c0}], {pad}"
            f" {{in_bounds = [true, true]}} : {mtype}, {vtype}"
        )
        return result, vtype

    def _gen_pack_2d_t(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # pack_2d_t(ptr, row_base, K, NB, M, dtype, col_off=0) -> vector<K x NB>
        #   Like load_2d_t but reads A in ROW-MAJOR order (unit inner stride) then
        #   transposes via a stack-allocated buffer — no extra DDR bandwidth.
        #
        #   load_2d_t strides=[1, M]: inner dim NB has stride M (vlse, strided gather).
        #   pack_2d_t strides=[M, 1]: inner dim K  has stride 1 (vle, contiguous).
        #   Transpose via linalg.generic → local buf → unit-stride read for batch_macc.
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_ssa, ptr_type = self._gen_expr(node.args[0])
        rb_ssa, _ = self._gen_expr(node.args[1])
        K = ast.literal_eval(node.args[2])
        NB = ast.literal_eval(node.args[3])
        m_node = node.args[4]
        dtype = ast.literal_eval(node.args[5]) if len(node.args) > 5 else "f16"
        col_off_node = node.args[6] if len(node.args) > 6 else kwargs.get("col_off")
        space = self._space_of(ptr_type)
        ranked = ptr_type.replace("memref<*x", "memref<?x", 1) if ptr_type.startswith("memref<*x") else ptr_type
        if ptr_type.startswith("memref<*x"):
            rcast = self._alloc_ssa("ranked")
            self._emit(f"{rcast} = memref.cast {ptr_ssa} : {ptr_type} to {ranked}")
            ptr_ssa = rcast
        m_static = isinstance(m_node, ast.Constant) and isinstance(m_node.value, int)
        if m_static:
            M = m_node.value
            m_ssa = self._const_int(M)
            row_stride = str(M)
        else:
            m_ssa, _ = self._gen_expr(m_node)
            row_stride = "?"
        # offset = row_base*M + col_off
        boff = self._alloc_ssa("boff")
        self._emit(f"{boff} = arith.muli {rb_ssa}, {m_ssa} : index")
        if col_off_node is not None:
            co_ssa, _ = self._gen_expr(col_off_node)
            boff2 = self._alloc_ssa("boff")
            self._emit(f"{boff2} = arith.addi {boff}, {co_ssa} : index")
            boff = boff2
        sp = f", {space}" if space else ""
        # Row-major view: NB rows × K cols, strides=[M, 1] (inner dim contiguous)
        row_type = f"memref<{NB}x{K}x{dtype}, strided<[{row_stride}, 1], offset: ?>{sp}>"
        stride_op = str(M) if m_static else m_ssa
        A_view = self._alloc_ssa("Aview")
        self._emit(
            f"{A_view} = memref.reinterpret_cast {ptr_ssa} to "
            f"offset: [{boff}], sizes: [{NB}, {K}], strides: [{stride_op}, 1]"
            f" : {ranked} to {row_type}"
        )
        # Local contiguous buffer K×NB on the stack
        buf_type = f"memref<{K}x{NB}x{dtype}>"
        buf = self._alloc_ssa("buf")
        self._emit(f"{buf} = memref.alloca() {{alignment = 64 : i64}} : {buf_type}")
        # linalg.generic: buf[k,n] = A_view[n,k]  (transpose)
        d0_map = f"affine_map<(d0, d1) -> (d1, d0)>"  # input:  A_view[n=d1, k=d0]
        d1_map = f"affine_map<(d0, d1) -> (d0, d1)>"  # output: buf   [k=d0, n=d1]
        self._emit(
            f'linalg.generic {{'
            f'indexing_maps = [{d0_map}, {d1_map}], '
            f'iterator_types = ["parallel", "parallel"]'
            f'}} ins({A_view} : {row_type}) outs({buf} : {buf_type}) {{'
        )
        self._emit(f'^bb0(%a: {dtype}, %_: {dtype}):')
        self._emit(f'  linalg.yield %a : {dtype}')
        self._emit(f'}}')
        # Unit-stride read from contiguous buf
        c0 = self._const_int(0)
        pad = self._const_float(0.0, dtype)
        vtype = f"vector<{K}x{NB}x{dtype}>"
        result = self._alloc_ssa(hint or "pkT")
        self._emit(
            f"{result} = vector.transfer_read {buf}[{c0}, {c0}], {pad}"
            f" {{in_bounds = [true, true]}} : {buf_type}, {vtype}"
        )
        return result, vtype

    def _gen_alloc_tcm_2d(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # alloc_tcm_2d(K, NB, dtype) -> memref<K×NB×dtype>
        # memref.alloc → MemReftoSpeRT pass converts to spert.alloc thread_local
        # → spine_thread_malloc (TCM), allocated ONCE before K-block loop.
        K = ast.literal_eval(node.args[0])
        NB = ast.literal_eval(node.args[1])
        dtype = ast.literal_eval(node.args[2]) if len(node.args) > 2 else "f16"
        mtype = f"memref<{K}x{NB}x{dtype}>"
        result = self._alloc_ssa(hint or "tcmbuf")
        self._emit(f"{result} = memref.alloc() {{alignment = 64 : i64}} : {mtype}")
        return result, mtype

    def _gen_pack_2d_t_into(self, node: ast.Call):
        # pack_2d_t_into(buf, ptr, row_base, K, NB, M, dtype, col_off=0)
        # Pack A[row_base:row_base+NB, col_off:col_off+K] transposed into existing buf.
        # No alloca — caller owns buf (typically a TCM buffer from alloc_tcm_2d).
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        buf_ssa, buf_type = self._gen_expr(node.args[0])
        ptr_ssa, ptr_type = self._gen_expr(node.args[1])
        rb_ssa, _ = self._gen_expr(node.args[2])
        K = ast.literal_eval(node.args[3])
        NB = ast.literal_eval(node.args[4])
        m_node = node.args[5]
        dtype = ast.literal_eval(node.args[6]) if len(node.args) > 6 else "f16"
        col_off_node = node.args[7] if len(node.args) > 7 else kwargs.get("col_off")
        space = self._space_of(ptr_type)
        ranked = ptr_type.replace("memref<*x", "memref<?x", 1) if ptr_type.startswith("memref<*x") else ptr_type
        if ptr_type.startswith("memref<*x"):
            rcast = self._alloc_ssa("ranked")
            self._emit(f"{rcast} = memref.cast {ptr_ssa} : {ptr_type} to {ranked}")
            ptr_ssa = rcast
        m_static = isinstance(m_node, ast.Constant) and isinstance(m_node.value, int)
        if m_static:
            M = m_node.value
            m_ssa = self._const_int(M)
            row_stride = str(M)
        else:
            m_ssa, _ = self._gen_expr(m_node)
            row_stride = "?"
        boff = self._alloc_ssa("boff")
        self._emit(f"{boff} = arith.muli {rb_ssa}, {m_ssa} : index")
        if col_off_node is not None:
            co_ssa, _ = self._gen_expr(col_off_node)
            boff2 = self._alloc_ssa("boff")
            self._emit(f"{boff2} = arith.addi {boff}, {co_ssa} : index")
            boff = boff2
        sp = f", {space}" if space else ""
        row_type = f"memref<{NB}x{K}x{dtype}, strided<[{row_stride}, 1], offset: ?>{sp}>"
        stride_op = str(M) if m_static else m_ssa
        A_view = self._alloc_ssa("Aview")
        self._emit(
            f"{A_view} = memref.reinterpret_cast {ptr_ssa} to "
            f"offset: [{boff}], sizes: [{NB}, {K}], strides: [{stride_op}, 1]"
            f" : {ranked} to {row_type}"
        )
        buf_type_clean = f"memref<{K}x{NB}x{dtype}>"
        # spestruct.pack (K3 hardware-accelerated pack, lowers to spe_pack_* runtime fn).
        # Expand 2D buf[K×NB] to 4D [1×1×K×NB] required by spestruct.pack.
        # inner_dims_pos=[1,0] + inner_tiles=[K,NB]: reads NB rows of K elements each
        # (unit-stride inner), writes transposed → same spe_pack_f16_m{NB}_n{K}_pi1_po0
        # that mm uses, with full hardware prefetch/scatter support.
        buf_type_clean = f"memref<{K}x{NB}x{dtype}>"
        d0_map = "affine_map<(d0, d1) -> (d1, d0)>"
        d1_map = "affine_map<(d0, d1) -> (d0, d1)>"
        self._emit(
            f'linalg.generic {{'
            f'indexing_maps = [{d0_map}, {d1_map}], '
            f'iterator_types = ["parallel", "parallel"]'
            f'}} ins({A_view} : {row_type}) outs({buf_ssa} : {buf_type_clean}) {{'
        )
        self._emit(f'^bb0(%a: {dtype}, %_: {dtype}):')
        self._emit(f'  linalg.yield %a : {dtype}')
        self._emit(f'}}')

    def _gen_proton_mark(self, node: ast.Call):
        name = ast.literal_eval(node.args[0])
        is_start = len(node.args) >= 2 and ast.literal_eval(node.args[1])
        action = "start" if is_start else "end"
        self._emit(f'proton.record {action} "{name}"')


    def _gen_free_tcm(self, node: ast.Call):
        # free_tcm(buf): memref.dealloc → spine_thread_free (TCM)
        buf_ssa, buf_type = self._gen_expr(node.args[0])
        self._emit(f"memref.dealloc {buf_ssa} : {buf_type}")

    def _gen_store_2d_at(self, node: ast.Call):
        # store_2d_at(ptr, elem_off, rows, cols, vec): write rows×cols block at offset
        ptr_ssa, ptr_type = self._gen_expr(node.args[0])
        off_ssa, _ = self._gen_expr(node.args[1])
        rows = ast.literal_eval(node.args[2])
        cols = ast.literal_eval(node.args[3])
        vec_ssa, vec_type = self._gen_expr(node.args[4])
        m = re.match(r'vector<\d+x\d+x(.+)>', vec_type)
        edtype = m.group(1) if m else "f32"
        space = self._space_of(ptr_type)
        ranked = ptr_type.replace("memref<*x", "memref<?x", 1) if ptr_type.startswith("memref<*x") else ptr_type
        if ptr_type.startswith("memref<*x"):
            rcast = self._alloc_ssa("ranked")
            self._emit(f"{rcast} = memref.cast {ptr_ssa} : {ptr_type} to {ranked}")
            ptr_ssa = rcast
        # output block element offset = elem_off * rows (here rows=1 typically) ... but
        # C is [1,N] row-major: block (1×cols) at element offset = elem_off (the col base).
        # block size = rows*cols; for rows=1 offset = elem_off. Use elem_off*rows generally.
        rows_c = self._const_int(rows)
        boff = self._alloc_ssa("boff")
        self._emit(f"{boff} = arith.muli {off_ssa}, {rows_c} : index")
        sp = f", {space}" if space else ""
        mtype = f"memref<{rows}x{cols}x{edtype}, strided<[{cols}, 1], offset: ?>{sp}>"
        view = self._alloc_ssa("view2d")
        self._emit(
            f"{view} = memref.reinterpret_cast {ptr_ssa} to "
            f"offset: [{boff}], sizes: [{rows}, {cols}], strides: [{cols}, 1]"
            f" : {ranked} to {mtype}"
        )
        c0 = self._const_int(0)
        self._emit(
            f"vector.transfer_write {vec_ssa}, {view}[{c0}, {c0}]"
            f" {{in_bounds = [true, true]}} : {vec_type}, {mtype}"
        )

    def _gen_splat_2d(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # splat_2d(val, rows, cols, dtype) -> vector<rows x cols x dtype>
        val_node = node.args[0]
        rows = ast.literal_eval(node.args[1])
        cols = ast.literal_eval(node.args[2])
        dtype = ast.literal_eval(node.args[3]) if len(node.args) > 3 else "f32"
        val_ssa, _ = self._gen_expr(val_node)
        vtype = f"vector<{rows}x{cols}x{dtype}>"
        result = self._alloc_ssa(hint or "splat2d")
        self._emit(f"{result} = vector.broadcast {val_ssa} : {dtype} to {vtype}")
        return result, vtype

    def _gen_batch_macc(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # batch_macc(lhs_memref<m x k>, rhs_vec<k x n>, acc_vec<m x n>)
        #   -> vector_ext.batch_macc, acc[m,n] = sum_k lhs[m,k]*rhs[k,n]
        # Use generic op form so spine-triton-opt (which doesn't register
        # vector_ext) can still parse the raw_linalg text; spine-opt (k3,
        # registered) lowers it via ExpandBatchMacc → vfwmacc.
        lhs_ssa, lhs_type = self._gen_expr(node.args[0])
        rhs_ssa, rhs_type = self._gen_expr(node.args[1])
        acc_ssa, acc_type = self._gen_expr(node.args[2])
        result = self._alloc_ssa(hint or "bmacc")
        self._emit(
            f'{result} = "vector_ext.batch_macc"({lhs_ssa}, {rhs_ssa}, {acc_ssa})'
            f' : ({lhs_type}, {rhs_type}, {acc_type}) -> {acc_type}'
        )
        return result, acc_type

    def _gen_store_2d(self, node: ast.Call):
        # store_2d(ptr, rows, cols, vec)
        ptr_ssa, ptr_type = self._gen_expr(node.args[0])
        rows = ast.literal_eval(node.args[1])
        cols = ast.literal_eval(node.args[2])
        vec_ssa, vec_type = self._gen_expr(node.args[3])
        # element dtype from vector<rows x cols x dtype>
        m = re.match(r'vector<\d+x\d+x(.+)>', vec_type)
        edtype = m.group(1) if m else "f32"
        space = self._space_of(ptr_type)
        ranked = ptr_type.replace("memref<*x", "memref<?x", 1) if ptr_type.startswith("memref<*x") else ptr_type
        if ptr_type.startswith("memref<*x"):
            rcast = self._alloc_ssa("ranked")
            self._emit(f"{rcast} = memref.cast {ptr_ssa} : {ptr_type} to {ranked}")
            ptr_ssa = rcast
        mtype = self._strided_2d_type(rows, cols, edtype, space)
        view = self._alloc_ssa("view2d")
        self._emit(
            f"{view} = memref.reinterpret_cast {ptr_ssa} to "
            f"offset: [0], sizes: [{rows}, {cols}], strides: [{cols}, 1]"
            f" : {ranked} to {mtype}"
        )
        c0 = self._const_int(0)
        self._emit(
            f"vector.transfer_write {vec_ssa}, {view}[{c0}, {c0}]"
            f" {{in_bounds = [true, true]}} : {vec_type}, {mtype}"
        )

    def _gen_store_vec(self, node: ast.Call):
        args = node.args
        ptr_ssa, ptr_type = self._gen_expr(args[0])
        idx_ssa, _        = self._gen_expr(args[1])
        vec_ssa, vec_type = self._gen_expr(args[2])

        # vector.store requires ranked memref; cast if unranked
        store_ptr_ssa = ptr_ssa
        store_ptr_type = ptr_type
        if ptr_type.startswith("memref<*x") or ptr_type == "memref<32xf32>":
            # For output memref<32xf32>, keep as-is (already ranked)
            # For memref<*x...>, cast to ranked
            if ptr_type.startswith("memref<*x"):
                ranked_type = ptr_type.replace("memref<*x", "memref<?x", 1)
                cast_ssa = self._alloc_ssa("ranked")
                self._emit(f"{cast_ssa} = memref.cast {ptr_ssa} : {ptr_type} to {ranked_type}")
                store_ptr_ssa = cast_ssa
                store_ptr_type = ranked_type

        self._emit(
            f"vector.transfer_write {vec_ssa}, {store_ptr_ssa}[{idx_ssa}]"
            f" {{in_bounds = [true]}} : {vec_type}, {store_ptr_type}"
        )

    def _gen_store_scalar(self, node: ast.Call):
        args = node.args
        ptr_ssa, ptr_type = self._gen_expr(args[0])
        idx_ssa, _        = self._gen_expr(args[1])
        val_ssa, val_type = self._gen_expr(args[2])

        # memref.store requires ranked memref; cast if unranked
        store_ptr_ssa = ptr_ssa
        store_ptr_type = ptr_type
        if ptr_type.startswith("memref<*x"):
            ranked_type = ptr_type.replace("memref<*x", "memref<?x", 1)
            cast_ssa = self._alloc_ssa("ranked")
            self._emit(f"{cast_ssa} = memref.cast {ptr_ssa} : {ptr_type} to {ranked_type}")
            store_ptr_ssa = cast_ssa
            store_ptr_type = ranked_type

        self._emit(
            f"memref.store {val_ssa}, {store_ptr_ssa}[{idx_ssa}]"
            f" : {store_ptr_type}"
        )

    # ------------------------------------------------------------------
    # svector-level ops (feishu 3.3 mv 示例). Fixed VL, no dynamic vsetvl.
    # ------------------------------------------------------------------
    def _gen_vconfig_assign(self, target: str, node: ast.Call):
        # vconfig(avl, sew_bytes) → fixed compile-time VL. avl (dynamic tail
        # length) is ignored this round: we run full VL tiles only.
        sew = ast.literal_eval(node.args[1]) if len(node.args) > 1 else 2
        vl = _vl_from_sew(int(sew))
        self._constexpr_ints[target] = vl
        self._active_vl = vl

    def _require_vl(self) -> int:
        if self._active_vl is None:
            raise ValueError(
                "spine_raw svector op used before tle.vconfig(...) set the VL")
        return self._active_vl

    def _gen_vzero(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # vzero(dtype) → vector<VL x dtype> of zeros
        dtype = _resolve_dtype(node.args[0] if node.args else None, "f32")
        vl = self._require_vl()
        vec_type = f"vector<{vl}x{dtype}>"
        zero = self._const_float(0.0, dtype)
        result = self._alloc_ssa(hint or "vzero")
        self._emit(f"{result} = vector.broadcast {zero} : {dtype} to {vec_type}")
        return result, vec_type

    def _gen_vload(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # vload(ptr, idx_tuple[, stride], dtype=?) → vector<VL x dtype>
        #   1D external idx (ki,)      : transfer_read ptr[ki]
        #   2D external idx (ni, ki)   : flat offset = ni*stride + ki (row-major
        #                                B is N×K, stride = K passed explicitly)
        #   ND ranked idx (packed_B)   : transfer_read reads the innermost dim
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_node = node.args[0]
        idx_node = node.args[1]
        assert isinstance(idx_node, ast.Tuple), "vload index must be a tuple"
        idx_elts = idx_node.elts
        stride_node = node.args[2] if len(node.args) > 2 else kwargs.get("stride")
        dtype = _resolve_dtype(kwargs.get("dtype"), "f16")
        vl = self._require_vl()
        vec_type = f"vector<{vl}x{dtype}>"
        pad = self._const_float(0.0, dtype)

        ptr_ssa, ptr_type = self._gen_expr(ptr_node)

        if not ptr_type.startswith("memref<*x"):
            # Ranked memref (e.g. local packed_B): index every dim directly,
            # transfer_read pulls the innermost VL-length slice. in_bounds has
            # one entry per *vector* dim (rank 1 here), not per index element.
            idx_ssas = [self._gen_expr(e)[0] for e in idx_elts]
            result = self._alloc_ssa(hint or "vld")
            self._emit(
                f"{result} = vector.transfer_read {ptr_ssa}[{', '.join(idx_ssas)}], {pad}"
                f" {{in_bounds = [true]}} : {ptr_type}, {vec_type}"
            )
            return result, vec_type

        # External unranked pointer: cast to ranked 1D, compute a flat offset.
        ranked_ssa, ranked_type = self._ranked_cast(ptr_ssa, ptr_type)
        if len(idx_elts) == 1:
            off_ssa, _ = self._gen_expr(idx_elts[0])
        elif len(idx_elts) == 2:
            assert stride_node is not None, \
                "vload of a 2D index into an external pointer needs a row stride"
            r_ssa, _ = self._gen_expr(idx_elts[0])
            s_ssa, _ = self._gen_expr(stride_node)
            c_ssa, _ = self._gen_expr(idx_elts[1])
            mul = self._alloc_ssa("roff")
            self._emit(f"{mul} = arith.muli {r_ssa}, {s_ssa} : index")
            off_ssa = self._alloc_ssa("off")
            self._emit(f"{off_ssa} = arith.addi {mul}, {c_ssa} : index")
        else:
            raise NotImplementedError(
                f"vload with {len(idx_elts)}-D index into external pointer unsupported")
        result = self._alloc_ssa(hint or "vld")
        self._emit(
            f"{result} = vector.transfer_read {ranked_ssa}[{off_ssa}], {pad}"
            f" {{in_bounds = [true]}} : {ranked_type}, {vec_type}"
        )
        return result, vec_type

    def _gen_vmacc(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # vmacc(acc, x, y) → acc + extf(x) * extf(y)  (widening f16 → f32 fma)
        acc_ssa, acc_type = self._gen_expr(node.args[0])
        x_ssa, x_type = self._gen_expr(node.args[1])
        y_ssa, y_type = self._gen_expr(node.args[2])
        acc_elem = _vec_elem(acc_type)
        n = _vec_n(acc_type)
        wide_type = f"vector<{n}x{acc_elem}>"
        # Widen operands to the accumulator element type if needed.
        if x_type != wide_type:
            xw = self._alloc_ssa("xw")
            self._emit(f"{xw} = arith.extf {x_ssa} : {x_type} to {wide_type}")
            x_ssa = xw
        if y_type != wide_type:
            yw = self._alloc_ssa("yw")
            self._emit(f"{yw} = arith.extf {y_ssa} : {y_type} to {wide_type}")
            y_ssa = yw
        result = self._alloc_ssa(hint or "vmacc")
        self._emit(f"{result} = math.fma {x_ssa}, {y_ssa}, {acc_ssa} : {acc_type}")
        return result, acc_type

    def _gen_vreduce_sum(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # vreduce_sum(vec) → scalar horizontal add
        v_ssa, v_type = self._gen_expr(node.args[0])
        elem = _vec_elem(v_type)
        result = self._alloc_ssa(hint or "vrsum")
        self._emit(f"{result} = vector.reduction <add>, {v_ssa} : {v_type} into {elem}")
        return result, elem

    def _gen_vstore(self, node: ast.Call):
        # vstore(ptr, idx_tuple, scalar) → memref.store  (1D scalar output)
        ptr_ssa, ptr_type = self._gen_expr(node.args[0])
        idx_node = node.args[1]
        assert isinstance(idx_node, ast.Tuple), "vstore index must be a tuple"
        assert len(idx_node.elts) == 1, "vstore currently supports a 1D scalar index"
        idx_ssa, _ = self._gen_expr(idx_node.elts[0])
        val_ssa, _ = self._gen_expr(node.args[2])
        store_ssa, store_type = self._ranked_cast(ptr_ssa, ptr_type)
        self._emit(f"memref.store {val_ssa}, {store_ssa}[{idx_ssa}] : {store_type}")

    def _try_const_int(self, node) -> int | None:
        """Fold a shape/index AST node to a compile-time int if possible.

        Handles int literals, constexpr ints (nvl from vconfig), and +/-/*//
        of those. Returns None when any leaf is a runtime value (e.g. K)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in self._constexpr_ints:
            return self._constexpr_ints[node.id]
        if isinstance(node, ast.BinOp):
            l = self._try_const_int(node.left)
            r = self._try_const_int(node.right)
            if l is None or r is None:
                return None
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.FloorDiv):
                return l // r
        return None

    def _gen_alloc(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # alloc(shape_tuple, dtype) → memref.alloc (写法3 packed_B scratch).
        # Static int dims stay literal; expression dims (e.g. K // nvl with K a
        # runtime arg) become '?' with a dynamic size operand.
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        shape_node = node.args[0]
        assert isinstance(shape_node, ast.Tuple), "alloc shape must be a tuple"
        dt_node = node.args[1] if len(node.args) > 1 else kwargs.get("dtype")
        dtype = _resolve_dtype(dt_node, "f16")

        dims: list[str] = []
        dyn_ssas: list[str] = []
        for e in shape_node.elts:
            cval = self._try_const_int(e)
            if cval is not None:
                dims.append(str(cval))
            else:
                ssa, _ = self._gen_expr(e)
                dims.append("?")
                dyn_ssas.append(ssa)
        mtype = f"memref<{'x'.join(dims)}x{dtype}>"
        result = self._alloc_ssa(hint or "packed")
        operands = ", ".join(dyn_ssas)
        self._emit(
            f"{result} = memref.alloc({operands}) {{alignment = 64 : i64}} : {mtype}")
        return result, mtype

    def _gen_vpack(self, node: ast.Call):
        # vpack(src, (row0, col0), dst, dst_shape, stride):
        #   pack src's ROWS-row block starting at row0 into dst laid out as
        #   (1, K//nvl, ROWS, nvl), so dst[0, kb, r, :] = src[row0+r, kb*nvl:+nvl].
        #   src is an external row-major pointer with row stride = `stride`
        #   (== K here); col0 is assumed 0 (full rows). dst is a ranked memref
        #   from tle.alloc. The kb loop runs over the runtime column extent.
        src_node, src_idx, dst_node, dst_shape, stride_node = node.args[:5]
        assert isinstance(src_idx, ast.Tuple) and len(src_idx.elts) == 2, \
            "vpack src index must be a (row, col) tuple"
        assert isinstance(dst_shape, ast.Tuple) and len(dst_shape.elts) == 4, \
            "vpack dst_shape must be 4-D (1, K//nvl, ROWS, nvl)"

        vl = self._require_vl()
        rows = self._try_const_int(dst_shape.elts[2])
        assert rows is not None, \
            "vpack ROWS (dst_shape[2]) must be a compile-time constant"

        dst_ssa, dst_type = self._gen_expr(dst_node)
        dtype = _memref_elem(dst_type)
        vec_type = f"vector<{vl}x{dtype}>"

        src_ssa, src_type = self._gen_expr(src_node)
        ranked_ssa, ranked_type = self._ranked_cast(src_ssa, src_type)
        row0_ssa, _ = self._gen_expr(src_idx.elts[0])
        stride_ssa, _ = self._gen_expr(stride_node)

        pad = self._const_float(0.0, dtype)
        c0 = self._const_int(0)
        cvl = self._const_int(vl)

        loop_ssa = self._alloc_ssa("pk")
        self._emit(f"scf.for {loop_ssa} = {c0} to {stride_ssa} step {cvl} {{")
        self._indent += 2
        kb_ssa = self._alloc_ssa("kb")
        self._emit(f"{kb_ssa} = arith.divui {loop_ssa}, {cvl} : index")
        for r in range(rows):
            if r == 0:
                nir = row0_ssa
            else:
                cr = self._const_int(r)
                nir = self._alloc_ssa("nir")
                self._emit(f"{nir} = arith.addi {row0_ssa}, {cr} : index")
            roff = self._alloc_ssa("roff")
            self._emit(f"{roff} = arith.muli {nir}, {stride_ssa} : index")
            off = self._alloc_ssa("off")
            self._emit(f"{off} = arith.addi {roff}, {loop_ssa} : index")
            vec = self._alloc_ssa("pkv")
            self._emit(
                f"{vec} = vector.transfer_read {ranked_ssa}[{off}], {pad}"
                f" {{in_bounds = [true]}} : {ranked_type}, {vec_type}")
            cr_idx = self._const_int(r)
            self._emit(
                f"vector.transfer_write {vec}, {dst_ssa}[{c0}, {kb_ssa}, {cr_idx}, {c0}]"
                f" {{in_bounds = [true]}} : {vec_type}, {dst_type}")
        self._indent -= 2
        self._emit("}")

