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
            raise ValueError(f"Parameter '{pname}' of @spine_raw function '{fn.__name__}' "
                             f"must have an In[...] or InOut[...] annotation.")
        if not isinstance(ann, _TypedAnnotation):
            raise ValueError(f"Parameter '{pname}' annotation must be In[...] or InOut[...], got {ann!r}")
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


_SPINE_RAW_BUILTIN_NAMES = {
    "range", "proton_mark", "vconfig", "vzero", "vload", "vmacc", "vreduce_sum", "vstore", "alloc", "pack", "vfwmadot",
    "vpack", "mmt4d", "vmin", "vmax", "sqrt", "rsqrt", "abs", "cast", "select"
}

# Element-type classification for §6.4 elementwise dispatch.
_FLOAT_ELEMS = {"f16", "f32", "bf16", "f64"}
_ELEM_BITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "f16": 16, "bf16": 16, "f32": 32, "f64": 64}


def _is_float_elem(elem: str) -> bool:
    return elem in _FLOAT_ELEMS


def _elem_bits(elem: str) -> int:
    return _ELEM_BITS[elem]


# §6.4 binary operators → (float arith op, int arith op). None = not defined for
# that domain (e.g. bitwise on floats, true division on ints).
_BINOP_ARITH = {
    ast.Add: ("addf", "addi"),
    ast.Sub: ("subf", "subi"),
    ast.Mult: ("mulf", "muli"),
    ast.Div: ("divf", None),  # a / b  真除(浮点) → vfdiv
    ast.FloorDiv: (None, "divsi"),  # a // b 整数向下取整除 → vdiv
    ast.Mod: ("remf", "remsi"),  # a % b  取余 → vrem
    ast.BitAnd: (None, "andi"),
    ast.BitOr: (None, "ori"),
    ast.BitXor: (None, "xori"),
    ast.LShift: (None, "shli"),
    ast.RShift: (None, "shrsi"),
}

# §6.4 comparisons → (arith.cmpf predicate, arith.cmpi predicate). Signed int.
_CMP_PRED = {
    ast.Lt: ("olt", "slt"),
    ast.LtE: ("ole", "sle"),
    ast.Gt: ("ogt", "sgt"),
    ast.GtE: ("oge", "sge"),
    ast.Eq: ("oeq", "eq"),
    ast.NotEq: ("one", "ne"),
}


# RVV vector config (SPEC §6.1). VLEN is the physical scalable-register width;
# SEW is the element width derived from the dtype (SPEC §3.1), not a vconfig
# parameter. The svector eDSL's element granularity is f16 (all svector kernels
# load f16 and count VL in f16 elements); f32 accumulators are the same element
# count at a wider LMUL group. VLMAX = lmul * VLEN / SEW.
_VLEN_BITS = 1024  # K3 scalable register width
_BASE_SEW_BITS = 16  # f16 element width (SPEC §3.1); the svector loop's VL granularity


def _vlmax(lmul: int, sew_bits: int = _BASE_SEW_BITS) -> int:
    """VLMAX (element count) for K3 at the given LMUL, per SPEC §6.1.

    VLMAX = lmul * VLEN / SEW. With VLEN=1024 and SEW=16 (f16):
      lmul=1 -> 64, lmul=2 -> 128, lmul=4 -> 256, lmul=8 -> 512.
    """
    return lmul * _VLEN_BITS // sew_bits


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
        self._preamble: list[str] = []  # constant defs, always at indent=2
        self._lines: list[str] = []  # body ops
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

    def _const_int_typed(self, n: int, itype: str) -> str:
        """Integer constant in a specific integer element type (e.g. i8/i32),
        as opposed to _const_int which always emits `index`."""
        key = (n, itype)
        if key not in self._const_ints:
            ssa = self._alloc_ssa(f"c{abs(n)}{'_neg' if n < 0 else ''}_{itype}")
            self._const_ints[key] = ssa
            self._preamble.append(f"  {ssa} = arith.constant {n} : {itype}")
        return self._const_ints[key]

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

        # Resolve closure/global int free-vars as compile-time consts (so shapes
        # baked into a kernel via closure — e.g. tle.mmt4d(B, A, C, N, K, 32) with
        # N/K captured — fold to literals for _try_const_int).
        freevars: dict[str, object] = {}
        if getattr(fn, "__closure__", None):
            names = fn.__code__.co_freevars
            for nm, cell in zip(names, fn.__closure__):
                try:
                    freevars[nm] = cell.cell_contents
                except ValueError:
                    pass
        for nm, val in {**(fn.__globals__ or {}), **freevars}.items():
            if isinstance(val, int) and not isinstance(val, bool):
                self._constexpr_ints.setdefault(nm, val)

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

        # 写法4 pattern:body 里出现 tle.vfwmadot(矩阵单元 mv)→ 整段折成结构化
        # linalg.pack+mmt4d+unpack(cube 布局交下游 spe_pack;raw 逐 cube vfwmadot 在
        # 当前 build 数值不对,唯结构化路正确)。dims 从闭包常量 N(输出行)/K 取。
        if not (self._body_has_vfwmadot(node.body) and self._emit_mmt4d_from_pattern(params)):
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

        # vconfig(avl, lmul) → compile-time VL constant, tracked separately from
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
            for_line = (f"{res_part} = scf.for {loop_ssa} = {lb_ssa} to {ub_ssa} step {step_ssa}"
                        f" iter_args({ia_part}) -> ({types_part}) {{")
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
            self._emit(f"scf.yield {', '.join(yield_ssas)} : {', '.join(yield_types)}")

        self._indent -= 2
        self._emit("}")

        self._loop_iter_args = prev_loop_iter_args

        # After loop: rebind iter_arg names to result SSAs
        for v, _, typ in ia_data:
            self._bind(v, result_ssas[v], typ)

    def _gen_call_stmt(self, node: ast.Call):
        if _is_spine_raw_attr(node.func, "proton_mark", self._aliases):
            self._gen_proton_mark(node)
        elif _is_spine_raw_attr(node.func, "vstore", self._aliases):
            self._gen_vstore(node)
        elif _is_spine_raw_attr(node.func, "pack", self._aliases):
            self._gen_pack(node)
        elif _is_spine_raw_attr(node.func, "mmt4d", self._aliases):
            self._gen_mmt4d(node)
        else:
            raise NotImplementedError(f"Unsupported call statement: {ast.dump(node.func)}")

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
        if isinstance(node, ast.UnaryOp):
            return self._gen_unaryop(node, hint)
        if isinstance(node, ast.Compare):
            return self._gen_compare(node, hint)
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

    def _broadcast_to(self, ssa: str, typ: str, vec_type: str) -> str:
        """Broadcast a scalar (or narrower value) to vec_type via vector.broadcast.
        Returns the vector SSA. Assumes typ is the scalar element type of vec_type."""
        out = self._alloc_ssa("bcast")
        self._emit(f"{out} = vector.broadcast {ssa} : {typ} to {vec_type}")
        return out

    def _match_operands(self, lssa, ltype, rssa, rtype):
        """§6.4: bring a (vec, scalar) or (scalar, vec) pair to a common vector
        type by broadcasting the scalar side. Returns (lssa, rssa, vec_type)."""
        l_is_vec = ltype.startswith("vector<")
        r_is_vec = rtype.startswith("vector<")
        if l_is_vec and r_is_vec:
            if ltype != rtype:
                raise NotImplementedError(f"elementwise between mismatched vectors {ltype} / {rtype}")
            return lssa, rssa, ltype
        if l_is_vec and not r_is_vec:
            return lssa, self._broadcast_to(rssa, _vec_elem(ltype), ltype), ltype
        if r_is_vec and not l_is_vec:
            return self._broadcast_to(lssa, _vec_elem(rtype), rtype), rssa, rtype
        return lssa, rssa, None  # both scalar

    def _gen_binop(self, node: ast.BinOp, hint: str) -> tuple[str, str]:
        lssa, ltype = self._gen_expr(node.left)
        rssa, rtype = self._gen_expr(node.right)
        op = type(node.op)
        result = self._alloc_ssa(hint or "t")

        # Index arithmetic fast-path (loop bounds / flat offsets, compile-time-ish).
        if ltype == "index" and rtype == "index":
            opname = {ast.Add: "addi", ast.Mult: "muli", ast.Sub: "subi", ast.FloorDiv: "divui"}.get(op)
            if opname is None:
                raise NotImplementedError(f"BinOp {op.__name__} not supported for index")
            self._emit(f"{result} = arith.{opname} {lssa}, {rssa} : index")
            return result, "index"

        # §6.4 elementwise: broadcast the scalar side, then dispatch by float/int.
        lssa, rssa, vec_type = self._match_operands(lssa, ltype, rssa, rtype)
        if vec_type is None:
            raise NotImplementedError(f"BinOp between {ltype!r} and {rtype!r} not supported")
        elem = _vec_elem(vec_type)
        is_float = _is_float_elem(elem)
        arith = _BINOP_ARITH.get(op)
        if arith is None:
            raise NotImplementedError(f"§6.4 operator {op.__name__} not supported")
        opname = arith[0] if is_float else arith[1]
        if opname is None:
            domain = "float" if is_float else "integer"
            raise NotImplementedError(f"§6.4 operator {op.__name__} not defined for {domain} elements ({elem})")
        self._emit(f"{result} = arith.{opname} {lssa}, {rssa} : {vec_type}")
        return result, vec_type

    def _gen_unaryop(self, node: ast.UnaryOp, hint: str) -> tuple[str, str]:
        # §6.4: -a (negate) and ~a (bitwise not).
        vssa, vtype = self._gen_expr(node.operand)
        result = self._alloc_ssa(hint or "u")
        if not vtype.startswith("vector<"):
            raise NotImplementedError(f"unary {type(node.op).__name__} on non-vector {vtype}")
        elem = _vec_elem(vtype)
        if isinstance(node.op, ast.USub):
            if _is_float_elem(elem):
                self._emit(f"{result} = arith.negf {vssa} : {vtype}")
            else:
                zero = self._broadcast_to(self._const_int_typed(0, elem), elem, vtype)
                self._emit(f"{result} = arith.subi {zero}, {vssa} : {vtype}")
            return result, vtype
        if isinstance(node.op, ast.Invert):  # ~a  = xor -1 (integers only)
            if _is_float_elem(elem):
                raise NotImplementedError(f"§6.4 ~a not defined for float elements ({elem})")
            ones = self._broadcast_to(self._const_int_typed(-1, elem), elem, vtype)
            self._emit(f"{result} = arith.xori {vssa}, {ones} : {vtype}")
            return result, vtype
        raise NotImplementedError(f"§6.4 unary op {type(node.op).__name__} not supported")

    def _gen_compare(self, node: ast.Compare, hint: str) -> tuple[str, str]:
        # §6.4: a <cmp> b → mask vector<Nxi1> (§6.8). Single comparison only.
        if len(node.ops) != 1:
            raise NotImplementedError("chained comparison not supported (write as separate compares)")
        lssa, ltype = self._gen_expr(node.left)
        rssa, rtype = self._gen_expr(node.comparators[0])
        op = type(node.ops[0])
        lssa, rssa, vec_type = self._match_operands(lssa, ltype, rssa, rtype)
        if vec_type is None:
            raise NotImplementedError(f"comparison between {ltype!r} and {rtype!r} not supported")
        pred = _CMP_PRED.get(op)
        if pred is None:
            raise NotImplementedError(f"§6.4 comparison {op.__name__} not supported")
        elem = _vec_elem(vec_type)
        cmp_op, predicate = ("cmpf", pred[0]) if _is_float_elem(elem) else ("cmpi", pred[1])
        n = _vec_n(vec_type)
        mask_type = f"vector<{n}xi1>"
        result = self._alloc_ssa(hint or "cmp")
        self._emit(f"{result} = arith.{cmp_op} {predicate}, {lssa}, {rssa} : {vec_type}")
        return result, mask_type

    def _gen_call_expr(self, node: ast.Call, hint: str) -> tuple[str, str]:
        if _is_spine_raw_attr(node.func, "vzero", self._aliases):
            return self._gen_vzero(node, hint)
        if _is_spine_raw_attr(node.func, "vload", self._aliases):
            return self._gen_vload(node, hint)
        if _is_spine_raw_attr(node.func, "vmacc", self._aliases):
            return self._gen_vmacc(node, hint)
        if _is_spine_raw_attr(node.func, "vreduce_sum", self._aliases):
            return self._gen_vreduce_sum(node, hint)
        if _is_spine_raw_attr(node.func, "vfwmadot", self._aliases):
            return self._gen_vfwmadot(node, hint)
        if _is_spine_raw_attr(node.func, "vpack", self._aliases):
            return self._gen_vpack(node, hint)
        if _is_spine_raw_attr(node.func, "alloc", self._aliases):
            return self._gen_alloc(node, hint)
        for _nm in ("vmin", "vmax"):
            if _is_spine_raw_attr(node.func, _nm, self._aliases):
                return self._gen_vminmax(node, hint, _nm)
        if _is_spine_raw_attr(node.func, "sqrt", self._aliases):
            return self._gen_unary_math(node, hint, "sqrt")
        if _is_spine_raw_attr(node.func, "rsqrt", self._aliases):
            return self._gen_unary_math(node, hint, "rsqrt")
        if _is_spine_raw_attr(node.func, "abs", self._aliases):
            return self._gen_abs(node, hint)
        if _is_spine_raw_attr(node.func, "cast", self._aliases):
            return self._gen_cast(node, hint)
        if _is_spine_raw_attr(node.func, "select", self._aliases):
            return self._gen_select(node, hint)
        raise NotImplementedError(f"Unsupported call: {ast.dump(node.func)}")

    # ------------------------------------------------------------------
    # spine_raw builtin implementations
    # ------------------------------------------------------------------

    def _ranked_cast(self, ptr_ssa: str, ptr_type: str) -> tuple[str, str]:
        """Cast memref<*xT> to memref<?xT>; return (ssa, type)."""
        if ptr_type.startswith("memref<*x"):
            ranked_type = ptr_type.replace("memref<*x", "memref<?x", 1)
            cast_ssa = self._alloc_ssa("ranked")
            self._emit(f"{cast_ssa} = memref.cast {ptr_ssa} : {ptr_type} to {ranked_type}")
            return cast_ssa, ranked_type
        return ptr_ssa, ptr_type

    # ------------------------------------------------------------------
    # profiling
    # ------------------------------------------------------------------

    def _gen_proton_mark(self, node: ast.Call):
        name = ast.literal_eval(node.args[0])
        is_start = len(node.args) >= 2 and ast.literal_eval(node.args[1])
        action = "start" if is_start else "end"
        self._emit(f'proton.record {action} "{name}"')

    def _gen_vconfig_assign(self, target: str, node: ast.Call):
        # SPEC §6.1: vconfig(avl, lmul) -> VL = min(avl, VLMAX),
        # VLMAX = lmul * VLEN / SEW. SEW is derived from the dtype (§3.1), not a
        # parameter; the svector loop counts VL in f16 elements (base SEW=16).
        # avl (runtime tail narrowing / true strip-mine) is deferred (SPEC §6.1),
        # so VL is the fixed VLMAX for the requested LMUL this round.
        lmul = ast.literal_eval(node.args[1]) if len(node.args) > 1 else 1
        vl = _vlmax(int(lmul))
        self._constexpr_ints[target] = vl
        self._active_vl = vl

    def _require_vl(self) -> int:
        if self._active_vl is None:
            raise ValueError("spine_raw svector op used before tle.vconfig(...) set the VL")
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
        # SPEC §6.2:vload(ptr, index, stride=None, idx=None) -> vec
        #   index — 起始元素偏移(基址),扁平标量;二维坐标由用户自行压平(如 ni*K + ki)。
        #   stride — 逐元素间隔 → vlse(待扩);idx — 索引向量 → vluxei/gather(待扩)。
        #   不带 stride/idx → 连续访存 vle(transfer_read)。
        #   例外:对 alloc 出的 ranked memref(如 packed_B),index 为多维元组 = 逐维下标,
        #   transfer_read 读最内维 VL 长切片(这是 ranked scratch 的自然寻址,非 2D 压平)。
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_node = node.args[0]
        idx_node = node.args[1]
        stride_node = node.args[2] if len(node.args) > 2 else kwargs.get("stride")
        idx_vec_node = kwargs.get("idx")
        dtype = _resolve_dtype(kwargs.get("dtype"), "f16")
        vl = self._require_vl()
        vec_type = f"vector<{vl}x{dtype}>"
        pad = self._const_float(0.0, dtype)

        ptr_ssa, ptr_type = self._gen_expr(ptr_node)

        if not ptr_type.startswith("memref<*x"):
            # Ranked memref (e.g. local packed_B): index every dim directly,
            # transfer_read pulls the innermost VL-length slice. in_bounds has
            # one entry per *vector* dim (rank 1 here), not per index element.
            assert isinstance(idx_node, ast.Tuple), \
                "vload of a ranked memref (alloc) needs a per-dim index tuple"
            idx_ssas = [self._gen_expr(e)[0] for e in idx_node.elts]
            result = self._alloc_ssa(hint or "vld")
            self._emit(f"{result} = vector.transfer_read {ptr_ssa}[{', '.join(idx_ssas)}], {pad}"
                       f" {{in_bounds = [true]}} : {ptr_type}, {vec_type}")
            return result, vec_type

        # External unranked pointer (SPEC canonical): index = 扁平标量元素偏移。
        if idx_vec_node is not None:
            raise NotImplementedError("vload idx (gather → vluxei) 未实现(SPEC §6.2 待扩)")
        if stride_node is not None:
            raise NotImplementedError("vload stride (跨步 → vlse) 未实现(SPEC §6.2 待扩)")
        assert not isinstance(idx_node, ast.Tuple), ("vload 的 index 须为扁平标量元素偏移(二维坐标请自行压平,如 ni*K + ki);"
                                                     "SPEC §6.2")
        ranked_ssa, ranked_type = self._ranked_cast(ptr_ssa, ptr_type)
        off_ssa, _ = self._gen_expr(idx_node)
        result = self._alloc_ssa(hint or "vld")
        self._emit(f"{result} = vector.transfer_read {ranked_ssa}[{off_ssa}], {pad}"
                   f" {{in_bounds = [true]}} : {ranked_type}, {vec_type}")
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

    # ---- §6.4 named elementwise functions -----------------------------------

    def _gen_vminmax(self, node: ast.Call, hint: str, which: str) -> tuple[str, str]:
        # vmin/vmax(a, b) → 逐元素 min/max. Float: arith.minimumf/maximumf;
        # int: arith.minsi/maxsi (signed). b may be a broadcast scalar (§6.4).
        lssa, ltype = self._gen_expr(node.args[0])
        rssa, rtype = self._gen_expr(node.args[1])
        lssa, rssa, vec_type = self._match_operands(lssa, ltype, rssa, rtype)
        if vec_type is None:
            raise NotImplementedError(f"{which}: needs at least one vector operand")
        elem = _vec_elem(vec_type)
        if _is_float_elem(elem):
            opname = "minimumf" if which == "vmin" else "maximumf"
        else:
            opname = "minsi" if which == "vmin" else "maxsi"
        result = self._alloc_ssa(hint or which)
        self._emit(f"{result} = arith.{opname} {lssa}, {rssa} : {vec_type}")
        return result, vec_type

    def _gen_unary_math(self, node: ast.Call, hint: str, which: str) -> tuple[str, str]:
        # sqrt(a) → math.sqrt; rsqrt(a) → math.rsqrt. Float only.
        vssa, vtype = self._gen_expr(node.args[0])
        if not vtype.startswith("vector<") or not _is_float_elem(_vec_elem(vtype)):
            raise NotImplementedError(f"{which}: float vector operand required, got {vtype}")
        result = self._alloc_ssa(hint or which)
        self._emit(f"{result} = math.{which} {vssa} : {vtype}")
        return result, vtype

    def _gen_abs(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # abs(a) → |a|. Float: math.absf; int: math.absi.
        vssa, vtype = self._gen_expr(node.args[0])
        if not vtype.startswith("vector<"):
            raise NotImplementedError(f"abs: vector operand required, got {vtype}")
        opname = "absf" if _is_float_elem(_vec_elem(vtype)) else "absi"
        result = self._alloc_ssa(hint or "abs")
        self._emit(f"{result} = math.{opname} {vssa} : {vtype}")
        return result, vtype

    def _gen_cast(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # cast(a, dtype) → 类型转换. Same element count, new element type.
        #   f16→f32 widening = arith.extf (直达); float→float narrowing = truncf;
        #   int↔float = sitofp/fptosi; int width change = extsi/trunci.
        vssa, vtype = self._gen_expr(node.args[0])
        if not vtype.startswith("vector<"):
            raise NotImplementedError(f"cast: vector operand required, got {vtype}")
        src_elem = _vec_elem(vtype)
        dst_elem = _resolve_dtype(node.args[1], src_elem)
        if dst_elem == src_elem:
            return vssa, vtype
        n = _vec_n(vtype)
        dst_type = f"vector<{n}x{dst_elem}>"
        src_f, dst_f = _is_float_elem(src_elem), _is_float_elem(dst_elem)
        if src_f and dst_f:
            op = "extf" if _elem_bits(dst_elem) > _elem_bits(src_elem) else "truncf"
        elif src_f and not dst_f:
            op = "fptosi"
        elif not src_f and dst_f:
            op = "sitofp"
        else:  # int → int
            op = "extsi" if _elem_bits(dst_elem) > _elem_bits(src_elem) else "trunci"
        result = self._alloc_ssa(hint or "cast")
        self._emit(f"{result} = arith.{op} {vssa} : {vtype} to {dst_type}")
        return result, dst_type

    def _gen_select(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # select(m, a, b) → a if m else b, per lane. §6.8: falls back to
        # arith.select (native vmerge deferred). m is a vector<Nxi1> mask.
        mssa, mtype = self._gen_expr(node.args[0])
        assa, atype = self._gen_expr(node.args[1])
        bssa, btype = self._gen_expr(node.args[2])
        # Broadcast scalar a/b against the other vector operand if needed.
        assa, bssa, vec_type = self._match_operands(assa, atype, bssa, btype)
        if vec_type is None:
            raise NotImplementedError("select: a/b must include at least one vector")
        if not mtype.startswith("vector<"):
            raise NotImplementedError(f"select: mask must be a vector<Nxi1>, got {mtype}")
        result = self._alloc_ssa(hint or "sel")
        self._emit(f"{result} = arith.select {mssa}, {assa}, {bssa} : {mtype}, {vec_type}")
        return result, vec_type

    def _gen_vreduce_sum(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # vreduce_sum(vec) → scalar horizontal add
        v_ssa, v_type = self._gen_expr(node.args[0])
        elem = _vec_elem(v_type)
        result = self._alloc_ssa(hint or "vrsum")
        self._emit(f"{result} = vector.reduction <add>, {v_ssa} : {v_type} into {elem}")
        return result, elem

    def _gen_vfwmadot(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # vfwmadot(acc, x, y) → "vector_ext.matmul"(x, y, acc) <{m,n,k}> (写法4, 矩阵单元).
        #   矩阵引擎直接产出宽结果 (不需 vreduce_sum);K3 spine-opt 注册了
        #   vector_ext::MatmulOp + ConvertOpToLLVMPattern,lower 到
        #   llvm.riscv.smt.vfwmadot (需 xsmtvdotii mattr, 已在 compiler.py 配)。
        #   用 generic form 让未注册 vector_ext 的 spine-triton-opt 也能 parse。
        #   operand 均为整寄存器宽 (f16=64, f32=64);tile 规格 m=n=k=8 (SMT 单元固定)。
        acc_ssa, acc_type = self._gen_expr(node.args[0])
        x_ssa, x_type = self._gen_expr(node.args[1])
        y_ssa, y_type = self._gen_expr(node.args[2])
        m = n = k = 8
        result = self._alloc_ssa(hint or "vfwmadot")
        self._emit(f'{result} = "vector_ext.matmul"({x_ssa}, {y_ssa}, {acc_ssa})'
                   f' <{{m = {m} : i64, n = {n} : i64, k = {k} : i64}}>'
                   f' : ({x_type}, {y_type}, {acc_type}) -> {acc_type}')
        return result, acc_type

    def _body_has_vfwmadot(self, body) -> bool:
        """body(含嵌套 for)里是否出现 tle.vfwmadot 调用 → 判定写法4 矩阵单元 mv。"""
        for n in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(n, ast.Call) and _is_spine_raw_attr(n.func, "vfwmadot", self._aliases):
                return True
        return False

    def _emit_mmt4d_from_pattern(self, params) -> bool:
        """写法4(文档 svector vpack/vfwmadot 循环)折成结构化 mmt4d。
        约定:kernel 前 3 个 memref 参数 = (B, Apad, C);M(输出行)/K 从闭包常量
        N/K 取(_constexpr_ints),N_pad=32。成功 emit 返回 True。"""
        mem_params = [(nm, ann) for nm, ann in params if "memref" in ann.mlir_type]
        if len(mem_params) < 3:
            return False
        K = self._constexpr_ints.get("K")
        if K is None:
            return False
        # M = 每 program 处理的行数(闭包常量 N)。有 row_base 形参(index)时为 grid 并发:
        # B/C 按 row_base 运行期动态偏移;否则单 program、无偏移。
        M = self._constexpr_ints.get("N")
        if M is None:
            return False
        idx_params = [nm for nm, ann in params if ann.mlir_type == "index"]
        row_off_ssa = self._env[idx_params[0]][0] if idx_params else None
        B_nm, A_nm, C_nm = mem_params[0][0], mem_params[1][0], mem_params[2][0]
        B_ssa, B_ty = self._env[B_nm]
        A_ssa, A_ty = self._env[A_nm]
        C_ssa, C_ty = self._env[C_nm]
        self._emit_mmt4d_block(B_ssa, B_ty, A_ssa, A_ty, C_ssa, C_ty, M, K, 32, row_off_ssa=row_off_ssa)
        return True

    def _gen_mmt4d(self, node: ast.Call):
        # mmt4d(B, Apad, C, M, K, N): 结构化矩阵乘 C[M,N] = B[M,K] @ Apad[K,N].
        B_ssa, B_ty = self._gen_expr(node.args[0])
        A_ssa, A_ty = self._gen_expr(node.args[1])
        C_ssa, C_ty = self._gen_expr(node.args[2])
        M = self._try_const_int(node.args[3])
        K = self._try_const_int(node.args[4])
        N = self._try_const_int(node.args[5])
        assert None not in (M, K, N), "mmt4d M/K/N must be compile-time ints"
        self._emit_mmt4d_block(B_ssa, B_ty, A_ssa, A_ty, C_ssa, C_ty, M, K, N)

    def _emit_mmt4d_block(self, B_ssa, B_ty, A_ssa, A_ty, C_ssa, C_ty, M, K, N, row_off_ssa=None):
        # 发 linalg.pack + linalg.mmt4d + linalg.unpack;下游 spe_pack → smt.vfwmadot
        #   自动生成 cube 布局(数值正确,已在 K3/179 对拍 torch.mv 通过 max_diff 7e-3)。
        #   tile:mb=16, nb=32, kb=8(满足 mb>=8,nb>=8,kb==8);M%16==0,K%8==0,N%32==0。
        #   row_off_ssa:grid 并发时本 program 的起始行(运行期);B/C 按 row_off*K / row_off*N
        #   动态偏移(memref 带 offset: ?),M=每 program 行数(BLOCK,编译期)。
        MB, NB, KB = 16, 32, 8
        assert M % MB == 0 and K % KB == 0 and N % NB == 0, \
            f"mmt4d needs M%{MB}==0,K%{KB}==0,N%{NB}==0, got M={M},K={K},N={N}"
        et = "f16"
        sp = "#ptr.generic_space"
        mr = lambda r, c: f"memref<{r}x{c}xf16, strided<[{c}, 1]>, {sp}>"
        # 动态行偏移(grid):offset 类型带 ?,offset 值 = row_off * 列数
        dyn = row_off_ssa is not None
        mro = (lambda r, c: f"memref<{r}x{c}xf16, strided<[{c}, 1], offset: ?>, {sp}>") if dyn else mr

        def _off(cols):  # B/C 的元素偏移 = row_off * cols;A 不偏移
            if not dyn:
                return "0"
            o = self._alloc_ssa("roff")
            ccols = self._const_int(cols)
            self._emit(f"{o} = arith.muli {row_off_ssa}, {ccols} : index")
            return o

        cst = self._alloc_ssa("cst")
        self._emit(f"{cst} = arith.constant 0.000000e+00 : {et}")
        # B[M,K] → pack <M/MB,K/KB,MB,KB>
        offB = _off(K)
        rB = self._alloc_ssa("rB")
        self._emit(f"{rB} = memref.reinterpret_cast {B_ssa} to offset: [{offB}], sizes: [{M}, {K}], "
                   f"strides: [{K}, 1] : {B_ty} to {mro(M, K)}")
        tB = self._alloc_ssa("tB")
        self._emit(f"{tB} = bufferization.to_tensor {rB} restrict : {mro(M, K)} to tensor<{M}x{K}x{et}>")
        eB = self._alloc_ssa("eB")
        self._emit(f"{eB} = tensor.empty() : tensor<{M//MB}x{K//KB}x{MB}x{KB}x{et}>")
        pB = self._alloc_ssa("packB")
        self._emit(f"{pB} = linalg.pack {tB} padding_value({cst} : {et}) outer_dims_perm = [0, 1] "
                   f"inner_dims_pos = [0, 1] inner_tiles = [{MB}, {KB}] into {eB} : "
                   f"tensor<{M}x{K}x{et}> -> tensor<{M//MB}x{K//KB}x{MB}x{KB}x{et}>")
        # Apad[K,N] → pack perm[1,0] <N/NB,K/KB,NB,KB>
        rA = self._alloc_ssa("rA")
        self._emit(f"{rA} = memref.reinterpret_cast {A_ssa} to offset: [0], sizes: [{K}, {N}], "
                   f"strides: [{N}, 1] : {A_ty} to {mr(K, N)}")
        tA = self._alloc_ssa("tA")
        self._emit(f"{tA} = bufferization.to_tensor {rA} restrict : {mr(K, N)} to tensor<{K}x{N}x{et}>")
        eA = self._alloc_ssa("eA")
        self._emit(f"{eA} = tensor.empty() : tensor<{N//NB}x{K//KB}x{NB}x{KB}x{et}>")
        pA = self._alloc_ssa("packA")
        self._emit(f"{pA} = linalg.pack {tA} padding_value({cst} : {et}) outer_dims_perm = [1, 0] "
                   f"inner_dims_pos = [1, 0] inner_tiles = [{NB}, {KB}] into {eA} : "
                   f"tensor<{K}x{N}x{et}> -> tensor<{N//NB}x{K//KB}x{NB}x{KB}x{et}>")
        # mmt4d → <M/MB,N/NB,MB,NB>
        eO = self._alloc_ssa("eO")
        self._emit(f"{eO} = tensor.empty() : tensor<{M//MB}x{N//NB}x{MB}x{NB}x{et}>")
        fO = self._alloc_ssa("fill")
        self._emit(f"{fO} = linalg.fill ins({cst} : {et}) outs({eO} : "
                   f"tensor<{M//MB}x{N//NB}x{MB}x{NB}x{et}>) -> tensor<{M//MB}x{N//NB}x{MB}x{NB}x{et}>")
        mm = self._alloc_ssa("mm")
        self._emit(f"{mm} = linalg.mmt4d ins({pB}, {pA} : tensor<{M//MB}x{K//KB}x{MB}x{KB}x{et}>, "
                   f"tensor<{N//NB}x{K//KB}x{NB}x{KB}x{et}>) outs({fO} : "
                   f"tensor<{M//MB}x{N//NB}x{MB}x{NB}x{et}>) -> tensor<{M//MB}x{N//NB}x{MB}x{NB}x{et}>")
        # unpack → C[M,N]
        offC = _off(N)
        rC = self._alloc_ssa("rC")
        self._emit(f"{rC} = memref.reinterpret_cast {C_ssa} to offset: [{offC}], sizes: [{M}, {N}], "
                   f"strides: [{N}, 1] : {C_ty} to {mro(M, N)}")
        tC = self._alloc_ssa("tC")
        self._emit(f"{tC} = bufferization.to_tensor {rC} restrict writable : {mro(M, N)} to tensor<{M}x{N}x{et}>")
        up = self._alloc_ssa("unpack")
        self._emit(f"{up} = linalg.unpack {mm} inner_dims_pos = [0, 1] inner_tiles = [{MB}, {NB}] "
                   f"into {tC} : tensor<{M//MB}x{N//NB}x{MB}x{NB}x{et}> -> tensor<{M}x{N}x{et}>")
        self._emit(f"bufferization.materialize_in_destination {up} in writable {rC} : "
                   f"(tensor<{M}x{N}x{et}>, {mro(M, N)}) -> ()")

    def _gen_vpack(self, node: ast.Call, hint: str) -> tuple[str, str]:
        # vpack(a, b, group_len) → "vector_ext.interleave"(a, b) <{groupLen}>
        #   → lower 到 smt.vpack.vv(硬件 cube pack)。RVV-faithful:1:1 映射硬件
        #   vpack.vv。a/b 同型 1D vector<VLxdtype>,out 为 vector<2VLxdtype>(交织)。
        a_ssa, a_type = self._gen_expr(node.args[0])
        b_ssa, b_type = self._gen_expr(node.args[1])
        group_len = ast.literal_eval(node.args[2])
        n = _vec_n(a_type)
        elem = _vec_elem(a_type)
        out_type = f"vector<{2 * n}x{elem}>"
        result = self._alloc_ssa(hint or "ilv")
        self._emit(f'{result} = "vector_ext.interleave"({a_ssa}, {b_ssa})'
                   f' <{{groupLen = {group_len} : i64}}>'
                   f' : ({a_type}, {b_type}) -> {out_type}')
        return result, out_type

    def _gen_vstore(self, node: ast.Call):
        # SPEC §6.2:vstore(ptr, index, value, stride=None, idx=None)
        #   index — 起始元素偏移(扁平标量,二维坐标由用户压平)。
        #   value — 向量 → 写 VL 个元素(transfer_write → vse);标量 → 只写 1 个元素
        #           (memref.store,reduce 回写)。
        #   stride → vsse(待扩);idx → vsuxei/scatter(待扩)。
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_ssa, ptr_type = self._gen_expr(node.args[0])
        idx_node = node.args[1]
        stride_node = node.args[3] if len(node.args) > 3 else kwargs.get("stride")
        if stride_node is not None or kwargs.get("idx") is not None:
            raise NotImplementedError("vstore stride/idx (vsse/vsuxei) 未实现(SPEC §6.2 待扩)")
        assert not isinstance(idx_node, ast.Tuple), ("vstore 的 index 须为扁平标量元素偏移(二维坐标请自行压平);SPEC §6.2")
        idx_ssa, _ = self._gen_expr(idx_node)
        val_ssa, val_type = self._gen_expr(node.args[2])
        store_ssa, store_type = self._ranked_cast(ptr_ssa, ptr_type)
        if val_type.startswith("vector<"):
            # 宽结果向量写回 (写法4 vfwmadot): 只写 acc 的前 m(=8) 宽有效元素。
            self._emit(f"vector.transfer_write {val_ssa}, {store_ssa}[{idx_ssa}]"
                       f" {{in_bounds = [true]}} : {val_type}, {store_type}")
        else:
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
        self._emit(f"{result} = memref.alloc({operands}) {{alignment = 64 : i64}} : {mtype}")
        return result, mtype

    def _gen_pack(self, node: ast.Call):
        # pack(src, (row0, col0), dst, dst_shape, stride):
        #   pack src's ROWS-row block starting at row0 into dst laid out as
        #   (1, K//nvl, ROWS, nvl), so dst[0, kb, r, :] = src[row0+r, kb*nvl:+nvl].
        #   src is an external row-major pointer with row stride = `stride`
        #   (== K here); col0 is assumed 0 (full rows). dst is a ranked memref
        #   from tle.alloc. The kb loop runs over the runtime column extent.
        src_node, src_idx, dst_node, dst_shape, stride_node = node.args[:5]
        assert isinstance(src_idx, ast.Tuple) and len(src_idx.elts) == 2, \
            "pack src index must be a (row, col) tuple"
        assert isinstance(dst_shape, ast.Tuple) and len(dst_shape.elts) == 4, \
            "pack dst_shape must be 4-D (1, K//nvl, ROWS, nvl)"

        vl = self._require_vl()
        rows = self._try_const_int(dst_shape.elts[2])
        assert rows is not None, \
            "pack ROWS (dst_shape[2]) must be a compile-time constant"

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
            self._emit(f"{vec} = vector.transfer_read {ranked_ssa}[{off}], {pad}"
                       f" {{in_bounds = [true]}} : {ranked_type}, {vec_type}")
            cr_idx = self._const_int(r)
            self._emit(f"vector.transfer_write {vec}, {dst_ssa}[{c0}, {kb_ssa}, {cr_idx}, {c0}]"
                       f" {{in_bounds = [true]}} : {vec_type}, {dst_type}")
        self._indent -= 2
        self._emit("}")
