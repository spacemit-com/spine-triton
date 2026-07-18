# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""SpineMLIRBuilderCodegen — translates @spine_raw Python functions to MLIR ops.

Phase 1: AST visitor for the spine_raw eDSL subset.
Builds vector/arith/memref/scf ops straight through the C++ builder API
(create_tle_dsl_region_direct), with no MLIR text emission.
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap
from typing import Callable

from .builtins import mma_cube as _mma_cube
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


def _vec_elem_last(mlir_type: str) -> str:
    """Element dtype of a rank-N vector (last component), e.g. vector<2x256xf16> -> f16."""
    m = re.match(r'vector<(?:\d+x)+(bf16|f16|f32|f64|i8|i16|i32|i64)>', mlir_type)
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
    "range", "proton_mark", "vconfig", "vzero", "vload", "vmacc",
    "vreduce_sum", "vreduce_max", "vreduce_min", "vreduce_mul",
    "vstore", "alloc", "pack", "vmadot",
    "vpack", "vbroadcast", "vshape", "spread", "imin", "vmin", "vmax", "sqrt", "rsqrt", "vexp", "vlog", "vscalar", "viota", "abs", "cast", "select"
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
# Builder-API codegen  (no string emission)
# ---------------------------------------------------------------------------

class SpineMLIRBuilderCodegen:
    """Translate @spine_raw fn → C++ builder API calls.

    generate_builder(fn) → (param_type_strs, body_builder)
    body_builder(b, block_args) is the callback for create_tle_dsl_region_direct.
    """

    def __init__(self):
        self._b = None
        self._env: dict[str, tuple] = {}
        self._const_index_cache: dict[int, object] = {}
        self._const_int_typed_cache: dict[tuple, object] = {}
        self._const_float_cache: dict[tuple, object] = {}
        self._constexpr_ints: dict[str, int] = {}
        self._constexpr_floats: dict[str, float] = {}
        self._active_vl: int | None = None
        self._active_valid = None
        self._aliases: set[str] = set()
        self._all_iter_arg_names: set[str] = set()
        self._loop_iter_args: set[str] = set()

    # --- Type helpers ---

    def _t(self, s: str):
        return self._b.parse_type(s)

    def _tf(self, name: str):
        if name == "f16":   return self._b.get_f16_type()
        if name == "f32":   return self._b.get_f32_type()
        return self._t(name)

    # --- Constant helpers (no caching — caches cause dominance violations across regions) ---

    def _const_int(self, n: int):
        return self._b.create_arith_constant_index(n)

    def _const_int_typed(self, n: int, elem: str):
        return self._b.create_arith_constant_int(n, self._t(elem))

    def _const_float(self, v: float, ftype: str = "f32"):
        return self._b.create_arith_constant_float(v, self._tf(ftype))

    # --- Env helpers ---

    def _bind(self, name: str, val, typ: str):
        self._env[name] = (val, typ)

    def _get(self, name: str) -> tuple:
        if name not in self._env:
            raise ValueError(f"Undefined variable: {name!r}")
        return self._env[name]

    def _require_vl(self) -> int:
        if self._active_vl is None:
            raise ValueError("spine_raw svector op used before vconfig() set VL")
        return self._active_vl

    def _try_const_int(self, node) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in self._constexpr_ints:
            return self._constexpr_ints[node.id]
        if isinstance(node, ast.BinOp):
            l = self._try_const_int(node.left)
            r = self._try_const_int(node.right)
            if l is None or r is None:
                return None
            if isinstance(node.op, ast.Add):    return l + r
            if isinstance(node.op, ast.Sub):    return l - r
            if isinstance(node.op, ast.Mult):   return l * r
            if isinstance(node.op, ast.FloorDiv): return l // r
        return None

    # --- broadcast helper ---

    def _broadcast_to(self, val, scalar_type_str: str, vec_type_str: str):
        return self._b.create_vector_broadcast(val, self._t(vec_type_str))

    def _match_operands(self, lv, lt: str, rv, rt: str):
        lv_is = lt.startswith("vector<")
        rv_is = rt.startswith("vector<")
        if lv_is and rv_is:
            if lt != rt:
                raise NotImplementedError(f"mismatched vectors {lt}/{rt}")
            return lv, rv, lt
        if lv_is and not rv_is:
            return lv, self._broadcast_to(rv, _vec_elem(lt), lt), lt
        if rv_is and not lv_is:
            return self._broadcast_to(lv, _vec_elem(rt), rt), rv, rt
        return lv, rv, None

    def _scalar_index_to_float(self, v, ftype: str):
        """index → ftype scalar: index_cast to i64, then sitofp. `index` is not
        an integer type in MLIR so sitofp can't take it directly."""
        i64_v = self._b.create_arith_index_cast(v, self._t("i64"))
        return self._b.create_arith_sitofp(i64_v, self._tf(ftype))

    def _promote_scalar_pair(self, lv, lt: str, rv, rt: str):
        """Promote a pair of scalar operands to a common type, returning
        (lv, rv, result_type_str). Handles index↔float mixes (mean = sum / N)
        by lifting index to the float side; identical types pass through."""
        if lt == rt:
            return lv, rv, lt
        l_f, r_f = _is_float_elem(lt), _is_float_elem(rt)
        if l_f and rt == "index":
            return lv, self._scalar_index_to_float(rv, lt), lt
        if r_f and lt == "index":
            return self._scalar_index_to_float(lv, rt), rv, rt
        if l_f and r_f:
            # differing float widths: widen the narrower to the wider
            wide = lt if _elem_bits(lt) >= _elem_bits(rt) else rt
            if lt != wide:
                lv = self._b.create_arith_extf(lv, self._tf(wide))
            if rt != wide:
                rv = self._b.create_arith_extf(rv, self._tf(wide))
            return lv, rv, wide
        raise NotImplementedError(f"scalar promote between {lt!r} and {rt!r}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate_builder(self, fn):
        """Return (param_type_strs, body_builder) for create_tle_dsl_region_direct."""
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if not func_nodes:
            raise ValueError(f"No function definition in {fn.__name__!r}")
        func_node = func_nodes[0]

        params = _parse_signature(fn)
        param_type_strs = [ann.mlir_type for _, ann in params]

        # Detect spine_raw module aliases
        try:
            import spine_raw as _sr_mod
        except ModuleNotFoundError:
            try:
                from triton.language.extra import spine_raw as _sr_mod
            except (ModuleNotFoundError, ImportError):
                _sr_mod = None
        aliases: set[str] = set()
        if _sr_mod is not None:
            for k, v in (fn.__globals__ or {}).items():
                if v is _sr_mod:
                    aliases.add(k)
        if not aliases:
            aliases = {"spine_raw", "sr"}

        # Closure / global constexprs
        freevars: dict[str, object] = {}
        if getattr(fn, "__closure__", None):
            for nm, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    freevars[nm] = cell.cell_contents
                except ValueError:
                    pass
        constexpr_ints: dict[str, int] = {}
        constexpr_floats: dict[str, float] = {}
        for nm, val in {**(fn.__globals__ or {}), **freevars}.items():
            if isinstance(val, int) and not isinstance(val, bool):
                constexpr_ints.setdefault(nm, val)
            elif isinstance(val, float):
                constexpr_floats.setdefault(nm, val)

        # Pre-scan: find iter_args for all top-level for loops
        all_iter_arg_names: set[str] = set()
        defined_so_far: set[str] = set(p for p, _ in params)
        for stmt in func_node.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        defined_so_far.add(t.id)
            elif isinstance(stmt, ast.For):
                all_iter_arg_names |= _find_reassigned(stmt.body, defined_so_far)

        def body_builder(b, block_args):
            # Fresh state for each invocation
            self.__init__()
            self._b = b
            self._aliases = aliases
            self._constexpr_ints = dict(constexpr_ints)
            self._constexpr_floats = dict(constexpr_floats)
            self._all_iter_arg_names = all_iter_arg_names
            # Bind params to block args
            for (pname, ann), barg in zip(params, block_args):
                self._env[pname] = (barg, ann.mlir_type)
            # Generate body statements
            for stmt in func_node.body:
                if isinstance(stmt, ast.Pass):
                    continue
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    continue
                self._gen_stmt(stmt)

        return param_type_strs, body_builder

    # ------------------------------------------------------------------
    # Statement generators
    # ------------------------------------------------------------------

    def _gen_stmt(self, node):
        if isinstance(node, ast.Assign):
            self._gen_assign(node)
        elif isinstance(node, ast.For):
            self._gen_for(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self._gen_call_stmt(node.value)
        elif isinstance(node, (ast.Return, ast.Pass)):
            pass
        else:
            raise NotImplementedError(f"Unsupported stmt: {ast.dump(node)}")

    def _gen_assign(self, node: ast.Assign):
        assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        target = node.targets[0].id
        if isinstance(node.value, ast.Call) and \
                _is_spine_raw_attr(node.value.func, "vconfig", self._aliases):
            self._gen_vconfig_assign(target, node.value)
            return
        val, typ = self._gen_expr(node.value)
        self._bind(target, val, typ)

    def _gen_for(self, node: ast.For):
        assert isinstance(node.target, ast.Name)
        loop_var = node.target.id
        assert _is_spine_raw_attr(node.iter.func, "range", self._aliases)
        rargs = node.iter.args
        assert len(rargs) in (1, 3)
        if len(rargs) == 1:
            lb = self._const_int(0)
            ub, _ = self._gen_expr(rargs[0])
            step = self._const_int(1)
        else:
            lb, _ = self._gen_expr(rargs[0])
            ub, _ = self._gen_expr(rargs[1])
            step, _ = self._gen_expr(rargs[2])

        outer_vars = set(self._env.keys())
        reassigned = sorted(_find_reassigned(node.body, outer_vars))
        ia_data = [(v, *self._get(v)) for v in reassigned]  # (name, val, typ_str)
        ia_vals = [d[1] for d in ia_data]

        prev_loop_iter = self._loop_iter_args
        self._loop_iter_args = set(reassigned)
        # vconfig inside the body sets _active_vl/_active_valid; _active_valid can
        # be a region-internal SSA (arith.minsi on N-i). Snapshot and restore so a
        # tail-loop vconfig doesn't leak that value into sibling loops → otherwise
        # a later vload references an SSA from a dead sibling region ('operand does
        # not dominate ... neither in a parent nor in a child region').
        saved_active_vl = self._active_vl
        saved_active_valid = self._active_valid

        def for_body(b, iv, region_iter_args):
            # Rebind iter_args to their region block args
            saved = {}
            for d, ria in zip(ia_data, region_iter_args):
                v, _, ts = d
                saved[v] = self._env.get(v)
                self._env[v] = (ria, ts)
            self._env[loop_var] = (iv, "index")
            for stmt in node.body:
                self._gen_stmt(stmt)
            yield_vals = [self._get(v)[0] for v in reassigned]
            # Restore (for_body is called once; restore for post-loop rebind below)
            for v, old in saved.items():
                if old is None:
                    self._env.pop(v, None)
                else:
                    self._env[v] = old
            return yield_vals

        result_vals = self._b.create_scf_for(lb, ub, step, ia_vals, for_body)
        self._loop_iter_args = prev_loop_iter
        # Restore active vconfig state clobbered inside the body.
        self._active_vl = saved_active_vl
        self._active_valid = saved_active_valid

        # Bind results back to the iter_arg names
        for d, rv in zip(ia_data, result_vals):
            v, _, ts = d
            self._env[v] = (rv, ts)

    def _gen_call_stmt(self, node: ast.Call):
        if _is_spine_raw_attr(node.func, "proton_mark", self._aliases):
            pass  # skip profiling marks in builder path
        elif _is_spine_raw_attr(node.func, "vstore", self._aliases):
            self._gen_vstore(node)
        elif _is_spine_raw_attr(node.func, "pack", self._aliases):
            self._gen_pack(node)
        else:
            raise NotImplementedError(f"Unsupported call stmt: {ast.dump(node.func)}")

    # ------------------------------------------------------------------
    # Expression generators
    # ------------------------------------------------------------------

    def _gen_expr(self, node, hint: str = "") -> tuple:
        if isinstance(node, ast.Name):
            if node.id in self._constexpr_ints:
                return self._const_int(self._constexpr_ints[node.id]), "index"
            if node.id in self._constexpr_floats:
                return self._const_float(self._constexpr_floats[node.id]), "f32"
            return self._get(node.id)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, int):   return self._const_int(v), "index"
            if isinstance(v, float): return self._const_float(v), "f32"
            raise NotImplementedError(f"Unsupported literal: {v!r}")
        if isinstance(node, ast.BinOp):
            return self._gen_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._gen_unaryop(node)
        if isinstance(node, ast.Compare):
            return self._gen_compare(node)
        if isinstance(node, ast.Call):
            return self._gen_call_expr(node)
        raise NotImplementedError(f"Unsupported expr: {ast.dump(node)}")

    def _gen_binop(self, node: ast.BinOp) -> tuple:
        lv, lt = self._gen_expr(node.left)
        rv, rt = self._gen_expr(node.right)
        op = type(node.op)
        if lt == "index" and rt == "index":
            m = {ast.Add: "addi", ast.Mult: "muli", ast.Sub: "subi", ast.FloorDiv: "divui"}
            opname = m.get(op)
            if opname is None:
                raise NotImplementedError(f"BinOp {op.__name__} on index")
            fn = getattr(self._b, f"create_arith_{opname}")
            return fn(lv, rv), "index"
        # Scalar arithmetic (neither operand a vector). Covers reduce-then-scale
        # (mean = vreduce_sum(v) / N): promote index→f32 so a f32 scalar and an
        # index (e.g. row count N) can divide/multiply. _match_operands only
        # broadcasts scalars into vectors, so scalar×scalar must be handled here.
        if not lt.startswith("vector<") and not rt.startswith("vector<"):
            lv, rv, st = self._promote_scalar_pair(lv, lt, rv, rt)
            is_f = _is_float_elem(st)
            arith = _BINOP_ARITH.get(op)
            if arith is None:
                raise NotImplementedError(f"Operator {op.__name__} not in _BINOP_ARITH")
            opname = arith[0] if is_f else arith[1]
            if opname is None:
                raise NotImplementedError(
                    f"Operator {op.__name__} not defined for scalar {'float' if is_f else 'int'}")
            fn = getattr(self._b, f"create_arith_{opname}")
            return fn(lv, rv), st
        lv, rv, vt = self._match_operands(lv, lt, rv, rt)
        if vt is None:
            raise NotImplementedError(f"BinOp between {lt!r} and {rt!r}")
        elem = _vec_elem(vt)
        is_f = _is_float_elem(elem)
        arith = _BINOP_ARITH.get(op)
        if arith is None:
            raise NotImplementedError(f"Operator {op.__name__} not in _BINOP_ARITH")
        opname = arith[0] if is_f else arith[1]
        if opname is None:
            raise NotImplementedError(f"Operator {op.__name__} not defined for {'float' if is_f else 'int'}")
        fn = getattr(self._b, f"create_arith_{opname}")
        return fn(lv, rv), vt

    def _gen_unaryop(self, node: ast.UnaryOp) -> tuple:
        vv, vt = self._gen_expr(node.operand)
        # Scalar negation (e.g. -1e38 as fill= argument, or -mean in a formula)
        if not vt.startswith("vector<"):
            if isinstance(node.op, ast.USub) and _is_float_elem(vt):
                return self._b.create_arith_negf(vv), vt
            raise NotImplementedError(f"unary on non-vector {vt}")
        elem = _vec_elem(vt)
        if isinstance(node.op, ast.USub):
            if _is_float_elem(elem):
                return self._b.create_arith_negf(vv), vt
            zero = self._broadcast_to(self._const_int_typed(0, elem), elem, vt)
            return self._b.create_arith_subi(zero, vv), vt
        if isinstance(node.op, ast.Invert):
            if _is_float_elem(elem):
                raise NotImplementedError(f"~a not defined for float {elem}")
            ones = self._broadcast_to(self._const_int_typed(-1, elem), elem, vt)
            return self._b.create_arith_xori(vv, ones), vt
        raise NotImplementedError(f"Unary {type(node.op).__name__}")

    def _gen_compare(self, node: ast.Compare) -> tuple:
        if len(node.ops) != 1:
            raise NotImplementedError("chained comparison not supported")
        lv, lt = self._gen_expr(node.left)
        rv, rt = self._gen_expr(node.comparators[0])
        op = type(node.ops[0])
        lv, rv, vt = self._match_operands(lv, lt, rv, rt)
        if vt is None:
            raise NotImplementedError(f"compare between {lt!r} and {rt!r}")
        pred_pair = _CMP_PRED.get(op)
        if pred_pair is None:
            raise NotImplementedError(f"Compare {op.__name__}")
        elem = _vec_elem(vt)
        if _is_float_elem(elem):
            pred = pred_pair[0]
            return self._b.create_arith_cmpf(pred, lv, rv), f"vector<{_vec_n(vt)}xi1>"
        pred = pred_pair[1]
        return self._b.create_arith_cmpi(pred, lv, rv), f"vector<{_vec_n(vt)}xi1>"

    def _gen_call_expr(self, node: ast.Call) -> tuple:
        b = self._aliases
        if _is_spine_raw_attr(node.func, "vzero", b):      return self._gen_vzero(node)
        if _is_spine_raw_attr(node.func, "vload", b):      return self._gen_vload(node)
        if _is_spine_raw_attr(node.func, "vmacc", b):      return self._gen_vmacc(node)
        if _is_spine_raw_attr(node.func, "vreduce_sum", b): return self._gen_vreduce_sum(node)
        if _is_spine_raw_attr(node.func, "vreduce_max", b): return self._gen_vreduce_max(node)
        if _is_spine_raw_attr(node.func, "vreduce_min", b): return self._gen_vreduce_min(node)
        if _is_spine_raw_attr(node.func, "vreduce_mul", b): return self._gen_vreduce_mul(node)
        if _is_spine_raw_attr(node.func, "vmadot", b):     return self._gen_vmadot(node)
        if _is_spine_raw_attr(node.func, "vpack", b):      return self._gen_vpack(node)
        if _is_spine_raw_attr(node.func, "vshape", b):     return self._gen_vshape(node)
        if _is_spine_raw_attr(node.func, "vbroadcast", b): return self._gen_vbroadcast(node)
        if _is_spine_raw_attr(node.func, "alloc", b):      return self._gen_alloc(node)
        if _is_spine_raw_attr(node.func, "spread", b):     return self._gen_spread(node)
        if _is_spine_raw_attr(node.func, "imin", b):
            av, _ = self._gen_expr(node.args[0])
            bv, _ = self._gen_expr(node.args[1])
            return self._b.create_arith_minsi(av, bv), "index"
        for nm in ("vmin", "vmax"):
            if _is_spine_raw_attr(node.func, nm, b):
                return self._gen_vminmax(node, nm)
        if _is_spine_raw_attr(node.func, "sqrt", b):  return self._gen_unary_math(node, "sqrt")
        if _is_spine_raw_attr(node.func, "rsqrt", b): return self._gen_unary_math(node, "rsqrt")
        if _is_spine_raw_attr(node.func, "vexp", b):  return self._gen_unary_math(node, "exp")
        if _is_spine_raw_attr(node.func, "vlog", b):  return self._gen_unary_math(node, "log")
        if _is_spine_raw_attr(node.func, "vscalar", b): return self._gen_vscalar(node)
        if _is_spine_raw_attr(node.func, "viota", b): return self._gen_viota(node)
        if _is_spine_raw_attr(node.func, "abs", b):   return self._gen_abs(node)
        if _is_spine_raw_attr(node.func, "cast", b):  return self._gen_cast(node)
        if _is_spine_raw_attr(node.func, "select", b): return self._gen_select(node)
        raise NotImplementedError(f"Unsupported call: {ast.dump(node.func)}")

    # ------------------------------------------------------------------
    # vconfig
    # ------------------------------------------------------------------

    def _gen_vconfig_assign(self, target: str, node: ast.Call):
        lmul = ast.literal_eval(node.args[1]) if len(node.args) > 1 else 1
        vl = _vlmax(int(lmul))
        self._constexpr_ints[target] = vl
        self._active_vl = vl
        try:
            avl_const = ast.literal_eval(node.args[0])
        except Exception:
            avl_const = None
        if avl_const is not None and avl_const >= _vlmax(1):
            self._active_valid = None
        elif avl_const is not None and avl_const < 0:
            self._active_valid = None
        else:
            avl_val, _ = self._gen_expr(node.args[0])
            vlmax_val = self._const_int(vl)
            self._active_valid = self._b.create_arith_minsi(vlmax_val, avl_val)

    # ------------------------------------------------------------------
    # vzero / vmacc / vreduce_sum
    # ------------------------------------------------------------------

    def _gen_vzero(self, node: ast.Call) -> tuple:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        dtype = _resolve_dtype(node.args[0] if node.args else None, "f32")
        vl = self._require_vl()
        group = self._try_const_int(kwargs["group"]) if "group" in kwargs else None
        vt = f"vector<{group}x{vl}x{dtype}>" if group else f"vector<{vl}x{dtype}>"
        zero = self._const_float(0.0, dtype)
        return self._b.create_vector_broadcast(zero, self._t(vt)), vt

    def _gen_vmacc(self, node: ast.Call) -> tuple:
        acc_v, acc_t = self._gen_expr(node.args[0])
        x_v, x_t = self._gen_expr(node.args[1])
        y_v, y_t = self._gen_expr(node.args[2])
        acc_elem = _vec_elem(acc_t)
        n = _vec_n(acc_t)
        wide_t = f"vector<{n}x{acc_elem}>"
        wide_T = self._t(wide_t)
        if x_t != wide_t:
            x_v = self._b.create_arith_extf(x_v, wide_T)
        if y_t != wide_t:
            y_v = self._b.create_arith_extf(y_v, wide_T)
        return self._b.create_math_fma(x_v, y_v, acc_v), acc_t

    def _gen_vreduce_sum(self, node: ast.Call) -> tuple:
        v_v, v_t = self._gen_expr(node.args[0])
        elem = _vec_elem(v_t)
        return self._b.create_vector_reduction("add", v_v), elem

    def _gen_vreduce_max(self, node: ast.Call) -> tuple:
        v_v, v_t = self._gen_expr(node.args[0])
        elem = _vec_elem(v_t)
        if not _is_float_elem(elem):
            raise NotImplementedError(f"vreduce_max on integer element {elem!r} not yet wired (add maxsi to binding)")
        return self._b.create_vector_reduction("maxf", v_v), elem

    def _gen_vreduce_min(self, node: ast.Call) -> tuple:
        v_v, v_t = self._gen_expr(node.args[0])
        elem = _vec_elem(v_t)
        if not _is_float_elem(elem):
            raise NotImplementedError(f"vreduce_min on integer element {elem!r} not yet wired (add minsi to binding)")
        return self._b.create_vector_reduction("minf", v_v), elem

    def _gen_vreduce_mul(self, node: ast.Call) -> tuple:
        v_v, v_t = self._gen_expr(node.args[0])
        elem = _vec_elem(v_t)
        return self._b.create_vector_reduction("mul", v_v), elem

    # ------------------------------------------------------------------
    # vmadot / vminmax / unary math / abs / cast / select
    # ------------------------------------------------------------------

    def _gen_vmadot(self, node: ast.Call) -> tuple:
        acc_v, acc_t = self._gen_expr(node.args[0])
        x_v,  x_t  = self._gen_expr(node.args[1])
        y_v,  y_t  = self._gen_expr(node.args[2])
        xm = re.match(r'vector<(\d+)x(\d+)x(f16|bf16)>', x_t)
        ym = re.match(r'vector<(\d+)x(\d+)x(f16|bf16)>', y_t)
        am = re.match(r'vector<(\d+)x(\d+)xf32>', acc_t)
        if not (xm and ym and am):
            raise ValueError(f"vmadot type error: x={x_t} y={y_t} acc={acc_t}")
        b1, b2, B = int(xm.group(1)), int(ym.group(1)), int(am.group(1))
        if B != b1 * b2:
            raise ValueError(f"vmadot acc rows must be b1·b2={b1*b2}, got {B}")
        m_s, n_s, k_s = _mma_cube(xm.group(3))
        lane = n_s * k_s
        result_T = self._t(acc_t)
        res = self._b.create_generic_op(
            "vector_ext.cross_batch_matmul",
            [x_v, y_v, acc_v],
            {"k": k_s, "m": m_s, "n": n_s},
            [result_T])
        return res[0], acc_t

    def _gen_vminmax(self, node: ast.Call, which: str) -> tuple:
        lv, lt = self._gen_expr(node.args[0])
        rv, rt = self._gen_expr(node.args[1])
        lv, rv, vt = self._match_operands(lv, lt, rv, rt)
        if vt is None:
            raise NotImplementedError(f"{which}: needs at least one vector")
        elem = _vec_elem(vt)
        if _is_float_elem(elem):
            fn = self._b.create_arith_minimumf if which == "vmin" else self._b.create_arith_maximumf
        else:
            fn = self._b.create_arith_minsi if which == "vmin" else self._b.create_arith_maxsi
        return fn(lv, rv), vt

    def _gen_unary_math(self, node: ast.Call, which: str) -> tuple:
        vv, vt = self._gen_expr(node.args[0])
        fn = getattr(self._b, f"create_math_{which}")
        return fn(vv), vt

    def _gen_abs(self, node: ast.Call) -> tuple:
        vv, vt = self._gen_expr(node.args[0])
        elem = _vec_elem(vt)
        fn = self._b.create_math_absf if _is_float_elem(elem) else self._b.create_math_absi
        return fn(vv), vt

    def _gen_cast(self, node: ast.Call) -> tuple:
        vv, vt = self._gen_expr(node.args[0])
        # Scalar cast (e.g. cast(i, f32) where i is a scalar index) — used by
        # index-tracking reductions to combine loop counters with float lanes.
        if not vt.startswith("vector<"):
            dst_elem = _resolve_dtype(node.args[1], vt)
            if dst_elem == vt:
                return vv, vt
            if vt == "index" and _is_float_elem(dst_elem):
                return self._scalar_index_to_float(vv, dst_elem), dst_elem
            raise NotImplementedError(f"scalar cast {vt!r} → {dst_elem!r} not supported")
        # Element token: _vec_elem_last handles rank-N (vector<16x32xf32>→f32);
        # fall back to index detection since _vec_elem_last's regex omits index.
        src_elem = _vec_elem_last(vt)
        if src_elem is None:
            src_elem = "index" if vt.endswith("xindex>") else _vec_elem(vt)
        dst_elem = _resolve_dtype(node.args[1], src_elem)
        if dst_elem == src_elem:
            return vv, vt
        # Rebuild the vector type by swapping only the trailing element token
        # (can't rfind("x") because "index" itself contains an 'x').
        prefix = vt[:vt.rfind("x" + src_elem)] + "x"
        dst_type_str = prefix + dst_elem + ">"
        dst_T = self._t(dst_type_str)
        # index-element vector → float: index has no bit width for extf/sitofp
        # directly; go index → i64 → float (mirrors scalar path).
        if src_elem == "index" and _is_float_elem(dst_elem):
            i64_vt = prefix + "i64>"
            i64_v = self._b.create_arith_index_cast(vv, self._t(i64_vt))
            return self._b.create_arith_sitofp(i64_v, dst_T), dst_type_str
        sf, df = _is_float_elem(src_elem), _is_float_elem(dst_elem)
        if sf and df:
            fn = self._b.create_arith_extf if _elem_bits(dst_elem) > _elem_bits(src_elem) else self._b.create_arith_truncf
        elif sf and not df:
            fn = self._b.create_arith_fptosi
        elif not sf and df:
            fn = self._b.create_arith_sitofp
        else:
            fn = self._b.create_arith_extsi if _elem_bits(dst_elem) > _elem_bits(src_elem) else self._b.create_arith_trunci
        return fn(vv, dst_T), dst_type_str

    def _gen_select(self, node: ast.Call) -> tuple:
        mv, mt = self._gen_expr(node.args[0])
        av, at = self._gen_expr(node.args[1])
        bv, bt = self._gen_expr(node.args[2])
        av, bv, vt = self._match_operands(av, at, bv, bt)
        if vt is None:
            raise NotImplementedError("select: a/b must include at least one vector")
        return self._b.create_arith_select(mv, av, bv), vt

    # ------------------------------------------------------------------
    # vshape / vbroadcast
    # ------------------------------------------------------------------

    def _gen_vshape(self, node: ast.Call) -> tuple:
        vv, vt = self._gen_expr(node.args[0])
        elem = _vec_elem_last(vt)
        dims = [self._try_const_int(e) for e in node.args[1].elts] \
            if isinstance(node.args[1], ast.Tuple) else [self._try_const_int(node.args[1])]
        if any(d is None for d in dims):
            raise ValueError("vshape shape must be compile-time ints")
        out_t = f"vector<{'x'.join(str(d) for d in dims)}x{elem}>"
        return self._b.create_vector_shape_cast(vv, self._t(out_t)), out_t

    def _gen_vbroadcast(self, node: ast.Call) -> tuple:
        vv, vt = self._gen_expr(node.args[0])
        n = self._try_const_int(node.args[1])
        if n is None:
            raise ValueError("vbroadcast n must be a compile-time int")
        elem = _vec_elem_last(vt)
        inner = _vec_n(vt)
        out_t = f"vector<{n}x{inner}x{elem}>"
        return self._b.create_vector_broadcast(vv, self._t(out_t)), out_t

    # ------------------------------------------------------------------
    # alloc
    # ------------------------------------------------------------------

    def _gen_alloc(self, node: ast.Call) -> tuple:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        shape_node = node.args[0]
        assert isinstance(shape_node, ast.Tuple)
        dt_node = node.args[1] if len(node.args) > 1 else kwargs.get("dtype")
        dtype = _resolve_dtype(dt_node, "f16")
        dims: list[str] = []
        dyn_vals = []
        for e in shape_node.elts:
            cv = self._try_const_int(e)
            if cv is not None:
                dims.append(str(cv))
            else:
                vv, _ = self._gen_expr(e)
                dims.append("?")
                dyn_vals.append(vv)
        mtype_str = f"memref<{'x'.join(dims)}x{dtype}>"
        return self._b.create_memref_alloc(self._t(mtype_str), dyn_vals if dyn_vals else None, 64), mtype_str

    # ------------------------------------------------------------------
    # _ranked_cast helper
    # ------------------------------------------------------------------

    def _ranked_cast(self, ptr_v, ptr_t: str) -> tuple:
        if ptr_t.startswith("memref<*x"):
            ranked_t = ptr_t.replace("memref<*x", "memref<?x", 1)
            return self._b.create_memref_cast(self._t(ranked_t), ptr_v), ranked_t
        return ptr_v, ptr_t

    # ------------------------------------------------------------------
    # vload
    # ------------------------------------------------------------------

    def _gen_vload(self, node: ast.Call) -> tuple:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_v, ptr_t = self._gen_expr(node.args[0])
        idx_node = node.args[1]
        dtype = _resolve_dtype(kwargs.get("dtype"), "f16")
        vl = self._require_vl()
        vt = f"vector<{vl}x{dtype}>"
        # fill= kwarg: value for padding of tail tiles (default 0.0).
        # Softmax exp-accumulation needs fill=-1e38 so exp(fill-xmax)≈0.
        fill_node = kwargs.get("fill")
        if fill_node is not None:
            fill_v, _ = self._gen_expr(fill_node)
            pad = fill_v
        else:
            pad = self._const_float(0.0, dtype)

        # Packed cube tensor from vpack(memref)
        group_kw = kwargs.get("group")
        if ptr_t.startswith("tensor<") and isinstance(idx_node, ast.Tuple) and group_kw is not None:
            group = self._try_const_int(group_kw)
            assert group is not None
            elem = re.search(r'(bf16|f16|f32|f64|i8|i16|i32|i64)>$', ptr_t).group(1)
            idx_vs = [self._gen_expr(e)[0] for e in idx_node.elts]
            c0 = self._const_int(0)
            flat_t = f"vector<{group * vl}x{elem}>"
            flat_v = self._b.create_vector_transfer_read(self._t(flat_t), ptr_v, idx_vs + [c0], pad, [True])
            out_t = f"vector<{group}x{vl}x{elem}>"
            return self._b.create_vector_shape_cast(flat_v, self._t(out_t)), out_t

        if not ptr_t.startswith("memref<*x"):
            # Ranked memref (alloc scratch)
            assert isinstance(idx_node, ast.Tuple)
            idx_vs = [self._gen_expr(e)[0] for e in idx_node.elts]
            return self._b.create_vector_transfer_read(self._t(vt), ptr_v, idx_vs, pad, [True]), vt

        # External unranked pointer with valid= or _active_valid fill-0 path
        valid_node = kwargs.get("valid")
        valid_v_active = None if valid_node is not None else self._active_valid
        if valid_node is not None or valid_v_active is not None:
            assert "group" not in kwargs
            valid_v = self._gen_expr(valid_node)[0] if valid_node is not None else valid_v_active
            off_v, _ = self._gen_expr(idx_node)
            sp = "#ptr.generic_space"
            src_mr_t = f"memref<?x{dtype}, strided<[?], offset: ?>, {sp}>"
            rsrc = self._b.create_memref_reinterpret_cast(
                self._t(src_mr_t), ptr_v, [off_v], [valid_v], [self._const_int(1)])
            tens_t = f"tensor<?x{dtype}>"
            tsrc = self._b.create_bufferization_to_tensor(rsrc, self._t(tens_t))
            fill_tens_t = f"tensor<{vl}x{dtype}>"
            escr = self._b.create_tensor_empty(self._t(fill_tens_t))
            fscr = self._b.create_linalg_fill(pad, escr)
            c0 = self._const_int(0)
            filled = self._b.create_tensor_insert_slice(tsrc, fscr, [c0], [valid_v], [self._const_int(1)])
            return self._b.create_vector_transfer_read(self._t(vt), filled, [c0], pad, [True]), vt

        ranked_v, ranked_t = self._ranked_cast(ptr_v, ptr_t)
        off_v, _ = self._gen_expr(idx_node)
        group = self._try_const_int(group_kw) if group_kw is not None else None
        if group:
            flat_t = f"vector<{group * vl}x{dtype}>"
            flat_v = self._b.create_vector_transfer_read(self._t(flat_t), ranked_v, [off_v], pad, [True])
            out_t = f"vector<{group}x{vl}x{dtype}>"
            return self._b.create_vector_shape_cast(flat_v, self._t(out_t)), out_t
        return self._b.create_vector_transfer_read(self._t(vt), ranked_v, [off_v], pad, [True]), vt

    # ------------------------------------------------------------------
    # vscalar — scalar load from a pointer at a dynamic index
    # ------------------------------------------------------------------

    def _gen_vscalar(self, node: ast.Call) -> tuple:
        """vscalar(ptr, idx, dtype=f32) → scalar element load from ptr[idx].

        Useful for gather-like access (e.g. cross_entropy: logits[target]).
        Uses _ranked_cast + memref.load — no C++ changes required.
        """
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_v, ptr_t = self._gen_expr(node.args[0])
        idx_v, _ = self._gen_expr(node.args[1])
        dtype = _resolve_dtype(kwargs.get("dtype"), "f32")
        ranked_v, _ = self._ranked_cast(ptr_v, ptr_t)
        return self._b.create_memref_load(ranked_v, [idx_v]), dtype

    def _gen_viota(self, node: ast.Call) -> tuple:
        """viota() → vector<VLxindex> = [0, 1, .., VL-1] (vector.step).

        Index vector for index-tracking reductions (argmax/argmin).
        """
        vl = self._require_vl()
        vt = f"vector<{vl}xindex>"
        return self._b.create_vector_step(self._t(vt)), vt

    # ------------------------------------------------------------------
    # vstore
    # ------------------------------------------------------------------

    def _gen_vstore(self, node: ast.Call):
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        ptr_v, ptr_t = self._gen_expr(node.args[0])
        idx_node = node.args[1]
        val_v, val_t = self._gen_expr(node.args[2])
        shape_node = kwargs.get("shape")
        if shape_node is not None:
            dims = [self._try_const_int(e) for e in shape_node.elts]
            if any(d is None for d in dims) or len(dims) != 2:
                raise ValueError("vstore shape= must be 2-tuple of compile-time ints")
            R, C = dims
            elem = _vec_elem_last(val_t)
            off_v, _ = self._gen_expr(idx_node)
            sp = "#ptr.generic_space"
            m2t = f"memref<{R}x{C}x{elem}, strided<[{C}, 1], offset: ?>, {sp}>"
            # 结果类型两维全静态 RxC:mixed 传 int,否则 static_sizes 全 dynamic 冲突。
            r2 = self._b.create_memref_reinterpret_cast_mixed(
                self._t(m2t), ptr_v, [off_v],
                [R, C], [C, 1])
            c0 = self._const_int(0)
            self._b.create_vector_transfer_write(val_v, r2, [c0, c0], [False, False])
            return
        assert not isinstance(idx_node, ast.Tuple)
        idx_v, _ = self._gen_expr(idx_node)
        if val_t.startswith("vector<"):
            vn = _vec_n(val_t)
            elem = _vec_elem(val_t)
            sp = "#ptr.generic_space"
            c0 = self._const_int(0)
            if self._active_valid is None:
                # Full-tile path: static memref<VLxT, strided<[1], offset:?>>.
                # Dynamic memref<?xT> causes VL to be clamped by descriptor size
                # → only lane0 written. Static size bypasses clamping.
                m1t = f"memref<{vn}x{elem}, strided<[1], offset: ?>, {sp}>"
                r1 = self._b.create_memref_reinterpret_cast_mixed(
                    self._t(m1t), ptr_v, [idx_v], [vn], [1])
                self._b.create_vector_transfer_write(val_v, r1, [c0], [True])
            else:
                # Tail-tile path: only _active_valid < VL elements are valid.
                # Use dynamic memref<?xT> with size=valid + in_bounds=[false]
                # so transfer_write generates a masked store respecting the bound.
                valid_v = self._active_valid
                m1t = f"memref<?x{elem}, strided<[?], offset: ?>, {sp}>"
                r1 = self._b.create_memref_reinterpret_cast(
                    self._t(m1t), ptr_v, [idx_v], [valid_v], [self._const_int(1)])
                self._b.create_vector_transfer_write(val_v, r1, [c0], [False])
        else:
            store_v, store_t = self._ranked_cast(ptr_v, ptr_t)
            self._b.create_memref_store(val_v, store_v, [idx_v])

    # ------------------------------------------------------------------
    # vpack
    # ------------------------------------------------------------------

    def _gen_vpack(self, node: ast.Call) -> tuple:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        first_v, first_t = self._gen_expr(node.args[0])
        if first_t.startswith("memref<"):
            # memref branch: linalg.pack path
            it = kwargs.get("inner_tiles")
            assert it is not None and isinstance(it, ast.Tuple) and len(it.elts) == 2
            rt = self._try_const_int(it.elts[0])
            kt = self._try_const_int(it.elts[1])
            K = self._try_const_int(kwargs["stride"]) if "stride" in kwargs else None
            rows = self._try_const_int(kwargs["rows"]) if "rows" in kwargs else None
            assert None not in (rt, kt, K, rows)
            et = _memref_elem(first_t) if "memref<*x" not in first_t else \
                re.search(r'memref<\*x([a-z0-9]+)', first_t).group(1)
            sp = "#ptr.generic_space"
            vr_node = kwargs.get("valid_rows")
            Mp = ((rows + rt - 1) // rt) * rt
            Kp = ((K + kt - 1) // kt) * kt
            need_pad = (Mp != rows) or (Kp != K) or (vr_node is not None)
            off_node = kwargs.get("offset")
            off_v = self._gen_expr(off_node)[0] if off_node is not None else self._const_int(0)
            cst = self._const_float(0.0, et)
            Kv = self._const_int(K)
            if vr_node is not None:
                vr_v = self._gen_expr(vr_node)[0]
                mr_t = f"memref<?x{K}x{et}, strided<[{K}, 1], offset: ?>, {sp}>"
                # dim0 动态(vr_v), dim1 静态 K:必须用 mixed,否则 static_sizes 把
                # K 也标成 dynamic → 'expected result type with size = dynamic instead of K'。
                r2 = self._b.create_memref_reinterpret_cast_mixed(
                    self._t(mr_t), first_v, [off_v], [vr_v, K],
                    [K, 1])
                tsrc = self._b.create_bufferization_to_tensor(r2, self._t(f"tensor<?x{K}x{et}>"))
                dyn_rows_v = vr_v
            else:
                mr_t = (f"memref<{rows}x{K}x{et}, strided<[{K}, 1], offset: ?>, {sp}>"
                        if off_node is not None else f"memref<{rows}x{K}x{et}, strided<[{K}, 1]>, {sp}>")
                # 两维全静态:mixed 传 int 保持 static_sizes=[rows, K] 与结果类型一致。
                r2 = self._b.create_memref_reinterpret_cast_mixed(
                    self._t(mr_t), first_v, [off_v],
                    [rows, K], [K, 1])
                tsrc = self._b.create_bufferization_to_tensor(r2, self._t(f"tensor<{rows}x{K}x{et}>"))
                dyn_rows_v = None
            if need_pad:
                ep = self._b.create_tensor_empty(self._t(f"tensor<{Mp}x{Kp}x{et}>"))
                fp = self._b.create_linalg_fill(cst, ep)
                # source tsrc 是 tensor<?xKx> 或 tensor<rowsxKx>:dim1=K 静态,
                # insert_slice sizes 须 mixed(dim1 传 int K),否则 static_sizes 全 dynamic
                # 与 source 静态维冲突 → 'expected type tensor<?x?xf16>' rank/size mismatch。
                if dyn_rows_v is not None:
                    ins = self._b.create_tensor_insert_slice_mixed(tsrc, fp,
                        [0, 0], [dyn_rows_v, K], [1, 1])
                else:
                    ins = self._b.create_tensor_insert_slice_mixed(tsrc, fp,
                        [0, 0], [rows, K], [1, 1])
                src_v, src_rows, src_K = ins, Mp, Kp
            else:
                src_v, src_rows, src_K = tsrc, rows, K
            oc, kc = src_rows // rt, src_K // kt
            eP = self._b.create_tensor_empty(self._t(f"tensor<{oc}x{kc}x{rt}x{kt}x{et}>"))
            pk = self._b.create_linalg_pack(src_v, eP, cst, [rt, kt], [0, 1], [0, 1])
            col_t = f"tensor<{oc}x{kc}x{rt * kt}x{et}>"
            col = self._b.create_tensor_collapse_shape(pk, [[0], [1], [2, 3]])
            return col, col_t

        # vector branch: group_interleave
        group_len = self._try_const_int(node.args[1])
        if group_len is None:
            raise ValueError("vpack(vector) group_len must be compile-time int")
        m_re = re.match(r'vector<(\d+)x(\d+)x(f16|bf16|f32)>', first_t)
        if not m_re:
            raise ValueError(f"vpack(vector) needs rank-2 vector, got {first_t}")
        b, ncol, elem = int(m_re.group(1)), int(m_re.group(2)), m_re.group(3)
        out_t = f"vector<{b // 2}x{ncol * 2}x{elem}>"
        res = self._b.create_generic_op(
            "vector_ext.group_interleave", [first_v], {"groupLen": group_len},
            [self._t(out_t)])
        return res[0], out_t

    # ------------------------------------------------------------------
    # spread
    # ------------------------------------------------------------------

    def _gen_spread(self, node: ast.Call) -> tuple:
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        cs_node = kwargs.get("cube_shape") or (node.args[1] if len(node.args) > 1 else None)
        assert isinstance(cs_node, ast.Tuple) and len(cs_node.elts) == 3
        kc = self._try_const_int(cs_node.elts[0])
        n  = self._try_const_int(cs_node.elts[1])
        k  = self._try_const_int(cs_node.elts[2])
        assert None not in (kc, n, k)
        src_v, src_t = self._gen_expr(node.args[0])
        et = re.search(r'([a-z0-9]+)(?:,|>)', src_t.split("memref<")[1]).group(1) \
            if "memref<*x" not in src_t else re.search(r'memref<\*x([a-z0-9]+)', src_t).group(1)
        sp = "#ptr.generic_space"
        total = kc * k
        k_real = self._try_const_int(kwargs["k_real"]) if "k_real" in kwargs else total
        assert k_real is not None and k_real <= total
        pad_k = k_real < total
        # src → 1D <k_real>
        # src → 1D <k_real>:dim0 静态 k_real(mixed 传 int),但 stride 是 strided<[?]>
        # 动态(earlier strided fix 为满足 to_tensor),故 stride 仍传 Value;offset 静态 0。
        src1d_t = f"memref<{k_real}x{et}, strided<[?]>, {sp}>"
        rsrc = self._b.create_memref_reinterpret_cast_mixed(
            self._t(src1d_t), src_v, [0],
            [k_real], [self._const_int(1)])
        scr_t = f"memref<{kc}x{n}x{k}x{et}>"
        scr = self._b.create_memref_alloc(self._t(scr_t), None, 64)
        c0, c1 = self._const_int(0), self._const_int(1)
        ckc, cn, ck = self._const_int(kc), self._const_int(n), self._const_int(k)
        ckreal = self._const_int(k_real) if pad_k else None
        ckrm1  = self._const_int(k_real - 1) if pad_k else None
        zcst   = self._const_float(0.0, et) if pad_k else None

        def outer_body(b, li, _):
            def mid_body(b, lni, _):
                def inner_body(b, lki, _):
                    ck8 = b.create_arith_muli(li, ck)
                    idx = b.create_arith_addi(ck8, lki)
                    if pad_k:
                        inb = b.create_arith_cmpi("slt", idx, ckreal)
                        cl  = b.create_arith_minsi(idx, ckrm1)
                        ld  = b.create_memref_load(rsrc, [cl])
                        av  = b.create_arith_select(inb, ld, zcst)
                    else:
                        av = b.create_memref_load(rsrc, [idx])
                    b.create_memref_store(av, scr, [li, lni, lki])
                    return []
                b.create_scf_for(c0, ck, c1, [], inner_body)
                return []
            b.create_scf_for(c0, cn, c1, [], mid_body)
            return []
        self._b.create_scf_for(c0, ckc, c1, [], outer_body)
        col_t = f"memref<{kc}x{n * k}x{et}>"
        col = self._b.create_memref_collapse_shape(scr, [[0], [1, 2]])
        return col, col_t

    # ------------------------------------------------------------------
    # pack (statement)
    # ------------------------------------------------------------------

    def _gen_pack(self, node: ast.Call):
        src_node, src_idx, dst_node, dst_shape, stride_node = node.args[:5]
        assert isinstance(src_idx, ast.Tuple) and len(src_idx.elts) == 2
        assert isinstance(dst_shape, ast.Tuple) and len(dst_shape.elts) == 4
        vl = self._require_vl()
        rows = self._try_const_int(dst_shape.elts[2])
        assert rows is not None
        dst_v, dst_t = self._gen_expr(dst_node)
        dtype = _memref_elem(dst_t)
        vt = f"vector<{vl}x{dtype}>"
        src_v, src_t = self._gen_expr(src_node)
        ranked_v, ranked_t = self._ranked_cast(src_v, src_t)
        row0_v, _ = self._gen_expr(src_idx.elts[0])
        stride_v, _ = self._gen_expr(stride_node)
        pad = self._const_float(0.0, dtype)
        c0 = self._const_int(0)
        cvl = self._const_int(vl)
        sp = "#ptr.generic_space"
        src_mr_t = f"memref<?x{dtype}, strided<[?], offset: ?>, {sp}>"

        def loop_body(b, loop_v, _):
            kb = b.create_arith_divui(loop_v, cvl)
            for r in range(rows):
                nir = row0_v if r == 0 else b.create_arith_addi(row0_v, self._const_int(r))
                roff = b.create_arith_muli(nir, stride_v)
                off  = b.create_arith_addi(roff, loop_v)
                rem   = b.create_arith_subi(stride_v, loop_v)
                valid = b.create_arith_minsi(cvl, rem)
                rsrc  = b.create_memref_reinterpret_cast(
                    self._t(src_mr_t), ranked_v, [off], [valid], [self._const_int(1)])
                tsrc  = b.create_bufferization_to_tensor(rsrc, self._t(f"tensor<?x{dtype}>"))
                escr  = b.create_tensor_empty(self._t(f"tensor<{vl}x{dtype}>"))
                fscr  = b.create_linalg_fill(pad, escr)
                filled = b.create_tensor_insert_slice(tsrc, fscr, [c0], [valid], [self._const_int(1)])
                vec   = b.create_vector_transfer_read(self._t(vt), filled, [c0], pad, [True])
                cr_idx = self._const_int(r)
                # 1D vector<VL> 写入 rank-4 memref:permutation_map 只 1 个 result(d3),
                # in_bounds 须与 map results 同 rank(=1),不是索引数(4)。
                b.create_vector_transfer_write(vec, dst_v, [c0, kb, cr_idx, c0], [True])
            return []
        self._b.create_scf_for(c0, stride_v, cvl, [], loop_body)
