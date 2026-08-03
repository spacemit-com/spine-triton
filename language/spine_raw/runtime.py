# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""@spine_raw decorator and SpineLinalgJITFunction.

SpineLinalgJITFunction wraps a Python function annotated with In/InOut and,
on first call to make_body_builder(), runs SpineMLIRBuilderCodegen to build
the raw kernel body straight through the C++ builder API (no MLIR text).
"""
from __future__ import annotations

from typing import Callable

from .codegen import SpineMLIRBuilderCodegen


class SpineLinalgJITFunction:
    """Wrapper around a @spine_raw function that emits its body via builder API.

    Attributes:
        _fn                  : original Python function
        _body_builder_cache  : cached (param_type_strs, body_builder) | None
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self._body_builder_cache = None   # (param_type_strs, body_builder) | None
        self.__triton_builtin__ = True
        # LLVM-direct: mark functions using only llvm_* primitives for direct llvm.func emission
        self._llvm_direct = self._detect_llvm_direct(fn)

    def _detect_llvm_direct(self, fn: Callable) -> bool:
        """Detect if fn uses only llvm-direct (llvm_*) primitives by scanning its source.

        Routes to LLVMDirectTextCodegen (sibling llvm.func, bypasses spine-opt)
        ONLY when every primitive is llvm-direct — i.e. no svector DATA helpers
        (vload/vzero/vmacc/vreduce_*/sstore/vconfig/...). `range` is path-agnostic
        control-flow and used by both, so it doesn't count as a svector marker.
        A mixed kernel (svector helpers + call_intrinsic) goes to
        SpineMLIRBuilderCodegen, whose _gen_call_intrinsic handles
        tle.call_intrinsic inline.
        """
        import ast
        import inspect
        from .codegen import _SPINE_RAW_BUILTIN_NAMES
        # path-agnostic control-flow primitives used by BOTH codegens
        _PATH_AGNOSTIC = {"range", "proton_mark"}
        try:
            src = inspect.getsource(fn)
            tree = ast.parse(src)
            has_llvm_direct = False
            has_svector = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                    if name.startswith("llvm_") or name == "call_intrinsic":
                        has_llvm_direct = True
                    elif name in _PATH_AGNOSTIC:
                        pass  # control-flow, not a svector marker
                    elif name in _SPINE_RAW_BUILTIN_NAMES and not name.startswith("llvm_"):
                        has_svector = True
            # Pure llvm-direct kernel → LLVMDirectTextCodegen.
            # Mixed or pure-svector → SpineMLIRBuilderCodegen (svector path).
            return has_llvm_direct and not has_svector
        except Exception:
            return False

    @property
    def __name__(self) -> str:
        return self._fn.__name__

    def make_body_builder(self):
        """Return (param_type_strs, body_builder) for create_tle_dsl_region_direct."""
        if self._body_builder_cache is None:
            gen = SpineMLIRBuilderCodegen()
            self._body_builder_cache = gen.generate_builder(self._fn)
        return self._body_builder_cache

    def __repr__(self) -> str:
        return f"SpineLinalgJITFunction({self._fn.__name__!r})"


_REGISTRY: dict[str, type] = {
    "linalg": SpineLinalgJITFunction,
}


def spine_raw(*, name: str = "linalg") -> Callable:
    """Decorator: mark a Python function as a raw Linalg MLIR kernel.

    Usage:
        @spine_raw(name="linalg")
        def mv_acc_raw_inner(A: In["memref<*xf16, #ptr.generic_space>"], ...):
            ...
    """
    if name not in _REGISTRY:
        raise ValueError(f"spine_raw: unknown backend {name!r}. Available: {list(_REGISTRY)}")
    cls = _REGISTRY[name]

    def decorator(fn: Callable) -> SpineLinalgJITFunction:
        return cls(fn)

    return decorator
