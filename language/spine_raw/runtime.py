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
        # Mode-1: mark functions using only llvm_* primitives for direct llvm.func emission
        self._mode1 = self._detect_mode1(fn)

    def _detect_mode1(self, fn: Callable) -> bool:
        """Detect if fn uses only mode-1 (llvm_*) primitives by scanning its source."""
        import ast
        import inspect
        try:
            src = inspect.getsource(fn)
            tree = ast.parse(src)
            # Scan for calls to tle.llvm_* or sr.llvm_* (mode-1 markers)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr.startswith("llvm_") or node.func.attr == "call_intrinsic":
                            return True
            return False
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
