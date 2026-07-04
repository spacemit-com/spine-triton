# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""@spine_raw decorator and SpineLinalgJITFunction.

SpineLinalgJITFunction wraps a Python function annotated with In/InOut,
triggers SpineMLIRCodeGenerator on first call to make_linalg(), and
caches the resulting MLIR string.
"""
from __future__ import annotations

from typing import Callable

from .codegen import SpineMLIRCodeGenerator


class SpineLinalgJITFunction:
    """Wrapper around a @spine_raw function that compiles Python → Linalg MLIR.

    Attributes:
        _fn         : original Python function
        _mlir_text  : cached bare func.func string (None until make_linalg() called)
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self._mlir_text: str | None = None
        # Tell Triton's JIT not to track this as a mutable global (same as
        # FlagTree's MLIRJITFunction.__triton_builtin__)
        self.__triton_builtin__ = True

    @property
    def __name__(self) -> str:
        return self._fn.__name__

    def make_linalg(self) -> str:
        """Trigger AST → MLIR compilation (lazy, cached).

        Returns a module-wrapped MLIR string:
            module {
              func.func @name(...) { ... }
            }
        """
        if self._mlir_text is None:
            gen = SpineMLIRCodeGenerator()
            func_text = gen.generate(self._fn)
            self._mlir_text = "module {{\n{}\n}}\n".format(func_text)
        return self._mlir_text

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
