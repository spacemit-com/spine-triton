# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""In / InOut type annotations for @spine_raw kernel parameters.

Usage:
    def my_fn(A: In["memref<*xf16, #ptr.generic_space>"], n: In["i32"]):
        ...
"""
from __future__ import annotations
from typing import Generic, TypeVar

T = TypeVar("T")


class _TypedAnnotation:
    """Base for In/InOut; carries the MLIR type string."""
    def __init__(self, mlir_type: str, writable: bool):
        self.mlir_type = mlir_type
        self.writable = writable

    def __repr__(self) -> str:
        cls = "InOut" if self.writable else "In"
        return f"{cls}[{self.mlir_type!r}]"


class In(Generic[T]):
    """Read-only input parameter. Maps to the given MLIR type (no return)."""
    _instance: _TypedAnnotation | None = None

    def __class_getitem__(cls, mlir_type: str) -> _TypedAnnotation:
        return _TypedAnnotation(mlir_type, writable=False)


class InOut(Generic[T]):
    """Read-write parameter. The raw function receives it and may mutate it in place.
    For SSA-clean MLIR the caller passes a memref that the function writes into."""
    def __class_getitem__(cls, mlir_type: str) -> _TypedAnnotation:
        return _TypedAnnotation(mlir_type, writable=True)
