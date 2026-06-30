# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""spine_raw — Python eDSL for writing raw Linalg/memref/vector MLIR kernels.

Public API:
    spine_raw   : decorator to mark a function as a raw MLIR kernel
    In          : read-only parameter annotation
    InOut       : read-write parameter annotation
    call        : inside @triton.jit, emit tle.dsl_region (C++ DSLRegionOpPattern
                  lowers it to spine_ext.raw_region)
"""

from .types import In, InOut
from .runtime import spine_raw, SpineLinalgJITFunction
from .call_registry import call
from .builtins import splat, load_vec, store_vec, store_scalar, fma, extf, reduce_add, matmul
from .builtins import load_tile, pad_vec, extract_elem
from .builtins import batch_macc, view_2d, load_2d, splat_2d, store_2d
from .builtins import load_2d_at, store_2d_at, load_2d_t, pack_2d_t
from .builtins import alloc_tcm_2d, pack_2d_t_into, free_tcm, proton_mark
from .builtins import range as range  # noqa: A001 (shadows builtin intentionally)

__all__ = [
    "spine_raw",
    "SpineLinalgJITFunction",
    "In",
    "InOut",
    "call",
    "splat",
    "load_vec",
    "store_vec",
    "store_scalar",
    "fma",
    "extf",
    "reduce_add",
    "matmul",
    "load_tile",
    "pad_vec",
    "extract_elem",
    "batch_macc",
    "view_2d",
    "load_2d",
    "load_2d_at",
    "load_2d_t",
    "pack_2d_t",
    "alloc_tcm_2d",
    "pack_2d_t_into",
    "free_tcm",
    "proton_mark",
    "splat_2d",
    "store_2d",
    "store_2d_at",
    "range",
]
