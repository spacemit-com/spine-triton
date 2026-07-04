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

from .types import In, InOut, mem, index
from .runtime import spine_raw, SpineLinalgJITFunction
from .call_registry import call
from .builtins import batch_macc, view_2d, load_2d, splat_2d, store_2d_at
from .builtins import alloc_tcm_2d, pack_2d_t_into, proton_mark
from .builtins import vconfig, vzero, vload, vmacc, vreduce_sum, vstore, alloc, vpack, vmadot
from .builtins import f16, f32, bf16
from .builtins import range as range  # noqa: A001 (shadows builtin intentionally)

# raw_kernel: bare decorator alias for @spine_raw(name="linalg") to match the
# feishu 3.3 surface (`@tle.raw_kernel`).
raw_kernel = spine_raw(name="linalg")

__all__ = [
    "spine_raw",
    "raw_kernel",
    "SpineLinalgJITFunction",
    "In",
    "InOut",
    "mem",
    "index",
    "call",
    "batch_macc",
    "view_2d",
    "load_2d",
    "alloc_tcm_2d",
    "pack_2d_t_into",
    "proton_mark",
    "splat_2d",
    "store_2d_at",
    "vconfig",
    "vzero",
    "vload",
    "vmacc",
    "vreduce_sum",
    "vstore",
    "alloc",
    "vpack",
    "vmadot",
    "f16",
    "f32",
    "bf16",
    "range",
]
