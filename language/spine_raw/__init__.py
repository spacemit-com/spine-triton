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
from .builtins import proton_mark
from .builtins import vconfig, vzero, vload, vmacc, vreduce_sum, vreduce_max, vreduce_min, vreduce_mul, vstore, alloc, pack, vpack, vmadot, vshape, vbroadcast, spread
from .builtins import vmin, vmax, sqrt, rsqrt, vexp, vlog, sload, sstore, viota, abs, cast, select  # §6.4 elementwise + transcendental
from .builtins import call_intrinsic, llvm_poison, llvm_const, llvm_base_ptr, llvm_gep, llvm_size  # LLVM-dialect llvm-direct
from .builtins import f16, f32, bf16
from .builtins import mma_cube
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
    "proton_mark",
    "vconfig",
    "vzero",
    "vload",
    "vmacc",
    "vreduce_sum",
    "vreduce_max",
    "vreduce_min",
    "vreduce_mul",
    "vstore",
    "alloc",
    "pack",
    "vpack",
    "vmadot",
    "vshape",
    "vbroadcast",
    "spread",
    "vmin",
    "vmax",
    "sqrt",
    "rsqrt",
    "vexp",
    "vlog",
    "sload",
    "sstore",
    "viota",
    "abs",
    "cast",
    "select",
    "call_intrinsic",
    "llvm_poison",
    "llvm_const",
    "llvm_base_ptr",
    "llvm_gep",
    "llvm_size",
    "f16",
    "f32",
    "bf16",
    "mma_cube",
    "range",
]
