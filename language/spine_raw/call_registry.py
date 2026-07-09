from __future__ import annotations
from triton.language.core import builtin


@builtin
def call(fn, outputs=None, inputs=None, _semantic=None):
    """Inside @triton.jit: emit tle.dsl_region TTIR op holding the raw kernel body.

    The linalg body is generated as MLIR text at trace time and parsed in-process
    by create_tle_dsl_region into tle.dsl_region's real region (no serialized
    raw_linalg string attr — keeps the TTIR readable). The C++ DSLRegionOpPattern
    (TLEToLinalg) then clones that region into spine_ext.raw_region during
    --triton-to-linalg-experimental.
    """
    if inputs is None:
        inputs = []
    linalg_text = fn.make_linalg()
    _semantic.builder.create_tle_dsl_region(fn.__name__, linalg_text, [v.handle for v in inputs])
