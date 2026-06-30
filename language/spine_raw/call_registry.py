from __future__ import annotations
from triton.language.core import builtin


@builtin
def call(fn, outputs=None, inputs=None, _semantic=None):
    """Inside @triton.jit: emit tle.dsl_region TTIR op with full raw_linalg text.

    The linalg body is generated at trace time and embedded in the op's
    raw_linalg attr, so the C++ DSLRegionOpPattern (TLEToLinalg) can parse it
    and build spine_ext.raw_region during --triton-to-linalg-experimental.
    Mirrors FlagTree's tle_raw.call() which embeds LLVM text in DSLRegionOp.
    """
    if inputs is None:
        inputs = []
    linalg_text = fn.make_linalg()
    _semantic.builder.create_tle_dsl_region(
        fn.__name__, linalg_text, [v.handle for v in inputs])
