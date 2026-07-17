from __future__ import annotations


def call(fn, outputs=None, inputs=None, _semantic=None):
    """Inside @triton.jit: emit tle.dsl_region TTIR op holding the raw kernel body.

    Uses create_tle_dsl_region_direct — body_builder builds ops via C++ builder API
    with no MLIR text round trip.

    When _semantic is None (interpreter mode / outside JIT) the call is a no-op.
    """
    if _semantic is None:
        return
    if inputs is None:
        inputs = []
    param_type_strs, body_builder = fn.make_body_builder()
    builder = _semantic.builder
    param_types = [builder.parse_type(s) for s in param_type_strs]
    builder.create_tle_dsl_region_direct(
        fn.__name__,
        [v.handle for v in inputs],
        param_types,
        body_builder,
    )


# Mark as triton builtin so JIT AST visitor injects _semantic automatically.
call.__triton_builtin__ = True
