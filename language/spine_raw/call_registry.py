from __future__ import annotations


def _to_handle(v, builder, param_type_str: str):
    """Convert a JIT input value to an MLIR Value handle.

    tl.constexpr (triton specializes small ints, e.g. P=1, as constexpr
    inside @triton.jit) has no .handle; emit arith.constant instead.
    """
    if hasattr(v, "handle"):
        return v.handle
    # tl.constexpr case
    val = v.value if hasattr(v, "value") else v
    if isinstance(val, int):
        if param_type_str == "index":
            return builder.create_arith_constant_index(val)
        # fallback: i32/i64 — use index_cast path
        return builder.create_arith_constant_index(val)
    if isinstance(val, float):
        import re
        bits = int(re.search(r"f(\d+)", param_type_str).group(1)) if re.search(r"f(\d+)", param_type_str) else 32
        ft = builder.get_f32_type() if bits <= 32 else builder.parse_type("f64")
        return builder.create_arith_constant_float(val, ft)
    raise TypeError(f"spine_raw.call: cannot convert constexpr {val!r} (type {param_type_str!r}) to IR handle")


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
    handles = [_to_handle(v, builder, pt) for v, pt in zip(inputs, param_type_strs)]
    builder.create_tle_dsl_region_direct(
        fn.__name__,
        handles,
        param_types,
        body_builder,
    )


# Mark as triton builtin so JIT AST visitor injects _semantic automatically.
call.__triton_builtin__ = True

