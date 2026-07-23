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


# Mode-1 handoff: the @triton.jit body (which calls call()) runs during
# make_ir, strictly before the "ttir" stage's make_ttir reads it. Triton
# compiles one kernel at a time, so a process-global holder is a safe, C++-free
# channel from call() → make_ttir (avoids binding get_module/set_attr, which
# don't exist in this libtriton API and would need a full riscv64 rebuild).
_PENDING_MODE1_MODULE: dict[str, str] = {}


def take_pending_mode1_module(kernel_name: str = None):
    """make_ttir calls this to retrieve + clear a pending mode-1 module.

    Returns (module_text, emitted_symbol_name), or (None, None) if the
    just-compiled kernel was not a mode-1 kernel.
    """
    text = _PENDING_MODE1_MODULE.pop("text", None)
    name = _PENDING_MODE1_MODULE.pop("name", None)
    return text, name


def call(fn, outputs=None, inputs=None, _semantic=None):
    """Inside @triton.jit: emit tle.dsl_region TTIR op holding the raw kernel body.

    Uses create_tle_dsl_region_direct — body_builder builds ops via C++ builder API
    with no MLIR text round trip.

    Mode-1 bypass: if fn has _mode1=True, emit llvm.func module text and pass it
    via module attr to metadata (compiler.py reads it in make_ttir).

    When _semantic is None (interpreter mode / outside JIT) the call is a no-op.
    """
    if _semantic is None:
        return
    if inputs is None:
        inputs = []

    # Mode-1 bypass: detect and emit
    if getattr(fn, '_mode1', False):
        from .mode1_text import emit_mode1_module
        # emit_mode1_module needs the raw Python function, not the JIT wrapper
        raw_fn = fn._fn if hasattr(fn, '_fn') else fn
        llvm_module_text = emit_mode1_module(raw_fn)
        # Stash for make_ttir via the process-global holder (no C++ module attr:
        # get_module/set_attr aren't bound in this libtriton API). This runs
        # inside make_ir, before the ttir stage reads it — same-process, same
        # single-kernel compile, so the handoff is safe.
        _PENDING_MODE1_MODULE["text"] = llvm_module_text
        # The emitted llvm.func is named after the raw kernel (raw_fn), NOT the
        # @triton.jit host wrapper. The launcher looks up metadata["name"] as the
        # binary symbol, so make_ttir must override name to the emitted symbol.
        _PENDING_MODE1_MODULE["name"] = raw_fn.__name__
        return  # skip dsl_region emission

    # Normal path
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

