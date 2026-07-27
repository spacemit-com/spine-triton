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


# LLVM-direct handoff: the @triton.jit body (which calls call()) runs during
# make_ir, strictly before the "ttir" stage's make_ttir reads it. Triton
# compiles one kernel at a time, so a process-global holder is a safe, C++-free
# channel from call() → make_ttir (avoids binding get_module/set_attr, which
# don't exist in this libtriton API and would need a full riscv64 rebuild).
_PENDING_LLVM_DIRECT_MODULE: dict[str, any] = {}


def take_pending_llvm_direct_module(kernel_name: str = None):
    """make_ttir calls this to retrieve + clear a pending llvm-direct module.

    Returns (module_text, emitted_symbol_name), or (None, None) if the
    just-compiled kernel was not a llvm-direct kernel.
    """
    text = _PENDING_LLVM_DIRECT_MODULE.pop("text", None)
    name = _PENDING_LLVM_DIRECT_MODULE.pop("name", None)
    return text, name


def take_pending_llvm_funcs():
    """make_ttir calls this to retrieve + clear pending llvm.func siblings for mixed mode.

    Returns list of llvm.func text strings (no module wrapper), or empty list.
    Used when host contains both normal ops and llvm-direct calls.
    """
    funcs = _PENDING_LLVM_DIRECT_MODULE.pop("llvm_funcs", [])
    return funcs


def take_pending_llvm_calls():
    """Retrieve + clear pending mixed-mode llvm.call bridge specs.

    Returns a list of {"callee": str, "arg_bridge": [{"pos": int,
    "kind": "ptr"|"scalar"}, ...]}, one per _sr_call to an llvm-direct kernel
    inside a host that also does other work. `pos` is the host func.func
    argument index; `kind` selects the bridge (memref→data-ptr-i64 vs i32→i64).
    Empty list if the just-compiled kernel had no mixed llvm-direct calls.
    """
    return _PENDING_LLVM_DIRECT_MODULE.pop("llvm_calls", [])


def call(fn, outputs=None, inputs=None, _semantic=None):
    """Inside @triton.jit: emit tle.dsl_region TTIR op holding the raw kernel body.

    Uses create_tle_dsl_region_direct — body_builder builds ops via C++ builder API
    with no MLIR text round trip.

    LLVM-direct bypass: if fn has _llvm_direct=True, emit llvm.func module text and pass it
    via module attr to metadata (compiler.py reads it in make_ttir).

    When _semantic is None (interpreter mode / outside JIT) the call is a no-op.
    """
    if _semantic is None:
        return
    if inputs is None:
        inputs = []

    # LLVM-direct bypass: detect and emit
    if getattr(fn, '_llvm_direct', False):
        from .llvm_direct_text import emit_llvm_func_for_inline
        from .codegen import _parse_signature
        # emit_llvm_func_for_inline needs the raw Python function, not the JIT wrapper
        raw_fn = fn._fn if hasattr(fn, '_fn') else fn

        # Arity guard: inputs must match kernel signature
        n_params = len(_parse_signature(raw_fn))
        if len(inputs) != n_params:
            raise ValueError(
                f"spine_raw.call: LLVM-direct kernel {raw_fn.__name__!r} declares "
                f"{n_params} parameter(s) but got {len(inputs)} input(s).")

        # Emit the sibling llvm.func (no module wrapper). param_types is the
        # sibling ABI (every param → i64: memref=data-ptr-as-i64, scalar=i64).
        llvm_func_text, param_types = emit_llvm_func_for_inline(raw_fn)
        if "llvm_funcs" not in _PENDING_LLVM_DIRECT_MODULE:
            _PENDING_LLVM_DIRECT_MODULE["llvm_funcs"] = []
        _PENDING_LLVM_DIRECT_MODULE["llvm_funcs"].append(llvm_func_text)

        # Mixed-mode host bridge is emitted by compiler.py at the *linalgdir*
        # stage (func.func form), where memrefs exist and llvm.call is legal —
        # not here at TTIR (tt.ptr, no bridge ops). We can't reference the host
        # func.func's SSA args from here, but they map 1:1 by POSITION to the
        # host's entry-block args (verified: tt.func user params → func.func
        # %arg0.. in the same order). So record each input's host-arg index.
        builder = _semantic.builder
        entry = builder.get_insertion_block()
        n_block_args = entry.get_num_arguments()
        # NOTE: .id is a bound method on this libtriton build (pybind11), not a
        # property — call it. Using the method object as a dict key silently never
        # matches, so every input would look like a non-host-arg. (K3-verified.)
        argid_to_pos = {entry.get_argument(i).id(): i for i in range(n_block_args)}

        params = _parse_signature(raw_fn)  # [(pname, ann), ...]
        arg_bridge = []  # per-input: {"pos": int, "kind": "ptr"|"scalar"}
        for (pname, ann), v in zip(params, inputs):
            if not hasattr(v, "handle"):
                raise ValueError(
                    f"spine_raw.call: LLVM-direct kernel {raw_fn.__name__!r} in "
                    f"mixed mode requires every input to be a host launch arg "
                    f"(a tt.func parameter); got a computed/constexpr value for "
                    f"{pname!r}. Compute derived values INSIDE the kernel from "
                    f"tle.program_id(axis).")
            pos = argid_to_pos.get(v.handle.id())
            if pos is None:
                raise ValueError(
                    f"spine_raw.call: input for {pname!r} of {raw_fn.__name__!r} "
                    f"is not a host entry-block argument. In mixed mode inputs must "
                    f"be the host's own launch parameters (bridged to the sibling "
                    f"llvm.func by position at the linalgdir stage).")
            kind = "ptr" if ann.mlir_type.startswith("memref") else "scalar"
            arg_bridge.append({"pos": pos, "kind": kind})

        if "llvm_calls" not in _PENDING_LLVM_DIRECT_MODULE:
            _PENDING_LLVM_DIRECT_MODULE["llvm_calls"] = []
        _PENDING_LLVM_DIRECT_MODULE["llvm_calls"].append({
            "callee": raw_fn.__name__,
            "arg_bridge": arg_bridge,   # ordered per sibling param
        })

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

