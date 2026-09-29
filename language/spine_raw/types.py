# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""In / InOut type annotations for @spine_raw kernel parameters.

Usage:
    def my_fn(A: In["memref<*xf16, #ptr.generic_space>"], n: In["i32"]):
        ...
"""
from __future__ import annotations


class _TypedAnnotation:
    """Base for In/InOut; carries the MLIR type string.

    ``kind`` is fixed once, at construction: "mem" for memref pointer params,
    "scalar" otherwise. Consumers must branch on ``kind`` instead of re-sniffing
    the ``mlir_type`` text. Free-form ``In["..."]`` annotations derive ``kind``
    here (the single place that reads the annotation grammar); the ``mem()`` /
    ``index`` sugars pass it explicitly.
    """

    def __init__(self, mlir_type: str, writable: bool, kind: str | None = None):
        self.mlir_type = mlir_type
        self.writable = writable
        if kind is None:
            kind = "mem" if mlir_type.startswith("memref") else "scalar"
        if kind not in ("mem", "scalar"):
            raise ValueError(f"invalid annotation kind {kind!r} (expected 'mem' or 'scalar')")
        self.kind = kind

    def __repr__(self) -> str:
        cls = "InOut" if self.writable else "In"
        return f"{cls}[{self.mlir_type!r}]"


class In:
    """Read-only input parameter. Maps to the given MLIR type (no return)."""

    def __class_getitem__(cls, mlir_type: str) -> _TypedAnnotation:
        return _TypedAnnotation(mlir_type, writable=False)


class InOut:
    """Read-write parameter. The raw function receives it and may mutate it in place.
    For SSA-clean MLIR the caller passes a memref that the function writes into."""

    def __class_getitem__(cls, mlir_type: str) -> _TypedAnnotation:
        return _TypedAnnotation(mlir_type, writable=True)


# ---------------------------------------------------------------------------
# Document-facing sugar (feishu 3.3): tle.mem(f16) / tle.mem(f32, out=True) /
# tle.index. These produce the same In/InOut annotations used by @spine_raw.
# ---------------------------------------------------------------------------
def mem(dtype: str, out: bool = False) -> _TypedAnnotation:
    """Pointer-parameter annotation for a raw kernel.

    tle.mem(f16)             -> In["memref<*xf16, #ptr.generic_space>"]
    tle.mem(f32, out=True)   -> InOut["memref<*xf32, #ptr.generic_space>"]
    """
    mlir_type = f"memref<*x{dtype}, #ptr.generic_space>"
    return _TypedAnnotation(mlir_type, writable=out, kind="mem")


# Scalar index parameter annotation, e.g.  K: tle.index
index = _TypedAnnotation("index", writable=False, kind="scalar")


# ---------------------------------------------------------------------------
# Structured MLIR type metadata (Ty hierarchy).
#
# codegen tracks the type of every SSA value as a Ty object instead of an MLIR
# text string: shape/element/layout queries are attribute reads (no regex, no
# substring sniffing), and type text is produced only at the C++ builder
# boundary via ``mlir()`` (fed to ``builder.parse_type``). Constructed types
# never round-trip through text; ``parse_ty`` is the single boundary parser for
# the two places text legitimately enters (free-form ``In["..."]`` annotation
# strings and LLVM-channel user literals).
# ---------------------------------------------------------------------------

# Exact widths for the scalar types the DSL documents; ``ScalarTy.bits``
# extends this with the regular iN/fN/bfN pattern for anything else.
_SCALAR_BITS = {
    "i1": 1, "i4": 4, "i8": 8, "i16": 16, "i32": 32, "i64": 64,
    "f16": 16, "bf16": 16, "f32": 32, "f64": 64,
}
_FLOAT_SCALARS = {"f16", "bf16", "f32", "f64"}


def _is_scalar_name(tok: str) -> bool:
    """index / iN / fN / bfN — the builtin scalar type names parse_ty accepts."""
    if tok == "index":
        return True
    for prefix in ("bf", "f", "i"):
        if tok.startswith(prefix) and tok[len(prefix):].isdigit():
            return True
    return False


def _dim_text(d) -> str:
    return "?" if d is None else str(d)


class Ty:
    """Base of the structured type hierarchy. ``mlir()`` prints the MLIR text."""

    __slots__ = ()

    def mlir(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.mlir()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.mlir()!r})"


class ScalarTy(Ty):
    """A builtin scalar type: ``index``, ``i1``..``i64``, ``f16``/``bf16``/``f32``/``f64``."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def mlir(self) -> str:
        return self.name

    @property
    def is_float(self) -> bool:
        if self.name in _FLOAT_SCALARS:
            return True
        n = self.name
        if n.startswith("bf") and n[2:].isdigit():
            return True
        return n.startswith("f") and n[1:].isdigit()

    @property
    def bits(self) -> int:
        if self.name in _SCALAR_BITS:
            return _SCALAR_BITS[self.name]
        for prefix in ("bf", "f", "i"):
            if self.name.startswith(prefix) and self.name[len(prefix):].isdigit():
                return int(self.name[len(prefix):])
        raise KeyError(f"no bit width for scalar type {self.name!r}")

    def __eq__(self, other) -> bool:
        return isinstance(other, ScalarTy) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("ScalarTy", self.name))


class VecTy(Ty):
    """``vector<D0xD1x...xT>``.

    ``dims`` entries are ints (static, the only form codegen constructs) or
    preformatted text tokens ("[8]" scalable, "?" dynamic) which only ever come
    from parsed LLVM-channel user literals and are never inspected arithmetically.
    ``elem`` is the element Ty (a ScalarTy for everything the DSL constructs).
    """

    __slots__ = ("dims", "elem")

    def __init__(self, dims, elem: Ty):
        self.dims = tuple(dims)
        self.elem = elem

    def mlir(self) -> str:
        return "vector<" + "x".join(_dim_text(d) for d in self.dims) + "x" + self.elem.mlir() + ">"

    @property
    def first_dim(self) -> int:
        d = self.dims[0]
        if not isinstance(d, int):
            raise ValueError(f"Cannot extract a static size from {self.mlir()!r}")
        return d

    def __eq__(self, other) -> bool:
        return isinstance(other, VecTy) and other.dims == self.dims and other.elem == self.elem

    def __hash__(self) -> int:
        return hash(("VecTy", self.dims, self.elem))


class TensorTy(Ty):
    """``tensor<D0xD1x...xT>``; dims entries are ints (static) or None (``?``)."""

    __slots__ = ("dims", "elem")

    def __init__(self, dims, elem: Ty):
        self.dims = tuple(dims)
        self.elem = elem

    def mlir(self) -> str:
        return "tensor<" + "x".join(_dim_text(d) for d in self.dims) + "x" + self.elem.mlir() + ">"

    def __eq__(self, other) -> bool:
        return isinstance(other, TensorTy) and other.dims == self.dims and other.elem == self.elem

    def __hash__(self) -> int:
        return hash(("TensorTy", self.dims, self.elem))


class StridedLayout:
    """``strided<[S0, S1, ...]>`` or ``strided<[...], offset: ?>``.

    ``strides`` entries are ints (static) or None (``?``). ``has_offset`` is
    True for ``offset: ?`` (the only form codegen constructs) or an int for a
    static offset (parse-only).
    """

    __slots__ = ("strides", "has_offset")

    def __init__(self, strides, has_offset=False):
        self.strides = tuple(strides)
        self.has_offset = has_offset

    def mlir(self) -> str:
        inner = ", ".join(_dim_text(s) for s in self.strides)
        if self.has_offset is True:
            off = ", offset: ?"
        elif self.has_offset is False:
            off = ""
        else:
            off = f", offset: {self.has_offset}"
        return f"strided<[{inner}]{off}>"

    def __eq__(self, other) -> bool:
        return (isinstance(other, StridedLayout)
                and other.strides == self.strides and other.has_offset == self.has_offset)

    def __hash__(self) -> int:
        return hash(("StridedLayout", self.strides, self.has_offset))

    def __repr__(self) -> str:
        return f"StridedLayout({self.mlir()!r})"


class MemTy(Ty):
    """``memref`` type.

    ``dims`` is None for the unranked form (``memref<*xT[, space]>``, the raw
    pointer-param annotation) or a tuple of ints/None (``?``) for ranked forms.
    ``layout`` is a StridedLayout or None; ``space`` is the memory-space attr
    text (e.g. ``#ptr.generic_space``) or None.
    """

    __slots__ = ("dims", "elem", "layout", "space")

    def __init__(self, dims, elem: Ty, layout: StridedLayout | None = None,
                 space: str | None = None):
        self.dims = None if dims is None else tuple(dims)
        self.elem = elem
        self.layout = layout
        self.space = space

    @property
    def unranked(self) -> bool:
        return self.dims is None

    def mlir(self) -> str:
        if self.dims is None:
            s = "memref<*x" + self.elem.mlir()
        else:
            s = ("memref<" + "x".join(_dim_text(d) for d in self.dims)
                 + "x" + self.elem.mlir())
        if self.layout is not None:
            s += ", " + self.layout.mlir()
        if self.space is not None:
            s += ", " + self.space
        return s + ">"

    def as_dynamic_ranked(self) -> "MemTy":
        """The _ranked_cast transform: memref<*xT, ...> -> memref<?xT, ...>."""
        if not self.unranked:
            raise ValueError(f"as_dynamic_ranked on a ranked type {self.mlir()!r}")
        return MemTy((None,), self.elem, self.layout, self.space)

    def __eq__(self, other) -> bool:
        return (isinstance(other, MemTy) and other.dims == self.dims
                and other.elem == self.elem and other.layout == self.layout
                and other.space == self.space)

    def __hash__(self) -> int:
        return hash(("MemTy", self.dims, self.elem, self.layout, self.space))


class OpaqueTy(Ty):
    """User-literal MLIR type text carried through uninterpreted.

    Only for the LLVM-direct channel, where kernel authors pass type strings
    verbatim ("!llvm.ptr", "!llvm.struct<...>", "vector<[8]xf16>", "()" for
    void). Never constructed by codegen; never inspected beyond passthrough.
    """

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def mlir(self) -> str:
        return self.text

    def __eq__(self, other) -> bool:
        return isinstance(other, OpaqueTy) and other.text == self.text

    def __hash__(self) -> int:
        return hash(("OpaqueTy", self.text))


# Frequently used singletons.
GENERIC_SPACE = "#ptr.generic_space"
INDEX = ScalarTy("index")
I1 = ScalarTy("i1")
I8 = ScalarTy("i8")
I16 = ScalarTy("i16")
I32 = ScalarTy("i32")
I64 = ScalarTy("i64")
F16 = ScalarTy("f16")
BF16 = ScalarTy("bf16")
F32 = ScalarTy("f32")
F64 = ScalarTy("f64")
LLVM_PTR = OpaqueTy("!llvm.ptr")
LLVM_DESC = OpaqueTy("!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>")
VOID = OpaqueTy("()")


# ---------------------------------------------------------------------------
# parse_ty — the single boundary text parser (hand scanner, no regex).
# Grammar subset: scalar | vector<dims xT> | tensor<dims xT> |
#                 memref<(*|dims) xT [, strided<[strides][, offset: ?]>] [, #space]>
# dims: int | "?" (memref/tensor) | "[N]" (vector scalable, parse-only).
# ---------------------------------------------------------------------------

def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _parse_err(s: str, i: int, what: str):
    return ValueError(f"parse_ty: expected {what} at position {i} in {s!r}")


def _parse_scalar_at(s: str, i: int):
    j = i
    while j < len(s) and (s[j].isalnum() or s[j] == "_"):
        j += 1
    tok = s[i:j]
    if not tok or not _is_scalar_name(tok):
        raise _parse_err(s, i, "a scalar type name (index/iN/fN/bfN)")
    return ScalarTy(tok), j


def _parse_dim_at(s: str, i: int):
    """(dim, next_i) where dim is int | None ('?') | str ('[N]' scalable)."""
    if s[i] == "?":
        return None, i + 1
    if s[i] == "[":
        j = s.find("]", i)
        if j < 0 or not s[i + 1:j].isdigit():
            raise _parse_err(s, i, "'[N]' scalable dim")
        return s[i:j + 1], j + 1
    j = i
    while j < len(s) and s[j].isdigit():
        j += 1
    if j == i:
        raise _parse_err(s, i, "a dimension (int, '?' or '[N]')")
    return int(s[i:j]), j


def _parse_dims_then_elem(s: str, i: int):
    """Parse `D0xD1x...xT` (dims then scalar elem); returns (dims, elem, next_i)."""
    dims = []
    while True:
        i = _skip_ws(s, i)
        if s[i].isdigit() or s[i] in "?[":
            d, i = _parse_dim_at(s, i)
            dims.append(d)
            i = _skip_ws(s, i)
            if i >= len(s) or s[i] != "x":
                raise _parse_err(s, i, "'x' after dimension")
            i += 1
        else:
            break
    elem, i = _parse_scalar_at(s, i)
    return tuple(dims), elem, i


def _parse_vector_at(s: str, i: int):
    i += len("vector<")
    dims, elem, i = _parse_dims_then_elem(s, i)
    i = _skip_ws(s, i)
    if i >= len(s) or s[i] != ">":
        raise _parse_err(s, i, "'>' to close vector type")
    return VecTy(dims, elem), i + 1


def _parse_tensor_at(s: str, i: int):
    i += len("tensor<")
    dims, elem, i = _parse_dims_then_elem(s, i)
    i = _skip_ws(s, i)
    if i >= len(s) or s[i] != ">":
        raise _parse_err(s, i, "'>' to close tensor type")
    return TensorTy(dims, elem), i + 1


def _parse_strided_at(s: str, i: int):
    i += len("strided<")
    i = _skip_ws(s, i)
    if i >= len(s) or s[i] != "[":
        raise _parse_err(s, i, "'[' to start stride list")
    i += 1
    strides: list = []
    while True:
        i = _skip_ws(s, i)
        if s[i] == "]":
            i += 1
            break
        if s[i] == "?":
            strides.append(None)
            i += 1
        else:
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            if j == i:
                raise _parse_err(s, i, "a stride (int or '?')")
            strides.append(int(s[i:j]))
            i = j
        i = _skip_ws(s, i)
        if s[i] == ",":
            i += 1
        elif s[i] == "]":
            i += 1
            break
        else:
            raise _parse_err(s, i, "',' or ']' in stride list")
    has_offset: bool | int = False
    i = _skip_ws(s, i)
    if i < len(s) and s[i] == ",":
        i = _skip_ws(s, i + 1)
        if not s.startswith("offset:", i):
            raise _parse_err(s, i, "'offset:' section")
        i = _skip_ws(s, i + len("offset:"))
        if s[i] == "?":
            has_offset = True
            i += 1
        else:
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            if j == i:
                raise _parse_err(s, i, "offset value ('?' or int)")
            has_offset = int(s[i:j])
            i = j
        i = _skip_ws(s, i)
    if i >= len(s) or s[i] != ">":
        raise _parse_err(s, i, "'>' to close strided layout")
    return StridedLayout(strides, has_offset), i + 1


def _parse_memref_at(s: str, i: int):
    i += len("memref<")
    i = _skip_ws(s, i)
    dims = None
    if s[i] == "*":
        i += 1
        if i >= len(s) or s[i] != "x":
            raise _parse_err(s, i, "'x' after '*'")
        i += 1
        elem, i = _parse_scalar_at(s, i)
    else:
        dims, elem, i = _parse_dims_then_elem(s, i)
    layout = None
    space = None
    i = _skip_ws(s, i)
    while i < len(s) and s[i] == ",":
        i = _skip_ws(s, i + 1)
        if s.startswith("strided<", i):
            layout, i = _parse_strided_at(s, i)
        elif s[i] == "#":
            j = i
            while j < len(s) and s[j] not in ",>":
                j += 1
            space = s[i:j].strip()
            i = j
        else:
            raise _parse_err(s, i, "'strided<...>' layout or '#...' memory space")
        i = _skip_ws(s, i)
    if i >= len(s) or s[i] != ">":
        raise _parse_err(s, i, "'>' to close memref type")
    return MemTy(dims, elem, layout, space), i + 1


def parse_ty(text: str) -> Ty:
    """Parse the MLIR type-text subset used by @spine_raw.

    The single boundary parser: annotation strings (``In["..."]``) enter here
    once, everything else is constructed as Ty objects directly. Unknown text
    raises ValueError instead of being guessed at.
    """
    s = text.strip()
    if s.startswith("vector<"):
        ty, i = _parse_vector_at(s, 0)
    elif s.startswith("tensor<"):
        ty, i = _parse_tensor_at(s, 0)
    elif s.startswith("memref<"):
        ty, i = _parse_memref_at(s, 0)
    else:
        ty, i = _parse_scalar_at(s, 0)
    i = _skip_ws(s, i)
    if i != len(s):
        raise ValueError(f"parse_ty: trailing text at position {i} in {text!r}")
    return ty


def parse_ty_or_opaque(text: str) -> Ty:
    """parse_ty with an OpaqueTy fallback — for LLVM-channel user literals
    ("!llvm.ptr", "()", ...) that are carried through uninterpreted."""
    try:
        return parse_ty(text)
    except ValueError:
        return OpaqueTy(text)
