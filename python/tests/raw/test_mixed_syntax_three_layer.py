"""Mixed-syntax composition — THREE syntax layers in one pipeline.

(PLAN_mixed_syntax_composition.md — extends the two-layer composition to cover
all three lowering routes spine-triton exposes.)

A single pipeline `out = (Mat @ (vec * alpha)) * beta` split so each stage is
written in a DIFFERENT syntax layer:

  stage 1  pre_scale_tl   : vec_s[k] = vec[k] * alpha      —— 普通 tl 语法
  stage 2  gemv_spine_raw : scores[n] = Σ_k Mat[n,k]*vec_s —— 普通 spine_raw
  stage 3  post_scale_llvm: out[n]    = scores[n] * beta   —— call_intrinsic (LLVM-direct)

Why THREE separate launches (not one inlined host):
  • stage 1 (pure @triton.jit) lowers through the normal triton→linalg path.
  • stage 2 (@tle.raw_kernel, dsl_region) inlines into a triton host as a
    `tle.dsl_region` op (call_registry.py:99-109) — the composable path.
  • stage 3 uses `call_intrinsic` → the LLVM-direct emitter, which REPLACES the
    whole module with a standalone top-level `llvm.func` (AGENT.md §8, §8.1).
    It therefore CANNOT be inlined next to a dsl_region — it must be its own
    launch. The three stages compose through shared torch buffers instead
    (program-serial, results handed stage→stage via memory).

Shape constraints: N % 8 == 0 (stage 3 vle/vse fixed VL=8, no tail); K arbitrary
(stage 2 spine_raw handles the K tail; stage 1 tl masks its tail).
"""
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import call as _sr_call

f16 = tle.f16
f32 = tle.f32

_BETA = 0.5   # baked into the LLVM-direct stage as a constant vector splat


# ── 层级 1: 普通 tl 语法 —— elementwise pre-scale vec_s = vec * alpha ────────
# Standard Triton: program-per-block, tl.arange + masked load/store. alpha is a
# runtime f32 scalar; output kept f16 so stage 2's widening vmacc sees f16×f16.
@triton.jit
def pre_scale_tl(vec_ptr, vec_s_ptr, alpha, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(vec_ptr + offs, mask=mask, other=0.0)
    y = (x.to(tl.float32) * alpha).to(tl.float16)
    tl.store(vec_s_ptr + offs, y, mask=mask)


# ── 层级 2: 普通 spine_raw (dsl_region) —— GEMV scores = Mat @ vec_s ─────────
# Mat/vec_s f16, acc f32: tle.vmacc IS the widening vfwmacc (f16×f16→f32), the
# K3-proven idiom. Tail loop handles arbitrary K.
@tle.raw_kernel
def gemv_spine_raw(Mat: tle.mem(f16), vec_s: tle.mem(f16), scores: tle.mem(f32, out=True),
                   K: tle.index, row_base: tle.index, row_end: tle.index):
    nvl = tle.vconfig(-1, 1)              # f16 lmul=1 → VLMAX=64
    Kfloor = (K // nvl) * nvl
    for n in tle.range(row_base, row_end, 1):
        acc = tle.vzero(f32)
        # Main loop: process full vectors
        for ki in tle.range(0, Kfloor, nvl):
            vm = tle.vload(Mat, n * K + ki)
            vv = tle.vload(vec_s, ki)
            acc = tle.vmacc(acc, vm, vv)
        # Tail loop: step=nvl, but reconfigure inside (style from test_raw_mv_svector.py)
        for ki in tle.range(Kfloor, K, nvl):
            nvl = tle.vconfig(K - ki, 1)  # Reconfigure for tail length
            tm = tle.vload(Mat, n * K + ki)
            tv = tle.vload(vec_s, ki)
            acc = tle.vmacc(acc, tm, tv)
        tle.sstore(scores, n, tle.vreduce_sum(acc))


@triton.jit(do_not_specialize=["K", "N"])
def gemv_host(Mat, vec_s, scores, K, N, BLOCK: tl.constexpr):
    # Use row_base/row_end pattern to avoid constant specialization
    pid = tl.program_id(0)
    row_base = pid * BLOCK
    row_end = min(row_base + BLOCK, N)
    _sr_call(gemv_spine_raw, outputs=[], inputs=[Mat, vec_s, scores, K, row_base, row_end])


# ── 层级 3: call_intrinsic (LLVM-direct) —— post-scale out = scores * beta ───
# vle → fmul(by beta splat) → vse. LLVM-direct = standalone llvm.func module, so
# this is its OWN launch (cannot inline beside a dsl_region). N % 8 == 0 → VL=8
# tiles cover N exactly, no tail. grid=1: single program strides the whole N.
@tle.raw_kernel
def post_scale_llvm(scores: tle.mem(f32), out: tle.mem(f32, out=True), N: tle.index):
    vl = tle.llvm_const(8, "i64")
    zero = tle.llvm_const(0, "i64")
    beta = tle.llvm_const("5.000000e-01", "vector<[8]xf32>")   # 0.5 splat
    for i in tle.range(zero, N, vl):
        p = tle.llvm_poison("vector<[8]xf32>")
        gs = tle.llvm_gep(tle.llvm_base_ptr(scores), i, "f32")
        v = tle.call_intrinsic("llvm.riscv.vle", [p, gs, vl], result_type="vector<[8]xf32>")
        r = tle.call_intrinsic("llvm.fmul", [v, beta], result_type="vector<[8]xf32>")
        go = tle.llvm_gep(tle.llvm_base_ptr(out), i, "f32")
        tle.call_intrinsic("llvm.riscv.vse", [r, go, vl], result_type="()")


@triton.jit
def post_scale_host(scores, out, N):
    _sr_call(post_scale_llvm, outputs=[], inputs=[scores, out, N])


def _run(N, K, alpha=1.5, BLOCK=64):
    assert N % 8 == 0, "stage 3 (llvm-direct vle/vse) needs N % 8 == 0"
    torch.manual_seed(0)
    Mat = torch.randn(N, K, dtype=torch.float16)
    vec = torch.randn(K, dtype=torch.float16)
    vec_s = torch.zeros(K, dtype=torch.float16)     # stage1 → stage2 buffer
    scores = torch.zeros(N, dtype=torch.float32)    # stage2 → stage3 buffer
    out = torch.zeros(N, dtype=torch.float32)

    # stage 1: pure tl elementwise pre-scale
    grid1 = ((K + BLOCK - 1) // BLOCK,)
    pre_scale_tl[grid1](vec.contiguous(), vec_s, alpha, K, BLOCK=BLOCK)
    # stage 2: spine_raw GEMV (Mat @ vec_s), grid=1 with BLOCK=N
    gemv_host[(1,)](Mat.contiguous().reshape(-1), vec_s, scores, K, N, BLOCK=N)
    # stage 3: llvm-direct post-scale (scores * beta), grid=1
    post_scale_host[(1,)](scores, out, N)

    # golden from the SAME f16-rounded inputs each stage actually reads
    ref = torch.mv(Mat.float(), (vec.float() * alpha).half().float()) * _BETA
    max_diff = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-1), \
        f"N={N} K={K} alpha={alpha} max_diff={max_diff:.4e}"
    return max_diff


_SHAPES = [(8, 64), (16, 128), (32, 100), (8, 65), (24, 130)]


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mixed_three_layer(N, K):
    _run(N, K)


if __name__ == "__main__":
    print("=== Mixed-syntax THREE layers: tl → spine_raw → call_intrinsic ===")
    all_ok = True
    for N, K in _SHAPES:
        try:
            md = _run(N, K)
            print(f"PASS  N={N:3d} K={K:3d}  max_diff={md:.4e}")
        except Exception as e:
            all_ok = False
            print(f"FAIL  N={N:3d} K={K:3d}  {type(e).__name__}: {str(e)[:100]}")
    print("ALL_PASS" if all_ok else "HAS_FAILURES")
