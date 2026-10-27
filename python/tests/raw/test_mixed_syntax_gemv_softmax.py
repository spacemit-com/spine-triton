"""Mixed-syntax composition — two-stage GEMV → Softmax pipeline.

(PLAN_mixed_syntax_composition.md §121-167 "分层调用 / Host-Level Composition")

One @triton.jit host orchestrates TWO smt_rvisa sub-kernels that pass an
intermediate result through a shared memory buffer:

  stage 1  gemv_stage    : scores[n] = sum_k Mat[n,k] * vec[k]      (→ scores buf)
  stage 2  softmax_stage : out[n]    = softmax(scores)[n]           (scores → out)

Both _sr_call sites are inlined as sequential `tle.dsl_region` ops into the same
host body (call_registry.py:99-109). Stage 2 reads the buffer stage 1 wrote —
program-serial, compile-time inlined, zero call/dispatch overhead (PLAN §144-148,
§347 "通过 memory 传递中间结果"). grid=1: single program does both stages.

This is the composable half of the PLAN. NOTE (fail-loud, AGENT.md §8.1 / §10):
an LLVM-direct sub-kernel CANNOT be composed this way — its emitter replaces the
whole module, discarding the host body and any other dsl_region. So both stages
here are smt_rvisa (dsl_region path), which genuinely inlines and composes.
"""
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
import triton.language.extra.smt_rvisa as sr
from triton.language.extra.smt_rvisa import call as _sr_call

f16 = sr.f16
f32 = sr.f32


# ── stage 1: GEMV (smt_rvisa) — Mat @ vec → scores ─────────────────────────
# Mat/vec f16, acc f32: sr.vmacc IS the widening vfwmacc (f16×f16→f32), the
# K3-proven idiom. f32 vmacc builds a 2048-bit vector<64xf32> fma that mis-tiles
# for K>VL. `scores` stays f32 — softmax stage 2 reads it as f32.
@sr.raw_kernel
def gemv_stage(Mat: sr.mem(f16), vec: sr.mem(f16), scores: sr.mem(f32, out=True), K: sr.index, N: sr.index):
    nvl = sr.vconfig(-1, 1)
    Kfloor = (K // nvl) * nvl
    for n in sr.range(0, N, 1):
        acc = sr.vzero(f32)
        for ki in sr.range(0, Kfloor, nvl):
            vm = sr.vload(Mat, n * K + ki)  # f16 (vload default)
            vv = sr.vload(vec, ki)
            acc = sr.vmacc(acc, vm, vv)  # widening f16×f16→f32
        for ki in sr.range(Kfloor, K, nvl):
            nvl = sr.vconfig(K - ki, 1)
            tm = sr.vload(Mat, n * K + ki)
            tv = sr.vload(vec, ki)
            acc = sr.vmacc(acc, tm, tv)
        sr.sstore(scores, n, sr.vreduce_sum(acc))


# ── stage 2: Softmax (smt_rvisa) — scores → out, stable via max-subtract ───
@sr.raw_kernel
def softmax_stage(scores: sr.mem(f32), out: sr.mem(f32, out=True), N: sr.index):
    nvl = sr.vconfig(-1, 1)
    Nfloor = (N // nvl) * nvl

    # pass 1: max
    acc_max = sr.vload(scores, 0, dtype=f32)
    for i in sr.range(0, Nfloor, nvl):
        va = sr.vload(scores, i, dtype=f32)
        acc_max = sr.vmax(acc_max, va)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        ta = sr.vload(scores, i, dtype=f32)
        acc_max = sr.vmax(acc_max, ta)
    xmax = sr.vreduce_max(acc_max)

    # pass 2: sum(exp(x - max))
    acc_sum = sr.vzero(f32)
    for i in sr.range(0, Nfloor, nvl):
        vb = sr.vload(scores, i, dtype=f32)
        acc_sum = acc_sum + sr.vexp(vb - xmax)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tb = sr.vload(scores, i, dtype=f32, fill=-1e38)  # padded lanes → exp≈0
        acc_sum = acc_sum + sr.vexp(tb - xmax)
    denom = sr.vreduce_sum(acc_sum)

    # pass 3: exp(x - max) / denom
    inv_denom = 1.0 / denom
    for i in sr.range(0, Nfloor, nvl):
        vc = sr.vload(scores, i, dtype=f32)
        sr.vstore(out, i, sr.vexp(vc - xmax) * inv_denom)
    for i in sr.range(Nfloor, N, nvl):
        sr.vconfig(N - i, 1)
        tc = sr.vload(scores, i, dtype=f32)
        sr.vstore(out, i, sr.vexp(tc - xmax) * inv_denom)


# ── Triton host: compose stage1 → stage2 through `scores` buffer ───────────
@triton.jit(do_not_specialize=["K", "N"])
def gemv_softmax_host(Mat, vec, scores, out, K, N):
    # stage 1: Mat @ vec → scores   (smt_rvisa dsl_region #1, inlined)
    _sr_call(gemv_stage, outputs=[], inputs=[Mat, vec, scores, K, N])
    # stage 2: softmax(scores) → out (smt_rvisa dsl_region #2, inlined; reads #1's output)
    _sr_call(softmax_stage, outputs=[], inputs=[scores, out, N])


def _run(N, K):
    torch.manual_seed(1)
    Mat = torch.randn(N, K, dtype=torch.float16)  # f16 GEMV inputs (widening vmacc)
    vec = torch.randn(K, dtype=torch.float16)
    scores = torch.zeros(N, dtype=torch.float32)  # f32 intermediate buffer (host-allocated)
    out = torch.zeros(N, dtype=torch.float32)
    gemv_softmax_host[(1, )](Mat.contiguous().reshape(-1), vec.contiguous(), scores, out, K, N)
    # golden from the SAME f16-rounded GEMV inputs; softmax over f32 scores
    ref = torch.softmax(torch.mv(Mat.float(), vec.float()), dim=0)
    max_diff = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), \
        f"N={N} K={K} max_diff={max_diff:.4e}"
    return max_diff


_SHAPES = [(64, 64), (128, 128), (100, 65), (200, 130)]


@pytest.mark.parametrize("N, K", _SHAPES)
def test_mixed_gemv_softmax(N, K):
    _run(N, K)


if __name__ == "__main__":
    print("=== Mixed-syntax two-stage: GEMV → Softmax (composed in one host) ===")
    all_ok = True
    for N, K in _SHAPES:
        try:
            md = _run(N, K)
            print(f"PASS  N={N:3d} K={K:3d}  max_diff={md:.4e}")
        except Exception as e:
            all_ok = False
            print(f"FAIL  N={N:3d} K={K:3d}  {type(e).__name__}: {str(e)[:100]}")
    print("ALL_PASS" if all_ok else "HAS_FAILURES")
