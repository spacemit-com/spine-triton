# PROTON_0UTPUT=/mnt_aเิ_ws2/zuoweixia/mv_m512_bm8_pn64.json python3 test_mv_proton_scopes.py 512 4096 8 64

import sys
import torch
import triton
import triton.language as tl
from triton.language.extra.spine_raw import spine_raw, In, InOut, call as _sr_call
import triton.language.extra.spine_raw as sr_mod
from flag_gems.runtime import torch_device_fn

_BLOCK_M = 32
_PACK_N = 64


@spine_raw(name="linalg")
def mv(B: In["memref<*xf16, #ptr.generic_space>"], A: In["memref<*xf16, #ptr.generic_space>"], col: In["index"],
       M: In["index"], nk: In["index"], C: InOut["memref<*xf32, #ptr.generic_space>"]):
    sr_mod.proton_mark("alloc", True)
    b0 = sr_mod.alloc_tcm_2d(32, 64, "f16")
    b1 = sr_mod.alloc_tcm_2d(32, 64, "f16")
    b2 = sr_mod.alloc_tcm_2d(32, 64, "f16")
    b3 = sr_mod.alloc_tcm_2d(32, 64, "f16")
    sr_mod.proton_mark("alloc", False)
    sr_mod.proton_mark("splat", True)
    a0 = sr_mod.splat_2d(0.0, 1, 64, "f32")
    a1 = sr_mod.splat_2d(0.0, 1, 64, "f32")
    a2 = sr_mod.splat_2d(0.0, 1, 64, "f32")
    a3 = sr_mod.splat_2d(0.0, 1, 64, "f32")
    sr_mod.proton_mark("splat", False)
    for kb in sr_mod.range(nk):
        koff = kb * 32
        lhs = sr_mod.view_2d(B, 1, 32, "f16", koff)
        sr_mod.proton_mark("pack", True)
        sr_mod.pack_2d_t_into(b0, A, col, 32, 64, M, "f16", koff)
        sr_mod.pack_2d_t_into(b1, A, col + 64, 32, 64, M, "f16", koff)
        sr_mod.pack_2d_t_into(b2, A, col + 128, 32, 64, M, "f16", koff)
        sr_mod.pack_2d_t_into(b3, A, col + 192, 32, 64, M, "f16", koff)
        sr_mod.proton_mark("pack", False)
        sr_mod.proton_mark("load", True)
        r0 = sr_mod.load_2d(b0, 32, 64, "f16")
        r1 = sr_mod.load_2d(b1, 32, 64, "f16")
        r2 = sr_mod.load_2d(b2, 32, 64, "f16")
        r3 = sr_mod.load_2d(b3, 32, 64, "f16")
        sr_mod.proton_mark("load", False)
        sr_mod.proton_mark("macc", True)
        a0 = sr_mod.batch_macc(lhs, r0, a0)
        a1 = sr_mod.batch_macc(lhs, r1, a1)
        a2 = sr_mod.batch_macc(lhs, r2, a2)
        a3 = sr_mod.batch_macc(lhs, r3, a3)
        sr_mod.proton_mark("macc", False)
    sr_mod.proton_mark("store", True)
    sr_mod.store_2d_at(C, col, 1, 64, a0)
    sr_mod.store_2d_at(C, col + 64, 1, 64, a1)
    sr_mod.store_2d_at(C, col + 128, 1, 64, a2)
    sr_mod.store_2d_at(C, col + 192, 1, 64, a3)
    sr_mod.proton_mark("store", False)


@triton.jit(do_not_specialize=["M", "NK"])
def host(B, A, C, M, NK, NB: tl.constexpr):
    pid = tl.program_id(0)
    _sr_call(mv, outputs=[], inputs=[B, A, pid * NB, M, NK, C])


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

    total_nb = 4 * _PACK_N
    nk = M // _BLOCK_M

    a = torch.randn(N, M, dtype=torch.float16).contiguous()
    b = torch.randn(M, dtype=torch.float16)
    c = torch.empty(N, device=a.device, dtype=torch.float32)
    for _ in range(30):
        with torch_device_fn.device(a.device):
            host[(N // total_nb, )](b, a, c, M, nk, NB=total_nb)

    try:
        from triton.backends.spine_triton.proton import profiler
        profiler.dump()
    except Exception as e:
        print("dump failed:", e)
