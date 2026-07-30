from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, spine_triton
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import sys
import tempfile
import shutil
import os
import re
import subprocess
import functools
from pathlib import Path
from . import (
    get_spine_triton_opt_path,
    dump_ir_if_needed,
    get_llvm_bin_path,
    get_spine_mlir_opt_path,
    extract_kernel_name,
    get_cpu_name_from_arch_id,
    get_spine_mlir_opt_options,
    get_cpu_arch,
    get_target_arch,
    get_cross_toolchain,
)


def _ttir_to_linalgdir(mod, metadata):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    metadata["smt_parallel_inside"] = ("bind_sub_block = true" in ttir_code)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "linalg.mlir")
        Path(src_path).write_text(ttir_code)
        dump_ir_if_needed([src_path], metadata["name"])
        spine_triton_opt_path = get_spine_triton_opt_path()
        subprocess.check_call([
            spine_triton_opt_path,
            # spine_ext.raw_region (emitted by DSLRegionOpPattern when lowering
            # tle.dsl_region) is an unregistered op here — its dialect lives in
            # spine-mlir's spine-opt downstream. Allow it so the conversion can
            # create it in generic form.
            "--allow-unregistered-dialect",
            src_path,
            "--triton-to-linalg-experimental",
            "-o",
            dst_path,
        ])
        dump_ir_if_needed([dst_path], metadata["name"])
        return Path(dst_path).read_text()


def _optimize_linalgdir(linalgdir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return linalgdir


def _host_func_arg_types(linalgdir: str, func_name: str) -> list[str]:
    """Parse the host func.func arg types (in order) from linalgdir text.

    Returns e.g. ["memref<*xf16, #ptr.generic_space>", ..., "i32", ...].
    Splits the top-level arg list on commas that are NOT nested inside
    <>/{}/() — MLIR types embed commas (strided<[1], offset: ?>).
    """
    m = re.search(rf"func\.func\s+@{re.escape(func_name)}\s*\(", linalgdir)
    if not m:
        raise RuntimeError(f"mixed-mode: host func @{func_name} not found in linalgdir")
    i = m.end()
    depth, start, args = 1, i, []
    ang = cur = par = 0
    while i < len(linalgdir):
        c = linalgdir[i]
        if c == '<': ang += 1
        elif c == '>': ang -= 1
        elif c == '{': cur += 1
        elif c == '}': cur -= 1
        elif c == '(': par += 1
        elif c == ')':
            if par == 0:
                args.append(linalgdir[start:i]); break
            par -= 1
        elif c == ',' and ang == cur == par == 0:
            args.append(linalgdir[start:i]); start = i + 1
        i += 1
    types = []
    for a in args:
        a = a.strip()
        if not a:
            continue
        # "%argN: TYPE {attrs}" -> TYPE (strip name and trailing attr dict)
        after = a.split(":", 1)[1].strip()
        # drop a trailing " {...}" attribute dict at top level
        if after.endswith("}"):
            b = after.rfind(" {")
            if b != -1:
                after = after[:b].strip()
        types.append(after)
    return types


# The lowered host memref descriptor: spine-opt lowers each memref<*xT> param
# to (i64 rank, !llvm.ptr desc), where desc points to a StridedMemRefType
# {allocated, aligned, offset, sizes[1], strides[1]}. The aligned data ptr is
# field [1]. (Confirmed from gemv_host_ll.mlir.)
_LL_DESC = "!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"


def _ttir_pos_to_ll_argidx(host_arg_types):
    """Map a TTIR host-arg position → its start index in the LOWERED ll.mlir
    signature. Each memref param expands to 2 ll args (i64 rank, !llvm.ptr);
    each scalar stays 1. Returns (ll_start_index_list, is_memref_list)."""
    starts, is_mem = [], []
    ll = 0
    for t in host_arg_types:
        starts.append(ll)
        mem = t.strip().startswith("memref")
        is_mem.append(mem)
        ll += 2 if mem else 1
    return starts, is_mem


def _inject_mixed_llvm_llmlir(llmlir: str, func_name: str, host_arg_types,
                              llvm_funcs, llvm_calls) -> str:
    """Graft llvm.func siblings + host→sibling bridge into the LOWERED ll.mlir.

    Done post-lowering (uniform llvm dialect) so spine-opt never sees llvm ops
    it can't lower. For each pending call, bridge host descriptor args to the
    sibling's i64 ABI:
      - memref: load its (i64 rank,!llvm.ptr) descriptor's [1] aligned ptr,
        llvm.ptrtoint → i64  (sibling recovers it via llvm.inttoptr)
      - scalar i32: llvm.sext → i64
    Then `llvm.call @callee(...) : (i64,...) -> ()` before the host llvm.return.
    Validated by mlir-translate on real gemv ll.mlir + emit_llvm_func_for_inline.
    """
    ll_start, _is_mem = _ttir_pos_to_ll_argidx(host_arg_types)

    bridge = []
    uid = 0
    for spec in llvm_calls:
        callee = spec["callee"]
        operands = []
        for item in spec["arg_bridge"]:
            pos, kind = item["pos"], item["kind"]
            base = ll_start[pos]
            if kind == "ptr":
                desc_ptr = f"%arg{base + 1}"  # (rank=base, desc ptr=base+1)
                d = f"%mix{uid}_d"; p = f"%mix{uid}_p"; i = f"%mix{uid}_i"
                bridge.append(f"    {d} = llvm.load {desc_ptr} : !llvm.ptr -> {_LL_DESC}")
                bridge.append(f"    {p} = llvm.extractvalue {d}[1] : {_LL_DESC}")
                bridge.append(f"    {i} = llvm.ptrtoint {p} : !llvm.ptr to i64")
                operands.append(i)
            else:  # scalar: host passes it as i32 → sext to i64
                s = f"%mix{uid}_s"
                bridge.append(f"    {s} = llvm.sext %arg{base} : i32 to i64")
                operands.append(s)
            uid += 1
        argstr = ", ".join(operands)
        tystr = ", ".join("i64" for _ in operands)
        bridge.append(f"    llvm.call @{callee}({argstr}) : ({tystr}) -> ()")

    # Insert before the host func's FIRST llvm.return (host is single-block).
    host_at = llmlir.find(f"llvm.func @{func_name}")
    if host_at < 0:
        raise RuntimeError(f"mixed-mode: host llvm.func @{func_name} not found in ll.mlir")
    ret_m = None
    for m in re.finditer(r"^[ \t]*llvm\.return\b.*$", llmlir, re.MULTILINE):
        if m.start() > host_at:
            ret_m = m
            break
    if ret_m is None:
        raise RuntimeError("mixed-mode: no llvm.return in host func to anchor llvm.call")
    out = llmlir[:ret_m.start()] + "\n".join(bridge) + "\n" + llmlir[ret_m.start():]

    # Append the sibling llvm.func(s) before the module's closing brace.
    close = out.rfind("}")
    out = out[:close] + "\n" + "\n".join(llvm_funcs) + "\n" + out[close:]
    return out


def _spine_mlir_linalgdir_to_llir_ref(linalgdir: str, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        linalg_path = os.path.join(tmpdir, "linalg.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(linalg_path).write_text(linalgdir)
        # SpineTriton-MLIR to LLVM-MLIR
        spine_mlir_path = get_spine_mlir_opt_path()

        pipeline_option_str = get_spine_mlir_opt_options()
        if pipeline_option_str == "":
            pipeline_option_str = "enable-always-tls={}".format("0" if metadata["smt_parallel_inside"] else "1")

        cmd_str = '{} {} --spine-triton-e2e-ref-pipeline="{}" -o {}'.format(spine_mlir_path, linalg_path,
                                                                            pipeline_option_str, llmlir_path)
        subprocess.check_call(
            cmd_str,
            shell=True,
        )

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path])
        dump_ir_if_needed([llmlir_path, llir_path], metadata["name"])
        return Path(llir_path).read_text()


def _spine_mlir_linalgdir_to_llir(linalgdir: str, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        linalg_path = os.path.join(tmpdir, "linalg.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, ".ll")
        Path(linalg_path).write_text(linalgdir)
        spine_mlir_path = get_spine_mlir_opt_path()

        pipeline_option_str = get_spine_mlir_opt_options()
        if pipeline_option_str == "":
            pipeline_option_str = "enable-always-tls={} enable-fuse-group=false".format(
                "0" if metadata["smt_parallel_inside"] else "1")

        cmd_str = '{} {} --spine-triton-e2e-pipeline="{}" -o {}'.format(spine_mlir_path, linalg_path,
                                                                        pipeline_option_str, llmlir_path)
        subprocess.check_call(
            cmd_str,
            shell=True,
        )

        # Mixed-mode: splice llvm.func siblings + host-side llvm.call bridges into
        # the lowered ll.mlir (uniform llvm dialect, memrefs already descriptors).
        # Done here (post spine-opt, pre dump/translate) so both the debug-info
        # re-run path and the direct mlir-translate path see the injected module.
        if "mixed_llvm_funcs" in metadata and "mixed_llvm_calls" in metadata:
            _ll = Path(llmlir_path).read_text()
            _ll = _inject_mixed_llvm_llmlir(
                _ll, metadata["name"], metadata["mixed_host_arg_types"],
                metadata["mixed_llvm_funcs"], metadata["mixed_llvm_calls"])
            Path(llmlir_path).write_text(_ll)

        dump_ir_if_needed([llmlir_path], metadata["name"])

        llmlir_new_path = llmlir_path
        base_path = os.getenv("SPINE_TRITON_DUMP_PATH", "")
        if base_path:
            llmlir_new_path = os.path.join(tmpdir, "ll_with_debuginfo.mlir")
            subprocess.check_call([
                spine_mlir_path,
                os.path.join(
                    base_path,
                    metadata["name"] + "_" + os.path.basename(llmlir_path),
                ),
                "--ensure-debug-info-scope-on-llvm-func",
                "-mlir-print-debuginfo",
                "-o",
                llmlir_new_path,
            ])

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_new_path, "--mlir-to-llvmir", "-o", llir_path])
        dump_ir_if_needed([llir_path], metadata["name"])
        return Path(llir_path).read_text()


def _llvm_direct_to_llir(llvm_module_text: str, metadata):
    """LLVM-direct bypass: llvm.func module → LLVM IR (skip spine-opt, only mlir-translate)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        llmlir_path = os.path.join(tmpdir, "llvm_direct.mlir")
        llir_path = os.path.join(tmpdir, ".ll")
        Path(llmlir_path).write_text(llvm_module_text)
        mlir_translate_path = get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path])
        dump_ir_if_needed([llir_path], metadata["name"])
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return llir


def _llir_to_so(llir: str, metadata):
    cpu_arch = get_cpu_arch()
    target_arch_id = metadata["target"].arch_id
    ai_cpu_arch = get_cpu_name_from_arch_id(target_arch_id)

    target_arch = get_target_arch()
    cross_toolchain = get_cross_toolchain()

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, ".ll")
        src_opt_path = os.path.join(tmpdir, ".opt.ll")
        asm_path = os.path.join(tmpdir, ".s")
        dst_path = os.path.join(tmpdir, ".o")
        Path(src_path).write_text(llir)

        llopt_path = get_llvm_bin_path("opt")
        llopt_flags = []
        if target_arch == "riscv64":
            llopt_flags.extend([
                "--march=riscv64", "-passes=loop-vectorize", "--pass-remarks-missed", "-force-vector-width=32",
                "-force-vector-interleave=2"
            ])

        subprocess.check_call([llopt_path, src_path, *llopt_flags, "-o", src_opt_path])

        llc_path = get_llvm_bin_path("llc")
        llc_flags = ["-O3", "--float-abi=hard", "--relocation-model=pic"]
        if target_arch == "riscv64":
            mattr_list = ["64bit", "a", "b", "c", "d", "f", "i", "m", "v", "zfh", "zvfh", "zicbop", "zicbom", "zicboz"]
            if ai_cpu_arch in {"spacemit-a200", "spacemit-a200m"}:
                mattr_list.extend(["xsmtvsfu", "zmatrix"])
            elif ai_cpu_arch in {"spacemit-a100", "spacemit-x100", "spacemit-x60", "spacemit-a60"}:
                mattr_list.append("xsmtvdotii")

            llc_flags.extend(["--march=riscv64", "--mattr=" + ",".join(mattr_list)])

        # Generate assembly for dump if SPINE_TRITON_DUMP_PATH exists, but still generate object file for the final output
        if (dum_dir := os.environ.get("SPINE_TRITON_DUMP_PATH", "")) != "" and os.path.exists(dum_dir):
            subprocess.check_call([llc_path, src_opt_path, *llc_flags, "-filetype=asm", "-o", asm_path])
            kernel_name = metadata.get("name", "unknown_kernel")
            asm_dump_path = os.path.join(dum_dir, "{}.s".format(kernel_name))
            shutil.copy(asm_path, asm_dump_path)

        subprocess.check_call([llc_path, src_opt_path, *llc_flags, "-filetype=obj", "-o", dst_path])
        rpc_host = os.environ.get("SPINE_TRITON_RPC_HOST", "")
        if rpc_host:
            # For RPC mode, we don't need to create a shared library
            with open(dst_path, "rb") as f:
                return f.read()

        dump_ir_if_needed([dst_path], metadata["name"])

        cpu_backend_path = Path(__file__).resolve().parent
        include_dir = os.path.join(cpu_backend_path, "include")
        so_path = os.path.join(tmpdir, ".so")
        runtime_lib_dir = os.path.join(cpu_backend_path.parent.parent, "_C")

        if target_arch == "riscv64" and cpu_arch != "riscv64":
            assert os.path.exists(cross_toolchain), "Cross-compilation toolchain path does not exist: {}".format(
                cross_toolchain)
            # Cross-compilation mode: use cross-compile toolchain
            cc = os.path.join(cross_toolchain, "bin", "clang++")
            sysroot = os.path.join(cross_toolchain, "sysroot")
            subprocess.check_call([
                cc,
                "--target=riscv64-unknown-linux-gnu",
                f"--sysroot={sysroot}",
                "-std=c++17",
                "-march=rv64gcv_zfh_zba_zicbop",
                "-mabi=lp64d",
                "-O3",
                dst_path,
                f"-I{include_dir}",
                "-shared",
                "-fPIC",
                "-fuse-ld=lld",
                "-nostdlib++",
                "-o",
                so_path,
            ])
        else:
            # Native compilation mode
            py_version = sys.version_info
            py_include_dir = os.path.join(
                sys.base_prefix,
                "include",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
            )
            py_lib_dir = os.path.join(sys.base_prefix, "lib")
            py_lib = "{name}{major}.{minor}".format(name="python", major=py_version.major, minor=py_version.minor)

            gcc_flags = []
            if target_arch == "riscv64":
                gcc_flags.extend(["-march=rv64gcv_zfh_zba_zicbop", "-mabi=lp64d", "-O3"])
            subprocess.check_call([
                "g++",
                "-std=c++17",
                *gcc_flags,
                dst_path,
                f"-I{py_include_dir}",
                f"-I{include_dir}",
                f"-L{py_lib_dir}",
                f"-L{runtime_lib_dir}",
                "-shared",
                f"-l{py_lib}",
                "-lSpineTritonRuntime",
                "-fPIC",
                "-o",
                so_path,
            ])

        dump_ir_if_needed([so_path], metadata["name"])
        with open(so_path, "rb") as f:
            return f.read()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True

    def __post_init__(self):
        pass

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = "so"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        if "instrumentation_mode" in opts:
            opts.pop("instrumentation_mode")
        args = {"arch": self.target.arch}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        return CPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `spine-triton.cc`
    def load_dialects(self, ctx):
        spine_triton.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttir")
        num_threads = metadata['target'].num_threads
        attrs = []
        attrs.append(num_threads)
        arch_id = metadata['target'].arch_id
        attrs.append(arch_id)
        force_vector_interleave = metadata['target'].force_vector_interleave
        attrs.append(force_vector_interleave)
        builder = ir.builder(mod.context)
        mod.set_attr("tt.num_threads", builder.get_int32_attr(num_threads))
        mod.set_attr("tt.arch_id", builder.get_string_attr(arch_id))
        mod.set_attr("tt.force_vector_interleave", builder.get_int32_attr(force_vector_interleave))

        # LLVM-direct: pick up a pending llvm.func module text stashed by
        # spine_raw.call() during make_ir (process-global handoff — see
        # call_registry.take_pending_llvm_direct_module). None for non-llvm-direct kernels.
        _llvm_direct_text, _llvm_direct_name = None, None
        try:
            from triton.language.extra.spine_raw.call_registry import take_pending_llvm_direct_module
            _llvm_direct_text, _llvm_direct_name = take_pending_llvm_direct_module()
            if _llvm_direct_text:
                metadata["llvm_direct_module"] = _llvm_direct_text
                # LLVM-direct bypasses _ttir_to_linalgdir, which normally seeds
                # smt_parallel_inside (read by the launcher + pipeline option).
                # LLVM-direct kernels are single-program (no bind_sub_block), so False.
                metadata["smt_parallel_inside"] = False
        except Exception:
            pass

        # Mixed-mode (coexistence): the host keeps its func.func body (tl +
        # spine_raw dsl_region) AND calls one or more llvm-direct siblings. Unlike
        # the pure-LLVM path above (which REPLACES the module), here we stash the
        # sibling func text + per-call arg bridge. Injection happens at the LOWERED
        # ll.mlir layer (_inject_mixed_llvm_llmlir, post spine-opt) — the linalgdir
        # layer can't host it because memrefs still carry #ptr.generic_space, which
        # crashes extract_aligned_pointer lowering. At ll.mlir the host is uniform
        # llvm dialect with memrefs already descriptors, so the llvm.call + sibling
        # splice is legal. Independent of llvm_direct_module (unset in mixed).
        try:
            from triton.language.extra.spine_raw.call_registry import (
                take_pending_llvm_funcs, take_pending_llvm_calls)
            _mixed_funcs = take_pending_llvm_funcs()
            _mixed_calls = take_pending_llvm_calls()
            if _mixed_funcs and _mixed_calls:
                metadata["mixed_llvm_funcs"] = _mixed_funcs
                metadata["mixed_llvm_calls"] = _mixed_calls
        except Exception:
            pass

        tt_pattern = r"tt\.func\s+public\s+@(\w+)\s*\("
        kernel_name = extract_kernel_name(tt_pattern, str(mod))
        metadata["name"] = kernel_name
        # LLVM-direct: the binary exports the emitted llvm.func's symbol (the raw
        # kernel name), not the @triton.jit host wrapper. The launcher looks up
        # metadata["name"] as the symbol, so override it to the emitted name.
        # (Mixed mode keeps the host name — the entry point is the host func.func.)
        if _llvm_direct_text and _llvm_direct_name:
            metadata["name"] = _llvm_direct_name
        return mod

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)

        def _linalgdir_stage(src, metadata):
            # LLVM-direct bypass: if metadata has pre-emitted llvm.func module, return it
            if "llvm_direct_module" in metadata:
                return metadata["llvm_direct_module"]
            linalgdir = _optimize_linalgdir(_ttir_to_linalgdir(src, metadata))
            # Mixed mode: record the host func.func arg TYPES (ordered) now, while
            # they're still memref/scalar. The sibling llvm.call is grafted later
            # at the *ll.mlir* stage (post-lowering, uniform llvm dialect) — NOT
            # here: extract_aligned_pointer_as_index on a #ptr.generic_space memref
            # crashes spine-opt (getMemorySpaceAsInt). We only need the type list
            # to map each TTIR arg position → its (i64 rank, !llvm.ptr) descriptor
            # slot in the lowered host signature.
            if "mixed_llvm_calls" in metadata:
                metadata["mixed_host_arg_types"] = _host_func_arg_types(
                    linalgdir, metadata["name"])
            return linalgdir

        stages["linalgdir"] = _linalgdir_stage

        use_ref_pipeline = int(os.getenv("SPINE_TRITON_USE_REF_PIPELINE", "0")) > 0

        def _llir_stage(src, metadata):
            # LLVM-direct bypass: skip spine-opt, only mlir-translate
            if "llvm_direct_module" in metadata:
                return _optimize_llir(_llvm_direct_to_llir(src, metadata))
            # Normal path
            if not use_ref_pipeline:
                return _optimize_llir(_spine_mlir_linalgdir_to_llir(src, metadata))
            else:
                return _optimize_llir(_spine_mlir_linalgdir_to_llir_ref(src, metadata))

        stages["llir"] = _llir_stage
        stages["so"] = lambda src, metadata: _llir_to_so(src, metadata)

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}


def get_cache_sizes():

    unit_map = {"k": 1024, "m": 1024**2, "g": 1024**3, "": 1}

    cache_cmd = {
        "L1": "lscpu | grep -E 'L1d|Cache|一级数据' | grep -v 'combined' | awk '{print $3, $4, $5, $6}'",
        "L2": "lscpu | grep -E 'L2|Cache|二级数据' | grep -v 'combined' | awk '{print $3, $4, $5, $6}'",
        "L3": "lscpu | grep -E 'L3|Cache|三级数据' | grep -v 'combined' | awk '{print $3, $4, $5, $6}'",
    }

    results = []
    for cache_level in ["L1", "L2", "L3"]:  # Enforce order
        try:
            output = subprocess.check_output(cache_cmd[cache_level], shell=True).decode()
        except Exception as e:
            print(f"Command execution failed: {e}")
            results.append(0)
            continue

        match = re.search(r"(\d+)\s*([KMG]?i?B)\s*\((\d+)\s*instances\)", output)
        matchNoinstances = re.search(r"(\d+)\s*([KMG]?i?B)", output)
        if match:
            total_size, unit, instances = match.groups()
        elif matchNoinstances:
            total_size, unit = matchNoinstances.groups()
            instances = 1
        else:
            results.append(0)
            continue

        unit = unit.lower().rstrip("ib")
        bytes_per_instance = (int(total_size) * unit_map[unit]) // int(instances)
        results.append(bytes_per_instance)

    return results  # Format: [L1_size, L2_size, L3_size] in bytes
