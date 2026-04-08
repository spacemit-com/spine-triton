import hashlib
import tempfile
import os
import subprocess
import platform
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
import time
import triton
from triton.runtime.cache import get_cache_manager
from triton.runtime.build import compile_module_from_src
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from . import (get_cpu_name_from_arch_id, get_spine_mlir_cc_debug)

dirname = os.path.dirname(os.path.realpath(__file__))
include_dir = os.path.join(dirname, "include")

# -------------------- Utility Functions ----------------------------


def _ty_to_cpp(ty):
    if ty[0] == "*":
        return "void*"
    if ty == "constexpr":
        return "PyObject*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def _extracted_type(ty):
    if ty[0] == "*":
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)


def _format_of(ty):
    return {
        "PyObject*": "O",
        "constexpr": "O",
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "l",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[ty]


def _generate_launcher(constants, signature, smt_parallel_inside=False, kernel_name="unknown_kernel"):
    # Check if kernel-level proton capture is enabled at compile time
    enable_proton_kernel_capture = os.environ.get("PROTON_KERNEL_CAPTURE", "0") != "0"

    arg_decls = ", ".join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = "".join([_format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiKKOOOO" + args_format
    args_list = (", " + ", ".join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else "")

    kernel_arg_decls = ", ".join(
        _ty_to_cpp(ty) if ty[0] != "*" else "int64_t, void*" for i, ty in signature.items() if ty != "constexpr")
    kernel_arg_decls += ", " if kernel_arg_decls else ""

    kernel_parameters = ", ".join(f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"0, &ptr_arg{i}"
                                  for i, ty in signature.items()
                                  if ty != "constexpr")
    kernel_parameters += ", " if kernel_parameters else ""

    smt_parallel_inside_arg = "constexpr bool smt_parallel_inside = {};".format(
        "true" if smt_parallel_inside else "false")

    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include <memory>
#include <optional>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

namespace mlir {{
namespace speir {{
void *spineGetMultiStream(int64_t);

template <int64_t rank>
void spineMultiStreamDispatch(
    void *multi_stream,
    const std::function<void(const std::array<int64_t, rank> &)> &fn,
    const std::array<int64_t, rank> &block_size);

void spineStreamDispatch(
    void *stream, const std::function<void(const std::array<int64_t, 3> &)> &fn,
    const std::array<int64_t, 3> &grid_size);

}} // namespace speir
}}// namespace mlir

extern "C" {{
int64_t spine_get_stream_threads();
int64_t spine_require_stream();
void spine_release_stream(int64_t);
{'''// Proton kernel-level profiling APIs
void proton_enter_kernel(const char *kernel_name, int gridX, int gridY, int gridZ);
void proton_exit_kernel(const char *kernel_name, int gridX, int gridY, int gridZ);''' if enable_proton_kernel_capture else ''}
}}

using kernel_ptr_t = void(*)({kernel_arg_decls} int, int, int, int, int, int);


typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (void*) PyLong_AsLongLong(ret);
    if(!ptr_info.dev_ptr) {{
      return ptr_info;
    }}
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}}


{f'static constexpr const char* KERNEL_NAME = "{kernel_name}";' if enable_proton_kernel_capture else ''}

static void _launch(int gridX, int gridY, int gridZ, int64_t stream, kernel_ptr_t kernel_ptr, {arg_decls}) {{
  {smt_parallel_inside_arg}
  if (gridX*gridY*gridZ <= 0) return;
  {'// Auto kernel capture: record kernel entry' if enable_proton_kernel_capture else ''}
  {'proton_enter_kernel(KERNEL_NAME, gridX, gridY, gridZ);' if enable_proton_kernel_capture else ''}
  int64_t stream_threads = spine_get_stream_threads();
  int64_t gridX_out = (gridX + stream_threads - 1) / stream_threads;
  const char* force_nested_dispatch_1d_env = getenv("SPINE_TRITON_FORCE_NESTED_DISPATCH_1D");
  bool force_nested_dispatch_1d =
      force_nested_dispatch_1d_env != nullptr && strcmp(force_nested_dispatch_1d_env, "0") != 0;
  // 1D grid kernels are sensitive to pid(0) mapping; keep a safe one-level mapping by default.
  bool use_safe_pid_mapping = (gridY == 1 && gridZ == 1 && !force_nested_dispatch_1d);
  {' '.join(f'StridedMemRefType<char, 0> ptr_arg{i} = {{static_cast<char *>(arg{i}), static_cast<char *>(arg{i}), 0}};'
            for i, ty in signature.items() if i not in constants and ty[0] == "*")}
    if constexpr (!smt_parallel_inside) {{
        if (use_safe_pid_mapping) {{
            mlir::speir::spineMultiStreamDispatch<3>(reinterpret_cast<void*>(stream), [&](const std::array<int64_t, 3> &block){{
                int x = block[0];
                int y = block[1];
                int z = block[2];
                (*kernel_ptr)({kernel_parameters}
                                     gridX, gridY, gridZ, x, y, z);
            }},
             {{gridX, gridY, gridZ}});
        }} else {{
            mlir::speir::spineMultiStreamDispatch<3>(reinterpret_cast<void*>(stream), [&](const std::array<int64_t, 3> &block){{
                int x_out = block[0];
                int y_out = block[1];
                int z_out = block[2];
                int64_t current_stream = spine_require_stream();
                mlir::speir::spineStreamDispatch(reinterpret_cast<void*>(current_stream),
                [&] (const std::array<int64_t, 3> & cur_grid) {{
                    int x = cur_grid[0] + x_out * stream_threads;
                    if (x >= gridX) {{
                            return;
                    }}
                    (*kernel_ptr)({kernel_parameters}
                                         gridX, gridY, gridZ, x, y_out, z_out);
                }},
                    {{stream_threads, 1, 1}});

                spine_release_stream(current_stream);
            }},
                 {{gridX_out, gridY, gridZ}});
        }}
  }} else {{
    mlir::speir::spineMultiStreamDispatch<3>(reinterpret_cast<void*>(stream), [&](const std::array<int64_t, 3> &block){{
      int x = block[0];
      int y = block[1];
      int z = block[2];
      (*kernel_ptr)({kernel_parameters}
                   gridX, gridY, gridZ, x, y, z);
    }},
       {{gridX, gridY, gridZ}});
  }}
  {'// Auto kernel capture: record kernel exit' if enable_proton_kernel_capture else ''}
  {'proton_exit_kernel(KERNEL_NAME, gridX, gridY, gridZ);' if enable_proton_kernel_capture else ''}
  }}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  uint64_t _stream;
  uint64_t _function;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  kernel_ptr_t kernel_ptr = reinterpret_cast<kernel_ptr_t>(_function);

  // [CPULauncher-specific]: We don't need the metadata below but just put them
  // here anyway to be consistent with others.
  // This will make updating the driver easier in the future.

  //  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  //  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
  //    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
  //    return NULL;
  //  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, _stream, kernel_ptr, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0] == "*" else f"_arg{i}"for i, ty in signature.items())});
  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__spine_triton_kernel_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___spine_triton_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def compile_module(src, name):
    py_version = sys.version_info
    cpu_arch = platform.machine()
    if platform.system() == "Windows":
        py_include_dir = os.path.join(sys.base_prefix, "include")
        py_lib_dir = os.path.join(sys.base_prefix, "libs")
        py_lib = "{name}{major}{minor}.lib".format(name="python", major=py_version.major, minor=py_version.minor)
    else:
        py_include_dir = os.path.join(
            sys.base_prefix,
            "include",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        )
        py_lib_dir = os.path.join(sys.base_prefix, "lib")
        py_lib = "{name}{major}.{minor}".format(name="python", major=py_version.major, minor=py_version.minor)
    cpu_backend_path = Path(__file__).resolve().parent
    include_dir = os.path.join(cpu_backend_path, "include")
    spine_opt_debug = get_spine_mlir_cc_debug()
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    if platform.system() == "Windows":
        filename = f"{name}.pyd"
    else:
        filename = f"{name}.so"
    cache_path = cache.get_file(filename)
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            if platform.system() == "Windows":
                launcher_src_path = os.path.join(tmpdir, "main.cxx")
                so_path = os.path.join(tmpdir, "kernel.pyd")
                Path(launcher_src_path).write_text(src)
                # Compile it together.
                subprocess.check_call([
                    "cl",
                    "/LD",
                    "/std:c++17",
                    launcher_src_path,
                    f"-I{py_include_dir}",
                    f"-I{include_dir}",
                    "/link",
                    f"/LIBPATH:{py_lib_dir}",
                    "/link",
                    f"{py_lib}",
                    f"/OUT:{so_path}",
                ])
            else:
                launcher_src_path = os.path.join(tmpdir, "main.cxx")
                so_path = os.path.join(tmpdir, "kernel.so")

                Path(launcher_src_path).write_text(src)

                with open(launcher_src_path, "rb") as f:
                    launcher_src_path = cache.put(f.read(), os.path.basename(launcher_src_path), binary=False)

                gcc_flags = []
                if cpu_arch == "riscv64":
                    gcc_flags.extend(["-march=rv64gcv_zfh_zba_zicbop_zihintpause", "-mabi=lp64d"])
                if spine_opt_debug:
                    gcc_flags.append("-g")
                    gcc_flags.append("-O0")
                else:
                    gcc_flags.append("-O3")

                # Compile it together.
                subprocess.check_call([
                    "g++",
                    "-std=c++17",
                    *gcc_flags,
                    launcher_src_path,
                    f"-I{py_include_dir}",
                    f"-I{include_dir}",
                    f"-L{py_lib_dir}",
                    "-shared",
                    f"-l{py_lib}",
                    "-fPIC",
                    "-o",
                    so_path,
                ])

            with open(so_path, "rb") as f:
                cache_path = cache.put(f.read(), filename, binary=True)

    spec = importlib.util.spec_from_file_location(name, cache_path)
    if spec is None:
        raise RuntimeError(f"Cannot find {name} module in {cache_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class AICPUTarget(GPUTarget):
    backend: str
    arch: str
    core: int
    ai_core: int
    arch_id: str
    num_threads: int
    force_vector_interleave: int


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        # Runtime library path for spine_print_unranked_memref
        runtime_lib_dir = os.path.join(dirname, "..", "..", "_C")

        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="cpu_utils",
            library_dirs=[runtime_lib_dir],
            include_dirs=[include_dir],
            libraries=["dl"],
        )
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self._get_current_stream = mod.get_current_stream
        self._get_arch_id = mod.get_arch_id
        self._get_num_cores = mod.get_num_cores
        self._get_stream_threads = mod.get_stream_threads

    def get_current_stream(self):
        return self._get_current_stream()

    def get_arch_id(self):
        return self._get_arch_id()

    def get_num_cores(self):
        return self._get_num_cores()

    def get_stream_threads(self):
        return self._get_stream_threads()


class CPULauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()

        def cst_key(i):
            return src.fn.arg_names.index(i) if isinstance(i, str) else i

        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        smt_parallel_inside = metadata.smt_parallel_inside
        # Get kernel name for auto proton capture
        kernel_name = src.fn.__name__ if hasattr(src, 'fn') and hasattr(src.fn, '__name__') else "unknown_kernel"
        launcher_src = _generate_launcher(constants, signature, smt_parallel_inside, kernel_name)
        mod = compile_module(launcher_src, "__spine_triton_kernel_launcher")
        self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.compiler.CompiledKernel.launch_enter_hook = None
        triton.compiler.CompiledKernel.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.compiler.CompiledKernel.launch_enter_hook = (lambda arg: self._enter_hook())
        triton.compiler.CompiledKernel.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()


class CPUDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        self.binary_ext = "so"
        self.current_arch_id = self.utils.get_arch_id()
        self.cpu_arch = get_cpu_name_from_arch_id(self.current_arch_id)
        self.num_cores = self.utils.get_num_cores()
        self.num_of_stream_threads = self.utils.get_stream_threads()
        self.force_vector_interleave = 2

    # CPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(CPUDriver())
    @staticmethod
    def is_active():
        return False

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_device_capability(self):
        return ("cpu", 0)

    def get_current_stream(self, device):
        return self.utils.get_current_stream()

    def get_current_device(self):
        # CPU doesn't have a device to return. Return something.
        return "cpu"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "cpu"
        return

    def get_current_target(self):
        return AICPUTarget("cpu", self.cpu_arch, 0, self.num_cores, self.num_of_stream_threads, self.current_arch_id,
                           self.num_of_stream_threads, self.force_vector_interleave)

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def get_device_interface(self):
        return CPUDeviceInterface()

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="cpu")

    def clear_cache(self, cache):
        cache.zero_()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)
