#!/bin/bash
set -euo pipefail
# bash build_whl.sh ${LLVM_INSTALL_DIR} {arch/x86_64/riscv64} {spine-mlir-install-dir} {spine-runtime-install-dir} {mlir-bindings-dir} [py-riscv64-dir]

# Resolve all input paths to absolute paths to avoid breakage after pushd
LLVM_INSTALL_DIR=$(cd "${1}" && pwd)
BUILD_DIR=build-wheel-${2}
SPINE_MLIR_INSTALL_DIR=$(cd "${3}" && pwd)
chmod a+x "${SPINE_MLIR_INSTALL_DIR}"/bin/*

# spine-runtime install dir (its own `installed/` layout: include/ + lib/).
# libspert.so + spert headers come from here, separate from spine-mlir.
if [ -z "${4:-}" ]; then
    echo "ERROR: SPINE_RUNTIME_INSTALL_DIR (arg 4) is required" >&2
    echo "Usage: bash build_whl.sh <LLVM> <arch> <spine-mlir-install> <spine-runtime-install>" >&2
    exit 1
fi
SPINE_RUNTIME_INSTALL_DIR=$(cd "${4}" && pwd)

CUR_DIR=${PWD}
VERSION_NUMBER=$(cat VERSION_NUMBER)
MAX_JOBS=${MAX_JOBS:-20}

echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"
echo "SPINE_MLIR_INSTALL_DIR: ${SPINE_MLIR_INSTALL_DIR}"
echo "SPINE_RUNTIME_INSTALL_DIR: ${SPINE_RUNTIME_INSTALL_DIR}"
echo "BUILD_DIR: ${BUILD_DIR}"

export TRITON_PLUGIN_DIRS=${PWD}

# Vendor spert headers from SPINE_RUNTIME_INSTALL_DIR/include into backend/include
# (the generated launcher #include "spert.hpp"); libspert.so* is picked up by
# setup.py from SPINE_RUNTIME_INSTALL_DIR/lib.
source "$(dirname "$(readlink -f "$0")")/common.sh"
vendor_spert_headers "${SPINE_RUNTIME_INSTALL_DIR}"

# Vendor MLIR Python bindings (mlir_core) into backend/mlir_core so the wheel
# is self-contained: llvm_direct.py / mixed_bridge.py resolve them at runtime
# (language/smt_rvisa/_mlir_loader.py) without any PYTHONPATH export.
# Arg 5 (required): the riscv64-python3XX bindings release package from
# https://github.com/spacemit-com/spine-mlir/releases (top level contains mlir/).
if [ -z "${5:-}" ]; then
    echo "ERROR: mlir-bindings dir (arg 5) is required" >&2
    echo "Usage: bash build_whl.sh <LLVM> <arch> <spine-mlir-install> <spine-runtime-install> <mlir-bindings-dir> [py-riscv64-dir]" >&2
    exit 1
fi
vendor_mlir_bindings "${5}"

mkdir -p ${TRITON_PLUGIN_DIRS}/${BUILD_DIR}

pushd triton
git reset
git checkout .
git clean -fd
ls ${CUR_DIR}/patch/*.patch | xargs -n1 git apply

export SPINE_MLIR_INSTALL_DIR=${SPINE_MLIR_INSTALL_DIR}
export SPINE_RUNTIME_INSTALL_DIR=${SPINE_RUNTIME_INSTALL_DIR}
export SPINE_TRITON_VERSION_NUMBER=${VERSION_NUMBER}
# NOTE: never override TRITON_CODEGEN_BACKENDS here — setup.py passes
# "-DTRITON_CODEGEN_BACKENDS=nvidia;amd" earlier on the same cmake command line
# and the later -D from TRITON_APPEND_CMAKE_ARGS would win; an empty override
# skips add_subdirectory(third_party/amd) and triton-opt.cpp then fails on the
# missing tablegen'd TritonAMDGPUTransforms/Passes.h.inc.
# Cross builds: the host interpreter's Python3 headers are x86; point cmake at
# the riscv64 Python (py-riscv64-3xx package) headers instead — only headers are
# needed (Linux python modules do not link libpython), Python3_LIBRARY is passed
# only to satisfy FindPython's LOCATION strategy. zlib is not in the toolchain
# sysroot either; the CI builds it into the toolchain sysroot
# (.github/scripts/build_rv_zlib.sh) and this script defaults ZLIB_LIBDIR there
# (overridable when zlib lives elsewhere).
PY_RISCV_ROOT=${PY_RISCV_ROOT:-${6:-}}
if [ "${2}" = "riscv64" ]; then
    if [ -z "${PY_RISCV_ROOT}" ]; then
        echo "ERROR: py-riscv64 dir (arg 6 or PY_RISCV_ROOT env) is required for riscv64 cross builds" >&2
        echo "Usage: bash build_whl.sh <LLVM> <arch> <spine-mlir-install> <spine-runtime-install> [mlir-bindings] <py-riscv64-dir>" >&2
        exit 1
    fi
    PY_VER=$(ls "${PY_RISCV_ROOT}/include" | grep -m1 '^python3\.')
    if [ -z "${PY_VER}" ]; then
        echo "ERROR: no python3.* include dir under ${PY_RISCV_ROOT}/include" >&2
        exit 1
    fi
    EXTRA_CMAKE_ARGS="\
-DPython3_INCLUDE_DIR:PATH=${PY_RISCV_ROOT}/include/${PY_VER} \
-DPython3_LIBRARY:FILEPATH=${PY_RISCV_ROOT}/lib/lib${PY_VER}.so \
-DPython3_FIND_STRATEGY=LOCATION \
-DPython3_FIND_REGISTRY=NEVER \
-DPython3_FIND_FRAMEWORK=NEVER \
"
else
    EXTRA_CMAKE_ARGS=""
fi
# zlib is not in the toolchain tarball's sysroot; CI builds it into
# ${RISCV_ROOT_PATH}/sysroot/usr (see .github/scripts/build_rv_zlib.sh), so
# default ZLIB_LIBDIR to the sysroot and fail explicitly when no zlib is
# found there. Override with ZLIB_LIBDIR env when zlib lives elsewhere.
ZLIB_LIBDIR=${ZLIB_LIBDIR:-${RISCV_ROOT_PATH}/sysroot/usr/lib}
if [ ! -e "${ZLIB_LIBDIR}/libz.a" ] && [ ! -e "${ZLIB_LIBDIR}/libz.so" ]; then
    echo "ERROR: no libz under ${ZLIB_LIBDIR}" >&2
    echo "Build zlib for riscv64 into the toolchain sysroot (.github/scripts/build_rv_zlib.sh) or set ZLIB_LIBDIR" >&2
    exit 1
fi
export TRITON_APPEND_CMAKE_ARGS="-DLLVM_LIBRARY_DIR=${LLVM_INSTALL_DIR}/lib \
-DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm \
-DLLD_DIR=${LLVM_INSTALL_DIR}/lib/cmake/lld \
-DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir \
-DCMAKE_TOOLCHAIN_FILE=${CUR_DIR}/cmake/linux_riscv64.toolchain.cmake \
-DCMAKE_SHARED_LINKER_FLAGS=-L${ZLIB_LIBDIR} \
-DCMAKE_EXE_LINKER_FLAGS=-L${ZLIB_LIBDIR} \
${EXTRA_CMAKE_ARGS}"
export CC=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc
export CXX=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-g++
TRITON_BUILD_PROTON=false TRITON_BUILD_WITH_CLANG_LLD=false TRITON_BUILD_UT=false TRITON_OFFLINE_BUILD=true \
TRITON_BUILD_WITH_CCACHE=false TRITON_IN_TREE_BACKENDS= LLVM_ROOT_DIR=${LLVM_INSTALL_DIR} LLVM_SYSPATH=${LLVM_INSTALL_DIR} MAX_JOBS=${MAX_JOBS} \
python3 setup.py bdist_wheel --plat=linux-${2}

cp dist/*.whl ${CUR_DIR}/${BUILD_DIR}/
popd

echo "whl package generated successfully at: ${CUR_DIR}/${BUILD_DIR}/"
