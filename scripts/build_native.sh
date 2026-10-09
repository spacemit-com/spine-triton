# !/bin/bash
# bash build.sh ${LLVM_INSTALL_DIR} {arch/x86_64/riscv64} {spine-mlir-install-dir} [{mlir-bindings-llvm}]

LLVM_INSTALL_DIR=${1}
BUILD_DIR=build-${2}
SPINE_MLIR_INSTALL_DIR=${3}
chmod a+x "${SPINE_MLIR_INSTALL_DIR}"/bin/*

CUR_DIR=${PWD}
VERSION_NUMBER=$(cat VERSION_NUMBER)

echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"

export TRITON_PLUGIN_DIRS=${PWD}

# Vendor MLIR Python bindings (mlir_core) into backend/mlir_core so the build
# is self-contained: llvm_direct.py / mixed_bridge.py resolve them at runtime
# (language/smt_rvisa/_mlir_loader.py) without any PYTHONPATH export.
# Source (arg 4, optional): either the "-python" release package (a directory
# containing mlir_core/) or an LLVM install with python_packages/mlir_core;
# defaults to the build LLVM when it ships python_packages.
if [ -n "${4:-}" ]; then
    MLIR_BINDINGS_ROOT=$(cd "${4}" && pwd)
else
    MLIR_BINDINGS_ROOT=${LLVM_INSTALL_DIR}
fi
if [ -d "${MLIR_BINDINGS_ROOT}/python_packages/mlir_core" ]; then
    MLIR_BINDINGS_SRC=${MLIR_BINDINGS_ROOT}/python_packages/mlir_core
elif [ -d "${MLIR_BINDINGS_ROOT}/mlir_core" ]; then
    MLIR_BINDINGS_SRC=${MLIR_BINDINGS_ROOT}/mlir_core
elif [ -d "${MLIR_BINDINGS_ROOT}/mlir" ]; then
    # arg is the mlir_core directory itself
    MLIR_BINDINGS_SRC=${MLIR_BINDINGS_ROOT}
else
    echo "ERROR: MLIR Python bindings (mlir_core) not found under ${MLIR_BINDINGS_ROOT}" >&2
    echo "Pass the -python release package or an LLVM install with python_packages/mlir_core as arg 4" >&2
    exit 1
fi
rm -rf backend/mlir_core
mkdir -p backend/mlir_core
cp -a "${MLIR_BINDINGS_SRC}/." backend/mlir_core/
if [ ! -d backend/mlir_core/mlir ]; then
    echo "ERROR: MLIR Python bindings (mlir_core) not found at ${MLIR_BINDINGS_SRC}" >&2
    exit 1
fi
echo "vendored MLIR Python bindings from: ${MLIR_BINDINGS_SRC}"

mkdir -p ${TRITON_PLUGIN_DIRS}/${BUILD_DIR}

pushd triton
git reset
git checkout .
git clean -fd
ls ${CUR_DIR}/patch/*.patch | xargs -n1 git apply

export SPINE_MLIR_INSTALL_DIR=${SPINE_MLIR_INSTALL_DIR}
export SPINE_TRITON_VERSION_NUMBER=${VERSION_NUMBER}
export TRITON_APPEND_CMAKE_ARGS="-DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm -DLLVM_LIBRARY_DIR=${LLVM_INSTALL_DIR}/lib -DLLD_DIR=${LLVM_INSTALL_DIR}/lib/cmake/lld -DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir"
TRITON_BUILD_PROTON=false TRITON_BUILD_WITH_CLANG_LLD=false TRITON_BUILD_UT=false TRITON_OFFLINE_BUILD=true \
TRITON_BUILD_WITH_CCACHE=false LLVM_ROOT_DIR=${LLVM_INSTALL_DIR} MAX_JOBS=20 \
python3 setup.py install --prefix=${TRITON_PLUGIN_DIRS}/${BUILD_DIR}
popd

rm -rf ${BUILD_DIR}/triton

if ls -d ${BUILD_DIR}/lib/python*/site-packages/triton >/dev/null 2>&1; then
    cp -r ${BUILD_DIR}/lib/python*/site-packages/triton* ${BUILD_DIR}/
    rm -rf ${BUILD_DIR}/lib
elif  ls -d ${BUILD_DIR}/local/lib/python*/dist-packages/triton >/dev/null 2>&1; then
    cp -r ${BUILD_DIR}/local/lib/python*/dist-packages/triton* ${BUILD_DIR}/
    rm -rf ${BUILD_DIR}/local
else
    echo "Error: Cannot find triton package"
    exit 1
fi
