#!/bin/bash
# Shared vendoring helpers sourced by the build scripts (build.sh,
# build_whl.sh, build_x86_rpc.sh, build_native.sh). Not a standalone script:
#   source scripts/common.sh
# then call vendor_spert_headers / vendor_mlir_bindings below.

# Vendor spert headers from <spine-runtime-install>/include into
# backend/include/SpineRuntime so the generated launcher can #include
# "spert.hpp" (see backend/driver.py). libspert.so* is picked up separately
# by setup.py from <spine-runtime-install>/lib. Both come from the
# spine-runtime install package, shipped independently of spine-mlir.
#
# Usage: vendor_spert_headers <spine-runtime-install-dir>
vendor_spert_headers() {
    local rt=$1
    if [ ! -f "${rt}/include/spert.hpp" ]; then
        echo "ERROR: spert.hpp not found in ${rt}/include" >&2
        return 1
    fi
    mkdir -p backend/include/SpineRuntime
    local h
    for h in spert.hpp spert_engine.hpp spert_abi.h; do
        [ -f "${rt}/include/${h}" ] && \
            cp "${rt}/include/${h}" "backend/include/SpineRuntime/${h}"
    done
    echo "vendored spert headers from ${rt}/include"
}

# Vendor MLIR Python bindings (mlir_core) into backend/mlir_core so the
# built package is self-contained: llvm_direct.py / mixed_bridge.py resolve
# them at runtime (language/smt_rvisa/_mlir_loader.py) without any PYTHONPATH
# export. The vendored layout is always backend/mlir_core/mlir/ (the mlir
# namespace package itself); the source's extra top-level files are ignored.
#
# The source package layout: <src>/mlir is the bindings tree itself (the
# f6ded0be release packages, e.g. llvm-f6ded0be...-riscv64-python3XX or
# llvm-f6ded0be...-x86-python312, whose top level contains mlir/).
#
# Usage: vendor_mlir_bindings <src-dir>
vendor_mlir_bindings() {
    if [ ! -d "${1}" ]; then
        echo "ERROR: MLIR bindings source dir '${1}' does not exist" >&2
        return 1
    fi
    if [ ! -d "${1}/mlir" ]; then
        echo "ERROR: MLIR Python bindings (mlir/) not found under '${1}'" >&2
        echo "Pass a bindings package whose top level contains mlir/" >&2
        return 1
    fi
    rm -rf backend/mlir_core
    mkdir -p backend/mlir_core/mlir
    cp -a "${1}/mlir/." backend/mlir_core/mlir/
    echo "vendored MLIR Python bindings from: ${1}/mlir"
}
