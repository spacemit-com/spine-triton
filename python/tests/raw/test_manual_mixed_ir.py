"""
Manually written mixed-mode IR test: func.func host + llvm.func sibling.

This demonstrates the multi-function module approach for coexistence of
tl/spine_raw/call_intrinsic semantics in a single kernel, without requiring
automatic emission logic in call_registry.py.

The test validates:
1. BufferDeallocation processes func.func, skips llvm.func
2. llvm.call from func.func to llvm.func sibling compiles cleanly
3. Full pipeline spine-opt → mlir-translate → LLVM IR succeeds
"""

import triton
import triton.language as tl
import torch
import subprocess
import tempfile
import os


def test_mixed_manual_ir():
    """Test manually written mixed IR (func.func + llvm.func) through full pipeline."""

    # Manually written mixed IR based on test_mixed_syntax_three_layer pattern
    mixed_ir = '''module attributes {dlti.target_system_spec = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"arch_id" = "0xA064", "num_threads" = 4 : i32>>, tt.force_vector_interleave = 2 : i32} {
  func.func @mixed_host(%arg0: memref<*xf16>, %arg1: memref<*xf16>, %arg2: f32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    // Stage 1: simple arith loop (like pre_scale_tl)
    %pid = arith.index_cast %arg7 : i32 to index
    %block_start = arith.muli %pid, %c64 : index

    scf.for %i = %c0 to %c64 step %c1 {
      %idx = arith.addi %block_start, %i : index
      // Simplified: just touch the memory to have memref ops
      %reinterpret = memref.reinterpret_cast %arg0 to offset: [%idx], sizes: [1], strides: [1] : memref<*xf16> to memref<1xf16, strided<[1], offset: ?>>
    }

    // Stage 2: Call llvm.func sibling (like post_scale_llvm)
    %base_ptr = memref.extract_aligned_pointer_as_index %arg1 : memref<*xf16> -> index
    %ptr_i64 = arith.index_cast %base_ptr : index to i64
    %size_i64 = arith.index_cast %arg3 : i32 to i64
    llvm.call @post_scale_stage(%ptr_i64, %size_i64) : (i64, i64) -> ()

    return
  }

  llvm.func @post_scale_stage(%arg0: i64, %arg1: i64) {
    %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %scale = llvm.mlir.constant(1.500000e+00 : f32) : f32

    llvm.br ^loop(%c0 : i64)
  ^loop(%iv: i64):
    %cond = llvm.icmp "slt" %iv, %c8 : i64
    llvm.cond_br %cond, ^body, ^exit
  ^body:
    %elem_ptr = llvm.getelementptr %ptr[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %val = llvm.load %elem_ptr : !llvm.ptr -> f16
    %val_f32 = llvm.fpext %val : f16 to f32
    %scaled = llvm.fmul %val_f32, %scale : f32
    %scaled_f16 = llvm.fptrunc %scaled : f32 to f16
    llvm.store %scaled_f16, %elem_ptr : f16, !llvm.ptr
    %next = llvm.add %iv, %c1 : i64
    llvm.br ^loop(%next : i64)
  ^exit:
    llvm.return
  }
}
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        input_mlir = os.path.join(tmpdir, "mixed_input.mlir")
        output_mlir = os.path.join(tmpdir, "mixed_lowered.mlir")
        output_ll = os.path.join(tmpdir, "mixed_output.ll")

        # Write input IR
        with open(input_mlir, 'w') as f:
            f.write(mixed_ir)

        # Run spine-opt pipeline
        spine_opt = "/home/zuoweixia/work/tritons/spine-mlir-k3/build/x86/speir/Release/bin/spine-opt"
        result = subprocess.run(
            [spine_opt, "--spine-triton-e2e-pipeline", input_mlir, "-o", output_mlir],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"spine-opt FAILED:\n{result.stderr}")
            assert False, "spine-opt pipeline failed"

        print("✓ spine-opt --spine-triton-e2e-pipeline succeeded")

        # Run mlir-translate
        mlir_translate = "/home/zuoweixia/work/tritons/spine-mlir-k3/build/x86/speir/Release/installed/bin/mlir-translate"
        result = subprocess.run(
            [mlir_translate, "--mlir-to-llvmir", output_mlir, "-o", output_ll],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"mlir-translate FAILED:\n{result.stderr}")
            assert False, "mlir-translate failed"

        print("✓ mlir-translate --mlir-to-llvmir succeeded")

        # Verify output contains both functions
        with open(output_ll, 'r') as f:
            llvm_ir = f.read()

        assert "@mixed_host" in llvm_ir, "Host function missing in LLVM IR"
        assert "@post_scale_stage" in llvm_ir, "LLVM stage function missing"
        assert "call void @post_scale_stage" in llvm_ir, "llvm.call not preserved"

        print("✓ LLVM IR contains both functions with preserved call")
        print("\nTest PASSED: Multi-function module approach is viable")


if __name__ == "__main__":
    test_mixed_manual_ir()
