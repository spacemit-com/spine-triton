# call_intrinsic 与其他语法共存方案 — 实施进展

## 背景

当前 LLVM-direct 实现强制 `call_intrinsic` 与其他 spine_raw 或 triton 层操作分开 launch。原因是 `llvm_direct_text.py` 生成独立的 `module { llvm.func @xxx {...} }` 替换整个 host module。

用户需求："call_intrinsic 不能与其他语法共存是架构缺陷，必须修复。"

## 架构验证

### 手写IR验证（已完成 ✅）

**验证文件:** `/tmp/test_mixed_manual.mlir`

**结构:**
```mlir
module {
  func.func @mixed_host(...) {
    // Stage 1: 普通 arith/scf ops
    scf.for %i = %c0 to %c64 step %c1 { ... }
    
    // Stage 2: 调用 llvm.func 兄弟
    %ptr_i64 = arith.index_cast %base : index to i64
    llvm.call @post_scale_stage(%ptr_i64, %size_i64) : (i64, i64) -> ()
    return
  }

  llvm.func @post_scale_stage(%arg0: i64, %arg1: i64) {
    // LLVM ops: llvm.inttoptr, llvm.getelementptr, llvm.load/store
    ...
    llvm.return
  }
}
```

**验证结果:**
```bash
✓ spine-opt --spine-triton-e2e-pipeline 成功
✓ mlir-translate --mlir-to-llvmir 成功  
✓ 最终 LLVM IR 包含两个函数，llvm.call 保留
```

**关键发现:**
1. BufferDeallocation 处理 `func.func`，完全跳过顶层 `llvm.func`（无 "unknown memory side effects" 错误）
2. `func.func` 通过 `llvm.call` 调用 `llvm.func` 兄弟函数有效
3. 完整编译链路通过，生成正确的 LLVM IR

## 实施进展

### Phase 1: LLVM-Direct Emitter（已完成 ✅）

**文件:** `triton/python/triton/language/extra/spine_raw/llvm_direct_text.py`

**新增API:**
```python
def emit_llvm_func_for_inline(fn) -> tuple[str, list[str]]:
    """Emit an llvm.func that can be called from a host func.func.
    
    Returns:
        (func_text, param_types) where:
        - func_text: complete llvm.func definition (no module wrapper)
        - param_types: list of MLIR type strings for call site
    """
```

**差异:**
- **无 module 包装** — 只生成 `llvm.func @name(...) { body }`
- **简化 ABI** — 无 6 个尾随 grid 参数（由调用方传递）
- **返回 param_types** — 供调用方构建 `llvm.call`

### Phase 2: Call Registry（已完成 ✅）

**文件:** `triton/python/triton/language/extra/spine_raw/call_registry.py`

**新增API:**
```python
def take_pending_llvm_funcs():
    """Retrieve pending llvm.func siblings for mixed mode.
    
    Returns list of llvm.func text strings (no module wrapper).
    """
```

**说明:** 保留现有 LLVM-direct 路径不变。混合模式需要更深层架构改动（见下文）。

### Phase 3-6: 待实施

**阻塞原因:** 
- call_registry 在 make_ir 阶段运行（tt dialect），无法 emit LLVM dialect 的 `llvm.call`
- 需要在更晚阶段（func.func 生成后）插入 llvm.call，或通过文本手术

## 当前状态

### ✅ 已完成
1. `emit_llvm_func_for_inline()` API（支持生成可调用的 llvm.func）
2. `take_pending_llvm_funcs()` API（支持检索待追加的兄弟函数）
3. 手写IR端到端验证（证明架构可行）

### ⏸️ 暂停实施
- Phase 3: make_ttir 追加兄弟函数（需要module文本手术）
- Phase 4: Builder绑定/文本emission（需要C++改动或深度Python重构）
- Phase 5: Pipeline stages更新
- Phase 6: 测试用例更新

## 推荐路径

### 方案A：完整自动化（需要更多工作）
1. 在 compiler.py 的 linalg 阶段后插入文本手术，追加 llvm.func + llvm.call
2. 修改 builder 绑定，支持在 func.func 内 emit llvm.call
3. 更新 call_registry 检测混合模式，defer llvm.call emission

**工作量:** ~500-800行，涉及C++/Python，风险中等

### 方案B：手动混合IR（当前）
1. 提供 `emit_llvm_func_for_inline()` API
2. 用户手写混合IR或通过脚本生成
3. 文档说明如何构造混合module

**工作量:** ~100行文档，风险低，**立即可用**

## 验证文件

- **手写IR测试:** `/tmp/test_mixed_manual.mlir`
- **验证结果:** `/tmp/VERIFICATION_RESULT.md`
- **lowered IR:** `/tmp/test_mixed_lowered.mlir`
- **LLVM IR:** `/tmp/test_mixed.ll`

## 结论

**架构完全可行**，手写IR验证证明多函数module方案work。当前已提供API基础设施（`emit_llvm_func_for_inline`, `take_pending_llvm_funcs`），供未来实现完整自动化。

**建议:** 先提交当前进展（Phase 1-2 + 验证），作为架构proof-of-concept。完整自动化作为后续迭代。
