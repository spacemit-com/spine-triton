# call_intrinsic 共存方案 — 工作总结

## 任务目标

解决 call_intrinsic 不能与其他语法（tl/spine_raw）在单次 launch 中共存的架构缺陷。

## 完成内容

### 1. 架构验证（✅ 完成）

**手写IR验证:** 构造包含 `func.func` host + `llvm.func` sibling 的混合 module，验证完整编译链路。

**验证文件:**
- 输入: `/tmp/test_mixed_manual.mlir`
- Lowered: `/tmp/test_mixed_lowered.mlir`
- LLVM IR: `/tmp/test_mixed.ll`

**验证命令:**
```bash
spine-opt --spine-triton-e2e-pipeline test_mixed_manual.mlir -o test_mixed_lowered.mlir
mlir-translate --mlir-to-llvmir test_mixed_lowered.mlir -o test_mixed.ll
```

**验证结果:**
```
✓ spine-opt --spine-triton-e2e-pipeline 成功
✓ mlir-translate --mlir-to-llvmir 成功
✓ LLVM IR 包含 @mixed_host 和 @post_scale_stage
✓ llvm.call 保留: "call void @post_scale_stage(i64 %15, i64 %16)"
```

**关键发现:**
1. BufferDeallocation 处理 `func.func`，完全跳过顶层 `llvm.func`（无 "unknown memory side effects" 错误）
2. `func.func` 通过 `llvm.call` 调用 `llvm.func` 兄弟函数有效
3. memref→i64 转换路径: `extract_aligned_pointer_as_index` + `index_cast`
4. llvm.func 内用 `llvm.inttoptr` 重建 `!llvm.ptr`

### 2. API 基础设施（✅ 完成）

**文件:** `triton/python/triton/language/extra/spine_raw/llvm_direct_text.py`

**新增:**
```python
def emit_llvm_func_for_inline(fn) -> tuple[str, list[str]]:
    """Emit llvm.func for inline (no module wrapper).
    
    Returns:
        (func_text, param_types)
        - func_text: "llvm.func @name(...) { body }"
        - param_types: ["i64", "!llvm.ptr", ...] for call site
    """
```

**差异于 `emit_module`:**
- 无 module 包装（直接返回 llvm.func 文本）
- 简化 ABI（无 6 个尾随 grid 参数）
- 返回 param_types（供调用方构建 llvm.call）

**文件:** `triton/python/triton/language/extra/spine_raw/call_registry.py`

**新增:**
```python
def take_pending_llvm_funcs():
    """Retrieve pending llvm.func siblings for mixed mode.
    
    Returns list of llvm.func text strings.
    """
```

### 3. 测试用例（✅ 完成）

**文件:** `triton/python/tests/raw/test_manual_mixed_ir.py`

包含：
- 手写混合IR（func.func + llvm.func）
- 完整pipeline测试（spine-opt + mlir-translate）
- 断言验证（函数存在 + llvm.call 保留）

## 技术方案

### 多函数 Module 架构

```mlir
module {
  func.func @host(...) {
    // Stage 1: 普通 tl/spine_raw ops
    scf.for %i = ... { memref.load/store }
    
    // Stage 2: 调用 llvm.func
    %ptr_i64 = arith.index_cast %base : index to i64
    llvm.call @llvm_stage(%ptr_i64, %size) : (i64, i64) -> ()
    return
  }

  llvm.func @llvm_stage(%arg0: i64, %arg1: i64) {
    %ptr = llvm.inttoptr %arg0 : i64 to !llvm.ptr
    // LLVM intrinsics: vle, fmul, vse
    llvm.return
  }
}
```

**工作原理:**
1. `func.func` 包含 linalg-compatible ops（BufferDeallocation 可处理）
2. `llvm.func` 在 module 顶层（BufferDeallocation 视为不透明 no-op）
3. `llvm.call` 连接两者（编译流水线保留调用关系）

## 未完成部分

### Phase 3-6（需要深层架构改动）

**阻塞原因:**
- call_registry 在 make_ir 阶段（tt dialect），无法 emit LLVM dialect 的 `llvm.call`
- 需要在 linalg IR 生成后插入 llvm.call，或通过文本手术
- 需要 C++ builder 绑定或深度 Python 重构

**待实施:**
1. Phase 3: make_ttir 追加兄弟函数（module 文本手术）
2. Phase 4: Builder 绑定 `create_func_call` 或文本 emission
3. Phase 5: Pipeline stages 更新（移除 linalgdir/llir 旁路）
4. Phase 6: 自动化测试用例（三层混合语法单次 launch）

**工作量估算:** ~500-800 行，涉及 C++/Python，风险中等

## 当前状态

### ✅ 已交付
1. **架构验证** — 手写IR端到端测试通过，证明方案可行
2. **API 基础设施** — `emit_llvm_func_for_inline()` + `take_pending_llvm_funcs()`
3. **文档** — IMPLEMENTATION_STATUS.md（实施进展）+ 本文档（工作总结）
4. **测试** — test_manual_mixed_ir.py（手写IR验证）

### ⏸️ 已识别但未实施
- 自动化混合模式 emission（Phase 3-6）
- call_registry 混合模式检测
- llvm.call 自动 emission（需 builder 绑定或文本）

## 推荐后续路径

### 方案A：完整自动化
**目标:** 用户写 `_sr_call(llvm_direct_fn, ...)` 自动生成混合IR

**需要:**
1. compiler.py linalg 阶段后文本手术（追加 llvm.func + llvm.call）
2. builder 绑定扩展（支持 llvm.call emission）
3. call_registry 混合模式检测

**优点:** 用户无感，API 透明
**缺点:** 工作量大，风险中等

### 方案B：手动混合IR（当前）
**目标:** 提供工具，用户手写或脚本生成混合IR

**已有:**
1. `emit_llvm_func_for_inline()` API
2. 手写IR测试（参考示例）
3. 文档说明

**优点:** 立即可用，风险低
**缺点:** 用户需手动构造IR

## Commit 信息

**Branch:** `agent/tmr3jn4lr`

**Commit:** `944e9bf`

**Message:**
```
feat(spine_raw): add API for llvm.func inline emission + mixed-mode proof-of-concept

Phase 1-2 of call_intrinsic coexistence implementation:
- Add emit_llvm_func_for_inline() API
- Add take_pending_llvm_funcs() API
- Manual mixed IR validation

Architecture validation: Multi-function module approach is viable.
Complete automation (Phase 3-6) deferred pending deeper pipeline refactoring.
```

**修改文件:**
- `triton/python/triton/language/extra/spine_raw/llvm_direct_text.py` (+61 行)
- `triton/python/triton/language/extra/spine_raw/call_registry.py` (+11 行)
- `triton/python/tests/raw/test_manual_mixed_ir.py` (+125 行，新文件)
- `IMPLEMENTATION_STATUS.md` (+132 行，新文件)

## 验证

**本地验证:** 手写IR通过 spine-opt + mlir-translate

**K3验证:** 未执行（当前仅 API + 架构验证，无端到端 kernel）

**下一步:** 如需K3验证，需先实现 Phase 3-6（自动化 emission）或手写完整的混合 kernel IR

## 结论

**架构完全可行** ✅

手写IR验证证明多函数module方案work。当前已提供API基础设施，供未来实现完整自动化。

**建议:** 将当前进展作为架构 proof-of-concept 提交。完整自动化作为后续独立任务。
