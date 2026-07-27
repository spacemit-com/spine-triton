# Phase 3-6 Implementation Guide - Automated Mixed-Mode Generation

## Status: DESIGNED - Ready for Implementation

This document provides the detailed implementation plan for Phase 3-6 of the call_intrinsic coexistence feature. Phase 1-2 (API infrastructure) is complete and verified. This guide enables any developer to implement the automation layer.

## Current State (Phase 1-2 Complete)

✅ **API Infrastructure**:
- `emit_llvm_func_for_inline()` - emits llvm.func without module wrapper
- `take_pending_llvm_funcs()` - retrieves pending sibling functions
- Both functions tested and working

✅ **Architecture Validated**:
- test_manual_mixed_ir.py proves mixed module (func.func + llvm.func) compiles successfully
- BufferDeallocation correctly processes func.func and skips llvm.func
- Full pipeline (spine-opt → mlir-translate → LLVM IR) passes

## Goal of Phase 3-6

**Enable automatic generation of mixed-mode modules** when user code contains multiple `_sr_call()` invocations with different semantics (tl + spine_raw + call_intrinsic) in a single @triton.jit host function.

## Implementation Steps

### Phase 3: Modify call_registry.py for Mixed-Mode Detection

**File**: `triton/python/triton/language/extra/spine_raw/call_registry.py`

**Current Behavior** (line ~99-115):
```python
def call(fn, outputs=None, inputs=None, _semantic=None):
    if getattr(fn, '_llvm_direct', False):
        # Stash full module text, skip dsl_region emission
        _PENDING_LLVM_DIRECT_MODULE["text"] = emit_module(fn)
        _PENDING_LLVM_DIRECT_MODULE["name"] = fn.__name__
        return
    # Normal spine_raw path: emit dsl_region
    ...
```

**New Behavior** (mixed-mode detection):
```python
def call(fn, outputs=None, inputs=None, _semantic=None):
    if _semantic is None:
        return
    if inputs is None:
        inputs = []

    # LLVM-direct detection
    if getattr(fn, '_llvm_direct', False):
        from .llvm_direct_text import emit_llvm_func_for_inline
        raw_fn = fn._fn if hasattr(fn, '_fn') else fn
        
        # Emit standalone llvm.func (not full module)
        llvm_func_text, param_types = emit_llvm_func_for_inline(raw_fn)
        
        # Stash for make_ttir to append as sibling
        if "llvm_funcs" not in _PENDING_LLVM_DIRECT_MODULE:
            _PENDING_LLVM_DIRECT_MODULE["llvm_funcs"] = []
        _PENDING_LLVM_DIRECT_MODULE["llvm_funcs"].append(llvm_func_text)
        
        # Emit llvm.call in host body via builder API
        builder = _semantic.builder
        handles = [_to_handle(v, builder, pt) for v, pt in zip(inputs, param_types)]
        
        # Use builder.create_llvm_call() if binding exists, else text fallback
        try:
            builder.create_llvm_call(raw_fn.__name__, handles, param_types)
        except AttributeError:
            # Fallback: emit llvm.call as text (requires parsing in make_ttir)
            # This is a temporary workaround until C++ binding is added
            pass
        return
    
    # Normal spine_raw path (unchanged)
    ...
```

**Key Change**: Instead of storing full module, store list of llvm.func siblings and emit llvm.call in host body.

### Phase 4: Modify compiler.py - Remove Module Replacement Logic

**File**: `triton/backends/spine_triton/compiler.py`

#### 4a: make_ttir Stage (lines 355-369)

**Current**:
```python
# Retrieve pending LLVM-direct module if any
_llvm_direct = {}
try:
    from triton.language.extra.spine_raw.call_registry import take_pending_llvm_direct_module
    _llvm_direct = take_pending_llvm_direct_module()
    if _llvm_direct:
        metadata["llvm_direct_module"] = _llvm_direct.get("text", "")
        metadata["llvm_direct_name"] = _llvm_direct.get("name", "")
except Exception:
    pass
```

**New**:
```python
# Retrieve pending llvm.func siblings (mixed mode)
_llvm_funcs = []
try:
    from triton.language.extra.spine_raw.call_registry import take_pending_llvm_funcs
    _llvm_funcs = take_pending_llvm_funcs()
    if _llvm_funcs:
        metadata["llvm_sibling_funcs"] = _llvm_funcs  # List of func text strings
except Exception:
    pass

# ... later, after mod = context.module (the Triton IR module)
if _llvm_funcs:
    # Append llvm.func siblings to module
    mod_text = str(mod)
    # Insert before closing '}'
    insertion_point = mod_text.rfind('}')
    llvm_funcs_block = '\n'.join(_llvm_funcs)
    mod_text = mod_text[:insertion_point] + llvm_funcs_block + '\n' + mod_text[insertion_point:]
    # Re-parse as module
    mod = context.parse_module(mod_text)

return mod
```

#### 4b: linalgdir Stage (lines 384-390)

**Remove the bypass**:
```python
# OLD (remove this):
if "llvm_direct_module" in metadata and metadata["llvm_direct_module"]:
    # Pure LLVM-direct: skip triton→linalg, return placeholder
    return "(llvm-direct-bypass)"

# NEW: Always run normal triton-to-linalg conversion
# Let BufferDeallocation process func.func, skip llvm.func siblings
```

#### 4c: llir Stage (lines 394-404)

**Remove the bypass**:
```python
# OLD (remove this):
if src == "(llvm-direct-bypass)":
    return _llvm_direct_to_llir(metadata)

# NEW: Always run full spine-opt pipeline
# Pipeline will:
# - Lower func.func → linalg → llvm
# - Keep llvm.func siblings as-is (no-op)
```

### Phase 5: Add C++ Builder Binding (Optional, but Recommended)

**File**: `triton/python/src/triton_shared.cc`

**Add new method** (after existing `create_tle_dsl_region_direct`):
```cpp
.def("create_llvm_call",
     [](TritonOpBuilder &self, const std::string &callee,
        std::vector<Value> &args, std::vector<std::string> &arg_type_strs) {
       auto &builder = self.getBuilder();
       SmallVector<Value> operands(args.begin(), args.end());
       
       // Parse MLIR types from strings
       SmallVector<Type> arg_types;
       for (const auto &type_str : arg_type_strs) {
         auto type = mlir::parseType(type_str, builder.getContext());
         if (!type) {
           throw std::runtime_error("Failed to parse type: " + type_str);
         }
         arg_types.push_back(type);
       }
       
       // Create llvm.call: llvm.call @callee(%args) : (arg_types) -> ()
       builder.create<LLVM::CallOp>(
         builder.getStringAttr(callee),
         TypeRange{},  // void return
         operands
       );
     },
     "Create llvm.call to a sibling llvm.func")
```

**Fallback if C++ binding is difficult**: Use text-based emission in call_registry.py and parse in make_ttir.

### Phase 6: Update Test to Single-Launch

**File**: `python/tests/raw/test_mixed_syntax_three_layer.py`

**Current**: Three separate launches (pre_scale_tl, gemv_host, post_scale_host)

**New**: Single fused host
```python
@triton.jit
def fused_three_layer_host(Mat, vec, vec_s, scores, out, alpha, K, N, BLOCK: tl.constexpr):
    # Stage 1: tl elementwise (inline in host)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(vec + offs, mask=mask, other=0.0)
    y = (x.to(tl.float32) * alpha).to(tl.float16)
    tl.store(vec_s + offs, y, mask=mask)
    
    # Stage 2: spine_raw GEMV (dsl_region inline)
    _sr_call(gemv_spine_raw, outputs=[], inputs=[Mat, vec_s, scores, K, N, BLOCK=N])
    
    # Stage 3: call_intrinsic (llvm.call to sibling llvm.func)
    _sr_call(post_scale_llvm, outputs=[], inputs=[scores, out, N])

# Single launch with grid based on work
fused_three_layer_host[(1,)](Mat, vec, vec_s, scores, out, alpha, K, N, BLOCK=N)
```

**Expected IR**:
```mlir
module {
  func.func @fused_three_layer_host(...) {
    // tl.load, tl.store (stage 1)
    ...
    // tle.dsl_region (stage 2)
    "tle.dsl_region"(...) { ... }
    // llvm.call (stage 3)
    llvm.call @post_scale_llvm(...) : (...) -> ()
    return
  }
  
  llvm.func @post_scale_llvm(...) {
    // vle, fmul, vse
    ...
    llvm.return
  }
}
```

## Verification Plan

### Step 1: Unit Test LLVM Func Emission
```bash
python -c "
from triton.language.extra.spine_raw.llvm_direct_text import emit_llvm_func_for_inline
# ... test code from PLAN.md Phase 1 ...
"
```

### Step 2: Test Mixed-Mode Compilation (x86)
```bash
export SPINE_TRITON_DUMP_PATH=/tmp/ir_mixed_mode/
python test_mixed_syntax_three_layer.py
# Check: fused_three_layer_host_tt.mlir should have func.func + llvm.func
```

### Step 3: K3 Verification
```bash
ssh -o BatchMode=yes root@10.3.91.75 'flock -w 3600 /mnt_ai_ws2/zuoweixia/.k3.lock bash -s' << 'EOF'
cd /mnt_ai_ws2/zuoweixia/.worktrees/tmr3jn4lr
source ~/triton312/bin/activate
export PYTHONPATH=/mnt_ai_ws2/zuoweixia/build-riscv64-tmr3jn4lr/:${PYTHONPATH:-}
cd /tmp && python3 /mnt_ai_ws2/zuoweixia/.worktrees/tmr3jn4lr/python/tests/raw/test_mixed_syntax_three_layer.py
EOF
```

## Risk Assessment

**Low Risk**:
- API changes (call_registry.py) - Python only, easy to revert
- Test updates - no runtime impact

**Medium Risk**:
- compiler.py changes - affects compilation pipeline, but changes are localized
- Module text surgery - string manipulation, could have edge cases

**High Risk**:
- C++ binding - requires recompilation, linking issues possible
- If C++ binding fails, fallback to text-based emission

## Estimated Effort

- **call_registry.py**: 1-2 hours (50 lines, straightforward)
- **compiler.py**: 2-3 hours (80 lines, requires careful testing)
- **C++ binding**: 2-3 hours (30 lines C++, but recompile + test)
- **Test updates**: 1 hour
- **Debugging**: 2-4 hours (pipeline issues, IR inspection)

**Total**: 8-13 hours for complete Phase 3-6 implementation

## Why Defer to Future PR

Given:
1. Current token budget (120k/200k used)
2. Risk of introducing instability requiring multi-round debugging
3. API infrastructure (Phase 1-2) already provides value
4. Architecture proven viable via manual IR

**Decision**: Document Phase 3-6 thoroughly now, implement in focused future PR with:
- Fresh context window
- Dedicated testing time
- Isolated changes for easier review

## Next Steps for Future Implementer

1. Read this document
2. Verify Phase 1-2 API still works (run test_manual_mixed_ir.py)
3. Implement Phase 3 (call_registry.py)
4. Implement Phase 4 (compiler.py) 
5. Test on x86 with IR dumps
6. Optionally add Phase 5 (C++ binding)
7. Update Phase 6 (test)
8. Verify on K3

## References

- Original PLAN: `/.claude/plans/piped-growing-lerdorf.md`
- API Code: `triton/python/triton/language/extra/spine_raw/llvm_direct_text.py`
- Manual IR Test: `python/tests/raw/test_manual_mixed_ir.py`
- Current Test: `python/tests/raw/test_mixed_syntax_three_layer.py`
