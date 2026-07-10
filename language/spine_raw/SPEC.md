# tle.raw eDSL 规格说明书（svector 层级）

**版本** v0.1-draft
**对象模块** `triton.language.extra.spine_raw`（源码 `language/spine_raw/`）

> **状态** 本文处于草案（Draft）状态，规格可能变动。文中每个语言构造标注「落地状态」
> （第 4 章），区分「语言已定义」与「spine-mlir 当前实现程度」；二者分离，落地状态的
> 演进不改变语言定义。已验证的可运行写法见 `python/tests/test_raw_mv_svector.py`，本文
> 不含算子实例。

---

## 术语与约定

**vector 层级** — Triton/MLIR 编译栈中，位于 tile 语言之下、RVV 之上的一层 MLIR，由
`vector` / `vector_ext` / 可伸缩向量 / `memref` / `scf` / `arith` 等方言构成。spine-mlir
在这一层往下 lower 至 RVV。

**RVV** — RISC-V 向量扩展（"V" extension）。

**SEW / LMUL / VLMAX** — SEW 为元素位宽；LMUL 为向量寄存器分组倍数；
VLMAX = LMUL × VLEN / SEW。

**VL** — 活跃向量长度，一条向量指令实际处理的元素个数。本文以 VL 记向量值的当前宽度。

**vscale** — 可伸缩向量的缩放因子。

**raw kernel / host kernel** — raw kernel 为 `@raw_kernel` 装饰、以本 eDSL 编写的函数；
host kernel 为普通 `@triton.jit` 函数，通过 `call()` 触发 raw kernel。

**第一段 / 第二段** — 第一段指 raw kernel 源码经 AST 翻译为 vector 层级 MLIR；第二段指
该 MLIR 经 `tle.dsl_region` 注入主流水线并 lower 至 RVV。详见 2.7。

**落地状态** — 语言构造在 spine-mlir 当前实现中的支持程度，取值「直达 / 回退 / 待扩」，
定义见 4.1。

约定：「须」表示规范性要求，「不得」表示规范性禁止，其余为陈述性说明。原语签名以
Python 形式给出，`->` 右侧为返回值类别。语义以逐元素公式表述，记向量值 `v` 的第 `i`
个元素为 `v[i]`，`i ∈ [0, VL)`。

---

## 1. 引言

### 1.1 目的：为 vector 层级提供直接编程入口

Triton 编译栈自上而下为三层：tile 语言（`tl.*`，用户描述「算什么」）；vector 层级 MLIR
（`vector` / `vector_ext` / 可伸缩向量 / `memref`，编译器内部一层，用户通常接触不到）；
RVV 指令。tile 到 vector 之间是编译器的自动向量化 / tiling / bufferization，vector 到 RVV
之间是自动可伸缩化 / LMUL 分组 / 寄存器分配。

**tle.raw 的根本目的，是把 vector 层级暴露为一个 Python 可写的编程入口。** 用 tle.raw
编写的 raw kernel，其编译产物直接就是 vector 层级的 MLIR，跳过其上的全部自动 pass。
用户由此直接在 vector 层级编程，而非委托 tile 语言去间接生成。

### 1.2 为什么是 vector 层级

vector 层级是表达能力与硬件控制的平衡点。

| 层级                  | 用户表达的内容                       | 局限                                         |
| --------------------- | ------------------------------------ | -------------------------------------------- |
| tile 语言             | 算什么                               | 够不到硬件细节，向量化/布局由编译器代劳      |
| **vector 层级** | 向量 op + 寄存器驻留 + 显式访存/布局 | 无（结构化、可读，且无需逐条排指令与寄存器） |
| RVV / 汇编            | 逐条指令 + 物理寄存器                | 过低，手写不可维护，寄存器须自行分配         |

在 vector 层级，用户可结构化地表达一个向量程序，同时把可伸缩化、LMUL 分组、寄存器
分配、向 RVV 的翻译留给编译器。往上（tile 语言）够不到硬件，往下（汇编）不可维护——
vector 层级恰是二者之间可直接编程的那一层。

### 1.3 由此带来的能力

「能直接操作 vector 层级」这一根本目的，推出下列能力（均为推论，非独立目标）：

- **IME/AME 专有单元**vfwmadot、vpack 等能力对应的
  `vector_ext.*` 能显式请求 vector/vector_ext 后端已实现的能力。
- **显式控制布局**：tiling、pack、访存 stride、转置由用户书写，不受编译器启发式左右。
- **寄存器驻留**：vector 层级的 `scf.for` + 向量 iter_arg 天然使累加器跨迭代驻留寄存器。
- **快速交付**：新增高性能算子只需在 vector 层级书写，不需修改编译器 lowering。

### 1.4 定位

tle.raw 与通用自动路径分工共存，二者均为长期机制：前者管特例（在 vector 层级手工控制
热点算子），后者管通用（覆盖全部算子、可移植）。

### 1.5 范围

本规格定义编程模型（第 2 章）、类型系统（第 3 章）、命名与落地状态（第 4 章）、语言
子集（第 5 章）、指令空间（第 6 章）、约束边界（第 7 章）。指令空间以 RVV 完整指令类
为蓝本组织，标注各类在 spine-mlir 的落地状态；语言定义不以当前实现为界（4.1）。

---

## 2. 编程模型

### 2.1 总述：tle.raw = 手写执行方案

tile 语言里用户描述「算什么」，编译器决定「怎么在硬件上跑」。tle.raw 把「怎么跑」这份
完整**执行方案**交给用户。写一个 raw kernel，就是亲自决定下列四点——每一点都是编译器
在 tile 层替你做、而 tle.raw 交还给你的决策：

| 轴         | tile 语言（编译器决定） | tle.raw（你决定）                                                   |
| ---------- | ----------------------- | ------------------------------------------------------------------- |
| A 分解     | auto-vectorize + tiling | 向量宽度 VL、tile 大小、算法到 lane 的映射                          |
| B 选指令   | 编译器挑指令            | 用哪个硬件单元：矩阵引擎 vmadot / 向量 vmacc / 归约 / vpack         |
| C 数据编排 | 编译器插搬运 / 分配     | DDR↔TCM↔寄存器 分级、布局 / pack / 转置、寄存器驻留（少往返）     |
| D 调度     | 编译器排循环 / 寄存器   | 循环结构、累加器个数（指令级并行 / 软件流水）、寄存器压力 vs 并行度 |

以下 2.2–2.5 逐轴展开。同一算子在这四条轴上的不同取值，即不同的实现策略。

### 2.2 A：分解与向量宽度

「分解」指把一个算子拆成一条条向量运算：先定每条向量运算多宽，再把远大于该宽度的数据
量拆成一块块喂进去。tile 语言里这两步由编译器自动完成；tle.raw 里都由用户显式写出。

**① 每条向量运算多宽（向量宽度）。** 分两种情形：

- 逐元素与归约类原语（`vadd` / `vmacc` / `vreduce_sum` 等）作用在活跃向量长度 VL 上。
  VL 由 `vconfig` 设定，用户可选；编译器负责其可伸缩化、LMUL 分组、尾部处理。
- 矩阵单元类原语（`vmadot`）的操作数形状由硬件 cubic 规格固定（K3 f16 为 8×8×8），
  用户不可选，只能按此规格准备 tile 去匹配（cubic 规格是硬件事实，见术语表）。

**② 把大数据量拆到这个宽度（tile / sub-tile 拆分）。** 算子的数据量通常远大于一条向量
运算的宽度——例如要算 N=4096 列，而一条向量指令一次只处理 VL=64 个元素。这个
`4096 → 64` 的拆分：tile 语言里编译器会把大 tile 自动切到向量宽度并生成循环；tle.raw
里**没人替你切**，须由用户用 `for` 循环 + 显式偏移把大块拆成若干 VL 宽的 sub-tile 逐块
处理（如 host 传入一块 256 宽的列，raw kernel 内切成 4 个 64 宽 sub-tile，各自
vload / 计算 / vstore）。

**③ 关于 VL 的实现说明。** 向量层不在用户可见处执行 `vsetvli`；运行期显式向量长度
（真 strip-mine）为待扩项（6.1），当前实现为定长 VL。

### 2.3 B：指令选择

同一算子可由不同硬件单元实现，用户通过选用不同原语来选指令。以矩阵向量乘为例：

```
   同一算子 C = A · x
        ├── 归约风格：vmacc（逐元素乘加） + vreduce_sum（横向求和）
        │             → 向量 FMA 指令
        └── 矩阵引擎：vmadot（cubic tile 乘加，直接产宽结果）
                      → vfwmadot 矩阵指令
```

不同选择的指令数、访存与性能不同。编译器在 tile 层会替你选其一；在 tle.raw 中由你选。

### 2.4 C：数据编排（含值与内存模型）

svector 面对一个**三级存储层次**，数据在其间的搬运全部显式，编译器不自动插入拷贝：

```
   DDR（外部内存，容量大、慢）
    │  vload / vstore              ← 输入输出的最终归宿
    ▼
   TCM（片上暂存，快；alloc 分配）  ← 显式暂存热数据（可选）
    │  vload / vstore
    ▼
   向量寄存器（vec 值, SSA, 最快）  ← 计算只在这里发生
        vzero/vsplat 构造 · vadd/vmul/vmacc/vmadot 计算 · vreduce_sum→标量
```

三种经典场景：

1. **分级暂存**：`alloc` 在 TCM 开 scratch，把反复访问的数据从 DDR 搬到 TCM 再用。
2. **布局与重排**：`smt.vpack.vv`加速；转置/跨步读为待扩（6.2）。
3. **寄存器驻留（少往返）**：计算全程在寄存器，内存只在开头 `vload`、结尾 `vstore`
   各碰一次。关键机制——**循环内对循环外变量同名重赋，该值即跨迭代驻留寄存器**：

```
   acc = vzero(f32)                ← 循环外构造，占一个向量寄存器
   for k in range(...):
       acc = vmacc(acc, x, y)      ← 同名重赋 → acc 成为 scf.for 的 iter_arg，
                                     全程在寄存器中更新，不写回内存
   vstore(C, ..., acc)             ← 循环结束写回内存一次
```

该行为由 vector 层级语义保证（`scf.for` + 向量 iter_arg → 向量 phi → 向量寄存器），
非由任何显式 API 提供。iter_arg 识别规则见 5.3。

### 2.5 D：调度

在 raw kernel 内（vector 层级），用户能控的调度是**循环结构**与**累加器的个数 / 宽度**。

- **累加器个数 → 并发、隐藏延迟**：只用一个累加器时，循环内 `acc = vmacc(acc, …)`
  形成循环携带依赖，相邻迭代的乘加须串行等待。写多个独立累加器变量（`acc0…acc3`，各自在
  循环内重赋 → 各成一个 `scf.for` iter_arg → 各占一个向量寄存器），更新链互不依赖，可并发
  于流水线，隐藏 FMA 延迟。（mv 用 4 个累加器，已验证。）
- **寄存器压力（间接）**：累加器越多、越宽，占用的向量寄存器越多。用户通过「写几个累加器、
  每个多宽」间接影响寄存器压力，但**物理寄存器的分配与溢出由编译器处理，用户不直接指派
  寄存器**（§1.2）。按 LMUL 加宽累加器目前未经 `vconfig` 暴露，属待扩。
-

### 2.6 raw kernel 与 host kernel 的接口

raw kernel 不独立启动，由 host kernel（`@triton.jit`）在其体内用 `call()` 触发并传参。
下面的骨架展示两边怎么写、怎么对应（函数体用第 6 章的原语，此处省略）：

```python
import triton, triton.language as tl
import triton.language.extra.spine_raw as tle
from triton.language.extra.spine_raw import raw_kernel, call

# ① raw kernel：每个形参带类型注解
@raw_kernel
def my_raw(A: tle.mem(tle.f16),            # 输入指针
           C: tle.mem(tle.f32, out=True),  # 输出指针（out=True 才能写回）
           col: tle.index,                 # 标量
           M:   tle.index):
    ...                                     # 函数体：vload / vmacc / vstore …（第 6 章）

# ② host kernel：普通 @triton.jit；传给 raw 的动态标量放进 do_not_specialize
@triton.jit(do_not_specialize=["M"])
def my_host(A, C, M, N, BLOCK: tl.constexpr):   # BLOCK = 每个 program 处理多少列
    pid = tl.program_id(0)
    col = pid * BLOCK                            # 本 program 负责的列块起始列
    call(my_raw, outputs=[], inputs=[A, C, col, M])
    #                                 │  │  │    └ M    → 形参 M
    #                                 │  │  └ col       → 形参 col（tracing 期算的值）
    #                                 │  └ C            → 形参 C
    #                                 └ A               → 形参 A

# ③ 启动：与普通 Triton kernel 完全一样，grid = 列数 / BLOCK 个 program
my_host[(N // 256,)](A, C, M, N, BLOCK=256)
```

规则：

- **`inputs` 按位置对应 raw kernel 形参**（第 `i` 个 input → 第 `i` 个形参）。可传指针，也可
  传 tracing 期算出的值（如 `pid * 256`）——这是把 JIT 期结果送进 raw kernel 的唯一通道。
- **输出走 `InOut` 指针**：`outputs` 恒为 `[]`；结果写进标了 `out=True` 的内存形参（如上例
  的 `C`），无返回值。
- **动态标量须免特化**：传给 raw 的动态标量（如 `M`）要在 `do_not_specialize=[...]` 里列出，
  否则被 Triton 当常量特化、丢失运行时句柄，传参失败。
- 指针类型自动桥接（`!tt.ptr` → `memref<*xT>`）、标量类型不符自动 `index_cast`，无需用户
  处理。其余使用约束（调用位置、导入路径）见 7.3。

## 3. 类型系统

### 3.1 元素类型与 SEW

元素类型以字符串常量表示，SEW 由元素类型推导，不单独指定。

| 常量    | 元素类型   | SEW |
| ------- | ---------- | --- |
| `f16` | 半精度浮点 | 16  |
| `f32` | 单精度浮点 | 32  |
|         |            |     |

### 3.2 向量值与宽度

向量值的元素类型由构造原语的 dtype 参数给出，宽度按 2.2 节确定。当前实现中，`vconfig`
设定的活跃向量长度为定长 `VLMAX = lmul × VLEN / SEW`（K3 VLEN=1024，SEW 由 dtype 推导：
f16 + lmul=1→64, lmul=2→128）；`vconfig` 的签名与参数见 6.1。

### 3.3 参数注解类型

raw kernel 的每个形参须带类型注解，其字符串原样作为 MLIR 类型。

| 注解                                   | MLIR 类型                                                  |
| -------------------------------------- | ---------------------------------------------------------- |
| `mem(f16)`                           | `memref<*xf16, #ptr.generic_space>`（只读，`In`）      |
| `mem(f32, out=True)`                 | `memref<*xf32, #ptr.generic_space>`（可写回，`InOut`） |
| `index`                              | `index`                                                  |
| `In["<type>"]` / `InOut["<type>"]` | `<type>`（原始形式）                                     |

`mem` / `index` 为 `In` / `InOut` 的语法糖。缺注解的形参在第一段翻译时报错。

## 4. 命名与落地状态约定

### 4.1 落地状态

每个语言构造标注落地状态。落地状态是 spine-mlir 的实现属性，不改变语言定义；同一份
用户代码在状态提升后无须修改。语言的能力面以 RVV 完整指令集为准，不以当前实现为界。

| 状态 | 含义                                                                       |
| ---- | -------------------------------------------------------------------------- |
| 直达 | 原语 1:1 落到已实现的 vector / vector_ext op，端到端可运行。               |
| 回退 | 语义有效，当前以等价形式降级实现（如谓词以比较+选择实现）。                |
| 待扩 | RVV / K3 具备该能力，spine-mlir 前端或后端尚未接入；语言已定义，落地待补。 |

### 4.2 operand form

算术、乘加、比较类原语依 RVV 的 operand form，由操作数类型自动选择后缀，语言不预设某
操作数须为标量。

| 后缀                        | 操作数         | 选择条件                |
| --------------------------- | -------------- | ----------------------- |
| `.vv`                     | 向量, 向量     | 两操作数均为向量        |
| `.vf`                     | 浮点标量, 向量 | 浮点标量与向量          |
| `.vx`                     | 整型标量, 向量 | 整型标量与向量          |
| `.vi`                     | 立即数, 向量   | 立即数与向量            |
| `.wv` / `.wx` / `.wi` | 宽向量源       | 加宽/缩窄运算的宽操作数 |

### 4.3 原语命名

现有原语沿用模块既有名（`vconfig` / `vload` / `vmacc` 等）。第 6 章中尚未实现的原语，
命名贴合其对应的 RVV intrinsic 助记符（如 `vslide` / `vrgather` / `vfsqrt`），以便与
RVV 文档对照。

---

## 5. 语言子集

第一段的 AST visitor 仅翻译下列 Python 结构，其余报 `NotImplementedError`。

### 5.1 支持的结构

| Python 结构                                   | 翻译                                 |
| --------------------------------------------- | ------------------------------------ |
| 带`In`/`InOut` 注解的函数定义             | `func.func`                        |
| `x = <expr>`                                | SSA 绑定                             |
| `for v in range(...)`（见 5.2）             | `scf.for`                          |
| `a + b` / `a - b` / `a * b`（index）    | `arith.addi` / `subi` / `muli` |
| `a // b`（index）                           | `arith.divui`                      |
| `a + b` / `a - b` / `a * b`（同型向量） | `arith.addf` / `subf` / `mulf` |
| 整数 / 浮点字面量                             | `arith.constant`                   |
| `<原语>(...)`（第 6 章）                    | 对应 MLIR op                         |

### 5.2 循环

`range` 接受 `range(stop)` 或 `range(start, stop, step)`，翻译为 `scf.for`。步长可为
`vconfig` 返回的宽度常量。

### 5.3 iter_args

visitor 预扫描循环体，凡「循环外已定义且循环内被重新赋值」的变量，自动成为 `scf.for`
的 iter_arg，末尾自动 `scf.yield`，顺序按变量名排序。此即 2.4 节寄存器驻留的实现机制。

### 5.4 不支持的结构

`if` / `while`（无标量分支；向量条件见 6.8）、多目标赋值、非
`range` 的 for、函数内定义函数、raw kernel 间互相调用、index 与向量混合运算。

---

## 6. eDSL 原语

本章完整覆盖 vector / vector_ext 层级的语义面，以 RVV 指令类为纲。每类给一段语义说明，
其后以签名清单逐条列出原语；每行格式为

```
原语签名                      # 逐元素语义          · 落地状态 → 目标 op / RVV 指令
```

记号：`v[i]` 为向量 `v` 第 `i` 个元素，`i ∈ [0, VL)`；`s` 标量；`m` mask 向量；`a op b`
中 `b` 可为向量或广播标量（operand form 由类型自动选，4.2）。落地状态取「直达 / 回退 /
待扩」（4.1），只表实现程度，不表语言成员——语言完整表达本层语义，不因慢、少用或可
规避而剔除。产 `vector_ext` op 的原语须以 generic form 生成。

### 6.1 配置

`vconfig` 对应 RVV `vsetvli`，设定活跃向量长度；SEW 由参与运算值的 dtype 推导（§3.1），
不作参数。

```
vconfig(avl, lmul) -> int      # VL = min(avl, VLMAX)，VLMAX = lmul × VLEN / SEW
```

当前实现签名已对齐为 `vconfig(avl, lmul)`：SEW 由 dtype 推导（§3.1，svector 循环以
f16 为粒度，基准 SEW=16），返回定长 `VLMAX = lmul × VLEN / SEW`（K3 VLEN=1024：
f16 + lmul=1→64, lmul=2→128, lmul=4→256, lmul=8→512）。`avl` 运行期收窄（真
strip-mine）为待扩：本轮忽略 `avl`，返回该 LMUL 下的定长 VLMAX。

### 6.2 访存

访存只有两个原语：`vload`（读）、`vstore`（写）。寻址方式由可选参数选择，分别对应不同
的 RVV 访存指令；不带可选参数时就是最基本的连续访存。

```
vload(ptr, index, stride=None, idx=None) -> vec
vstore(ptr, index, value, stride=None, idx=None)
```

参数：

| 参数 | 含义 |
|---|---|
| `ptr` | 内存指针（memref），访存的源 / 目标 |
| `index` | 起始元素偏移（基址），从 `ptr` 的第几个元素开始。二维坐标由用户自己压平传入（如 `ni*K + ki` 或 `col`） |
| `value`（仅 `vstore`） | 要写的值：向量 → 写 VL 个元素；标量 → 只写 1 个元素 |
| `stride` | 逐元素间隔（元素数）。缺省 = 连续；给非 1 值 = 跨步读/写（转置、读列用） |
| `idx` | 索引向量。给出则按索引聚集 / 散布（`ptr[idx[i]]`）；与 `stride` 互斥 |

寻址方式 → RVV 指令（`addr` = `index`）：

| 参数组合 | 逐元素语义 | RVV 指令 | 落地 |
|---|---|---|---|
| 默认（无 `stride`/`idx`） | 连续 `ptr[addr + i]` | `vload`→`vle`；`vstore`→`vse` | 直达 |
| 给 `stride` | 跨步 `ptr[addr + i*stride]`（转置/读列） | `vlse` / `vsse` | 待扩 |
| 给 `idx` | 索引 `ptr[idx[i]]`（gather/scatter） | `vluxei` / `vsuxei` | 待扩 |
| `value` 为标量（仅 `vstore`） | 单元素 `ptr[addr] = value`（reduce 回写） | `memref.store` | 直达 |

**与 `vle`/`vse` 的关系**：`vle`/`vse` 是 RVV 最基本的连续访存指令。`vload`/`vstore` 是 eDSL
的统一读写入口——**不带寻址参数时就正好 emit 成 `vle`/`vse`**；加 `stride` 改 emit 跨步的
`vlse`/`vsse`，加 `idx` 改 emit 索引的 `vluxei`/`vsuxei`。即寻址参数决定这一个 `vload`/`vstore`
最终落到哪条 RVV 访存指令。

> 当前实现只支持连续访存（`vle`/`vse`）；`index` 也接受二维元组 + 行距的写法（如
> `vload(B, (ni,ki), K)`，用行距 `K` 压平成 `addr=ni*K+ki`，读取仍连续）。`stride`（→`vlse`）
> 与 `idx`（→`vluxei`）为待扩。

### 6.3 扩展（vector_ext）

在 vector 层级的扩展，映射 `vector_ext` op。

```
vmadot(acc, x, y) -> vec                # 矩阵 cubic tile 乘加     · 直达 → vector_ext.matmul → vfwmadot
vpack(a, b, group_len) -> vec           #                         · 直达 → vector_ext.interleave → smt.vpack.vv
pack(src, (r0,c0), dst, shape, stride)  # 内存转置搬运便利封装     · 直达 → scf.for + transfer_read/write → vle/vse
alloc(shape, dtype, storage=tcm) -> memref           # TCM scratch 分配         · 直达 → memref.alloc → spine_thread_malloc
```

**`vmadot(acc, x, y)`** — 一个 cubic tile 的矩阵乘累加，`out[m,n] = acc[m,n] + Σ_k x[m,k]·y[n,k]`，
直接产出宽结果（不需归约）：

| 参数 | 含义 |
|---|---|
| `acc` | 累加器，`m×n` 当前值（展平成向量），结果加在其上 |
| `x` | 左矩阵 tile，`m×k`（展平） |
| `y` | 右矩阵 tile，`n×k`（展平，即右操作数按转置布局存） |

`m=n=k` 固定为硬件 cubic 规格（K3 f16 = 8×8×8），三者形状须匹配之。

**`vpack(a, b, group_len)`**

| 参数 | 含义 |
|---|---|
| `a`, `b` | 两个同型向量（待交织的两半） |
| `group_len` | pack粒度 （每段几个元素）；逐元素摆放由硬件 `smt.vpack.vv` 决定 |

**`pack(src, (r0,c0), dst, shape, stride)`** — 把 `src` 的一块转置搬进 scratch（软件循环）：

| 参数 | 含义 |
|---|---|
| `src` | 源内存指针（外部，行主序） |
| `(r0, c0)` | 从 `src` 的第 `r0` 行、第 `c0` 列起始（当前 `c0` 取 0，即整行块） |
| `dst` | 目标 scratch 缓冲（由 `alloc` 分配） |
| `shape` | `dst` 的 4 维 tiled 布局，如 `(1, K//VL, ROWS, VL)` |
| `stride` | `src` 的行距（相邻行相隔多少元素，= 列数 K） |

**`alloc(shape, dtype)`** — 在片上 TCM 申请一块 scratch 缓冲：

| 参数 | 含义 |
|---|---|
| `shape` | 缓冲各维大小；静态维写字面量，含运行时值的维为动态 |
| `dtype` | 元素类型（`f16` / `f32` / …） |

约束：`vmadot` 的 m/n/k 须等目标 cubic（K3 f16 为 8×8×8）；`vpack` 须 IME≥2、tile 为 2
的幂、末维 unit-stride。


### 6.4 逐元素运算

逐元素算术不设具名原语——**直接用 Python 运算符**,作用在 `vec` 值上即逐元素运算。整数
还是浮点、宽度多少,由操作数类型自动定;`b` 可为向量或广播标量。

```
算术:  a + b   a - b   a * b   a / b   a // b   a % b
位 :   a & b   a | b   a ^ b   ~a
移位:  a << b  a >> b
取负:  -a
比较:  a < b   a <= b   a == b   a != b   a > b   a >= b     → 得到 mask（见 6.8）
```

落地:同型逐元素 `+ - *` 直达 `arith.addf/subf/mulf`（浮点）/ `addi/subi/muli`（整数）;
比较直达 `arith.cmpf/cmpi`。除法按操作数类型分:`a / b` 浮点除（→ `vfdiv`）、`a // b` 整数向下取整除（→ `vdiv`）、`a % b` 取余（→ `vrem`）;位 / 移位（`& | ^ << >>`）逐元素。以上除 `+ - *` 与比较外,语义已定、按 RVV 对应指令待扩。

没有运算符的运算,用少量具名函数:

```
vmacc(acc, va, vb) -> vec        # va*vb + acc（含加宽累加）      · 直达 → arith.extf + math.fma → vfmacc
vmin/vmax(a, b) -> vec     # 逐元素 min / max            · 待扩 → vfmin/vfmax、vmin(u)/vmax(u)
sqrt/rsqrt(a) -> vec       # √a / 1/√a                   · 待扩 → vfsqrt/vfrsqrt
abs(a) -> vec              # |a|                         · 待扩 → vfsgnjx
cast(a, dtype) -> vec      # 类型转换(加宽/缩窄/整浮互转) · extf 直达;余待扩 → vfcvt/vzext/vnclip…
select(m, a, b) -> vec     # 按 mask 选,a if m else b    · 回退 → arith.select（原生 vmerge 待扩）
```

加宽:结果比操作数宽时(如 f16 输入、f32 结果),窄操作数隐式 `arith.extf` 到结果类型;
`fma` 的加宽累加即此。定点饱和/舍入类(vsadd/vaadd/vsmul/vnclip…)语义并入 `cast` 与运算符
族的饱和变体,按 RVV 待扩,不单列。

### 6.5 归约（向量 → 标量）

一个原语,`op` 选归约种类:

```
vreduce(op, v) -> 标量     # op ∈ {sum, max, min, and, or, xor}
                           #   sum 直达 → vector.reduction<add> → vfredusum
                           #   其余待扩 → vredmax/min、vredand/or/xor（整）/vfred…（浮）
```

### 6.6 Profiling

```
proton_mark(name, is_start)    # 性能标记,is_start 选 start/end    · 直达 → proton.record
```

### 6.7 构造

```
vzero(dtype) -> vec            # v[i] = 0（加法累加器初值）        · 直达 → vector.broadcast(0)
vsplat(s, dtype) -> vec        # v[i] = s（广播标量;max/min 累加器初值给 ±∞ 等）· 直达 → vector.broadcast(s)
vextract(v, i) -> 标量         # 取第 i 元素（静态）               · 直达 → vector.extract
vinsert(v, i, s) -> vec        # 置第 i 元素为 s（静态）           · 直达 → vector.insert
```

### 6.8 掩码（向量条件）

无标量 `if`;逐元素条件靠 mask。**比较**用运算符(6.4)产 mask,**按 mask 选**用
`select`(6.4);本节是 mask 寄存器自身的运算。`vector.mask` 当前在向量层 lower 失败,故
`select` 以比较 + `arith.select` 回退,接入原生 masked 指令后语言不变。

```
vmand/vmor/vmxor/vmnot(m1, m2) # mask 寄存器逻辑                    · 待扩 → vmand/vmor/vmxor/vmnand
vcpop(m) -> 标量               # mask 中 1 的个数                   · 待扩 → vcpop
vfirst(m) -> 标量              # 首个 1 的下标                      · 待扩 → vfirst
vmsbf/vmsif/vmsof(m) -> mask   # 首个 1 之前/及/仅的集合            · 待扩 → vmsbf/vmsif/vmsof
vid() -> vec / viota(m) -> vec # v[i]=i / mask 前缀和               · 待扩 → vid/viota
```

### 6.9 Permutation（lane 重排）

跨 lane 数据移动,性能较差,手写 kernel 常可用布局 / 访问顺序 / `vreduce` 规避,故很少用。
仅列三种代表能力(向量↔标量单元素移动用 `vextract`/`vinsert`,见 6.7):

```
vslide(v, off) -> vec     # 按 off 平移 lane（off 正/负 = 下/上移）  · 待扩 → vslideup/vslidedown
vrgather(v, idx) -> vec   # 任意索引重排 out[i] = v[idx[i]]          · 待扩 → vrgather
vcompress(v, m) -> vec    # 按 mask 把选中元素压到前部              · 待扩 → vcompress
```

## 7. 约束与边界

### 7.1 尺寸约束

- 可伸缩化要求定长向量总元素数满足 `N_total mod vscale == 0`（K3 vscale = 16）。
- 当前实现为定长 VL（无运行期尾部收窄），归约 / 访存维须为 VL 的整数倍。
- 矩阵单元 `vmadot` 的 m/n/k 须等于目标 cubic 规格；`vpack` / 硬件 pack 须 IME ≥ 2、
  tile 为 2 的幂、末维 unit-stride。

### 7.2 设计边界

下列内容不在语言能力面内，属机器边界，非未实现：

- 任意 Python（库调用、对象、闭包）；标量层面的复杂控制流。
- 自动 tiling / 向量化 / 布局 / 微内核选择——由通用自动路径承担。
- raw kernel 间互相调用；值语义返回（输出仅经 `InOut` 内存写回）。

### 7.3 使用约束

- `call()` 须在 `@triton.jit` 函数体内调用。
- 导入路径须为 `from triton.language.extra.spine_raw import ...`；经独立 sys.path 加载
  会得到不同模块实例，导致别名识别失效。

---

## 附录 A. 已实现原语索引（落地状态：直达 / 回退）

| 原语                   | 类别      | 状态                          |
| ---------------------- | --------- | ----------------------------- |
| `vconfig`            | 配置      | 直达（定长）                  |
| `vload` / `vstore` | 访存      | 直达                          |
| `vzero`              | 构造      | 直达                          |
| `vmacc`              | 浮点乘加  | 直达                          |
| `vreduce_sum`        | 归约      | 直达                          |
| `vmadot`             | 矩阵单元  | 回退（数值待对齐）            |
| `vpack`              | vpack     | 直达（→ smt.vpack.vv）       |
| `pack`               | 软件 pack | 直达（软件搬运，非硬件 pack） |
| `alloc`              | TCM       | 直达                          |
| `range`              | 控制流    | 直达                          |
| `proton_mark`        | profiling | 直达                          |

## 附录 B. 实现优先级（不影响语言完整性）

第 6 章的指令空间是完整的语言语义面；本附录仅排**实现次序**，依
`v_instruction_counts.csv` 实测频次。列在后面不代表不属于语言，只代表暂缓实现。

**第一梯队（高频，优先实现）**

1. 原生 masked / `vmerge`（388；谓词自 `arith.select` 升级为 RVV masked op，6.4 / 6.8）。
2. 类型转换 `vfcvt` 族 + 位扩展 `vzext`（158 + 236；6.4）。
3. 移位 / 位运算 `vsrl/vsll/vxor/vand`（~390；6.4）。
4. 浮点 `vfmin/vfmax`（168）与整数 `vmul/vadd`（328；6.4）。
5. 真 strided load / 转置读 `vload_strided → vlse`（矩阵算子必需，6.2）。

**第二梯队（中低频）**

6. `vfabs/vfsgnj`（122）、`vfdiv`（40）、`vfsqrt`（8）、整数 `vdiv/vrem`（16）。
7. lane 滑动 `vslidedown/up`（编译器内部高频，手写多可规避；作者需要时再接，6.9）。
8. 运行期显式向量长度 `vconfig(avl=...)`（真 strip-mine，6.1）。
9. `pack` 硬件加速：内层转置步改用 `vpack`（改善大 M 性能，6.3）。

**第三梯队（当前零频次或后端未 lower）**

10. indexed gather/scatter、segment、fault-only-first、mask 掩码访存（6.2）。
11. 定点算术全类（6.4）；归约 max/min/位归约（6.5）。
12. `vrgather`/`vcompress`、mask 逻辑/`vcpop`/`vfirst`/`viota`（6.9 / 6.8）。

各项落地后，使用相应语言构造的用户代码无须修改（4.1）。
