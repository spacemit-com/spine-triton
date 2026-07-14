# mv 性能对标 — spine_raw 三种写法 vs FlagGems 原生

在 K3-003(SpacemiT X100,arch 0xA064,vlen=1024,riscv64,16 核)上测量矩阵-向量乘
`C[N] = B[N,K] @ A[K]` 的四种实现,f16 输入 / f32 累加,对拍 `torch.mv`。

## 被测实现

| 实现 | 路径 | 计算方式 |
| --- | --- | --- |
| **style2 (pure svector)** | `python/tests/test_raw_mv_svector.py` | 纯向量:逐行 `vmacc`(extf+fma 宽化累加)+ `vreduce_sum` 横向求和,无打包 |
| **style3 (svector+pack)** | 同上 | 在 style2 基础上先 `alloc`+`pack` 把 B 的行块搬进连续 scratch,再逐行累加 |
| **cbm (matrix engine)** | `python/tests/test_raw_mv_cbm.py` | 矩阵引擎:mv 表达成 cube GEMM,`vpack`/`spread` 摆 cube → `vmadot`(cross_batch_matmul → `smt.vfwmadot`)→ `group_interleave` 还原 |
| **flaggems (native)** | `spine-FlagGems-github/src/flag_gems/ops/mv.py` | 原生 FlagGems `mv_kernel`:`tl.load` 分块 + `a*b` 逐元素 + `tl.sum`,`@triton.autotune` |

## 测量方法

- shape:`N = K = 64`,f16(方阵,让三种 raw 写法与 flaggems 共用同一 shape;cbm 当前固定 M=K=64、Npad=32)。
- host wall-clock,每实现 20 次 warmup(排除编译)后取 100 次迭代的**中位数**(微秒)。
- 脚本:`bench_mv.py`(NFS 根),`GEMS_VENDOR=spacemit`。
- 正确性:全部对拍 `torch.mv`,`max_diff < 5e-2` 通过(svector 走 f32 累加 max_diff=0;cbm/flaggems f16 矩阵引擎 max_diff≈7e-3)。

## 结果(3 次运行,中位数 us)

| 实现 | run1 | run2 | run3 | 典型 median_us | 加速比 vs FlagGems |
| --- | --- | --- | --- | --- | --- |
| flaggems (native) | 256.1 | 261.5 | 267.5 | **~262** | 1.00× (基线) |
| style2 (pure svector) | 161.7 | 166.1 | 179.9 | ~169 | **~1.55×** |
| style3 (svector+pack) | 158.6 | 172.7 | 181.2 | ~171 | **~1.53×** |
| **cbm (matrix engine)** | 124.8 | 125.2 | 131.3 | **~127** | **~2.05×** |

## 结论

1. **cbm(矩阵引擎)最快,约 2.0× FlagGems**。把 mv 折成 cube GEMM 直接喂 `smt.vfwmadot` 矩阵单元,单位时间算得多,是这个 shape 上的最优。
2. **两种 svector 写法各约 1.5× FlagGems**,彼此接近。这个规模下 style3 的预打包收益被搬运开销抵消,与 style2 基本持平(打包的价值要在更大的 K、B 行被反复复用时才显现)。
3. **FlagGems 原生最慢**:`tl.load`+逐元素 `a*b`+`tl.sum` 的通用向量路径,没有用到 K3 的矩阵单元,也没有 raw eDSL 那种显式布局/寄存器驻留控制。

三种 raw 写法均快于 FlagGems 原生,印证了 spine_raw 直接在 vector/matrix 层手写执行方案的收益:向量路(svector)约 1.5×,矩阵引擎路(cbm)约 2.0×。

## 边界与后续

- 本表只测了 `N=K=64` 单一方阵 shape。cbm 目前固定 M=K=64/Npad=32,要跨 shape 对标需先把 cbm 的 M/K/Npad 参数化(grid 已并发,行方向可扩)。
- host wall-clock 含 Python dispatch 开销;绝对值随负载波动(见三次运行),但相对加速比稳定。
- 更大 K / N 下 svector 打包(style3)与 cbm 的相对优势预计拉开,值得后续按 shape 扫描补测。
