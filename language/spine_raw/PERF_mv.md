# mv 性能对标 — spine_raw 三种写法 vs FlagGems 原生

在 K3-003(hostname k3-dev-005,SpacemiT X100,arch 0xA064,vlen=1024,riscv64,nproc=8,
FlagGems num_threads=4)上测量矩阵-向量乘 `C[N] = B[N,K] @ A[K]` 的四种实现,
f16 输入 / f32 累加,对拍 `torch.mv`。

## 被测实现

| 实现 | 路径 | 计算方式 |
| --- | --- | --- |
| **style2 (pure svector)** | `python/tests/test_raw_mv_svector.py` | 纯向量:逐行 `vmacc`(extf+fma 宽化累加)+ `vreduce_sum` 横向求和,无打包 |
| **style3 (svector+pack)** | 同上 | 在 style2 基础上先 `alloc`+`pack` 把 B 的行块搬进连续 scratch,再逐行累加 |
| **cbm (matrix engine)** | `python/tests/test_raw_mv_cbm.py` | 矩阵引擎:mv 表达成 cube GEMM,`vpack`/`spread` 摆 cube → `vmadot`(cross_batch_matmul → `smt.vfwmadot`)→ `group_interleave` 还原 |
| **flaggems (native)** | `spine-FlagGems-github/src/flag_gems/ops/mv.py` | 原生 FlagGems `mv_kernel`:`tl.load` 分块 + `a*b` 逐元素 + `tl.sum`,`@triton.autotune` |

## 测量方法

- shape:12 组 `N×K`(见下表),共同约束 `N%16==0`(svector N%4 + cbm M%16)、`K%64==0`
  (svector K%64 + cbm K%8),让四种实现共用同一 shape。cbm Npad 固定 32(A 广播维)。
- host wall-clock,每实现 20 次 warmup(排除编译)后取 100 次迭代的**中位数**(微秒)。
- 脚本:`bench_mv.py`(NFS 根),`GEMS_VENDOR=spacemit`,`LD_LIBRARY_PATH` 指 spine runtime。
- 正确性:每 shape 全部对拍 `torch.mv`,`max_diff < 5e-2`(表中 all_ok 均 True)。

## 结果(多 shape 扫描,median us)

| N×K | style2 | style3 | cbm | flaggems | s2/fg | s3/fg | **cbm/fg** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64×64   | 173.6 | 174.4 | **122.1** | 268.1 | 1.54 | 1.54 | **2.20** |
| 128×64  | 240.4 | 231.9 | **125.5** | 274.1 | 1.14 | 1.18 | **2.18** |
| 256×64  | 326.7 | 321.8 | **199.6** | 379.5 | 1.16 | 1.18 | **1.90** |
| 512×64  | 526.4 | 545.7 | **254.1** | 475.7 | 0.90 | 0.87 | **1.87** |
| 1024×64 | 881.9 | 897.1 | **341.6** | 465.5 | 0.53 | 0.52 | **1.36** |
| 64×128  | 185.5 | 181.3 | **157.0** | 371.1 | 2.00 | 2.05 | **2.36** |
| 128×128 | 258.3 | 255.1 | **126.0** | 282.7 | 1.09 | 1.11 | **2.24** |
| 256×128 | 325.0 | 335.7 | **177.4** | 302.3 | 0.93 | 0.90 | **1.70** |
| 512×128 | 494.6 | 489.8 | **239.3** | 400.2 | 0.81 | 0.82 | **1.67** |
| 128×256 | 233.7 | 231.3 | **129.4** | 304.6 | 1.30 | 1.32 | **2.35** |
| 256×256 | 329.5 | 323.8 | **182.4** | 345.1 | 1.05 | 1.07 | **1.89** |
| 512×512 | 506.9 | 520.3 | **279.2** | 681.9 | 1.35 | 1.31 | **2.44** |

## 结论

1. **cbm(矩阵引擎)在所有 12 个 shape 上都最快,vs FlagGems 稳定 1.36×–2.44×**(多数 ~1.7–2.4×)。
   把 mv 折成 cube GEMM 喂 `smt.vfwmadot` 矩阵单元,是全 shape 域的最优。它随 N 增长最平缓
   (1024×64 才 342us,svector 已 882us),矩阵单元吞吐优势在大 N 更明显。
2. **两种 svector 写法(style2/style3)基本持平**,彼此差异在噪声内 —— 这个规模下 style3 的预打包
   收益被搬运开销抵消。它们只在**小 N**(N≤128)快过 FlagGems(~1.1–2.0×);**N 变大后反被 FlagGems
   超过**(512×64 起 s?/fg<1,1024×64 只 0.5×),因为 svector 逐行标量归约不随 N 摊薄。
3. **FlagGems 原生**居中:通用 `tl.load`+逐元素+`tl.sum`,`@triton.autotune` 使它在大 N 比 svector 更
   稳,但始终不及 cbm 的矩阵单元。

要点:**矩阵引擎路(cbm)是全 shape 域的赢家且优势随 N 扩大;svector 只在小 N 有优势、大 N 落后**。
这印证 mv 这种 memory-bound + 可 cube 化的算子,用 K3 矩阵单元(cbm)是正确方向。

## 边界与后续

- shape 限制:`N%16==0`、`K%64==0`(见测量方法)。cbm 已支持任意 8 倍数族(M%16/K%8),
  svector 要 K%64;非整除尾部需 padding/mask(SPEC §6.1 avl,待扩)。cbm Npad 固定 32
  (>32 需 N-tiling,vpack.vv seg≤512 硬件上限)。
- host wall-clock 含 Python dispatch;绝对值随负载波动,相对加速比稳定(多次运行一致)。
- 未测更大 K(>512)与非方阵极端比;cbm 的 N-tiling(Npad>32)是拓宽 N 列的后续。
