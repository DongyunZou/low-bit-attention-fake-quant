# FA4 Kernel Implementation Roadmap

## 目标

在 5090 和 PRO 6000 机器上，从 FA4 CUTE per-head FP8 attention 出发，逐步实现当前 fake-quant 路线中的低比特 attention 技术，并保证每一步都能和 fake-quant oracle 对齐。

总体顺序：

1. FA4 baseline bring-up。
2. per-head quant 改成 FP32 scale 的 per-block quant。
3. K smooth。
4. Q smooth + Q k-means permute。
5. estimated rowmax，减少 online softmax 中的 rowmax update / rescale / exp 开销。
6. V smooth + V k-means。

每一步只改变一个主要因素。kernel 输出先对齐 fake-quant attention，再做 Wan layer / saved sensitive call / end-to-end 视频验证。

## M0: FA4 Baseline Bring-Up

### Fake-quant 对应逻辑

这一步不引入新 fake-quant 逻辑，只用 FA4 原始 CUTE per-head FP8 attention 作为真实 kernel baseline。

需要记录：

- 输入布局：项目内 fake-quant 使用 NHD `(B, S, H, D)`，FA4 kernel 若使用 HND 或 packed layout，需要固定转换边界。
- dtype：Q/K/V 通常是 BF16 输入，kernel 内走 FP8 MMA / BF16 输出。
- softmax：以 FA4 CUTE bf16/fp8 路径的 online softmax 行为作为 baseline。

### 实现要点

- 分别在 5090 和 PRO 6000 上确认 CUTE kernel 可以编译和运行。
- 固定一个最小 Wan self-attention shape smoke test，例如 `B=1, S=32760, H=40, D=128`，并处理 padding 到 tile 对齐。
- 加一层 thin wrapper，把 FA4 kernel 接到当前 bench 脚本的 Q/K/V dump 上。

### 验收

- kernel 输出和 FA4 reference 路径一致。
- 小 shape、Wan layer0 dump、saved sensitive call 都能跑通，无 NaN/Inf。
- 记录 baseline kernel time，作为后续每一步性能对比基准。

## M1: FP32 Scale Per-Block Quant

### Fake-quant 对应逻辑

对应当前 fake-quant 的 Q/K `fp8_block`：

```text
scale = max(amax(block), 1e-4) / 448
q_fp8 = cast_fp8(q / scale)
k_fp8 = cast_fp8(k / scale)
```

当前 end-to-end 主要配置：

- `fp8_block_size = 64`
- scale 使用 FP32 保存和计算。
- Q/K 沿 token block 统计 amax，不再使用 FA4 原始 per-head scale。

### 实现要点

- 在 FA4 load path 中加入 Q/K per-block scale load。
- scale 粒度应和 fake-quant 完全一致，优先支持 `(B, H, ceil(S / block_size))`。
- kernel hot loop 中按 Q block 和 K block 应用 scale：
  - 如果直接用 FP8 MMA，需要把 `s_q * s_k` 合并到 score accumulator。
  - 如果先 dequant 到寄存器/共享内存，需要确认 dequant 精度和 fake-quant oracle 一致。
- padding token 只参与对齐，不参与 amax、rowmax、softmax denominator 和输出。

### 验收

- 单独验证 quant/dequant 与 `fp8_block_quant` 对齐。
- attention 输出对齐 fake-quant `qk_quant="fp8_block"`。
- 记录 per-head scale baseline 到 per-block FP32 scale 的精度收益和 kernel 时间变化。

## M2: K Smooth

### Fake-quant 对应逻辑

对应 `smoothing="k_only"`：

```text
k_mean = mean(K, dim=S)
K_smooth = K - k_mean
```

这个变换对 softmax 不改变语义，因为对所有 key 减同一个向量，只会给同一 query row 的所有 logits 加同一个常数，softmax 会抵消。

### 实现要点

- 在 K quant 前预处理 `K_smooth`，再对 `K_smooth` 做 per-block FP8 quant。
- `k_mean` 不需要进入主 attention kernel，因此理论上主 kernel 无速度损失。
- 预处理可以先用独立 CUDA/Triton kernel，后续再融合到 K quant preprocess。

### 验收

- 未量化时验证 softmax invariance。
- 量化后比较 `off` vs `k_only` 的 Q/K block amax、attention MSE、cosine。
- kernel time 应接近 M1，主要额外成本只在 preprocess。

## M3: Q Smooth + Q K-Means Permute

### Fake-quant 对应逻辑

对应当前主要配置：

```text
q_kmeans_k = 32
q_smooth_block_size = 256
```

流程：

1. 对每个 `(B, H)` 的 Q token 做 k-means label。
2. 按 label stable sort Q token，记录 inverse order。
3. 每 256 个 reordered Q token 计算一个 `q_m`。
4. 量化 `Q_centered = Q - q_m`。
5. attention score 中补回 correction：

```text
score = dot(Q_centered_fp8, K_smooth_fp8) + dot(q_m, K_smooth)
```

correction 必须走 FP32/BF16 高精度路径，不能再次用 FP8 K 近似。

### 实现要点

- Q k-means 只 permute Q 轴；K/V 轴不跟随 Q k-means permute。
- 输出 O 需要用 inverse order 还原到原 token 顺序。
- `q_smooth_block_size=256`，而 attention `block_m` 可以是 64；因此 4 个 M tile 共享同一个 `q_m`。
- kernel 中每个 K tile 需要为当前 Q smooth group 计算或加载 `q_m @ K_smooth_tile.T`。
- 对 cluster boundary block 要额外关注，因为一个 256-token Q smooth block 可能跨 k-means label 边界，这是 rowmax estimate outlier 的主要来源之一。

### 验收

- order/inverse roundtrip。
- Q block amax 分布应明显改善。
- attention 输出对齐 fake-quant `smoothing="full", q_kmeans_k=32, q_smooth_block_size=256`。
- 单独统计 cluster boundary block 和 single-cluster block 的 rowmax estimate error。

## M4: Estimated Rowmax

### Fake-quant 对应逻辑

利用 Q smooth 的 `q_m` 预估 softmax rowmax：

```text
rowmax_est ~= max_K(q_m @ K_smooth.T)
```

kernel 不再对每个 K tile 都无条件推进 online rowmax，而是从 estimate 初始化 softmax coordinate，只在 estimate 明显不合适时 fallback 到 online-style update。

当前 fake-quant / Triton debug candidate：

```text
up_threshold = 16
down_threshold = 32
```

逻辑：

- under-estimate：如果 `tile_max - m_i > up_threshold`，向上更新 `m_i`。
- over-estimate：如果 `m_i - max_seen_i > down_threshold`，向下更新 `m_i`。
- 触发更新时像 online softmax 一样 rescale 旧的 `acc` 和 `l_i`，保证数值正确性。

### 实现要点

- 真 kernel 中优先按 Q smooth group 存 `rowmax_est`，而不是按 row 存，减少内存。
- `rowmax_est` 可以在 preprocess 中用 `q_m @ K_smooth.T` reduction 得到。
- 主 kernel 仍保留 online-softmax state：`m_i, l_i, acc, max_seen_i`。
- 加 debug counter：
  - rowmax update 次数。
  - real rescale 次数。
  - rows with update。
  - upward / downward update 分开计数。
- counter kernel 只用于 profiling，不用于正式性能测试。

### 已有参考数据

在 saved Wan bad self-attention call `call_066.pt` 上，配置为 `q_smooth_block=256`、`q_kmeans_k=32`、`fp8_block_size=64`、`block_m=block_n=64`：

| Method | Rowmax updates | Rescales | Rows with update |
|---|---:|---:|---:|
| FA4-style online, threshold 8 | 2,717,371 | 1,406,651 | 1,310,720 |
| Estimated rowmax, up 16 / down 32 | 677,294 | 595,885 | 153,999 |

对应减少：

- rowmax updates: 75.1%
- real rescales: 57.6%
- rows with update: 88.3%

### 验收

- 对齐 fake-quant online rowmax dynamic-P 输出，而不是只对齐 BF16 SDPA。
- Wan layer0 和 saved sensitive calls 都无 NaN/Inf。
- 统计 update/rescale 减少量，并确认正式 kernel 时间确实下降。

## M5: V Smooth + V K-Means

### Fake-quant 对应逻辑

对应当前 V 路线：

```text
v_smooth = per_block
v_kmeans_k = 64
v_smooth_block_size = 64
```

流程：

1. 对 V token 做 k-means label。
2. 按 V label permute K/V 轴，K 必须和 V 使用同一个 order。
3. 对 reordered V 按 block 做 smooth / quant。
4. PV accumulation 后补回 V smooth correction。

直觉上，V smooth 降低 V quant error；V k-means 让同一个 V smooth block 内 token 更相似，从而降低 block 内动态范围。

### 实现要点

- V k-means permute 是 K/V 轴 permute，不影响 Q 轴 order。
- Q k-means 和 V k-means 可以同时存在：
  - Q k-means 影响 Q order 和 output inverse。
  - V k-means 影响 K/V order 和 softmax K loop traversal。
- K smooth、Q correction、rowmax estimate 都要使用 V-permuted 后的 K 顺序。
- V smooth correction 需要依赖每个 V block 的 softmax probability sum，因此要在 PV loop 中同步维护 block-level P sum。

### 验收

- K/V permute 和 inverse-free output 语义正确，因为输出只在 Q 轴还原。
- V quant MSE 和 full attention MSE 相比 M4 下降。
- end-to-end 视频指标不回退，kernel 时间可接受。

## 统一测试矩阵

每个 milestone 都跑同一组测试，避免只在 layer0 上过拟合：

| Level | 内容 |
|---|---|
| Unit | quant scale shape、padding mask、permute roundtrip、correction term |
| Attention | Wan layer0 六个 workload |
| Sensitive | saved bad/sensitive calls，例如 `call_066.pt`、`call_236.pt` |
| End-to-end | 短配置视频 smoke，再跑正式 5-noise 指标 |
| Perf | kernel time、preprocess time、rowmax update/rescale counter |

精度对比顺序：

1. kernel vs fake-quant oracle。
2. fake-quant oracle vs BF16 SDPA。
3. end-to-end video vs SDPA baseline。

## 主要风险

- FA4 内部 layout 和 fake-quant NHD layout 不一致，容易引入 hidden transpose cost。
- padding 同时影响 quant block、Q smooth block、softmax mask，必须统一处理。
- Q smooth correction 和 rowmax estimate 必须使用同一个 K smooth / K order 语义。
- k-means preprocess 可能抵消 kernel 加速，需要单独拆分 preprocess time 和 attention kernel time。
- estimated rowmax 的 counter 收益不等于最终 wall-clock 收益，必须在正式 kernel 上测。

## 推荐开发顺序

1. 先只支持 non-causal Wan self-attention，固定 `D=128`。
2. 每完成一个 milestone，冻结一个 bench config 和 JSON 结果。
3. 5090 用于快速 compile/smoke，PRO 6000 用于长 shape 和 end-to-end。
4. 所有新优化默认先实现 debug counter，再关闭 counter 做真实性能。
5. 任何 end-to-end 崩溃先回到 saved QKV call 单独复现，再进入视频指标。
