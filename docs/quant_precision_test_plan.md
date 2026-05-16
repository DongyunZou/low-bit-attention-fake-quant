# Quant 精度测试计划

## 目标

这个项目用于搭建低比特 fake-quant 精度测试框架，重点服务 Wan / video
diffusion 的 end-to-end 视频生成质量验证。核心要求是：Q/K/V 预处理、
量化、P requant 尽量用 Triton 实现，避免 PyTorch eager reference 和真实
kernel 之间存在过大的语义差距。

第一阶段不直接追求最快 kernel，而是建立可复现实验矩阵、可插拔 helper
和稳定的精度指标。

## 需要覆盖的量化组合

固定输入布局为 NHD：`(B, S, H, D)`，`D in {64, 128}`。

### Q/K 量化

1. `fp8_block`
   - e4m3fn FP8。
   - 一个 FP32 scale 覆盖 `(B, S/128, H)`。
   - scale 公式：`scale = max(amax, 1e-4) / 448`。
   - helper: `low_bit_fake_quant.quant_triton.fp8_block_quant`。

2. `mxfp8`
   - MXFP8 风格 block scale。
   - Q/K 沿 D 轴每 32 个元素一个 scale。
   - scale 是 power-of-two：`2 ** ceil(log2(max(amax, floor) / 448))`。
   - helper: `low_bit_fake_quant.quant_triton.mxfp8_qk_quant`。

### V 量化

默认使用 `fp8_channel`：

- e4m3fn FP8。
- 一个 FP32 scale 覆盖 `(B, H, D)`，沿 S reduce。
- helper: `fp8_per_channel_quant`。

保留 `mxfp8_s` 作为实验分支：

- 沿 S 轴每 32 个 token 一个 scale。
- helper: `mxfp8_v_quant`。

### Q/K smoothing

1. `off`
   - 不做 smoothing。

2. `k_only`
   - `K_smooth = K - mean(K, dim=S)`。
   - 这是 softmax-invariant 变换，主要降低 K 的 block amax。
   - helper: `preprocess.smooth_k`，当前为 Triton 实现。

3. `full`
   - `k_only` 加 Q centering。
   - `Q_centered = Q - qm[group]`。
   - `qm` 默认按 `q_smooth_block_size=256` 生成，也要覆盖 128/512。
   - correction: `correction = qm @ K_smooth^T`，不预乘 `sm_scale`。
   - helpers: `preprocess.group_mean_q` + `pipeline.prepare_qkv`。

### Q k-means reorder

维度：

- `q_kmeans = off`
- `q_kmeans = on`，默认 `k=32, iters=10, seed=0`

当前 helper:

- `kmeans.q_kmeans_reorder`
- chunked torch 实现，作为 correctness helper。

后续 Triton 化拆成两个 kernel：

1. assignment kernel：对每个 `(B,H)`，按 token chunk 计算
   `||q - centroid||^2` 并输出 label。
2. centroid reduce kernel：按 label 聚合 sum/count，更新 centroid。

排序和 inverse order 可以继续先用 `torch.argsort`，因为它不改变 fake-quant
数值，只改变 token 顺序；如果 end-to-end 需要完全图内执行，再替换为
Triton radix/counting-sort 分支。

## P requant 测试

真实 FP8 attention kernel 中，P 不是 FP32 softmax 后直接参与 PV，而是：

```text
z = (score - row_max) * sm_scale * log2(e) + p_max_offset
P_scaled = exp2(z)
P_fp8 = P_scaled.to(float8_e4m3fn)
PV = P_fp8 @ V_fp8
O = PV * (1 / row_sum) * v_descale
```

`row_sum` 应累加 cast 前的 `P_scaled`，不是 FP8 cast 后的 P。最终
`1 / row_sum` 会抵消 `2 ** p_max_offset` 的全局 scale。

测试分两层：

1. 小 shape materialized P probe
   - helper: `p_requant.p_requant_rows`。
   - 用于确认 FP8 cast、`p_max_offset`、LSE 公式和饱和边界。

2. 大 shape streaming Triton attention
   - 不 materialize `(B,H,S,S)`。
   - 每次处理一个 Q block 和一个 K block。
   - online rowmax/rowsum 与 P_fp8 cast 在同一个 Triton kernel 内完成。
   - 输出用于 Wan/video end-to-end fake-quant oracle。

第二层是后续必须实现的核心 kernel。PyTorch reference 可以保留为 debug
上界，但不能作为最终 end-to-end 视频精度结论的唯一依据。

## 实验矩阵

基础矩阵：

| 维度 | 取值 |
|---|---|
| QK quant | `fp8_block`, `mxfp8` |
| smoothing | `off`, `k_only`, `full` |
| Q k-means | `off`, `on(k=32)` |
| V quant | `fp8_channel` first, `mxfp8_s` optional |
| P requant | `off` reference, `on(p_max_offset=8)` |

第一批必须跑：

1. `fp8_block × off × no-kmeans × fp8_channel × P-on`
2. `fp8_block × k_only × no-kmeans × fp8_channel × P-on`
3. `fp8_block × full × kmeans-on × fp8_channel × P-on`
4. `mxfp8 × off × no-kmeans × fp8_channel × P-on`
5. `mxfp8 × k_only × no-kmeans × fp8_channel × P-on`
6. `mxfp8 × full × kmeans-on × fp8_channel × P-on`

扩展 sweep：

- `q_smooth_block_size in {128, 256, 512}`
- `q_kmeans_k in {16, 32, 64}`
- `p_max_offset in {0, 4, 6, 8}`
- `V quant in {fp8_channel, mxfp8_s}`

## 精度指标

### Attention / activation 层

- cosine vs BF16/FP32 reference。
- relative MSE。
- max absolute error。
- LSE max abs / relative MSE。
- P saturation rate：`abs(P_scaled_fp8) == 448` 的比例。
- P underflow / zero rate。

### End-to-end 视频层

固定 seed、prompt、scheduler、resolution、num frames、guidance 参数。

每个量化配置输出：

- latent cosine / relative MSE vs BF16 baseline。
- decoded RGB PSNR。
- decoded RGB SSIM。
- LPIPS 或 DISTS。
- CLIP/Image embedding cosine，用于感知级 drift。
- 视频逐帧指标的 median / p95 / worst frame。

至少保留：

- BF16 baseline latent / video。
- 每个 fake-quant 配置的 latent / video。
- JSON metrics。
- 小尺寸 debug artifact，包含 P saturation/zero 统计。

## 项目结构

```text
low-bit-fake-quant/
├── pyproject.toml
├── docs/
│   └── quant_precision_test_plan.md
├── src/low_bit_fake_quant/
│   ├── config.py
│   ├── quant_triton.py
│   ├── preprocess.py
│   ├── kmeans.py
│   ├── p_requant.py
│   └── pipeline.py
└── tests/
    └── test_imports.py
```

## 实现顺序

1. 单元验证 Triton quant helper
   - FP8 block scale vs PyTorch reference。
   - FP8 per-channel V vs PyTorch reference。
   - MXFP8 Q/K and V scale shape / no saturation invariant。

2. 验证 smoothing helper
   - `smooth_k` softmax invariance。
   - `group_mean_q` vs PyTorch group mean。
   - `full` correction 的未缩放语义：`sm_scale` 只能在 attention 中应用一次。

3. 验证 Q k-means reorder
   - order/inverse roundtrip。
   - reorder 后 inverse output 与原 token 顺序对齐。
   - kmeans 开关对 Q block amax 分布的影响。

4. 实现 streaming Triton P-requant attention oracle
   - 先支持 dense non-causal MHA。
   - 输入使用 dequantized Q/K/V 或 quantized+scale 两套路径。
   - 输出 BF16/FP32 selectable。
   - 统计 P saturation/zero。

5. 接入 Wan/video end-to-end
   - 在 attention 或 transformer block 边界插入 fake-quant pipeline。
   - 固定配置生成短视频 smoke，再扩展到正式评测 prompts。

## uv 命令

```bash
uv sync --extra dev --extra bench
uv run pytest
```

如需匹配 CUDA 13.2 PyTorch wheel，`pyproject.toml` 已配置
`pytorch-test-cu132` index。
