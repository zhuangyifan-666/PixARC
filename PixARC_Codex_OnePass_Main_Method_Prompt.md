# Codex 任务：实现单次采样、双参数的 Pixel-Remainder Taylor，并只运行新的 1K 主方法实验

你是一名资深扩散模型推理加速研究工程师。请直接在当前 `PixARC` 仓库中实现一个新的像素空间自适应 TaylorSeer 方法，并使用仓库现有 1K 协议完成新方法实验与已有基线对比。

不要重新运行已有 Full、TaylorSeer、SeaCache、SpeCa 或 DiCache 基线。不要建立额外标定集，不要训练误差预测器，不要运行影子完整前向，不要在线调用 DINO、LPIPS 或其他视觉编码器。新方法对每张图只进行一次正常扩散采样轨迹；轨迹内部的 Full/Taylor NFE 由新控制器决定。

不要只给计划或伪代码。请完成代码、测试、真实模型冒烟、1K 生成、评测、追踪和与已有 CSV 的汇总。若 GPU、权重或评测器确实不可用，仍须完成所有代码和测试，并给出精确阻塞证据与可直接复制执行的命令；禁止伪造实验结果。

---

## 0. 方法概述

方法名：

**Pixel-Remainder Taylor（像素余项控制泰勒预测）**

核心思想：

1. TaylorSeer 在轨迹中本来就会周期性执行 Full NFE；
2. 每个 Full NFE 都会得到引导后的干净图像预测 `x0_full`；
3. 只使用这些已经存在的 Full 输出，在像素空间维护 0～3 阶有限差分；
4. 用下一遗漏阶的像素差分估计一阶或二阶 Taylor 预测在未来 `h` 个 NFE 上的余项；
5. 在给定像素误差预算 `tau` 下，自动选择下一段使用的一阶/二阶 Taylor 以及连续 Taylor NFE 数量；
6. 不增加任何模型前向。每个 NFE 仍然只执行一次既定的 Full 或 Taylor 路径。

该方法只有两个对外参数：

```text
tau              # 允许的像素余项预算
max_taylor_span  # 两次 Full 之间最多允许的连续 Taylor NFE 数
```

第一轮实验固定 `max_taylor_span=3`，只扫描 `tau`。只有在结果显示绝大多数锚点都命中上限且质量仍然很好时，才额外运行 `max_taylor_span=4`。

---

## 1. 首先检查仓库和协议

必须先阅读：

- `BASELINE_BATCH_PROTOCOL.md`
- `results/README.md`
- `results/taylorseer_1k_summary.csv`
- `results/seacache_1k_summary.csv`
- `results/speca_1k_summary.csv`
- `results/dicache_1k_summary.csv`
- `JiT/baselines/taylorseer-style/`
- `PixelGen/baselines/taylorseer-style/`
- 两个 TaylorSeer 目录中的 README、RUNBOOK、配置、调度器、运行时、有限差分、模型适配器、采样器、追踪、生成和评测代码

开始前执行并记录：

```bash
git status --short
git log -1 --oneline
```

不得覆盖用户已有修改，不得修改：

- `third-party/`
- 任何已有 `baselines/` 代码、配置和结果
- 已有 `results/*_1k_summary.csv`

新代码放到独立目录：

```text
JiT/methods/pixel-remainder-taylor/
PixelGen/methods/pixel-remainder-taylor/
```

Python 包名统一为：

```text
pixel_remainder_taylor
```

可以复制对应 TaylorSeer 目录作为脚手架，但必须重命名包、模式、配置模式、元数据、输出目录和结果名称。

---

## 2. 必须保持的实验公平性

沿用已有 1K 实验的全部条件：

- 使用已有 1K manifest、sample id、类别、随机种子和显式噪声；
- JiT：保持现有真实批量、conditional/unconditional 两次前向、CFG 区间、50 步 exact Heun、BF16 和编译模式；
- PixelGen：保持现有真实批量、`[unconditional, conditional]` 合并 2B 前向、guidance 函数、50 步 exact Heun、BF16 和编译模式；
- 输出命名、分片、恢复运行、计时边界和评测器与已有协议一致；
- 新方法的速度必须包含控制器、像素分解和追踪开销；
- 不得排除任何运行时开销；
- 不得重新生成已有基线。

最重要的调用数约束：

```text
JiT：每个 NFE 仍为 2 次网络前向（conditional + unconditional）
PixelGen：每个 NFE 仍为 1 次合并 2B 网络前向
```

新方法不允许出现：

- shadow forward；
- 同一 NFE 的候选阶数重复前向；
- 为计算判据额外执行 Full；
- 额外完整参考轨迹；
- 额外模型副本。

99 个 NFE 的总模型调用数必须与相同模型族现有 TaylorSeer 一致。

---

## 3. 基础 Taylor 预测保持不变

复用现有 TaylorSeer 的以下语义：

- 缓存对象仍为每层 gate 前的 attention 输出和 MLP 输出；
- 只有 Full 分支更新有限差分历史；
- Taylor 分支只读历史；
- 使用现有单调 NFE 坐标 `q=total_nfe-1-nfe_index`；
- CFG 流共享同一个 Full/Taylor 动作，但保留现有流状态语义；
- 缓存数据类型、编译公平性和 exact Heun 生命周期保持不变。

特征历史固定保存到二阶：

```text
stored_feature_order = 2
```

运行时增加：

```text
active_forecast_order in {1, 2}
```

`ModuleTaylorState.forecast` 或等价函数必须支持只读的 `order_override`：

```python
result = sum(
    factor[k] * offset**k / factorial(k)
    for k in range(min(order_override, available_order) + 1)
)
```

不得删除高阶因子，不得因当前选择一阶而破坏二阶历史。

---

## 4. 像素锚点历史

### 4.1 锚点对象

在每个 **Full NFE** 完成 CFG 后，得到真实批次上的引导后干净图像预测：

```text
x0_anchor: [B, 3, H, W]
```

必须使用引导后的真实样本预测，而不是：

- conditional/unconditional 分别计算；
- CFG 扩展后的 2B 样本重复统计；
- 未引导模型输出；
- 最终保存的 uint8 图片。

JiT 当前网络直接预测 `x0`，但应从采样器实际使用的 guided velocity 统一恢复：

```python
x0_anchor = current_state + (1.0 - t) * guided_velocity
```

PixelGen 同样从当前实际状态和 guidance 后 velocity 恢复：

```python
x0_anchor = current_state + (1.0 - t) * guided_velocity
```

使用传入该 NFE 的真实 `current_state`；predictor 和 corrector 必须使用各自真实输入状态。只将 `x0_anchor.detach().float()` 用于控制器，求解器继续使用原有 BF16 输出。

### 4.2 像素有限差分

为 guided `x0_anchor` 建立独立的 exact-only 历史：

```text
pixel_factors = [P0, P1, P2, P3]
```

使用与现有 TaylorSeer 相同的坐标和递推更新规则，但：

```text
pixel_max_order = 3
pixel_cache_dtype = float32
```

只有 Full NFE 更新 `pixel_factors`。Taylor NFE 绝对不能写入该历史。

- `P0`：当前完整像素预测；
- `P1`：一阶变化估计；
- `P2`：二阶变化估计，用于估计一阶预测的遗漏项；
- `P3`：三阶变化估计，用于估计二阶预测的遗漏项。

像素历史只保存一个 guided real-batch 流，不为 CFG 分支分别复制。

---

## 5. 固定的低频/高频分解

不使用 FFT、小波、DINO 或 LPIPS。固定使用一次平均池化构造低频：

```text
pool_kernel = 8
```

定义：

```python
def split_bands(x):
    low_small = avg_pool2d(x, kernel_size=8, stride=8)
    low = interpolate(
        low_small,
        size=x.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    high = x - low
    return low, high
```

为了减少开销，低频范数可以直接在 `low_small` 上计算，但公式和测试必须保持等价语义。

所有范数按每张真实图片独立计算：

```python
mean(abs(tensor), dim=(1, 2, 3))
```

固定：

```text
eps = 1e-6
```

---

## 6. 像素 Taylor 余项估计

当前 Full 锚点坐标为 `q_anchor`。对候选特征预测阶数：

```text
o in {1, 2}
```

其下一遗漏项为 `P[o+1]`。对于未来连续 `h` 个 Taylor NFE，定义：

```text
scale(o, h) = abs(h) ** (o + 1) / factorial(o + 1)
```

对每张图片计算低频余项：

```text
r_low_i(o, h) = scale(o, h)
                * mean(abs(low(P[o+1])_i))
                / (mean(abs(low(P0)_i)) + eps)
```

高频余项：

```text
r_high_i(o, h) = scale(o, h)
                 * mean(abs(high(P[o+1])_i))
                 / (mean(abs(high(P0)_i)) + eps)
```

定义采样进度：

```text
progress = clamp(nfe_index / (total_nfe - 1), 0, 1)
```

使用固定的高频时间门控：

```text
risk_i(o, h) = max(r_low_i(o, h), progress * r_high_i(o, h))
```

一个批次共享动作。主控制使用真实样本平均值：

```text
risk(o, h) = mean_i(risk_i(o, h))
```

同时记录批次最大值作为诊断，但不参与主决策：

```text
risk_max(o, h) = max_i(risk_i(o, h))
```

严禁对 conditional/unconditional 或合并 CFG 的重复项重复计数。

---

## 7. 用两个参数自动选择阶数和跨度

配置只有：

```text
tau
max_taylor_span
```

固定候选：

```text
order_candidates = [1, 2]
h_candidates = [1, ..., max_taylor_span]
```

对每个可用阶数 `o`，找最大安全跨度：

```python
safe_h[o] = max(
    [h for h in range(1, max_taylor_span + 1)
     if risk(o, h) <= tau],
    default=0,
)
```

阶数可用条件：

- `o=1`：像素 `P2` 已成熟，所有特征分支至少有一阶历史；
- `o=2`：像素 `P3` 已成熟，所有特征分支至少有二阶历史。

选择规则必须固定，不再增加阈值：

```python
best_order = argmax_o (safe_h[o], -o)
best_span = safe_h[best_order]
```

即：

1. 选择安全跨度最大的阶数；
2. 若跨度相同，选择更低阶数；
3. 若所有 `safe_h=0`，下一 NFE 继续 Full；
4. 若选择 `best_span=h`，当前 Full 之后连续执行 `h` 个 Taylor NFE，均使用 `best_order`，之后下一 NFE Full；
5. 每次新的 Full 锚点重新计算，不沿用旧风险。

这里的动态 `N` 定义为：

```text
当前 Full 锚点之后实际连续执行的 Taylor NFE 数
```

若需要与现有 TaylorSeer `interval` 对照：

```text
interval = N + 1
```

### 7.1 暖启动

固定：

```text
warmup_full_nfe = 3
```

- 前 3 个 NFE 强制 Full；
- 第 3 个 Full 后通常已有 `P2`，允许规划一阶 Taylor 段；
- `P3` 成熟后才允许选择二阶；
- 历史不足、上下文变化、非有限值或状态异常时，下一 NFE Full；
- 不额外强制最后一个 NFE Full，保持与现有 TaylorSeer 主协议一致，除非仓库协议明确要求。

---

## 8. 动态调度器状态

新增独立调度器，至少维护：

```text
nfe_index
full_count
taylor_count
warmup_remaining
active_forecast_order
remaining_taylor_nfe
planned_span
last_anchor_q
last_risk_table
```

动作逻辑：

```python
if nfe_index < warmup_full_nfe:
    action = FULL
elif remaining_taylor_nfe > 0:
    action = TAYLOR
    remaining_taylor_nfe -= 1
else:
    action = FULL
```

Full NFE 完成并更新像素历史后，调用：

```python
plan_next_segment(...)
```

设置：

```text
active_forecast_order = best_order
remaining_taylor_nfe = best_span
planned_span = best_span
```

注意：当前 Full 锚点已经完成；`best_span` 从下一个 NFE 开始计数。

如果 `best_span=0`，下一个 NFE仍为 Full。

所有 CFG 流在同一 NFE 必须共享动作和 `active_forecast_order`。

---

## 9. 配置文件

新配置模式使用：

```yaml
schema_version: pixarc-pixel-remainder-taylor-v1
method:
  mode: pixel_remainder_taylor
  tau: null
  max_taylor_span: 3
  stored_feature_order: 2
  pixel_max_order: 3
  warmup_full_nfe: 3
  pool_kernel: 8
  batch_reduction: mean
  cache_dtype: inherit
  trace_mode: full
```

`tai: null` 或 `tau: null` 必须被严格拒绝；不要静默使用默认值。

本轮只正式运行以下三组：

```text
prt_t0p01_h3: tau=0.01, max_taylor_span=3
prt_t0p02_h3: tau=0.02, max_taylor_span=3
prt_t0p04_h3: tau=0.04, max_taylor_span=3
```

如果三组的动作轨迹几乎完全相同，或全部 `best_span=0`，只生成建议命令，不擅自无限追加：

```text
更保守：tau=0.005
更激进：tau=0.08
```

只有当 `t0p02_h3` 或质量最佳点满足以下两个条件时，才额外运行 `h4`：

```text
1. planned_span == 3 的锚点比例 >= 0.60
2. 与 Full 相比 delta_FID <= 0.5 且 mean_LPIPS <= 0.10
```

额外配置只运行一个：

```text
prt_<selected_tau>_h4
```

不要自动根据 1K FID 宣称最终论文超参数；这里只是先导手工调参。

---

## 10. 追踪要求

每个 NFE 至少记录：

```text
trajectory_id
sample_ids
nfe_index
q
macro_step_index
solver_stage
continuous_t
action
full_reason
active_forecast_order
remaining_taylor_before
remaining_taylor_after
available_feature_order_min
available_feature_order_max
pixel_available_order
planned_span
```

每个 Full 锚点额外记录：

```text
anchor_q
progress
risk_o1_h1 ... risk_o1_hH
risk_o2_h1 ... risk_o2_hH
risk_max_o1_h1 ...
risk_max_o2_h1 ...
safe_h_order1
safe_h_order2
selected_order
selected_span
tau
max_taylor_span
pixel_low_factor_norms
pixel_high_factor_norms
```

每条轨迹汇总至少记录：

```text
total_nfe
full_nfe
taylor_nfe
full_ratio
taylor_ratio
order1_taylor_nfe
order2_taylor_nfe
mean_selected_order
mean_planned_span
max_planned_span
span_0_ratio
span_1_ratio
span_2_ratio
span_3_ratio
span_cap_hit_ratio
network_forward_count
expected_network_forward_count
controller_time_ms
pixel_history_update_time_ms
forecast_time_ms
history_update_time_ms
cache_bytes
pixel_history_bytes
peak_memory_allocated
peak_memory_reserved
```

必须断言：

```text
network_forward_count == existing TaylorSeer expected_network_forward_count
```

不能把 Full/Taylor 动作数当作网络前向数；JiT 的 CFG 两流与 PixelGen 的合并流语义必须保持现状。

---

## 11. 单元测试

至少新增以下 CPU 测试，JiT 与 PixelGen 可共享核心测试：

1. `test_pixel_factors_exact_only`
   - Full 更新像素历史；Taylor 不更新；
2. `test_remainder_formula_order1`
   - 验证 `h^2/2 * P2`；
3. `test_remainder_formula_order2`
   - 验证 `h^3/6 * P3`；
4. `test_low_high_split_reconstruction`
   - `low + high == x`；
5. `test_progress_gates_high_frequency`
   - progress=0 时高频不控制；progress=1 时完整参与；
6. `test_safe_h_monotonic_in_tau`
   - tau 增大时安全跨度不减；
7. `test_safe_h_bounded_by_cap`
   - 不超过 `max_taylor_span`；
8. `test_tie_prefers_lower_order`
   - 相同跨度选一阶；
9. `test_order2_requires_mature_histories`
   - P3 或特征二阶不足时不能选二阶；
10. `test_dynamic_schedule_counts`
    - 给定计划序列，Full/Taylor 动作正确；
11. `test_no_extra_forward_contract_jit`
    - 模拟 99 NFE，JiT 前向数等于现有预期；
12. `test_no_extra_forward_contract_pixelgen`
    - PixelGen 前向数等于现有预期；
13. `test_cfg_real_batch_reduction`
    - CFG 扩展不重复计算样本；
14. `test_state_dict_clean`
    - 控制器和像素历史不进入 checkpoint、EMA、参数或 buffer；
15. `test_nonfinite_forces_full`
    - 非有限风险产生安全 Full 行为；
16. `test_fixed_schedule_parity_mode`
    - 调试模式能复现现有 TaylorSeer 指定 interval/order 的动作和输出语义。

运行两个新目录的完整测试，并记录命令和结果。

---

## 12. 真实模型冒烟

在 1K 前必须依次完成：

### JiT

1. 2 张图片 instrumented Full 对齐；
2. 固定 Taylor 调试模式与原 TaylorSeer `interval=3, order=2` 对齐；
3. 新方法 8 张图片，确认：
   - 只有一次采样轨迹；
   - 无 shadow；
   - 前向总数与原 TaylorSeer 一致；
   - 至少产生一次动态计划；
   - 追踪字段完整；
   - 输出没有 NaN/Inf。

### PixelGen

执行相同流程，额外确认：

- 合并 CFG 顺序仍为 `[unconditional, conditional]`；
- 只对真实 B 样本计算像素控制分数；
- 一个 NFE 只有一个合并模型前向；
- guidance interval 保持上游 predictor/corrector 语义。

冒烟失败时先修复，不得直接启动 1K。

---

## 13. 1K 实验

只运行新方法，不重新运行任何基线。

对 JiT 与 PixelGen 分别运行：

```text
prt_t0p01_h3
prt_t0p02_h3
prt_t0p04_h3
```

必须使用已有 1K manifest 和相同 Full 参考图路径。若已有 Full 图片不在仓库内，从已有协议元数据和环境中定位；不要重新生成 Full，除非用户环境中确实不存在，并明确报告。

每个配置输出到独立目录，支持恢复，禁止覆盖已有结果。

评测：

- FID
- sFID
- Inception Score
- precision
- recall
- 配对 MSE
- PSNR
- SSIM
- LPIPS
- elapsed seconds
- images/s
- speedup vs 已有 Full
- 峰值显存
- Full/Taylor 比例
- 一阶/二阶使用比例
- 计划跨度分布
- 上限命中率

配对评测必须使用相同 sample id/seed 的已有 Full 图片。

---

## 14. 与已有基线汇总

创建：

```text
results/pixel_remainder_taylor_1k_summary.csv
results/pixel_remainder_taylor_1k_trace.csv
results/pixel_remainder_taylor_1k_comparison.csv
results/PIXEL_REMAINDER_TAYLOR_1K_REPORT.md
```

`comparison.csv` 读取而不是重跑：

- `results/taylorseer_1k_summary.csv`
- `results/seacache_1k_summary.csv`
- `results/speca_1k_summary.csv`
- `results/dicache_1k_summary.csv`

每个模型族至少比较：

- Full；
- TaylorSeer 中最相关的 `interval=3, order=1/2`；
- SeaCache 的保守、中等和激进点；
- SpeCa 参考点和最优速度—质量点；
- DiCache 参考点和最优速度—质量点；
- 新方法三个 tau 点。

生成速度—质量 Pareto 标记，至少使用：

```text
speedup vs delta_FID
speedup vs mean_LPIPS
speedup vs aggregate_MSE
```

不要只以 1K FID 的微小随机差异宣称胜出。报告中必须同时讨论配对误差、吞吐和动作轨迹。

主方法是否有初步价值，至少检查：

1. 相比固定 `interval=3, order=2`，能否在相近或更低 LPIPS/MSE 下提高速度；
2. 相比固定 `interval=4`，能否避免明显质量崩坏；
3. 一阶和二阶是否都实际被使用；
4. 计划跨度是否随时间变化，而不是退化成固定 interval；
5. 控制器开销是否远小于节省的 Full 计算。

---

## 15. 完成定义

只有满足以下条件才算完成：

- 新方法没有修改已有基线和 third-party；
- 代码同时支持 JiT 与 PixelGen；
- 方法只有 `tau` 和 `max_taylor_span` 两个对外控制参数；
- 第一轮固定 `max_taylor_span=3`，只扫描三个 tau；
- 不使用标定集、训练、DINO、LPIPS 在线判据、影子前向或额外完整轨迹；
- 每个 NFE 的模型前向数与现有 TaylorSeer 相同；
- 只有 Full 更新特征和像素历史；
- 阶数和连续 Taylor 数由像素余项公式决定；
- CPU 测试通过；
- 真实模型冒烟通过；
- GPU 可用时完成两模型族 1K 新方法实验、评测和已有基线对比；
- 结果、命令、环境、git diff、失败项和未完成项被准确记录；
- 禁止伪造任何实验结果。

最终回复必须包含：

1. 修改文件列表；
2. 方法实现摘要；
3. 测试命令与结果；
4. 冒烟命令与结果；
5. 1K 命令与结果表；
6. 与已有基线的对比结论；
7. 关键 trace 统计；
8. 任何阻塞与可直接运行的后续命令；
9. `git diff --stat`；
10. 明确声明是否存在额外模型前向。
