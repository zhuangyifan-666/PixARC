# Codex 任务：修复 Pixel-Remainder Taylor，使其可可靠完成 1K 实验，并具备后续 50K 扩展能力

你是一名资深扩散模型推理加速研究工程师和实验基础设施工程师。请在当前 `PixARC` 仓库中直接完成代码修复、测试、真实模型冒烟、1K 运行准备，以及 50K 可扩展基础设施。

不要只输出计划、建议或伪代码。你必须实际检查仓库、修改代码、运行测试、验证真实入口，并在条件允许时执行 1K 实验。若 GPU、权重、环境或评测器确实不可用，仍须完成所有不依赖这些资源的代码和测试，并生成精确的阻塞报告与可直接复制运行的剩余命令。禁止伪造实验结果。

---

## 一、最终目标

修复并冻结一个可复现版本的 **Pixel-Remainder Taylor（像素余项控制泰勒预测）**，满足：

1. 新方法每张图片只运行一条正常扩散采样轨迹；
2. 不运行影子预测，不额外运行完整参考轨迹，不在线调用 LPIPS、DINO 或其他视觉编码器；
3. 每个 NFE 的模型前向次数与现有 TaylorSeer 完全一致；
4. 动态刷新造成非均匀 Full 锚点时，特征预测和像素余项估计必须在数学上正确；
5. 可安全恢复中断任务，恢复后的累计耗时不能被低估；
6. 现有 1K manifest 的原始字节、随机种子、分组与已有基线保持不变；
7. 只运行新方法，不重新运行任何已有 Full、TaylorSeer、SeaCache、SpeCa 或 DiCache 基线；
8. 1K 阶段只扫描：

```text
tau = 0.01, 0.02, 0.04
max_taylor_span = 3
```

9. 后续可将 1K 中选出的一个配置直接扩展为 50K，不需要重写采样器、调度器、manifest、恢复、计时或评测管线。

---

## 二、已知仓库状态与已发现问题

先验证，不要盲目假设，但当前快照预计具有以下状态：

```text
当前 HEAD：c97b45d
本地 origin/main：444fba3
```

当前工作区可能有大量换行符变化，以及少量真实未提交修改。已知问题包括：

1. `PixelGen/.../pixelgen_sampler.py` 中错误使用 `sampler.t_eps`，会触发 `NameError`；应使用 `self.t_eps`。
2. 仓库实际旁路元数据为 `manifest_1k.meta.json`，旧代码却只寻找 `manifest_1k.jsonl.meta.json`。
3. 归档或 Windows 换行可能将关键 manifest 从 LF 改成 CRLF，导致 SHA256 与冻结旁路元数据不一致。
4. 旧实现使用针对均匀间隔的递归有限差分，但动态刷新产生非均匀 Full 锚点，二阶及以上预测不再数学正确。
5. PixelGen 的 `pixelgen_main.py` 需要显式绑定上游 `DataModule`：

```python
from src.lightning_data import DataModule

LightningCLI(
    PixelRemainderTaylorLightning,
    DataModule,
    auto_configure_optimizers=False,
    save_config_callback=None,
)
```

6. 旧恢复计时会覆盖上一次耗时，导致中断恢复后速度被高估。
7. 轨迹末尾如果只剩 1 个 NFE，控制器仍可能记录计划跨度 3，污染统计。
8. JiT 和 PixelGen 测试在同一 Python 进程中可能因同名包缓存互相污染，必须分开运行。
9. 当前工作区不干净时，只记录 `git_commit` 不足以复现实验；必须记录实际可执行代码树哈希，并在正式运行前形成干净提交。

本地提交 `444fba3` 已经尝试修复其中多项问题。优先以它为修复基础，但必须审查、测试和补全，而不是无条件认为它正确。

---

## 三、不可违反的算法与公平性约束

### 3.1 不得增加模型前向

50 步精确 Heun 应为 99 个 NFE。

```text
JiT：每个 NFE 条件与无条件各一次，整条轨迹 198 次网络前向
PixelGen：每个 NFE 一次合并的 2B CFG 前向，整条轨迹 99 次网络前向
```

严禁：

- shadow forward；
- 同一 NFE 为多个阶数重复运行模型；
- 为判据额外运行 Full；
- 额外完整采样轨迹；
- 在线 LPIPS、DINO、CLIP 或其他额外神经网络；
- 隐藏控制器、像素分解、缓存或追踪耗时。

### 3.2 只有 Full 更新历史

- Full NFE 更新特征精确锚点；
- Full NFE 更新引导后的像素 `x0` 精确锚点；
- Taylor NFE 只能读取历史，不能写入任何精确锚点；
- 当前选择一阶时，不能删除二阶历史；
- 每条新轨迹必须清空所有运行时状态；
- 运行时状态不得进入模型 `state_dict`。

### 3.3 动态模式必须使用非均匀精确锚点预测

动态 Full 锚点可能位于：

```text
q = 98, 95, 91, 89, ...
```

因此动态模式不能继续把旧递归差分当成统一坐标下的泰勒导数。

动态模式应保存最近精确锚点：

```text
coordinates = [q_{m-k}, ..., q_m]
values      = [F_{m-k}, ..., F_m]
```

使用最近 `order + 1` 个精确锚点进行非均匀拉格朗日多项式外推：

\[
\hat F(q)=\sum_i w_i(q)F(q_i),
\qquad
w_i(q)=\prod_{j\ne i}\frac{q-q_j}{q_i-q_j}.
\]

要求：

- 坐标必须有限、互异；
- 允许坐标单调递减；
- BF16/FP16 缓存预测时使用 FP32 累加，最后只转换一次输出类型；
- 对病态插值权重进行内部安全检查；
- 若候选在执行前发现历史不足、坐标异常或插值病态，必须安全退回当前 NFE 的 Full，不能崩溃，也不能静默使用错误结果；
- 固定调度对齐模式 `fixed_schedule_parity` 继续保留原 TaylorSeer 递归算术，以便与已有固定 `interval=3, order=2` 基线逐像素对齐；动态主方法才使用非均匀多项式预测。

### 3.4 像素锚点与风险判据

采样路径满足：

\[
x_t=t x_0+(1-t)\epsilon.
\]

每个 Full NFE 在 CFG 后，从采样器实际使用的状态和速度恢复：

\[
\hat x_0=x_t+(1-t)v_{\mathrm{guided}}.
\]

像素历史保存最近 4 个 Full 锚点，以支持 0～3 阶多项式；特征历史保存最近 3 个 Full 锚点，以支持 0～2 阶预测。

固定使用：

```text
order_candidates = [1, 2]
pixel_max_order = 3
stored_feature_order = 2
warmup_full_nfe = 3
pool_kernel = 8
batch_reduction = mean
```

低频与高频：

```python
low_small = avg_pool2d(x, kernel_size=8, stride=8)
low = interpolate(low_small, x.shape[-2:], mode="bilinear", align_corners=False)
high = x - low
```

对于候选阶数 `o` 和跨度 `h`，动态模式使用同一目标坐标处的两种多项式候选：

```text
selected  = degree-o forecast
protected = degree-(o+1) forecast
omitted   = protected - selected
```

而不是继续使用均匀网格下的 `h^(o+1)/(o+1)! * P[o+1]`。

每张真实图片的低频和高频相对风险为：

\[
r_{low}=\frac{\operatorname{mean}|L(omitted)|}
{\operatorname{mean}|L(x_{0,anchor})|+10^{-6}},
\]

\[
r_{high}=\frac{\operatorname{mean}|H(omitted)|}
{\operatorname{mean}|H(x_{0,anchor})|+10^{-6}}.
\]

采样进度：

\[
p=\frac{nfe\_index}{total\_nfe-1}.
\]

每张图片：

\[
r_i(o,h)=\max(r_{low,i},p\,r_{high,i}).
\]

主决策使用真实批次平均值：

\[
r(o,h)=\operatorname{mean}_i r_i(o,h).
\]

批次最大值只记录为诊断，不参与主决策。

选择规则：

```python
safe_h[o] = largest h <= min(max_taylor_span, remaining_future_nfe)
            such that risk[o, h] <= tau

choose order with largest safe_h;
on tie choose lower order;
if both safe_h are zero, next NFE is Full.
```

不得增加新的可调权重。主方法对外只保留：

```text
tau
max_taylor_span
```

---

## 四、安全的 Git 工作流程

当前工作区可能不干净。禁止直接执行会丢失用户修改的：

```text
git reset --hard
git clean -fd
git checkout .
git restore .
```

### 4.1 先记录和备份

执行并保存输出：

```bash
git status --porcelain=v1
git rev-parse HEAD
git rev-parse origin/main 2>/dev/null || true
git log --oneline --decorate -8
git diff --binary > ../pixarc_before_codex_worktree.patch
git diff --cached --binary > ../pixarc_before_codex_index.patch
```

### 4.2 优先创建独立干净 worktree

若本地存在提交 `444fba3`，在仓库同级建立：

```bash
git worktree add -b codex/prt-ready ../PixARC_prt_ready 444fba3
```

后续修改、测试和正式实验都在 `../PixARC_prt_ready` 中进行。不要破坏原工作区。

若该提交不存在，则从可用的最新方法提交创建独立 worktree，并手动实现本任务要求；不得依赖网络 fetch 才能继续。

### 4.3 只迁移真实需要的未提交修改

从原工作区重新加入 PixelGen `DataModule` 绑定及其真实行为测试。不要迁移大面积 CRLF 差异、已有结果变化或 third-party 变化。

### 4.4 换行策略

在修复 worktree 中：

```bash
git config core.autocrlf false
```

确保 `.gitattributes` 至少对以下文件强制 LF：

```gitattributes
*.py text eol=lf
*.sh text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
*.jsonl text eol=lf
*.json text eol=lf
*.csv text eol=lf
*.md text eol=lf
```

不要执行全仓库 `git add --renormalize .`，避免制造无关大 diff。只恢复和修改本任务涉及的文件。

冻结 1K manifest 必须恢复为提交中的原始 LF 字节，不能重新生成旁路元数据来迁就 CRLF 文件。修复后校验：

```text
JiT manifest SHA256：e8ddfb2a2470661b7fbc46bd9077c2432195ae2b6986a5b466a760f68797bc1c
PixelGen manifest SHA256：31536470eacf69e07ccd72305e7866957d15859b2091eec7daed2a309cedf5c0
```

若仓库冻结元数据与这里不同，以仓库旁路元数据为真，但必须说明差异，不能静默改写。

---

## 五、必须完成的代码修复

### 5.1 非均匀预测核心

审查并补全：

```text
JiT/methods/pixel-remainder-taylor/pixel_remainder_taylor/finite_difference.py
JiT/methods/pixel-remainder-taylor/pixel_remainder_taylor/state.py
JiT/methods/pixel-remainder-taylor/pixel_remainder_taylor/controller.py
JiT/methods/pixel-remainder-taylor/pixel_remainder_taylor/runtime.py
```

必须具备：

- `nonuniform_lagrange_weights`；
- `nonuniform_polynomial_forecast`；
- 最近精确锚点坐标和值；
- 动态模式使用非均匀预测；
- 固定对齐模式保留旧递归实现；
- FP32 累加；
- 坐标重复、非有限、历史不足、病态权重的 fail-closed 处理；
- 当前 NFE 开始模型分支前进行预测可用性预检，无法安全预测时改为 Full；
- Taylor 分支不改变历史张量、坐标、计数或数据指针；
- 末尾跨度限制为剩余 NFE 数。

不要把 `max_weight_l1` 暴露为新的实验参数；它只能是内部数值安全常数，并在元数据中记录其固定值。

### 5.2 PixelGen 立即崩溃问题

修复：

```text
PixelGen/methods/pixel-remainder-taylor/pixel_remainder_taylor/pixelgen_sampler.py
```

将错误的：

```python
sampler.t_eps
```

改为：

```python
self.t_eps
```

新增真正调用 `_guided_velocity` 的单元测试，不能只做字符串检查。

测试至少验证：

- `1-t < t_eps` 时发生正确截断；
- CFG 的 `[unconditional, conditional]` 顺序不被颠倒；
- 输出批次为真实 B，而不是 2B；
- 不出现未定义变量。

### 5.3 PixelGen 数据模块绑定

修复：

```text
PixelGen/methods/pixel-remainder-taylor/scripts/pixelgen_main.py
```

显式传入：

```python
from src.lightning_data import DataModule
```

并使用：

```python
LightningCLI(
    PixelRemainderTaylorLightning,
    DataModule,
    auto_configure_optimizers=False,
    save_config_callback=None,
)
```

测试不能只搜索字符串。至少使用 monkeypatch 或子进程导入，验证 `LightningCLI` 接收到正确的模型类和数据模块类，并验证生成脚本构造的 resolved YAML 能被 CLI 解析到预测数据集。

### 5.4 manifest 旁路元数据兼容

实现一个唯一的公共解析函数，兼容：

```text
manifest.jsonl.meta.json
manifest.meta.json
```

规则：

- 一个存在：使用它；
- 两个都不存在：明确失败；
- 两个都存在且字节不同：明确失败；
- 两个都存在且字节相同：确定性选择规范名称；
- 仍然调用冻结 baseline 的原始 validator；
- 不得降低 manifest、PyTorch 版本、随机数设备、形状、批次、world size 等验证强度。

启动脚本、JiT 生成脚本、PixelGen 生成脚本和评测脚本必须全部使用同一个解析实现。

### 5.5 运行身份与可复现性

每个输出根目录的 `run_manifest.json` 至少记录：

- `git_commit`；
- 当前可执行方法树 SHA256，包含 `pixel_remainder_taylor/`、`scripts/`、`configs/` 中所有 `.py/.sh/.yaml/.yml`；
- 输入配置 SHA256；
- manifest 和旁路元数据 SHA256；
- checkpoint 路径、大小及已有身份字段；
- 模型族；
- 采样器、步数、CFG、dtype、compile mode；
- `tau`、`max_taylor_span`；
- predictor backend；
- 内部固定数值安全常数；
- 期望 NFE 数和网络前向数；
- Python、PyTorch、CUDA 版本；
- world size、真实 batch size、CFG 执行形式；
- 图像后处理协议。

运行开始后，若同一输出目录的任何身份字段不同，必须拒绝恢复。

正式 1K 前形成一个干净提交，并确保：

```bash
git status --porcelain
```

为空。生成结果不得写进 Git 仓库。

### 5.6 累计计时与安全恢复

采用追加式、不可变的 invocation ledger：

```text
launcher_invocations/<invocation_id>.json
launcher_timing.json
```

每次恢复运行记录：

- 开始/结束时间；
- 本次开始前已有样本数；
- 本次新增样本数；
- 累计样本数；
- manifest SHA256；
- world size；
- 启动返回码。

最终速度使用所有必要 invocation 的累计 wall clock：

```text
cumulative_elapsed_seconds
```

不能只使用最后一次恢复的耗时。

恢复必须满足：

- 已完成样本的 PNG 与 metadata 一致；
- 不重复生成已完成 sample id；
- 不允许重复 metadata sample id；
- 输入配置、manifest、sidecar、checkpoint、代码树哈希不变；
- 中断后 `--resume` 能继续缺失组；
- 已完成的 batch group 不重新计时为生成样本，但恢复启动和检查开销属于累计 wall clock。

### 5.7 启动器硬化

当前方法启动器应吸收已有 TaylorSeer `launch_4gpu_50k.sh` 的成熟保护逻辑，而不是保留一个脆弱简化版。要求：

- 必须显式设置 GPU 运行确认变量；
- `CUDA_VISIBLE_DEVICES` 恰好 4 个唯一设备；
- 检查 GPU UUID，防止同一物理 GPU 被别名重复指定；
- 检查计算进程、明显利用率和显存占用；
- 使用全局 GPU 协调锁和输出目录锁；
- 不杀死、不暂停任何外部进程；
- 输入配置、manifest、sidecar 在输出目录中做不可变快照；
- 非空输出目录没有 `--resume` 时拒绝；
- signal 到来时安全记录状态，不产生虚假的 completed timing；
- 每个 rank 独立日志；
- 支持 JiT 和 PixelGen 使用不同 Python 环境：

```text
PIXEL_REMAINDER_PYTHON=/absolute/path/to/python
```

- 接受任意正整数 `--expected-count`，因此同一启动器可用于 1K 和 50K；
- 启动前校验 manifest 的实际记录数等于 `--expected-count`；
- 输出必须位于仓库外。

### 5.8 追踪模式与 50K 存储

1K 配置保持：

```text
trace_mode = full
```

为 50K 增加：

```text
trace_mode = summary
```

`summary` 模式必须：

- 不改变任何 Full/Taylor 决策；
- 不改变生成图片；
- 不保存每个 NFE 的完整嵌套 trace；
- 仍保存每条 batch trajectory 的：Full/Taylor 数、order1/order2 数、跨度直方图、阶段统计、最大/平均风险摘要、控制器耗时、缓存内存、调用数契约；
- 能被通用聚合脚本读取；
- 1K 与 50K 使用相同采样代码，仅追踪详细程度不同。

新增测试证明，在确定性虚拟模型和相同输入下，`full` 与 `summary` 的动作序列、输出、调用数与汇总统计一致。

### 5.9 控制器开销优化

保持公式与决定完全一致的前提下，尽量把所有候选阶数和跨度的风险计算向量化。

避免在一个 Full 锚点内对每个阶数、跨度、频带反复调用 `.item()` 或 `float(cuda_tensor)` 导致多次 GPU 同步。目标是：

- GPU 上完成候选风险张量计算；
- 每个 Full 锚点最多一次小张量 CPU 传输用于决策和 trace；
- 不引入额外模型前向；
- 与朴素参考实现的决策逐项一致。

添加参考实现对照测试。

---

## 六、必须新增或补强的测试

JiT 和 PixelGen 测试必须使用两个独立 Python 进程运行，避免同名 `taylorseer_style` 包缓存污染。

创建统一入口，例如：

```text
JiT/methods/pixel-remainder-taylor/scripts/run_all_cpu_tests.sh
```

内部先运行 JiT tests，再启动新的 Python 进程运行 PixelGen tests。

### 6.1 数学正确性

至少包括：

1. 常数、一次、二次、三次多项式在非均匀、递减坐标上的精确外推；
2. 关键反例：

```text
f(q)=q^2
anchors q=[10, 7, 3]
target q=2
order=2
expected=4
```

3. BF16/FP16 使用 FP32 累加；
4. 重复坐标、非有限坐标、病态权重明确失败；
5. order override 只读，不破坏更高阶历史；
6. Taylor NFE 不更新精确锚点；
7. 动态模式使用非均匀后端，固定对齐模式使用旧后端；
8. `protected degree o+1 - selected degree o` 的风险计算正确；
9. `tau` 增大时安全跨度不减；
10. 末尾跨度不超过剩余 NFE；
11. 相同安全跨度时优先低阶；
12. 非有限像素锚点或风险强制 Full。

### 6.2 调用数与生命周期

至少包括：

```text
JiT exact Heun 50 steps -> 99 NFE -> 198 forwards
PixelGen exact Heun 50 steps -> 99 NFE -> 99 combined forwards
```

使用带计数器的轻量虚拟模型真实经过采样器路径，不能仅用 `sum(range(99))` 伪测试。

验证：

- predictor/corrector/final Euler 的 NFE 顺序；
- CFG 批次语义；
- 每条轨迹结束后状态完全清空；
- 异常退出也清空状态；
- 当前 Taylor 预检失败时，在模型分支执行前退回 Full；
- 没有额外模型前向。

### 6.3 PixelGen 入口与速度转换

- 真实调用 `_guided_velocity`；
- `self.t_eps`；
- CFG 顺序；
- 真实 B 输出；
- `LightningCLI` 正确绑定 `DataModule`；
- resolved YAML 能加载预测数据集；
- combined 2B 网络前向只执行一次。

### 6.4 真实 manifest 与协议

测试必须直接读取仓库中的两个真实 1K manifest 和 sidecar，验证：

- 两种 sidecar 命名解析；
- SHA256 匹配；
- record count 1000；
- JiT batch 32、world size 4、generator device CUDA；
- PixelGen batch 4、world size 4、generator device CPU；
- CRLF 文件会被拒绝，而不是静默接受；
- 旁路元数据不同会被拒绝；
- 输入快照和恢复身份绑定。

### 6.5 恢复与累计计时

使用临时目录模拟：

1. 第一次运行完成一部分并失败；
2. 第二次 `--resume` 完成剩余部分；
3. 验证累计耗时为两次 invocation 之和；
4. 样本没有重复；
5. manifest 或配置改变后恢复被拒绝；
6. 完整运行后再次 `--resume` 不重新生成样本，但验证仍通过。

### 6.6 50K 通用性

不运行真实 50K，但使用轻量临时 manifest 测试：

- `expected-count` 不再硬编码 1000；
- 50 samples/class、1000 classes 得到 50,000 records；
- 四个 shard 各 12,500；
- JiT batch 32 与 PixelGen batch 4 的分组都有效；
- 50K seed 集与 1K seed 集可通过 `--disjoint-from` 验证互不重叠；
- 通用输出验证器、聚合器和评测器接受 50,000；
- `summary` 追踪不会产生与样本数乘 99 成正比的巨大嵌套 JSON。

---

## 七、代码质量检查

执行：

```bash
python -m compileall \
  JiT/methods/pixel-remainder-taylor \
  PixelGen/methods/pixel-remainder-taylor
```

若环境已有 `ruff`、`pyflakes` 或 `shellcheck`，运行它们；不要为了本任务修改全局环境或安装不必要软件。

分别运行：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
python -m pytest -q -p no:cacheprovider \
  JiT/methods/pixel-remainder-taylor/tests

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
python -m pytest -q -p no:cacheprovider \
  PixelGen/methods/pixel-remainder-taylor/tests
```

再运行统一测试入口。所有测试必须通过。

检查：

```bash
git diff --check
git status --short
```

不得留下 `__pycache__`、`.pytest_cache`、临时日志、输出图片或生成结果。

---

## 八、真实 GPU 冒烟门槛

只有 CPU 测试全部通过后才能使用 GPU。不要抢占繁忙 GPU。

由于正式协议是四卡固定批次，为保证生产入口真实覆盖，构造一个“每个 shard 恰好一个完整 batch group”的 smoke manifest：

```text
JiT：32 images/shard × 4 = 128 images
PixelGen：4 images/shard × 4 = 16 images
```

从冻结 1K manifest 中选择每个 shard 的第一个完整 batch group，保留原 sample id、class、seed、batch_group_id 与 position，并生成与当前模型环境兼容的新 smoke sidecar。smoke manifest 只放在仓库外临时目录，不提交。

对每个模型族依次运行：

### 8.1 instrumented Full

要求：

- 99 NFE；
- JiT 198 forwards；PixelGen 99 forwards；
- 与已有相同 sample id 的 Full 1K 图片逐像素一致；若已有图片根目录未提供，则记录此项为外部阻塞，不可伪造；
- 输出、metadata、trace、timing 和恢复验证通过。

### 8.2 fixed schedule parity

使用：

```text
interval=3
order=2
```

要求与已有 TaylorSeer `i3_k2` 同 sample id 输出逐像素一致。若参考图片根目录不可用，至少验证动作序列、调用数和固定模式算术，并将逐像素对齐列为正式 1K 前的未完成门槛。

### 8.3 动态方法

至少运行：

```text
tau=0.01, max_taylor_span=3
tau=0.04, max_taylor_span=3
```

要求：

- 无 NaN/Inf；
- 99 NFE；
- 调用数准确；
- `full_nfe + taylor_nfe = 99`；
- `order1_taylor_nfe + order2_taylor_nfe = taylor_nfe`；
- 至少一个配置出现 Taylor NFE；
- 较大 tau 的平均 Taylor 比例不得低于较小 tau；若违反，检查实现和样本差异并报告；
- 末尾实际跨度与计划跨度一致；
- 控制器耗时、缓存内存和动作统计完整；
- `--resume` 在人工中断或模拟部分完成后可正确继续。

真实冒烟未通过时，禁止启动 1K。

---

## 九、1K 实验管线

### 9.1 只运行新方法

不要重新运行任何已有 baseline。

对 JiT 和 PixelGen 分别运行：

```text
tau=0.01, max_taylor_span=3
tau=0.02, max_taylor_span=3
tau=0.04, max_taylor_span=3
```

沿用已有冻结 1K manifest：

```text
results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl
results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl
```

输出必须在仓库外，例如：

```text
$RUN_ROOT/1k/jit_prt_t0p01_h3
$RUN_ROOT/1k/jit_prt_t0p02_h3
$RUN_ROOT/1k/jit_prt_t0p04_h3
$RUN_ROOT/1k/pixelgen_prt_t0p01_h3
$RUN_ROOT/1k/pixelgen_prt_t0p02_h3
$RUN_ROOT/1k/pixelgen_prt_t0p04_h3
```

启动器必须支持失败后：

```bash
--resume
```

且累计时间正确。

### 9.2 1K 输出验证

每个运行要求：

- 1000 个唯一 sample id；
- 1000 张有效 256×256 RGB PNG；
- 无缺失、重复和额外文件；
- 每个 sample id 与 manifest 的 class/seed 对应；
- run manifest、输入快照和代码树哈希完整；
- 每条轨迹 99 NFE；
- JiT 每条 198 forwards；PixelGen 每条 99 forwards；
- timing 显示 completed、累计样本数 1000、provenance complete；
- trace 覆盖全部真实样本且无重复；
- 所有统计有限。

### 9.3 通用评测脚本

不要把评测逻辑继续硬编码为 1000。实现通用：

```text
scripts/evaluate_run.py
```

参数至少包括：

```text
--model
--run
--candidate-root
--manifest
--expected-count
--reference-npz
--evaluator
--timing
--trace
--output-dir
--baseline-summary          # 可选，用于速度对比
--paired-reference-root     # 可选，仅 1K 有对应 Full 时使用
--reference-manifest        # 可选
```

1K 时同时计算：

- FID；
- sFID；
- Inception Score；
- precision；
- recall；
- 与已有 Full 的配对 MSE、PSNR、SSIM、LPIPS；
- wall-clock images/s；
- speedup vs Full；
- peak allocated/reserved memory；
- Full/Taylor 比例；
- order1/order2 比例；
- span 直方图；
- 分阶段动作统计；
- 控制器和预测开销。

可以保留 `evaluate_1k.py`，但它应成为调用通用评测器的薄封装，不能复制一套硬编码逻辑。

### 9.4 与已有 baseline 汇总

读取现有：

```text
results/taylorseer_1k_summary.csv
results/seacache_1k_summary.csv
results/speca_1k_summary.csv
results/dicache_1k_summary.csv
```

以及新方法结果，生成：

```text
results/pixel_remainder_taylor_1k_summary.csv
results/pixel_remainder_taylor_1k_trace_summary.csv
results/pixel_remainder_taylor_1k_comparison.csv
```

不得改写已有 baseline CSV。

比较表按模型族分别计算 Pareto 状态，不能跨 JiT 与 PixelGen 混合。

不要由脚本替用户秘密选择最佳 tau。完整输出三点，由用户根据速度—质量曲线选择。

---

## 十、50K 扩展基础设施

本任务不自动运行真实 50K，但必须让 50K 只需“选择 1K 最佳 tau + 构建正式 manifest + 启动”即可。

### 10.1 通用 50K manifest

复用冻结 TaylorSeer 的 manifest 实现，提供明确命令分别构建 JiT 与 PixelGen manifest：

```text
samples_per_class = 50
num_classes = 1000
record_count = 50000
world_size = 4
JiT batch_size = 32, generator_device = cuda
PixelGen batch_size = 4, generator_device = cpu
```

`base_seed` 必须由命令行显式传入，且通过：

```text
--disjoint-from <1k manifest>
```

保证与 1K 调参集不共享随机种子。

写出 manifest 后立即验证：

- 50,000 records；
- 每类 50；
- 每个 shard 12,500；
- batch group 完整；
- sidecar 绑定实际 Python/PyTorch/设备协议；
- manifest 与 sidecar SHA256 记录到运行说明。

### 10.2 50K 配置物化

提供一个小工具或清晰模板，将用户选择的：

```text
tau
max_taylor_span
```

物化成不可变 YAML。50K 默认：

```text
max_taylor_span = 3
trace_mode = summary
```

不得引入新的质量调参项。

### 10.3 50K 启动与恢复

同一个生产启动器接受：

```text
--expected-count 50000
```

并具备：

- 四卡固定真实 batch；
- 任意次数恢复；
- 累计 wall clock；
- 每次 invocation ledger；
- 不重复图片；
- 不覆盖已有输出；
- 输入与代码身份冻结；
- 完成后严格验证 50,000。

### 10.4 50K 评测

通用评测器在没有 Full 50K 配对图时也能工作：

- distribution metrics 必须可运行；
- paired metrics 仅当显式提供匹配 Full 根目录时运行；
- speedup 仅当提供匹配的 Full 50K wall clock 或可信冻结 summary 时计算，否则报告原始 images/s，不伪造 speedup；
- `expected-count=50000`；
- 类别平衡验证；
- summary trace 聚合。

生成一个：

```text
JiT/methods/pixel-remainder-taylor/RUNBOOK_1K_50K.md
```

其中包含从 1K 选择 tau、构建两个 50K manifest、启动、恢复、验证、评测的可直接复制命令。

---

## 十一、正式完成标准

只有以下全部满足时，才能报告“1K ready”：

1. 工作在独立干净分支/worktree；
2. `sampler.t_eps` 问题修复且有真实调用测试；
3. PixelGen `DataModule` 正确绑定且有行为测试；
4. 冻结 1K manifest 原始 SHA256 匹配；
5. 动态模式非均匀多项式预测通过数学反例测试；
6. 固定对齐模式保留旧算法并可与基线对齐；
7. 当前 NFE 的不安全预测可在模型分支前回退 Full；
8. JiT 和 PixelGen CPU 测试分别全部通过；
9. `compileall` 和 `git diff --check` 通过；
10. 四卡真实 smoke 的 Full、fixed parity、dynamic 均通过；
11. 99 NFE 与 198/99 网络前向契约通过；
12. 生产启动器、恢复、累计 timing、输出 validator 通过；
13. 正式代码形成干净提交，结果目录位于仓库外；
14. 1K 三个 tau 配置命令可直接运行；
15. 通用评测和已有 baseline 汇总可运行。

只有以下全部满足时，才能报告“50K infrastructure ready”：

1. 通用 manifest 可构建 50,000 且与 1K seeds 不重叠；
2. 启动器和 validator 无 1000 硬编码；
3. `trace_mode=summary` 不改变采样结果和动作；
4. 恢复和累计计时经过测试；
5. 通用评测接受 50,000；
6. RUNBOOK 命令完整；
7. 不需要修改核心采样代码即可从 1K 扩展到 50K。

---

## 十二、最终交付物

请完成并给出：

1. 一个干净 Git 提交，提交信息建议：

```text
make pixel remainder Taylor 1k-ready and 50k-scalable
```

2. 修改文件清单和关键设计说明；
3. 所有 CPU 测试的精确命令与结果；
4. GPU smoke 的精确命令、输出根目录和结果；
5. 若资源可用，六个新方法 1K 运行及评测结果；
6. 若资源不可用，精确阻塞项，不得使用笼统的“环境问题”；
7. `RUNBOOK_1K_50K.md`；
8. `CODEX_REPAIR_REPORT.md`，包含：
   - 原始 HEAD 和修复提交；
   - manifest SHA256；
   - 代码树 SHA256；
   - 已修复问题；
   - 尚未解决风险；
   - 1K readiness：PASS/FAIL；
   - 50K infrastructure readiness：PASS/FAIL；
   - 可直接复制的下一步命令。

最终回复必须按以下格式：

```text
STATUS
- 1K readiness: PASS/FAIL
- 50K infrastructure readiness: PASS/FAIL
- repaired commit: ...
- worktree: ...

FIXES
- ...

TESTS
- command: ...
  result: ...

GPU SMOKE
- ...

1K RUNS
- completed / not completed
- exact output roots

BLOCKERS
- none / exact blocker evidence

NEXT COMMANDS
- exact copy-paste commands
```

不要在存在任何阻断错误、未通过真实 smoke、manifest hash 不匹配或工作区未冻结时宣称可以开始 1K。
