# JiT implementation report: required 49 questions

1. **DiCache commit是什么？** `fdbe20b669c9174bbed5ec994de073fd881c8010`。
2. **参考了哪些官方文件？** 本地 `baselines/DiCache/FLUX/run_flux_dicache.py`、`WAN2.1/run_wan_dicache.py` 及其 `wan/` 改动、`HunyuanVideo/run_hunyuanvideo_dicache.py` 及其 `hyvideo/` 改动；JiT 侧参考 `third-party/JiT/model_jit.py` 与 `denoiser.py`。
3. **FLUX、WAN、Hunyuan有哪些实现差异？** FLUX 使用严格 `<`、inclusive warmup、last Full、gamma `[1,1.5]`；WAN 用 cond/uncond 交替双 slot、严格 `<`、无 last Full、gamma `[1,2]`；Hunyuan 用 combined CFG 全局状态、`<=`、无 last Full、gamma `[1,1.5]`。
4. **主profile为什么选择FLUX image行为？** JiT 是图像模型，body/probe 都能在 image-token 边界精确定义；FLUX 同样以 image hidden state 做公开主路径，且其 strict gate、resume 与 DCTA anchor 语义最完整。该选择称为 `released_code_faithful_image_profile`，不声称是官方 JiT 实现。
5. **probe depth是多少？** 主 profile 固定 1；2/3 仅作明确标注的 ablation。
6. **probe feature精确定义是什么？** 从 `body_input` 精确执行前 `probe_depth` 个 JiT block 后的 image-token 输出；完整 resume state 另含全部 token、下一 block index 与 context 是否已插入。
7. **probe是否只包含image tokens？** 是。context 已插入时取 32 个 context token 之后的 image suffix。
8. **body input是什么？** `x_embedder(x) + pos_embed`，即 block 0 之前的 `[B,256,768]` image tensor（JiT-B/16 256）。
9. **full residual是什么？** `exact_body_output - body_input`；`exact_body_output` 是所有 block 后、移除 context、进入 final layer 前的 image tensor。
10. **probe residual是什么？** `probe_feature - body_input`，与对应 exact Full residual 同步写入 anchor。
11. **error_choice是什么？** 主 profile 是 `delta_y`；`delta_minus=abs(delta_y-delta_x)` 仅作 ablation。
12. **delta_y公式是什么？** `mean(abs(probe_t-probe_prev))/mean(abs(probe_prev))`，对一个 CFG stream 的完整 batch/tokens/channels 全局求均值。
13. **是否有epsilon？** 主 `official_no_epsilon` 没有；`stable_eps_ablation` 才显式加入 epsilon。
14. **threshold比较符是什么？** 严格 `<` 才 Reuse。
15. **threshold等于accumulator时做什么？** Full，并清 accumulator；CPU equality test 已覆盖。
16. **ret_ratio如何实现？** 每个 JiT CFG stream 独立用 `call_index <= int(ret_ratio*total_calls)` 直接 Full。
17. **warmup是否有off-by-one？** 有意保留 FLUX inclusive 行为：N=30、0.2 时 0–6；N=99 时 0–19。`ret_ratio=0` 仍 Full index 0。
18. **是否强制last full？** 主 profile 是，每个 cond/uncond stream 的最后调用都 Full。
19. **probe后Full如何resume？** 从保存的 `next_block_index` 与完整 token/context state 继续执行 suffix，使用当前位置对应 RoPE。
20. **probe blocks是否重复计算？** 不重复。CPU mock block visitation test 验证 prefix 只执行一次。
21. **DCTA使用哪两个anchors？** 同一 stream 最近两个 exact Full 产生的同步 `(full_residual, probe_residual)` anchor。
22. **gamma公式是什么？** `mean(abs(P_cur-P_old))/mean(abs(P_new-P_old))`，再用 `R_old + gamma*(R_new-R_old)` 估计 residual。
23. **gamma范围是什么？** 主 profile clamp 到 `[1.0,1.5]`。
24. **单anchor如何fallback？** 直接使用最新 exact full residual，记为 zero-order fallback。
25. **Reuse是否写anchor？** 不写；有 CPU exact-only-anchor test。
26. **previous probe是否每个调用更新？** direct warmup Full 会初始化；之后每个 eligible Full 或 Reuse 都更新为本次 probe，从而保持 adjacent-call 差分。
27. **JiT CFG如何处理？** 保留 upstream 固定顺序：conditional forward 后 unconditional forward；两者是显式隔离的 `cond`/`uncond` state。
28. **JiT两个branch能否动作不同？** 能。各自 gate/accumulator/anchor 独立；summary 统计 disagreement。
29. **PixelGen combined 2B如何处理？** 这是 sibling PixelGen 的一条 combined `[uncond,cond]` stream/一个 batch-global action；JiT 不复用它，而保持两个 B-sized stream。文档明确禁止混同。
30. **batch-global语义如何保持？** 每个 JiT branch 内对完整 `[B,T,D]` 归约为单一 error/gamma/action。主协议 B=1；B>1 必须标为 grouped-batch。
31. **Heun NFE如何定义？** 前 N−1 个 macro step 各 predictor+corrector，最后一步 Euler，因此 `2N-1`；重复连续时间仍是不同 solver-stage observation。
32. **50-step真实NFE和forward数是多少？** 99 NFE；cond/uncond 各 99 次，共 198 JiT network forwards。
33. **context token如何处理？** JiT-B/16 在 block 4 前 prepend 32 token，只插一次；probe state 记录插入状态；body output 取 context 后 image suffix；RoPE 在边界切换。
34. **return_layer/return_last如何处理？** vendored standalone JiT 的 `forward` 没有这两个参数，本 adapter 不虚构接口；标准 upstream `forward` 保持不变。该行为仅属于 PixelGen 变体。
35. **runtime是否进入state_dict？** 不进入。用 `object.__setattr__` 附着非 module runtime；CPU state-dict test 已验证无 cache key/tensor。
36. **EMA/deepcopy是否安全？** standalone JiT 不 deepcopy EMA model；严格加载 checkpoint model 后逐参数复制 `model_ema1`，runtime 仍为空且不进 checkpoint。真实 checkpoint GPU load 尚待 smoke。
37. **compile有什么风险？** Python 动态 action、`.item()` 同步、suffix 分支会 graph-break/recompile。主比较用相同 `matched_eager`；`blockwise` 与 upstream compile 仅 deferred 对照。
38. **cache显存是多少？** JiT-B/16、B=1、双 CFG stream、BF16、两个 anchor pair 的 persistent analytic lower bound 是 4,718,592 bytes/12 tensors；FP32 是 9,437,184 bytes。实际 peak 未测，需 CUDA allocated/reserved 报告。
39. **probe开销尚未实测哪些部分？** first-block CUDA 时间、gate reductions、finite/threshold/clip scalar sync、DCTA、suffix、cache clone/cast、compile effect、allocator/workspace 与端到端 CUDA-event 占比均未实测。
40. **当前Full能否做paired metrics？** 没有完整 immutable manifest/noise/batch/checkpoint/EMA/compile 证据的当前或 legacy Full 不能；状态为 `PAIRED_METRICS_BLOCKED`。需从同一新 manifest 生成 matched instrumented Full。
41. **官方CPU parity误差是多少？** 最新 deterministic core parity 的 global max absolute error `0.0`、global max relative error `0.0`、warmup mismatch `0`。
42. **哪些CPU测试完成？** 最终完整 suite 为 `98 passed`、`0 skipped`、`0 failed`：error/gate/DCTA parity，strict equality，warmup，gamma clipping/non-finite/zero-order，exact-only anchors，probe/resume/context，CFG isolation/disagreement，Heun counts，reset/state_dict，manifest/noise/sharding/resume，JSON-safe durable trace/call-count/output validation，compile-mode structure，三类 selection 证据同一身份，source-bound smoke/compile/release gate，release-prefix SHA/resume identity，累计 launcher wall clock，memory estimator，preflight 和 shadow trace aggregation。
43. **哪些GPU测试deferred？** checkpoint/EMA1 load、upstream-vs-instrumented tensor parity、scratch-vs-resume parity、adaptive smoke、compile、shadow/1K/8K quality selection、latency/peak memory、4-GPU 50K 与所有真实 metrics。
44. **是否修改上游？** 否；修改仅在 `JiT/baselines/dicache-style`。本地 DiCache 与 `third-party/JiT` 只读。
45. **是否启动CUDA？** 否。所有验证使用 `CUDA_VISIBLE_DEVICES=''`；真实模型因 constructor CUDA allocation 未在 CPU test 实例化。
46. **是否干扰已有任务？** 否。未 attach、signal、暂停、终止或改写任何已有任务/输出；GPU launcher 本轮未运行。
47. **threshold是否已选择？** 否；`rel_l1_thresh: null`。此外 `gamma_nonfinite_policy: null`，二者都必须在 disjoint 1K/8K 方案中 preregister/materialize。
48. **50K是否已运行？** 否。launcher 仅实现且 fail-closed，未执行。
49. **还有哪些未验证风险？** real checkpoint key/EMA correctness、GPU numerical parity、context/RoPE 在真实 kernel 下的等价性、BF16/FP32 cache quality、非有限策略、compile graph/recompile、scalar-sync attribution、probe成本、threshold泛化、branch disagreement、quality/FID evaluator、peak memory/OOM、四卡恢复/文件系统并发与最终 50K 完整性均待验证。

最新 CPU 结果必须以最终交接时的重新运行日志为准；上述 parity 数字不等价于真实 JiT GPU 输出 parity。
