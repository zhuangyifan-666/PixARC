# SpeCa-style port for JiT

This directory is a self-contained, unofficial JiT integration of the behavior released in Cache4Diffusion's SpeCa-DiT code. The main mode is `released_code_faithful`: it intentionally reproduces the released previous-error scheduler and local verifier, not the paper's idealized current-step reject/rollback description.

No GPU result is included in this implementation pass. Smoke, compile, quality, latency, verification overhead, peak memory, parameter selection, 8K, and 50K runs are all deferred until the user separately authorizes GPU work.

## Method contract

At every model-evaluation (NFE) coordinate, one shared scheduler selects Full or Taylor:

- Full recomputes every JiT block exactly and updates per-block, per-module finite-difference factors.
- Taylor uses layerwise TaylorSeer forecasts for separate gate-pre attention and MLP outputs. It still recomputes timestep/class conditioning, AdaLN modulation, and current gates; it skips ordinary block norms, attention, and MLP.
- Only Full writes history. Forecasts and local exact-verifier features never become anchors.
- Patch/context embeddings, fresh final norm/projection, context removal, and unpatchify always execute normally.

For a checked Taylor NFE, the final Transformer block is evaluated twice from the same speculative-prefix input: once with forecast attention/MLP and once exactly. The complete block outputs are compared, but the main trajectory continues with the speculative output. No current solver state is rejected, repaired, replayed, or rolled back. The new error is written after the NFE and may force the **next** NFE to Full.

The main verifier uses all tokens and the released batch-global metric:

```text
relative_l1 = mean(abs(pred - exact) / (abs(exact) + 1e-10))
failure     = previous_error > max(
                base_threshold * decay_rate**((total_nfe-q)/total_nfe),
                0.01)
```

Equality passes. `.item()`/scalar synchronization remains inside the measured sampling path.

## Scheduler details

Main defaults fixed by released behavior are `first_enhance=3`, `threshold_floor=0.01`, `error_metric=relative_l1`, `error_eps=1e-10`, `verify_layer=-1`, `verification_token_scope=all_tokens`, `gate_mode=batch_global`, `coordinate_mode=official_nfe_index`, and `force_last_full=false`.

`first_enhance=3` does not produce three consecutive Full calls: from the released initial state the first three actions are Full/Taylor/Full. Any Full unconditionally forces the following action to Taylor and clears the stored error in that following branch. Verification begins on the `(min_taylor_steps+1)`-th Taylor after a Full; `max_taylor_steps` forces Full after that many completed Taylor actions. `interval` is ignored in adaptive SpeCa. The separate `taylor_draft_fixed` parity mode uses `first_enhance=2` to match the local TaylorSeer baseline; that value is never substituted into main SpeCa. See [`OFFICIAL_BEHAVIOR.md`](OFFICIAL_BEHAVIOR.md).

## Runtime modes

| mode | Purpose | Valid for main result? |
|---|---|---|
| `upstream_full` | untouched upstream forward, no scheduler/history/verifier | auxiliary Full reference |
| `instrumented_full` | local split exact block path at every NFE; no Taylor history allocation/update and no verifier | matched Full and block parity |
| `taylor_draft_fixed` | fixed action schedule matching TaylorSeer draft behavior | parity/ablation only |
| `speca` | released adaptive scheduler plus local verifier | yes |
| `shadow_verify` | extra exact/draft diagnostics on tiny samples | diagnostics only; never latency/50K |

Config schema is `pixarc-speca-config-v1`, with top-level `model`, `sampling`, `speca`, and `runtime` sections. SpeCa fields include `mode`, `scheduler_mode`, order/threshold/min/max parameters, verification settings, coordinate/gate modes, cache dtype, trace mode, and an interval that is meaningful only for `taylor_draft_fixed`.

## JiT integration

`SpeCaJiT` reuses the upstream JiT embeddings, RoPE, blocks, context layout, final layer, and unpatchify without monkey patching or copying the full model. `SpeCaDenoiser` connects one trajectory lifecycle to JiT Heun sampling, and `SpeCaRuntime` owns non-persistent scheduler/history/trace state. Model parameters and checkpoint/state-dict keys remain upstream-compatible; runtime factors are neither parameters nor persistent buffers.

JiT inserts 32 class-context tokens at `in_context_start`. Full and verifier-exact blocks select the correct image-only or image-plus-context RoPE. Context remains through the last block, so main all-token verification includes it; context is removed before the fresh output head.

The upstream JiT CFG path calls conditional and unconditional models separately. The port keeps independent factor histories but one shared action, coordinate, check flag, threshold, and previous error. Verification sufficient statistics are combined exactly as if cond/uncond payloads were concatenated before applying the official metric. The scheduler advances only after both streams complete.

This differs deliberately from the sibling PixelGen port: PixelGen preserves one combined effective-`2B` `[unconditional, conditional]` forward and one combined history, whereas JiT preserves two batch-1 forwards and two histories. Neither port changes its upstream CFG layout.

At 50 Heun steps, the derived sequence has 99 NFE decisions and 198 JiT forwards (99 per stream). Continuous `t` repeats across corrector/predictor boundaries with different states, so Taylor coordinates use monotonic `q=total_nfe-1-nfe_index`, not `t`. See [`HEUN_ADAPTATION.md`](HEUN_ADAPTATION.md).

## Batch protocol

Main generation uses one real sample per GPU process. Each NFE has two batch-1 stream forwards; this preserves a per-real-sample interpretation despite the released batch-global gate. Every main manifest also fixes one real sample per `batch_group_id`.

A real batch larger than one is a separately labeled grouped-batch SpeCa experiment, requires a new manifest and tuning, and cannot be mixed with main results. Dynamic per-sample ragged grouping is deliberately not implemented. See [`BATCH_SEMANTICS.md`](BATCH_SEMANTICS.md).

## Checkpoint, EMA, compile, and memory

Checkpoint and EMA selection/loading order are inherited. Explicit manifest noise bypasses any fresh `torch.randn` path; runtime state is reset per prediction batch and never saved to a checkpoint.

The primary speedup comparison is `instrumented_full` versus `speca` with identical `matched_eager` or validated `blockwise` mode. The denominator executes the same split exact blocks but deliberately carries no draft-cache allocation/history update/verifier overhead; only a Full action inside adaptive `speca` updates Taylor history. Upstream-compiled Full divided by eager SpeCa is invalid. Compile time is separate and scheduler/dynamic state stays outside compiled regions. See [`COMPILE_COMPATIBILITY.md`](COMPILE_COMPATIBILITY.md).

For main BF16 batch 1, order 4, the analytic JiT Taylor cache is 102,236,160 bytes (240 tensors); explicit verifier temporaries add a lower-bound 4,423,680 bytes. Kernel workspaces and CUDA peaks are unmeasured. No Lite predictor, cache quantization, order reduction, CPU offload, or layer dropping is allowed silently. See [`MEMORY_REPORT.md`](MEMORY_REPORT.md).

## Reproducible generation

The manifest records sample/class IDs, per-sample seed, four-way shard, position, batch group, split, generator protocol, and sidecar hash. Formal final generation is 50,000 images, exactly 50 per ImageNet class and 12,500 per rank; validation is 8,000 images, exactly 8 per class, with disjoint seeds. Noise is constructed independently per sample, so rank order, resume, and prior scheduler decisions do not change it.

Outputs use an explicit external `OUTPUT_ROOT`, atomic temporary-file rename, per-rank JSONL metadata, resolved config, run manifest, summaries, and logs. A non-empty mismatched destination fails. Resume skips only validated matching records and never changes batch grouping or carries history across batches.

The existing/planned JiT Full batch-32 reference is **`PAIRED_METRICS_BLOCKED`** against main batch-1 SpeCa. It may be used for independently validated distribution metrics, but never PSNR/SSIM/LPIPS by filename. A new batch-1 manifest-backed matched Full is required; see [`BASELINE_COMPATIBILITY_REPORT.md`](BASELINE_COMPATIBILITY_REPORT.md).

## Evaluation and performance

After exact output validation, distribution evaluation supports FID, sFID, Inception Score, precision, and recall with one fixed local ADM evaluator/reference NPZ. Strict paired evaluation supports aggregate/per-image PSNR, SSIM, and AlexNet LPIPS from saved RGB uint8 PNGs only after metadata proves identical noise and generation conditions. Missing evaluator, reference NPZ, LPIPS package, or local weights is a hard error/skip; tools do not download or substitute.

Latency uses synchronized CUDA events after inputs are resident and before CPU copy/PNG. It includes embeddings, AdaLN/gates, exact work, forecasting, history, verifier/reduction/scalar sync, scheduler, CFG, Heun, fresh head/unpatchify, cache I/O, and reset. It reports batch-1 latency, common-batch throughput, and four-GPU wall clock separately. `speedup = median_matched_full / median_speca`; the old 50K wall clock is not a denominator.

Real timing reports `verification_block_time_ms`, `metric_reduction_time_ms`/`error_reduction_time_ms`, and `scalar_sync_time_ms`. The registered total-sampling definition is `verification_overhead_ratio = (verification_block_time + error_reduction_time + scalar_sync_time) / total_sampling_time`; the trace aggregator's ratio within its measured subcomponents is diagnostic and cannot replace the CUDA-event denominator.

Summary trace records Full/Taylor/verification ratios and reasons, speculative spans/horizons/orders, error and threshold quantiles, verifier/predictor/history/sync/scheduler/cache timing, cache/peak memory, and real/effective batch. Full trace distinguishes `verification_fail_at_current_nfe` from `next_nfe_forced_full_due_previous_failure`; it never says `rejected_current_step`.

## Required staged workflow

Use [`RUNBOOK.md`](RUNBOOK.md), in order:

1. read-only environment audit and immutable manifest;
2. CPU analytic memory and 2–8-image deferred smoke;
3. upstream-versus-instrumented Full parity;
4. fixed-schedule Taylor draft parity and verification-timing checks;
5. 1K correctness/behavior proxy;
6. independent 8K threshold/order selection;
7. matched single-GPU benchmark;
8. freeze config/hash, then four-GPU 50K;
9. validate outputs, compute distribution metrics, then strict paired metrics only with a matched Full;
10. aggregate trace/latency/memory artifacts.

[`configs/official_single_example_defaults.yaml`](configs/official_single_example_defaults.yaml) and [`configs/official_ddp_defaults.yaml`](configs/official_ddp_defaults.yaml) are read-only released starting points, not claimed JiT optima. Tune order, base threshold, decay, and min/max Taylor spans on 1K/8K only; select conservative/medium/aggressive points by measured latency or speedup, freeze before 50K, and never tune on final data.

## Documents

- [`AUDIT.md`](AUDIT.md): local revisions, inspected sources, defaults, and scope.
- [`OFFICIAL_BEHAVIOR.md`](OFFICIAL_BEHAVIOR.md): exact released control flow and fixtures.
- [`PAPER_CODE_GAP.md`](PAPER_CODE_GAP.md): paper/code/adaptation separation.
- [`VERIFICATION_SEMANTICS.md`](VERIFICATION_SEMANTICS.md): speculative-prefix local check and no writeback.
- [`HEUN_ADAPTATION.md`](HEUN_ADAPTATION.md), [`BATCH_SEMANTICS.md`](BATCH_SEMANTICS.md): solver and batch contracts.
- [`BASELINE_COMPATIBILITY_REPORT.md`](BASELINE_COMPATIBILITY_REPORT.md): blocked old-reference pairing.
- [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md): required 48-question handoff.
- [`COMPILE_COMPATIBILITY.md`](COMPILE_COMPATIBILITY.md), [`MEMORY_REPORT.md`](MEMORY_REPORT.md), and [`NOTICE.md`](NOTICE.md): system/fairness/provenance limits.

## Known limitations

No GPU parity, quality, LPIPS, FID, latency, compile, verification-overhead, or real peak-memory result has been run. Thresholds are not selected for JiT, 3090 feasibility is not measured, and 50K is not generated. The Heun NFE mapping is a necessary adaptation that still needs real sampler instrumentation. Local verification is not full-model error, released previous-error semantics do not repair the current trajectory, batch-global metrics can be dominated by outliers, and paper-style rollback is absent by design.
