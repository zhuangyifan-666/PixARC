# Unofficial DiCache-style port for JiT

This directory is a self-contained JiT integration of Online Probe Profiling and Dynamic Cache Trajectory Alignment (DCTA). It subclasses the vendored model, preserves checkpoint keys and JiT’s conditional-then-unconditional CFG order, owns all cache state per model/trajectory/stream, and does not modify `third-party/JiT` or the local DiCache clone.

Status: CPU core parity, state transitions, context-aware probe/resume, exact-Heun lifecycle, deterministic manifests, metric guards, CPU preflight, and launch safety are implemented. Real checkpoint/model parity, CUDA execution, compile behavior, latency, quality, peak GPU memory, and 50K generation were not run. The candidate config intentionally leaves `rel_l1_thresh` and `gamma_nonfinite_policy` null; no JiT threshold, speedup, or quality result is claimed.

## Main semantics

For the image-token body input `x_t` and exact first-block feature `p_t`:

```text
delta_x = mean(abs(x_t - x_prev)) / mean(abs(x_prev))
delta_y = mean(abs(p_t - p_prev)) / mean(abs(p_prev))
accumulated_error += delta_y
Reuse iff accumulated_error < rel_l1_thresh
delta_minus = abs(delta_y - delta_x)             # ablation only
```

Main profile `flux_image_released` fixes image tokens, probe depth 1, `delta_y`, strict `<`, FLUX inclusive `ret_ratio=.2`, final Full, batch-global reductions, two synchronized exact anchor pairs, first-order DCTA, gamma `[1,1.5]`, and no epsilon. Equality is Full. `delta_minus`, stable epsilon, exact-count warmup, and zero-order-only reuse are explicit ablations.

Inclusive warmup deliberately preserves the released off-by-one rule `call_index <= int(ret_ratio*total_calls)`: at 99 calls and 0.2, indices 0–19 are direct Full. `ret_ratio=0` still makes index 0 Full, and the last call is independently forced Full.

With two exact anchors:

```text
P_cur = current_probe_feature - current_body_input
gamma = clamp(mean(abs(P_cur-P_old))/mean(abs(P_new-P_old)), 1.0, 1.5)
R_hat = R_old + gamma * (R_new-R_old)
approximated_body_output = current_body_input + R_hat
```

Only exact Full calls append anchors. Reuse updates adjacent-call body/probe observations but never writes an estimated anchor. One anchor uses the latest residual as a counted zero-order fallback.

## JiT boundary and execution

```text
fresh x_embedder(x) + pos_embed
  -> 12-block cacheable body
     (32 context tokens prepended before block 4; matching RoPE transition)
  -> image-token suffix
  -> fresh final RMSNorm/AdaLN/projection/unpatchify
```

The saved probe internal state contains tokens, next block index, and context-insertion state. An eligible refresh resumes after the probe without repeating prefix blocks. FP32 residual caches are converted back to the body dtype before the fresh head.

The exact anchor definitions are `full_residual = exact_body_output - body_input` and `probe_residual = probe_feature - body_input`. Their synchronized exact pairs are the only DCTA history.

Execution actions are `DIRECT_FULL`, `FULL_RESUME_FROM_PROBE`, and `REUSE`. `probe_count/probe_time_ms` cover eligible gate probes only; direct Full prefix work is part of exact `suffix_time_ms` and is not double-counted.

JiT performs conditional then unconditional as separate calls. They have independent observations, accumulators, anchors, actions, and refresh histories. For 50-step exact Heun there are 99 NFEs per stream and 198 network forwards. Repeated continuous times remain distinct solver-stage observations.

This intentionally differs from sibling PixelGen, which keeps one combined `[unconditional, conditional]` effective-2B forward, one state, and one action. JiT uses the registered batch-32 grouped protocol: each branch decision is batch-global over all 32 samples.

## Modes

| Mode | Purpose |
|---|---|
| `upstream_full` | untouched upstream oracle, no runtime/cache path |
| `instrumented_full` | matched local split body; every call exact |
| `probe_shadow_full` | exact execution plus counterfactual gate/DCTA diagnostics |
| `dicache_zero_order` | latest-residual reuse ablation |
| `dicache` | main Online Probe + DCTA candidate |
| `probe_only_ablation` | probe/gate instrumentation with exact suffix |

Compile modes are explicit: `upstream` only for upstream Full, `matched_eager` for the primary instrumented-Full/DiCache comparison, and `blockwise` as a deferred ablation. Dynamic scheduling and scalar decisions remain outside compiled blocks.

## Reproducibility and fairness

The main protocol is real batch 32/effective CFG work 64 through two branch forwards. The gate and gamma reduction are batch-global within each branch, so threshold selection and every matched Full/DiCache comparison use the same frozen groups. Each manifest sample uses an independent CPU float32 generator and is copied to the assigned GPU, making noise invariant to sharding/resume order. Runs archive config, manifest, sidecar, and (for final runs) the exact release gate; bind checkpoint path/size/SHA-256, Git/tree IDs, port/upstream source-byte hashes, sampler, CFG, dtype, batch/grouping, compile mode, and DiCache profile; write numeric RGB uint8 PNGs atomically; and reject partial or mismatched resume groups.

Paired PSNR/SSIM/LPIPS requires mechanically identical manifests, per-sample noise, batch grouping, checkpoint/EMA, sampler, CFG, dtype, compile mode, and postprocessing. Existing or future Full outputs without that evidence may support validated unpaired distribution metrics only.

Generation strictly loads `checkpoint["model"]`, then copies `checkpoint["model_ema1"]` into parameters; missing keys fail. The run manifest pins checkpoint path/size and records `EMA1`. A Full output without the same checkpoint/EMA1 and immutable-noise evidence is `PAIRED_METRICS_BLOCKED`.

Standalone JiT does not deepcopy an EMA denoiser: EMA1 is copied into this single model before CUDA evaluation. The non-module runtime remains empty during loading and absent from `state_dict`; trajectory reset releases it after every batch. Resume accepts only complete, identity-matched batch groups and never silently advances RNG state.

Threshold and gamma safety policy must be selected/preregistered on disjoint pilot/validation data, materialized into a new immutable config, and frozen before final 50K. The template cannot run because both fields are null. A selected report must bind one common candidate/checkpoint/manifest identity across the independent 8K paired, trace, and matched single-GPU benchmark artifacts. The final launcher additionally requires a release gate binding both final configs, the manifest and sidecar, selection, smoke/resume parity, compile-matrix evidence, and the unchanged port/upstream source bytes used to create those gates.

Summary aggregation weights gamma, accumulated-error, and refresh-gap means by their underlying event counts. The cross-trajectory P95 fields are deliberately named `*_over_trajectory_p95`: they are P95 over trajectory-level P95 values, not an exact pooled-call P95. Full/shadow traces retain scalar calls only on small diagnostics; the 50K summary never retains feature tensors.

## Timing and memory

Primary latency uses CUDA events around complete already-on-GPU generation through the final image tensor. Component fields use host `perf_counter` and are diagnostic on asynchronous CUDA. Gate-error finite checks, threshold comparison, and DCTA finite/clip `.item()` calls are accumulated in `scalar_sync_time_ms`; that host interval can absorb queued work and must not be treated as kernel attribution. Compile/first-execution time is separate.

Persistent cache accounting includes previous body/probe observations and two full/probe residual pairs per CFG stream. The CPU estimator excludes model weights, attention/workspace temporaries, allocator overhead, compilation, and final-head buffers. Actual allocated/reserved peak memory must come from the guarded GPU benchmark.

## Deferred experiment stages

1. Run CPU tests, official-core parity, common-core comparison, and config preflight.
2. On 2–8 samples, check checkpoint/EMA1 loading, upstream-Full versus instrumented-Full parity, full-body versus probe-resume parity, 99/198 call counts, CFG isolation, reset, dtype, and non-finite handling.
3. Run `probe_shadow_full` and probe-depth 1/2/3 diagnostics; report probe/exact-residual Spearman correlation and DCTA improvement over zero order by solver stage.
4. Use a disjoint 1K pilot for the preregistered coarse sweep, then a disjoint 8K set for frozen threshold/policy selection. Never tune on final 50K.
5. Run the matched single-GPU CUDA-event benchmark for instrumented Full and DiCache, including first execution, host component diagnostics, and allocated/reserved peaks.
6. Freeze config and run new matched Full/DiCache 50K generations on four idle assigned GPUs with identical manifests.
7. Validate artifacts, then report FID, sFID, IS, precision, and recall with pinned evaluator/reference hashes. Report PSNR, SSIM, and LPIPS only if strict pairing passes.

FLUX, WAN, and Hunyuan differ in CFG state layout, threshold equality, warmup/final behavior, and gamma range; [OFFICIAL_VARIANTS.md](OFFICIAL_VARIANTS.md) gives the concrete table. Known limitations are the unresolved candidate fields, no real-model/CUDA parity, no compile/latency/memory measurement, no selected operating point, and no completed quality/50K result.

## Safety

All GPU entry points require `DICACHE_GPU_TESTS_ALLOWED=1`, explicit allocated devices, and idle-target telemetry checks. Four-GPU launch uses physical-GPU locks, never signals existing jobs, archives immutable inputs and the release gate, refuses unsafe non-empty output roots, validates complete groups on resume, and accumulates immutable invocation wall clocks rather than relabeling the last resumed suffix as 50K throughput. This implementation turn did not use CUDA.

Use [RUNBOOK.md](RUNBOOK.md) for exact commands, [AUDIT.md](AUDIT.md) for source evidence, [OFFICIAL_VARIANTS.md](OFFICIAL_VARIANTS.md) and [PAPER_CODE_GAP.md](PAPER_CODE_GAP.md) for semantic differences, and [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) for the 49-item status record.
