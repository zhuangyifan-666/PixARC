# Unofficial DiCache-style port for PixelGen

This directory is a self-contained, non-monkey-patched PixelGen integration of Online Probe Profiling and Dynamic Cache Trajectory Alignment (DCTA). It targets the local PixelGen-XL 256 checkpoint and exact-Heun sampler, keeps the upstream combined-CFG call, and does not modify `third-party/PixelGen` or the local DiCache clone.

Status: CPU formulas, state transitions, structural model paths, exact-Heun lifecycle, deterministic manifests, metrics, and launch safety are implemented and tested. Real checkpoint loading, GPU parity, quality, latency, compile behavior, and 50K generation are deferred. `rel_l1_thresh` is intentionally unset in candidate templates; no PixelGen threshold or speedup is claimed.

## Method

DiCache avoids most transformer suffix work when a cheap exact prefix indicates that the current denoising trajectory remains predictable. Each eligible model evaluation executes the first transformer block, measures the change in its image-token output, and either resumes the exact suffix or estimates the full-body residual from exact history.

For current body input `x_t` and probe feature `p_t`:

```text
delta_x = mean(abs(x_t - x_prev)) / mean(abs(x_prev))
delta_y = mean(abs(p_t - p_prev)) / mean(abs(p_prev))
delta_minus = abs(delta_y - delta_x)
accumulated_error += delta_y                 # main profile
Reuse iff accumulated_error < rel_l1_thresh # strict
```

There is no epsilon in `official_no_epsilon`. Equality is Full. `stable_eps_ablation` and `delta_minus` are explicit non-main options. Previous body/probe observations update on every eligible Full or Reuse, so differences are between adjacent observed calls, not relative to the last refresh.

The selected `flux_image_released` profile fixes probe depth 1, `delta_y`, `ret_ratio=.2`, strict `<`, final Full, image tokens, two exact anchor pairs, first-order DCTA, gamma `[1,1.5]`, batch-global reductions, and no epsilon. It is labeled `released_code_faithful_image_profile`; it is not an official native PixelGen implementation.

## Cache boundary and probe

`body_input` is PixelGen’s `x_embedder(x) + pos_embed` image tensor before block 0. `probe_internal_state` is the complete state after exactly `probe_depth` blocks, including context-insertion state and next block index. `probe_feature` is only the image-token slice of that state.

PixelGen prepends class context immediately before block `in_context_start` and keeps it through the remaining blocks. Before insertion, image tokens start at 0; afterward they start at `in_context_len`. The adapter records this mapping, selects the upstream RoPE for the current block region, inserts context once, resumes at the exact original block index, and removes context once before the body output. It never truncates an arbitrary token range to force shapes.

```text
exact_body_output  = image tokens after all transformer blocks, before final_layer
full_residual      = exact_body_output - body_input
probe_residual     = probe_feature - body_input
```

Body input, image probe, exact body output, and both residuals must match in shape/device/dtype at their boundary. Final AdaLN/norm, output projection, and `unpatchify` always run fresh. `return_layer` and `return_last` force direct exact Full and return genuine upstream-compatible intermediates.

## DCTA

Only exact Full calls append synchronized `(full_residual, probe_residual)` anchors. Reuse never writes either history. With one exact anchor, Reuse uses the latest full residual (zero-order fallback). With two:

```text
P_cur = current_probe_feature - current_body_input
gamma_raw = mean(abs(P_cur - P_old)) / mean(abs(P_new - P_old))
gamma = clamp(gamma_raw, 1.0, 1.5)
R_hat = R_old + gamma * (R_new - R_old)
approximated_body_output = current_body_input + R_hat
```

Gamma is a single scalar over the complete combined CFG tensor. The window is a bounded two-anchor deque: released FLUX may append indefinitely, but only reads the last two, so this is a tested memory-equivalent optimization. FP32 cache mode preserves FP32 residual arithmetic but converts the approximated body output back to the model body dtype before the fresh head.

`gamma_nonfinite_policy` supports `official_propagate`, `latest_residual_fallback`, and `force_full`. The policy must be preregistered after smoke/1K validation. `dicache_zero_order` is an ablation and must not be presented as full DiCache.

## Three execution paths

- `DIRECT_FULL`: warmup, first/missing state, last call, diagnostic return, or explicit Full mode. The model executes every block once and captures the probe while crossing the prefix.
- `PROBE_THEN_FULL`: an eligible probe reaches the threshold; execution resumes at `next_block_index`. Prefix blocks are not repeated.
- `PROBE_THEN_REUSE`: the exact prefix runs, the suffix is skipped, DCTA or the single-anchor fallback estimates the body output, then the fresh head runs.

`probe_count` reports eligible gate probes only. A direct Full captures the prefix feature in-line but does not perform or count an extra probe/gate execution.

FLUX warmup preserves its off-by-one condition: `call_index <= int(ret_ratio * total_calls)`. For 30 calls at `.2`, indices 0–6 are Full; for PixelGen exact-Heun 50 (`99` calls), indices 0–19 are Full. `ret_ratio=0` still makes index 0 Full, and `ret_ratio=1` makes every call Full. The final actual combined call is also Full and forced Full clears the accumulator.

## Combined CFG, batches, and Heun

Upstream PixelGen concatenates `[unconditional, conditional]` and performs one effective-2B forward. This port preserves one forward, one explicit `combined_cfg` stream, one action, one accumulator, and one anchor window. It never splits CFG or allows the halves to choose different residuals. This differs from WAN’s two independent calls.

The main protocol is real batch 4/effective CFG batch 8. All four samples share the batch-global action and gamma; this is a `grouped_batch`, not strictly sample-adaptive, protocol. Batch size and fixed grouping affect the threshold and must match Full.

With exact Heun and N macro steps, the adapter records predictor/corrector evaluations independently and derives `2N-1` NFEs. At 50 steps there are 99 combined network forwards, not 50 and not 198. Repeated continuous time is not deduplicated. The corrector retains the upstream use of `t_cur` for the CFG interval test. Warmup and last Full are based on the 99-call stream plan.

Every prediction batch begins one trajectory and ends/resets it in a `finally`-protected lifecycle. Previous inputs, probes, accumulators, anchors, indices, and trace tensors cannot leak across batches.

## PixelGen integration

`DiCachePixelGenJiT` subclasses the local upstream JiT and reuses embeddings, blocks, RoPE, context embeddings, final layer, unpatchify, parameter names, and checkpoint keys. No global monkey patch or copied model file is used. `DiCacheHeunSamplerJiT` mirrors the exact upstream numerical loop while surrounding each real model evaluation with an NFE lifecycle. Generation is deliberately EMA-only: `DiCachePixelGenLightning` rejects `eval_original_model=true`, and both generation and benchmark loaders require the strict `ema_denoiser.*` checkpoint namespace.

PixelGen deep-copies the denoiser for EMA. `DiCacheRuntime.__deepcopy__` creates an independent empty runtime; histories and tensors are not shared. Runtime is attached as a non-module object, is absent from `state_dict`, and is never written to a checkpoint. Checkpoint loading remains strict and uses `ema_denoiser.*` for generation/benchmark paths.

## Modes and configurations

| Mode | Purpose |
|---|---|
| `upstream_full` | untouched upstream oracle; no scheduler/cache overhead |
| `instrumented_full` | matched local body path, every call exact, no persistent cache |
| `probe_shadow_full` | every call exact while recording counterfactual gate/DCTA errors |
| `dicache_zero_order` | probe gate plus latest-residual reuse ablation |
| `dicache` | main probe + DCTA candidate |
| `probe_only_ablation` | probe every eligible call but execute exact suffix |

Main YAMLs use PixelGen-XL 256, BF16 mixed precision, real batch 4/effective CFG batch 8, EMA, 50-step exact Heun, CFG 2.25, timeshift 2, and guidance interval `(0.1,0.9]`. The main candidate contains both `rel_l1_thresh: null` and `gamma_nonfinite_policy: null`; use `scripts/materialize_dicache_config.py` to create an immutable resolved config only with a matching selection report. The generator rejects either unresolved field. A selected report binds one common candidate/checkpoint/manifest identity across the independent 8K paired, trace, and matched single-GPU benchmark evidence. The final launcher also requires a release gate binding both configs, manifest plus sidecar, selection, smoke/resume parity, compile matrix, and the unchanged port/upstream source bytes that produced the evidence. Shadow/ablation reports remain provisional and cannot authorize final 50K.

Compile modes are explicit. `matched_eager` is the main Full/DiCache comparison. `upstream` is valid only for `upstream_full`. `blockwise` is a deferred ablation. Dynamic Python state and `.item()` decisions remain outside compiled blocks. Primary speedup is `instrumented_full / dicache` under the same compile mode; an upstream whole-model result is supplemental.

## Timing and memory

Primary latency uses `torch.cuda.Event` around the complete already-on-GPU batch generation through the final image tensor, with matched warmups. Compile/first-execution time is reported separately. Host `perf_counter` component fields are diagnostics: CUDA is asynchronous, so they do not replace total event timing. Probe-finiteness, gate-threshold, and DCTA finite/clipping `.item()` intervals are accumulated in `scalar_sync_time_ms` and subtracted from gate/DCTA host components to avoid double counting. Probe overhead is reported as probe + gate + scalar sync over measured batch latency, with this limitation attached.

Memory planning and measured peaks use PixelGen-XL BF16 B=4. The CPU estimator excludes attention workspaces, allocator overhead, compilation, model/EMA parameters, and final-head temporaries. Guarded sampler summaries collect peak allocated/reserved memory.

## Reproducibility and evaluation

The manifest fixes sample ID, class, independent CPU seed, shard position, batch group, and within-group order. Each initial Gaussian is constructed with its own CPU generator, so resume/rank/worker scheduling cannot shift later samples. Config, manifest, sidecar, checkpoint size/path/SHA-256, port/upstream source-byte hashes, Git/tree identities, sampler, batch, CFG, dtype, compile mode, DiCache profile, and the final release-gate SHA-256 are recorded. Outputs are numeric RGB uint8 PNGs written atomically. Resume validates complete groups and refuses partial/corrupt/mismatched state or a missing/different archived release gate.

Legacy PixelGen Full outputs are `PAIRED_METRICS_BLOCKED` unless they prove immutable per-sample noise and batch-4 group replay. Generate a new matched `instrumented_full` from the same manifest for strict pairing.

Evaluation stages are:

1. 2–8 image checkpoint/EMA/path smoke.
2. `upstream_full` versus `instrumented_full` exact parity.
3. full-from-body versus probe-then-resume parity.
4. shadow diagnostics and probe-depth 1/2/3 cost/correlation.
5. disjoint 1K coarse threshold sweep, then disjoint 8K fine validation. Do not use 50K to choose a threshold.
6. matched single-GPU latency, compile, and memory benchmark.
7. frozen four-GPU 50K Full and DiCache runs.
8. validated FID, sFID, IS, precision, recall; and only for a mechanically accepted pair, PSNR, SSIM, LPIPS.

Trace summaries include Full/Reuse ratios, probe error, accumulated error, DCTA and zero-order counts, gamma/clipping/nonfinite counts, refresh gaps, component timing, cache size, and optional scalar event traces. Cross-run means are event-count weighted. Fields named `*_over_trajectory_p95` are explicitly P95 over trajectory-level P95 values rather than an exact pooled-call P95. These diagnostics do not replace quality or total latency.

## Safety and limitations

All GPU entry points fail closed unless `DICACHE_GPU_TESTS_ALLOWED=1` and an explicit device allocation are present. Launchers query only target-device telemetry, refuse existing compute PIDs/material utilization, coordinate locks, never signal jobs, refuse non-empty outputs without validated resume, archive immutable inputs plus the exact release gate, and preserve cumulative invocation wall-clock evidence across resumes.

Known unresolved items: threshold and gamma nonfinite policy are not preregistered; upstream/full/resume GPU parity is untested; PixelGen’s constructor has an unconditional CUDA allocation and therefore cannot be safely instantiated in CPU unit tests; compile graph behavior, probe cost, total latency, quality, RTX 3090 peak memory, and 50K completion are unmeasured. The Hunyuan/WAN/FLUX differences and paper/code gaps are documented separately.

See [RUNBOOK.md](RUNBOOK.md) for exact deferred commands and [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) for the 49-question status record.
