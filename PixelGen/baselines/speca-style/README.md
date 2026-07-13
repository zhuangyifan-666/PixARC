# SpeCa-style port for PixelGen

This directory is a self-contained, unofficial PixelGen integration of the behavior released in Cache4Diffusion's SpeCa-DiT code. Main `released_code_faithful` mode preserves released previous-error scheduling and local verification; it does not silently implement the paper's idealized reject-and-rollback semantics.

This implementation pass contains no GPU result. Model smoke, compile, parity, quality, latency, verification overhead, peak memory, tuning, 8K, and 50K work are all deferred until separately authorized.

## Method contract

One scheduler action is selected for every expensive Heun model evaluation:

- Full executes exact blocks and updates independent per-layer attention/MLP finite-difference factors.
- Taylor forecasts the complete gate-pre attention and MLP outputs separately. Current timestep/class conditioning, AdaLN modulation, and gates are still recomputed; ordinary draft blocks skip norms, attention, and MLP.
- Only Full updates exact history. Predictions and exact local-verifier tensors never update it.
- Patch/context embeddings, final norm/projection, context removal, and unpatchify stay fresh.

When a Taylor NFE is checked, the final Transformer block has a draft and local exact branch from the same speculative-prefix input. Complete block outputs are compared, but the current main path retains the draft output. Failure does not roll back, replay, replace, or repair the current solver step. Its scalar is written after the NFE and may force the next NFE Full.

Main verification is all-token, batch-global released `relative_l1`:

```text
relative_l1 = mean(abs(pred - exact) / (abs(exact) + 1e-10))
failure     = previous_error > max(
                base_threshold * decay_rate**((total_nfe-q)/total_nfe),
                0.01)
```

Equality passes and scalar synchronization is part of latency.

## Scheduler behavior

Fixed released settings are `first_enhance=3`, `threshold_floor=0.01`, `error_metric=relative_l1`, `error_eps=1e-10`, `verify_layer=-1`, `verification_token_scope=all_tokens`, `gate_mode=batch_global`, `coordinate_mode=official_nfe_index`, and `force_last_full=false`.

The released startup is Full/Taylor/Full, not three Full actions. Any Full forces the next action Taylor and clears stored error in that branch. Checking starts on the `(min_taylor_steps+1)`-th Taylor after Full; the next Full is forced after `max_taylor_steps` completed Taylors. `interval` is ignored by adaptive SpeCa. The separate `taylor_draft_fixed` parity mode uses `first_enhance=2` to match the local TaylorSeer baseline; that value is never substituted into main SpeCa. See [`OFFICIAL_BEHAVIOR.md`](OFFICIAL_BEHAVIOR.md).

## Runtime modes

| mode | Purpose | Main result? |
|---|---|---|
| `upstream_full` | original PixelGen forward; no history/scheduler/verifier | auxiliary reference |
| `instrumented_full` | local exact split-block path on every NFE; no Taylor history allocation/update and no verifier | matched Full/parity |
| `taylor_draft_fixed` | fixed schedule for TaylorSeer draft parity | ablation only |
| `speca` | released adaptive scheduler/local verifier | yes |
| `shadow_verify` | extra exact/draft diagnostics on tiny data | never latency/50K |

Config schema `pixarc-speca-config-v1` contains top-level `model`, `sampling`, `speca`, and `runtime`; SpeCa fields cover mode/scheduler, order/threshold/min/max, verifier, batch gate, NFE coordinate, cache dtype, trace, and fixed-only interval.

## PixelGen integration

The inner package is `speca_style` because the outer `speca-style` path contains a hyphen. `SpeCaPixelGenJiT` reuses upstream modules/weights, `SpeCaHeunSamplerJiT` applies one runtime action per model evaluation, and `SpeCaPixelGenLightning` preserves prediction/EMA behavior with an inference-only trainer wrapper where required. Use `PYTHONPATH=$PIXARC_ROOT/third-party/PixelGen:$PIXARC_ROOT/PixelGen/baselines/speca-style:$PYTHONPATH`.

PixelGen's CFG layout is unchanged:

```text
cfg_x         = cat([x, x], dim=0)
cfg_condition = cat([uncondition, condition], dim=0)
```

One combined `[unconditional, conditional]` effective-`2B` forward owns one history, one scheduler, and one metric. It is never split, reordered, half-verified, or independently scheduled. Main real batch is 1, hence effective batch 2.

This differs deliberately from the sibling JiT port: JiT keeps separate conditional and unconditional Taylor histories while sharing one scheduler and combines verification sufficient statistics exactly as if the two stream payloads were concatenated. Neither port changes its upstream CFG execution layout.

Context remains through the final block, so all-token verification includes image and context tokens. The context is removed before the always-fresh output head. `return_layer` or `return_last` requests force Full and return exact features while recording `diagnostic_return`; diagnostics are not latency/50K paths.

For `exact_henu=true`, 50 steps derive 99 decisions/combined forwards. Continuous `t` can repeat for different predictor/corrector states, so the monotonic Taylor coordinate is `q=total_nfe-1-nfe_index`; continuous times are logs only. See [`HEUN_ADAPTATION.md`](HEUN_ADAPTATION.md).

## EMA, deepcopy, lifecycle, and batch

Checkpoint/EMA selection remains upstream-compatible. PixelGen may deepcopy the denoiser for `ema_denoiser`; runtime state is non-persistent and every copy starts empty/independent. Prediction keeps the same EMA policy. Each batch calls trajectory begin and guarantees end/reset in `finally`, preventing factor/error/counter leakage.

Real batch 1 is the registered protocol because the released metric/action is batch-global. A larger real batch is separately labeled grouped-batch SpeCa, uses effective `2B`, requires a new manifest/tuning, and cannot be mixed with main results or called strictly sample-adaptive. See [`BATCH_SEMANTICS.md`](BATCH_SEMANTICS.md).

## Compile and memory

Primary speedup compares `instrumented_full` and `speca` under the same `matched_eager` or validated `blockwise` mode. The denominator executes the same split exact blocks but carries no Taylor history allocation/update or verifier; only Full actions inside adaptive `speca` update draft history. An upstream-compiled Full/eager-SpeCa quotient is invalid. Scheduler/state/`.item()` stay outside compiled regions and compile time is separate. EMA/deepcopy compile independence must be tested on GPU. See [`COMPILE_COMPATIBILITY.md`](COMPILE_COMPATIBILITY.md).

At real batch 1/effective 2, BF16 order 4, the analytic Taylor cache is 359,792,640 bytes (280 tensors); explicit verifier temporary lower bound is 10,616,832 bytes. This excludes attention/compiler/CUDA workspaces and is not a 3090 peak. No silent Lite, order reduction, dtype change, offload, compression, or layer dropping is allowed. See [`MEMORY_REPORT.md`](MEMORY_REPORT.md).

## Manifest, outputs, and current reference

The immutable manifest records sample/class IDs, per-sample seeds, four-way shard, position, one-sample batch group, split, noise protocol, and sidecar hash. Final is 50,000 samples, exactly 50/class and 12,500/rank; validation is 8,000, exactly 8/class, with disjoint seeds. Noise is per-sample explicit and independent of rank order, resume, earlier samples, and SpeCa actions.

Generation writes only to explicit external `OUTPUT_ROOT` using atomic rename, resolved config/run manifest, per-rank JSONL, summaries, and logs. A non-empty incompatible destination fails. Resume skips only matching valid records, preserves grouping, and resets SpeCa per batch.

The current PixelGen Full real-batch-4 reference is **`PAIRED_METRICS_BLOCKED`** against main batch-1 SpeCa because exact noise/RNG/group replay is not proven. It can support validated unpaired distribution metrics but never PSNR/SSIM/LPIPS by filename. A new manifest-backed batch-1 matched Full is required; see [`BASELINE_COMPATIBILITY_REPORT.md`](BASELINE_COMPATIBILITY_REPORT.md).

## Evaluation, latency, and trace

Validated 50K outputs support FID, sFID, Inception Score, precision, and recall with one fixed local ADM evaluator/reference NPZ. Strict paired output supports PSNR, SSIM, and local AlexNet LPIPS from saved RGB uint8 PNG only after run metadata proves identical conditions. Tools never download missing reference/evaluator/weights and never substitute another LPIPS backbone.

Synchronized CUDA-event timing starts after resident inputs/reset and ends after the clamped image tensor but before CPU/PNG. It includes embeddings, gates, exact/draft/history/verifier/reduction/`.item()`/scheduler, combined CFG, Heun, fresh head/unpatchify, cache I/O, and reset. Report batch-1 latency, common-batch throughput, and four-GPU wall clock separately. `speedup = median_matched_full / median_speca`; old 50K wall time is not a denominator.

Real timing reports `verification_block_time_ms`, `metric_reduction_time_ms`/`error_reduction_time_ms`, and `scalar_sync_time_ms`. The registered total-sampling definition is `verification_overhead_ratio = (verification_block_time + error_reduction_time + scalar_sync_time) / total_sampling_time`; the trace aggregator's subcomponent-only ratio is diagnostic and does not replace the CUDA-event denominator.

Summary trace records Full/Taylor/verification ratios/reasons, speculative spans/horizons/order, error/threshold quantiles, fine-grained timing, cache and peak memory, and real/effective batch. Full trace says `verification_fail_at_current_nfe` and, if applicable, `next_nfe_forced_full_due_previous_failure`, never `rejected_current_step`.

## Staged workflow

Follow [`RUNBOOK.md`](RUNBOOK.md) in order: read-only audit; immutable batch-1 manifests; analytic memory; deferred 2–8 image smoke; upstream/instrumented Full parity; fixed-schedule draft parity; verification semantics; 1K proxy; independent 8K selection; matched single-GPU benchmark; frozen four-GPU 50K; output validation; distribution metrics; strict paired metrics only against a new matched Full; trace/latency/memory aggregation.

[`configs/official_single_example_defaults.yaml`](configs/official_single_example_defaults.yaml) and [`configs/official_ddp_defaults.yaml`](configs/official_ddp_defaults.yaml) are read-only released starting points, not PixelGen optima. Tune order, base/decay threshold, and min/max spans on 1K/8K, select operating points by measured latency/speedup, freeze the selected config/hash, and never tune on final 50K.

## Documents

- [`AUDIT.md`](AUDIT.md), [`OFFICIAL_BEHAVIOR.md`](OFFICIAL_BEHAVIOR.md), [`PAPER_CODE_GAP.md`](PAPER_CODE_GAP.md).
- [`VERIFICATION_SEMANTICS.md`](VERIFICATION_SEMANTICS.md), [`HEUN_ADAPTATION.md`](HEUN_ADAPTATION.md), [`BATCH_SEMANTICS.md`](BATCH_SEMANTICS.md).
- [`BASELINE_COMPATIBILITY_REPORT.md`](BASELINE_COMPATIBILITY_REPORT.md), [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md).
- [`COMPILE_COMPATIBILITY.md`](COMPILE_COMPATIBILITY.md), [`MEMORY_REPORT.md`](MEMORY_REPORT.md), [`NOTICE.md`](NOTICE.md).

## Known limitations

No GPU parity, smoke, quality, LPIPS/FID, latency, compile, verifier-overhead, or real peak-memory result exists yet. PixelGen thresholds are not selected, 3090 feasibility is unknown, and 50K is not generated. Heun mapping, EMA/deepcopy and diagnostic-return behavior still require real-model checks. The local verifier is not a full exact trajectory check; previous-error behavior never repairs the current output; batch-global reduction can hide/outweight individual behavior; paper-style rollback is intentionally absent.
