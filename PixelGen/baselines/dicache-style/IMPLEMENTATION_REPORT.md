# PixelGen DiCache implementation report

This report answers the required 49 questions for the PixelGen port. “Verified” means CPU/static verification only unless a GPU stage is named explicitly.

1. **DiCache commit?** `fdbe20b669c9174bbed5ec994de073fd881c8010`.
2. **Official files referenced?** `FLUX/run_flux_dicache.py`; `WAN2.1/run_wan_dicache.py` and its `wan/` DiCache changes; `HunyuanVideo/run_hunyuanvideo_dicache.py` and its `hyvideo/` changes; the local paper’s probe/DCTA sections. PixelGen references were `src/models/transformer/JiT.py`, `src/diffusion/flow_matching/sampling.py`, and Lightning/EMA code.
3. **FLUX/WAN/Hunyuan differences?** FLUX uses strict `<`, inclusive warmup, final Full, gamma `[1,1.5]`, one image stream, and partial reset. WAN uses separate cond/uncond calls and two independent slots, a different warmup boundary, no final Full, gamma `[1,2]`, and full cleanup. Hunyuan uses one combined batch-global state, `<=`, no final Full, and gamma `[1,1.5]`. See `OFFICIAL_VARIANTS.md`.
4. **Why FLUX image profile?** PixelGen is an image generator and FLUX supplies the closest released image-token behavior. PixelGen’s combined CFG is preserved rather than importing WAN’s two-forward layout.
5. **Probe depth?** Main profile exactly 1. Depth 2/3 exists only as an ablation template.
6. **Exact probe feature?** Image tokens after exact execution of block 0, extracted from the resumable internal state at the upstream context-aware image start index.
7. **Image tokens only?** Yes. Prepended class-context tokens never enter the gate or residual.
8. **Body input?** `x_embedder(x) + pos_embed`, before transformer block 0, on the effective combined 2B tensor.
9. **Full residual?** Exact image-only output after all blocks and context removal, before final layer, minus body input.
10. **Probe residual?** Current image-only probe feature minus the same call’s body input.
11. **Error choice?** Main `delta_y`; `delta_minus` is parity/ablation support.
12. **`delta_y` formula?** `mean(abs(probe_t-probe_prev))/mean(abs(probe_prev))`, reduced over the complete tensor.
13. **Epsilon?** None in main `official_no_epsilon`; an explicit `stable_eps_ablation` exists.
14. **Threshold comparator?** Strict `<` for Reuse.
15. **Equality behavior?** Full resumed from the already-computed probe; accumulator resets.
16. **`ret_ratio` implementation?** Per-stream call index with FLUX `call_index <= int(ret_ratio*total_stream_calls)`.
17. **Warmup off-by-one?** Preserved. N=30/r=.2 has 7 warmup Full calls; N=99/r=.2 has 20.
18. **Forced last Full?** Yes, for the final actual `combined_cfg` call.
19. **How does Full resume?** `ProbeResult` carries internal tokens, next block index, context-inserted flag, and image start; suffix starts at `probe_depth` without reconstructing the prefix.
20. **Are probe blocks repeated?** No. Direct Full captures while traversing once; eligible Full reuses the already-computed internal state.
21. **DCTA anchors?** The second-latest and latest synchronized exact `(full_residual, probe_residual)` pairs.
22. **Gamma formula?** `mean(abs(Pcur-Pold))/mean(abs(Pnew-Pold))`, followed by clipping; `Pcur=probe_feature-body_input`.
23. **Gamma range?** Main `[1.0,1.5]`.
24. **Single-anchor fallback?** Reuse the latest exact full residual (zero order).
25. **Does Reuse write anchors?** No; tests assert exact-only history.
26. **Does previous probe update every call?** Every eligible Reuse or resumed Full updates current previous body/probe. Direct Full also establishes them. Instrumented Full intentionally allocates no cache.
27. **JiT CFG handling?** Not applicable inside this PixelGen directory; the sibling JiT port owns two explicit streams. No cross-import exists.
28. **Can JiT branches disagree?** The sibling design permits it, following WAN. PixelGen cannot because it performs one combined call.
29. **PixelGen combined 2B handling?** Preserve `[unconditional, conditional]`, one forward, one `combined_cfg` runtime stream, one batch-global decision/gamma, and 2B-shaped state.
30. **Batch-global semantics?** All reductions include every token and both CFG halves. Main B=1 gives effective 2; B>1 must be called grouped-batch and needs new threshold validation.
31. **Heun NFE definition?** Every actual expensive model evaluation is one NFE; predictor/corrector/final-euler are distinct, even at repeated continuous time.
32. **50-step NFE/forward count?** `2*50-1=99` NFEs and 99 combined model forwards, derived rather than hard-coded.
33. **Context tokens?** Inserted exactly before upstream `in_context_start`, prepended, retained internally, with the correct RoPE region. Image extraction uses explicit start indices; suffix cannot reinsert context.
34. **`return_layer`/`return_last`?** Their presence forces direct exact Full; upstream-compatible exact feature/last tensors are returned and approximation is never labeled exact.
35. **Runtime in `state_dict`?** No. It is instance-owned non-module state attached with `object.__setattr__`; state-dict tests pass.
36. **EMA/deepcopy safe?** `__deepcopy__` creates a distinct empty runtime and histories/tensors are not shared. This evaluation port is intentionally EMA-only: it rejects `eval_original_model=true`, and generation/benchmark paths strictly load `ema_denoiser.*`. CPU structural tests pass; real checkpoint/EMA execution is deferred.
37. **Compile risks?** Dynamic actions, Python mutation, and `.item()` synchronize/graph-break whole-model compilation. Main comparison is matched eager; blockwise and upstream compile are separately labeled and GPU-deferred.
38. **Cache memory?** Analytic PixelGen-XL BF16 B=1 result: 7,077,888 persistent bytes, six tensors, plus 1,179,648 temporary probe lower bound. Real allocator/3090 peak is unknown.
39. **Unmeasured probe overhead?** Actual prefix kernels, gate/DCTA sync attribution, final-head overlap, attention workspaces, compiler effects, and allocator behavior. Total must use CUDA events; host component timers are diagnostic.
40. **Can current Full support paired metrics?** No: `PAIRED_METRICS_BLOCKED`. Legacy batch 4/continuous RNG does not prove replay for main batch 1. A new manifest-matched `instrumented_full` is required.
41. **Official CPU parity error?** Deterministic finite float32/BF16 fixtures produced global maximum absolute error `0.0` and maximum relative error `0.0`; strict/equality and warmup boundaries matched. The script pins the local commit and checks audited FLUX source expressions before evaluating.
42. **Completed CPU tests?** Final full suite: `110 passed`, `0 skipped`, `0 failed`. It covers official formulas/gate/DCTA, warmup edges, numeric/nonfinite behavior, anchors, previous-state updates, probe resume/context indexing, exact-only history, combined CFG, deepcopy/state dict, exact-Heun lifecycle/counts, shadow counterfactuals and aggregation, JSON-safe durable traces/sample IDs/call counts, fp32-cache output dtype, manifest/sharding/metadata, metrics, compile modes, cross-evidence selection identity, source-bound smoke/compile/release gates, release-prefix SHA/resume identity, cumulative launcher wall clock, and memory estimation.
43. **Deferred GPU tests?** Checkpoint/EMA smoke, Full/Reuse/DCTA occurrence, upstream versus instrumented Full parity, probe-resume parity, real-feature official parity, shadow/probe-depth diagnostics, compile modes, 1K search, 8K validation, matched latency, peak memory, four-GPU 50K, distribution metrics, and paired metrics.
44. **Upstream modified?** No. All edits are below `PixelGen/baselines/dicache-style/`.
45. **CUDA started?** No new CUDA model, context, kernel, compile, generation, or metric workload was started. The overall task’s initial read-only driver query could not communicate; CPU-only imports may issue a failed device-count warning but do not create a workload.
46. **Existing tasks disturbed?** No. The initially observed PixelGen DDP PID `385579` and waiting JiT shell PID `406161` were not attached to, signaled, paused, or changed, and were not repeatedly polled.
47. **Threshold selected?** No. Main candidate and profile retain a null threshold. A disjoint 1K coarse search and 8K validation must precede materialization.
48. **50K run?** No.
49. **Remaining risks?** Real checkpoint compatibility; exact hidden/image parity; unconditional upstream CUDA allocation during construction; EMA/Lightning integration; context/RoPE behavior on GPU; Heun numerical parity; threshold and final gamma-nonfinite policy preregistration; nonfinite frequency; action distribution; compile graph breaks/recompiles; host timing attribution; probe overhead; RTX 3090 peak; quality/latency operating points; four-rank resume; evaluator/reference identity; and final 50K completion.

## Additional engineering decisions

The exact anchor window is `deque(maxlen=2)`, which is equivalent to released readers of the last two entries and prevents ineffective growth. Summary/off tracing does not build per-call event dictionaries or call trace-only `.item()` conversions. Small shadow runs preserve scalar `stream_trace` and `shadow_scalar_series`; aggregation reports probe/full Spearman, zero-order versus DCTA error/improvement, and solver-stage breakdown without retaining tensors.

Probe-finiteness, strict-gate, and DCTA finite/clipping scalar synchronization intervals are carried by the canonical common results and accumulated in `scalar_sync_time_ms`; they are subtracted from gate/DCTA host intervals to avoid double counting. Primary total latency remains CUDA-event based.

The main candidate also leaves `gamma_nonfinite_policy` null. A provisional safe policy may be used for smoke; the final choice must be recorded after smoke and before 1K/8K/final runs. The deferred guard refuses unresolved candidates.
