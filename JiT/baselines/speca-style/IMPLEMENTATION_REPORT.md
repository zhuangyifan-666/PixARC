# JiT SpeCa implementation report

This report answers the required handoff questions for the unofficial JiT port. “Passed” is never inferred from code inspection; CUDA-dependent work is explicitly deferred.

1. **Cache4Diffusion commit?** `91a1949fcc88acab46547f0b5f295f5de2df2870`.
2. **SpeCa reference files?** `dit/speca-dit/models.py`, `cache_functions/cal_type.py`, `cache_functions/cache_init.py`, `cache_functions/__init__.py`, `taylor_utils/__init__.py`, `sample.py`, `sample_ddp.py`, and the local diffusion/respace loop.
3. **TaylorSeer commit?** `704ee98c74f7f04da443daa3c0aa2cc7803d86e3`; the sibling JiT TaylorSeer port was read-only parity context.
4. **Paper or released code?** Main `scheduler_mode=released_code_faithful` follows released code.
5. **Paper/code differences?** Paper describes global relative-L2/sequential current rejection; code defaults to elementwise relative-L1, local speculative-prefix verification, exact metric-only output, no current rollback, previous-error next-NFE scheduling, and a 0.01 threshold floor.
6. **Draft object?** Each block's complete gate-pre attention output and complete gate-pre MLP output.
7. **Attention/MLP separate?** Yes, independent finite-difference states per layer, module, and CFG stream.
8. **Fresh gates?** Yes; timestep/class conditioning and AdaLN modulation execute every call.
9. **Fresh final head?** Yes; final norm/projection, context removal, and unpatchify always execute on the main path.
10. **Verification layer?** `verify_layer=-1`, resolved to the final Transformer block (JiT-B/16 block 11).
11. **Layer-27 hard-code removed?** Yes; resolution uses actual model depth and validates explicit overrides.
12. **Verifier input prefix?** Speculative, with draft and exact local branches starting from the identical `x_verify_in`.
13. **Exact output writeback?** No; only error statistics leave the branch.
14. **Current failure rollback?** No rollback, replay, solver rewind, or current output replacement.
15. **Error timing?** Current verification is stored at `end_nfe`; it influences the next `begin_nfe`.
16. **Exact error metric?** Main is `mean(abs(pred-exact)/(abs(exact)+eps))` over the mathematically combined cond/uncond tensor and all tokens/channels.
17. **Why relative L1?** It is the local released DiT single/DDP default, so choosing another metric for main would not be code-faithful.
18. **Epsilon?** `1e-10`.
19. **Threshold formula?** `base_threshold * decay_rate**((total_nfe-q)/total_nfe)`.
20. **Floor?** `max(formula, 0.01)` in main.
21. **Comparator?** Strict `previous_error > threshold`; equality passes.
22. **`first_enhance` behavior?** Main released SpeCa uses 3; due the Full-following special branch, startup is Full/Taylor/Full rather than three Full calls. Fixed-draft TaylorSeer parity separately uses 2.
23. **`min_taylor_steps` behavior?** The check flag starts on the `(min+1)`-th Taylor following Full under released counter ordering.
24. **`max_taylor_steps` behavior?** After `max` completed Taylor calls, the next decision is Full.
25. **Is interval ineffective?** In adaptive released SpeCa, yes. It is `null`/ignored; only fixed-draft ablation uses an interval.
26. **Scheduler parity?** Passed inside the 120-test CPU suite: None/low/equal/high errors, min/max boundaries, startup, Full-following Taylor, failure-next-Full, floor, decay 1, and 50/99/100 NFE fixtures are covered.
27. **Error-metric parity?** Passed inside the CPU suite. Toy anchors are L1 1.5, L2 1.581138849, relative-L1 1.25, relative-L2 1.457737923, and cosine error 0.292893291.
28. **Taylor predictor parity?** Passed. The dedicated script ran both float32 and float64 at order 4; each made 120 comparisons with local-TaylorSeer and released-SpeCa maximum absolute error `0.0` and `exact_match=true`.
29. **JiT cond/uncond aggregation?** Independent histories share one decision. Raw payload sufficient statistics yield exactly the metric on concatenated conditional and unconditional batches; the NFE advances after both.
30. **PixelGen combined 2B?** Not used in JiT. The separate PixelGen port preserves one combined `[unconditional, conditional]` state; this JiT port instead has two batch-1 forwards.
31. **Why main batch 1?** The released scalar is batch-global; one real sample prevents unrelated samples sharing the adaptive decision and fixes strict replay grouping.
32. **Heun NFE mapping?** One action per network evaluation, `q=total_nfe-1-nfe_index`; continuous `t` is log-only because it repeats for different states.
33. **50-step counts?** Derived 99 NFE decisions, 99 conditional and 99 unconditional forwards, total 198 JiT calls.
34. **Context tokens?** Inserted at upstream `in_context_start` with context position embedding/RoPE, retained through remaining blocks, then removed before the head.
35. **Context in verification?** Yes in main `all_tokens`, because the final block still carries context.
36. **`return_layer`/`return_last`?** Those are PixelGen-specific upstream diagnostics. JiT diagnostics that request exact intermediates must force Full and record `diagnostic_return`; no draft tensor is advertised as exact.
37. **Runtime in state dict?** No; histories/scheduler/trace are plain non-persistent runtime objects and reset per trajectory.
38. **EMA/deepcopy safe?** JiT preserves upstream checkpoint/EMA loading and keeps runtime out of weights. It does not rely on PixelGen's Lightning deepcopy path.
39. **Compile risks?** Dynamic actions, check toggles, Python state, variable order, dictionaries, and `.item()` can graph-break; scheduler stays outside compiled regions and comparisons must match compile mode.
40. **Cache memory?** Main BF16 order-4 SpeCa analytic cache is 102,236,160 bytes/240 tensors; verifier lower-bound temporaries are 4,423,680 bytes. Matched `instrumented_full` has zero Taylor cache/tensors and no verifier. CUDA peaks are unknown.
41. **Unmeasured verification cost?** Exact final block, input retention/clone, reduction, `.item()` sync, scheduler interaction, kernel workspace, and their ratio to total/matched Full all remain unmeasured on JiT.
42. **Can current Full be paired?** `PAIRED_METRICS_BLOCKED`: existing/planned Full batch 32 differs from registered batch 1 and lacks immutable proof of identical per-sample noise/RNG/grouping.
43. **CPU tests completed?** Yes: `CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/pixarc-speca-jit-pyc PYTHONPATH=PixARC/JiT/baselines/speca-style:PixARC/third-party/JiT /root/miniconda3/envs/jit/bin/python -m pytest -q PixARC/JiT/baselines/speca-style/tests` reported **120 passed, 0 skipped, 0 failed in 4.61 s**, including matched exact-oracle zero-cache, launcher-profile, trace-identity, and non-overlapping diagnostic-accounting regressions. Forced compileall of package/scripts/tests, `bash -n` for every shell script, safe-load of all 9 YAML files, and the 9-file common-core hash/API comparison also passed.
44. **GPU tests deferred?** All model smoke, block/full/draft parity, verification instrumentation, compile, 1K, 8K, benchmark, 50K, metrics, and CUDA memory runs.
45. **Upstream modified?** No Cache4Diffusion, TaylorSeer, vendored JiT/PixelGen, SeaCache, or TaylorSeer-style modification is part of this port.
46. **CUDA started?** No new CUDA work was started by this implementation/documentation task.
47. **Existing jobs disturbed?** No process was stopped, paused, attached, signaled, or modified; no existing reference output was written.
48. **Remaining risks?** GPU/model parity, upstream sampler instrumentation, compile behavior, threshold selection, verifier overhead, 3090 peak memory/OOM, reference checkpoint/EMA identity, matched batch-1 Full, LPIPS assets, evaluator availability, output-scale validation, and 50K completion are unresolved.

See [`RUNBOOK.md`](RUNBOOK.md) for the gated commands and [`BASELINE_COMPATIBILITY_REPORT.md`](BASELINE_COMPATIBILITY_REPORT.md) for the pairing decision.
