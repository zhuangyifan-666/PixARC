# PixelGen SpeCa implementation report

This report answers the required handoff questions for the unofficial PixelGen port. Inspection is not reported as an executed test, and every CUDA-dependent result is deferred.

1. **Cache4Diffusion commit?** `91a1949fcc88acab46547f0b5f295f5de2df2870`.
2. **SpeCa reference files?** `models.py`, `cache_functions/cal_type.py`, `cache_functions/cache_init.py`, `cache_functions/__init__.py`, `taylor_utils/__init__.py`, `sample.py`, `sample_ddp.py`, and local diffusion/respace files.
3. **TaylorSeer commit?** `704ee98c74f7f04da443daa3c0aa2cc7803d86e3`; sibling PixelGen TaylorSeer code was read-only context.
4. **Paper or released code?** Main `released_code_faithful` follows released executable behavior.
5. **Paper/code differences?** Paper global relative-L2/sequential rejection differs from code's default elementwise relative-L1, speculative-prefix local block check, no exact writeback/current rollback, next-NFE use of current error, and threshold floor.
6. **Draft object?** Complete gate-pre attention and MLP outputs per block.
7. **Attention/MLP separate?** Yes, independent states per layer/module within one combined CFG stream.
8. **Fresh gates?** Yes, current conditioning/AdaLN/gates are recomputed.
9. **Fresh final head?** Yes, final norm/projection/context removal/unpatchify are exact on the main path.
10. **Verification layer?** `verify_layer=-1`, resolving the final block (27 for current XL/2).
11. **Layer-27 hard-code removed?** Yes; 27 is a resolved property, not a scheduler literal.
12. **Verifier input prefix?** Speculative; draft and exact local branches share the same `x_verify_in`.
13. **Exact output writeback?** No.
14. **Current failure rollback?** No rollback, replay, state rewind, or current output replacement.
15. **Error timing?** Written at `end_nfe`, consumed at the next `begin_nfe`.
16. **Exact error metric?** `mean(abs(pred-exact)/(abs(exact)+eps))` over full combined CFG batch, all tokens, and channels.
17. **Why relative L1?** It is the audited released DiT default in both local entry points.
18. **Epsilon?** `1e-10`.
19. **Threshold formula?** `base_threshold * decay_rate**((total_nfe-q)/total_nfe)`.
20. **Floor?** `0.01` via `max`.
21. **Comparator?** Strict `>`; equality passes.
22. **`first_enhance` behavior?** Main released SpeCa uses 3 and starts Full/Taylor/Full due the Full-following Taylor branch. Fixed-draft TaylorSeer parity separately uses 2.
23. **`min_taylor_steps` behavior?** Check starts on the `(min+1)`-th Taylor after Full under released ordering.
24. **`max_taylor_steps` behavior?** After `max` completed Taylors, the next action is Full.
25. **Interval ineffective?** In adaptive SpeCa yes; null/ignored there, valid only in fixed-draft mode.
26. **Scheduler parity?** Passed inside the 122-test CPU suite: None/low/equal/high errors, boundaries, startup, Full-following action, failure timing, floor, decay 1, and 50/99/100 counts are covered.
27. **Metric parity?** Passed inside the CPU suite. Audited toy anchors are L1 1.5, L2 1.581138849, relative-L1 1.25, relative-L2 1.457737923, and cosine error 0.292893291.
28. **Predictor parity?** Passed: the dedicated order-4 comparison made 120 comparisons, with local-TaylorSeer and released-SpeCa maximum absolute error `0.0` and `exact_match=true`.
29. **JiT cond/uncond aggregation?** Not used here. The JiT sibling has two histories plus a concatenation-equivalent metric; PixelGen preserves a single combined state.
30. **PixelGen combined 2B?** One `[unconditional, conditional]` effective-`2B` forward, factor store, scheduler, and metric; no split/reorder/half verification.
31. **Why main batch 1?** Batch-global error then corresponds to one real sample while including both CFG halves and fixes strict group replay.
32. **Heun mapping?** One action per combined model evaluation with monotonic `q=total_nfe-1-nfe_index`; repeated continuous `t` is log-only.
33. **50-step counts?** Derived 99 NFE decisions and 99 combined effective-2 forwards for `exact_henu=true`.
34. **Context tokens?** Upstream insertion/position/context handling is preserved through blocks, then removed before head.
35. **Context in verification?** Yes under main `all_tokens`, because context remains at the final block.
36. **`return_layer`/`return_last`?** Either request forces Full and records `forced_full_reason=diagnostic_return`. `return_layer` preserves the upstream 2-/3-tuple exact-feature contract; `return_last=true` without `return_layer` still returns only the image output, exactly as upstream. Diagnostics are excluded from main latency/50K.
37. **Runtime in state dict?** No; factors/scheduler/trace are non-persistent and reset per trajectory.
38. **EMA/deepcopy safe?** Design preserves the same EMA selection; every deepcopy must get an independent empty runtime and never share factors. CPU tests cover this structurally; GPU Lightning parity remains deferred.
39. **Compile risks?** Dynamic action/check, Python state, variable order, combined-layout validation, dictionaries, and `.item()`; scheduler stays outside compiled kernels and comparison modes must match.
40. **Cache memory?** BF16 order-4 real-batch-1 SpeCa cache is 359,792,640 bytes/280 tensors; verifier temporary lower bound is 10,616,832 bytes. Matched `instrumented_full` has zero Taylor cache/tensors and no verifier. CUDA peak unknown.
41. **Unmeasured verifier cost?** Exact block, retained input/payloads, reduction, sync, scheduler/compile interaction, workspace, and ratios to total/matched Full.
42. **Can current Full be paired?** `PAIRED_METRICS_BLOCKED`: current real batch 4 versus main real batch 1 and no immutable proof of identical noise/RNG/grouping.
43. **CPU tests completed?** Yes: `cd /mnt/iset/nfs-main/private/zhuangyifan/PixARC/PixelGen/baselines/speca-style && CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/pixarc-mpl PYTHONPATH=../../third-party/PixelGen:. /root/miniconda3/envs/jit/bin/python -m pytest -q` reported **122 passed, 0 skipped, 0 failed in 4.74 s**, including launcher-profile, cross-shard/resume trajectory-identity, collision-detection, and non-overlapping diagnostic-accounting regressions. Forced compileall, `bash -n`, all five executable-config validators, and the duplicated common-tool hash/API comparison also passed.
44. **GPU tests deferred?** All real model smoke/parity/verification, EMA/deepcopy compile, 1K, 8K, benchmark, 50K, metrics, and CUDA memory.
45. **Upstream modified?** No Cache4Diffusion, TaylorSeer, vendored PixelGen/JiT, SeaCache, or TaylorSeer-style modification.
46. **CUDA started?** No new CUDA work by this task.
47. **Existing jobs disturbed?** No stop, pause, attach, signal, or output overwrite.
48. **Remaining risks?** Real model/block/sampler parity, combined CFG/diagnostic behavior, EMA/deepcopy, compile graphs, threshold selection, verifier overhead, 3090 peak/OOM, checkpoint/EMA/reference proof, LPIPS/evaluator assets, output validation, and 50K completion.

See [`RUNBOOK.md`](RUNBOOK.md) and [`BASELINE_COMPATIBILITY_REPORT.md`](BASELINE_COMPATIBILITY_REPORT.md).
