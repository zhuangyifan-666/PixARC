# Released SpeCa-DiT behavior

Source of truth: local Cache4Diffusion commit `91a1949fcc88acab46547f0b5f295f5de2df2870`, especially [`models.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/models.py), [`cal_type.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/cache_functions/cal_type.py), [`cache_init.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/cache_functions/cache_init.py), and [`taylor_utils`](../../../baselines/Cache4Diffusion/dit/speca-dit/taylor_utils/__init__.py).

The following answers describe released code, not an interpretation of the paper.

1. The draft predictor is layerwise TaylorSeer finite-difference extrapolation.
2. Every block keeps independent histories for attention and MLP.
3. Predicted tensors are complete attention/MLP outputs before current gates.
4. Taylor NFEs recompute timestep/class conditioning and AdaLN modulation, so gates are fresh.
5. Ordinary Taylor blocks skip `norm1`, attention, `norm2`, and MLP.
6. Verification occurs at the final Transformer block in released DiT-XL/2.
7. That implementation hard-codes block 27; this adaptation resolves the final block structurally.
8. It compares complete post-residual block outputs.
9. The exact branch starts from the same speculative-prefix input as the draft branch.
10. Exact verifier output is not written to the main path.
11. Failure does not roll back or replay the current NFE.
12. `last_layer_error` is produced after the current action.
13. It affects the next NFE decision.
14. Paper sequential reject/revert prose is therefore not identical to released execution.
15. Both released entry points default to `relative_l1`.
16. `relative_l1 = mean(abs(pred-exact)/(abs(exact)+1e-10))`.
17. `relative_l2 = sqrt(mean((abs(pred-exact)/(abs(exact)+1e-10))**2))`.
18. Released relative metrics use epsilon `1e-10`; cosine follows PyTorch cosine-similarity epsilon behavior.
19. Metrics call `.item()`, with possible GPU synchronization.
20. Reductions cover the entire batch/tokens/features; cosine flattens each sample then averages samples.
21. Failure is strictly `previous_error > threshold`; equality passes.
22. `progress = (total_nfe - q) / total_nfe`.
23. `threshold = max(base_threshold * decay_rate**progress, 0.01)`.
24. `first_enhance=3`.
25. It does not imply three consecutive Full actions.
26. A previous Full forces Taylor, sets its counter to 1, disables checking, and clears stored error.
27. Verification starts on the `(min_taylor_steps+1)`-th Taylor after Full.
28. Full follows `max_taylor_steps` completed consecutive Taylor actions.
29. Every Full has at least one following Taylor.
30. Error at Taylor NFE `i` can force Full only at `i+1`.
31. Stored error clears in the Full-following Taylor branch.
32. `activated_steps` initializes as `[total_nfe-1]`.
33. First Full appends the same coordinate, producing a duplicate initial list entry.
34. `interval` is not an adaptive scheduler input.
35. `cache_counter` is bookkeeping, not a decision variable.
36. Preallocated `cache[i][layer]` dictionaries do not hold the active factors.
37. Active factors live under `cache[-1]`.
38. Per-step layer dictionaries and `layer_outputs` are dead storage; `full_count` is write-only; depth 28/layer 27 and broad clones are DiT-XL/2-specific.
39. `sample.py` and `sample_ddp.py` defaults differ; see [`AUDIT.md`](AUDIT.md).
40. DDP default per-process real batch is 1.

## Exact released scheduler anchors

| Parameters | total NFE | error stream | Full | Taylor | verified Taylor |
|---|---:|---|---:|---:|---:|
| single defaults | 50 | all pass | 9 | 41 | 24 |
| single defaults | 50 | all fail | 13 | 37 | 12 |
| DDP defaults | 50 | all pass | 7 | 43 | 25 |
| DDP defaults | 50 | all fail | 11 | 39 | 9 |
| DDP defaults | 99 | all pass | 12 | 87 | 53 |
| DDP defaults | 99 | all fail | 21 | 78 | 19 |
| DDP defaults | 100 | all pass | 12 | 88 | 54 |
| DDP defaults | 100 | all fail | 21 | 79 | 19 |

These are scheduler parity fixtures, not PixelGen performance results.

## Adaptation boundary

`released_code_faithful` preserves the scheduler, metric, previous-error timing, exact-only factors, speculative-prefix verifier, and no-writeback behavior. Necessary adaptation consists of depth-relative verification, monotonic Heun NFE coordinates, PixelGen's existing combined `[unconditional, conditional]` `2B` layout, lifecycle/deepcopy validation, exact diagnostic returns, and cloning only when a check can run. See [`HEUN_ADAPTATION.md`](HEUN_ADAPTATION.md) and [`VERIFICATION_SEMANTICS.md`](VERIFICATION_SEMANTICS.md).

