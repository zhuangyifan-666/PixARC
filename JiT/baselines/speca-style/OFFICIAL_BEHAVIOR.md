# Released SpeCa-DiT behavior

Source of truth: local Cache4Diffusion commit `91a1949fcc88acab46547f0b5f295f5de2df2870`, especially [`models.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/models.py), [`cal_type.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/cache_functions/cal_type.py), [`cache_init.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/cache_functions/cache_init.py), and [`taylor_utils`](../../../baselines/Cache4Diffusion/dit/speca-dit/taylor_utils/__init__.py).

The following answers describe released code, not an interpretation of the paper.

1. The draft predictor is layerwise TaylorSeer finite-difference extrapolation.
2. Every block keeps independent histories for attention and MLP.
3. The predicted tensors are complete attention/MLP outputs before current `gate_msa`/`gate_mlp` multiplication.
4. Taylor NFEs recompute the timestep/class conditioning and AdaLN modulation, so gates are fresh.
5. Ordinary Taylor blocks do not recompute `norm1`, attention, `norm2`, or MLP.
6. Verification occurs at the final Transformer block in the released DiT-XL/2 model.
7. That implementation hard-codes `layer == 27`; the adaptation must resolve the final block structurally.
8. It compares the complete block output, not block input or an isolated module output.
9. The exact branch starts from the same speculative-prefix input as the draft branch, not from a counterfactual fully exact prefix.
10. The exact verifier result is not written to the main path.
11. A failed verification does not roll back or replay the current NFE.
12. `last_layer_error` is produced after the current action executes.
13. It affects the next NFE decision.
14. Therefore the paper's sequential reject/revert description is not identical to released execution.
15. Both released sampling entry points default to `relative_l1`.
16. `relative_l1 = mean(abs(pred-exact)/(abs(exact)+1e-10))`.
17. `relative_l2 = sqrt(mean((abs(pred-exact)/(abs(exact)+1e-10))**2))`.
18. The released L1/L2-relative epsilon is `1e-10`; cosine follows PyTorch cosine-similarity behavior rather than applying that argument.
19. Metrics call `.item()`, producing a host-visible scalar and possible GPU synchronization.
20. Reductions aggregate the full tensor, including batch and tokens; cosine flattens each sample then averages samples.
21. Failure is strictly `previous_error > threshold`; equality passes.
22. `progress = (total_nfe - q) / total_nfe`.
23. `threshold = max(base_threshold * decay_rate**progress, 0.01)`.
24. `first_enhance` is hard-coded to 3 in the released initialization.
25. It does not mean three consecutive Full actions.
26. If the previous type is Full, the next action is unconditionally Taylor, the Taylor counter becomes 1, checking is disabled, and the stored error is cleared.
27. With counter checks performed before increment, verification starts on the `(min_taylor_steps + 1)`-th Taylor after a Full.
28. A Full is selected after `max_taylor_steps` completed consecutive Taylor actions.
29. A Full therefore always has at least one following Taylor action.
30. A failure detected at Taylor NFE `i` can force Full only at NFE `i+1`.
31. The stored error is cleared in the Full-following Taylor branch, not when the error is first computed.
32. `activated_steps` starts as `[total_nfe - 1]`.
33. The first Full appends the same coordinate, so the released list contains a duplicate initial anchor coordinate.
34. `interval` does not participate in released SpeCa scheduling.
35. `cache_counter` is bookkeeping and does not decide an action.
36. Preallocated `cache[i][layer]` dictionaries do not become the active tensor store.
37. The finite-difference factors actually live under `cache[-1]`.
38. Dead/write-only structures include per-step layer dictionaries, `layer_outputs`, and `full_count`; hard-coded DiT-XL/2 details include depth 28, verifier layer 27, and an unconditional per-block clone.
39. `sample.py` and `sample_ddp.py` defaults differ; see [`AUDIT.md`](AUDIT.md).
40. `sample_ddp.py` defaults to real batch 1 per process.

## Exact released scheduler anchors

For the local official function, deterministic synthetic traces produce these counts:

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

These are scheduler fixtures, not JiT quality or speed results.

## Adaptation boundary

The main runtime is named `released_code_faithful`. It preserves the action control flow, previous-error timing, metric, exact-only history, speculative-prefix verifier, and no-writeback behavior. Necessary non-method changes are: a model-depth-relative verify layer, `official_nfe_index` for Heun, two independent JiT CFG histories with one mathematically combined metric, lifecycle validation, and avoiding clones on blocks where no verifier can run. See [`HEUN_ADAPTATION.md`](HEUN_ADAPTATION.md) and [`VERIFICATION_SEMANTICS.md`](VERIFICATION_SEMANTICS.md).

