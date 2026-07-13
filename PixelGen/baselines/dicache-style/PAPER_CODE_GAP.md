# Paper/code gap

1. The paper’s main probe error is the relative L1 change of the shallow-probe output (`delta_y`).
2. For tensors with the same number of elements, `mean(abs(diff))/mean(abs(reference))` is algebraically the same as the L1-norm ratio because the element-count factors cancel.
3. `delta_minus=abs(delta_y-delta_x)` is a released-code option, not part of the paper’s main formula.
4. The paper does not specify the released `ret_ratio` warmup.
5. It does not specify FLUX’s forced last-step Full.
6. It introduces trajectory alignment but does not prescribe released-code gamma clipping.
7. It does not prescribe a universal gamma upper bound.
8. Consequently, FLUX `[1,1.5]` versus WAN `[1,2]` is backend engineering behavior, not a derivable paper mandate.
9. The paper does not define NaN/Inf handling, epsilon insertion, or gamma fallback.
10. It does not prescribe independent conditional/unconditional decisions.
11. It does not prescribe batch-global versus per-sample tensor reduction.
12. “Sample-specific” describes adaptive trajectories conceptually; released tensor means couple every sample in a batch. This port calls B>1 `grouped_batch`, not strictly sample-adaptive.
13. DCTA equations use two exact trajectory anchors. They do not require an unbounded container.
14. FLUX appends indefinitely and WAN/Hunyuan retain an ineffective older element; all consume only the last two. A bounded deque of two synchronized pairs is semantically equivalent.
15. The paper reports probe costs (FLUX about 0.19 s/4%, WAN-1.3B about 3.98 s/5%, Hunyuan about 11.47 s/2%) but those figures are not PixelGen measurements.
16. Monkey patches and class-level state are released engineering choices, not algorithm requirements.

## Other operational gaps

Algorithm 1 uses a non-strict reuse boundary, whereas FLUX and WAN use strict `<`; the main port follows released FLUX. The paper says the first call is Full but not FLUX’s inclusive off-by-one warmup. The appendix’s principal DCTA is first-order (`k=1`); the higher-order coefficient experiment is not added here. Reset, CFG ordering, exact context-token boundaries, compile behavior, manifest/replay rules, and final-head freshness must be established by each backend integration.

The formal method label is `released_code_faithful_image_profile`, not “official PixelGen DiCache.” The stable-epsilon mode, nonfinite fallbacks, zero-order-only mode, deeper probes, grouped batches, and any future per-sample regrouping are explicit ablations or adaptations.

