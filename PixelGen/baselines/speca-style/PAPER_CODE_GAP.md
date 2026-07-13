# SpeCa paper versus released code

The registered PixelGen baseline follows local Cache4Diffusion commit `91a1949fcc88acab46547f0b5f295f5de2df2870`. Paper motivation and released execution are documented separately. No paper PDF is vendored in this checkout; the paper-described column records the version linked by the local [`Cache4Diffusion README`](../../../baselines/Cache4Diffusion/README.md), while local source remains the executable authority.

## Three contracts

| Topic | Paper-described behavior | Released-code behavior | PixelGen adaptation behavior |
|---|---|---|---|
| draft | TaylorSeer predicts several future states | one online per-NFE action; per-block attention/MLP factors | same, held in one combined effective-`2B` state |
| error | global relative L2 norm, paper epsilon `1e-8` | default elementwise batch-global relative L1, epsilon `1e-10` | same metric over full `[unconditional, conditional]` tensor |
| validation | sequential future acceptance | one local exact check on current Taylor NFE | one local check per combined Heun NFE |
| rejection | current offending prediction is described as rejected/reverted | current draft is retained; no rollback/replay | current combined output feeds CFG/Heun unchanged |
| exact input | can be read as an exact target | exact block starts from speculative prefix | same combined speculative-prefix input |
| exact writeback | reject semantics imply exact recovery | no writeback/history update | metric-only; diagnostic exact returns force Full instead |
| scheduler timing | current acceptance | new error drives next decision | `end_nfe` writes; next `begin_nfe` reads |
| final layer | final Transformer block/layer | hard-coded block 27 | resolved as `len(blocks)-1` (also 27 for XL/2, but not hard-coded) |
| threshold | base/decay schedule | same schedule plus floor 0.01; first progress `1/N` | same on monotonic Heun NFE `q` |
| interval | not the adaptive rule | ineffective/inconsistently exposed | ignored/null in `speca`; fixed-only ablation |

## Metrics are not interchangeable

Main released metric:

```text
mean(abs(pred-exact)/(abs(exact)+1e-10))
```

Paper-style global relative L2 is a tensor norm ratio. Released `relative_l2` is instead an RMS of elementwise ratios. With `pred=[3,1]`, `exact=[1,2]`, released anchors are L1 1.5, L2 1.581138849, relative-L1 1.25, relative-L2 1.457737923, cosine error 0.292893291, whereas global relative-L2 is 1.0.

## Claim boundary

- `verification_fail_at_current_nfe` does not mean current-step rejection; only a subsequent `next_nfe_forced_full_due_previous_failure` may occur.
- The local verifier measures a complete block discrepancy conditional on a speculative prefix, not a full exact trajectory discrepancy.
- Exact verifier features do not repair output or become Taylor anchors.
- PixelGen's monotonic NFE coordinate and combined `2B` state are required sampler/framework adaptations, not paper rollback semantics.
- Results must not claim exact global verification, current recovery, ragged per-sample scheduling, or paper-equivalent sequential acceptance.

`paper_semantics_experimental` is reserved for a future explicit ablation. It is not the main mode, not selected by provided configs, and is not silently implemented.
