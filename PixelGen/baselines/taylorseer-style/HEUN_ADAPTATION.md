# Heun adaptation for PixelGen

## Why continuous time is not the Taylor coordinate

The official DiT implementation advances one model call per discrete sampler
step. PixelGen's exact Heun loop makes a predictor at `t_i`, a corrector on
the provisional state at `t_(i+1)`, and then the next predictor on the
corrected state at the same `t_(i+1)`. The two inputs at that repeated time are
not the same tensor. Using continuous `t` as the finite-difference coordinate
could therefore produce a zero denominator even though the feature changed.

The primary port uses `coordinate_mode=official_nfe_index`. For
`nfe_index=0..total_nfe-1`:

```text
q = total_nfe - 1 - nfe_index
```

Thus q is strictly decreasing, every expensive model evaluation has one
coordinate, signed exact-anchor gaps are retained, and shifted continuous
time is trace metadata only. This is a necessary solver mapping, not a claim
that Heun is identical to the original DDIM loop.

## PixelGen call sequence

The upstream exact-Heun sampler preserves this order:

```text
cfg_x         = cat([x, x])
cfg_condition = cat([uncondition, condition])
begin_nfe(nfe_index, q, macro_step, stage, t, t_next)
  one combined 2B model forward -- stream "combined_cfg"
  guidance split/combine
end_nfe(seen_streams={"combined_cfg"})
```

It must not be split into conditional and unconditional forwards. For N
sampler steps with `exact_henu=true`, there are N predictors and N-1
correctors:

```text
total_nfe = 2*N - 1
combined network forwards = total_nfe
```

At N=50 this is 99 decisions and 99 combined 2B forwards. q runs from 98
through 0. Solver stages are recorded as `predictor`, `corrector`, and
`final_euler` for the final predictor/integration call.

For the faithful fixed scheduler at 99 NFE:

| interval | Full NFE | Taylor NFE |
|---:|---:|---:|
| 1 | 99 | 0 |
| 2 | 50 | 49 |
| 3 | 34 | 65 |
| 4 | 26 | 73 |
| 5 | 21 | 78 |

These counts follow `first_enhance=2`, do not force the last call Full, and
must also be checked by the CPU scheduler tests.

## State and lifecycle implications

- All blocks and both modules use the single NFE action.
- Taylor calls never change the combined stream's anchor or factors.
- The cache batch dimension is 2B and preserves unconditional-first ordering.
- Every Lightning prediction batch begins a new trajectory and resets in
  `finally`.
- A last short batch is a new trajectory with new shapes; it cannot reuse the
  previous batch's history.
- Diagnostic `return_layer` or `return_last` requires a forced Full decision,
  recorded as `forced_full_reason=diagnostic_return`.
- Expected call counts are derived from sampler settings, not hard-coded.

## Heun-specific alternative

Maintaining separate predictor and corrector histories could reduce mixing of
solver stages, but it is a **HEUN-SPECIFIC ABLATION, NOT ORIGINAL
TAYLORSEER**. The shipped faithful scheduler implements only
`official_nfe_index`; `stage_separated` is deferred and must never silently
replace the primary result.

