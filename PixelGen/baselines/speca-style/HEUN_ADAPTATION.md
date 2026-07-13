# Heun adaptation for PixelGen

## Why continuous time is not a Taylor coordinate

Released SpeCa-DiT associates one action with one DDIM network call. PixelGen's `HeunSamplerJiT` with `exact_henu=true` normally evaluates predictor and corrector calls. A corrector's continuous `t_next` can equal the next predictor's `t`, yet the solver states differ. Taylor finite differences on continuous `t` would see duplicate coordinates and `delta q=0` for distinct calls.

Main `coordinate_mode=official_nfe_index` defines:

```text
nfe_index = 0, ..., total_nfe - 1
q         = total_nfe - 1 - nfe_index
```

This is strictly monotonic and places one action on each expensive combined model evaluation. Continuous `t` is trace metadata only.

## PixelGen call structure

For `num_steps=S`, exact Heun derives `total_nfe=2*S-1`; the final interval is an Euler evaluation. PixelGen duplicates latent state and concatenates conditioning as `[unconditional, conditional]`, then performs one combined effective-`2B` forward per NFE. At 50 steps this yields 99 decisions and 99 combined forwards, not 198 separate calls.

Runtime logic derives the count and stages rather than hard-coding 99. It logs `macro_step_index`, `nfe_index`, `q`, `solver_stage` (`predictor`, `corrector`, `final_euler`), continuous `t`, and `t_next`. A repeated continuous time is expected; repeated `q` is invalid.

## Threshold progress

The released formula operates on adapted `q`:

```text
progress  = (total_nfe - q) / total_nfe
threshold = max(base_threshold * decay_rate**progress, threshold_floor)
```

The first progress is `1/total_nfe`, as in released control flow. One scheduler, combined Taylor state, and combined verification metric advance once per model forward.

## Scope

The mapping is necessary sampler adaptation, not a new method. It leaves PixelGen's Heun equations, timeshift, guidance interval, `[unconditional, conditional]` order, combined call, EMA, checkpoint, output head, no-rollback semantics, and Taylor polynomial unchanged. CPU sequence tests cover monotonic coordinates, stage order, repeated `t`, arbitrary step counts, and 99 combined calls at 50 steps; real sampler instrumentation is deferred to an authorized GPU run.

