# Heun adaptation for JiT

## Why continuous time is not the Taylor coordinate

Released SpeCa-DiT makes one model call per DDIM timestep. JiT uses a Heun solver: a macro step normally evaluates a predictor at `t`, evaluates a corrector at `t_next`, and the next macro step can evaluate a new predictor at the same continuous `t_next` but with a different solver state. Using continuous `t` would therefore create duplicate coordinates (`delta q = 0`) for distinct network evaluations and invalidate finite differences.

The main protocol uses `coordinate_mode=official_nfe_index`:

```text
nfe_index = 0, ..., total_nfe - 1
q         = total_nfe - 1 - nfe_index
```

`q` is strictly decreasing, preserves the sign of finite-difference gaps, and assigns one SpeCa action to each expensive model evaluation. Continuous time remains trace metadata only.

## JiT call structure

For `num_steps=S`, the local helpers derive counts from the sampler rather than embedding constants:

```text
total_nfe = 2*S - 1              # exact Heun with final Euler evaluation
per-stream forwards = total_nfe
JiT.forward calls = 2*total_nfe  # separate conditional/unconditional streams
```

At 50 steps this means 99 shared decisions, 99 conditional forwards, 99 unconditional forwards, and 198 `JiT.forward` calls. Both streams receive the same `(nfe_index, q, action, check, threshold)` and each updates only its own factors. The NFE advances once after both stream payloads have been combined.

Stages are logged as `predictor`, `corrector`, and `final_euler`, together with `macro_step_index`, `continuous_t`, and `t_next`. Repeated continuous time is expected; repeated `q` is an error.

## Threshold progress

The released formula is applied on the adapted discrete coordinate:

```text
progress  = (total_nfe - q) / total_nfe
threshold = max(base_threshold * decay_rate**progress, threshold_floor)
```

For the first NFE, `q=total_nfe-1`, so progress begins at `1/total_nfe`, matching the released code rather than being reset to zero.

## Scope of the adaptation

This mapping is necessary sampler plumbing, not a new rejection algorithm. It does not change Heun states, predictor/corrector equations, CFG, checkpoint, output head, SpeCa scheduler branches, current-output no-rollback semantics, or the Taylor polynomial. No action is made once per macro step, and no `99`/`198` literal is required by runtime logic.

CPU sequence tests cover monotonic `q`, stage order, repeated continuous time without duplicate `q`, 99/198 counts at 50 steps, and arbitrary step counts. A real upstream-sampler instrumentation run remains a deferred GPU check.

