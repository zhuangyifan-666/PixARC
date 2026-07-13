# Heun adaptation for JiT

## Why continuous time is not the Taylor coordinate

The official DiT implementation advances one model call per discrete sampler
step. JiT's exact Heun loop makes a predictor at `t_i`, a corrector on the
provisional state at `t_(i+1)`, and then the next predictor on the corrected
state at the same `t_(i+1)`. The two inputs at that repeated time are not the
same tensor. Using continuous `t` as the finite-difference coordinate could
therefore produce a zero denominator even though the feature changed.

The primary port uses `coordinate_mode=official_nfe_index`. For
`nfe_index=0..total_nfe-1`:

```text
q = total_nfe - 1 - nfe_index
```

Thus q is strictly decreasing, every expensive model evaluation has one
coordinate, signed exact-anchor gaps are retained, and continuous time is
trace metadata only. This is a necessary solver mapping, not a claim that
Heun is identical to the original DDIM loop.

## JiT call sequence

One NFE contains both CFG model calls:

```text
begin_nfe(nfe_index, q, macro_step, stage, t, t_next)
  conditional forward   -- stream "cond", same action and q
  unconditional forward -- stream "uncond", same action and q
  CFG combination
end_nfe(seen_streams={"cond", "uncond"})
```

The scheduler advances after both streams, never between them. Histories are
independent because class conditioning and current gates differ. For N sampler
steps, upstream JiT executes N-1 predictor/corrector pairs followed by one
final Euler evaluation:

```text
total_nfe = 2*(N-1)+1
network_forwards = 2*total_nfe
```

At N=50 this is 99 decisions, 99 conditional forwards, 99 unconditional
forwards, and 198 `JiT.forward` calls. q runs from 98 through 0. Solver stages
are recorded as `predictor`, `corrector`, and `final_euler`.

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
- Taylor calls never change either stream's anchor or factors.
- Both stream calls must complete before `end_nfe`; missing or duplicate
  stream execution is an error.
- Every manifest batch begins a new trajectory and resets in `finally`.
- A last short batch is a new trajectory with new shapes; it cannot reuse the
  previous batch's history.
- Expected NFE and network-forward counts are computed from sampler settings,
  not hard-coded to 99 or 198.

## Heun-specific alternative

Maintaining separate predictor and corrector histories could reduce mixing of
solver stages, but it is a **HEUN-SPECIFIC ABLATION, NOT ORIGINAL
TAYLORSEER**. The shipped faithful scheduler implements only
`official_nfe_index`; `stage_separated` is deferred and must never silently
replace the primary result.

