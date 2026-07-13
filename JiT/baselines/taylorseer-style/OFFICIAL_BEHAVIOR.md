# Audited TaylorSeer-DiT behavior

This document records executable behavior from the local TaylorSeer clone at
commit `704ee98c74f7f04da443daa3c0aa2cc7803d86e3`. It is the semantic contract
for this **unofficial TaylorSeer-style port for JiT**. The paper and
Cache4Diffusion are secondary references.

## Source path and action boundary

The relevant files are:

- `baselines/TaylorSeer/TaylorSeer-DiT/models.py`;
- `TaylorSeer-DiT/taylor_utils/__init__.py`;
- `TaylorSeer-DiT/cache_functions/cal_type.py`;
- `TaylorSeer-DiT/cache_functions/cache_init.py`;
- `TaylorSeer-DiT/sample.py` and `sample_ddp.py`.

`cal_type` runs once before the block loop. Consequently one model call has
one global `full` or `Taylor` action. Every block and both its attention and
MLP branches use that action. The action is not chosen per layer or sample.

## What is forecast

For each `(layer, module)` where `module` is `attn` or `mlp`, the official
implementation stores the branch output before multiplying the current gate:

```text
attention: block.attn(modulate(block.norm1(x), shift_msa, scale_msa))
MLP:       block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))
```

The attention return already includes its output projection/dropout, and the
MLP return is the complete MLP output. Attention and MLP histories are
separate. A Taylor call still evaluates timestep/class embeddings and each
block's AdaLN modulation, then multiplies the forecast by the **current** gate.
It skips norm1/attention/norm2/MLP. The final normalization, final AdaLN,
output projection, and unpatchify are always exact.

## Finite-difference history

At exact coordinate `q_j`, with previous exact coordinate `q_(j-1)`, the
signed gap is `delta_q = q_j - q_(j-1)`. The official recursive update is:

```text
new D0     = f(q_j)
new D(k+1) = (new Dk - old Dk) / delta_q
```

Old factors must be retained until the recursion completes. `max_order=K`
means highest order K and at most K+1 tensors. The first exact call provides
only D0; later exact anchors mature one additional order until K. Only exact
calls update factors and the anchor. A forecast is read-only:

```text
offset = q - latest_exact_q
f_hat(q) = sum(Dk * offset**k / factorial(k), k=0..available_order)
```

The denominator and forecast offset keep their signs. There is no polynomial
fit, clipping, damping, normalization, quantization, CPU offload, or forecast
write-back. `max_order=0` is exact zero-order reuse.

## Faithful fixed schedule

The audited default is `first_enhance=2`: the first two calls are Full and
each resets `cache_counter=0`. Thereafter:

```text
if cache_counter == interval - 1:
    Full; cache_counter = 0
else:
    Taylor; cache_counter += 1
```

`interval=1` is all Full. `last_steps` is computed in `cal_type.py` but never
used, so the faithful default does **not** force a final Full. A separate
`force_last_full=true` option is an explicitly labeled safety ablation.

For the official 50-call loop, local source inspection gives:

| interval | Full | Taylor | final action |
|---:|---:|---:|---|
| 1 | 50 | 0 | Full |
| 2 | 26 | 24 | Full |
| 3 | 18 | 32 | Full |
| 4 | 14 | 36 | Full |
| 5 | 11 | 39 | Taylor |

The CPU schedule parity test imports the local official helper and is the
authoritative executable check; these counts are not a substitute for it.

## Hard-coded and dead storage in the source clone

The official initialization uses `activated_steps=[49]`, loops over 28
layers, and creates per-timestep dictionaries. Those are 50-step/DiT-XL
assumptions, not mathematical requirements. Tensor factors are actually
stored only under `cache[-1][layer][module]`; the per-step dictionaries remain
empty. This port removes the 49, 50, and depth-28 constants, allocates state
from the real model depth, and retains only the latest anchor's factors.
Behavioral parity is required even though dead dictionaries are omitted.

## JiT mapping

JiT-B/16 has 12 blocks. Each block has its own attention and MLP state; no
whole-body residual is forecast. Conditional and unconditional model calls
share one schedule decision and coordinate but use independent tensor
histories. The final head remains fresh in both streams. See
`HEUN_ADAPTATION.md` for the 99-NFE coordinate mapping.

## Distinction from TaylorSeer-Lite

Cache4Diffusion contains easier whole-block/final-output forecasting variants,
including TaylorSeer-Lite. They do not preserve the original DiT branch-level
boundary and are not used for the primary baseline. No Lite result may be
reported under the main TaylorSeer-style label.

