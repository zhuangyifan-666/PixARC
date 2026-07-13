# Released variants and paper behavior

The table records the audited local DiCache revision `fdbe20b669c9174bbed5ec994de073fd881c8010`. “Paper” means what is specified algorithmically, not an inferred implementation default.

| Dimension | FLUX | WAN 2.1 | HunyuanVideo | Paper |
|---|---|---|---|---|
| probe depth | 1 default | 1 default | 1 | shallow prefix; implementation-specific |
| error choice | `delta_y` default; `delta_minus` supported | `delta_y`; computes `delta_x` but gate does not use it | `delta_y` | Eq. 5 probe-output relative change |
| denominator | mean absolute previous value, no epsilon | same, no epsilon | same, no epsilon | L1 ratio; no operational epsilon policy |
| threshold comparison | reuse iff accumulated `<` threshold | reuse iff accumulated `<` threshold | reuse iff accumulated `<=` threshold | Algorithm 1 uses `<=` reuse |
| equality action | Full | Full | Reuse | Reuse |
| `ret_ratio` | direct Full for `cnt <= int(r*N)` | probe starts at `cnt >= int(r*N)` | probe starts at `cnt >= floor(r*N)` | not specified |
| first Full | implied by inclusive warmup | implied by warmup | implied by warmup | explicitly Full |
| last Full | forced | not forced | not forced | not specified |
| CFG state | one FLUX image stream; text remains separate | two alternating slots, cond then uncond | one combined batch-global state | not specified |
| independent CFG actions | not applicable to one call | yes | no, one combined decision | not specified |
| gamma range | `[1,1.5]` | `[1,2]` | `[1,1.5]` | clipping/bounds not fixed |
| residual window | appends without a bound; only last two read | grows to three because of `len<=2`; only last two read | similarly three effective entries | two exact points in Eqs. 7–11 |
| reset | counter/windows reset; several previous fields persist | full dual-slot cleanup | partial cleanup | not specified |
| batch aggregation | global mean over input tensor | global mean within the active CFG call | global mean over combined batch | “sample-specific” motivation, aggregation not operationally fixed |
| cache boundary | image transformer body; text state separate | transformer body residual | transformer body residual | costly denoising body |
| final head | fresh norm/projection | fresh head | fresh head | not separately prescribed |
| monkey patch | yes | patched/copied model path | patched pipeline/model path | no engineering requirement |
| runtime state | class attributes | explicit two-element arrays/attributes | global/class attributes | abstract algorithm state |

## FLUX details

After image embedding, the body input is the image hidden-state tensor. The first `probe_depth` transformer blocks run exactly once. `delta_x = mean(abs(x_t-x_prev))/mean(abs(x_prev))`; `delta_y` is the analogous probe-state ratio; `delta_minus=abs(delta_y-delta_x)`. The accumulator begins as Python `0` and becomes a scalar tensor. Equality triggers Full. For 30 calls and `ret_ratio=.2`, indices 0–6 are direct Full. Direct warmup and final Full clear the accumulator. An eligible Full resumes at the suffix. Full and probe residual anchors come from the same exact call. Reuse updates previous input/probe but does not write anchors. The final normalization/projection is always recomputed.

DCTA uses the last two exact pairs: `gamma=clamp(mean(abs(Pcur-Pold))/mean(abs(Pnew-Pold)),1,1.5)` and `Rhat=Rold+gamma*(Rnew-Rold)`. With one anchor it reuses the latest exact full residual. There is no epsilon. The shipped threshold `0.4` is a demo default, not transferable evidence for PixelGen.

## WAN 2.1 details

Each diffusion step calls conditional then unconditional model forwards. `cnt % 2` maps them to two independently accumulated state slots and permits action disagreement. `num_steps=sampling_steps*2` counts total model forwards. The warmup boundary differs from FLUX, there is no forced final Full, and gamma is clipped to `[1,2]`. The CLI default is `0.08`; the reported WAN experiment uses `0.2`. End-of-trajectory cleanup is more complete than FLUX.

## HunyuanVideo details

The pipeline concatenates negative/unconditional then positive/conditional batches and performs one 2B forward when CFG is active. One batch-global state is used. Probe depth 1, threshold `0.1`, `ret_ratio=.2`, equality Reuse, no forced final Full, no epsilon, gamma `[1,1.5]`, and last-two effective anchors were observed. The demo may run guidance-scale 1 with embedded guidance, so combined CFG is conditional on the invoked path.

## Main port choice

PixelGen is an image generator, so FLUX is the closest released image behavior. WAN is not copied because PixelGen does not perform separate CFG forwards. Hunyuan confirms combined-batch state is a released pattern, but its non-strict boundary and missing last Full conflict with the selected FLUX image profile.

