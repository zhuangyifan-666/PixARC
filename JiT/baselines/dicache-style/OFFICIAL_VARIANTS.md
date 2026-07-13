# Released DiCache variants

The released backends are not one interchangeable algorithm. The JiT main profile deliberately follows the FLUX image-state behavior; WAN and Hunyuan are references, not executable JiT defaults.

| Dimension | FLUX release | WAN2.1 release | HunyuanVideo release | Paper / Algorithm 1 | JiT main profile |
|---|---|---|---|---|---|
| Probe depth | configurable; demo 1 | configured 1 | configured 1 | shallow prefix, implementation commonly 1; not a universal backend constant | fixed 1 |
| Error choice | `delta_y`; optional `delta_minus` | `delta_y` (`delta_x` computed but unused) | `delta_y` | `delta_y` in Eq. 5; no `delta_minus` | `delta_y` |
| Error denominator | mean abs of previous image probe; `delta_x` uses previous input | mean abs of previous probe in the same `cnt%2` slot | mean abs of previous image probe | previous probe L1 magnitude | previous image probe mean abs |
| Epsilon / numeric mode | no epsilon | no epsilon | no epsilon | no epsilon/non-finite prescription | `official_no_epsilon`; stable mode only ablation |
| Demo threshold | `0.4` for `delta_y` | CLI/demo `0.08` | demo `0.1` | no universal model threshold | deliberately null until JiT validation |
| Threshold comparison | strict `<` Reuse | strict `<` Reuse | `<=` Reuse | Algorithm 1 writes `<=` | strict `<` Reuse |
| `ret_ratio` | demo `0.2` | demo/default `0.2` | demo `0.2` | not prescribed | fixed `0.2` |
| First/warmup behavior | `cnt <= int(ret*N)` direct Full (inclusive) | probe eligible at `cnt >= int(ret*N)` | probe eligible at `cnt >= int(ret*N)` | first call Full; no `ret_ratio` schedule | FLUX inclusive, independently per stream |
| Last-call behavior | forced Full | no forced last Full | no forced last Full | no forced-last rule | forced Full per stream |
| CFG layout | FLUX pipeline image state; release demo holds one transformer state | cond/uncond are separate calls using alternating two slots | CFG-active input is one combined `[uncond,cond]` 2B batch | no concrete CFG state-layout rule | cond then uncond calls with explicit isolated streams |
| Batch aggregation | one scalar mean over the complete current tensor | one scalar mean per alternating stream tensor | one scalar mean over combined batch | described as sample-specific, but does not specify batched reduction | one batch-global scalar within each B-sized CFG stream; main B=1 |
| Gamma formula | probe-trajectory ratio and first-order residual extrapolation | same per slot | same global history | Eqs. 7–11 use two exact anchors | same per stream |
| Gamma minimum | 1.0 | 1.0 | 1.0 | clipping bounds not mandated | 1.0 |
| Gamma maximum | 1.5 | 2.0 | 1.5 | no universal upper bound | 1.5 |
| Residual window | Python lists grow, but only last two pairs are read | per-slot windows can retain a small extra stale entry; last two drive DCTA | bounded/stale short history; latest two drive formula | two exact anchor states in the first-order derivation | `deque(maxlen=2)` exact synchronized pairs |
| Single-anchor behavior | latest/previous full residual | latest residual in the slot | latest residual | first-order unavailable; latest residual is the natural zero-order case | explicit counted latest-residual fallback |
| Reset behavior | official reset clears counter/windows but is not a full lifecycle contract | resets alternating states/windows | partial cleanup | no systems reset prescription | complete trajectory/failure reset |
| Cached boundary | image hidden state across transformer body | latent `x` residual across WAN blocks | image/latent stream across double blocks | generic denoiser body residual | JiT image blocks from `x_embedder+pos` to pre-head image suffix |
| Final head | norm/projection after cached body are recomputed | post-block output path remains current | post-block/final output path remains current | not specified as an integration boundary | final norm/AdaLN/projection/unpatchify always fresh |
| Monkey patch | global transformer class replacement and class attributes | model class method/state patching | transformer class method/state patching | not an algorithm requirement | none; subclass plus instance runtime |
| State storage | mutable transformer/class attributes | mutable class attributes with two slots | mutable transformer/class attributes | abstract per-sample history | per-model, per-trajectory, per-stream objects outside `state_dict` |

For `N=30`, `ret_ratio=.2`, FLUX direct Full includes indices 0 through 6. For a 99-call JiT stream it includes indices 0 through 19. `ret_ratio=0` still makes index 0 Full, and `ret_ratio=1` makes every call Full. Equality at the accumulated threshold selects Full.

Released FLUX stores an unbounded Python list but only reads the latest two anchors. The port’s `deque(maxlen=2)` is therefore behavior-equivalent for first-order DCTA while making persistent memory bounded. It does not import or copy the official global monkey-patching machinery.

The main config is named `flux_image_released` and labeled `released_code_faithful_image_profile`. That label means the audited scheduling/formulas were preserved at the JiT image-token boundary. It does not mean an official JiT integration exists.
