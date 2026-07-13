# Online Probe Profiling semantics

The one PixelGen stream, `combined_cfg`, stores the previous body input, previous image-only probe feature, accumulated scalar error, and exact anchor window. On an eligible call:

```text
delta_x = mean(abs(body - previous_body)) / mean(abs(previous_body))
delta_y = mean(abs(probe - previous_probe)) / mean(abs(previous_probe))
error   = delta_y                         # main profile
accumulated_candidate = accumulated + error
Reuse iff accumulated_candidate < rel_l1_thresh
```

`delta_minus=abs(delta_y-delta_x)` is implemented for parity/ablation. Main `official_no_epsilon` does not alter denominators. `stable_eps_ablation` adds the explicit configured epsilon and must be labeled as an ablation. Reductions are over the complete combined 2B tensor, so each value is one batch-global scalar.

Equality is Full. Reuse preserves the candidate accumulator; Full resets it. Both eligible Reuse and eligible resumed Full update previous body/probe to the current observation. Direct Full captures its probe while executing the body once and also updates the previous observations and an exact anchor pair.

FLUX inclusive warmup is exact: `call_index <= int(ret_ratio * total_stream_calls)`. Thus `N=30,r=.2` gives 7 direct Full calls and `N=99,r=.2` gives 20. Index 0 remains Full at `r=0`; all calls are warmup Full at `r=1`. The last actual combined call is also Full. Missing state, missing anchors, diagnostic returns, incompatible context, and selected nonfinite safety handling force Full.

`official_compare` keeps PyTorch comparison behavior: Inf normally makes `<` false and NaN also compares false. It never silently maps NaN to zero. `force_full_reset_and_log` is a separately recorded minimum safety adaptation.

Summary `probe_count` means eligible gate probes only. A direct Full captures the depth-1 prefix feature during its single exact body execution, but that in-line capture is not counted as a separate probe and has no gate decision.
