# DCTA semantics

Let the two latest exact synchronized anchors be `(R_old,P_old)` and `(R_new,P_new)`, where `R` is the full-body residual and `P` the probe residual. For the current call:

```text
P_cur     = current_probe_feature - current_body_input
gamma_raw = mean(abs(P_cur - P_old)) / mean(abs(P_new - P_old))
gamma     = clamp(gamma_raw, 1.0, 1.5)
R_hat     = R_old + gamma * (R_new - R_old)
output    = current_body_input + R_hat
```

Gamma is one batch-global scalar per JiT CFG stream. With only one exact anchor, Reuse falls back to that latest residual; this is counted as zero-order fallback. `dicache_zero_order` deliberately disables first-order alignment and is an ablation, not the main method.

Only exact Full calls append anchors. Reuse cannot write an estimated residual or probe pair. The two-anchor deque is equivalent to released FLUX’s accessed history. Anchor shape/device must match the current body. FP32 cache arithmetic is allowed, but `approximated_body_output` is cast back to the body dtype before the current final head.

`official_no_epsilon` reproduces the released denominator. `stable_eps_ablation` is non-main. Supported gamma non-finite policies are official propagation, latest-residual fallback, and force-Full. The candidate template leaves this choice null until preregistration; no final policy or result is claimed.

The runtime separately times DCTA scalar `.item()` synchronization (finite and clipping decisions) and subtracts it from diagnostic DCTA host elapsed. These host intervals are approximate on asynchronous CUDA and never replace total CUDA-event latency.
