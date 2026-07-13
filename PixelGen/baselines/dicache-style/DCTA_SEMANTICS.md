# Dynamic Cache Trajectory Alignment

The runtime stores at most two synchronized exact `ResidualAnchor` objects. Each contains the full-body residual and shallow-probe residual from the same exact call plus NFE/call/stage metadata. A bounded window is equivalent to released FLUX’s unbounded append because only the last two elements are read.

With one anchor, Reuse is zero-order: `Rhat=Rlatest`. With two anchors:

```text
Pcur      = current_probe_feature - current_body_input
gamma_raw = mean(abs(Pcur - Pold)) / mean(abs(Pnew - Pold))
gamma     = clamp(gamma_raw, 1.0, 1.5)
Rhat      = Rold + gamma * (Rnew - Rold)
body_hat  = current_body_input + Rhat
```

Gamma is one scalar over all tokens and both CFG halves. It is not per token, channel, sample, or branch. Main numeric mode has no denominator epsilon. `dicache_zero_order` deliberately disables DCTA and cannot be reported as full DiCache. The main candidate template leaves `gamma_nonfinite_policy` null and therefore cannot run until a post-smoke choice is materialized.

Only exact Full calls append anchors. Reuse updates adjacent-call previous input/probe but never writes either exact anchor series. If gamma is nonfinite, `official_propagate`, `latest_residual_fallback`, or `force_full` may be selected. The choice must be frozen after smoke/1K validation; no 50K result exists yet. CPU tests cover clipping, fallback, nonfinite policies, synchronized windows, exact-only updates, and direct-formula parity.
