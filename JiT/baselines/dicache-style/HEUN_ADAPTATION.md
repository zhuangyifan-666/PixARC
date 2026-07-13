# Exact-Heun adaptation

The adapter preserves the vendored JiT sampler rather than treating a displayed timestep as a cache key.

For `N` macro steps:

```text
for macro steps 0 .. N-2:
    predictor evaluation at t
    provisional Euler state
    corrector evaluation at t_next
final macro step:
    one Euler evaluation
```

This is `2N-1` NFEs. Each JiT NFE evaluates conditional then unconditional, so the network-forward count is `2*(2N-1)`. At 50 steps the exact invariants are 99 NFEs, 99 calls in each CFG stream, and 198 network forwards.

Predictor and corrector are separate DiCache observations even when a continuous time value repeats across neighboring solver stages. Every NFE records macro-step index, solver stage, current `t`, and `t_next`. Warmup and forced-last logic use per-stream call indices over all 99 calls.

One generation opens exactly one trajectory before the first NFE and closes it only after all counts validate. A `finally` path resets all stream observations, accumulators, anchors, indices, and trace tensors on failure. Explicit input noise allows Full and DiCache to replay the same sample-specific CPU Gaussian tensor.
