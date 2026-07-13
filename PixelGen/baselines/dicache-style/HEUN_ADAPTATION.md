# Exact-Heun adaptation

PixelGen’s `exact_henu` spelling is retained for configuration compatibility. A macro step is not an NFE. With `N` macro steps and exact Heun, the actual sequence is predictor plus corrector for the first `N-1` steps and one `final_euler` predictor at the last step:

```text
total_nfe = 2*N - 1
PixelGen combined model forwards = total_nfe
```

For 50 steps this is 99 NFEs and 99 combined 2B forwards. Counts are derived by `expected_nfe_count`/`expected_network_forward_count`; business logic does not hard-code 99. Predictor and corrector are separate probe observations even when continuous time repeats. The trace records macro-step index, NFE index, stage, `t`, and `t_next`.

Each NFE concatenates `x` as `[unconditional, conditional]`, concatenates conditioning in the same order, and performs exactly one model call. The upstream corrector’s CFG interval test uses `t_cur` even though the model is evaluated at `t_hat`; this behavior is preserved. The configured interval is open-low/closed-high.

Warmup and last Full use the combined stream’s 99-call plan, not 50 macro steps. The sampler begins one trajectory for every prediction batch, opens/closes one NFE around every model evaluation, verifies expected counts, ends the trajectory, and resets all tensor state in `finally` on failure. No state crosses batches.

GPU validation still required: upstream versus adapted exact Full parity, repeated-time traces, 50-step count instrumentation, and numerical parity of predictor/corrector outputs.

