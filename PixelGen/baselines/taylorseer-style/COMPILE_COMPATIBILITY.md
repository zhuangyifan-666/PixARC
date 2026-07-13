# PixelGen compile and latency compatibility

Upstream PixelGen Lightning deep-copies the denoiser, then calls `compile()` on
both original and EMA instances in `configure_model`. The Taylor adapter adds
Python scheduling/history around per-branch attention and MLP work. Comparing
an upstream-compiled Full model with an eager TaylorSeer model would confound
the algorithm with compilation and is prohibited.

## Declared modes

- `matched_eager`: correctness and primary fallback. Original denoiser and EMA
  must have equivalent eager regions, as must matched Full and TaylorSeer.
- `upstream`: preserves upstream compilation and is valid for reporting
  `upstream_full` separately. It is not the TaylorSeer speedup denominator
  unless an equivalent candidate region is proven.
- `blockwise`: scheduler/history remain outside compiled graphs while pure
  exact attention/MLP, forecast, and final-head kernels may be evaluated as
  separately compiled regions. Both methods must share the same setting.

Dynamic Full/Taylor actions, diagnostic-return forcing, factor-list growth,
runtime deepcopy state, Python trace records, and the combined CFG lifecycle
can cause graph breaks or recompilation. The scheduler and history mutations
must remain outside compiled regions. `denoiser` and `ema_denoiser` must never
share runtime tensor state after deepcopy.

## Fair speedup

The primary definition is:

```text
speedup = median_instrumented_matched_full_latency
          / median_taylorseer_latency
```

Both measurements must have the same GPU, checkpoint/EMA, manifest batch,
initial noise, BF16 policy, exact-Heun sampler, guidance/timeshift, compile
mode, warmup, batch size, and timing boundary. Report upstream compiled Full
separately; never divide unlike modes.

Timing begins after conditions and explicit noise are on GPU and state is
reset. It includes all 2B forwards, CFG, sampler, embeddings, AdaLN/gates,
exact branches, forecasts, history, scheduler, final head, unpatchify, and
cleanup. It ends after upstream-equivalent decoded/image conversion and before
CPU copy/PNG. Checkpoint load, first compile, dataloader, PNG, and metrics are
excluded.

Use CUDA events with synchronization, at least 10 warmup and 30 measured
batches. Report median/mean/std/p90/p95/p99 ms per image, compile time, graph
breaks/recompiles, batch/effective 2B batch, Full/Taylor counts, forecast and
history time, cache bytes/count, and peak allocated/reserved memory.

## Deferred validation gate

No compile or GPU latency test was run during implementation. Before a result
is reportable, test output parity for `upstream_full`, `instrumented_full`, and
TaylorSeer interval 1 in every claimed mode; inspect graph breaks; then run the
same benchmark factory for matched Full and TaylorSeer. The current four-GPU
50K wall clock is not an automatic single-GPU speedup denominator.

