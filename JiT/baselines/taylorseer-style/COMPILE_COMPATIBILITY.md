# JiT compile and latency compatibility

JiT's upstream block and final-layer forwards are decorated with
`torch.compile`. The Taylor adapter calls attention and MLP modules separately,
so it does not traverse the same compiled block-forward boundary. Comparing a
compiled upstream Full against eager TaylorSeer would confound the algorithm
with compilation and is prohibited.

## Declared modes

- `matched_eager`: correctness and primary fallback. Full and TaylorSeer must
  use equivalent eager regions. Any inherited compiled wrapper that remains
  active must be recorded and matched before using the result.
- `upstream`: preserves upstream compilation and is valid for reporting
  `upstream_full` separately. It is not the TaylorSeer speedup denominator
  unless an exactly equivalent candidate region is proven.
- `blockwise`: scheduler/history remain outside compiled graphs while pure
  exact attention/MLP and final-head kernels may be compiled. Full and
  TaylorSeer must use identical blockwise settings.

Dynamic action selection, Python dataclasses, factor-list growth, tensor
shape checks, and trace collection can cause graph breaks or recompilation.
The scheduler and trajectory state must remain outside compiled regions.

## Fair speedup

The primary definition is:

```text
speedup = median_instrumented_matched_full_latency
          / median_taylorseer_latency
```

Both measurements must have the same GPU, checkpoint/EMA, manifest batch,
initial noise, dtype, sampler/steps/CFG, compile mode, warmup, batch size, and
timing boundary. Report upstream compiled Full as a separate number; never
divide unlike modes.

Timing begins after labels and explicit noise are resident on the GPU and
Taylor state is reset. It includes sampler, both CFG branches, embeddings,
AdaLN/gates, exact branches, forecasts, history update/read, scheduler, final
head, unpatchify, and cleanup. It ends after the final image tensor and upstream
equivalent clamp/conversion, before CPU copy/PNG. Checkpoint load, first
compile, dataloader, encoding, and metrics are excluded.

Use CUDA events with synchronization, at least 10 warmup and 30 measured
batches. Report median/mean/std/p90/p95/p99 ms per image, compile time, graph
breaks/recompiles, batch/effective CFG batch, Full/Taylor counts, forecast and
history time, cache bytes/count, and peak allocated/reserved memory.

## Deferred validation gate

No compile or GPU latency test was run during implementation. Before a result
is reportable, test output parity for `upstream_full`, `instrumented_full`, and
TaylorSeer interval 1 in every claimed mode; inspect graph breaks; then run the
same benchmark factory for matched Full and TaylorSeer. The current four-GPU
50K wall clock is not an automatic single-GPU speedup denominator.

