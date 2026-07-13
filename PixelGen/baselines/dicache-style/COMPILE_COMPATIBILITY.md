# Compile compatibility

PixelGen DiCache exposes three compilation modes, but a release decision uses
five independently constructed rows:

| Matrix row | `dicache.mode` | `runtime.compile_mode` | What is compiled |
|---|---|---|---|
| `upstream_whole_model` | `upstream_full` | `upstream` | Upstream whole `JiT` call path |
| `matched_eager_full` | `instrumented_full` | `matched_eager` | No Dynamo wrapper; exact instrumented path |
| `matched_eager_dicache` | `dicache` | `matched_eager` | Same eager body plus DiCache scheduling |
| `blockwise_full` | `instrumented_full` | `blockwise` | Each transformer block and final layer |
| `blockwise_dicache` | `dicache` | `blockwise` | Same blockwise bodies plus DiCache scheduling |

The upstream row instantiates the original
`src.models.transformer.JiT.JiT` class and loads the same EMA weights; it does
not use the instrumented subclass with a disabled runtime.

The upstream `JiTBlock.forward` has a decorator-installed Dynamo wrapper.
`matched_eager` removes it.  The deferred benchmark factory also removes it
before applying explicit blockwise `nn.Module.compile`, so the blockwise rows
do not accidentally nest two compiler wrappers.  The Python NFE lifecycle,
gate, DCTA state, and dynamic suffix selection remain outside compiled blocks.
Whole-model compilation is restricted to `upstream_full`; it is never silently
used as a DiCache speedup denominator.

## Executable matrix and hard gates

[`scripts/run_compile_matrix.py`](scripts/run_compile_matrix.py) validates all
five materialized config surfaces before loading a model, then launches each
row through [`scripts/benchmark_single_gpu.sh`](scripts/benchmark_single_gpu.sh)
in a fresh process.  Process isolation prevents the upstream whole-model graph,
matched model, and blockwise model from occupying GPU memory simultaneously.

Every row records:

- steady-state latency using `torch.cuda.Event` and end-event synchronization;
- first-execution wall time as an upper bound that includes compilation;
- steady-state and first-execution CUDA peak allocated/reserved bytes;
- Dynamo counter deltas, graph-break count, guard-failure/recompile count, and
  retained `TORCH_LOGS=graph_breaks,recompiles` output;
- sampler-derived expected and observed network-forward counts; and
- byte identity of the decoded RGB uint8 tensor before CPU copy or PNG encoding.

The untimed validation trajectory inspects the raw sampler tensor and raw VAE
decode for finite values, verifies the derived forward count and runtime
lifecycle, and checks that cache tensors are released.  The final matrix is
fail-closed.  It requires byte-exact final outputs for:

1. upstream whole-model versus matched-eager instrumented Full;
2. matched-eager versus blockwise instrumented Full; and
3. matched-eager versus blockwise DiCache.

This is stricter than a floating tolerance at the final image boundary.  Any
row failure, non-finite raw tensor, count mismatch, lifecycle/cache failure, or
correctness mismatch produces a top-level `"passed": false` artifact and a
non-zero exit.  No benchmark result is claimed in this repository; the matrix
must be executed on an idle allocated GPU.

## Deferred command

First materialize five immutable configs with the indicated mode pairs and the
same checkpoint, sampler, precision, manifest, selected threshold, and gamma
policy.  Then, only after the allocated GPU is confirmed idle, run:

```bash
export CUDA_VISIBLE_DEVICES=<one-allocated-idle-gpu>
export DICACHE_GPU_TESTS_ALLOWED=1

python PixelGen/baselines/dicache-style/scripts/run_compile_matrix.py \
  --upstream-whole-model-config "$UPSTREAM_COMPILED_CONFIG" \
  --matched-eager-full-config "$EAGER_FULL_CONFIG" \
  --matched-eager-dicache-config "$EAGER_DICACHE_CONFIG" \
  --blockwise-full-config "$BLOCKWISE_FULL_CONFIG" \
  --blockwise-dicache-config "$BLOCKWISE_DICACHE_CONFIG" \
  --manifest "$SMOKE_MANIFEST" \
  --output-dir "$OUTPUT_ROOT/compile_matrix_rows" \
  --output-json "$OUTPUT_ROOT/compile_matrix.json" \
  --warmup-batches 10 \
  --measured-batches 30
```

The output directory and matrix JSON must not already exist.  Keep the row
JSON files and Dynamo logs alongside the top-level artifact; the latter is the
mechanically bindable release gate and is valid only when `passed` is exactly
`true`.  It records SHA-256 identities for every row report and retained log,
while each row binds its input config, checkpoint size, manifest, sample IDs,
seeds, labels, and sampler-derived call count.

## Interpretation limits

The primary speedup denominator remains the matched pair
`instrumented_full` versus `dicache` in `matched_eager`, using the existing
pair benchmark API.  The five-row matrix diagnoses compiler compatibility; it
does not make an unmatched upstream-compiled speedup a fair headline number.

Total latency is a CUDA-event measurement through the final image tensor.
Component timers (`probe`, gate, DCTA, suffix, cache I/O) remain host
`perf_counter` diagnostics around asynchronous GPU launches.  A scalar sync
can absorb earlier queued work, so component attribution must not replace the
CUDA-event total.  CPU structural tests cover config routing, artifact gates,
and exact fingerprint behavior, but cannot establish GPU compiler correctness,
latency, memory, graph-break, or recompile results.
