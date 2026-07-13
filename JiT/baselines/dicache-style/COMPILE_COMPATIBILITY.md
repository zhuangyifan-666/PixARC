# Compile compatibility

JiT DiCache exposes three compiler modes and five measured roles:

| Process mode | Full role | Candidate role | Compilation boundary |
|---|---|---|---|
| `upstream` | `upstream_full` | none | whole upstream `JiT` model |
| `matched_eager` | `instrumented_full` | `dicache` | no Dynamo wrapper |
| `blockwise` | `instrumented_full` | `dicache` | transformer blocks and final layer |

The blockwise factory first removes the decorator-installed upstream block
wrappers, then applies explicit `nn.Module.compile` wrappers. This prevents a
nested compiler configuration. The Python trajectory lifecycle, gate, DCTA,
and dynamic suffix scheduling remain outside the compiled blocks. Whole-model
compilation is restricted to `upstream_full`; it is never used for a DiCache
candidate.

## Executable matrix and hard gates

[`scripts/run_compile_matrix.py`](scripts/run_compile_matrix.py) launches the
three modes in fresh subprocesses, so their Dynamo caches and CUDA allocators
cannot contaminate one another. The eager and blockwise processes each measure
Full followed by DiCache, giving five role measurements in total. Every role
records:

- steady-state latency with `torch.cuda.Event` and end-event synchronization;
- first-execution wall time and CUDA-event time;
- separate first-execution and steady-state CUDA peak allocated/reserved bytes;
- Dynamo graph-break, guard-failure, and guard/recompile counter deltas, with
  `TORCH_LOGS=graph_breaks,recompiles` stderr retained; and
- exact-Heun NFE and dual-CFG network-forward counts derived from the materialized
  sampler configuration.

The worker checks raw floating sampler outputs before uint8 conversion, finite
values, trajectory call partitions, Full-only behavior where required, runtime
closure, and zero live cache tensors after every trajectory. Matrix assembly
also requires identical checkpoint, sampler, precision, inputs, selected
DiCache configuration, and derived counts across modes.

Correctness is fail-closed for three raw-tensor comparisons:

1. upstream whole-model Full versus matched-eager instrumented Full;
2. matched-eager versus blockwise instrumented Full; and
3. matched-eager versus blockwise DiCache.

Shape, dtype, and finiteness must match, followed by `torch.allclose` under the
explicit `--atol`/`--rtol` values. Both tolerances default to zero. A worker
failure, non-finite value, count/lifecycle mismatch, configuration mismatch, or
correctness failure writes a top-level `"passed": false` artifact and exits
non-zero. The release evidence is bindable only when top-level `passed` is
exactly `true`.

## Deferred command

Use one resolved DiCache runner JSON (checkpoint, selected threshold/gamma
policy, sample IDs, seeds, and class IDs). The shell wrapper refuses to start
unless exactly one allocated GPU is visible, the explicit authorization flag
is set, `nvidia-smi` process telemetry is empty, utilization is at most 5%, and
used memory is at most 1024 MiB. It never signals another process.

```bash
export CUDA_VISIBLE_DEVICES=<one-allocated-idle-gpu>
export DICACHE_GPU_TESTS_ALLOWED=1

bash JiT/baselines/dicache-style/scripts/run_compile_matrix.sh \
  --runner-config "$RESOLVED_JIT_RUNNER_JSON" \
  --output-dir "$OUTPUT_ROOT/jit_compile_matrix_rows" \
  --output-json "$OUTPUT_ROOT/jit_compile_matrix.json" \
  --warmup-batches 10 \
  --measured-batches 30 \
  --atol 0 \
  --rtol 0
```

The output directory and matrix JSON must not already exist. Keep the three
worker JSON files, five `.pt` raw-output tensors, stdout logs, and Dynamo logs
beside the top-level matrix artifact. No GPU result is claimed in this
repository; the command is a deferred path and has not been executed here.

## Interpretation limits

The primary speedup comparison remains matched-eager `instrumented_full` versus
matched-eager `dicache`. The upstream whole-model and blockwise rows diagnose
compiler compatibility; their different graph boundaries are not a fair
headline speedup denominator.

Total latency is CUDA-event based through the final raw image tensor. Runtime
component timers use host `perf_counter` around asynchronous work; a scalar
synchronization can absorb earlier queued kernels, so those diagnostics cannot
replace the CUDA-event total. CPU tests establish routing and artifact gates,
not GPU correctness, speed, memory, graph-break, or recompile results.
