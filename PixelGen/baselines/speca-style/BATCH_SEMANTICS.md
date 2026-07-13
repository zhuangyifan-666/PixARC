# PixelGen batch and CFG semantics

## Registered main protocol

```text
real generation batch per GPU process = 1
combined model batch                  = 2
combined order                        = [unconditional, conditional]
Taylor histories                      = 1 combined state
scheduler / error                     = 1 per real sample trajectory
gate_mode                             = batch_global
manifest batch grouping               = one real sample per group
```

The released metric produces one scalar for the full model input. Real batch 1 ensures that the scalar/action corresponds to one real sample while still including both required CFG halves. It avoids adding a new per-sample ragged scheduler and is part of strict noise replay.

## One combined state

At each NFE, PixelGen duplicates the latent, concatenates conditioning in upstream order, calls the model once, forecasts/updates factors whose batch dimension is `2B`, verifies the complete combined tensor, and advances once. It is invalid to split into cond/uncond network calls, reverse order, maintain two histories/schedulers, check only half, or reduce each half independently.

The Lightning denoiser and its EMA deepcopy each receive independent empty runtime state. A prediction batch begins one trajectory and ends/resets it in `finally`; factors, previous error, counters, and NFE index never cross batches. Its trajectory identity is derived from the manifest `batch_group_id`, so it remains globally unique across the four independent shards and stable across resume; aggregation accepts repeated per-sample rows only when both batch group and complete trajectory summary match.

## Batch larger than one

Any real `B>1` run is separately labeled **grouped-batch SpeCa**: one batch-global decision covers multiple real samples and effective batch is `2B`. It requires a new immutable batch-group manifest, fixed order, fresh threshold selection, separate latency/memory reporting, and cannot be called strictly sample-adaptive or merged with main batch-1 results.

Changing runtime batch against a batch-1 manifest is rejected. Resume preserves frozen groups; per-sample explicit noise remains independent of rank execution order.

## Fairness consequences

- Matched Full and SpeCa paired runs both use real batch 1/effective batch 2 and the same manifest.
- JiT's two separate stream calls are a different model-family layout; cross-family latencies are not divided into a speedup.
- Common-batch throughput is a separate scenario and does not replace batch-1 latency or four-GPU wall clock.
- The current PixelGen Full real-batch-4 output is `PAIRED_METRICS_BLOCKED`; see [`BASELINE_COMPATIBILITY_REPORT.md`](BASELINE_COMPATIBILITY_REPORT.md).
