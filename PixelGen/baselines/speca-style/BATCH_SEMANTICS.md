# PixelGen batch and CFG semantics

## Registered main protocol

```text
real generation batch per GPU process = 4
combined model batch                  = 8
combined order                        = [unconditional, conditional]
Taylor histories                      = 1 combined state
scheduler / error                     = 1 per fixed batch group
gate_mode                             = batch_global
manifest batch grouping               = four real samples per group
```

The released metric produces one scalar for the full model input. Under the unified protocol, that scalar/action covers four real samples and both CFG halves. This grouped-batch interpretation, fixed order, and threshold are part of strict noise replay.

## One combined state

At each NFE, PixelGen duplicates the latent, concatenates conditioning in upstream order, calls the model once, forecasts/updates factors whose batch dimension is `2B`, verifies the complete combined tensor, and advances once. It is invalid to split into cond/uncond network calls, reverse order, maintain two histories/schedulers, check only half, or reduce each half independently.

The Lightning denoiser and its EMA deepcopy each receive independent empty runtime state. A prediction batch begins one trajectory and ends/resets it in `finally`; factors, previous error, counters, and NFE index never cross batches. Its trajectory identity is derived from the manifest `batch_group_id`, so it remains globally unique across the four independent shards and stable across resume; aggregation accepts repeated per-sample rows only when both batch group and complete trajectory summary match.

## Fixed grouped-batch execution

The main run is **grouped-batch SpeCa** at real batch 4/effective batch 8. One decision covers all four real samples and cannot be called strictly sample-adaptive. Changing runtime batch against a frozen manifest is rejected. Resume preserves frozen groups; per-sample explicit noise remains independent of rank execution order.

## Fairness consequences

- Matched Full and SpeCa paired runs both use real batch 4/effective batch 8 and the same manifest.
- JiT's two separate stream calls are a different model-family layout; cross-family latencies are not divided into a speedup.
- Primary timing reports batch-4 per-image latency and throughput alongside four-GPU wall clock.
- Legacy Full outputs without immutable batch-4 replay evidence remain `PAIRED_METRICS_BLOCKED`; see [`BASELINE_COMPATIBILITY_REPORT.md`](BASELINE_COMPATIBILITY_REPORT.md).
