# JiT batch and CFG semantics

## Registered main protocol

```text
real generation batch per GPU process = 1
conditional model batch               = 1
unconditional model batch             = 1
effective CFG samples per NFE          = 2, executed as two forwards
gate_mode                              = batch_global
manifest batch grouping                = one real sample per group
```

The released error reduces an entire input batch to one scalar. Real batch 1 is therefore required to support a sample-adaptive interpretation without inventing per-sample ragged scheduling. It is also part of strict Full/SpeCa noise replay and cannot be changed after a manifest is frozen.

## One scheduler, two histories

Each NFE has exactly one decision and two stream executions:

1. `begin_nfe` returns one action/check/threshold/coordinate.
2. `cond` executes with its own factors and exact anchors.
3. `uncond` executes with a distinct factor store but the same decision.
4. Verification payloads are combined with sufficient statistics equivalent to concatenation.
5. `end_nfe` validates that both streams completed and advances once.

It is invalid to share factor tensors, increment after the conditional call, use different actions or coordinates, verify one stream only, average incompatible already-reduced scalars, or create two schedulers.

## Batch larger than one

A real batch larger than one is permitted only as a separately named **grouped-batch SpeCa** experiment. Its single error/action covers multiple real samples; it is not strictly sample-adaptive. Such a run requires a separately generated immutable manifest, fixed grouping/order, independently tuned thresholds, its own memory/latency report, and no mixing with batch-1 main results.

Changing only the runtime batch against a batch-1 manifest is rejected. Resume also preserves the original batch groups, while per-sample explicit noise makes output independent of rank execution order.

## Fairness consequences

- Both matched Full and SpeCa paired runs use real batch 1 and the identical manifest.
- PixelGen's combined `2B` arrangement is not numerically the same execution layout as JiT's two forwards; results are reported per model family, not cross-divided.
- Common-batch throughput can be reported separately, but it does not replace batch-1 latency or four-GPU 50K wall time.
- The existing JiT Full batch-32 reference is `PAIRED_METRICS_BLOCKED` for this protocol; see [`BASELINE_COMPATIBILITY_REPORT.md`](BASELINE_COMPATIBILITY_REPORT.md).
