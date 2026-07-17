# JiT batch and CFG semantics

## Registered main protocol

```text
real generation batch per GPU process = 32
conditional model batch               = 32
unconditional model batch             = 32
effective CFG samples per NFE          = 64, executed as two forwards
gate_mode                              = batch_global
manifest batch grouping                = 32 real samples per group
```

The released error reduces an entire input batch to one scalar. Under the unified baseline protocol, one action covers the fixed group of 32 real samples in each branch. This is a grouped-batch rather than sample-adaptive interpretation; its grouping and threshold are frozen parts of strict Full/SpeCa replay.

## One scheduler, two histories

Each NFE has exactly one decision and two stream executions:

1. `begin_nfe` returns one action/check/threshold/coordinate.
2. `cond` executes with its own factors and exact anchors.
3. `uncond` executes with a distinct factor store but the same decision.
4. Verification payloads are combined with sufficient statistics equivalent to concatenation.
5. `end_nfe` validates that both streams completed and advances once.

It is invalid to share factor tensors, increment after the conditional call, use different actions or coordinates, verify one stream only, average incompatible already-reduced scalars, or create two schedulers.

## Fixed grouped-batch execution

The main experiment is **grouped-batch SpeCa** at batch 32. Its single error/action covers multiple real samples and is not described as sample-adaptive. Changing runtime batch or regrouping a frozen manifest is rejected. Resume preserves the original groups, while per-sample explicit noise makes output independent of rank execution order.

## Fairness consequences

- Both matched Full and SpeCa paired runs use real batch 32 and the identical manifest.
- PixelGen's combined `2B` arrangement is not numerically the same execution layout as JiT's two forwards; results are reported per model family, not cross-divided.
- Primary timing reports batch-32 per-image latency and throughput alongside four-GPU 50K wall time.
- Legacy Full outputs without immutable batch-32 replay evidence remain `PAIRED_METRICS_BLOCKED`; see [`BASELINE_COMPATIBILITY_REPORT.md`](BASELINE_COMPATIBILITY_REPORT.md).
