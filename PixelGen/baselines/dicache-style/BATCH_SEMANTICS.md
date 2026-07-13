# Batch and manifest semantics

The released-code-faithful gate reduces over the whole tensor. For real batch B, PixelGen’s effective model batch is 2B and all real samples share the same action and gamma.

The primary template uses:

```text
protocol: sample_adaptive
real batch: 1
effective combined CFG batch: 2
```

This is the closest available interpretation of sample-specific scheduling. B>1 must be labeled `grouped_batch`: it changes probe errors, refresh positions, DCTA, and often latency. A threshold selected at B=1 cannot be reused at another batch size. Dynamic ragged splitting is intentionally absent because it would be a new algorithm.

The immutable manifest binds sample ID, class ID, per-sample seed, shard, within-shard position, `batch_group_id`, and `position_in_batch`. Initial noise uses an independent CPU `torch.Generator.manual_seed` per sample; it does not depend on rank scheduling, prior samples, resume point, or worker count. Resume skips only complete groups whose PNG and metadata validate. A partial group, corrupt PNG, duplicate ID, or identity/config mismatch fails closed.

Formal Full and DiCache runs must use the same manifest, batch grouping, checkpoint/EMA, sampler, CFG, dtype, compile mode, and postprocessing. Batch grouping is part of the pairing contract, not a throughput-only option.

