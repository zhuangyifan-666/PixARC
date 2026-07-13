# Batch semantics

The frozen main protocol uses real batch size 1. This makes each `cond` and `uncond` gate sample-adaptive while retaining two independent branch decisions.

At real batch `B>1`, error and gamma reductions are batch-global within each branch. All B real samples in that branch share one action and residual estimate. Such a run is a grouped-batch experiment: its threshold, grouping, latency, and quality are not directly comparable to batch-1 results.

Manifest rows fix sample ID, class, signed 63-bit seed, shard, batch-group ID, and position within a group. Each initial Gaussian is created by an independent CPU `torch.Generator`, drawn as float32, stacked in fixed group order, and then copied to the assigned GPU. This makes noise replay independent of rank scheduling and resume order.

Resume skips only complete groups whose PNG and metadata identities validate. A partial group, corrupt image, changed checkpoint/config/manifest, or regrouped batch fails closed. Full and DiCache paired metrics require the same manifest and immutable grouping; numeric filenames alone are not pairing proof.
