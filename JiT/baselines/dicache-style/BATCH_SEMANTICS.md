# Batch semantics

The frozen main protocol uses real batch size 32. Error and gamma reductions are batch-global within each branch, so all 32 real samples share one action and residual estimate. This grouped-batch interpretation, threshold, grouping, latency, and quality protocol is shared by Full and DiCache.

Manifest rows fix sample ID, class, signed 63-bit seed, shard, batch-group ID, and position within a group. Each initial Gaussian is created by an independent CPU `torch.Generator`, drawn as float32, stacked in fixed group order, and then copied to the assigned GPU. This makes noise replay independent of rank scheduling and resume order.

Resume skips only complete groups whose PNG and metadata identities validate. A partial group, corrupt image, changed checkpoint/config/manifest, or regrouped batch fails closed. Full and DiCache paired metrics require the same manifest and immutable grouping; numeric filenames alone are not pairing proof.
