# Unified baseline batch protocol

The primary PixARC baseline experiments use one fixed real batch per GPU process:

| Model family | Real batch | CFG execution | Effective CFG work |
|---|---:|---|---:|
| JiT | 32 | separate conditional and unconditional forwards | 64 samples/NFE |
| PixelGen | 4 | one combined `[unconditional, conditional]` forward | 8 samples/NFE |

SeaCache, TaylorSeer, SpeCa, DiCache, and each method's matched Full use the same family-specific batch, immutable manifest grouping, checkpoint/EMA, sampler, dtype, compile mode, and timing boundary.

SpeCa and DiCache retain their released batch-global reductions. Consequently, batch-32 JiT and batch-4 PixelGen runs are grouped-batch experiments: every sample in a group shares the cache action. Thresholds selected at batch 1 are invalid for this protocol and must be selected again on the registered 1K/8K splits.

Latency is reported as CUDA-event milliseconds per image at the fixed batch, together with images/second and peak allocated/reserved memory. Batch-1 diagnostic timings and outputs generated under old manifests cannot be used as the primary denominator or for strict paired metrics.

If a method OOMs at the registered batch, record the failure. A lower-batch run is a separately labeled ablation and cannot silently replace the primary comparison.
