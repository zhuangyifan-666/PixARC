# PixelGen SpeCa memory report

## Analytic main-protocol estimate

Registered estimate: PixelGen JiT-XL/2 at 256 px, depth 28, hidden 1152, 256 image tokens, 32 context tokens from block 8 onward, real batch 4, one combined effective-8 state, BF16 factors, `max_order=4`, verifier on block 27.

| Component | Analytic value |
|---|---:|
| Taylor factor tensors | 280 = 1 state × 28 layers × 2 modules × 5 factors |
| Taylor cache | 1,439,170,560 bytes = 1.34033203125 GiB |
| one verify-layer feature (effective batch 8) | 5,308,416 bytes |
| retained draft/exact payloads | 10,616,832 bytes |
| cloned prefix + exact attention + exact MLP lower bound | 15,925,248 bytes |
| elementwise metric temporaries lower bound | 15,925,248 bytes |
| verifier temporaries lower bound | 42,467,328 bytes |
| analytic cache + verifier increment | 1,481,637,888 bytes = 1.3798828125 GiB |

The single history has effective tensor batch `2B` in PixelGen's unchanged `[unconditional, conditional]` layout. Context-aware token count is applied per block.

## Limits of the estimate

This verifier value is a lower-bound feature accounting. It excludes QKV/attention/SDPA, SwiGLU, allocator fragmentation, compiler caches, base model/solver activations, model weights, CUDA libraries, and actual lifetime overlap. It is not a measured peak and cannot certify 3090 feasibility.

Deferred runs must report factor bytes/count, verifier estimates, `peak_memory_allocated`, `peak_memory_reserved`, and delta against matched Full. Verification block, reduction, and scalar-sync time are reported separately.

Matched `instrumented_full` allocates no Taylor history and has no verifier, so its draft-cache contribution is exactly zero; observed delta versus matched Full includes the complete SpeCa cache and verifier/runtime effects.

## Scaling and OOM policy

- Storage grows linearly with real batch/effective `2B`, token/hidden dimensions, layers, bytes per element, and `max_order+1`.
- Main `cache_dtype=inherit`; no silent FP16, quantization, compression, CPU offload, layer removal, order reduction, or TaylorSeer-Lite.
- If batch 4 OOMs, record the failure. A lower-batch rerun is a separately labeled ablation and cannot replace the unified primary protocol.
- Every prediction batch starts and ends a trajectory; reset must remove all factor storage, including across EMA/deepcopy module lifecycles.

Only an authorized CUDA run can establish real peak memory.
