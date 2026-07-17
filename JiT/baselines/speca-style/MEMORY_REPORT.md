# JiT SpeCa memory report

## Analytic main-protocol estimate

Registered estimate: JiT-B/16 at 256 px, depth 12, hidden size 768, 256 image tokens, 32 context tokens from block 4 onward, real batch 32, two separate CFG histories, BF16 factors, `max_order=4`, verifier on block 11.

| Component | Analytic value |
|---|---:|
| Taylor factor tensors | 240 = 2 streams × 12 layers × 2 modules × 5 factors |
| Taylor cache | 3,271,557,120 bytes = 3.046875 GiB |
| one verify-layer feature per stream | 14,155,776 bytes |
| retained draft/exact payloads | 56,623,104 bytes |
| cloned prefix + exact attention + exact MLP lower bound | 42,467,328 bytes |
| elementwise metric temporaries lower bound | 42,467,328 bytes |
| verifier temporaries lower bound | 141,557,760 bytes |
| analytic cache + verifier increment | 3,413,114,880 bytes = 3.1787109375 GiB |

The Taylor formula counts `max_order+1` tensors for independent attention and MLP histories at each layer. The context-aware token layout is used per layer; conditional and unconditional histories never alias.

## What the estimate does not prove

Verifier temporary memory is an accounting lower bound. It excludes QKV, attention/SDPA, SwiGLU, allocator fragmentation, compiler caches, model weights, base solver activations, CUDA library workspaces, and overlap/lifetime differences. It therefore is not an OOM certificate and is not a measured `peak_memory_allocated` or `peak_memory_reserved` value.

Deferred GPU runs must report Taylor cache bytes/tensor count, verifier temporary estimate, allocated/reserved peaks, and delta against matched Full. The trace must retain verification block, error-reduction, and scalar-sync costs separately.

Matched `instrumented_full` allocates no Taylor history and has no verifier, so its draft-cache contribution is exactly zero; observed delta versus matched Full includes the complete SpeCa cache and verifier/runtime effects.

## Scaling and OOM policy

- Cache storage is linear in real batch, CFG streams, layers/tokens, hidden size, bytes per element, and `max_order+1`.
- `cache_dtype=inherit` is the main protocol. There is no silent FP16 conversion, quantization, compression, CPU offload, layer dropping, order reduction, or TaylorSeer-Lite substitution.
- If batch 32 OOMs, record the failure. A lower-batch rerun is a separately labeled ablation and cannot replace the unified primary protocol.
- Runtime storage is reset at every trajectory boundary and must return zero live factor tensors afterward.

The CPU estimator is useful for preregistration, while only an authorized CUDA run can establish the real peak.
