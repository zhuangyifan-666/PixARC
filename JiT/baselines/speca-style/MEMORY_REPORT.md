# JiT SpeCa memory report

## Analytic main-protocol estimate

Registered estimate: JiT-B/16 at 256 px, depth 12, hidden size 768, 256 image tokens, 32 context tokens from block 4 onward, real batch 1, two separate CFG histories, BF16 factors, `max_order=4`, verifier on block 11.

| Component | Analytic value |
|---|---:|
| Taylor factor tensors | 240 = 2 streams × 12 layers × 2 modules × 5 factors |
| Taylor cache | 102,236,160 bytes = 0.09521484375 GiB |
| one verify-layer feature per stream | 442,368 bytes |
| retained draft/exact payloads | 1,769,472 bytes |
| cloned prefix + exact attention + exact MLP lower bound | 1,327,104 bytes |
| elementwise metric temporaries lower bound | 1,327,104 bytes |
| verifier temporaries lower bound | 4,423,680 bytes = 0.004119873047 GiB |
| analytic cache + verifier increment | 106,659,840 bytes = 0.099334716797 GiB |

The Taylor formula counts `max_order+1` tensors for independent attention and MLP histories at each layer. The context-aware token layout is used per layer; conditional and unconditional histories never alias.

## What the estimate does not prove

Verifier temporary memory is an accounting lower bound. It excludes QKV, attention/SDPA, SwiGLU, allocator fragmentation, compiler caches, model weights, base solver activations, CUDA library workspaces, and overlap/lifetime differences. It therefore is not an OOM certificate and is not a measured `peak_memory_allocated` or `peak_memory_reserved` value.

Deferred GPU runs must report Taylor cache bytes/tensor count, verifier temporary estimate, allocated/reserved peaks, and delta against matched Full. The trace must retain verification block, error-reduction, and scalar-sync costs separately.

Matched `instrumented_full` allocates no Taylor history and has no verifier, so its draft-cache contribution is exactly zero; observed delta versus matched Full includes the complete SpeCa cache and verifier/runtime effects.

## Scaling and OOM policy

- Cache storage is linear in real batch, CFG streams, layers/tokens, hidden size, bytes per element, and `max_order+1`.
- `cache_dtype=inherit` is the main protocol. There is no silent FP16 conversion, quantization, compression, CPU offload, layer dropping, order reduction, or TaylorSeer-Lite substitution.
- If a 3090 OOMs, reduce the real batch consistently for every compared method; the main protocol is already batch 1. If batch 1 still OOMs, report it rather than changing the method.
- Runtime storage is reset at every trajectory boundary and must return zero live factor tensors afterward.

The CPU estimator is useful for preregistration, while only an authorized CUDA run can establish the real peak.
