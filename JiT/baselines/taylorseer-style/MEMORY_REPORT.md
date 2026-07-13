# JiT Taylor cache memory report

No GPU memory measurement or OOM search was run in this implementation round.
The values below are analytical upper bounds for factor storage only; they do
not include model parameters, activations, compiler workspace, allocator
fragmentation, sampler state, or trace objects.

## Estimator model

For each stream, layer, and module (`attn`, `mlp`), order K stores up to K+1
tensors shaped `[batch,tokens,hidden]`. JiT has two independent streams. At
256x256, JiT-B/16 has 256 image tokens; layers 0--3 use 256 and layers 4--11
use 288 after 32 context tokens are inserted.

```text
bytes = streams * modules * factors * batch
        * sum_layer(tokens) * hidden * bytes_per_element
```

For JiT-B/16, `sum_layer(tokens)=3328`, hidden=768, streams=2, modules=2.
With inherited BF16 factors:

| batch/rank | max_order | max factors/module | factor tensors | cache GiB |
|---:|---:|---:|---:|---:|
| 32 | 0 | 1 | 48 | 0.6094 |
| 32 | 4 | 5 | 240 | 3.0469 |
| 1 | 4 | 5 | 240 | 0.0952 |

The factor count matures with exact anchors, so early calls allocate less than
the K=4 upper bound. `cache_dtype=fp32` is a non-faithful explicit ablation and
doubles these BF16 numbers. It must not be selected silently.

## Runtime accounting

Runtime reports unique tensor storage as `cache_allocated_bytes`/
`cache_bytes`, tensor count, available order per module, history/forecast
times, and trajectory summaries. GPU benchmarks must additionally report
`torch.cuda.max_memory_allocated`, `max_memory_reserved`, and the delta against
matched Full after resetting peak statistics at the same boundary. Reset must
return cache bytes to zero and release factor references.

## 3090 status and OOM policy

Maximum viable batch on a 24-GB RTX 3090 is **unknown and unmeasured**. Batch
32/rank is inherited from the queued Full protocol, not a promise that the
K=4 Taylor cache fits with all model/compiler memory. Run the estimator first,
then deferred small smoke tests. If any method OOMs, lower the batch for **all**
methods in that latency comparison and regenerate matching manifests/configs.
Never substitute TaylorSeer-Lite, lower order, FP16 cache, compression, or CPU
offload without labeling a separate ablation.

