# PixelGen Taylor cache memory report

No GPU memory measurement or OOM search was run in this implementation round.
The values below are analytical upper bounds for factor storage only; they do
not include model parameters, activations, compiler workspace, allocator
fragmentation, sampler state, VAE, or trace objects.

## Estimator model

For each layer and module (`attn`, `mlp`), order K stores up to K+1 tensors
shaped `[2*batch,tokens,hidden]`. PixelGen uses one `combined_cfg` stream, not
two separate histories. At 256x256, PixelGen-XL has 256 image tokens; layers
0--7 use 256 and layers 8--27 use 288 after 32 context tokens are inserted.

```text
bytes = streams * modules * factors * effective_2B
        * sum_layer(tokens) * hidden * bytes_per_element
```

For PixelGen-XL, `sum_layer(tokens)=7808`, hidden=1152, streams=1, modules=2.
With inherited BF16 factors:

| image batch/rank | effective CFG batch | max_order | factor tensors | cache GiB |
|---:|---:|---:|---:|---:|
| 4 | 8 | 0 | 56 | 0.2681 |
| 4 | 8 | 4 | 280 | 1.3403 |
| 1 | 2 | 4 | 280 | 0.3351 |

The factor count matures with exact anchors, so early calls allocate less than
the K=4 upper bound. `cache_dtype=fp32` is a non-faithful explicit ablation and
doubles these BF16 numbers. It must not be selected silently.

## Runtime accounting

Runtime reports unique tensor storage as `cache_allocated_bytes`/
`cache_bytes`, tensor count, available order per module, history/forecast
times, and trajectory summaries. GPU benchmarks must additionally report
`torch.cuda.max_memory_allocated`, `max_memory_reserved`, and the delta against
matched Full after resetting peak statistics at the same boundary. Reset must
return cache bytes to zero and release factor references, including on
exceptions and after the last short batch.

## 3090 status and OOM policy

Maximum viable batch on a 24-GB RTX 3090 is **unknown and unmeasured**. Batch
4/rank is inherited from the active Full protocol, not a promise that every
order/compile mode fits. Run the estimator first, then deferred small smoke
tests. If any method OOMs, lower the batch for **all** methods in that latency
comparison and regenerate matching manifests/configs. Never substitute Lite,
lower order, FP16 cache, compression, or CPU offload without labeling a
separate ablation.

