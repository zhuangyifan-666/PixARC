# Memory report

For JiT-B/16 256, the image boundary has 256 tokens and hidden size 768. With real batch 32, two independent CFG streams, BF16/FP16 cache storage, and two exact anchor pairs, the persistent lower-bound model is:

```text
per feature tensor = 32 * 256 * 768 * 2 bytes = 12,582,912 bytes
per stream tensors = previous body + previous probe
                   + 2 full residuals + 2 probe residuals = 6
two streams        = 12 tensors = 150,994,944 bytes
temporary probe-state lower bound per active stream = 12,582,912 bytes
```

FP32 cache storage doubles persistent bytes to 301,989,888 while the fresh-head input is cast back to the body dtype. Run the estimator rather than relying on this prose:

```bash
python scripts/estimate_cache_memory.py \
  --preset jit-b16-256 --batch-size 32 --probe-depth 1 \
  --cache-dtype bfloat16
```

The estimator excludes attention matrices, block activations/workspaces, allocator fragmentation, model/EMA/checkpoint weights, compilation caches, final-head temporaries, image output, and framework state. Runtime summaries deduplicate tensor storages to report live cache bytes/tensors. Guarded generation/benchmark also records CUDA peak allocated and reserved memory.

No GPU memory run was performed, so no RTX 3090 or other device peak is reported. Compare measured instrumented Full versus DiCache under the same process/protocol and report both allocated and reserved deltas.
