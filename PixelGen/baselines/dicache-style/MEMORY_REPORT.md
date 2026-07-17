# Cache-memory report

The estimator is analytic and does not construct a model or allocate CUDA memory. For PixelGen-XL 256, patch size 16, hidden size 1152, real batch 4/effective combined batch 8, BF16 inherited cache, probe depth 1, it reports:

```text
persistent cache bytes: 28,311,552
persistent tensor count: 6
temporary probe lower bound: 4,718,592 bytes
```

The six persistent tensors are previous body input, previous probe feature, and two synchronized exact anchor pairs (full and probe residual for each of the last two Full calls). Scalar counters and Python metadata are excluded. A bounded two-anchor deque removes FLUX’s ineffective older residual history without changing DCTA outputs.

The temporary lower bound accounts for one resumable probe state; actual attention/MLP activations, compiler workspaces, allocator fragmentation, final-head tensors, model parameters, EMA parameters, and Lightning overhead are excluded. `cache_dtype=fp32` changes cache size and is not the main inherited-BF16 protocol.

At runtime, adapted and upstream-Full sampler summaries reset and read CUDA peak allocated/reserved only inside the guarded, explicitly authorized GPU path; CPU summaries use zero as “not measured.” They also report live cache bytes/tensor count before trajectory state is released. The generation aggregator preserves per-trajectory peak maxima. No RTX 3090 peak-memory measurement has been run. Use:

```bash
python scripts/estimate_cache_memory.py --preset pixelgen-xl-256 \
  --batch-size 4 --probe-depth 1 --cache-dtype bfloat16
```
