# PixelGen DiCache 1K threshold sweep

Completed on 2026-07-16 UTC. Source run root:
/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/dicache/pixelgen_1k_v2.

## Fixed protocol

| Field | Value |
| --- | --- |
| Model | PixelGen-JiT |
| Samples | 1,000 (1 per ImageNet class) |
| Base seed | 202607140000 |
| Resolution | 256 x 256 |
| GPUs / batch | 4 GPUs, real batch 4 per rank / effective CFG batch 8 |
| Sampler | Exact Heun, 50 steps / 99 NFE |
| CFG / interval | 2.25 / [0.1, 0.9] |
| Precision / compile | bf16-mixed / matched_eager |
| Noise scale / timeshift | 1.0 / 2.0 |
| CFG execution | combined [unconditional, conditional] CFG stream |
| DiCache fixed settings | profile=flux_image_released, error=delta_y, probe_depth=1, ret_ratio=0.2, DCTA order=1, gamma=[1.0,1.5], batch-global gate |
| Full throughput | 0.926207 images/s |
| CUDA benchmark | Not run; efficiency below is validated four-GPU generation wall clock |
| PyTorch | 2.7.1+cu126 |
| Repository commit | de6d80e7722d7fe8a12486f96028c43ec4e57a72 |
| DiCache commit | fdbe20b669c9174bbed5ec994de073fd881c8010 |
| Port source SHA-256 | 271afc3f176e0da1263641e345a8fb6b692906674d567823975dea3d3c55cb67 |
| Manifest SHA-256 | 31536470eacf69e07ccd72305e7866957d15859b2091eec7daed2a309cedf5c0 |

## Efficiency and ADM distribution metrics

FID/sFID are lower-is-better; IS/Precision/Recall are higher-is-better.
Delta is DiCache minus Full. Speedup is candidate four-GPU generation
throughput divided by the matched Full throughput; it is not a CUDA-event
microbenchmark.

| Threshold | Time (s) | img/s | Speedup | FID | Delta FID | sFID | IS | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 1079.672 | 0.926 | 1.000x | 38.4539 | - | 202.9272 | 279.8703 | 0.808 | 0.8250 |
| 0.01 | 683.914 | 1.462 | 1.579x | 38.5327 | +0.0787 | 202.4898 | 281.0082 | 0.809 | 0.8246 |
| 0.02 | 684.096 | 1.462 | 1.578x | 38.5302 | +0.0762 | 202.4784 | 281.1316 | 0.812 | 0.8247 |
| 0.04 | 524.238 | 1.908 | 2.060x | 38.6351 | +0.1811 | 202.1087 | 277.3130 | 0.795 | 0.8223 |
| 0.08 | 416.768 | 2.399 | 2.591x | 38.5881 | +0.1342 | 200.5371 | 273.0484 | 0.802 | 0.8222 |
| 0.16 | 355.052 | 2.816 | 3.041x | 39.0197 | +0.5657 | 197.5242 | 266.8529 | 0.792 | 0.8197 |
| 0.32 | 330.064 | 3.030 | 3.271x | 39.8721 | +1.4181 | 191.8508 | 239.2215 | 0.759 | 0.8115 |
| 0.64 | 298.479 | 3.350 | 3.617x | 46.2749 | +7.8210 | 184.7652 | 159.3790 | 0.640 | 0.7661 |

## Paired metrics against Full

PSNR/SSIM are higher-is-better and LPIPS is lower-is-better. Means are
computed over 1,000 strictly paired samples.

| Threshold | PSNR | SSIM | LPIPS | Aggregate MSE | Exact pixel pairs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 49.3252 | 0.9975 | 0.00257 | 0.00002821 | 0 |
| 0.02 | 49.2430 | 0.9975 | 0.00259 | 0.00002838 | 0 |
| 0.04 | 42.5739 | 0.9904 | 0.01476 | 0.00018798 | 0 |
| 0.08 | 34.7595 | 0.9711 | 0.04329 | 0.00071884 | 0 |
| 0.16 | 29.2321 | 0.9316 | 0.09203 | 0.00263157 | 0 |
| 0.32 | 23.9553 | 0.8260 | 0.19404 | 0.00964627 | 0 |
| 0.64 | 19.8189 | 0.6410 | 0.34998 | 0.01174183 | 0 |

All paired runs report zero NaN and zero Inf for PSNR, SSIM, and LPIPS.
Complete percentiles, per-class summaries, worst samples, and per-sample
rows are retained under raw/.

## DiCache execution trace

Counts sum all four ranks. Direct Full + Resumed Full + Reuse equals
the total stream-call count. Reuse fraction is Reuse / total calls.

| Threshold | Direct Full | Resumed Full | Reuse | Reuse fraction | DCTA | Max gap | Max cache (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 24948 | 0 | 0 | 0.0000 | 0 | 1 | 0.0 |
| 0.01 | 5292 | 9894 | 9762 | 0.3913 | 9762 | 2 | 45.0 |
| 0.02 | 5292 | 9828 | 9828 | 0.3939 | 9828 | 2 | 45.0 |
| 0.04 | 5292 | 6016 | 13640 | 0.5467 | 13640 | 4 | 45.0 |
| 0.08 | 5292 | 3427 | 16229 | 0.6505 | 16229 | 8 | 45.0 |
| 0.16 | 5292 | 1789 | 17867 | 0.7162 | 17867 | 16 | 45.0 |
| 0.32 | 5292 | 978 | 18678 | 0.7487 | 18678 | 32 | 45.0 |
| 0.64 | 5292 | 504 | 19152 | 0.7677 | 19152 | 56 | 45.0 |

All candidate traces report zero non-finite gamma values and zero
zero-order fallbacks.

## Artifact scope

- protocol/manifest_1k.jsonl and its sidecar bind sample ID, class, seed,
  shard, generator device, noise construction, and batch grouping.
- protocol/configs/ contains the frozen Full and seven DiCache configs;
  protocol/selection/ preserves the coarse-grid selection records.
- raw/<run>/ contains run manifests, validation, wall-clock records, ADM,
  paired and trace metrics, per-sample paired CSV, and rank metadata/summaries.
- Generated PNGs and large ADM sample NPZ files remain under the source
  run root above.

This is a deterministic single-run 1K proxy sweep, not a final 50K
ImageNet estimate. It has no across-seed confidence interval. Operating-
point selection must be treated as a quality-throughput Pareto decision.
No CUDA-event benchmark result is claimed or archived.

