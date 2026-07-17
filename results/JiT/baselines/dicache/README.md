# JiT DiCache 1K threshold sweep

Completed on 2026-07-16 UTC. Source run root:
/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/dicache/jit_1k.

## Fixed protocol

| Field | Value |
| --- | --- |
| Model | JiT-B/16 |
| Samples | 1,000 (1 per ImageNet class) |
| Base seed | 202607140000 |
| Resolution | 256 x 256 |
| GPUs / batch | 4 GPUs, real batch 32 per rank / effective CFG batch 64 |
| Sampler | Exact Heun, 50 steps / 99 NFE |
| CFG / interval | 3.0 / [0.1, 1.0] |
| Precision / compile | bfloat16 / matched_eager |
| Noise scale / timeshift | 1.0 / none |
| CFG execution | separate conditional/unconditional streams |
| DiCache fixed settings | profile=flux_image_released, error=delta_y, probe_depth=1, ret_ratio=0.2, DCTA order=1, gamma=[1.0,1.5], batch-global gate |
| Full throughput | 3.120559 images/s |
| CUDA benchmark | Not run; efficiency below is validated four-GPU generation wall clock |
| PyTorch | 2.5.1+cu124 |
| Repository commit | de6d80e7722d7fe8a12486f96028c43ec4e57a72 |
| DiCache commit | fdbe20b669c9174bbed5ec994de073fd881c8010 |
| Port source SHA-256 | d0593f831ece2a9d8b40c6d9075a58b99cd7f6f9212125ae42c9aa8e638cb18d |
| Manifest SHA-256 | e8ddfb2a2470661b7fbc46bd9077c2432195ae2b6986a5b466a760f68797bc1c |

## Efficiency and ADM distribution metrics

FID/sFID are lower-is-better; IS/Precision/Recall are higher-is-better.
Delta is DiCache minus Full. Speedup is candidate four-GPU generation
throughput divided by the matched Full throughput; it is not a CUDA-event
microbenchmark.

| Threshold | Time (s) | img/s | Speedup | FID | Delta FID | sFID | IS | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 320.455 | 3.121 | 1.000x | 38.9068 | - | 201.0475 | 258.9089 | 0.816 | 0.8084 |
| 0.01 | 220.053 | 4.544 | 1.456x | 38.9152 | +0.0084 | 200.8122 | 258.0685 | 0.812 | 0.8093 |
| 0.02 | 233.114 | 4.290 | 1.375x | 38.9152 | +0.0084 | 200.8122 | 258.0685 | 0.812 | 0.8093 |
| 0.04 | 181.257 | 5.517 | 1.768x | 38.9433 | +0.0365 | 200.3986 | 259.5123 | 0.816 | 0.8083 |
| 0.08 | 156.064 | 6.408 | 2.053x | 39.0021 | +0.0953 | 199.6952 | 257.6139 | 0.807 | 0.8075 |
| 0.16 | 136.454 | 7.328 | 2.348x | 39.1836 | +0.2768 | 198.0607 | 254.7288 | 0.802 | 0.8036 |
| 0.32 | 126.172 | 7.926 | 2.540x | 40.2896 | +1.3828 | 194.3511 | 235.8331 | 0.766 | 0.7931 |
| 0.64 | 116.062 | 8.616 | 2.761x | 45.5108 | +6.6039 | 191.3429 | 169.7140 | 0.681 | 0.7674 |

## Paired metrics against Full

PSNR/SSIM are higher-is-better and LPIPS is lower-is-better. Means are
computed over 1,000 strictly paired samples.

| Threshold | PSNR | SSIM | LPIPS | Aggregate MSE | Exact pixel pairs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 51.2330 | 0.9982 | 0.00051 | 0.00000814 | 0 |
| 0.02 | 51.2330 | 0.9982 | 0.00051 | 0.00000814 | 0 |
| 0.04 | 47.5358 | 0.9968 | 0.00170 | 0.00002224 | 0 |
| 0.08 | 43.5019 | 0.9942 | 0.00527 | 0.00005133 | 0 |
| 0.16 | 35.6008 | 0.9765 | 0.02307 | 0.00029800 | 0 |
| 0.32 | 28.1419 | 0.9097 | 0.08751 | 0.00168616 | 0 |
| 0.64 | 21.8296 | 0.7791 | 0.21779 | 0.00699132 | 0 |

All paired runs report zero NaN and zero Inf for PSNR, SSIM, and LPIPS.
Complete percentiles, per-class summaries, worst samples, and per-sample
rows are retained under raw/.

## DiCache execution trace

Counts sum all four ranks. Direct Full + Resumed Full + Reuse equals
the total stream-call count. Reuse fraction is Reuse / total calls.

| Threshold | Direct Full | Resumed Full | Reuse | Reuse fraction | DCTA | Max gap | Max cache (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 6336 | 0 | 0 | 0.0000 | 0 | 1 | 0.0 |
| 0.01 | 1344 | 2496 | 2496 | 0.3939 | 2496 | 2 | 240.0 |
| 0.02 | 1344 | 2496 | 2496 | 0.3939 | 2496 | 2 | 240.0 |
| 0.04 | 1344 | 1585 | 3407 | 0.5377 | 3407 | 4 | 240.0 |
| 0.08 | 1344 | 935 | 4057 | 0.6403 | 4057 | 6 | 240.0 |
| 0.16 | 1344 | 512 | 4480 | 0.7071 | 4480 | 10 | 240.0 |
| 0.32 | 1344 | 257 | 4735 | 0.7473 | 4735 | 20 | 240.0 |
| 0.64 | 1344 | 128 | 4864 | 0.7677 | 4864 | 34 | 240.0 |

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

