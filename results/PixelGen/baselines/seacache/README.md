# PixelGen SeaCache 1K threshold sweep

Completed on 2026-07-15 UTC. Source run root:
`/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/seacache/pixelgen_1k`.

## Fixed protocol

| Field | Value |
| --- | --- |
| Model | PixelGen-JiT |
| Samples | 1,000 (1 per ImageNet class) |
| Base seed | 202607140000 |
| Resolution | 256 x 256 |
| GPUs / batch | 4 GPUs, batch 4 per rank |
| Sampler | Exact Heun, 50 steps |
| CFG / interval | 2.25 / [0.1, 0.9] |
| Precision / compile | bf16-mixed / matched_eager |
| Noise scale / timeshift | 1.0 / 2.0 |
| Full throughput | 0.945141 images/s |
| PyTorch | 2.7.1+cu126 |
| Repository commit | de6d80e7722d7fe8a12486f96028c43ec4e57a72 |
| SeaCache commit | 8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2 |
| Manifest SHA-256 | 31536470eacf69e07ccd72305e7866957d15859b2091eec7daed2a309cedf5c0 |

## Efficiency and ADM distribution metrics

Metric directions are FID/sFID lower, IS/Precision/Recall higher. Delta is
SeaCache minus Full; speedup is candidate throughput divided by Full
throughput.

| Threshold | Time (s) | img/s | Speedup | FID | Delta FID | sFID | IS | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 1058.043 | 0.945 | 1.000x | 38.4539 | - | 202.9272 | 279.8703 | 0.808 | 0.8250 |
| 0.02 | 563.233 | 1.775 | 1.879x | 38.1516 | -0.3023 | 203.4207 | 281.6956 | 0.809 | 0.8236 |
| 0.05 | 537.421 | 1.861 | 1.969x | 38.1161 | -0.3378 | 203.0274 | 281.0384 | 0.813 | 0.8244 |
| 0.10 | 379.074 | 2.638 | 2.791x | 38.1734 | -0.2806 | 200.8107 | 277.4614 | 0.810 | 0.8216 |
| 0.20 | 265.896 | 3.761 | 3.979x | 38.5828 | +0.1289 | 197.7004 | 267.6060 | 0.816 | 0.8191 |
| 0.40 | 175.483 | 5.699 | 6.029x | 39.3013 | +0.8473 | 193.0614 | 270.0007 | 0.826 | 0.8140 |

## Paired metrics against Full

PSNR/SSIM are higher and LPIPS is lower. Values are means over the 1,000
strictly paired samples.

| Threshold | PSNR | SSIM | LPIPS | Aggregate MSE | Exact pixel pairs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 34.9103 | 0.9624 | 0.05113 | 0.00109188 | 0 |
| 0.05 | 35.0866 | 0.9628 | 0.05109 | 0.00107744 | 0 |
| 0.10 | 32.9917 | 0.9488 | 0.05855 | 0.00115243 | 0 |
| 0.20 | 28.6330 | 0.8959 | 0.09298 | 0.00186662 | 0 |
| 0.40 | 22.9059 | 0.7480 | 0.24592 | 0.00621258 | 0 |

All paired runs report zero NaN and zero Inf for PSNR, SSIM, and LPIPS.
Complete percentiles, per-class summaries, worst samples, and all per-sample
rows are retained under `raw/`.

## Cache trace

Counts below sum all four ranks. They are batched combined-CFG denoiser calls,
not per-image diffusion steps.

| Threshold | Full calls | Reuse calls | Reuse fraction |
| ---: | ---: | ---: | ---: |
| Full | 24,948 | 0 | 0.0000 |
| 0.02 | 12,852 | 12,096 | 0.4848 |
| 0.05 | 12,026 | 12,922 | 0.5180 |
| 0.10 | 8,408 | 16,540 | 0.6630 |
| 0.20 | 5,757 | 19,191 | 0.7692 |
| 0.40 | 3,396 | 21,552 | 0.8639 |

## Artifact scope

- `protocol/manifest_1k.jsonl` and metadata bind sample ID, class, seed, shard,
  and batch grouping.
- `protocol/configs/` contains the five frozen threshold configs.
- `raw/<run>/` contains the copied result artifacts for each operating point.
- Generated PNGs and ADM NPZs remain under the source run root above.

This is a single-run 1K proxy sweep. It has no across-seed confidence interval
and must not be reported as the final 50K ImageNet score.
