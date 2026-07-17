# JiT SeaCache 1K threshold sweep

Completed on 2026-07-14 UTC. Source run root:
`/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/seacache/jit_1k`.

## Fixed protocol

| Field | Value |
| --- | --- |
| Model | JiT-B/16 |
| Samples | 1,000 (1 per ImageNet class) |
| Base seed | 202607140000 |
| Resolution | 256 x 256 |
| GPUs / batch | 4 GPUs, batch 32 per rank |
| Sampler | Heun, 50 steps |
| CFG / interval | 3.0 / [0.1, 1.0] |
| Precision / compile | bfloat16 / matched_eager |
| Noise scale / timeshift | 1.0 / none |
| Full throughput | 3.169411 images/s |
| PyTorch | 2.5.1+cu124 |
| Repository commit | de6d80e7722d7fe8a12486f96028c43ec4e57a72 |
| SeaCache commit | 8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2 |
| Manifest SHA-256 | e8ddfb2a2470661b7fbc46bd9077c2432195ae2b6986a5b466a760f68797bc1c |

## Efficiency and ADM distribution metrics

Metric directions are FID/sFID lower, IS/Precision/Recall higher. Delta is
SeaCache minus Full; speedup is candidate throughput divided by Full
throughput.

| Threshold | Time (s) | img/s | Speedup | FID | Delta FID | sFID | IS | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 315.516 | 3.169 | 1.000x | 38.9845 | - | 201.4650 | 261.0170 | 0.812 | 0.8053 |
| 0.02 | 149.868 | 6.673 | 2.105x | 39.0504 | +0.0659 | 201.8413 | 267.1178 | 0.806 | 0.8082 |
| 0.05 | 137.373 | 7.279 | 2.297x | 39.1256 | +0.1411 | 200.6500 | 266.5899 | 0.810 | 0.8067 |
| 0.10 | 109.474 | 9.135 | 2.882x | 39.5665 | +0.5820 | 198.7603 | 255.2751 | 0.808 | 0.8030 |
| 0.20 | 80.272 | 12.458 | 3.931x | 41.3029 | +2.3184 | 195.4610 | 233.8701 | 0.764 | 0.7926 |
| 0.40 | 63.452 | 15.760 | 4.973x | 44.4266 | +5.4421 | 190.9715 | 202.1211 | 0.699 | 0.7805 |

## Paired metrics against Full

PSNR/SSIM are higher and LPIPS is lower. Values are means over the 1,000
strictly paired samples.

| Threshold | PSNR | SSIM | LPIPS | Aggregate MSE | Exact pixel pairs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 32.9179 | 0.9623 | 0.05624 | 0.00110327 | 0 |
| 0.05 | 33.5719 | 0.9575 | 0.05842 | 0.00103530 | 0 |
| 0.10 | 30.8948 | 0.9304 | 0.07452 | 0.00132084 | 0 |
| 0.20 | 25.6253 | 0.8492 | 0.12968 | 0.00317042 | 0 |
| 0.40 | 20.2414 | 0.6834 | 0.23504 | 0.01006525 | 0 |

All paired runs report zero NaN and zero Inf for PSNR, SSIM, and LPIPS.
Complete percentiles, per-class summaries, worst samples, and all per-sample
rows are retained under `raw/`.

## Cache trace

Counts below sum all four ranks. They are batched combined-CFG denoiser calls,
not per-image diffusion steps.

| Threshold | Full calls | Reuse calls | Reuse fraction |
| ---: | ---: | ---: | ---: |
| Full | 6,336 | 0 | 0.0000 |
| 0.02 | 3,264 | 3,072 | 0.4848 |
| 0.05 | 2,376 | 3,960 | 0.6250 |
| 0.10 | 1,797 | 4,539 | 0.7164 |
| 0.20 | 1,232 | 5,104 | 0.8056 |
| 0.40 | 832 | 5,504 | 0.8687 |

## Artifact scope

- `protocol/manifest_1k.jsonl` and metadata bind sample ID, class, seed, shard,
  and batch grouping.
- `protocol/configs/` contains the five frozen threshold configs.
- `raw/<run>/` contains the copied result artifacts for each operating point.
- Generated PNGs and ADM NPZs remain under the source run root above.

This is a single-run 1K proxy sweep. It has no across-seed confidence interval
and must not be reported as the final 50K ImageNet score.

The Full and SeaCache run manifests have different `model_config_hash` values
because the frozen configs represent the same resolved checkpoint path
differently (relative versus absolute) and include the cache integration. The
resolved checkpoint path, checkpoint size, model architecture, sampler, and
weights agree across the comparison.

