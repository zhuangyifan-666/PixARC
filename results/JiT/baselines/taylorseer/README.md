# JiT TaylorSeer 1K interval/order sweep

Completed on 2026-07-15 UTC. Source run root:
`/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/taylorseer/jit_1k`.

## Fixed protocol

| Field | Value |
| --- | --- |
| Model | JiT-B/16 |
| Samples | 1,000 (1 per ImageNet class) |
| Base seed | 202607140000 |
| Resolution | 256 x 256 |
| GPUs / batch | 4 GPUs, batch 32 per rank |
| Sampler | Exact Heun, 50 steps / 99 NFE |
| CFG / interval | 3.0 / [0.1, 1.0] |
| Precision / compile | bfloat16 / matched_eager |
| Noise scale / timeshift | 1.0 / none |
| TaylorSeer fixed settings | first_enhance=2, official_nfe_index, force_last_full=false |
| Full throughput | 3.297669 images/s |
| PyTorch | 2.5.1+cu124 |
| Repository commit | de6d80e7722d7fe8a12486f96028c43ec4e57a72 |
| TaylorSeer commit | 704ee98c74f7f04da443daa3c0aa2cc7803d86e3 |
| Port source SHA-256 | 05217172d5f5317563c1171ce7b93f3be82cd1b58ba11dd8371d71abc544d525 |
| Manifest SHA-256 | e8ddfb2a2470661b7fbc46bd9077c2432195ae2b6986a5b466a760f68797bc1c |

## Efficiency and ADM distribution metrics

Metric directions are FID/sFID lower and IS/Precision/Recall higher. Delta is
TaylorSeer minus Full; speedup is candidate throughput divided by Full throughput.

| Interval | Order | Time (s) | img/s | Speedup | FID | Delta FID | sFID | IS | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | - | 303.244 | 3.298 | 1.000x | 38.9845 | - | 201.4650 | 261.0170 | 0.812 | 0.8053 |
| 2 | 1 | 185.822 | 5.381 | 1.632x | 39.6302 | +0.6458 | 203.0135 | 251.4935 | 0.786 | 0.8113 |
| 2 | 2 | 202.289 | 4.943 | 1.499x | 39.4702 | +0.4857 | 202.9348 | 250.5097 | 0.789 | 0.8117 |
| 2 | 3 | 197.350 | 5.067 | 1.537x | 39.4695 | +0.4850 | 202.9601 | 250.2103 | 0.780 | 0.8119 |
| 3 | 1 | 140.972 | 7.094 | 2.151x | 39.6413 | +0.6569 | 200.8410 | 262.0131 | 0.790 | 0.8167 |
| 3 | 2 | 145.998 | 6.849 | 2.077x | 39.5235 | +0.5390 | 200.9760 | 262.5371 | 0.796 | 0.8183 |
| 3 | 3 | 156.059 | 6.408 | 1.943x | 39.5479 | +0.5634 | 200.9389 | 262.9890 | 0.791 | 0.8174 |
| 4 | 1 | 111.004 | 9.009 | 2.732x | 40.5882 | +1.6038 | 201.8528 | 244.5671 | 0.768 | 0.8070 |
| 4 | 2 | 125.074 | 7.995 | 2.425x | 40.2975 | +1.3131 | 202.3350 | 244.8687 | 0.796 | 0.8043 |
| 4 | 3 | 146.000 | 6.849 | 2.077x | 40.2935 | +1.3090 | 202.2962 | 243.9487 | 0.800 | 0.8084 |

## Paired metrics against Full

PSNR/SSIM are higher and LPIPS is lower. Values are means over 1,000
strictly paired samples.

| Interval | Order | PSNR | SSIM | LPIPS | Aggregate MSE | Exact pixel pairs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 27.5749 | 0.9256 | 0.10008 | 0.00227759 | 0 |
| 2 | 2 | 27.9168 | 0.9296 | 0.09578 | 0.00214474 | 0 |
| 2 | 3 | 27.9135 | 0.9295 | 0.09584 | 0.00214598 | 0 |
| 3 | 1 | 27.2236 | 0.9185 | 0.10848 | 0.00250135 | 0 |
| 3 | 2 | 28.3802 | 0.9326 | 0.09359 | 0.00199671 | 0 |
| 3 | 3 | 28.3390 | 0.9321 | 0.09392 | 0.00201321 | 0 |
| 4 | 1 | 22.9695 | 0.8421 | 0.18443 | 0.00587404 | 0 |
| 4 | 2 | 23.6272 | 0.8588 | 0.17083 | 0.00513040 | 0 |
| 4 | 3 | 23.5328 | 0.8567 | 0.17239 | 0.00523714 | 0 |

All paired runs report zero NaN and zero Inf for PSNR, SSIM, and LPIPS.
Complete percentiles, per-class summaries, worst samples, and per-sample rows
are retained under `raw/`.

## TaylorSeer execution trace

Counts sum all four ranks and operate on manifest batch trajectories. Taylor
fraction is Taylor-forecast NFE divided by total NFE.

| Interval | Order | Full NFE | Taylor NFE | Taylor fraction | Network forwards | Max cache (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | - | 3168 | 0 | 0.0000 | 6336 | 0.000 |
| 2 | 1 | 1600 | 1568 | 0.4949 | 6336 | 1.219 |
| 2 | 2 | 1600 | 1568 | 0.4949 | 6336 | 1.828 |
| 2 | 3 | 1600 | 1568 | 0.4949 | 6336 | 2.438 |
| 3 | 1 | 1088 | 2080 | 0.6566 | 6336 | 1.219 |
| 3 | 2 | 1088 | 2080 | 0.6566 | 6336 | 1.828 |
| 3 | 3 | 1088 | 2080 | 0.6566 | 6336 | 2.438 |
| 4 | 1 | 832 | 2336 | 0.7374 | 6336 | 1.219 |
| 4 | 2 | 832 | 2336 | 0.7374 | 6336 | 1.828 |
| 4 | 3 | 832 | 2336 | 0.7374 | 6336 | 2.438 |

## Artifact scope

- `protocol/manifest_1k.jsonl` and metadata bind sample ID, class, seed, shard,
  generator device, noise construction, and batch grouping.
- `protocol/configs/` contains the frozen Full and nine TaylorSeer configs.
- `raw/<run>/` contains run manifests, validation, wall-clock records, ADM and
  paired metrics, per-sample paired CSV, and all four rank metadata/summaries.
- Generated PNGs and ADM NPZs remain under the source run root above.

This is a deterministic single-run 1K proxy sweep, not a final 50K ImageNet
estimate. It has no across-seed confidence interval; operating-point selection
should be treated as a quality-throughput Pareto decision.
