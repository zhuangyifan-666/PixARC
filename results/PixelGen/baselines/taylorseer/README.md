# PixelGen TaylorSeer 1K interval/order sweep

Completed on 2026-07-15 UTC. Source run root:
`/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/taylorseer/pixelgen_1k`.

## Fixed protocol

| Field | Value |
| --- | --- |
| Model | PixelGen-JiT |
| Samples | 1,000 (1 per ImageNet class) |
| Base seed | 202607140000 |
| Resolution | 256 x 256 |
| GPUs / batch | 4 GPUs, batch 4 per rank |
| Sampler | Exact Heun, 50 steps / 99 NFE |
| CFG / interval | 2.25 / [0.1, 0.9] |
| Precision / compile | bf16-mixed / matched_eager |
| Noise scale / timeshift | 1.0 / 2.0 |
| TaylorSeer fixed settings | first_enhance=2, official_nfe_index, force_last_full=false |
| Full throughput | 0.915721 images/s |
| PyTorch | 2.7.1+cu126 |
| Repository commit | de6d80e7722d7fe8a12486f96028c43ec4e57a72 |
| TaylorSeer commit | 704ee98c74f7f04da443daa3c0aa2cc7803d86e3 |
| Port source SHA-256 | 6c5b8237ae08c5b50875e78e281a577a42d15dd8bf213bbaae447e4092162f6c |
| Manifest SHA-256 | 31536470eacf69e07ccd72305e7866957d15859b2091eec7daed2a309cedf5c0 |

## Efficiency and ADM distribution metrics

Metric directions are FID/sFID lower and IS/Precision/Recall higher. Delta is
TaylorSeer minus Full; speedup is candidate throughput divided by Full throughput.

| Interval | Order | Time (s) | img/s | Speedup | FID | Delta FID | sFID | IS | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | - | 1092.036 | 0.916 | 1.000x | 38.4806 | - | 202.8929 | 280.1995 | 0.807 | 0.8248 |
| 2 | 1 | 605.539 | 1.651 | 1.803x | 38.9390 | +0.4584 | 202.7689 | 267.0165 | 0.799 | 0.8175 |
| 2 | 2 | 634.632 | 1.576 | 1.721x | 38.9152 | +0.4346 | 202.8709 | 270.7324 | 0.801 | 0.8172 |
| 2 | 3 | 660.697 | 1.514 | 1.653x | 38.9175 | +0.4369 | 202.9843 | 270.2230 | 0.801 | 0.8181 |
| 3 | 1 | 465.280 | 2.149 | 2.347x | 38.5324 | +0.0519 | 202.4743 | 272.4709 | 0.809 | 0.8175 |
| 3 | 2 | 497.279 | 2.011 | 2.196x | 38.5626 | +0.0821 | 202.4550 | 277.9644 | 0.797 | 0.8189 |
| 3 | 3 | 527.578 | 1.895 | 2.070x | 38.5560 | +0.0754 | 202.5072 | 278.2957 | 0.800 | 0.8165 |
| 4 | 1 | 383.764 | 2.606 | 2.846x | 39.1341 | +0.6536 | 202.0233 | 261.8008 | 0.786 | 0.8269 |
| 4 | 2 | 410.254 | 2.438 | 2.662x | 39.2422 | +0.7616 | 202.2710 | 262.3144 | 0.777 | 0.8234 |
| 4 | 3 | 432.241 | 2.314 | 2.526x | 39.0740 | +0.5935 | 202.2265 | 263.7378 | 0.784 | 0.8242 |

## Paired metrics against Full

PSNR/SSIM are higher and LPIPS is lower. Values are means over 1,000
strictly paired samples.

| Interval | Order | PSNR | SSIM | LPIPS | Aggregate MSE | Exact pixel pairs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 27.6690 | 0.9066 | 0.12100 | 0.00286759 | 0 |
| 2 | 2 | 27.8288 | 0.9081 | 0.11936 | 0.00280694 | 0 |
| 2 | 3 | 27.8179 | 0.9075 | 0.11983 | 0.00281500 | 0 |
| 3 | 1 | 29.1542 | 0.9218 | 0.10466 | 0.00242941 | 0 |
| 3 | 2 | 30.6140 | 0.9352 | 0.08774 | 0.00186343 | 0 |
| 3 | 3 | 30.5157 | 0.9342 | 0.08913 | 0.00190153 | 0 |
| 4 | 1 | 24.3346 | 0.8506 | 0.18568 | 0.00557105 | 0 |
| 4 | 2 | 25.0345 | 0.8637 | 0.17192 | 0.00472444 | 0 |
| 4 | 3 | 24.9911 | 0.8629 | 0.17282 | 0.00484463 | 0 |

All paired runs report zero NaN and zero Inf for PSNR, SSIM, and LPIPS.
Complete percentiles, per-class summaries, worst samples, and per-sample rows
are retained under `raw/`.

## TaylorSeer execution trace

Counts sum all four ranks and operate on manifest batch trajectories. Taylor
fraction is Taylor-forecast NFE divided by total NFE.

| Interval | Order | Full NFE | Taylor NFE | Taylor fraction | Network forwards | Max cache (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | - | 24948 | 0 | 0.0000 | 24948 | 0.000 |
| 2 | 1 | 12600 | 12348 | 0.4949 | 24948 | 0.536 |
| 2 | 2 | 12600 | 12348 | 0.4949 | 24948 | 0.804 |
| 2 | 3 | 12600 | 12348 | 0.4949 | 24948 | 1.072 |
| 3 | 1 | 8568 | 16380 | 0.6566 | 24948 | 0.536 |
| 3 | 2 | 8568 | 16380 | 0.6566 | 24948 | 0.804 |
| 3 | 3 | 8568 | 16380 | 0.6566 | 24948 | 1.072 |
| 4 | 1 | 6552 | 18396 | 0.7374 | 24948 | 0.536 |
| 4 | 2 | 6552 | 18396 | 0.7374 | 24948 | 0.804 |
| 4 | 3 | 6552 | 18396 | 0.7374 | 24948 | 1.072 |

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
