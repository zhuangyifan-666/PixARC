# PixelGen SpeCa 1K parameter sweep

Completed on 2026-07-16 UTC. Source run root:
/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/speca/pixelgen_1k/runs.

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
| CFG execution | single combined [unconditional, conditional] effective-2B forward |
| SpeCa fixed settings | first_enhance=3, threshold_floor=0.01, relative-L1 error (eps=1e-10), released-code-faithful scheduler, official NFE index, all-token last-layer verification, batch-global gate, inherited cache dtype |
| Full throughput | 0.913764 images/s |
| CUDA benchmark | Not run; efficiency below is validated four-GPU generation wall clock |
| PyTorch | 2.7.1+cu126 |
| Repository commit | de6d80e7722d7fe8a12486f96028c43ec4e57a72 |
| TaylorSeer/SpeCa source commit | 704ee98c74f7f04da443daa3c0aa2cc7803d86e3 |
| Port source SHA-256 | a5d9c6d3035d6dce0056faa7b08f8633dead1e83e4c5c0ba5f7463ba23e11efe |
| Manifest SHA-256 | fe29e475202c9e711009610dfec6a647fd246cd2019cbc3b7bfa263e47bd7138 |

## Efficiency and ADM distribution metrics

FID/sFID are lower-is-better; IS/Precision/Recall are higher-is-better.
Delta FID is SpeCa minus Full. Speedup is candidate four-GPU generation
throughput divided by the matched Full throughput; it is not a CUDA-event
microbenchmark. O/T/D/S denote max order, base threshold, decay rate, and
minimum-maximum Taylor span.

| Point | Setting (O/T/D/S) | Time (s) | img/s | Speedup | FID | Delta FID | sFID | IS | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | O4/T0.3/D0.05/S3-8 | 1094.375 | 0.914 | 1.000x | 38.4539 | - | 202.9272 | 279.8703 | 0.8080 | 0.8250 |
| speca_ref | O4/T0.3/D0.05/S3-8 | 398.486 | 2.509 | 2.746x | 38.4086 | -0.0453 | 203.3156 | 285.8260 | 0.8110 | 0.8248 |
| order1 | O1/T0.3/D0.05/S3-8 | 347.475 | 2.878 | 3.150x | 38.2829 | -0.1711 | 203.2900 | 286.4025 | 0.8080 | 0.8234 |
| order2 | O2/T0.3/D0.05/S3-8 | 378.692 | 2.641 | 2.890x | 38.4790 | +0.0250 | 203.7210 | 288.4922 | 0.8070 | 0.8258 |
| order3 | O3/T0.3/D0.05/S3-8 | 401.056 | 2.493 | 2.729x | 38.3982 | -0.0557 | 203.4569 | 288.3271 | 0.8120 | 0.8247 |
| threshold0p1 | O4/T0.1/D0.05/S3-8 | 413.377 | 2.419 | 2.647x | 38.4086 | -0.0453 | 203.3156 | 285.8260 | 0.8110 | 0.8248 |
| threshold0p2 | O4/T0.2/D0.05/S3-8 | 412.003 | 2.427 | 2.656x | 38.4086 | -0.0453 | 203.3156 | 285.8260 | 0.8110 | 0.8248 |
| threshold0p4 | O4/T0.4/D0.05/S3-8 | 407.754 | 2.452 | 2.684x | 38.4086 | -0.0453 | 203.3156 | 285.8260 | 0.8110 | 0.8248 |
| decay0p01 | O4/T0.3/D0.01/S3-8 | 412.155 | 2.426 | 2.655x | 38.4086 | -0.0453 | 203.3156 | 285.8260 | 0.8110 | 0.8248 |
| decay0p1 | O4/T0.3/D0.1/S3-8 | 413.387 | 2.419 | 2.647x | 38.4086 | -0.0453 | 203.3156 | 285.8260 | 0.8110 | 0.8248 |
| minstep2 | O4/T0.3/D0.05/S2-8 | 464.560 | 2.153 | 2.356x | 38.2334 | -0.2206 | 202.6817 | 292.5203 | 0.8300 | 0.8269 |
| minstep4 | O4/T0.3/D0.05/S4-8 | 379.487 | 2.635 | 2.884x | 38.2438 | -0.2102 | 201.8181 | 281.6172 | 0.8100 | 0.8226 |
| maxstep5 | O4/T0.3/D0.05/S3-5 | 416.860 | 2.399 | 2.625x | 38.4086 | -0.0453 | 203.3156 | 285.8260 | 0.8110 | 0.8248 |
| maxstep10 | O4/T0.3/D0.05/S3-10 | 411.457 | 2.430 | 2.660x | 38.4181 | -0.0358 | 203.3255 | 286.2307 | 0.8120 | 0.8224 |

## Paired metrics against Full

PSNR/SSIM are higher-is-better and LPIPS is lower-is-better. Means are
computed over 1,000 strictly paired samples.

| Point | Setting (O/T/D/S) | PSNR | SSIM | LPIPS | Aggregate MSE | Exact pixel pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| speca_ref | O4/T0.3/D0.05/S3-8 | 26.8129 | 0.8815 | 0.15601 | 0.00432550 | 0 |
| order1 | O1/T0.3/D0.05/S3-8 | 26.6009 | 0.8784 | 0.16086 | 0.00470193 | 0 |
| order2 | O2/T0.3/D0.05/S3-8 | 26.8687 | 0.8822 | 0.15487 | 0.00422241 | 0 |
| order3 | O3/T0.3/D0.05/S3-8 | 26.8256 | 0.8816 | 0.15568 | 0.00429738 | 0 |
| threshold0p1 | O4/T0.1/D0.05/S3-8 | 26.8129 | 0.8815 | 0.15601 | 0.00432550 | 0 |
| threshold0p2 | O4/T0.2/D0.05/S3-8 | 26.8129 | 0.8815 | 0.15601 | 0.00432550 | 0 |
| threshold0p4 | O4/T0.4/D0.05/S3-8 | 26.8129 | 0.8815 | 0.15601 | 0.00432550 | 0 |
| decay0p01 | O4/T0.3/D0.01/S3-8 | 26.8129 | 0.8815 | 0.15601 | 0.00432550 | 0 |
| decay0p1 | O4/T0.3/D0.1/S3-8 | 26.8129 | 0.8815 | 0.15601 | 0.00432550 | 0 |
| minstep2 | O4/T0.3/D0.05/S2-8 | 26.2571 | 0.8790 | 0.15791 | 0.00403697 | 0 |
| minstep4 | O4/T0.3/D0.05/S4-8 | 25.1819 | 0.8587 | 0.18477 | 0.00540769 | 0 |
| maxstep5 | O4/T0.3/D0.05/S3-5 | 26.8129 | 0.8815 | 0.15601 | 0.00432550 | 0 |
| maxstep10 | O4/T0.3/D0.05/S3-10 | 26.8055 | 0.8813 | 0.15613 | 0.00433781 | 0 |

All paired runs report zero NaN and zero Inf for PSNR, SSIM, and LPIPS.
Complete percentiles, per-class summaries, worst samples, and per-sample
rows are retained under raw/.

## SpeCa execution trace

Counts sum all four ranks. Taylor fraction is Taylor NFE / total NFE.
The cache and peak-memory columns are means across recorded trajectories.

| Point | Full NFE | Taylor NFE | Taylor fraction | Network forwards | Verify pass rate | Mean cache (GiB) | Mean peak allocated (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 24948 | 0 | 0.0000 | 24948 | - | 0.000 | 5.558 |
| speca_ref | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 1.330 | 6.966 |
| order1 | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 0.532 | 6.148 |
| order2 | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 0.798 | 6.417 |
| order3 | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 1.064 | 6.686 |
| threshold0p1 | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 1.330 | 6.965 |
| threshold0p2 | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 1.330 | 6.965 |
| threshold0p4 | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 1.330 | 6.966 |
| decay0p01 | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 1.330 | 6.965 |
| decay0p1 | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 1.330 | 6.966 |
| minstep2 | 6552 | 18396 | 0.7374 | 24948 | 0.0000 | 1.330 | 6.985 |
| minstep4 | 4536 | 20412 | 0.8182 | 24948 | 0.0000 | 1.330 | 6.965 |
| maxstep5 | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 1.330 | 6.966 |
| maxstep10 | 5292 | 19656 | 0.7879 | 24948 | 0.0000 | 1.330 | 6.965 |

Verification pass rate is diagnostic rather than a quality metric. The
full trace has no verification attempts; its displayed dash is not missing
experimental data.

## Artifact scope

- protocol/manifest_1k.jsonl and its sidecar bind sample ID, class, seed,
  shard, generator device, noise construction, and batch grouping.
- protocol/configs/ contains the frozen Full and 13 SpeCa configurations.
- raw/<run>/ contains run manifests, validation, wall-clock records, ADM,
  paired and trace metrics, per-sample paired CSV, ADM sidecars, and rank
  metadata/summaries.
- Generated PNGs and large ADM sample NPZ files remain under the source
  run root above.

This is a deterministic single-run 1K proxy sweep, not a final 50K
ImageNet estimate. It has no across-seed confidence interval. Operating-
point selection must be treated as a quality-throughput Pareto decision.
No CUDA-event benchmark result is claimed or archived.
