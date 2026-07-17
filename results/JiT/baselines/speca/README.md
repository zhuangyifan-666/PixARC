# JiT SpeCa 1K parameter sweep

Completed on 2026-07-16 UTC. Source run root:
/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/speca/jit_1k/runs.

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
| CFG execution | separate cond then uncond forwards; one shared scheduler |
| SpeCa fixed settings | first_enhance=3, threshold_floor=0.01, relative-L1 error (eps=1e-10), released-code-faithful scheduler, official NFE index, all-token last-layer verification, batch-global gate, inherited cache dtype |
| Full throughput | 3.185995 images/s |
| CUDA benchmark | Not run; efficiency below is validated four-GPU generation wall clock |
| PyTorch | 2.5.1+cu124 |
| Repository commit | de6d80e7722d7fe8a12486f96028c43ec4e57a72 |
| TaylorSeer/SpeCa source commit | 704ee98c74f7f04da443daa3c0aa2cc7803d86e3 |
| Port source SHA-256 | 6d6454d5e33735bf3c4282b30e41322780d3877949cd226f9d0a444676852462 |
| Manifest SHA-256 | 3082e12dc36c8c7d7b41467a7c7f391182e033b854c53f89aa87fc8c2aacfa95 |

## Efficiency and ADM distribution metrics

FID/sFID are lower-is-better; IS/Precision/Recall are higher-is-better.
Delta FID is SpeCa minus Full. Speedup is candidate four-GPU generation
throughput divided by the matched Full throughput; it is not a CUDA-event
microbenchmark. O/T/D/S denote max order, base threshold, decay rate, and
minimum-maximum Taylor span.

| Point | Setting (O/T/D/S) | Time (s) | img/s | Speedup | FID | Delta FID | sFID | IS | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | O4/T0.3/D0.05/S3-8 | 313.874 | 3.186 | 1.000x | 38.9845 | - | 201.4650 | 261.0170 | 0.8120 | 0.8053 |
| speca_ref | O4/T0.3/D0.05/S3-8 | 110.723 | 9.032 | 2.835x | 38.7810 | -0.2035 | 200.7715 | 274.8570 | 0.8190 | 0.8104 |
| order1 | O1/T0.3/D0.05/S3-8 | 107.860 | 9.271 | 2.910x | 38.9397 | -0.0447 | 200.4430 | 275.7073 | 0.8090 | 0.8106 |
| order2 | O2/T0.3/D0.05/S3-8 | 115.055 | 8.691 | 2.728x | 38.7124 | -0.2721 | 200.8560 | 274.7113 | 0.8230 | 0.8079 |
| order3 | O3/T0.3/D0.05/S3-8 | 116.137 | 8.610 | 2.703x | 38.7834 | -0.2010 | 200.8323 | 275.5755 | 0.8190 | 0.8116 |
| threshold0p1 | O4/T0.1/D0.05/S3-8 | 122.532 | 8.161 | 2.562x | 38.7810 | -0.2035 | 200.7715 | 274.8570 | 0.8190 | 0.8104 |
| threshold0p2 | O4/T0.2/D0.05/S3-8 | 123.855 | 8.074 | 2.534x | 38.7810 | -0.2035 | 200.7715 | 274.8570 | 0.8190 | 0.8104 |
| threshold0p4 | O4/T0.4/D0.05/S3-8 | 124.743 | 8.016 | 2.516x | 38.7810 | -0.2035 | 200.7715 | 274.8570 | 0.8190 | 0.8104 |
| decay0p01 | O4/T0.3/D0.01/S3-8 | 121.458 | 8.233 | 2.584x | 38.7810 | -0.2035 | 200.7715 | 274.8570 | 0.8190 | 0.8104 |
| decay0p1 | O4/T0.3/D0.1/S3-8 | 122.178 | 8.185 | 2.569x | 38.7810 | -0.2035 | 200.7715 | 274.8570 | 0.8190 | 0.8104 |
| minstep2 | O4/T0.3/D0.05/S2-8 | 142.935 | 6.996 | 2.196x | 38.7255 | -0.2590 | 199.5915 | 277.1816 | 0.8170 | 0.8114 |
| minstep4 | O4/T0.3/D0.05/S4-8 | 113.405 | 8.818 | 2.768x | 39.0455 | +0.0610 | 199.0536 | 280.7462 | 0.8160 | 0.8116 |
| maxstep5 | O4/T0.3/D0.05/S3-5 | 122.051 | 8.193 | 2.572x | 38.7810 | -0.2035 | 200.7715 | 274.8570 | 0.8190 | 0.8104 |
| maxstep10 | O4/T0.3/D0.05/S3-10 | 121.715 | 8.216 | 2.579x | 38.7810 | -0.2035 | 200.7715 | 274.8570 | 0.8190 | 0.8104 |

## Paired metrics against Full

PSNR/SSIM are higher-is-better and LPIPS is lower-is-better. Means are
computed over 1,000 strictly paired samples.

| Point | Setting (O/T/D/S) | PSNR | SSIM | LPIPS | Aggregate MSE | Exact pixel pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| speca_ref | O4/T0.3/D0.05/S3-8 | 26.9986 | 0.9097 | 0.13193 | 0.00310383 | 0 |
| order1 | O1/T0.3/D0.05/S3-8 | 26.9867 | 0.9084 | 0.13603 | 0.00310528 | 0 |
| order2 | O2/T0.3/D0.05/S3-8 | 26.9756 | 0.9096 | 0.13183 | 0.00310947 | 0 |
| order3 | O3/T0.3/D0.05/S3-8 | 26.9978 | 0.9097 | 0.13188 | 0.00310396 | 0 |
| threshold0p1 | O4/T0.1/D0.05/S3-8 | 26.9986 | 0.9097 | 0.13193 | 0.00310383 | 0 |
| threshold0p2 | O4/T0.2/D0.05/S3-8 | 26.9986 | 0.9097 | 0.13193 | 0.00310383 | 0 |
| threshold0p4 | O4/T0.4/D0.05/S3-8 | 26.9986 | 0.9097 | 0.13193 | 0.00310383 | 0 |
| decay0p01 | O4/T0.3/D0.01/S3-8 | 26.9986 | 0.9097 | 0.13193 | 0.00310383 | 0 |
| decay0p1 | O4/T0.3/D0.1/S3-8 | 26.9986 | 0.9097 | 0.13193 | 0.00310383 | 0 |
| minstep2 | O4/T0.3/D0.05/S2-8 | 26.5742 | 0.9073 | 0.13165 | 0.00315302 | 0 |
| minstep4 | O4/T0.3/D0.05/S4-8 | 26.0247 | 0.8961 | 0.15264 | 0.00359320 | 0 |
| maxstep5 | O4/T0.3/D0.05/S3-5 | 26.9986 | 0.9097 | 0.13193 | 0.00310383 | 0 |
| maxstep10 | O4/T0.3/D0.05/S3-10 | 26.9986 | 0.9097 | 0.13193 | 0.00310383 | 0 |

All paired runs report zero NaN and zero Inf for PSNR, SSIM, and LPIPS.
Complete percentiles, per-class summaries, worst samples, and per-sample
rows are retained under raw/.

## SpeCa execution trace

Counts sum all four ranks. Taylor fraction is Taylor NFE / total NFE.
The cache and peak-memory columns are means across recorded trajectories.

| Point | Full NFE | Taylor NFE | Taylor fraction | Network forwards | Verify pass rate | Mean cache (GiB) | Mean peak allocated (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 3168 | 0 | 0.0000 | 6336 | - | 0.000 | 1.314 |
| speca_ref | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 2.975 | 4.398 |
| order1 | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 1.190 | 2.609 |
| order2 | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 1.785 | 3.205 |
| order3 | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 2.380 | 3.802 |
| threshold0p1 | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 2.975 | 4.397 |
| threshold0p2 | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 2.975 | 4.398 |
| threshold0p4 | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 2.975 | 4.398 |
| decay0p01 | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 2.975 | 4.397 |
| decay0p1 | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 2.975 | 4.397 |
| minstep2 | 832 | 2336 | 0.7374 | 6336 | 0.0000 | 2.975 | 4.398 |
| minstep4 | 576 | 2592 | 0.8182 | 6336 | 0.0000 | 2.975 | 4.397 |
| maxstep5 | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 2.975 | 4.397 |
| maxstep10 | 672 | 2496 | 0.7879 | 6336 | 0.0000 | 2.975 | 4.398 |

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
