# PixelGen evaluation protocol

Generation, distribution metrics, paired metrics, and latency are separate
phases. Metric time is never generation latency. This task implemented entry
points but did not execute FID, LPIPS, or any GPU evaluation.

## Output preflight

Before any metric, validate exactly the manifest sample IDs, no duplicates or
missing files, class counts, metadata/config/manifest/checkpoint identities,
numeric sample ordering, and decodable RGB uint8 PNG at 256x256. Lexicographic
ordering (`10.png` before `2.png`) is forbidden. A 50K final run must contain
50 samples for each of 1,000 classes.

## Distribution quality

The wrapper invokes one explicit local ADM evaluator and one explicit local
ImageNet-256 reference NPZ for every Full/SeaCache/TaylorSeer comparison. It
supports FID, sFID, Inception Score, precision, and recall and records sample
count, manifest SHA-256, evaluator identity/commit, reference path, image
protocol, and timestamp. A missing reference or evaluator is an error; no
download or fallback evaluator occurs.

Report candidate-minus-Full deltas:

```text
delta_fid       = FID_candidate - FID_full
delta_sfid      = sFID_candidate - sFID_full
delta_is        = IS_candidate - IS_full
delta_precision = precision_candidate - precision_full
delta_recall    = recall_candidate - recall_full
```

The active upstream compressed Full may be usable here after complete-output
validation even though its initial noise cannot be paired.

## Strict paired fidelity

PSNR, SSIM, and LPIPS require manifest-backed Full and TaylorSeer runs with
matching sample ID, class, seed/noise protocol, model/checkpoint/EMA, sampler,
steps, CFG/timeshift, dtype, resolution, batch grouping, compile mode, and
postprocessing. Missing metadata, duplicates, or any mismatch fails closed.

Saved RGB uint8 PNGs are converted to float32 `[0,1]`:

- PSNR uses RGB and `data_range=1`; identical images retain infinity. Report
  aggregate MSE, PSNR from aggregate MSE, mean/median/p90/p95/p99, and exact
  pair count.
- SSIM uses `channel_axis=-1`, `data_range=1`; record scikit-image version,
  window settings, mean/median/p90/p95/p99, and per-class means.
- LPIPS uses local `lpips`, AlexNet, RGB normalized to `[-1,1]`; report
  mean/median/p90/p95/p99/max, per-class means, worst 20 classes, worst 100
  samples, and NaN/Inf counts.

The per-sample CSV includes sample/class/seed, all three scores, and both
paths. If LPIPS or its already-local weights are unavailable, the LPIPS test
is explicitly skipped/failed as configured; no alternative backbone and no
network download is allowed. The current PixelGen reference is blocked for
paired metrics; see `BASELINE_COMPATIBILITY_REPORT.md`.

## Performance

Single-image latency, common-batch throughput, and four-GPU 50K wall clock are
different metrics and must be reported separately. Primary `speedup` uses
matched compile modes and median per-image latency. See
`COMPILE_COMPATIBILITY.md` and `MEMORY_REPORT.md`.

