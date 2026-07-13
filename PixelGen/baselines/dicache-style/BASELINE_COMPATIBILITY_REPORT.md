# PixelGen Full-reference compatibility

## Decision

`PAIRED_METRICS_BLOCKED` for the main DiCache protocol.

The current PixelGen Full reference used real batch 4 and a continuous distributed RNG path. The main DiCache template uses real batch 1 (combined CFG batch 2), and the old output does not establish an immutable sample-ID/class/seed/initial-noise mapping that proves replay after regrouping. Filename or array position is not evidence of the same Gaussian input.

| Strict-pair requirement | Existing Full evidence | Status |
|---|---|---|
| sample/class identity | potentially recoverable only by separate audit | unproven |
| exact per-sample Gaussian noise | continuous batch-4 stream; no immutable replay binding | blocked |
| fixed batch group | batch 4 versus main batch 1 | mismatch |
| checkpoint and EMA | expected, must be pinned mechanically | unproven |
| exact Heun, 50 steps, CFG/timeshift | recorded in legacy command but needs run-manifest comparison | unproven |
| dtype and compile path | must match local `instrumented_full` denominator | unproven |
| combined CFG order and postprocessing | expected, not a complete strict-pair proof | unproven |

After validating its own completeness, the old Full output may be used for unpaired FID, sFID, IS, precision, and recall. It must not be used for PSNR, SSIM, or LPIPS against new DiCache output, and files must not be paired post hoc by numeric order.

A new matched Full must be generated from the same immutable batch-1 manifest using `instrumented_full`, then compared to `dicache`. Primary speedup also uses `instrumented_full` with identical local split-body path and compile mode; an upstream-compiled Full may be reported separately but is not the main denominator.

