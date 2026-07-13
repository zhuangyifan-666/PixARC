# PixelGen Full-reference compatibility

## Decision

**`PAIRED_METRICS_BLOCKED`** for the registered SpeCa main protocol.

The current PixelGen Full reference uses real batch 4, while released-code-faithful main SpeCa uses real batch 1 per GPU process (effective combined CFG batch 2). Batch grouping is part of the deterministic-noise/RNG contract, and no immutable run metadata proves that regrouping from 4 to 1 reproduces identical initial Gaussian tensors and all downstream conditions. Same PNG names or nominal seeds are insufficient.

## Strict-pair checklist

| Requirement | Existing Full evidence versus main SpeCa | Status |
|---|---|---|
| sample ID / class ID | recoverable only after manifest/run audit | unproven |
| initial Gaussian noise | batch-4 versus batch-1 replay not proven | blocked |
| checkpoint / EMA | expected but must be pinned | unproven |
| sampler / exact Heun / steps / CFG / timeshift | must be compared mechanically | unproven |
| dtype / compile mode | must match selected matched-Full path | unproven |
| `[unconditional, conditional]` layout | expected but must be recorded | unproven |
| batch grouping | 4 versus 1 | mismatch |
| RNG consumption / resume | no immutable replay proof | blocked |
| postprocessing | must match and be recorded | unproven |

## Permitted use

After its own integrity validation, the old Full output may support unpaired FID, sFID, IS, precision, and recall. It must not support PSNR, SSIM, or LPIPS against main SpeCa, and numeric filenames must not be forced into pairs.

## Required matched Full

Under later GPU authorization, generate an immutable batch-1 manifest and run `instrumented_full` and `speca` with identical checkpoint, EMA, exact-Heun sampler, steps, CFG/timeshift, dtype, compile mode, explicit per-sample noise, combined CFG order, postprocessing, and one-sample groups. Preserve config/manifest/checkpoint hashes and run metadata. Only a mechanically accepted pair may enter strict metrics.

Primary speedup uses matched `instrumented_full`, not the existing Full wall clock, because SpeCa runs the local split block path and both sides must share compile/path conditions.

