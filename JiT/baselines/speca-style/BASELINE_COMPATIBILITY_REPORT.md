# JiT Full-reference compatibility

> Protocol update (2026-07-14): active SpeCa and matched Full runs use batch 32. Legacy outputs remain blocked unless immutable batch-32 manifest/noise/group identities match; the former batch-32-versus-batch-1 mismatch is no longer the design target.

## Decision

**`PAIRED_METRICS_BLOCKED`** for the registered SpeCa main protocol.

The existing/planned JiT Full reference uses real batch 32, while released-code-faithful SpeCa is registered at real batch 1 per GPU process. Batch grouping is part of the random/noise replay contract and also changes a batch-global SpeCa decision. Matching PNG names or class order cannot prove that the initial Gaussian tensor and every RNG consumption are identical. No auditable immutable batch-1 manifest/run-metadata pair currently establishes all strict conditions.

This classification is intentionally conservative. It applies even if filenames, seeds written elsewhere, checkpoint, sampler, and nominal generation settings look similar.

## Strict-pair checklist

| Requirement | Existing Full evidence versus main SpeCa | Status |
|---|---|---|
| sample ID / class ID | recoverable only after manifest/run audit | unproven |
| per-sample initial Gaussian noise | batch-32 versus batch-1 replay not proven | blocked |
| checkpoint / model variant | expected but must be pinned in run metadata | unproven |
| EMA | expected but must be pinned | unproven |
| sampler / steps / CFG / timeshift | expected but must be compared mechanically | unproven |
| dtype / compile mode | must match the selected matched-Full path | unproven |
| batch grouping | 32 versus 1 | mismatch |
| RNG consumption / resume order | no immutable replay proof | blocked |
| postprocessing | must be matched and recorded | unproven |

## Permitted use of the old Full run

After its own integrity validation, it may be used as an unpaired distribution reference for FID, sFID, IS, precision, and recall. It must not be used for PSNR, SSIM, or LPIPS against main SpeCa, and files must not be paired by numeric filename alone.

## Required matched Full

When GPU work is separately authorized, build an immutable batch-1 manifest and run `instrumented_full` and `speca` with the same checkpoint, EMA, sampler, steps, CFG, timeshift, dtype, compile mode, sample grouping, explicit per-sample noise, and postprocessing. Preserve both resolved configs, manifest and sidecar hashes, checkpoint identity, run metadata, and output validation reports. Only then may strict paired evaluation proceed.

The primary speedup denominator is matched `instrumented_full`, because it uses the same local block path and compile regime as SpeCa. The existing wall-clock Full run is not automatically a valid latency denominator.
