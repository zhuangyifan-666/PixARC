# JiT Full-reference compatibility

> Protocol update (2026-07-14): active JiT DiCache and matched Full runs use real batch 32. Historical batch-1 instructions below are superseded; legacy outputs remain blocked unless their immutable batch-32 identities match.

## Decision

No existing JiT output is accepted for strict paired metrics until its immutable run manifest proves every pairing field. The safe default is `PAIRED_METRICS_BLOCKED`.

Strict pairing requires identical sample/class IDs, per-sample CPU Gaussian tensors, batch groups, checkpoint and EMA1, 50-step exact Heun sequence, CFG scale/interval/order, dtype, compile mode, image postprocessing, and code/source identities. Filename order or a legacy seed argument is insufficient evidence.

An existing Full tree may be used for unpaired FID/sFID/IS/precision/recall only after its completeness, class balance, image format, evaluator, and reference statistics validate. It must not be post-hoc paired for PSNR/SSIM/LPIPS.

Generate a new matched `instrumented_full` run from the same immutable batch-1 manifest as the resolved DiCache candidate. Primary latency also uses instrumented Full with the same local body/compile path. `upstream_full` is the numerical oracle and supplemental performance reference.

The compatibility checker compares run-manifest fields mechanically and refuses missing/mismatched evidence; it does not infer compatibility from directory names.
