# Notice and provenance

This directory contains an **unofficial SpeCa-style port for JiT**. It is not authored, endorsed, or maintained by the SpeCa, Cache4Diffusion, TaylorSeer, or JiT authors.

## Local sources

- PixARC integration commit: `f15b77ac684d7254fde1db4b001d728b11da6550`.
- Cache4Diffusion/SpeCa reference commit: `91a1949fcc88acab46547f0b5f295f5de2df2870`.
- SpeCa behavior references: `../../../baselines/Cache4Diffusion/dit/speca-dit/models.py`, `cache_functions/cal_type.py`, `cache_functions/cache_init.py`, `taylor_utils/__init__.py`, `sample.py`, `sample_ddp.py`, and local diffusion files.
- TaylorSeer reference commit: `704ee98c74f7f04da443daa3c0aa2cc7803d86e3`.
- Vendored JiT identity: PixARC tree `d697163e4899e279a3c969d429832efecc9da115`.
- Vendored PixelGen identity used for cross-port API checks: PixARC tree `3043acf90f255a264f1445bda9ea8d468ba91a58`.

## Licensing facts found locally

- `../../../baselines/TaylorSeer/LICENSE` is GNU GPL version 3.
- `../../../third-party/JiT/LICENSE` is MIT and names Tianhong Li (2025).
- No `LICENSE` or `COPYING` file was found at the Cache4Diffusion clone root in this checkout, despite source-file references to a root license.
- No license file was found at the vendored PixelGen root.
- No license file was found at the PixARC root.

Absence of a local license file is not permission to redistribute. Anyone distributing this port, derived artifacts, or a combined work must resolve Cache4Diffusion/SpeCa, PixelGen, PixARC, TaylorSeer GPL-3.0, model/checkpoint, dataset, and evaluator obligations independently. This notice is factual provenance, not legal advice and not a new license grant.

## Reimplementation statement

The SpeCa core is a clean, minimal reimplementation from observed formulas and control-flow behavior. Whole upstream files such as `models.py`, `cal_type.py`, and diffusion implementations were not copied. The port reuses upstream JiT modules by inheritance/composition and preserves their parameter/state-dict names. Manifest, metric, metadata, image-I/O, and latency interfaces were adapted from local sibling baselines where appropriate; those baseline directories were not modified and are not runtime imports.

The duplicated JiT/PixelGen common core declares `COMMON_CORE_VERSION="speca-core-v1"` (tooling artifact version `pixarc-speca-style-v1`). `scripts/compare_common_tool_interfaces.py` records SHA-256 values and public APIs for the duplicated contract files; neither port imports the other at runtime.

Two semantic-equivalent engineering changes are explicit: `verify_layer=-1` resolves the actual final Transformer block rather than hard-coding layer 27, and tensors are cloned only when the selected verifier actually needs them rather than unconditionally at every block. Neither change permits exact-verifier writeback or current-NFE rollback.

The main mode is `released_code_faithful`. Any future paper-style reject/rollback implementation must use an explicit experimental mode and cannot be reported as this baseline.
