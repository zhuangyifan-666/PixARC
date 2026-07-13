# Notice and provenance

This directory contains an **unofficial SpeCa-style port for PixelGen**. It is not authored, endorsed, or maintained by the SpeCa, Cache4Diffusion, TaylorSeer, PixelGen, or JiT authors.

## Local sources

- PixARC integration commit: `f15b77ac684d7254fde1db4b001d728b11da6550`.
- Cache4Diffusion/SpeCa reference commit: `91a1949fcc88acab46547f0b5f295f5de2df2870`.
- SpeCa references: `../../../baselines/Cache4Diffusion/dit/speca-dit/models.py`, `cache_functions/cal_type.py`, `cache_functions/cache_init.py`, `taylor_utils/__init__.py`, `sample.py`, `sample_ddp.py`, and local diffusion files.
- TaylorSeer reference commit: `704ee98c74f7f04da443daa3c0aa2cc7803d86e3`.
- Vendored PixelGen identity: PixARC tree `3043acf90f255a264f1445bda9ea8d468ba91a58`.
- Vendored JiT identity used for cross-port checks: PixARC tree `d697163e4899e279a3c969d429832efecc9da115`.

## Licensing facts found locally

- `../../../baselines/TaylorSeer/LICENSE` is GNU GPL version 3.
- `../../../third-party/JiT/LICENSE` is MIT and names Tianhong Li (2025).
- No `LICENSE`/`COPYING` was found at the Cache4Diffusion clone root, despite source references to a root license.
- No license file was found at the vendored PixelGen root.
- No license file was found at the PixARC root.

Missing license text is not permission to redistribute. Distribution of this port, combined works, checkpoints, outputs, or evaluators requires an independent review of Cache4Diffusion/SpeCa, PixelGen, PixARC, TaylorSeer GPL-3.0, JiT MIT, model/checkpoint, dataset, and evaluator obligations. This notice grants no rights and is not legal advice.

## Reimplementation statement

The core is a minimal clean reimplementation from audited formulas/control flow; no whole `models.py`, `cal_type.py`, or diffusion file was copied. PixelGen components are inherited/composed so upstream state-dict keys and APIs remain stable. Local sibling baseline interfaces informed self-contained manifest, metrics, metadata, image-I/O, and timing code; sibling directories are unchanged and not runtime imports.

The duplicated JiT/PixelGen common core declares `COMMON_CORE_VERSION="speca-core-v1"` (tooling artifact version `pixarc-speca-style-v1`). `scripts/compare_common_tool_interfaces.py` records SHA-256 values and public APIs for the duplicated contract files; neither port imports the other at runtime.

Explicit semantic-equivalent optimizations are depth-relative `verify_layer=-1` instead of hard-coded 27 and cloning only when a selected check needs the speculative-prefix input. PixelGen's combined `[unconditional, conditional]` order, one `2B` forward, metric-only exact branch, no rollback, and fresh head are unchanged.

Main is `released_code_faithful`. Any future paper-style reject/rollback path must be separately named, authorized, and evaluated.
