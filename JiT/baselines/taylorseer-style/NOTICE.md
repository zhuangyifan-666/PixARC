# Sources, licenses, and provenance

This directory is an **unofficial TaylorSeer-style port for JiT**. It is not
published, endorsed, or supported by the TaylorSeer or JiT authors.

## Audited sources

- TaylorSeer commit `704ee98c74f7f04da443daa3c0aa2cc7803d86e3`:
  `TaylorSeer-DiT/models.py`, `taylor_utils/__init__.py`,
  `cache_functions/cal_type.py`, `cache_functions/cache_init.py`, `sample.py`,
  and `sample_ddp.py`. The clone includes GNU GPL version 3.
- Cache4Diffusion commit `91a1949fcc88acab46547f0b5f295f5de2df2870`
  was inspected for TaylorSeer, TaylorSeer-Lite, state-management, and
  multi-GPU engineering patterns. No repository-level LICENSE was found.
- JiT is vendored in PixARC commit
  `f15b77ac684d7254fde1db4b001d728b11da6550`, tree
  `d697163e4899e279a3c969d429832efecc9da115`. Relevant files include
  `model_jit.py`, `denoiser.py`, `main_jit.py`, and `engine_jit.py`. JiT carries
  the MIT License, Copyright (c) 2025 Tianhong Li.
- PixelGen is vendored in the same PixARC revision, tree
  `3043acf90f255a264f1445bda9ea8d468ba91a58`; no LICENSE was found. It was
  audited to keep the two ports' common protocol consistent.
- PixARC itself had no repository-level LICENSE at the audit snapshot.

## What was implemented or adapted

The Taylor finite-difference recursion, fixed schedule, and branch boundaries
were reimplemented minimally from observed behavior and formulas; the official
TaylorSeer `models.py` was not copied wholesale. The adapter inherits the
upstream JiT class and calls its modules so parameter names/checkpoint keys are
unchanged. Taylor state is runtime-only.

Manifest, atomic-output, metadata, paired/distribution metric, and latency
interfaces were adapted from the existing read-only PixARC
`JiT/baselines/seacache-style` integration to preserve local data protocols.
They do not import that runtime. The corresponding common utilities are
duplicated locally so JiT and PixelGen ports have no cross-directory runtime
dependency. No code was copied from Cache4Diffusion into the primary method.

Code-copy statement: no TaylorSeer or Cache4Diffusion source snippet was
copied verbatim into the model algorithm. Several data/CLI plumbing functions
were copied and adapted from the sibling PixARC SeaCache-style integration;
no copyright header was removed. That local source has no separate license
notice and PixARC has no repository-level license, so this same-repository use
does not establish external redistribution rights.

This implementation intentionally differs from the official clone by removing
dead per-step dictionaries and 28-layer/49-step hard-codes, mapping Heun calls
to monotone NFE coordinates, adding trajectory validation, and supporting JiT
conditional/unconditional stream isolation. These engineering changes do not
authorize calling the result an official port.

## Redistribution note

Absence of a LICENSE is not permission. Anyone redistributing or publishing
this port must review TaylorSeer's GPL-3.0 obligations, preserve JiT's MIT
notice where applicable, and obtain/clarify rights for PixARC,
Cache4Diffusion-derived context, PixelGen, checkpoints, datasets, and evaluator
assets. This notice records provenance and is not legal advice.
