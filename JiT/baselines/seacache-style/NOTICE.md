# Notice and provenance

This directory contains an **unofficial SeaCache-style port for JiT**. It is
not part of, endorsed by, or supported by the SeaCache or JiT upstream
projects.

## Audited sources

- Local SeaCache clone: `$ROOT/baselines/SeaCache`
- SeaCache commit: `8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2`
- Implementation files consulted:
  - `FLUX/util_seacache.py`
  - `FLUX/seacache_generate.py`
- Those two implementation files last changed at SeaCache commit
  `0a91c2bacad77b56e4cce4183a0254ac40c0739c`.
- Local JiT snapshot: `$ROOT/third-party/JiT`
- Local PixARC commit containing that snapshot:
  `d54c1e26768d80bf7c067f50e28868cdbf59d431`
- JiT model and denoiser were originally imported into PixARC at
  `6db780076d444c98ea1e97dc2adbbb5a407a6724`.

The SeaCache filter equation, gate order, and residual boundary were
independently reimplemented in small local modules. No complete upstream model
file was copied.

## License status

The audited JiT snapshot includes an MIT License, copyright 2025 Tianhong Li.
The copyright and permission notice in `$ROOT/third-party/JiT/LICENSE` applies
to reused JiT material.

**The audited SeaCache clone contains no `LICENSE`, `COPYING`, or `NOTICE`
file at the commit above.** This provenance notice is not a license grant. Do
not claim that SeaCache is MIT, Apache, or otherwise licensed based on this
repository. Redistribution or publication of code derived from SeaCache
requires separate legal/license review. The implementation here deliberately
limits reuse to a documented mathematical formula and observed runtime
semantics.

## Scope of modification

This port adds runtime cache state, a body-level residual controller, JiT
subclasses/wrappers, deterministic manifests, evaluation helpers, and deferred
launch/benchmark tooling. It does not modify checkpoints or upstream training
code, and it does not claim official SeaCache support for JiT.

