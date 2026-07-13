# Notice and provenance

This directory implements an **unofficial DiCache-style port for PixelGen**. It is not an official DiCache or PixelGen release and is not endorsed by either project.

- DiCache local revision inspected: `fdbe20b669c9174bbed5ec994de073fd881c8010`.
- PixARC/PixelGen identity: PixARC `3377371d9b72fcdfa9407df3eca66759c91d2901`, vendored PixelGen subtree tree `3043acf90f255a264f1445bda9ea8d468ba91a58`.
- DiCache behavior references: `FLUX/run_flux_dicache.py`, `WAN2.1/run_wan_dicache.py` plus its `wan/` changes, and `HunyuanVideo/run_hunyuanvideo_dicache.py` plus its `hyvideo/` changes.
- PixelGen interface references: `src/models/transformer/JiT.py`, `src/diffusion/flow_matching/sampling.py`, Lightning/EMA integration, and existing local baseline launch/output code.

No whole official source file or substantial code fragment was copied into this port. The small formulas and branch behavior were independently re-expressed from the paper and audited runtime behavior. Generic manifest, metric, and metadata utilities are self-contained counterparts shared by byte/API contract with the local JiT port; there is no cross-directory runtime import.

The bounded `deque(maxlen=2)` anchor storage is a documented semantic optimization: released backends read only the last two exact anchor pairs. Explicit stream IDs, instance-owned runtime, complete reset, fail-closed manifests, and non-monkey-patched subclasses are integration safety choices rather than claims about official implementation structure.

At audit time, no top-level `LICENSE`, `COPYING`, or `NOTICE` was found in the local PixARC root or DiCache clone, and no `LICENSE`/`COPYING` was found at the vendored PixelGen root. Absence is not permission and this notice makes no legal conclusion. Any redistribution or publication must independently establish the applicable upstream and nested-project terms and preserve existing notices.

