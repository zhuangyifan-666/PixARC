# Notice and provenance

This directory is an **unofficial DiCache-style port for JiT**. It is not an official DiCache or JiT release and is not endorsed by either project.

- DiCache local revision inspected: `fdbe20b669c9174bbed5ec994de073fd881c8010`.
- PixARC identity inspected: `3377371d9b72fcdfa9407df3eca66759c91d2901`.
- Vendored JiT subtree tree: `d697163e4899e279a3c969d429832efecc9da115`.
- DiCache sources: `FLUX/run_flux_dicache.py`, `WAN2.1/run_wan_dicache.py`, and `HunyuanVideo/run_hunyuanvideo_dicache.py` plus their local model changes.
- JiT sources: `third-party/JiT/model_jit.py`, `denoiser.py`, and local baseline launcher/evaluation conventions.

No upstream file was modified. The adapter subclasses the vendored JiT model and re-expresses the audited formulas with instance-owned state; it does not install the released global class monkey patch. Generic manifest, metric, metadata, and core-formula files follow the same self-contained byte/API contract as the sibling PixelGen port, with no cross-directory runtime import.

The bounded two-anchor window is an intentional semantic optimization: released FLUX appends history but DCTA reads only the latest two synchronized exact anchor pairs. Explicit stream IDs, full lifecycle reset, strict resume identity checks, and fail-closed launch guards are integration safety additions, not claims about official implementation structure.

At audit time no top-level license file was found in the local PixARC or DiCache roots. Absence is not permission and this notice makes no legal conclusion. Redistribution must independently establish all applicable upstream and nested-project terms and preserve their notices.
