# Local-source audit

This port was derived from local sources. No dependency was installed and no network command was used.

## Recorded identities

| Component | Identity | Treatment |
|---|---|---|
| PixARC | `3377371d9b72fcdfa9407df3eca66759c91d2901` | unrelated worktree changes preserved |
| DiCache | `fdbe20b669c9174bbed5ec994de073fd881c8010` | read-only local clone |
| vendored JiT | tree `d697163e4899e279a3c969d429832efecc9da115` | read-only subtree, not an independent Git checkout |

The audit read the sibling SeaCache, TaylorSeer, and SpeCa integrations only for repository conventions such as immutable manifests, output validation, launch locks, and metric reports. None was modified by this JiT port.

## DiCache findings used

- FLUX caches only image hidden states, probes the first `probe_depth` dual blocks, uses batch-global `delta_x`/`delta_y`, strict `<`, inclusive warmup, optional final Full, and gamma clipped to `[1,1.5]`.
- An eligible refresh resumes from the already-computed probe state; it does not repeat the prefix.
- Only exact Full outputs create synchronized full/probe residual anchors. Reuse updates adjacent-call observations but not exact anchors.
- WAN performs separate conditional/unconditional calls with two independent alternating state slots, no forced final Full, and gamma `[1,2]`.
- Hunyuan uses one combined CFG batch/global state, non-strict `<=`, no forced final Full, and gamma `[1,1.5]`.
- The released FLUX demo’s `0.4` threshold is model/demo-specific evidence, not a valid JiT operating point.

`scripts/compare_official_core_parity.py` pins the local DiCache commit, checks the audited FLUX expressions still exist, and compares deterministic CPU fixtures. The latest run reported global maximum absolute error `0.0`, maximum relative error `0.0`, and zero warmup mismatches. This is formula parity, not GPU/model-output parity.

## JiT findings used

- JiT-B/16 has 12 blocks, hidden size 768, 256 image tokens, 32 prepended context tokens inserted before block 4, and separate RoPE objects before/after insertion.
- The image body input is `x_embedder(x) + pos_embed`. The body output is the image-token suffix after all blocks and before `final_layer`.
- `final_layer` and `unpatchify` always use current conditioning and must run fresh.
- Upstream CFG performs conditional then unconditional as two separate model calls. This requires isolated DiCache streams.
- Exact 50-step Heun evaluates 99 NFEs and therefore 198 JiT network forwards. Repeated continuous times are distinct observations.
- `VisionRotaryEmbeddingFast` allocates CUDA state during real model construction, so CPU verification imports but never instantiates the real JiT model.

## Result boundary

CPU formulas, state transitions, probe/resume structure, exact-Heun sequencing, deterministic manifests, metric guards, and entrypoint preflight are implemented and tested. Checkpoint/model numerical parity, CUDA execution, compilation behavior, latency, quality, peak GPU memory, and 50K completion remain deferred. No speedup or selected threshold is claimed.
