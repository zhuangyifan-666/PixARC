# Local-source audit

> Protocol update (2026-07-14): active PixelGen DiCache and matched Full runs use real batch 4/effective CFG batch 8. Historical batch-1 statements below describe the original port audit and are superseded for the primary experiment.

This is an unofficial, clean-room-style DiCache port for PixelGen. Executable behavior, code semantics, and paper/code gaps were derived from the local DiCache clone and its local paper/source artifacts. No network download or network shell command was used for implementation.

## Revisions and worktrees

| Component | Recorded identity | Status during audit |
|---|---:|---|
| PixARC | `3377371d9b72fcdfa9407df3eca66759c91d2901` | existing changes outside this directory were treated as user-owned |
| DiCache | `fdbe20b669c9174bbed5ec994de073fd881c8010` | clean and read-only |
| vendored PixelGen | PixARC revision above; subtree tree `3043acf90f255a264f1445bda9ea8d468ba91a58` | read-only; it is not a nested Git repository |

The audit also read the existing SeaCache-, TaylorSeer-, and SpeCa-style integration directories for launcher, manifest, output-safety, and report conventions. None was modified. At the beginning of the overall task, a single read-only process/GPU audit found active PixelGen DDP PID `385579` and a waiting JiT shell PID `406161`; the GPU driver query could not communicate. Those states were not re-polled, attached to, stopped, paused, or otherwise modified after the initial audit.

## DiCache files inspected

- `FLUX/run_flux_dicache.py`: body/probe boundaries, `delta_x`, `delta_y`, `delta_minus`, strict gate, inclusive warmup, last Full, DCTA, final fresh head, monkey patch and state cleanup.
- `WAN2.1/run_wan_dicache.py` and DiCache changes below `WAN2.1/wan/`: alternating conditional/unconditional forwards, two state slots, independent gates, gamma range, warmup, and reset.
- `HunyuanVideo/run_hunyuanvideo_dicache.py` and DiCache changes below `HunyuanVideo/hyvideo/`: combined CFG batch, one global state, non-strict boundary, DCTA, and partial cleanup.
- The local DiCache paper: Online Probe Profiling, Algorithm 1, DCTA equations and appendix, implementation details, and limitations.

Detailed answers are in [OFFICIAL_VARIANTS.md](OFFICIAL_VARIANTS.md) and [PAPER_CODE_GAP.md](PAPER_CODE_GAP.md).

## PixelGen files inspected

- `third-party/PixelGen/src/models/transformer/JiT.py`: `x_embedder + pos_embed` body input; context insertion at `in_context_start`; context-first token layout; image extraction before `final_layer`; `return_layer` and `return_last`.
- `third-party/PixelGen/src/diffusion/flow_matching/sampling.py`: exact-Heun predictor/corrector sequence; one `[unconditional, conditional]` 2B model call; corrector guidance interval uses `t_cur`.
- `third-party/PixelGen/src/lightning_model.py` and EMA callback code: denoiser deepcopy, EMA prediction selection, and compile lifecycle.
- PixelGen data and save paths: existing Full generation used a continuous/batched RNG path and does not establish immutable per-sample noise replay.

## Conclusions used by the port

The main executable profile is `released_code_faithful_image_profile`, configured as `flux_image_released`: image-only probe features, depth 1, `delta_y`, strict `<`, inclusive `ret_ratio=0.2`, last Full, two exact anchor pairs, first-order DCTA, gamma `[1,1.5]`, batch-global reduction, and no epsilon. PixelGen retains one combined CFG forward and one `combined_cfg` state over effective batch `2B`. Real batch 1 is the main protocol.

No threshold is claimed: `rel_l1_thresh` is deliberately unresolved in candidate templates and must be selected on disjoint 1K/8K data. No GPU correctness, latency, quality, or 50K result was produced in this implementation turn.
