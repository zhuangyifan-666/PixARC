# PixelGen TaylorSeer port audit

Audit snapshot: 2026-07-13 UTC. The audit was read-only. No model was loaded,
no CUDA inference or benchmark was started, and no active process was
signalled, paused, attached, or debugged.

## Local revisions and ownership

| Source | Local identity | Finding |
|---|---|---|
| PixARC | `f15b77ac684d7254fde1db4b001d728b11da6550` | repository revision at audit |
| TaylorSeer | `704ee98c74f7f04da443daa3c0aa2cc7803d86e3` | independent local clone; GPL-3.0 |
| Cache4Diffusion | `91a1949fcc88acab46547f0b5f295f5de2df2870` | independent local clone; no LICENSE found |
| `third-party/JiT` | PixARC revision above; tree `d697163e4899e279a3c969d429832efecc9da115` | vendored tree, not an independent Git worktree; MIT |
| `third-party/PixelGen` | PixARC revision above; tree `3043acf90f255a264f1445bda9ea8d468ba91a58` | vendored tree; no LICENSE found |

The tracked worktrees were clean before creating these new untracked port
files. The upstream clones, vendored trees, and both existing `seacache-style`
directories were treated as read-only.

## Active-reference snapshot

At the single process/GPU audit snapshot, PixelGen Full reference generation
was active under launcher PID `385579` from `third-party/PixelGen/main.py`.
The sanitized command used `configs_c2i/PixelGen_XL.yaml`, checkpoint
`PixelGen/checkpoints/PixelGen_XL_160ep.ckpt`, four DDP devices, and per-rank
batch 4. JiT was queued behind it and was not active. This is a timestamped
observation, not a live monitor.

The active configuration is PixelGen-XL at 256, seed_everything 1234, EMA
prediction, exact Heun 50, guidance 2.25 on `(0.1,0.9]`, timeshift 2.0,
BF16-mixed precision, and 99 combined CFG forwards. The upstream callback
produces a compressed `output.npz` at completion and only a small preview PNG
set; this port did not scan or modify that output.

## Official TaylorSeer findings

Executable source, not the README alone, confirms:

- `cal_type` chooses one action before the model block loop;
- every block keeps separate attention/MLP finite-difference dictionaries;
- each target is the complete branch output before the current gate;
- Taylor calls recompute AdaLN/gates, skip both norms and expensive branches;
- only Full calls change anchors or factors;
- the final head/unpatchify are always exact;
- differences use signed descending discrete-step gaps;
- `max_order=K` stores at most K+1 tensors;
- `first_enhance=2` forces the first two Full;
- `last_steps` is dead in the action test, so there is no faithful final-Full
  rule;
- `activated_steps=[49]` and depth 28 are source-specific hard-codes;
- only `cache[-1]` holds factor tensors; per-step dictionaries are dead
  storage.

Full detail and interval counts are in `OFFICIAL_BEHAVIOR.md`.

## Cache4Diffusion finding

Cache4Diffusion was inspected only as an auxiliary engineering reference.
Its original-style Flux double-transformer path forecasts branch-level values,
while its single-transformer/TaylorSeer-Lite paths forecast broader outputs.
Lite or whole-body forecasting is not semantically interchangeable with the
official DiT implementation and is excluded from the primary port. SpeCa's
adaptive verification is also out of scope. No Cache4Diffusion source file was
copied into this port.

## PixelGen model/sampler audit

The source of truth is `third-party/PixelGen/src/models/transformer/JiT.py`,
`src/diffusion/flow_matching/sampling.py`, and `src/lightning_model.py`:

- PixelGen-XL is depth 28, hidden size 1152, 16 heads, patch 16;
- 32 context tokens are inserted immediately before block 8, retained through
  the remaining blocks, and removed once before the final head;
- blocks before/after insertion use `feat_rope`/`feat_rope_incontext`;
- the model interface is `forward(x,t,y,return_layer=None,return_last=False)`
  and its tuple return forms must remain unchanged;
- attention and complete MLP returns are the gate-pre forecast targets;
- exact Heun concatenates `[x,x]` and
  `[uncondition,condition]` into one 2B model call;
- `exact_henu=true`, 50 steps executes 50 predictors plus 49 correctors, or
  99 combined forwards;
- guidance uses `t>0.1 and t<=0.9`; timeshift is 2.0;
- Lightning deep-copies the denoiser to `ema_denoiser`, compiles both during
  configuration, and predicts with EMA unless `eval_original_model` is true.

The local adapter therefore uses one `combined_cfg` history and makes runtime
deepcopy non-sharing and empty. Diagnostic intermediate returns force Full.

## RNG/output audit

The active upstream RandomN dataset/DDP path does not archive a stable mapping
of sample ID, class ID, per-image seed, rank RNG offset, batch group, and
initial-noise hash. The compressed NPZ alone cannot prove which Gaussian noise
belongs to each array position. Preview filenames also do not establish
pairing. A manifest-backed Full rerun is required for strict paired metrics.

## Safety result

Only source/config/text reads and CPU-safe repository operations were used.
No new CUDA workload was started. No current output directory was scanned
recursively, hashed as a large checkpoint/output, or modified.

