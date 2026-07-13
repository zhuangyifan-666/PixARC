# JiT TaylorSeer port audit

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
was active under launcher PID `385579`, using four DDP devices and per-rank
prediction batch 4. JiT was not yet running; PID `406161` was a shell waiting
to launch it after PixelGen. This is a timestamped observation, not a monitor.

The queued JiT command requests JiT-B/16 at 256, checkpoint
`JiT/checkpoints/JiT-B-16-256/checkpoint-last.pth`, EMA1, four ranks, per-rank
batch 32, base seed 0, BF16, Heun 50, CFG 3.0 on `[0.1,1.0]`, noise scale 1,
and 50,000 images. No JiT output existed at the audit snapshot.

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

## JiT model audit

The source of truth is `third-party/JiT/model_jit.py` and `denoiser.py`:

- JiT-B/16 is depth 12, hidden size 768, 12 heads, patch 16;
- image token count is 256 at 256x256;
- 32 class-context tokens are prepended immediately before block 4 and remain
  through blocks 4--11, then are removed once before the final head;
- blocks 0--3 use `feat_rope`; blocks 4--11 use `feat_rope_incontext`;
- `Attention.forward` includes the output projection/projection dropout;
- `SwiGLUFFN.forward` includes its final `w3` projection;
- both are therefore the correct gate-pre forecast targets;
- block AdaLN is fresh per call; final RMSNorm/AdaLN/linear/unpatchify are
  fresh;
- the network predicts x; the denoiser converts it to velocity using
  `(x_prediction-z)/(1-t)` with `t_eps` handling;
- `_forward_sample` calls conditional then unconditional and applies CFG;
- Heun 50 performs 99 NFE, hence 198 network forwards;
- `main_jit.py` loads base, `model_ema1`, and `model_ema2`; evaluation uses
  EMA1 in the queued reference;
- upstream block/final forwards use `torch.compile` decorators, which matters
  for a fair latency denominator.

The local adapter inherits upstream modules and preserves parameter names. It
does not monkey-patch upstream classes.

## RNG/output audit

The queued upstream run seeds each rank with `base_seed+rank` and consumes a
continuous CUDA RNG stream through full-batch `torch.randn`. Labels and
filenames are class-major (`sample_id//50`), but PNGs do not store per-image
seed, rank RNG offset, batch group, or noise hash. The final padded batch also
affects RNG consumption. This is why current-reference pairing is conditional,
not established merely by matching filenames.

## Safety result

Only source/config/text reads and CPU-safe repository operations were used.
No new CUDA workload was started. No current output directory was scanned
recursively, hashed as a large checkpoint/output, or modified.

