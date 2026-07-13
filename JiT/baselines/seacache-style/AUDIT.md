# JiT and SeaCache implementation audit

Audit date: 2026-07-13 UTC. The audit was read-only: no model was loaded, no
CUDA inference was started, and no active process was signalled or attached.

## Revisions and worktrees

| Source | Exact local revision | Notes |
|---|---|---|
| PixARC | `d54c1e26768d80bf7c067f50e28868cdbf59d431` | `third-party/JiT` resolves to this repository, not an independent Git worktree |
| SeaCache | `8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2` | FLUX implementation files originated at `0a91c2b...` |
| JiT model/denoiser import | `6db780076d444c98ea1e97dc2adbbb5a407a6724` | Imported snapshot provenance; not an upstream JiT commit identifier |

At the beginning of the audit, `git status --short --untracked-files=no` showed
no tracked changes in PixARC or the nested SeaCache clone. That command
intentionally did not claim the absence of untracked or ignored files.

## Active-reference snapshot

At the audit snapshot:

- PixelGen reference generation was **ACTIVE**. Its launcher PID was `385579`,
  with four Lightning/DDP prediction ranks and per-rank prediction batch 4.
- JiT reference generation was **SCHEDULED**, not active. PID `406161` was the
  shell scheduler waiting for PixelGen to finish.
- The scheduled JiT command is recorded in the workspace-level
  `$ROOT/../run_jit_after_pixelgen.sh:77-96`.
- No JiT 50K output existed yet at audit time, so no JiT quality metric or
  completion claim is made here.

The scheduled JiT entry is `$ROOT/third-party/JiT/main_jit.py`; it requests
four ranks, per-rank batch 32, seed 0, JiT-B/16 at 256, EMA1 from
`$ROOT/JiT/checkpoints/JiT-B-16-256/checkpoint-last.pth`, Heun 50, CFG 3.0,
interval `[0.1,1.0]`, and output directory
`$ROOT/JiT/checkpoints/JiT-B-16-256/heun-steps50-cfg3.0-interval0.1-1.0-image50000-res256`.

This is a time-stamped observation, not a live monitor.

## SeaCache local behavior

The three local `util_seacache.py` copies are byte-identical. The FLUX path is
the implementation reference.

### SEA filter

`FLUX/util_seacache.py:23-109`:

- makes the input contiguous and converts FFT computation to float32;
- constructs one Wiener gain per selected axis and multiplies the gains;
- uses `Sx0=power_const/(abs(f)**power_exp+eps)` and
  `H=(a*Sx0)/(a*a*Sx0+b*b+eps)` per axis;
- uses full `fftn/ifftn` in the audited FLUX call;
- normalizes the complete separable filter by its mean;
- converts the result back to the original dtype.

FLUX reshapes its image probe to `[B,H,W,C]` and calls with
`dims=(-2,-3)`, so FFT covers W and H but not batch or channel
(`FLUX/seacache_generate.py:98-112`).

### Coefficients and distance

`FLUX/util_seacache.py:112-166` maps flow sigma to `a=1-sigma`,
`b=sigma` after clamping sigma into `[1e-6,1-1e-6]`. JiT actually trains with
`z=t*x+(1-t)*noise` (`third-party/JiT/denoiser.py:52-59`), so the direct JiT
mapping is `a=t`, `b=1-t`, with equivalent endpoint clamping. A nonuniform t
inside one batch must fail rather than silently average.

`rel_l1(current, previous)` is
`mean(abs(current-previous))/(mean(abs(previous))+eps)` over the entire tensor
and returns a Python float (`FLUX/util_seacache.py:196-199`). The official gate
is therefore batch-global.

### Gate/update order

From `FLUX/seacache_generate.py:86-130`:

1. First call, final call, or missing previous probe forces full and resets the
   accumulator.
2. Ordinary calls SEA-filter the current probe, add relative L1 to the
   accumulator, then use strict `< threshold` for reuse.
3. A refresh at `>= threshold` clears the accumulator before the previous
   probe is written.
4. The first forced-full call stores the **raw** probe. Ordinary calls store
   the filtered probe. The final forced-full call again stores raw probe.
5. Reuse computes `current_body_input + previous_residual`.
6. Full computes and stores `body_output-body_input`.
7. Final norm/projection are always fresh.

The official gate does not include missing residual in its initial forced-full
condition. The execution branch nevertheless falls back to exact body when a
reuse decision has no residual; it leaves the already-updated accumulator
unchanged.

### Synchronization

The audited code contains intentional CPU/scalar synchronization: `.item()`
for grid dimensions, `.detach().cpu()` in `rel_l1`, tensor-valued Python
conditions during filter normalization, and possible scheduler tensor-to-float
conversion. Faithful latency includes these costs.

## JiT local behavior

### Transformer and context

- Patch embedding: `third-party/JiT/model_jit.py:17-37`.
- Fixed positional embedding and body entry: `:243-245,343-344`.
- `JiTBlock` modulation and exact attention input: `:183-202`.
- JiT-B/16 has depth 12, patch 16, context length 32, insertion index 4:
  `:362-364`.
- Class context is prepended at the configured block and removed before the
  head: `:346-357`.
- Final layer uses fresh RMSNorm/AdaLN/linear projection, then unpatchify:
  `:162-180,317-329,354-357`.

The faithful probe is exactly:

```python
shift_msa, scale_msa, *_ = blocks[0].adaLN_modulation(c).chunk(6, dim=-1)
probe_raw = modulate(blocks[0].norm1(body_input), shift_msa, scale_msa)
```

For JiT-B/16 the first four blocks precede context insertion, so this probe is
an image-token grid. The cached residual begins at image tokens after patch and
position embedding and ends after all blocks and context removal. Context
tokens are never part of the cached residual; the final head remains fresh.

### CFG and solver calls

`third-party/JiT/denoiser.py:91-105` executes conditional forward first and
unconditional forward second. They cannot share cache state.

For `steps=N`, the local Heun loop performs `N-1` predictor/corrector pairs and
one final Euler call (`denoiser.py:81-88,114-122`). Therefore:

```text
calls per stream = 2*(N-1)+1
50-step Heun     = 99 cond + 99 uncond = 198 net forwards
```

The final network evaluation is at t=0.98, followed by integration to t=1.0.

### Checkpoint and EMA

`main_jit.py:203-218` loads `checkpoint["model"]`, `model_ema1`, and
`model_ema2`. Evaluation replaces model parameters with EMA1 and later restores
base parameters (`engine_jit.py:144-151,189-192`). Runtime cache state must not
be a parameter, buffer, module, or checkpoint key.

### CPU-import limitation

`third-party/JiT/util/model_util.py:118-134` constructs rotary tensors with
hard-coded `.cuda()` during model instantiation. Adapter imports are CPU-safe,
but CPU tests use mock blocks instead of constructing a real JiT model.

## Current JiT RNG and filenames

The scheduled run uses four ranks, per-rank batch 32, base seed 0, Heun 50,
CFG 3.0, interval `[0.1,1.0]`, BF16 autocast, and EMA1. Main seeds each rank
with `base_seed+rank` and reseeds immediately before evaluation
(`main_jit.py:128-131,225-230`). Each rank then consumes a continuous CUDA RNG
stream through one full-batch `torch.randn` per generation iteration
(`denoiser.py:68-72`).

Labels are class-major, exactly 50 per class. Numeric filenames are
`00000.png` through `49999.png`; consequently class is recoverable as
`sample_id//50` (`engine_jit.py:153-185`). No PNG/sidecar stores class, seed,
rank RNG offset, or initial-noise hash. The final padded distributed batch also
draws noise for unsaved positions.

The current/scheduled reference is therefore not self-describing or
independently seeded per image. See `BASELINE_COMPATIBILITY_REPORT.md`.

## License finding

JiT contains an MIT license. The audited SeaCache clone contains no license
file. See `NOTICE.md`; no SeaCache license is inferred.
