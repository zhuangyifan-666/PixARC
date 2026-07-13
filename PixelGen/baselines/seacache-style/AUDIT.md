# PixelGen and SeaCache audit

Audit snapshot: 2026-07-13 UTC. PixARC revision:
`d54c1e26768d80bf7c067f50e28868cdbf59d431`; SeaCache revision:
`8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2`. PixelGen is part of the PixARC
worktree, not an independent repository.

## Active reference

PixelGen reference generation was **ACTIVE** from
`$ROOT/third-party/PixelGen/main.py`, using `configs_c2i/PixelGen_XL.yaml`,
`$ROOT/PixelGen/checkpoints/PixelGen_XL_160ep.ckpt`, four DDP devices and
per-rank batch 4. JiT was **SCHEDULED** behind it, not active. No process was
signalled, attached, paused, or modified.

The read-only snapshot identified launcher/rank-0 PID `385579` and worker PIDs
`385902`, `385903`, and `385904`, all rooted at
`$ROOT/third-party/PixelGen`. The sanitized command was:

```text
python main.py predict -c configs_c2i/PixelGen_XL.yaml \
  --ckpt_path=../../PixelGen/checkpoints/PixelGen_XL_160ep.ckpt \
  --data.train_dataset=null --trainer.devices=4 --trainer.strategy=ddp \
  --data.pred_batch_size=4
```

The config seed was 1234. Output targeted
`$ROOT/third-party/PixelGen/universal_pix_workdirs/exp_PixelGen_XL/val_ode50_cfg2.25/predict`.
The callback caps ordinary preview PNG writing at roughly ten images and writes
the final compressed `output.npz` only as the run completes; no recursive
output scan or large-file hash was performed.

Current PixelGen protocol: PixelGen-XL at 256, EMA prediction, exact Heun 50,
99 combined-CFG network calls, guidance 2.25 on `(0.1,0.9]`, timeshift 2.0,
BF16 mixed precision, and compressed output NPZ plus only preview PNGs.

## SeaCache semantics

The audited SEA implementation converts contiguous probes to float32, uses a
separable full FFT over patch-grid H/W, mean-normalizes the Wiener gain, and
restores dtype. JiT/PixelGen use `z=t*x+(1-t)*noise`, hence `a=t,b=1-t` with
official endpoint clamp. Relative L1 reduces over the complete tensor and
returns a synchronized Python float. First/final calls are full; ordinary calls
filter current probe, accumulate distance, use strict `< threshold` for reuse,
then preserve the audited raw/filtered previous-probe write order. Residual is
`body_output-body_input`; final norm/projection/unpatchify stay fresh.

## PixelGen structure

- JiT model: `third-party/PixelGen/src/models/transformer/JiT.py:408-578`.
- `forward(x,t,y,return_layer=None,return_last=False)` inserts class context at
  block 8 and removes it before the final head.
- Diagnostic `return_layer`/`return_last` change return tuples; the port forces
  exact full body for either option and never fabricates cached intermediate
  features.
- Heun sampler: `src/diffusion/flow_matching/sampling.py:312-410`.
- CFG concatenates `[x,x]` and `[uncondition,condition]` into one 2B forward.
  The faithful cache therefore uses one batch-global `combined_cfg` state.
- With `exact_henu=true`, 50 steps produce `2*(50-1)+1=99` combined forwards.
- Guidance applies when `t>0.1 and t<=0.9`; timeshift is 2.0.
- Lightning deep-copies denoiser to `ema_denoiser` and upstream normally calls
  `compile()` on both (`src/lightning_model.py:45,67-68`). Prediction defaults
  to EMA unless `eval_original_model` is enabled.

The body residual covers image tokens after patch/position embedding through
all blocks and after context removal. The first-block modulated norm input is
the probe. The final head is always fresh.

## Current RNG/output limitation

The active upstream dataset generates random noise under Lightning/DDP and the
compressed callback writes a single large NPZ after completion. The NPZ does
not carry a complete per-sample mapping of stable sample ID, class ID, seed,
rank RNG offset, and initial-noise hash. Preview filenames are insufficient to
prove noise identity. Exact world/rank/batch RNG consumption cannot be
recovered from the compressed result alone.

## License and CPU limitation

Neither local SeaCache nor PixelGen supplies a license file. PixelGen's rotary
embedding construction contains hard-coded `.cuda()`, so real-model parity is
GPU-deferred; CPU tests use mocks.
