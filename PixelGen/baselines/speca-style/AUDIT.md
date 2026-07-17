# SpeCa-style PixelGen audit

> Protocol update (2026-07-14): PixARC's unified primary experiment now uses real batch 4/effective CFG batch 8 for every PixelGen baseline. Historical batch-1 observations below describe the original port audit, not the active experiment protocol. SpeCa must be retuned under grouped batch 4.

This audit is pinned to the local checkout. Internet state is not an implementation source.

## Revisions and provenance

| Component | Local identity | Role |
|---|---|---|
| PixARC | `f15b77ac684d7254fde1db4b001d728b11da6550` | integration tree |
| Cache4Diffusion | `91a1949fcc88acab46547f0b5f295f5de2df2870` | released SpeCa-DiT behavior |
| TaylorSeer | `704ee98c74f7f04da443daa3c0aa2cc7803d86e3` | finite-difference reference |
| vendored JiT | PixARC tree `d697163e4899e279a3c969d429832efecc9da115` | cross-port reference |
| vendored PixelGen | PixARC tree `3043acf90f255a264f1445bda9ea8d468ba91a58` | model/sampler/Lightning reference |

`third-party/PixelGen` is vendored inside PixARC rather than an independent Git worktree; its tree object is recorded instead of calling the PixARC HEAD an upstream PixelGen commit.

## Files inspected

- SpeCa control: [`cal_type.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/cache_functions/cal_type.py), [`cache_init.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/cache_functions/cache_init.py), [`models.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/models.py).
- Sampling/defaults: [`sample.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/sample.py), [`sample_ddp.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/sample_ddp.py), and the local diffusion loop.
- Taylor math: [`taylor_utils/__init__.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/taylor_utils/__init__.py), the local TaylorSeer checkout, and read-only `../taylorseer-style/`.
- PixelGen JiT class, block/context/final APIs, `HeunSamplerJiT`/`exact_henu`, CFG concatenation, Lightning denoiser/EMA deepcopy/compile, prediction dataset/callback, seed/noise/output flow under [`third-party/PixelGen`](../../../third-party/PixelGen/).
- Existing sibling manifest, image I/O, sharding, metrics, metadata, and benchmark interfaces, read-only.

## Audit conclusions

- Released SpeCa drafts per-layer, separate gate-pre attention/MLP tensors; current conditioning and gates remain fresh and only Full updates history.
- Its final-block exact verifier starts from a speculative prefix, compares complete block outputs, does not write back, and cannot roll back the current NFE.
- Main metric is batch-global elementwise `relative_l1`, `eps=1e-10`. Current error is computed after execution and can affect only the next decision.
- Startup is Full/Taylor/Full at `first_enhance=3`; Full always forces one following Taylor. `interval` and `cache_counter` do not select adaptive actions.
- PixelGen preserves one combined `2B` forward in `[unconditional, conditional]` order, one factor history, one scheduler, and one metric over the complete tensor.
- `return_layer`/`return_last` diagnostic requests require exact features and therefore force the relevant NFE Full.
- PixelGen 50-step `exact_henu=true` produces 99 combined forwards. Main real batch is 1, so the model's effective batch is 2.
- Lightning/EMA deepcopy requires runtime independence and empty histories; runtime state is excluded from `state_dict` and reset at each prediction batch.

## Released defaults observed locally

| Script | steps | per-process batch | max order | base | decay | min/max Taylor | metric | CFG | seed | VAE |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| `sample.py` | 50 | 1 | 4 | 0.1 | 0.01 | 2 / 5 | `relative_l1` | 1.5 | 1 | `mse` |
| `sample_ddp.py` | 100 | 1 | 4 | 0.3 | 0.05 | 3 / 8 | `relative_l1` | 1.5 | 0 | `ema` |

Both use `first_enhance=3` and threshold floor `0.01`. `sample.py` accesses an undefined parser field `args.interval`; this is a released-code defect, not evidence for a fixed interval adaptive schedule.

## Safety and scope

The initial 2026-07-13 UTC read-only snapshot identified existing PixelGen Full work and JiT Full work/queueing. Process state is ephemeral and is not performance evidence. This pass launched no CUDA context, attached to no process, sent no signal, and wrote no reference output. GPU smoke, compile, LPIPS, FID, latency, memory peaks, tuning, and generation remain explicitly deferred.

No files under Cache4Diffusion, TaylorSeer, `third-party`, SeaCache, or TaylorSeer-style were modified by this port.
