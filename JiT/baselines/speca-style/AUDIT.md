# SpeCa-style JiT audit

> Protocol update (2026-07-14): PixARC's unified primary experiment now uses real batch 32 for every JiT baseline. Historical batch-1 observations below describe the original port audit, not the active experiment protocol. SpeCa must be retuned at batch 32.

This audit is pinned to the local checkout. Internet state is not an implementation source.

## Revisions and provenance

| Component | Local identity | Role |
|---|---|---|
| PixARC | `f15b77ac684d7254fde1db4b001d728b11da6550` | integration tree |
| Cache4Diffusion | `91a1949fcc88acab46547f0b5f295f5de2df2870` | released SpeCa-DiT behavior |
| TaylorSeer | `704ee98c74f7f04da443daa3c0aa2cc7803d86e3` | finite-difference reference |
| vendored JiT | PixARC tree `d697163e4899e279a3c969d429832efecc9da115` | model/sampler reference |
| vendored PixelGen | PixARC tree `3043acf90f255a264f1445bda9ea8d468ba91a58` | cross-port parity reference |

`third-party/JiT` is vendored inside PixARC rather than an independent Git worktree; its tree object is therefore recorded instead of mislabeling the PixARC HEAD as an upstream JiT commit.

## Files inspected

- SpeCa control: [`cal_type.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/cache_functions/cal_type.py), [`cache_init.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/cache_functions/cache_init.py), [`models.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/models.py).
- Sampling/defaults: [`sample.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/sample.py), [`sample_ddp.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/sample_ddp.py), and the local diffusion loop under `../../../baselines/Cache4Diffusion/dit/speca-dit/diffusion/`.
- Taylor math: [`taylor_utils/__init__.py`](../../../baselines/Cache4Diffusion/dit/speca-dit/taylor_utils/__init__.py) plus the local TaylorSeer checkout and read-only `../taylorseer-style/` port.
- JiT model/sampler/EMA/compile paths under [`third-party/JiT`](../../../third-party/JiT/).
- Existing SeaCache/TaylorSeer manifest, image I/O, sharding, metrics, metadata, and benchmark interfaces, read-only.

## Audit conclusions

- The released draft is per-block Taylor prediction of separate attention and MLP outputs before current AdaLN gates. Current modulation and gates remain fresh; ordinary Taylor blocks skip both norms and both expensive modules.
- Only a Full NFE updates exact history. A verifier execution never updates history because its input is a speculative prefix.
- Verification is local to block 27 in the released DiT-XL/2 code. This port resolves `verify_layer=-1` to `len(blocks)-1` and does not hard-code 27.
- The verifier compares complete draft/exact block outputs from the same speculative-prefix input. Its exact result is not written back, and an error never rolls back the current NFE.
- The error computed after the current Taylor NFE is stored for the next decision. The main metric is batch-global elementwise `relative_l1`, `eps=1e-10`, with a Python scalar conversion.
- The released scheduler starts Full/Taylor/Full for the first three decisions when `first_enhance=3`; it does not perform three consecutive Full steps. Full always forces the next action to Taylor.
- `interval` and `cache_counter` do not control the released adaptive schedule. Fixed interval exists only in `taylor_draft_fixed` as a parity/ablation mode.
- JiT 50-step Heun produces 99 model-evaluation decisions. Conditional and unconditional calls share one decision but keep independent Taylor histories, yielding 198 `JiT.forward` calls.
- Main SpeCa generation is real batch 1 per process. Any batch larger than one is grouped-batch SpeCa and is not the registered main protocol.

## Released defaults observed locally

| Script | steps | per-process batch | max order | base | decay | min/max Taylor | metric | CFG | seed | VAE |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| `sample.py` | 50 | 1 | 4 | 0.1 | 0.01 | 2 / 5 | `relative_l1` | 1.5 | 1 | `mse` |
| `sample_ddp.py` | 100 | 1 | 4 | 0.3 | 0.05 | 3 / 8 | `relative_l1` | 1.5 | 0 | `ema` |

Both use `first_enhance=3` and threshold floor `0.01`. The single-process script accesses `args.interval` although that parser does not define the argument; this released-code defect is recorded, not silently used to invent an interval rule.

## Safety and scope

The initial 2026-07-13 UTC read-only snapshot identified existing PixelGen Full work and JiT Full work/queueing. Process state is ephemeral; it is not reused as performance evidence. This implementation/documentation pass launched no CUDA context, attached to no process, sent no signal, and wrote no reference output. GPU smoke, compile, LPIPS, FID, latency, memory peaks, tuning, and generation remain explicitly deferred.

No files under Cache4Diffusion, TaylorSeer, `third-party`, SeaCache, or TaylorSeer-style were modified by this port.
