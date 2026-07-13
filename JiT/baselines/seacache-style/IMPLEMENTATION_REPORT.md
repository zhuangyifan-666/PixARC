# JiT SeaCache-style implementation report

This report describes an unofficial port. It does not report GPU quality,
latency, speedup, or 50K generation results.

## Required implementation questions

### 1. Which SeaCache revision and files were used?

SeaCache commit `8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2`, specifically
`FLUX/util_seacache.py` and `FLUX/seacache_generate.py`. Those implementation
files last changed at `0a91c2bacad77b56e4cce4183a0254ac40c0739c`.

### 2. Is the probe update order preserved?

Yes, in `compatibility_mode="official_faithful"`: first/final calls store raw
probe; ordinary calls store filtered current probe; distance is accumulated
before strict threshold comparison; a threshold refresh clears accumulation
before writing previous probe. Missing residual safely forces exact body
without undoing the already-updated accumulator.

### 3. What is SEA filter CPU parity error?

The integrated CPU suite compares float32, bfloat16-restored, rectangular,
peak-normalized, mean-normalized, full FFT, and rFFT paths against the local
official function with `rtol=0, atol=0`. All passed, so observed maximum
elementwise error for those fixtures was exactly 0. No GPU parity claim is
made.

### 4. Is the gate batch-global?

Yes. Relative L1 reduces over the complete batch/grid/channel tensor. No
per-sample gate is used in the main port.

### 5. What are the exact residual boundaries?

Input: image tokens after patch embedding plus fixed position embedding and
before block 0. Output: image tokens after all Transformer blocks and after the
temporary class context prefix is removed. Cached residual is
`body_output-body_input`.

### 6. Is the final head always fresh?

Yes. Final RMSNorm/AdaLN, projection, and unpatchify run after every full or
reuse body decision.

### 7. Why are two JiT states required?

Upstream `_forward_sample` executes conditional then unconditional as two
separate network forwards. Their label embedding, modulation probe, accumulated
distance, and residual differ. The port routes them to `cond` and `uncond`
states and tests their ordering/isolation.

### 8. Why is PixelGen combined 2B not used here?

JiT does not concatenate CFG into 2B. Changing it would alter upstream batch
semantics, batch-global decisions, kernels, and latency. This port preserves
two calls.

### 9. How is the 50-step call count derived?

There are 49 Heun predictor/corrector pairs plus one final Euler evaluation:
`2*(50-1)+1=99` calls per stream, 198 total JiT network forwards. The helper
derives this from sampler method and step count rather than hard-coding 99.

### 10. How are context tokens handled?

JiT-B/16 inserts 32 `y_emb+in_context_posemb` tokens before block 4, uses the
context-aware RoPE thereafter, and removes the first 32 tokens after all
blocks. Residuals contain only image tokens. Nonstandard context insertion at
block 0 is rejected because it lacks an image-only first-block grid probe.

### 11. How are `return_layer` and `return_last` handled?

The audited JiT model exposes neither option. The adapter preserves its
`forward(x,t,y)` result and only adds keyword-only cache/trace context.

### 12. Does runtime state enter `state_dict`?

No. The controller is attached with plain-object assignment and owns ordinary
dataclass/Python state. CPU tests confirm no SeaCache runtime key appears in
the model state dict.

### 13. Are EMA and deepcopy safe?

The adapter parameter namespace remains `net.*`, matching upstream. Evaluation
must retain upstream strict checkpoint load and EMA1 substitution. Real
checkpoint/EMA parity is deferred to GPU because upstream model construction
creates CUDA tensors. Runtime controller deepcopy clears live state by design.

### 14. What compile risks remain?

Mutable state, scalar synchronization, FFT, and dynamic branching can cause
graph breaks. `matched_eager` now unwraps upstream block/final compile wrappers
on each model instance; `blockwise` preserves them. This is symmetric for Full
and SeaCache and changes no upstream class. See `COMPILE_COMPATIBILITY.md`; no
GPU compile test has been run.

### 15. Can the current Full reference be used for paired metrics?

Conditionally. It was only scheduled at audit time and does not save per-image
noise metadata. Exact rank-stream replay and all conditions in
`BASELINE_COMPATIBILITY_REPORT.md` must first be proven. Distribution metrics
remain usable after successful completion/output validation.

### 16. Which tests were executed on CPU?

The command

```bash
ROOT="$(git rev-parse --show-toplevel)"
: "${JIT_PYTHON:=python}"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$ROOT/JiT/baselines/seacache-style:$ROOT/third-party/JiT" \
  "$JIT_PYTHON" -m pytest -q -p no:cacheprovider \
  "$ROOT/JiT/baselines/seacache-style/tests"
```

reported `37 passed, 2 subtests passed` with zero failures/skips. It covered
SEA parity, controller/state, manifests/sharding, metadata, toy paired metrics,
JiT stream isolation, exception reset, explicit noise, and mock body-split
parity, low-precision position-add semantics, compile-wrapper isolation, and
runtime batch-group rejection, including archived-input and per-rank metadata
identity binding. PyTorch printed a CUDA-availability warning
despite the empty visibility mask; no model or CUDA workload was launched.

### 17. Which tests require GPU availability?

Checkpoint/EMA load, upstream Full parity, force-full parity, real reuse,
NaN/Inf checks, actual 99-call instrumentation, compile compatibility, 1K
proxy, 8K threshold sweep, matched single-GPU latency, and final four-GPU 50K.

### 18. What risks remain?

Real-model parity, compile graph behavior, threshold choice, batch-global gate
sensitivity, rank-stream replay of the scheduled reference, and production
runner/checkpoint integration remain unverified on GPU. Run identity binds the
actual executable port bytes, raw/canonical manifest, model/sampler configs,
environment and checkpoint path/size, but this stage intentionally did not hash
the multi-GB checkpoint content; a same-path, same-size replacement remains a
documented integrity risk.

### 19. Were upstream clones modified?

No. All implementation files are under
`JiT/baselines/seacache-style/`; `third-party/JiT` and
`baselines/SeaCache` were not modified by this task.

### 20. Were current GPU processes started or disturbed?

No. This task started no CUDA model, sent no signal, attached to no process,
and wrote nothing into reference outputs.

## Adapter design

`SeaCacheJiT` inherits upstream JiT. In `mode=full` its first action is direct
`super().forward`. Non-full modes split only the minimum embedding/body/head
path and call the same blocks, RoPE, context, final layer, and unpatchify.

`SeaCacheDenoiser` inherits upstream Denoiser but initializes the same fields
without first constructing and discarding a second large CUDA JiT. It adds
explicit noise and trajectory metadata, preserves conditional-before-
unconditional CFG, validates per-stream call count, and resets both states in
`finally`.

## No experimental claims

Threshold candidates are only a validation template. No threshold is selected,
no speedup is measured, and no SeaCache 50K generation has been run.
