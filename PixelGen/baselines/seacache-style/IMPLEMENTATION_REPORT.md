# PixelGen SeaCache-style implementation report

This is an unofficial port. No GPU quality, latency, speedup, threshold, or
SeaCache 50K result is claimed.

## Required implementation questions

### 1. Which SeaCache revision and files were used?

The source of behavior is local SeaCache commit
`8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2`, specifically
`FLUX/util_seacache.py` and `FLUX/seacache_generate.py`. The implementation was
rewritten locally rather than copying an upstream model file. PixelGen is a
snapshot in PixARC commit `d54c1e26768d80bf7c067f50e28868cdbf59d431`, not an
independent Git checkout with a recoverable upstream commit.

### 2. Is the official probe-update order preserved?

Yes, under `compatibility_mode="official_faithful"`. First and final calls
store raw probe; ordinary calls store the SEA-filtered current probe. Distance
is accumulated before strict threshold comparison. A threshold refresh clears
the accumulator before the previous probe is written. A missing residual falls
back to exact body without undoing the gate's already-updated accumulator.

### 3. What was the CPU SEA parity error?

Float32, bfloat16 restoration, rectangular grids, full FFT, rFFT, peak and mean
normalization fixtures were compared to the audited local function with
`rtol=0, atol=0`. Observed maximum elementwise error was 0 on those fixtures.
No GPU parity claim is made.

### 4. Is the gate batch-global?

Yes. `rel_l1` reduces over the complete effective 2B CFG batch, H, W, and
channel tensor. There is no per-sample gate in the main port.

### 5. What are the exact residual boundaries?

Input is the image-token tensor after patch embedding and fixed positional
embedding, immediately before block 0. Output is the image-token tensor after
all Transformer blocks and after the temporary class-context prefix is
removed. The cached value is `body_output - body_input`.

### 6. Is the final head always fresh?

Yes. Final RMSNorm/AdaLN modulation, linear projection, and unpatchify execute
on every model call, including cache reuse.

### 7. Why does standalone JiT need two states?

Standalone JiT calls conditional then unconditional as two independent network
forwards, so their probes/residuals must be isolated. That behavior is not
ported into PixelGen.

### 8. Why does PixelGen use one combined 2B state?

The audited PixelGen sampler concatenates `[x,x]` and
`[uncondition,condition]`, then performs one model forward. The port preserves
that exact ordering and uses one stream named `combined_cfg`. Splitting it
would change the batch-global gate, kernels, numerical result, and latency.

### 9. How is the 50-step call count derived?

With `exact_henu=true`, every one of 50 macro-steps evaluates a predictor and
the first 49 also evaluate a corrector: `50+49=99` combined 2B forwards. The
sampler derives a call plan from `num_steps` and `exact_henu`; it does not
hard-code 99.

### 10. How are context tokens handled?

PixelGen-XL inserts 32 class-context tokens before block 8, uses the
context-aware RoPE thereafter, and removes the first 32 tokens after the body.
Only image tokens enter the residual cache.

### 11. How are `return_layer` and `return_last` handled?

Either diagnostic request forces an exact full-body execution and records
`forced_full_reason=diagnostic_return`. The adapter returns the same output or
tuple ordering as upstream and never fabricates an intermediate from a cached
residual.

### 12. Does runtime state enter `state_dict`?

No. The controller and dataclass states are attached as plain Python objects,
not parameters, buffers, or submodules. Tests verify that no SeaCache runtime
key appears in `state_dict`.

### 13. Are EMA and deepcopy safe?

The upstream Lightning model deep-copies `denoiser` to `ema_denoiser`.
Controller serialization deliberately creates an independent empty state, so
the two models do not share probe/residual tensors. Prediction still selects
`ema_denoiser` unless upstream `eval_original_model` requests the original.
Real checkpoint/EMA key parity remains GPU-deferred.

### 14. What compile risks remain?

Mutable Python state, scalar synchronization, FFT, and dynamic full/reuse
branches can graph-break or recompile. `matched_eager` unwraps upstream block
compile wrappers per denoiser/EMA instance and skips outer compile; `blockwise`
preserves block wrappers but skips outer compile; `upstream` retains the outer
compile path. See `COMPILE_COMPATIBILITY.md`; no GPU compile matrix has run.

### 15. Can the active Full reference be used for paired metrics?

No: `PAIRED_METRICS_BLOCKED`. The compressed final `output.npz` does not store
the complete per-sample seed/initial-noise mapping, while only a small preview
subset has seed-bearing filenames. Filename/order alone cannot prove equal
Gaussian noise. Once complete and validated, it may still be used for absolute
distribution metrics. A new manifest-driven Full run is required for strict
PSNR/SSIM/LPIPS.

### 16. Which tests were executed on CPU?

The final command was:

```bash
ROOT="$(git rev-parse --show-toplevel)"
: "${PIXELGEN_PYTHON:=python}"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/matplotlib-seacache-test \
  PYTHONPATH="$ROOT/third-party/PixelGen:$ROOT/PixelGen/baselines/seacache-style" \
  "$PIXELGEN_PYTHON" -m unittest discover \
  -s "$ROOT/PixelGen/baselines/seacache-style/tests" -v
```

It passed 47 tests with zero failures and zero skips. Coverage includes SEA
parity, controller/state transitions, combined 2B lifecycle, exact call plans,
deepcopy/state-dict isolation, 50K manifests/sharding, per-sample RNG,
manifest dataset metadata, strict pairing toy metrics, comparable distribution
deltas, low-precision position-add semantics, compile-wrapper isolation, and
runtime batch-group rejection, including archived-input and per-rank metadata
identity binding. PyTorch emitted a CUDA-availability warning
under the empty visibility mask; no CUDA model or workload was launched.

### 17. Which tests must wait for GPU availability?

Real checkpoint and EMA load, upstream Full parity, force-full parity, real
reuse, finite-output checks, actual 99-call instrumentation, compile modes,
1K proxy, 8K threshold sweep, matched single-GPU benchmark, and final 4-GPU
50K generation/evaluation.

### 18. What risks remain?

Real LightningCLI/checkpoint integration (including the parameter-free local
inference trainer), EMA selection, compile behavior, output callback integration
at scale, batch-global sensitivity, threshold selection, and all real
quality/performance measurements remain unverified. Run identity binds actual
port bytes, raw/canonical manifest, model/sampler configs, environment and
checkpoint path/size, but this stage intentionally did not hash the multi-GB
checkpoint content; a same-path, same-size replacement remains a documented
integrity risk.

### 19. Were upstream clones modified?

No. All task files are under `PixelGen/baselines/seacache-style/`.
`third-party/PixelGen` and `baselines/SeaCache` were not modified.

### 20. Were active GPU processes started or disturbed?

No. This task started no CUDA model, sent no signal, attached to no process,
and wrote nothing into the active reference output.

## Delivered integration

`SeaCacheJiT` directly delegates to upstream `forward` in `mode=full`. Non-full
modes split only embedding/body/fresh-head logic and use the exact upstream
block-0 norm/modulate probe. `SeaCacheHeunSamplerJiT` scopes one combined state
to each prediction batch in `try/finally`. `SeaCacheLightningModel` keeps
original/EMA compile treatment symmetric. Manifest-backed prediction and
atomic numeric PNG/metadata output are in `pixelgen_io.py`.

The included benchmark factory measures Full then SeaCache with identical
labels/noise on one GPU and reports median/mean latency, speedup, calls,
full-body ratio, gate/FFT/cache timings, and peak memory. It has not been run.
