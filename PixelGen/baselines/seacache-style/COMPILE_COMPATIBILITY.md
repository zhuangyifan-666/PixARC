# PixelGen compile compatibility

No GPU compile test was run. Upstream Lightning deep-copies the denoiser and
normally compiles both original and EMA models. SeaCache has mutable Python
state, scalar gates and dynamic branching, so outer whole-model compilation is
unsafe without evidence.

The local wrapper exposes `upstream`, `matched_eager`, and `blockwise` and
requires denoiser and EMA denoiser to match. `matched_eager` binds each
block's `_torchdynamo_orig_callable` on both model instances and avoids outer
compilation. `blockwise` preserves upstream block wrappers but avoids outer
compilation. `upstream` preserves block wrappers and calls the upstream outer
compile path. The shipped Full and SeaCache configs both request
`matched_eager`; all three remain GPU-unverified.

GPU-deferred checks must record graph breaks, recompilations, correctness,
compile time, steady latency and memory. Never use compiled Full as the
speedup denominator for an eager SeaCache run. Compilation time is reported
separately.
