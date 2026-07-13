# PixelGen implementation report

This report separates source-level implementation facts from experiments. No
GPU experiment was executed in this implementation round. The CPU-only test
results below were recorded with `CUDA_VISIBLE_DEVICES=''`.

| # | Required question | Answer |
|---:|---|---|
| 1 | TaylorSeer commit | `704ee98c74f7f04da443daa3c0aa2cc7803d86e3`. |
| 2 | Official files | `TaylorSeer-DiT/models.py`, `taylor_utils/__init__.py`, `cache_functions/cal_type.py`, `cache_functions/cache_init.py`, `sample.py`, `sample_ddp.py`. |
| 3 | Cache4Diffusion | Commit `91a1949fcc88acab46547f0b5f295f5de2df2870` was audited for contrast/engineering only; no primary algorithm code was copied. |
| 4 | Primary implementation | Original-style per-block, per-attention/MLP forecast; not Lite and not whole-body residual forecasting. |
| 5 | Attention target | Complete `block.attn(...)` output, including output projection/dropout, before `gate_msa`. |
| 6 | MLP target | Complete MLP output before `gate_mlp`. |
| 7 | Fresh gate | Yes. Current AdaLN modulation and gates are recomputed every call. |
| 8 | Norm on Taylor | No: norm1/attention/norm2/MLP are skipped. Embeddings/AdaLN/gates remain fresh. |
| 9 | Final head | Always fresh: final norm/AdaLN/linear and unpatchify. |
| 10 | Exact-only history | Yes. Forecast reads do not update factors, anchor, coordinate list, or order. |
| 11 | Official formula parity | Passed for orders 0--4, float32/float64, and positive/negative coordinate gaps against the local official helper. |
| 12 | Max formula errors | Maximum absolute error 0.0; maximum relative error 0.0. |
| 13 | Schedule parity | Passed for intervals 1--5; Full/Taylor sequences and counts match the official 50-NFE behavior exactly. |
| 14 | Remove depth 28 | Runtime iterates inherited `blocks`; core state has no fixed depth, although this XL config explicitly declares upstream depth 28. |
| 15 | Remove 49/50 hard-code | `total_nfe` is derived from sampler/steps; `q=total_nfe-1-nfe_index`. |
| 16 | `first_enhance` | Default 2: both initial NFE are Full and reset the counter. |
| 17 | Last Full | Faithful default false because official `last_steps` is unused. `force_last_full` is explicit ablation only. |
| 18 | NFE index rationale | Exact Heun repeats continuous time at different states; monotone q avoids zero gaps. |
| 19 | Repeated t | Corrector and next predictor receive successive q values; shifted t is trace-only. |
| 20 | JiT dual streams | Not used here. PixelGen preserves one combined stream. |
| 21 | Combined 2B | Upstream concatenates `[x,x]` and `[uncondition,condition]` into one forward, so one `combined_cfg` history is faithful. |
| 22 | 50-step NFE | 99: 50 predictors plus 49 correctors with `exact_henu=true`. |
| 23 | Network forwards | 99 combined 2B forwards. |
| 24 | Context tokens | Insert 32 before block 8, retain through later blocks, remove once before final head. Per-layer shapes are validated. |
| 25 | RoPE | Full pre/post-insertion blocks use image/context-aware RoPE; Taylor skips attention/RoPE. |
| 26 | `return_layer`/`return_last` | Interface and tuple shapes are preserved; either request forces the current NFE Full with `diagnostic_return`. |
| 27 | State/checkpoint | Runtime is not a parameter or persistent buffer and must not add state_dict keys. |
| 28 | EMA/deepcopy | Deepcopy creates empty independent runtimes with no shared factor tensors; prediction retains upstream EMA choice. |
| 29 | Compile risk | Lightning compiles both denoisers; dynamic state can graph-break. Matched eager is primary pending GPU tests. |
| 30 | Cache estimate | PixelGen-XL, BF16, image batch 4 (effective 8), K=4 upper bound: 280 tensors, 1.340332 GiB factors. |
| 31 | 3090 maximum batch | Unknown; no GPU/OOM search was performed. |
| 32 | Current Full pairing | Blocked. A manifest-backed Full rerun is required. |
| 33 | CPU tests | 61 pytest tests passed; compileall, common-core hash/API comparison, manifest CLI checks, and memory estimator passed. |
| 34 | Deferred GPU tests | Model load/key check, smoke, interval=1 parity, shadow, compile, 1K, 8K selection, benchmark, 50K, metrics. |
| 35 | Upstream modifications | None intended or authorized; all port changes are within this new directory. |
| 36 | New CUDA task | None started. |
| 37 | Current 50K interference | None: no signals, attachment, output writes, or new GPU work. |
| 38 | Remaining risks | Real-model/tuple parity, checkpoint/EMA integration, Lightning/compiler graphs, memory/maximum batch, and operating-point choice remain unverified. |

## Implementation surface

`TaylorSeerPixelGenJiT` inherits the upstream model and exposes the gate-pre
branches. `TaylorSeerHeunSamplerJiT` maps each combined forward to one NFE.
`TaylorSeerPixelGenLightning` retains EMA selection, scopes every prediction
batch to a trajectory, and cleans state in `finally`; `InferenceOnlyTrainer`
avoids training-only dependencies. Common modules provide finite differences,
scheduling, state, trace, memory, manifest/metadata, metrics, and latency.

Four runtime modes are explicit: `upstream_full`, `instrumented_full`,
`taylorseer`, and `shadow_forecast`. Shadow runs exact work and is diagnostic,
not a latency baseline. The Taylor template deliberately leaves interval/order
null until a frozen 8K-selected copy is created.
