# JiT compile compatibility

SpeCa's Python scheduler, action-dependent block path, verifier toggle, mutable finite-difference state, variable available order, dictionary/dataclass access, and metric `.item()` are hostile to a single whole-model graph. The scheduler and scalar synchronization therefore stay outside compiled regions.

## Modes

| `compile_mode` | Meaning | Valid comparison |
|---|---|---|
| `upstream` | preserve the upstream JiT compile arrangement for an auxiliary Full reference | upstream Full only; not the primary SpeCa denominator unless paths are actually matched |
| `matched_eager` | local split blocks run eager on both sides; `instrumented_full` has no Taylor cache/update/verifier | primary reliable comparison |
| `blockwise` | compile stable exact block/forecast/verifier kernels independently while scheduler remains eager | compare only after both Full and SpeCa use the same setting |

JiT upstream code already places compile decorators around blocks/final components. That does not authorize dividing an upstream-compiled Full latency by an eager SpeCa latency.

## Expected graph boundaries and breaks

- `begin_nfe`, previous-error thresholding, action selection, `end_nfe`, trace, and `.item()` remain host-side.
- Full/Taylor selection happens before entering a stable block kernel.
- Exact attention/MLP, forecast arithmetic, and the local exact verifier are possible blockwise compilation units.
- Shape/device/dtype/stream checks and history allocation are trajectory/runtime operations, not graph state.
- A new available Taylor order may compile a different forecast shape/path; warmups must cover the measured steady state.

## Fair timing contract

The primary result is median matched-instrumented-Full latency divided by median SpeCa latency with identical GPU, real batch 1, effective CFG layout, checkpoint/EMA, 50-step Heun, CFG/timeshift, dtype, initial noise, and compile mode. `instrumented_full` measures the split exact model path only: it allocates/updates no Taylor factors and runs no verifier. SpeCa timing includes its predictor, Full-action history updates, local verification, reductions, scalar sync, scheduler, cache I/O, CFG, sampler, fresh final head, unpatchify, and reset. Compile time, loading, data loading, CPU copy, PNG, and evaluators are excluded.

No GPU compile run was executed in this task. Graph breaks, compile time, parity, speedup, and memory for `blockwise` remain deferred. Until measured, `matched_eager` is the only registered primary comparison.
