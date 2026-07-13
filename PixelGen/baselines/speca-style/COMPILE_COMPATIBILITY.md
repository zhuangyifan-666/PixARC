# PixelGen compile compatibility

SpeCa's Python scheduler, dynamic Full/Taylor/check branches, mutable factor state, variable available order, dictionary/dataclass access, and `.item()` cannot safely be hidden in one whole-model graph. Scheduler and scalar synchronization remain outside compiled regions.

## Modes

| `compile_mode` | Meaning | Valid comparison |
|---|---|---|
| `upstream` | preserve PixelGen's upstream compile/deepcopy arrangement for an auxiliary Full reference | upstream Full only unless paths are genuinely matched |
| `matched_eager` | local split blocks eager on both sides; `instrumented_full` has no Taylor cache/update/verifier | primary reliable comparison |
| `blockwise` | compile stable exact block, forecast, and verifier kernels separately | valid only after parity and same mode on both sides |

PixelGen may deepcopy the denoiser for EMA and compile the resulting modules. Each copy must own an empty, independent runtime; compile wrappers cannot cause factor/scheduler sharing. Runtime is excluded from state dict and checkpoint loading.

## Boundaries and expected breaks

- `begin_nfe`, thresholding, `end_nfe`, trace, and scalar `.item()` are host-side.
- Full/Taylor/check selection precedes stable block kernels.
- Exact attention/MLP, forecast arithmetic, and local exact verification are potential blockwise units.
- Lifecycle validation, combined-CFG shape checks, history allocation, and diagnostic-return forcing remain eager.
- Variable available order can require distinct warmed graph paths.

## Fair timing

Primary `speedup` is median matched-instrumented-Full latency divided by median SpeCa latency with identical GPU, real batch 1/effective batch 2, checkpoint/EMA, exact-Heun settings, CFG/timeshift, dtype, explicit noise, and compile mode. `instrumented_full` measures split exact blocks without Taylor cache allocation/update or verification. SpeCa timing includes its predictor, Full-action history updates, verifier/reduction/scalar sync/scheduler, combined CFG, sampler, fresh head, unpatchify, cache I/O, and reset. First compile is separate.

No GPU compile was executed in this task. PixelGen deepcopy/EMA compile parity, graph breaks, compile time, latency, and peak memory remain deferred. Until measured, `matched_eager` is the registered primary regime.
