# JiT safety and limitations

## Execution safety

Implementation was limited to read-only audit, file authoring, and CPU-only
checks. It did not launch a CUDA model, generation, benchmark, compile test,
FID, or LPIPS job. It did not signal, attach to, inspect the environment of,
or alter the active PixelGen process or queued JiT launcher.

Deferred GPU tools require the explicit gate
`TAYLORSEER_GPU_TESTS_ALLOWED=1`, an operator-selected
`CUDA_VISIBLE_DEVICES`, and an idle-device check. A busy or unqueryable GPU is
a refusal condition. The tools do not kill jobs. Operators must still inspect
allocation policy and active runs immediately before setting the gate.

Generation requires an explicit output root outside the source repository.
Non-empty output roots are rejected unless validated resume is requested.
Inputs are archived immutably; images use temporary-file plus atomic rename.
Resume skips only exact metadata matches. A damaged image, changed config,
manifest/checkpoint mismatch, or incompatible batch grouping fails closed.

No tool downloads checkpoints, datasets, evaluator assets, LPIPS weights, or
packages. Do not point output at the current upstream Full directory.

## Scientific and engineering limitations

- This is an unofficial port; no author endorsement is implied.
- CPU formula/scheduler tests do not prove real-model numerical parity.
- GPU smoke, interval=1 parity, shadow error, compile behavior, latency,
  memory, and 50K generation remain deferred.
- The 24-GB RTX 3090 maximum batch is unknown. Analytical cache bytes exclude
  model/activation/compiler overhead.
- `interval` and `max_order` are deliberately unresolved until independent
  8K validation; final 50K must not select them.
- `official_nfe_index` is a documented Heun adaptation. Direct continuous t is
  unsafe because of repeated times; stage-separated history is not shipped as
  the primary method.
- The faithful schedule does not force the final call Full. Enabling
  `force_last_full` is a separately labeled ablation.
- Higher-order forecasting may amplify numerical error; the faithful baseline
  deliberately adds no clipping, damping, correction, or error gate.
- Cache cost scales with two streams, all blocks, two modules, batch, token
  count, and K+1 factors. No Lite fallback is automatic.
- The queued JiT Full is only conditionally pairable pending exact RNG replay.
- Only Heun/Euler paths explicitly supported by the adapter are in scope;
  training and adaptive scheduling are not.

