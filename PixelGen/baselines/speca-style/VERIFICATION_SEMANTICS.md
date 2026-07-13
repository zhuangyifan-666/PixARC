# PixelGen verification semantics

## Local check from a speculative prefix

On a Taylor NFE with `check=true`, earlier blocks use draft forecasts. At the selected block, the resulting `x_verify_in` is therefore speculative. Two branches use the same tensor and current conditioning:

```text
x_pred  = draft_block(x_verify_in, fresh gates, Taylor factors)
x_exact = exact_block(x_verify_in, fresh modulation/norm/attention/MLP/gates)
error   = official_metric(x_pred, x_exact)
main path continues with x_pred
```

`verify_layer=-1` resolves `len(model.blocks)-1`. PixelGen XL/2 currently resolves to 27, but no behavior depends on that literal. Other layers are explicit ablations.

The exact branch is exact only conditional on its speculative prefix. It compares complete post-residual block outputs, not an isolated module or fully exact-model hidden state. It never writes back, updates factors, replays the prefix, rewinds Heun, or repairs the current output. Current error is recorded at `end_nfe` for the next decision.

Final norm/projection, context removal, and unpatchify always execute fresh on the retained draft path.

## Combined CFG and tokens

PixelGen keeps upstream ordering:

```text
cfg_x         = cat([x, x], dim=0)
cfg_condition = cat([uncondition, condition], dim=0)
```

The verifier operates on the entire effective `2B` output with one history/scheduler/error. It must not split calls, reorder halves, verify only one half, or schedule halves independently.

Main `verification_token_scope=all_tokens`; the final block still has image and context tokens, so both enter the reduction. `image_tokens_only` is diagnostic-only.

## Diagnostic return contract

Upstream `forward(x,t,y,return_layer=None,return_last=False)` must retain exact semantics. If `return_layer` is requested or `return_last=True`, the relevant NFE is forced Full and trace records `forced_full_reason=diagnostic_return`. A predicted feature is never returned under an interface promising an exact intermediate. These diagnostic paths are excluded from main latency and 50K.

## Error and clone behavior

```text
relative_l1 = mean(abs(pred-exact)/(abs(exact)+1e-10))
failure     = current_error > current_threshold
```

The reduction includes both CFG halves, all selected tokens, and channels. Equality passes, and scalar conversion/synchronization is timed. The port saves/clones verifier input only at the chosen layer when checking, rather than cloning at every block. CPU mocks cover same-input branches, no writeback/history update, full combined reduction, diagnostic forcing, and previous-error timing; GPU block parity/memory remain deferred.

