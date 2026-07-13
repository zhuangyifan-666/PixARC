# Body/cache boundary

For PixelGen JiT, `body_input` is the image-token tensor after `x_embedder(x) + pos_embed` and before transformer block 0. It has shape `[2B, (H/P)*(W/P), hidden]` in the combined CFG call.

`probe_internal_state` is the complete resumable state after exactly `probe_depth` blocks: token tensor, next block index, whether class context has been inserted, and the image-token start index. Context is inserted before block `in_context_start`, prepended to image tokens, and retained through later blocks. The prefix uses `feat_rope` before insertion and `feat_rope_incontext` from the insertion block onward.

`probe_feature` extracts only image tokens. Before insertion its start is zero; after insertion it starts at `in_context_len`. It is never made shape-compatible by truncating an arbitrary end. `exact_body_output` is the image-only tensor after every block and after removing context exactly once, but before `final_layer` and `unpatchify`.

The stored quantities are:

```text
full_body_residual = exact_body_output - body_input
probe_residual     = probe_feature     - body_input
```

All four tensors must have equal shape, dtype, and device. Exact Full and probe residuals are appended atomically as one anchor pair. A Reuse output is `body_input + estimated_full_residual`; it never becomes an anchor.

The final AdaLN/norm, output projection, and unpatchify are fresh on every call. Diagnostic `return_layer` or `return_last` forces exact Full so an approximation is never exposed as an exact intermediate.

