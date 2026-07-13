# Cache boundary

The cacheable JiT body begins after fresh patch/position embedding and ends before the fresh output head:

```text
image x
  -> x_embedder(x) + pos_embed                    fresh body_input
  -> blocks 0..11                                 cacheable body
       context tokens prepended once before block 4
       pre-context RoPE -> post-context RoPE
  -> remove the 32-token context prefix           exact_body_output
  -> final RMSNorm/AdaLN + projection             always fresh
  -> unpatchify                                    always fresh
```

For every stream call:

```text
full_residual  = exact_body_output - body_input
probe_residual = probe_feature      - body_input
```

All three image tensors must have the same shape, dtype, and device. With JiT-B/16 at 256px the boundary shape is `[B,256,768]`; context tokens are internal state, never cached as the image residual.

Only exact Full calls append a synchronized `(full_residual, probe_residual)` anchor. Reuse never promotes an estimated residual to exact history. Previous body/probe observations do update after eligible Full and Reuse so the next error is adjacent-call. The final head consumes an approximated body output cast back to `body_input.dtype`, including when residual anchors use FP32 cache storage.

Runtime state is attached outside the module state dictionary and fully released at trajectory end or failure. Checkpoint parameter names remain unchanged.
