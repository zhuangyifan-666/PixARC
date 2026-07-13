# PixelGen combined-CFG semantics

PixelGen does not use the WAN two-forward layout. It constructs `cfg_x=cat([x,x])` and `cfg_condition=cat([unconditional,conditional])` and invokes the JiT once. The port preserves that ordering and exposes one explicit stream ID, `combined_cfg`.

All cached tensors therefore have leading dimension `2B`. One probe, one gate decision, one accumulator, one exact anchor window, and one DCTA gamma apply to the entire combined tensor. The halves cannot choose different actions and never read branch-specific residuals. `combined_cfg_sample_ids` validates that the B IDs are duplicated in the same order if a 2B list is supplied.

This differs from WAN’s two independent slots and is consistent with the audited combined-batch Hunyuan pattern. It also means conditional/unconditional disagreement metrics are structurally zero/not applicable for this backend; they must not be fabricated by splitting the forward.

Guidance and model-output chunking remain upstream operations after the one combined forward. Every call’s final PixelGen head is fresh. `return_layer`/`return_last` force an exact combined call.

