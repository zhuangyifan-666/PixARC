# CFG semantics

Vendored JiT executes classifier-free guidance as two model calls in fixed order:

```text
conditional(labels)
unconditional(null_class)
CFG combination
```

The port preserves that order and creates independent `cond` and `uncond` DiCache stream states. Each stream owns its previous body/probe tensors, accumulated error, exact anchors, action, and refresh history. One branch may Full while the other Reuses; the summary records both-Full, both-Reuse, conditional-only-Full, unconditional-only-Full, and disagreement rate.

This resembles WAN’s independent state intent but uses explicit stream IDs rather than a global counter modulo two. It is not Hunyuan/PixelGen combined-CFG semantics. Cross-branch anchors or accumulators are forbidden.

For real batch `B`, each branch reduces over its own `[B,T,D]` tensor and shares one action/gamma across those B samples. The effective amount of model work per NFE is two B-sized forwards. Metadata records real batch size, effective CFG batch size `2B`, stream order, and grouping.
