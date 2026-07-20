# PixelGen adapter

This directory contains the PixelGen model, sampler, Lightning lifecycle and
durable trace adapter for the shared implementation in
`JiT/methods/pixel-remainder-taylor`.

The sampler preserves upstream combined CFG ordering
`[unconditional, conditional]`, computes the controller on the real B samples,
and performs exactly one combined model forward per NFE. For exact 50-step Heun
that is 99 forwards, identical to the existing PixelGen TaylorSeer path.

Use the shared [runbook](../../../JiT/methods/pixel-remainder-taylor/README.md)
for CPU tests, four-GPU smoke, the new-method-only 1K sweep, evaluation and
result merging.
