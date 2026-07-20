# PixelGen adapter

This directory contains the PixelGen model, sampler, Lightning lifecycle and
durable trace adapter for the shared implementation in
`JiT/methods/pixel-remainder-taylor`.

The sampler preserves upstream combined CFG ordering
`[unconditional, conditional]`, computes the controller on the real B samples,
and performs exactly one combined model forward per NFE. For exact 50-step Heun
that is 99 forwards, identical to the existing PixelGen TaylorSeer path.

PixelGen child YAML files retain `extends` only as authoring inputs. Before a
production launch, the shared materializer expands the complete parent chain,
sets `template_only: false`, resolves the checkpoint to an existing absolute
path, and writes canonical `config_resolved.yaml`. The untouched child bytes
are separately archived as `input_config.yaml`. A resume rejects changes to
either the child bytes or the re-resolved parent semantics. The PixelGen
generator reads these launcher-owned files and never rewrites them.

Use `/root/miniconda3/envs/pixelgen/bin/python` with the shared
`scripts/preflight_run.py --model PixelGen` before any GPU command. The
preflight uses a temporary directory and makes no GPU query.

Use the shared [runbook](../../../JiT/methods/pixel-remainder-taylor/README.md)
for CPU tests, four-GPU smoke, the new-method-only 1K sweep, evaluation and
result merging.
