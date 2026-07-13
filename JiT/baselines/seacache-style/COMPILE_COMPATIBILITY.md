# JiT compile compatibility

No GPU compile experiment was run during this implementation. Every conclusion
below is based on static inspection and CPU mocks and must be validated after
the active reference workload finishes.

## Upstream behavior

The local JiT decorates every `JiTBlock.forward` and `FinalLayer.forward` with
`@torch.compile` (`third-party/JiT/model_jit.py:175-180,197-202`). The complete
`JiT.forward` is not decorated. Consequently the native behavior is already
blockwise compilation, with Python orchestration between compiled blocks.

## SeaCache graph boundary

The cache controller contains mutable Python state, a scalar relative-L1 gate,
dynamic full/reuse branching, trace bookkeeping, and intentional host
synchronization. It must remain outside a `fullgraph=True` region. The local
adapter calls the same upstream block and final-layer modules, so the intended
safe boundary is:

- eager: embeddings, probe, SEA FFT, scalar gate, cache state, branch;
- compiled where supported: exact Transformer blocks and final layer;
- eager: sampler trajectory lifecycle and trace aggregation.

The reuse branch does not execute the compiled Transformer body. This may
change graph-cache behavior and requires GPU observation.

## Compile modes

| Mode | Intended meaning | Current status |
|---|---|---|
| `blockwise` | Full and SeaCache both use upstream block/final decorators; controller stays eager | Implemented; GPU behavior unverified |
| `matched_eager` | Bind each block/final layer's `_torchdynamo_orig_callable` on each model instance for both methods | Implemented with fail-fast wrapper count; GPU parity unverified |
| `upstream` | Native upstream Full behavior | Must not be compared to a different SeaCache compile mode |

The generation and benchmark runners accept all three labels and apply them to
the actual model instance. Both shipped YAMLs request `matched_eager`. This
instance-level unwrapping changes no upstream class and must still pass deferred
GPU output parity. Never divide compiled Full time by eager SeaCache time.

## Deferred checks

After GPU availability, record for Full, force-full-with-gate, and SeaCache:

1. checkpoint and output correctness;
2. Dynamo graph breaks and recompilation count;
3. first-call compile time separately from steady state;
4. warm-up convergence;
5. peak allocated/reserved memory;
6. identical compile mode in the speedup numerator and denominator.

Compilation time is not generation latency. The main speedup is median Full
milliseconds/image divided by median SeaCache milliseconds/image after at least
10 warm-up and 30 measured batches.
