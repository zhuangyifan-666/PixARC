# Probe and resume semantics

The probe is the exact output after the first `probe_depth` JiT blocks. Main-profile depth is 1. `ProbeInternalState` stores the complete token tensor, the next block index, and whether context has already been inserted; `probe_feature` is only the image-token view.

JiT inserts 32 class-context tokens before block 4. Before insertion, image tokens occupy the full sequence. After insertion, context is the prefix and image tokens are the suffix. Resume selects the correct RoPE for every remaining block, inserts context at most once, and removes it once at the body boundary. A refresh resumes from `next_block_index`; it never recomputes the prefix.

Three paths exist:

- `DIRECT_FULL`: warmup, missing history/anchor, forced last call, diagnostic, or Full mode. The model crosses every block exactly once while capturing the prefix feature in-line.
- `FULL_RESUME_FROM_PROBE`: an eligible probe reaches the threshold; remaining blocks resume from the saved internal state.
- `REUSE`: the exact prefix runs, the suffix is skipped, DCTA estimates the body residual, and the fresh final head runs.

`probe_count` and `probe_time_ms` count only eligible gate probes, matching PixelGen. A direct Full prefix capture is not a second probe: its prefix time is included in `suffix_time_ms`. This convention prevents cross-backend denominators from mixing eligible probes with normal Full prefix work.

Probe errors reduce over the complete real batch within one CFG stream. Main-profile formulas contain no epsilon. A zero denominator can produce a non-finite value; the preregistered policy determines whether that forces Full/reset or follows the official comparison behavior.
