# Notice and provenance

This is an **unofficial SeaCache-style port for PixelGen**, not an official
SeaCache or PixelGen integration.

Audited sources:

- `$ROOT/baselines/SeaCache`, commit
  `8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2`, especially
  `FLUX/util_seacache.py` and `FLUX/seacache_generate.py`;
- `$ROOT/third-party/PixelGen`, represented by PixARC commit
  `d54c1e26768d80bf7c067f50e28868cdbf59d431`. It is not an independent Git
  worktree, so no separate upstream PixelGen SHA can be proven locally.

Neither the audited SeaCache clone nor the local PixelGen snapshot contains a
`LICENSE`, `COPYING`, or `NOTICE` file. No license is inferred. This provenance
record is not a license grant; redistribution requires separate review. The
port independently reimplements only the minimum filter/gate/body split and
does not copy a complete upstream model file.

