# Pixel-Remainder Taylor configuration-pipeline repair report

## Status

- **1K readiness: FAIL**
- **50K infrastructure readiness: PASS**
- Original HEAD: `e2727d8d3bea6aa41886bcba52a541d0a5675aff`
- Repair branch: `codex/prt-configfix-20260720T110522Z`
- Clean repair worktree:
  `/mnt/iset/nfs-main/private/zhuangyifan/PixARC_prt_configfix_20260720T110522Z`
- Verified implementation commit:
  `11ff4ee381648b5934a20176279f75bb12c531a4`
- Formal commands: `READY_1K_COMMANDS.md`

The implementation commit exists in the current object database and was the
clean `HEAD` used for the ten production preflights and both clean 50K
preflights. The final documentation is a follow-up commit so this report can
refer to a real implementation hash.

The requested root filename `CODEX_TASK_PRT_CONFIG_FIX.md` was absent. The
untracked 788-line root file `PixARC_Codex_ConfigPipeline_Repair_Prompt.md`
contained the exact requested task and was used as the task source. The
original checkout was preserved with its pre-existing status:

```text
 D PixARC_Codex_Repair_1K_50K_Prompt.md
?? PixARC_Codex_ConfigPipeline_Repair_Prompt.md
```

No reset, clean, restore, baseline run, or edit to the original dirty worktree
was performed.

## Original production blockers reproduced

### JiT raw/resolved collision

The original JiT `t0p02` YAML was snapshotted to
`config_resolved.yaml`, then the generator's canonical serialization was
calculated:

```text
source_bytes=667 normalized_bytes=678 equal=False
FileExistsError: immutable run input differs:
  /tmp/pixarc-prt-config-repro-jit.VfrIQo/config_resolved.yaml
```

This is a real byte mismatch, not a hypothetical code inspection result.

### PixelGen archived child loses its parent

The real PixelGen `t0p02` child was copied alone to a temporary run directory
and loaded from that new location:

```text
extends: pixelgen_xl_256_base.yaml
FileNotFoundError: .../pixelgen_xl_256_base.yaml
```

The failure was reproduced at
`/tmp/pixarc-prt-config-repro-pixelgen.XazH39/config_resolved.yaml`.

## Repair design

Every production run now uses four launcher-owned immutable inputs:

```text
input_config.yaml                 exact authoring YAML bytes
config_resolved.yaml              canonical, self-contained production YAML
input_manifest.jsonl              exact manifest bytes
input_manifest.jsonl.meta.json    exact resolved sidecar bytes
```

- `materialize_config.py` reuses the shared `load_config`, expands PixelGen
  `extends`, validates the root mapping, forces `template_only: false`, resolves
  the model-specific checkpoint to an existing absolute path, emits one
  canonical YAML representation, and verifies the result from an isolated
  temporary directory.
- One hard-link-based immutable writer is shared by config materialization and
  raw input snapshots. An identical existing file succeeds; different bytes
  fail without overwrite.
- New and resumed launcher invocations archive raw YAML separately and
  re-materialize the complete current parent chain into a fresh temporary file
  before comparing it to the archived resolved YAML.
- A formatting-only child change fails the raw-byte check. An unchanged child
  with a changed inherited parent fails the resolved-byte/semantic check.
- Both generators require launcher-owned `config_resolved.yaml`,
  `input_config.yaml`, manifest and sidecar paths. They validate and read these
  files but no longer compete with the launcher to write them.
- Relative checkpoints fail in production generators. The only production
  input is a materialized configuration with an absolute existing checkpoint.
- Run manifests and per-image metadata now distinguish:
  `input_config_sha256`, `resolved_config_sha256`, and
  `semantic_config_hash`. The old `input_config_hash` remains only as an
  explicitly documented semantic-hash compatibility field.
- Resume identity still includes manifest, sidecar, checkpoint size, method
  source tree, Git/Python/PyTorch/CUDA and all prior sampling/NFE fields.
- `preflight_run.py` performs the full config/checkpoint/manifest/sidecar/count
  chain in a temporary directory, requires a clean executable worktree, makes
  zero GPU queries, and does not touch the formal output directory.
- Dynamic non-uniform interpolation, fixed legacy parity, PixelGen DataModule
  binding, 99-NFE/198-or-99-forward contracts, summary tracing, and cumulative
  invocation timing were not changed.

## Changed files in implementation commit

```text
JiT/methods/pixel-remainder-taylor/README.md
JiT/methods/pixel-remainder-taylor/RUNBOOK_1K_50K.md
JiT/methods/pixel-remainder-taylor/pixel_remainder_taylor/__init__.py
JiT/methods/pixel-remainder-taylor/pixel_remainder_taylor/config.py
JiT/methods/pixel-remainder-taylor/scripts/generate_shard.py
JiT/methods/pixel-remainder-taylor/scripts/launch_4gpu.sh
JiT/methods/pixel-remainder-taylor/scripts/materialize_50k_config.py
JiT/methods/pixel-remainder-taylor/scripts/materialize_config.py
JiT/methods/pixel-remainder-taylor/scripts/preflight_all_configs.sh
JiT/methods/pixel-remainder-taylor/scripts/preflight_run.py
JiT/methods/pixel-remainder-taylor/scripts/snapshot_input.py
JiT/methods/pixel-remainder-taylor/scripts/validate_outputs.py
JiT/methods/pixel-remainder-taylor/tests/test_config_pipeline.py
JiT/methods/pixel-remainder-taylor/tests/test_protocol_scripts.py
PixelGen/methods/pixel-remainder-taylor/README.md
PixelGen/methods/pixel-remainder-taylor/pixel_remainder_taylor/__init__.py
PixelGen/methods/pixel-remainder-taylor/pixel_remainder_taylor/pixelgen_io.py
PixelGen/methods/pixel-remainder-taylor/scripts/generate_shard.py
PixelGen/methods/pixel-remainder-taylor/tests/conftest.py
PixelGen/methods/pixel-remainder-taylor/tests/test_config_pipeline.py
PixelGen/methods/pixel-remainder-taylor/tests/validate_real_manifest.py
READY_1K_COMMANDS.md
```

No file below `third-party/` or `results/` changed. No baseline implementation,
frozen manifest, sidecar, existing image, CSV, metric, or checkpoint changed.

## Frozen identities

| Model | 1K manifest SHA256 | sidecar SHA256 | records |
|---|---|---|---:|
| JiT | `e8ddfb2a2470661b7fbc46bd9077c2432195ae2b6986a5b466a760f68797bc1c` | `2762c6930dbf06979d2e7d9fd10fd8f749335df5aa7a92198f1afc5e84457f54` | 1000 |
| PixelGen | `31536470eacf69e07ccd72305e7866957d15859b2091eec7daed2a309cedf5c0` | `bad5838c17778acd0ab32234e2c68888b1dca965b34f8590c44251964880840f` | 1000 |

Executable identities at implementation commit:

- Shared JiT method tree:
  `70623a6dc72e33d944d02be0e95160213d11767746e0c6044bf8543d4cfdce82`
- PixelGen adapter tree:
  `ff757704a63b82021927622498e83c7040403e8cb0716c532b6ab260bdb163cb`
- Combined PixelGen production identity:
  `ed0ff31d7c68935acbb99e4635c739f03a16449f240e2b5ce528b45bdda7049d`

Verified checkpoint identities:

- JiT: 1,576,057,392 bytes,
  `/mnt/iset/nfs-main/private/zhuangyifan/PixARC/JiT/checkpoints/JiT-B-16-256/checkpoint-last.pth`
- PixelGen: 2,722,303,321 bytes,
  `/mnt/iset/nfs-main/private/zhuangyifan/PixARC/PixelGen/checkpoints/PixelGen_XL_160ep.ckpt`

## CPU and integration test results

```text
JiT/methods/pixel-remainder-taylor/scripts/run_all_cpu_tests.sh
  JiT:      51 passed in 4.92s
  PixelGen: 13 passed in 2.12s
  PixelGen frozen manifest production validation: PASS
  PixelGen five real config preflights: PASS
  PixelGen per-image raw/resolved/semantic identity metadata: PASS

PYTHONPYCACHEPREFIX=/tmp/pixarc-prt-configfix-compile \
  /root/miniconda3/envs/jit/bin/python -m compileall -q \
  JiT/methods/pixel-remainder-taylor \
  PixelGen/methods/pixel-remainder-taylor
  result: PASS

bash -n JiT/methods/pixel-remainder-taylor/scripts/*.sh
  result: PASS

git diff --check
  result: PASS

git diff --name-only HEAD -- third-party results
  result: no output
```

Tests exercise the real five JiT configurations, real five PixelGen child
configurations, portable isolated loading, idempotent materialization, raw
whitespace rejection, inherited-parent change rejection, tampered resolved
archive rejection, PixelGen Dataset metadata, real frozen manifests, CRLF and
sidecar rejection, no-op/partial resume, cumulative timing, non-uniform
quadratic counterexample, 99 NFE and 198/99 forward counts, and full/summary
trace parity.

`ruff`, `pyflakes`, and `shellcheck` were not installed. No environment or
package installation was performed.

## Ten real production preflights

Command:

```bash
JiT/methods/pixel-remainder-taylor/scripts/preflight_all_configs.sh
```

All ten ran from clean implementation commit `11ff4ee` and returned
`status=PASS`, `gpu_queries=0`.

| Model/config | input SHA256 | resolved SHA256 | semantic hash |
|---|---|---|---|
| JiT Full | `59485d8ed3648e1c405bb5c39c238549b117a1aa61e3a290bd31436a70ae8516` | `080b2fc7d39b252ced1e6908ab80bcec15ebf1d672047f69f20fc6baa1a319cc` | `d0fa7685a355dd9af9a50a992b637a026e34a41739a527851db57531a558a7c5` |
| JiT fixed | `8eaab2bbf1cdf23cf15ec3c575f8a98e03160d81b2dfcdf3642444b517cbb92c` | `cfdd7b4e0f139db9dc433cb988e2f3727984eeb4a90fc11e19ddee3b2c9adb5a` | `7b3a918ff6ebffae094ab4e2de6464354155c8c4e0c1be78579349770d1872e0` |
| JiT tau 0.01 | `81e70994a84ff215afaa87e9c43b406d0530892877a9f005735d9181800332f3` | `25f49d696e487d4b46e44ec493968045a1b7a993246274ae114f3a91185dde1e` | `6ed458ab4f5dc89014f9e615f93d4b9bf1eaca6dc0aa306a2bcf9ae3a8021902` |
| JiT tau 0.02 | `539e535a42b93bbbb88617f4c20d28b8d0e75ae78bd3a2228195e20504117262` | `bcd4125dad31cf3e409aec506ea3e402e1fb5da09c86401914e3bdb059d89a25` | `03d0ee68ab14c309708394f93917fc6badef21da25fbc4e80854dd5adf5c5fcf` |
| JiT tau 0.04 | `dc7c053a71700e84ee83f90953c437f87ebdec03fe0158ad42a6b594a67088e8` | `5b3b8d070d03a1ce1149c24e53874e92dad65cc7371c8746e07a9eafe561a27b` | `476989fe8fcea7ca8ed4e5040a138fd7e3fee245139297a5ce05d8d298068786` |
| PixelGen Full | `1698912ab2c59496c5393ab12a781fcc7761c1fea66788afd31594fb8d9538a9` | `981b2dafbcb7e2e257327b6f11cb9ea4f118b3d6da68962d8b025cc4a1d16289` | `9225510e0f3adbed841ccbea6420e04b5e27e32f6105f97c518bcd2ffa8d54d4` |
| PixelGen fixed | `f44bc005c038102ec30adfe36b04f74090d0fa56cf6c0c12ec4c2f9f4a65a767` | `23d7d25162a9459a889ad4c684f3f75bb4196644ffd6779ae8c0ce4c83b3dbe0` | `009467ea1d40914ea8a92a633d4290ce2b773478fb089bfbe76101568ae98ff1` |
| PixelGen tau 0.01 | `cf8f08f2736e0b0769d0a3ea5f6ee74c407285eca2badda7ca7edd4155b9386c` | `17851870f346c2416a9cf56fc97a979ca819c7959a159b6cd48172af1a00263b` | `8cffc04407de649e89df10466db012ed8c2666b5b654ce3502b79a4c97c74fb9` |
| PixelGen tau 0.02 | `efed43238246e6b9d70465f0ac3a8082261c5b3ca466b9707d7637f0229a1d90` | `7d1c5e35eda4745c1e07e06258ab6d71f21f4d1a34a1f57bc51a0c7639722c72` | `81c2d0f97f649b179eeeb5651a76c62fa947dede8f86e91bf1777eee725275ca` |
| PixelGen tau 0.04 | `42ebb6a32b6f296fb1f20b1d0f7eafc4b89a6368f24f84e2b243ee710ab06613` | `9badde02c666c314ea2938e8f8c0869103838c77893d9b3c48f5bdea613c2f79` | `9304185c81e3d32bac59913bd779228823627e6fc241a68ed50e4261e39598bb` |

The launcher integration was also invoked without GPU authorization. It first
completed the real JiT preflight, then refused GPU work with exit code 3. The
requested formal output path did not exist afterward, proving the preflight
does not pollute it.

## 50K no-model dry run

Artifacts are external at `/tmp/pixarc-prt-configfix-50k.gg2chN`.

| Model | records | per class | per shard | manifest SHA256 | sidecar SHA256 |
|---|---:|---:|---:|---|---|
| JiT | 50000 | 50 | 12500 | `ab95d51ba387ee312aab9ba5232d7d2e89dbcf9ae26f33e80efd7ac52bf3580b` | `9d34b6494ff75594aaa7894d972d5db77ce956ab4892f45a0db512ed04b9cf16` |
| PixelGen | 50000 | 50 | 12500 | `401b7929de19a6611c95710c9e4f0481b3c2d90abc12a0f0d45afa080e2f0075` | `731c8d0153801d59f382b7d5d7d6e288393099c39111539f23c77f68b17728ec` |

Both seed sets were validated disjoint from the corresponding frozen 1K set.
Both materialized configs independently loaded with `tau=0.02`,
`max_taylor_span=3`, `trace_mode=summary`, and the absolute checkpoint above.
Both clean CLI preflights passed with `expected-count=50000` and zero GPU
queries. No model and no 50K image generation was run.

## GPU smoke and exact blocker

The mandatory GPU matrix was not attempted:

```text
$ nvidia-smi -L
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
exit code: 9

$ ls -l /dev/nvidia*
No such file or directory

JiT torch 2.5.1+cu124:
  cuda_available=False, device_count=0, compiled CUDA=12.4
PixelGen torch 2.7.1+cu126:
  cuda_available=False, device_count=0, compiled CUDA=12.6
```

Therefore the following release gates remain incomplete:

- JiT four-GPU Full, fixed, tau 0.01 and tau 0.04 smoke;
- PixelGen four-GPU Full, fixed, tau 0.01 and tau 0.04 smoke;
- Full and fixed pixel alignment against existing reference images;
- dynamic Taylor occurrence/monotonicity and real GPU NFE/forward traces;
- one controlled four-GPU resume and cumulative timing validation.

No GPU, pixel-alignment, timing, or recovery result was fabricated. Because
these gates are mandatory, **1K readiness remains FAIL** and none of the six
formal 1K runs was started.

## Formal 1K launch commands

`READY_1K_COMMANDS.md` contains six explicit commands for:

```text
JiT:      tau=0.01, 0.02, 0.04; max_taylor_span=3
PixelGen: tau=0.01, 0.02, 0.04; max_taylor_span=3
```

Each command includes the model-family Python, GPU authorization, frozen 1K
manifest, authoring config, expected count 1000, and a distinct external output
root. The file also states exactly when `--resume` is allowed. These commands
are prepared deliverables, not permission to launch while readiness is FAIL.

## Next commands

On a host exposing four idle allocated GPUs:

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixARC_prt_configfix_20260720T110522Z
export PIXEL_REMAINDER_CONFIG_SOURCE=/mnt/iset/nfs-main/private/zhuangyifan/PixARC
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PIXEL_REMAINDER_GPU_RUN_ALLOWED=1

nvidia-smi -L
JiT/methods/pixel-remainder-taylor/scripts/run_all_cpu_tests.sh
JiT/methods/pixel-remainder-taylor/scripts/preflight_all_configs.sh
```

Then execute the `Required four-GPU smoke matrix`, pixel-comparison, and
controlled-resume sections of
`JiT/methods/pixel-remainder-taylor/RUNBOOK_1K_50K.md`. Only after every GPU
gate passes may the six commands in `READY_1K_COMMANDS.md` be used.
