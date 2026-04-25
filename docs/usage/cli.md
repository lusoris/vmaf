# `vmaf` — command-line reference

`vmaf` is the main CLI binary shipped with this fork. It takes a reference /
distorted video pair, runs one or more VMAF models (plus any additional feature
extractors), and writes per-frame + pooled scores to an XML / JSON / CSV /
subtitle log.

> **Scope.** This page is the canonical flag reference for the `vmaf` binary
> in the Lusoris fork. It supersedes the abbreviated help string in
> [`libvmaf/tools/README.md`](../../libvmaf/tools/README.md) — the code's
> `--help` is authoritative for the *set* of flags at any given commit; this
> page adds defaults, interactions, and runnable examples per
> [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md).
>
> For the `vmaf_bench` micro-benchmark binary, see [bench.md](bench.md).
> For FFmpeg integration (the `libvmaf` filter), see [ffmpeg.md](ffmpeg.md).
> For the Python bindings, see [python.md](python.md).

## Quick start

```shell
# .y4m pair — no geometry flags needed
vmaf --reference ref.y4m --distorted dist.y4m

# .yuv pair — geometry is mandatory
vmaf \
  --reference ref.yuv \
  --distorted dist.yuv \
  --width 1920 --height 1080 \
  --pixel_format 420 --bitdepth 8 \
  --output scores.xml
```

Default behaviour when no `--model` is passed: the built-in `vmaf_v0.6.1`
model is loaded automatically. Default output format when no
`--xml|--json|--csv|--sub` is passed: XML.

## Required input flags

| Flag | Short | Argument | Required | Notes |
| --- | --- | --- | --- | --- |
| `--reference` | `-r` | path | **yes** | `.y4m` or `.yuv` path. |
| `--distorted` | `-d` | path | **yes** | `.y4m` or `.yuv` path. |
| `--width` | `-w` | unsigned | **yes for `.yuv`** | Ignored for `.y4m` (embedded). |
| `--height` | `-h` | unsigned | **yes for `.yuv`** | Ignored for `.y4m`. |
| `--pixel_format` | `-p` | `420` \| `422` \| `444` | **yes for `.yuv`** | 420 covers the overwhelming majority of streamable content. |
| `--bitdepth` | `-b` | `8` \| `10` \| `12` \| `16` | **yes for `.yuv`** | 10 and 12 bit require a 10-/12-bit aware model (e.g. `vmaf_b_v0.6.3` for banding sensitivity). |

If any of `--width`, `--height`, `--pixel_format`, `--bitdepth` is supplied
the input is treated as raw YUV and **all four** become mandatory.

## Models

The `--model / -m` flag takes a colon-delimited key/value string:

```text
--model path=<file>         # load a .json model from disk
--model version=<builtin>   # load a built-in model by name
--model path=...:name=<str> # rename the metric in the output log
--model version=...:disable_clip          # disable score clipping to [0, 100]
--model version=...:enable_transform      # apply transform
```

Built-in model versions (compiled into `libvmaf` via `-Dbuilt_in_models=true`,
default `true`):

| Version | Purpose |
| --- | --- |
| `vmaf_v0.6.1` | Default. 1080p training set, classic release. |
| `vmaf_v0.6.1neg` | Negative-gain (NEG) — non-enhancing; recommended for encoder A/B where one encoder may artificially sharpen. |
| `vmaf_b_v0.6.3` | Banding-aware variant (used with CAMBI). |
| `vmaf_4k_v0.6.1` | 4K training set. |
| `vmaf_4k_v0.6.1neg` | 4K + NEG. |

Float-precision variants (`vmaf_float_v0.6.1`, `vmaf_float_v0.6.1neg`,
`vmaf_float_b_v0.6.3`, `vmaf_float_4k_v0.6.1`) are also resolvable but are
legacy — prefer the integer versions for performance, the float versions
only for bit-exact comparison with older reports.

`--model` can be passed multiple times to run several models in one pass; each
model must have a unique `name=` (or the CLI errors out). Example running VMAF +
VMAF-NEG side-by-side:

```shell
vmaf -r ref.y4m -d dist.y4m \
  --model version=vmaf_v0.6.1:name=vmaf \
  --model version=vmaf_v0.6.1neg:name=vmaf_neg \
  --output scores.json --json
```

## Additional features

The `--feature` flag enables extra metrics (beyond whatever the model already
consumes). Syntax is the same colon-delimited key/value form as `--model`:

```shell
--feature psnr
--feature psnr=enable_chroma=true:enable_apsnr=true
--feature float_ssim=enable_db=true:clip_db=true
--feature cambi
--feature ciede
--feature psnr_hvs
```

See [../metrics/features.md](../metrics/features.md) for the full list of feature
identifiers and per-feature options.

## Output

| Flag | Default | Notes |
| --- | --- | --- |
| `--output` / `-o <path>` | stdout line + no file | Writes the per-frame + pooled log to `<path>`. |
| `--xml` | **default** | XML report (upstream-compatible). |
| `--json` | | JSON report. |
| `--csv` | | One row per frame. |
| `--sub` | | SubRip subtitle format — useful for overlaying scores during playback. |

Stderr always carries a short progress line plus the final pooled-mean VMAF score,
regardless of `--output`.

### Score precision (fork-added)

```text
--precision N          # printf "%.<N>g", N in 1..17
--precision max|full   # printf "%.17g" — IEEE-754 round-trip lossless (opt-in)
--precision legacy     # printf "%.6f"  — synonym for the default
```

The fork's default is `%.6f` (see
[ADR-0119](../adr/0119-cli-precision-default-revert.md), which supersedes
[ADR-0006](../adr/0006-cli-precision-17g-default.md)), matching upstream
Netflix output byte-for-byte so the CPU golden gate passes without explicit
flags. Pass `--precision=max` whenever you need IEEE-754 round-trip lossless
output (cross-backend numeric diff, archival reports, any consumer that
re-parses scores into doubles and compares them). Affects XML, JSON, CSV,
SUB, and stderr consistently.

See [precision.md](precision.md) for the full table of when to pick each mode.

## Backend selection

Build-time: each backend is opt-in via a meson flag. At **runtime**, backend
selection is per-invocation through flags on `vmaf` — there is **no** environment
variable that overrides it.

| Flag | Default | Effect |
| --- | --- | --- |
| `--no_cuda` | off | Forbid CUDA dispatch even if the CUDA backend is built in. |
| `--no_sycl` | off | Forbid SYCL dispatch even if the SYCL backend is built in. |
| `--sycl_device <N>` | auto (first GPU) | Pick SYCL device by ordinal from the oneAPI device list. |
| `--cpumask <bitmask>` (`-c`) | all ISAs enabled | Mask out specific CPU ISAs (e.g. force scalar, disable AVX-512). Values are fork-internal — see `libvmaf/src/cpu.h`. |
| `--gpumask <bitmask>` | all GPU ops enabled | Mask out specific GPU ops. |
| `--threads <N>` | host `nproc` | CPU-side worker thread count. |

If neither backend is built in, these flags are silently inert. See
[../backends/index.md](../backends/index.md) for the runtime-dispatch rules and
which features have GPU/SIMD twins.

## Frame range

```text
--frame_cnt <N>           # stop after N frames (both streams)
--frame_skip_ref <N>      # skip the first N frames of the reference
--frame_skip_dist <N>     # skip the first N frames of the distorted
--subsample <N>           # compute scores only every Nth frame (default 1 = all frames)
```

`--subsample` trades precision for speed — pooled scores are still computed
over the sampled subset, so keep it at 1 for final reports.

## Preset bundles

```text
--aom_ctc v1.0 | v2.0 | v3.0 | v4.0 | v5.0 | v6.0 | v7.0
--nflx_ctc v1.0
```

These expand to a canonical model + feature list for
[AOM](../metrics/ctc/aom.md) and Netflix common-test-conditions reports. For
example, `--aom_ctc v7.0` is equivalent to:

```text
--model version=vmaf_v0.6.1:name=vmaf
--model version=vmaf_v0.6.1neg:name=vmaf_neg
--feature psnr=reduced_hbd_peak=true:enable_apsnr=true:min_sse=0.5
--feature ciede
--feature float_ssim=scale=1:enable_db=true:clip_db=true
--feature float_ms_ssim=enable_db=true:clip_db=true
--feature psnr_hvs
--feature cambi
# plus common_bitdepth=on (forces reference + distorted to the same bitdepth)
```

`--aom_ctc proposed` is deprecated (errors out with an explanation).

## Tiny-AI flags (fork-added)

```text
--tiny-model <path>            # load a .onnx tiny model alongside classic models
--tiny-device auto|cpu|cuda|openvino|rocm    # ORT execution provider (default: auto)
--tiny-threads <N>             # CPU EP intra-op threads (0 = ORT default)
--tiny-fp16                    # request fp16 I/O where the EP supports it
--no-reference                 # NR mode; requires a no-reference tiny model
```

Underscore aliases (`--tiny_model`, `--tiny_device`, `--tiny_threads`, `--tiny_fp16`,
`--no_reference`) are accepted for scripting symmetry with the underscore flags
upstream uses.

`--no-reference` requires a `--tiny-model` whose registry entry declares
`"reference_required": false`. With no tiny model loaded the flag is
rejected at parse time; with a reference-required tiny model the run
fails at init with a diagnostic naming the model. In NR mode the
classic VMAF score and any reference-needing per-feature outputs are
omitted from the JSON / XML / CSV report — only the tiny-AI score and
NR-only metrics are emitted.

See [../ai/inference.md](../ai/inference.md) for the full tiny-AI CLI
walkthrough and the per-model registry (`model/tiny/registry.json`,
sha256 pins, known limitations).

## Logging and misc

| Flag | Short | Effect |
| --- | --- | --- |
| `--quiet` | `-q` | Disable the FPS meter when run in a TTY. |
| `--no_prediction` | `-n` | Skip final model prediction; extract features only. Useful for feeding raw features into a custom pool. |
| `--version` | `-v` | Print `libvmaf` version + git SHA and exit. |

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success. |
| 1 | Any parse / I/O / runtime error. `vmaf` writes a diagnostic to stderr before exiting. |

`libvmaf` does not currently surface granular error codes at the process
boundary; the specific `VMAF_ERR_*` code from the C API is logged to stderr
but collapsed to exit 1 from the CLI.

## Worked example — reproducing the upstream golden pair

Download the canonical Netflix test pair from upstream:

```shell
curl -sSLO https://github.com/Netflix/vmaf_resource/raw/master/python/test/resource/yuv/src01_hrc00_576x324.yuv
curl -sSLO https://github.com/Netflix/vmaf_resource/raw/master/python/test/resource/yuv/src01_hrc01_576x324.yuv
```

Run VMAF plus PSNR:

```shell
./build/libvmaf/tools/vmaf \
  --reference  src01_hrc00_576x324.yuv \
  --distorted  src01_hrc01_576x324.yuv \
  --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
  --model version=vmaf_v0.6.1 \
  --feature psnr \
  --output scores.xml
```

Expected stderr (CPU path):

```text
VMAF version 3.x.y-lusoris.N
48 frames  44.72 FPS
vmaf_v0.6.1: 76.6689050197
```

Expected `scores.xml` head:

```xml
<VMAF version="3.x.y-lusoris.N">
  <params qualityWidth="576" qualityHeight="324" />
  <fyi fps="41.98" />
  <frames>
    <frame frameNum="0" integer_adm2="0.96208412..." ... psnr_y="34.76077932..." vmaf="83.8562851..." />
    ...
  </frames>
  <pooled_metrics>
    <metric name="vmaf" min="71.17655..." max="87.18142..." mean="76.66890501..." harmonic_mean="76.51000634..." />
    ...
  </pooled_metrics>
</VMAF>
```

Pooled-mean VMAF for this pair is **76.668905…**. This is one of the three Netflix
CPU goldens preserved verbatim as a required CI gate — see
[ADR-0024](../adr/0024-netflix-golden-preserved.md) and
[`python/test/quality_runner_test.py`](../../python/test/quality_runner_test.py).

## Flag interactions and pitfalls

- **`.yuv` without geometry**. Passing `--reference foo.yuv` without
  `--width/--height/--pixel_format/--bitdepth` errors out. `.y4m` carries geometry
  in the header; `.yuv` does not.
- **Duplicate model names**. Each `--model` must have a unique `name=`. If the same
  built-in version is loaded twice, set `name=` explicitly on at least one.
- **`--no_prediction` with `--model`**. `--no_prediction` skips model prediction
  but does not skip loading — the model is still used to select which features to
  extract. Omit `--model` entirely plus pass `--no_prediction` to extract only the
  features listed via `--feature`.
- **Default `%.6f` truncation**. The default (and `--precision legacy`)
  truncates differences ≤ 1e-6 that would be distinguishable under
  `--precision=max`. Use `max` whenever you need to compare scores
  numerically (cross-backend diff, archival reports). The default mode
  exists for byte-for-byte agreement with pre-fork Netflix output, which
  the CPU golden gate depends on.
- **`--tiny-model` vs `--model`**. These compose — tiny-AI models are
  **additional** scores layered on top of the classic SVM/XGBoost prediction, not
  a replacement for it. Use `--no_prediction` if you want tiny scores alone. See
  [ADR-0023](../adr/0023-tinyai-user-surfaces.md).
- **`--no_cuda` + `--no_sycl` together**. Forces CPU-only even on a build with both
  GPU backends compiled in. Useful for cross-backend diff sessions.

## Related

- [bench.md](bench.md) — `vmaf_bench` micro-benchmark harness.
- [ffmpeg.md](ffmpeg.md) — using the VMAF filter inside `ffmpeg`.
- [python.md](python.md) — Python bindings for the CLI.
- [precision.md](precision.md) — dedicated `--precision` flag walkthrough.
- [../backends/index.md](../backends/index.md) — runtime backend dispatch rules.
- [../metrics/features.md](../metrics/features.md) — per-feature identifiers
  and options.
- [../ai/inference.md](../ai/inference.md) — tiny-AI inference walkthrough.
- [ADR-0119](../adr/0119-cli-precision-default-revert.md) (current
  precision default; supersedes
  [ADR-0006](../adr/0006-cli-precision-17g-default.md)),
  [ADR-0023](../adr/0023-tinyai-user-surfaces.md),
  [ADR-0024](../adr/0024-netflix-golden-preserved.md),
  [ADR-0100](../adr/0100-project-wide-doc-substance-rule.md).
