# Change Log

> The "Unreleased / lusoris fork" section below tracks fork-specific changes
> on top of upstream Netflix/vmaf. From here on, release-please generates
> entries automatically from Conventional Commits.

## [Unreleased] — lusoris fork (3.0.0-lusoris.0)

### Added

<<<<<<< HEAD
- **`fr_regressor_v2` ensemble — real-corpus retrain harness +
  flip workflow (ADR-0309).** Follow-up to
  [ADR-0303](docs/adr/0303-fr-regressor-v2-ensemble-prod-flip.md) /
  PR #399 that ships the operational harness for actually running
  the 5-seed × 9-fold LOSO retrain against the locally available
  Netflix Public Dataset (`.workingdir2/netflix/`) and emitting a
  machine-checkable verdict file. Adds
  [`ai/scripts/run_ensemble_v2_real_corpus_loso.sh`](ai/scripts/run_ensemble_v2_real_corpus_loso.sh)
  (Bash wrapper that validates the corpus, loops the seeds through
  `train_fr_regressor_v2_ensemble_loso.py`, and tees timestamped
  per-seed logs under `runs/ensemble_v2_real/logs/`),
  [`ai/scripts/validate_ensemble_seeds.py`](ai/scripts/validate_ensemble_seeds.py)
  (Python validator that calls the ADR-0303 gate, snapshots the
  corpus YUV file list as sha256 over sorted `relpath\tsize`, and
  writes `PROMOTE.json` on gate-pass with a recommendation to flip
  the five `fr_regressor_v2_ensemble_v1_seed{0..4}` rows in
  `model/tiny/registry.json` from `smoke: true` to `smoke: false`,
  or `HOLD.json` on gate-fail with the failing-seed details and a
  recommendation to keep `smoke: true` and investigate diversity /
  hyperparameters), unit tests for both verdict paths, and a
  runbook
  [`docs/ai/ensemble-v2-real-corpus-retrain-runbook.md`](docs/ai/ensemble-v2-real-corpus-retrain-runbook.md)
  covering prerequisites, the two-command run, verdict
  interpretation, and rollback. The harness deliberately does
  **not** run the LOSO inside the PR (6–12 h GPU work) and does
  **not** flip the registry — the registry flip is a separate
  follow-up PR gated on a passing `PROMOTE.json` (preserves a clean
  revert surface and honours the new `ai/AGENTS.md` invariant that
  registry-flip never happens during a rebase). Companion research
  digest:
  [Research-0081](docs/research/0081-fr-regressor-v2-ensemble-real-corpus-methodology.md).
=======
- **libFuzzer harness expansion — `fuzz_yuv_input` + `fuzz_cli_parse`
  (ADR-0311, extends ADR-0270).** Lands two additional harnesses
  alongside [`fuzz_y4m_input`](libvmaf/test/fuzz/fuzz_y4m_input.c):
  [`fuzz_yuv_input.c`](libvmaf/test/fuzz/fuzz_yuv_input.c) wraps
  the headerless raw-YUV reader (`tools/yuv_input.c`) via
  `raw_input_open` against a fixed 32x32 envelope, rotating
  through six (pix_fmt, bitdepth) combos selected from the
  fuzzer's first input byte; up to 8 frames per iteration.
  [`fuzz_cli_parse.c`](libvmaf/test/fuzz/fuzz_cli_parse.c) wraps
  the CLI argv tokeniser + the colon-delimited
  `parse_model_config` / `parse_feature_config` sub-parsers
  (`tools/cli_parse.c`); treats fuzzer bytes as a NUL-separated
  argv vector (capped at 64 args) and uses a `-Wl,--wrap=exit`
  linker shim with a `setjmp`/`longjmp` trampoline to intercept
  `usage()`'s `exit(1)` so a single bad input does not terminate
  the fuzzer process. Each harness ships 6 hand-crafted seeds
  (8/10-bit × 4:2:0/4:2:2/4:4:4 plus a truncated-frame seed for
  YUV; minimal/feature/model/yuv-flag/help shapes for CLI). The
  nightly [`.github/workflows/fuzz.yml`](.github/workflows/fuzz.yml)
  matrix gains the two new targets at 60 s/harness/night
  (≈ 3 minutes wall total, well under the 15-minute job cap).
  Operator runbook updated in
  [`docs/development/fuzzing.md`](docs/development/fuzzing.md);
  ADR-0270 stays the parent scaffold ADR. Companion research
  digest:
  [`docs/research/0083-libfuzzer-harness-expansion-target-survey.md`](docs/research/0083-libfuzzer-harness-expansion-target-survey.md)
  ranks the wider surface inventory and queues `fuzz_model_load`
  + `fuzz_sidecar` as the next follow-up pair.

>>>>>>> 00d26013 (feat(security): libFuzzer harnesses — yuv_input + cli_parse (extends ADR-0270))
- **`fr_regressor_v2` codec-aware scaffold — first downstream consumer
  of the vmaf-tune Phase A JSONL corpus (ADR-0272, prereq for
  Phase B).** Ships
  [`ai/scripts/train_fr_regressor_v2.py`](ai/scripts/train_fr_regressor_v2.py)
  — a scaffold-only trainer that consumes the JSONL corpus emitted by
  `vmaf-tune corpus` (ADR-0237 Phase A) and trains the codec-aware
  variant of the v1 FR regressor. Two-input ONNX (`features` shape
  `(N, 6)` canonical-6 + `codec` shape `(N, 8)` block —
  `[encoder_onehot(6), preset_norm, crf_norm]`); reuses the existing
  `FRRegressor(num_codecs=8)` class plumbed by ADR-0235. A `--smoke`
  mode synthesises 100 fake corpus rows and trains 1 epoch so the
  pipeline is end-to-end exercisable in CI without hours of encode
  time. Registers `fr_regressor_v2` in `model/tiny/registry.json`
  with `smoke: true` until a follow-up PR runs production training on
  a real Phase A corpus and clears the ADR-0235 ship gate (≥0.005
  multi-codec PLCC lift over v1's 0.95 LOSO floor). Doc surface:
  [model card](docs/ai/models/fr_regressor_v2.md),
  [research digest](docs/research/0058-fr-regressor-v2-feasibility.md),
  [ADR-0272](docs/adr/0272-fr-regressor-v2-codec-aware-scaffold.md),
  `ai/AGENTS.md` invariant note pinning the codec block layout and
  encoder vocabulary. Smoke validated locally (`python
  ai/scripts/train_fr_regressor_v2.py --smoke` produces a valid
  opset-17 two-input ONNX, op-allowlist clean, torch-vs-ORT roundtrip
  within 1e-4 atol). No upstream-mirror file touched; pure additive
  fork-local PR.
- **libFuzzer scaffold for the YUV4MPEG2 parser — `fuzz_y4m_input`
  (ADR-0270, OSSF Scorecard `Fuzzing` remediation).**
  Lands [`libvmaf/test/fuzz/`](libvmaf/test/fuzz/) with one
  initial harness (`fuzz_y4m_input.c`) wrapping the public
  `video_input_open` / `video_input_fetch_frame` /
  `video_input_close` triple via POSIX `fmemopen`. Opt-in
  via the new `-Dfuzz=true` Meson option (default OFF; requires
  clang). Ships a six-input seed corpus covering 420, 422,
  420p10, monochrome, 411, and empty inputs; an operator
  runbook at [`docs/development/fuzzing.md`](docs/development/fuzzing.md);
  and a nightly GitHub Actions job
  ([`.github/workflows/fuzz.yml`](.github/workflows/fuzz.yml))
  that runs each harness for 5 minutes per night and uploads
  any `crash-*` / `oom-*` / `timeout-*` artefacts. Intended to
  flip the OSSF Scorecard `Fuzzing` check off zero (PR #337
  remediation tracker).

  **The 60-second smoke run already surfaced a real heap-buffer-
  overflow** in `y4m_convert_411_422jpeg`
  ([`libvmaf/tools/y4m_input.c:507`](libvmaf/tools/y4m_input.c)):
  the first sub-loop of the 411-to-422 upsample writes
  `_dst[1]` unconditionally when `c_w == 1` and the destination
  chroma width `dst_c_w == 1`, missing the
  `(x << 1 | 1) < dst_c_w` guard that the third sub-loop
  carries. Reproducer parked under
  [`libvmaf/test/fuzz/y4m_input_known_crashes/y4m_411_w2_h4_oob_dst.y4m`](libvmaf/test/fuzz/y4m_input_known_crashes/);
  excluded from the nightly job until the fix lands. Tracked in
  [`docs/state.md`](docs/state.md) as a new Open bug. The fix
  ships as a follow-up PR (separate ADR) per the project's
  "harness + bug-report PR first, fix PR second" convention.
- **HIP seventh + eighth kernel-template consumers — `float_motion_hip`
  and `float_ssim_hip` (T7-10b follow-up / ADR-0273 + ADR-0274).**
  Ships
  [`libvmaf/src/feature/hip/float_motion_hip.{c,h}`](libvmaf/src/feature/hip/float_motion_hip.c)
  and
  [`libvmaf/src/feature/hip/float_ssim_hip.{c,h}`](libvmaf/src/feature/hip/float_ssim_hip.c)
  — the seventh and eighth consumers of
  `libvmaf/src/hip/kernel_template.h` after the six already shipped
  (ADR-0241 / -0254 / -0259 / -0260) or in flight (ADR-0266 /
  -0267, PR #340). `float_motion_hip` mirrors
  `libvmaf/src/feature/cuda/float_motion_cuda.c` (361 LOC,
  smallest unported CUDA twin) call-graph-for-call-graph; pins the
  **three-buffer ping-pong** (raw-pixel cache `uintptr_t ref_in` +
  blurred-frame ping-pong `uintptr_t blur[2]`) and the
  `motion_force_zero` short-circuit posture (`fex->extract` swap
  with the kernel-template helpers nulled). `float_ssim_hip` mirrors
  `libvmaf/src/feature/cuda/integer_ssim_cuda.c` (384 LOC, registers
  `vmaf_fex_float_ssim_cuda` despite the integer_ filename); first
  **multi-dispatch** HIP consumer (`chars.n_dispatches_per_frame ==
  2`); pins the five-intermediate-float-buffer pyramid
  (`uintptr_t h_{ref_mu,cmp_mu,ref_sq,cmp_sq,refcmp}`) and the v1
  `scale=1` `-EINVAL` validation surface. Both consumers register
  under `#if HAVE_HIP`; `init()` returns `-ENOSYS` until T7-10b
  flips kernel-template helper bodies to live HIP calls
  (`float_ssim_hip` may instead return `-EINVAL` from
  `validate_dims_hip` if the input dimensions trigger
  auto-decimation, mirroring the CUDA twin's validation). Smoke
  test grows from 16 to 18 sub-tests
  (`test_float_motion_hip_extractor_registered` pins the TEMPORAL
  flag bit; `test_float_ssim_hip_extractor_registered` pins
  `n_dispatches_per_frame == 2`). CPU baseline 47/47 green; HIP
  scaffold build 48/48 green. No ROCm SDK required.
- **Dynamic-PTQ int8 sidecars for `vmaf_tiny_v3` + `vmaf_tiny_v4`
  (T5-3d follow-up / ADR-0275).** Adds
  [`model/tiny/vmaf_tiny_v3.int8.onnx`](model/tiny/vmaf_tiny_v3.int8.onnx)
  (4 267 B; sha256 `b4bdbb35…`) and
  [`model/tiny/vmaf_tiny_v4.int8.onnx`](model/tiny/vmaf_tiny_v4.int8.onnx)
  (7 769 B; sha256 `203a2590…`), produced via
  `ai/scripts/ptq_dynamic.py`. Both registered with `quant_mode:
  "dynamic"` + `int8_sha256` + `quant_accuracy_budget_plcc: 0.01` in
  [`model/tiny/registry.json`](model/tiny/registry.json) and the
  matching per-model sidecars. v3 PLCC drop = 0.000120 (×83 budget
  headroom); v4 PLCC drop = 0.000145 (×69) on Netflix canonical-6
  features (~11k rows); KoNViD cross-corpus drops 0.000177 / 0.000080
  (~270k rows) — both still well under budget. v4 shrinks 45 % on
  disk; v3 only 5 % because `mlp_medium`'s fp32 ONNX is dominated by
  op metadata + Constant scaler nodes rather than weights. The
  `ai-quant-accuracy` CI gate (ADR-0174) iterates over both
  transparently. v2 stays fp32 — too little weight mass to be worth
  quantising. See
  [`docs/ai/models/vmaf_tiny_v3.md`](docs/ai/models/vmaf_tiny_v3.md)
  +
  [`docs/ai/models/vmaf_tiny_v4.md`](docs/ai/models/vmaf_tiny_v4.md)
  for the per-model PTQ sections, and
  [ADR-0275](docs/adr/0275-vmaf-tiny-v3-v4-ptq.md) for the rationale.

- **`vmaf-tune recommend` subcommand — `--target-vmaf` and
  `--target-bitrate` flags (Phase B-lite, Research-0061 Buckets 4 +
  5).** New `recommend` subcommand on `tools/vmaf-tune/` that consumes
  the Phase A corpus (either pre-built JSONL via `--from-corpus` or
  generated on the fly from `--source` + grid flags) and applies a
  user-supplied predicate. `--target-vmaf T` returns the row with the
  smallest CRF whose `vmaf_score >= T`; `--target-bitrate KBPS`
  returns the row whose `bitrate_kbps` is closest to `KBPS`. The two
  flags are mutually exclusive (argparse rejects both with exit 2).
  When no row clears `--target-vmaf`, the closest miss is returned
  with the predicate annotated `(UNMET)`. Default output is a single
  human-readable line; `--json` switches to the full corpus row as a
  JSON object on stdout. No new schema — `recommend` reuses the
  existing `bitrate_kbps` + `vmaf_score` columns; new code paths are
  contained to `tools/vmaf-tune/src/vmaftune/recommend.py` and the
  CLI wrapper. 13-test suite under
  `tools/vmaf-tune/tests/test_recommend.py` covers predicate
  semantics, encoder/preset filtering, NaN/failed-encode rejection,
  and CLI exit codes. Implements Buckets 4 + 5 from
  [docs/research/0061-vmaf-tune-capability-audit.md](docs/research/0061-vmaf-tune-capability-audit.md);
  parent ADR is
  [ADR-0237](docs/adr/0237-quality-aware-encode-automation.md).
- **`vmaf-tune` codec adapter — libaom-av1 (ADR-0278).** Adds
  [`tools/vmaf-tune/src/vmaftune/codec_adapters/libaom.py`](tools/vmaf-tune/src/vmaftune/codec_adapters/libaom.py)
  exposing the canonical adapter contract (name, encoder, quality
  knob, range, default, invert flag, presets tuple, validation) plus
  two libaom-specific helpers: `cpu_used(preset) -> int` mapping the
  shared cross-codec preset vocabulary (`placebo=0, slowest=1, ...,
  ultrafast=9`) onto libaom's `-cpu-used` integer, and
  `ffmpeg_codec_args(preset, crf) -> tuple[str, ...]` returning the
  argv slice that goes after `-c:v libaom-av1`. CRF range is the full
  libaom window `[0, 63]` with default `35`. Companion to the parallel
  `libx265` and `libsvtav1` adapter PRs; populates the fourth slot of
  the `fr_regressor_v2` six-slot codec one-hot
  (`x264, x265, svtav1, libaom, ?, ?`). User-facing documentation in
  [`docs/usage/vmaf-tune.md`](docs/usage/vmaf-tune.md) covers the
  preset → cpu-used mapping and the libaom-vs-SVT-AV1 trade-off
  (libaom is meaningfully slower at matched presets but tends to
  deliver slightly higher quality at the same bitrate at slow
  presets per AOM benchmarks). Routing the libaom argv slice through
  `vmaftune.encode.build_ffmpeg_command` is a follow-up alongside the
  codec-pluggable encode wiring.
- **`vmaf-tune` resolution-aware model selection + per-resolution CRF
  offsets (ADR-0289, PR #354 audit Bucket #8).** New module
  [`tools/vmaf-tune/src/vmaftune/resolution.py`](tools/vmaf-tune/src/vmaftune/resolution.py)
  exposes `select_vmaf_model_version(width, height) -> str` (height
  ≥ 2160 picks `vmaf_4k_v0.6.1`, else `vmaf_v0.6.1`),
  `select_vmaf_model(width, height) -> Path`, and
  `crf_offset_for_resolution(width, height) -> int` (4K → -2,
  1080p → 0, 720p → +2, sub-720p → +4 — codec-agnostic conservative
  defaults; Phase B/C/D will learn per-codec offsets via the same
  signature). The `corpus` subcommand auto-picks the model per
  encode resolution by default; pass `--no-resolution-aware` to
  force a single model via `--vmaf-model`. The emitted JSONL row's
  `vmaf_model` field now records the *effective* model used per
  row, not the global option — required for mixed-ladder corpora to
  be unambiguous downstream. Decision rule mirrors Netflix's
  published guidance; see
  [ADR-0289](docs/adr/0289-vmaf-tune-resolution-aware.md) and
  [Research-0054](docs/research/0064-vmaf-tune-resolution-aware.md).
  17 new tests in
  [`tools/vmaf-tune/tests/test_resolution.py`](tools/vmaf-tune/tests/test_resolution.py).
- **`vmaf-tune` NVENC codec adapters — `h264_nvenc`, `hevc_nvenc`,
  `av1_nvenc` (ADR-0290).** Wires the NVIDIA NVENC family into
  `tools/vmaf-tune/src/vmaftune/codec_adapters/` as three new
  adapter files plus a shared `_nvenc_common.py` helper that owns
  the mnemonic preset map (`ultrafast`..`placebo` → `p1`..`p7`)
  and the constant-quantizer hard window `[0, 51]`. Hardware
  encoders are 10–100× faster than software at the cost of ~3–5
  VMAF points at matched bitrate; the R-D curve is codec-distinct
  not codec-shifted, so they ship as separate codec entries rather
  than a flag on `libx264`. Registry now exposes
  `known_codecs() == ("av1_nvenc", "h264_nvenc", "hevc_nvenc",
  "libx264")`. Test coverage: parametrised suite under
  `tools/vmaf-tune/tests/test_codec_adapter_nvenc.py` (preset
  mapping, CQ range validation, encoder-not-found subprocess
  simulation). Documentation:
  [`docs/usage/vmaf-tune.md`](docs/usage/vmaf-tune.md) "Hardware
  encoders (NVENC)" section. Follow-up filed: `fr_regressor_v2`
  codec one-hot expansion from 6 → ≥ 12 slots before training v2
  on a corpus that includes hardware codecs. Companion research
  digest:
  [`docs/research/0065-vmaf-tune-nvenc-adapters.md`](docs/research/0065-vmaf-tune-nvenc-adapters.md).
- **`vmaf-tune` AMD AMF codec adapters — `h264_amf` / `hevc_amf` /
  `av1_amf` (ADR-0282).** Three new adapters under
  [`tools/vmaf-tune/src/vmaftune/codec_adapters/`](tools/vmaf-tune/src/vmaftune/codec_adapters/)
  sharing
  [`_amf_common.py`](tools/vmaf-tune/src/vmaftune/codec_adapters/_amf_common.py)
  — companion to the parallel NVENC / QSV adapter PRs. Wires the AMD
  Advanced Media Framework path through ffmpeg with `-rc cqp` plus
  matched `-qp_i` / `-qp_p` (closest analogue to x264 CRF). Compresses
  the canonical 7-level preset vocabulary onto AMF's three quality
  rungs (`placebo`/`slowest`/`slower`/`slow` → `quality`; `medium` →
  `balanced`; `fast`/`faster`/`veryfast`/`superfast`/`ultrafast` →
  `speed`) — the AMF hardware pipeline does not expose finer steps.
  AV1 (`av1_amf`) requires RDNA3 silicon (Radeon RX 7000 series or
  newer). `ensure_amf_available` probes `ffmpeg -encoders` for the
  requested codec and rejects builds without `--enable-amf` or an AMD
  GPU. Tests mock the subprocess seam end-to-end (50/50 passing
  without ffmpeg or an AMD GPU on PATH); docs at
  [`docs/usage/vmaf-tune.md`](docs/usage/vmaf-tune.md) §Hardware
  encoders carry the preset compression table and the silicon
  requirements.

- **`vmaf-tune` Intel QSV codec adapters — `h264_qsv`, `hevc_qsv`,
  `av1_qsv` (ADR-0281).** Three thin frozen-dataclass adapters under
  [`tools/vmaf-tune/src/vmaftune/codec_adapters/`](tools/vmaf-tune/src/vmaftune/codec_adapters/),
  backed by a shared private `_qsv_common.py` that pins the QSV
  preset vocabulary (`veryslow…veryfast`, x264-style names verbatim),
  the ICQ `global_quality` window (`1..51`), the preset identity
  check, the range validator, and an `ffmpeg -encoders` probe that
  fails fast when libmfx / VPL is not compiled in. Registry widened
  from libx264-only to four codecs; the search loop continues to
  route through the registry without branching on codec identity.
  Hardware availability matrix documented in
  [`docs/usage/vmaf-tune.md`](docs/usage/vmaf-tune.md): H.264 / HEVC
  on Intel iGPU 7th-gen+ (Kaby Lake or newer) or Arc / Battlemage;
  AV1 on 12th-gen+ iGPU only or Arc / Battlemage. Encode pipeline
  (`encode.py`) widening to dispatch on `adapter.quality_knob`
  deferred to a follow-up PR — adapter classes are valid but inert
  until that lands. Sibling NVENC + AMF adapter PRs run in parallel.
  Companion research digest:
  [`docs/research/0054-vmaf-tune-qsv-adapters.md`](docs/research/0054-vmaf-tune-qsv-adapters.md).
- **`vmaf-tune` libvvenc + NN-VC codec adapter (ADR-0285).** Adds
  [`tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py`](tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py)
  as a Phase A codec adapter on the ADR-0237 contract: drives
  Fraunhofer HHI's VVC / H.266 encoder via FFmpeg's `-c:v libvvenc`
  wrapper. `quality_knob = "qp"`, `quality_range = (17, 50)`, default
  `32`. The harness's canonical 7-name preset vocabulary
  (`placebo / slowest / slower / slow / medium / fast / faster /
  veryfast / superfast / ultrafast`) compresses onto VVenC's native
  5-level scale (`faster / fast / medium / slow / slower`) via a
  static map. First-class **NN-VC (neural-network video coding)**
  surface: `nnvc_intra: bool = False` toggle that emits
  `-vvenc-params IntraNN=1` to enable VVC's learned 5×5 / 7×7 / 9×9
  conv intra-prediction (~1-3% bitrate gain at iso-VMAF, ~5-10×
  slower intra encode); NN loop filter and NN super-resolution
  toggles deferred to follow-up ADRs once the corpus carries enough
  VVenC rows to estimate quality / cost separately. Companion docs
  in [`docs/usage/vmaf-tune.md`](docs/usage/vmaf-tune.md). Sibling
  to the parallel x264 / x265 / svt-av1 / libaom + NVENC / QSV /
  AMF / VideoToolbox adapter PRs landing 2026-05-03. NN-VC is the
  closest thing the open-source video stack has to a "neural-
  augmented codec" today and is the natural counterpart to the
  fork's existing tiny-AI *measurement* surface (`vmaf_tiny_v2`,
  `fr_regressor_v1`, `nr_metric_v1`).
- **`vmaf-tune` Phase D scaffold — per-shot CRF tuning (ADR-0276).**
  Adds `tools/vmaf-tune/src/vmaftune/per_shot.py` plus the
  `vmaf-tune tune-per-shot` CLI subcommand. `detect_shots()` wraps
  the C-side `vmaf-perShot` binary (ADR-0222 / TransNet V2
  ADR-0223) with a single-shot fallback when the binary is
  unavailable. `tune_per_shot()` exposes a pluggable predicate
  seam that Phase B's bisect (PR #347) drops into. `merge_shots()`
  emits per-segment FFmpeg argv plus a concat-demuxer command.
  Closes the orchestration layer for Bucket #1 of Research-0061
  (the Netflix-style table-stakes per-shot encoding feature).
  Scaffold-only: the default predicate returns the codec
  adapter's default CRF, and per-codec native emission
  (`--qpfile`/`--zones`/SVT-AV1 segment tables) is deferred to
  per-codec PRs. 16 new tests pass with mocked `vmaf-perShot` +
  mocked encoder; total `vmaf-tune` suite is 29 tests, zero
  binaries required. First per-phase split off
  [ADR-0237](docs/adr/0237-quality-aware-encode-automation.md).

- **`vmaf-tune` SVT-AV1 codec adapter (ADR-0278).** Adds
  [`tools/vmaf-tune/src/vmaftune/codec_adapters/svtav1.py`](tools/vmaf-tune/src/vmaftune/codec_adapters/svtav1.py)
  wired into the registry as `"libsvtav1"`, the second codec the
  Phase A corpus harness can sweep after `libx264`. AV1 specifics
  encoded as data: CRF range `0..63` (vs x264's `0..51`) with Phase A
  informative window pinned to `(20, 50)`, integer presets `0..13`
  translated from x264-style names via a closed `PRESET_NAME_TO_INT`
  table (`medium`→`7`, the SVT-AV1 default). The corpus loop emits
  the integer into FFmpeg's `-preset` argv slot via an optional
  `ffmpeg_preset_token()` adapter hook while keeping the human name
  on the JSONL row, so downstream `fr_regressor_v2` consumers
  (ADR-0235) read libsvtav1 rows unchanged via
  `CODEC_VOCAB[2] = "libsvtav1"`. `parse_versions()` learns the
  SVT-AV1 banner pattern alongside the x264 one. Pure subprocess
  mocks in `tools/vmaf-tune/tests/test_codec_adapter_svtav1.py`
  (27 tests pass total); real-binary integration smoke deferred to
  the CI runner that ships libsvtav1.
- **vmaf-tune Phase E — per-title bitrate-ladder generator
  (ADR-0277).** Ships
  [`tools/vmaf-tune/src/vmaftune/ladder.py`](tools/vmaf-tune/src/vmaftune/ladder.py)
  and a `vmaf-tune ladder` CLI subcommand that mirrors the Netflix
  per-title encoding paper: sample the (resolution × target-VMAF)
  plane via a pluggable `SamplerFn`, take the Pareto upper-convex
  hull on (bitrate, vmaf), pick `n` rungs along the hull (log-bitrate
  or VMAF spacing), and emit an ABR manifest (HLS master playlist /
  DASH MPD / JSON descriptor). PR #354 capability audit Bucket #6
  flagged this as the single biggest "game-changer" for the fork.
  Currently scaffold-only: the production sampler that drives Phase
  B's target-VMAF bisect (PR #347) wires up in a follow-up PR; the
  default sampler raises `NotImplementedError` and tests inject a
  synthetic stub. 15 new ladder tests + 28 total in
  `tools/vmaf-tune/tests/`. Default rendition set is the canonical
  5-rung 1080p/720p/480p/360p/240p ladder. See
  [`docs/usage/vmaf-tune.md` § "Per-title ladder"](docs/usage/vmaf-tune.md).

- **`fr_regressor_v2` probabilistic head — deep-ensemble + conformal
  scaffold (ADR-0279).** Adds the trainer
  [`ai/scripts/train_fr_regressor_v2_ensemble.py`](ai/scripts/train_fr_regressor_v2_ensemble.py),
  the evaluator
  [`ai/scripts/eval_probabilistic_proxy.py`](ai/scripts/eval_probabilistic_proxy.py),
  and the manifest sidecar `model/tiny/fr_regressor_v2_ensemble_v1.json`.
  Trains N=5 copies of the v2 architecture under distinct seeds,
  exports each as a separate two-input ONNX
  (`fr_regressor_v2_ensemble_v1_seed{0..4}.onnx`), and aggregates
  outputs into `(mu, sigma)` at inference. Optional split-conformal
  calibration on a held-out fraction yields a distribution-free
  marginal coverage guarantee at 1 − α (Vovk 2005, Romano 2019).
  Surfaces the prediction interval the in-flight `vmaf-tune
  --quality-confidence 0.95` flag (ADR-0237) needs to answer
  "smallest CRF where the lower 95 % VMAF bound is ≥ 92". Smoke-only
  ship — synthetic 100-row corpus, 1 epoch / member; production
  training is gated on the multi-codec Phase A corpus and tracked as
  backlog item T7-FR-REGRESSOR-V2-PROBABILISTIC. Five new ensemble
  members register as `kind: "fr"` rows in
  `model/tiny/registry.json`; the manifest is the higher-level
  ensemble entry point. Closes PR #354 audit Bucket #18 (top-3
  ranked). Companion docs:
  [`docs/ai/models/fr_regressor_v2_probabilistic.md`](docs/ai/models/fr_regressor_v2_probabilistic.md),
  [`docs/research/0067-fr-regressor-v2-probabilistic.md`](docs/research/0067-fr-regressor-v2-probabilistic.md).
- **`vmaf-tune` codec-agnostic encode dispatcher (ADR-0294, T-VMAF-TUNE
  follow-up).** Refactors
  [`tools/vmaf-tune/src/vmaftune/encode.py`](tools/vmaf-tune/src/vmaftune/encode.py)
  away from the Phase A hard-coded `libx264` `-c:v / -preset / -crf`
  argv. `run_encode` now looks up the codec adapter via
  `codec_adapters.get_adapter(req.encoder)` and asks it for the
  FFmpeg argv slice via `adapter.ffmpeg_codec_args(preset, quality)`
  (codec-specific shape — `-cq` / `-qp` / `-global_quality` / `-q:v`
  / etc.) plus an optional `adapter.extra_params()`. Adapters
  without `ffmpeg_codec_args` fall back to the legacy x264-CRF
  shape so partial in-flight adapters stay drivable.
  `parse_versions(stderr, encoder=...)` selects a per-codec version
  probe (libx264, libx265, libsvtav1, libvpx-vp9, libaom-av1,
  libvvenc, NVENC, QSV, AMF, VideoToolbox); unknown encoders return
  `"unknown"` rather than raising. The `EncodeRequest.crf` field is
  preserved unchanged for the SCHEMA_VERSION=1 row contract; a
  `quality` property mirrors it for adapter-side codec-agnostic
  vocabulary. Existing 13-test x264 suite still green; new 19-test
  multi-codec suite
  ([`tools/vmaf-tune/tests/test_encode_multi_codec.py`](tools/vmaf-tune/tests/test_encode_multi_codec.py))
  covers 9 codec shapes plus the unknown-codec / missing-method
  fallback paths. **Unblocks 17 in-flight codec adapter PRs**
  (#360 libaom, #362 libx265, #364 NVENC, #366 AMF, #367 QSV,
  #368 libvvenc, #370 libsvtav1, #373 VideoToolbox, plus follow-on
  waves) which now drive end-to-end encodes without copying or
  mutating the harness. New
  [`docs/adr/0297-vmaf-tune-encode-multi-codec.md`](docs/adr/0297-vmaf-tune-encode-multi-codec.md),
  [`docs/research/0070-vmaf-tune-encode-multi-codec.md`](docs/research/0070-vmaf-tune-encode-multi-codec.md);
  [`docs/usage/vmaf-tune.md`](docs/usage/vmaf-tune.md#codec-adapter-contract)
  gains a "Codec adapter contract" section.
  [`docs/rebase-notes.md`](docs/rebase-notes.md) entry 0228 pins the
  codec-agnostic-harness invariant.
- **`vmaf-tune` GPU scoring backend — `--score-backend {auto|cpu|cuda|sycl|vulkan}`
  (ADR-0299).** Wires the libvmaf CLI's existing `--backend NAME` selector
  ([ADR-0127](docs/adr/0127-vulkan-compute-backend.md) /
  [ADR-0175](docs/adr/0175-vulkan-backend-scaffold.md) /
  [ADR-0186](docs/adr/0186-vulkan-image-import-impl.md)) into Phase A
  scoring. Default `auto` walks `cuda → vulkan → sycl → cpu`, intersecting
  vmaf-binary advertised support (parsed from `vmaf --help`) with cheap
  host probes (`nvidia-smi -L`, `vulkaninfo --summary`, `sycl-ls`).
  Explicit non-auto requests are strict — `--score-backend cuda` on a
  CPU-only host raises `BackendUnavailableError` rather than silently
  downgrading. Removes the CPU-only score floor (1–2 fps at 1080p) and
  delivers ~10–30× speedup on the score axis. Ships
  `tools/vmaf-tune/src/vmaftune/score_backend.py`, CLI flag wiring, 22
  unit tests against the mocked help/probe runners, and a docs/usage
  section with a wall-clock expectation table.

- **vmaf-tune content-addressed encode/score cache (ADR-0298).**
  Adds
  [`tools/vmaf-tune/src/vmaftune/cache.py`](tools/vmaf-tune/src/vmaftune/cache.py),
  a content-addressed cache keyed on
  `sha256(src_content) + encoder + preset + crf + adapter_version + ffmpeg_version`
  that short-circuits both the encode and score subprocesses on a
  hit. Cache lives at `$XDG_CACHE_HOME/vmaf-tune/` (or
  `~/.cache/vmaf-tune/`) with LRU eviction at a 10 GiB default cap.
  Wired into the corpus loop with default-ON; `--no-cache`,
  `--cache-dir`, and `--cache-size-gb` opt-out and override. JSONL
  row schema unchanged (`SCHEMA_VERSION` stays at 1) — the cache is
  an opaque sidecar, not baked into the row. Re-runs of the same
  `(preset, crf)` sweep collapse from minutes to milliseconds; the
  intended workflow win is "user adjusts a flag, re-runs the same
  sweep, only the changed cells re-encode." Twelve new contract
  tests under `tests/test_cache.py` cover key stability, eviction,
  and end-to-end miss-then-hit through the corpus loop.
- **`vmaf-tune` coarse-to-fine CRF search (ADR-0306).** New
  `coarse_to_fine_search()` orchestrator in
  [`tools/vmaf-tune/src/vmaftune/corpus.py`](tools/vmaf-tune/src/vmaftune/corpus.py)
  drives a 2-pass search instead of the full `0..51` CRF grid. With
  defaults (`coarse_step=10`, `fine_radius=5`, `fine_step=1`) the
  search visits 5 coarse + up to 10 fine = 15 encodes per (source,
  preset) instead of 52 — a 3.46× wall-time speedup. When the
  highest-CRF coarse point already meets `--target-vmaf` the fine
  pass is skipped (1-pass shortcut, ~10× speedup). Exposed via
  `vmaf-tune corpus --coarse-to-fine` (opt-in) and a new
  `vmaf-tune recommend --target-vmaf VMAF` subcommand which prints
  the cheapest passing CRF for a quality target. The libx264 codec
  adapter's `quality_range` widens from the old `(15, 40)`
  informative window to the codec's nominal `(0, 51)` so the
  search domain matches the user's CLI; existing full-grid users
  who only pass `--crf 23..28` are unaffected. JSONL row schema
  unchanged (`SCHEMA_VERSION=1`). New tests:
  `test_coarse_to_fine_canonical_visits_15_points` and
  `test_coarse_to_fine_one_pass_shortcut_when_coarse_max_meets_target`
  in
  [`tools/vmaf-tune/tests/test_corpus.py`](tools/vmaf-tune/tests/test_corpus.py).
  Docs:
  [`docs/usage/vmaf-tune.md`](docs/usage/vmaf-tune.md#coarse-to-fine-crf-search-adr-0296).

- **HIP third + fourth kernel-template consumers — `ciede_hip` and
  `float_moment_hip` (T7-10b follow-up / ADR-0259 + ADR-0260).**
  Ships
  [`libvmaf/src/feature/hip/ciede_hip.{c,h}`](libvmaf/src/feature/hip/ciede_hip.c)
  and
  [`libvmaf/src/feature/hip/float_moment_hip.{c,h}`](libvmaf/src/feature/hip/float_moment_hip.c)
  — the third and fourth consumers of
  `libvmaf/src/hip/kernel_template.h` after `integer_psnr_hip`
  (ADR-0241) and `float_psnr_hip` (ADR-0254 / PR #324).
  `ciede_hip` mirrors `libvmaf/src/feature/cuda/integer_ciede_cuda.c`
  call-graph-for-call-graph including the **intentional bypass** of
  `submit_pre_launch` (ciede's kernel writes one float per block, no
  atomic, no memset required) — pinning the kernel-template's
  "no-memset bypass" path pre-runtime. `float_moment_hip` mirrors
  `libvmaf/src/feature/cuda/integer_moment_cuda.c` with a four-uint64
  atomic-counter readback (`MOMENT_HIP_COUNTERS = 4u`), pinning the
  "memset multiple uint64 counters in one helper call" path pre-runtime.
  Registers `vmaf_fex_ciede_hip` and `vmaf_fex_float_moment_hip` in
  `feature_extractor_list` under `#if HAVE_HIP`. Same scaffold posture
  as the first two consumers: registration succeeds, `init()` returns
  `-ENOSYS` until the runtime PR (T7-10b) flips the kernel-template
  helper bodies to live HIP calls. Smoke test grows from 15 to 17
  sub-tests (new `test_ciede_hip_extractor_registered` +
  `test_float_moment_hip_extractor_registered`). CPU baseline + HIP
  scaffold both green. No ROCm SDK required. Picks
  `integer_ciede_cuda` (243 LOC) and `integer_moment_cuda` (230 LOC) —
  the two smallest CUDA twins remaining after the two PSNR variants
  were claimed — over `integer_motion_cuda` (503 LOC, stateful),
  `integer_motion_v2_cuda` (321 LOC, stateful + dual-feature), and
  `float_ansnr_cuda` (298 LOC, duplicates ADR-0254's precision
  posture).
- **Vulkan 1.4 API-version bump deferred — root-cause investigation
  (ADR-0264 / research-0053).** Docs-only PR. An exploratory
  `VK_API_VERSION_1_3` → `VK_API_VERSION_1_4` bump on
  `VkApplicationInfo.apiVersion` and `VmaAllocatorCreateInfo.vulkanApiVersion`
  moves NVIDIA driver ≥ 1.4.329 output above the `places=4` cross-backend
  gate (ADR-0214) on `integer_vif_scale2` (45/48 mismatches, max abs
  1.527e-02) and `ciede2000` (42/48 mismatches, max abs 1.67e-04); RADV
  (Mesa 26.0.6) and lavapipe stay clean. The investigation proves the
  compiled SPIR-V is byte-identical at `--target-env=vulkan1.3` vs
  `vulkan1.4` (verified by `cmp` after glslc 2026.1), so the regression
  is purely NVIDIA's runtime shader compiler flipping its default FMA
  contraction policy under core `shaderFloatControls2` (promoted to
  Vulkan 1.4). Vulkan SPIR-V exposes no module-wide `ContractionOff`
  execution mode (Kernel-only); the only knob is per-result
  `OpDecorate ... NoContraction` (GLSL `precise`). Backlog item
  **T-VK-1.4-BUMP** captures the two-step fix path: (A) tag load-bearing
  FP ops `precise` in `libvmaf/src/feature/vulkan/shaders/vif.comp` and
  `ciede.comp`, (B) bump only after the gate is clean on all three
  drivers. `master` stays on Vulkan 1.3; no code change in this PR.
- **HIP fifth + sixth kernel-template consumers (T7-10b / ADR-0266 +
  ADR-0267).** Adds `feature/hip/float_ansnr_hip.{c,h}` and
  `feature/hip/integer_motion_v2_hip.{c,h}` as the fifth and sixth
  consumers of `libvmaf/src/hip/kernel_template.h`, mirroring
  `feature/cuda/float_ansnr_cuda.c` (297 LOC, smallest unported
  CUDA twin) and `feature/cuda/integer_motion_v2_cuda.c` (320 LOC,
  smallest unported temporal CUDA twin) call-graph-for-call-graph.
  `float_ansnr_hip` pins the **interleaved (sig, noise) per-block
  float-partial readback** shape — same `submit_pre_launch` bypass
  as `ciede_hip` (ADR-0259), doubled per-block partial width, two
  features (`float_ansnr` + `float_anpsnr`) emitted from one kernel
  pass. `motion_v2_hip` pins the **temporal-extractor shape** —
  `VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag, `flush()` callback,
  `uintptr_t[2]` ping-pong buffer slots that the runtime PR (T7-10b)
  will swap for real device-buffer handles. Both consumers register
  under `#if HAVE_HIP`; `init()` returns `-ENOSYS` until T7-10b
  flips kernel-template helper bodies to live HIP calls. Smoke test
  grows from 14 to 16 sub-tests
  (`test_float_ansnr_hip_extractor_registered` +
  `test_motion_v2_hip_extractor_registered`). CPU baseline 47/47
  green; HIP scaffold build 48/48 green. No ROCm SDK required.

- **`iqa_convolve` block-of-N tap widen — failed-attempt research
  digest (Research-0053).** Records the bit-exactness post-mortem for
  the proposed AVX-512 block-of-4 reorder targeted at the largest
  CPU hot-spot (`iqa_convolve_avx512`, ~39.5 % self-time on the
  post-merge CPU profile). The hypothesis was to amortise the
  per-tap `vcvtps2pd` widen over a block of 4 taps (4+4+3 stagger
  on the 11-tap Gaussian) to cut widen count 3.7× and recover an
  expected +6–10 % end-to-end on `float_ssim` / `float_ms_ssim`.
  A 10 M random-input Monte Carlo on the actual Gaussian
  coefficients shows block-of-4 mismatches the scalar reference's
  float-cast output on **27.67 %** of pixels; block-of-2 mismatches
  on 16.68 %. Per-tap widen is load-bearing — collapsing N
  intermediate doubles into one before adding to the accumulator
  changes rounding by construction. The
  [ADR-0138](docs/adr/0138-iqa-convolve-avx2-bitexact-double.md)
  bit-exact contract holds; the SIMD path stays as-is. Future
  options sketched: widen earlier in the pipeline (carry `__m512d`
  across the h/v split), or negotiate a per-path ULP budget under
  [ADR-0140](docs/adr/0140-simd-dx-framework.md). No code change in
  this PR. Companion: [`docs/research/0054-iqa-convolve-block-widen-attempt.md`](docs/research/0054-iqa-convolve-block-widen-attempt.md).
- **ssimulacra2 Vulkan host-path AVX2 + NEON SIMD (T-GPU-OPT-VK-2 / ADR-0252).**
  Adds `feature/x86/ssimulacra2_host_avx2.c` and `feature/arm64/ssimulacra2_host_neon.c`
  with `plane_stride`-parameterised `linear_rgb_to_xyb` and `downsample_2x2` SIMD
  kernels. Wires runtime dispatch (AVX2 / NEON) into `ssimulacra2_vulkan.c` for the
  three host-side hot paths identified by the 2026-05-02 Vulkan profile: YUV→linear-RGB,
  linear-RGB→XYB, and 2×2 box downsample. Measured: 2× XYB speedup, 3.2× downsample
  speedup on the 576×324 benchmark. Bit-exact to scalar reference (`memcmp`-level)
  verified by two new test cases in `test_ssimulacra2_simd`. AVX-512 omitted — cbrtf
  is per-lane scalar in both AVX2 and AVX-512 paths so the marginal gain is below 30%.

- **Bristol VI-Lab feasibility digest + BVI-CC ingest ADR
  (Draft).** Reconnaissance only — no downloads, no code change.
  [`docs/research/0046-bristol-vi-lab-feasibility.md`](docs/research/0046-bristol-vi-lab-feasibility.md)
  surveys nine BVI-* datasets (BVI-CC, BVI-DVC, BVI-AOM, BVI-HD,
  BVI-HFR, BVI-SR, BVI-VFI, BVI-SynTex, BVI-RLV) along axes of
  size, format, MOS / DMOS availability, licence posture,
  use-case fit (tiny-AI fr_regressor_v2 codec-aware leg vs.
  cross-backend parity soak), and effort-to-extract.
  [ADR-0241](docs/adr/0241-bristol-bvi-cc-ingest.md) (Status:
  Draft) proposes BVI-CC as the second tiny-AI training corpus
  alongside Netflix Public — 9 references × 306 distorted across
  HM / AV1 / VTM, ~250–400 GB raw. The load-bearing risk is the
  DMOS-vs-MOS scale mismatch (Bristol uses DMOS where higher = worse,
  Netflix uses MOS where higher = better); the ADR's
  recommendation makes the explicit `mos_convention` column an
  in-scope schema change for the first ingest PR rather than a
  follow-up.

- **`vmaf_tiny_v3` (mlp_medium, ADR-0241).** Wider/deeper tiny VMAF
  fusion model shipped alongside (not replacing) `vmaf_tiny_v2`.
  Architecture `mlp_medium` (6 → 32 → 16 → 1, 769 params vs v2's 257);
  same input contract (`features [N, 6]` float32 canonical-6 order,
  output `vmaf [N]` float32, opset 17, StandardScaler baked into
  graph), same training recipe as v2 (90 ep Adam @ lr=1e-3 MSE bs=256
  on the 4-corpus parquet, 330 499 rows). Netflix LOSO mean PLCC
  0.9986 ± 0.0015 vs v2's 0.9978 ± 0.0021 (+0.0008 mean, -29 % std).
  Production default stays v2; v3 is documented as the higher-PLCC /
  lower-variance option. New scripts under `ai/scripts/`
  (`train_vmaf_tiny_v3.py`, `export_vmaf_tiny_v3.py`,
  `validate_vmaf_tiny_v3.py`, `eval_loso_vmaf_tiny_v3.py`); new
  model-card [`docs/ai/models/vmaf_tiny_v3.md`](docs/ai/models/vmaf_tiny_v3.md);
  registry entry kind `fr` `smoke: false`. ONNX size 4 496 B
  (+2 050 B over v2). Companion research digest:
  [`docs/research/0046-vmaf-tiny-v3-mlp-medium-evaluation.md`](docs/research/0046-vmaf-tiny-v3-mlp-medium-evaluation.md).
- **HIP backend ships first kernel-template consumer (`integer_psnr_hip`)
  via T7-10 / [ADR-0241](docs/adr/0241-hip-first-consumer-psnr.md).**
  Lands [`libvmaf/src/hip/kernel_template.{h,c}`](libvmaf/src/hip/kernel_template.h)
  (field-for-field mirror of `cuda/kernel_template.h` from
  [ADR-0221](docs/adr/0246-gpu-kernel-template.md):
  `VmafHipKernelLifecycle` private-stream + 2-event struct,
  `VmafHipKernelReadback` device-accumulator + pinned-host pair,
  six lifecycle helpers) plus
  [`libvmaf/src/feature/hip/integer_psnr_hip.{c,h}`](libvmaf/src/feature/hip/integer_psnr_hip.c)
  (first consumer; mirrors `integer_psnr_cuda.c`'s
  init/submit/collect/close call graph verbatim). Template helper
  bodies and the consumer's submit/collect return `-ENOSYS` while
  the runtime PR (T7-10b) is pending; bodies flip to live HIP
  without touching consumer call-sites. New `vmaf_fex_psnr_hip`
  registered in `feature_extractor_list` under `#if HAVE_HIP` so
  callers asking for `psnr_hip` by name get "extractor found,
  runtime not ready" instead of "no such extractor". New
  `VMAF_FEATURE_EXTRACTOR_HIP = 1 << 6` flag bit reserved (unused
  until the picture buffer-type plumbing lands in T7-10b). Smoke
  test `libvmaf/test/test_hip_smoke.c` grows 5 sub-tests pinning
  the scaffold contract; 14/14 pass under `-Denable_hip=true`. CPU
  baseline (47/47) + HIP scaffold (48/48) green. No ROCm SDK
  required.
- **Vulkan VmafPicture preallocation surface (T-VULKAN-PREALLOC /
  ADR-0238).** Closes the API parity gap with CUDA / SYCL. New
  public entry points `vmaf_vulkan_preallocate_pictures` +
  `vmaf_vulkan_picture_fetch`; new enum
  `VmafVulkanPicturePreallocationMethod` (`NONE` / `HOST` /
  `DEVICE`); new struct `VmafVulkanPictureConfiguration`. Mirrors
  the SYCL surface — `HOST` uses `vmaf_picture_alloc`; `DEVICE`
  backs each picture's luma plane with a host-visible Vulkan
  buffer (VMA `AUTO_PREFER_HOST`) so callers write directly into
  the memory the kernel descriptor sets read. Pool depth is fixed
  at the canonical `frames-in-flight = 2` (matches SYCL); fetch
  falls back to a host-backed picture if the caller skipped
  preallocation. New `VMAF_PICTURE_BUFFER_TYPE_VULKAN_DEVICE`
  picture buffer-type tag. Six smoke tests in
  `libvmaf/test/test_vulkan_pic_preallocation.c` pin the contract
  under ASan/UBSan. FFmpeg-side adoption (per CLAUDE.md §12 r14)
  is a deliberate follow-up — no in-tree patch consumes the
  preallocation surface today, matching the SYCL precedent. See
  [`docs/api/gpu.md`](docs/api/gpu.md) and
  [ADR-0238](docs/adr/0238-vulkan-picture-preallocation.md).
- **`VmafVulkanConfiguration.max_outstanding_frames` knob (ADR-0235
  follow-up #3, T7-29 part 4).** The v2 async pending-fence ring
  depth is now caller-tunable via the public Vulkan configuration
  struct. `0` selects the canonical default (4); values are clamped
  to `[1, VMAF_VULKAN_RING_MAX]` (currently 8) internally. The
  clamped result is observable via the new
  `vmaf_vulkan_state_max_outstanding_frames()` accessor, so callers
  that pass a value can log what they actually got. External-handles
  callers (`vmaf_vulkan_state_init_external`) still receive the
  default — extending `VmafVulkanExternalHandles` is a separate ABI
  bump. Contract pinned in
  `libvmaf/test/test_vulkan_async_pending_fence.c`
  (`test_ring_size_*` group, 4 cases). See
  [`docs/api/gpu.md`](docs/api/gpu.md) and
  [ADR-0251](docs/adr/0251-vulkan-async-pending-fence.md).
- **`tools/vmaf-tune/` quality-aware encode automation spec
  (T-VMAF-TUNE / ADR-0237 — Proposed).** Umbrella spec for a new
  fork-local automation surface that closes the loop between the
  fork's quality stack and FFmpeg's encoders. The tool drives FFmpeg
  with parameter grids, captures bitrate + per-metric quality, and
  recommends encoder parameters per source × codec × target. Six-phase
  roadmap (A: harness MVP — `libx264` only, ~1 week; B: target-VMAF
  bisect; C: per-title CRF predictor; D: per-shot dynamic CRF gated
  on T6-3b; E: Pareto ABR ladder; F: MCP tools), each phase
  standalone-shippable. Multi-codec from day one via a codec-adapter
  interface — `libx264` / `libx265` / `libsvtav1` / `libvpx-vp9` /
  `libvvenc` / LCEVC / EVC / AVS3 / neural-codec adapters in an
  opt-in extra. Closes the loop on the codec-aware FR regressor
  (ADR-0235 codec collision, currently BLOCKED on corpus) and the
  per-shot CRF predictor (T6-3b). **Spec only — no code in this PR.**
  Companion research digest:
  [`docs/research/0044-quality-aware-encode-automation.md`](docs/research/0044-quality-aware-encode-automation.md).
- **Codec-aware FR regressor surface (T7-CODEC-AWARE / ADR-0235).**
  New `ai/src/vmaf_train/codec.py` ships a closed, order-stable
  6-bucket codec vocabulary (`x264`, `x265`, `libsvtav1`,
  `libvvenc`, `libvpx-vp9`, `unknown`) with `codec_index` + alias
  table (`h264` → `x264`, `hevc` → `x265`, `av1` → `libsvtav1`,
  `vp9` → `libvpx-vp9`, `vvc` / `h266` → `libvvenc`) and one-hot
  helpers. `FRRegressor` gains an optional `num_codecs` constructor
  arg that concatenates the one-hot codec id to the
  `FULL_FEATURES` input vector before the first MLP layer; the
  default `num_codecs=0` keeps the v1 single-input contract so
  existing checkpoints load unchanged. Feature-dump scripts
  (`ai/scripts/bvi_dvc_to_full_features.py`,
  `ai/scripts/extract_full_features.py`) now emit a `codec` column
  in their per-clip parquet — BVI-DVC defaults to `"x264"` (the
  internal libx264 encode), Netflix Public defaults to `"unknown"`
  (pre-encoded distortions, no in-band metadata). New
  `docs/adr/0235-codec-aware-fr-regressor.md`,
  `docs/research/0040-codec-aware-fr-conditioning.md`, and
  `docs/ai/models/fr_regressor_v2_codec_aware.md`. **Training run
  + PLCC delta measurement is BLOCKED** in this PR (the cached
  Netflix Public training corpus is not reachable from the
  authoring sandbox); the follow-up PR re-runs the trainer +
  ships `model/tiny/fr_regressor_v2_codec_aware.onnx` only if the
  empirical PLCC lift exceeds the 0.005 bar. Cites the 2026
  Bristol VI-Lab review §5.3 + Bampis 2018 (ST-VMAF) + Zhang 2021
  (Bull lab "Enhancing VMAF").

### Changed

- **`integer_ms_ssim_cuda` joins the engine-scope CUDA fence-batching
  helper (`drain_batch`).** Previously the extractor host-blocked 6
  times per frame: one `cuStreamSynchronize` at the end of `submit()`
  and five inside `collect()` (one per pyramid scale, forced by a
  shared partials buffer). The PR allocates **per-scale** partials
  buffers (5× `l_partials[]` / `c_partials[]` / `s_partials[]` device
  + matching pinned host shadows), enqueues all 5 SSIM scales' `horiz`
  + `vert_lcs` launches and DtoH copies back-to-back on
  `s->lc.str` inside `submit()`, records `s->lc.finished` once after
  the last DtoH, and calls `vmaf_cuda_drain_batch_register(&s->lc)` so
  the engine's `vmaf_cuda_drain_batch_flush` waits on the lifecycle
  alongside the rest of the CUDA feature stack. `collect()` becomes
  a host-side reduction only — `vmaf_cuda_kernel_collect_wait`
  short-circuits when the engine has already drained the lifecycle.
  Bit-exact (same kernels, same stream, same submission order; only
  the host wait point moves; cross-backend `places=4` gate unchanged).
  Expected ms_ssim wall-clock improvement on the Netflix CUDA
  benchmark: +3-5%. See [ADR-0271](docs/adr/0271-cuda-drain-batch-ms-ssim.md)
  and the per-frame syscall profile in
  [research-0061](docs/research/0061-cuda-ms-ssim-drain-batch-profile.md).

- **CI: `actions/upload-artifact` aligned to `@v7` across all
  workflows.** One leftover `@v4` pin in `.github/workflows/
  tests-and-quality-gates.yml` (the GPU parity-gate report upload
  step) is bumped to match the rest of the workflow tree, which is
  already pinned to `@v7` (or its commit-SHA equivalent
  `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a`). Pure CI maintenance —
  no library or runtime change.

- **`ffmpeg-patches/` series verified against pinned n8.1
  (2026-05-03).** Doc-only verification PR: replayed all six
  patches (0001..0006) onto a pristine FFmpeg `n8.1` checkout with
  `git am --3way` and confirmed every patch applies cleanly without
  context drift. Audited the libvmaf C-API symbols referenced by
  the patches (`vmaf_use_tiny_model`, `VmafDnnConfig`,
  `vmaf_vulkan_state_init_external`, `VmafVulkanExternalHandles`,
  `vmaf_vulkan_import_image`, `vmaf_vulkan_read_imported_pictures`,
  `vmaf_sycl_state_init`, `vmaf_sycl_dmabuf_import`, …) against
  `libvmaf/include/libvmaf/`; all symbols still exported, no
  renames or signature drift detected. Recent libvmaf API
  additions (PR #264 picture preallocation surface
  `vmaf_vulkan_preallocate_pictures` / `vmaf_vulkan_picture_fetch`,
  PR #260 `max_outstanding_frames` knob, PR #319 vulkan submit-side
  template, PR #320 cuda async upload) were reviewed for
  ffmpeg-side exposure: the preallocation surface is an alternative
  to the import path that the FFmpeg filter does not need (the
  filter is firmly on the zero-copy `vmaf_vulkan_import_image`
  path); the submit-side and CUDA changes are libvmaf-internal
  and do not affect the public surface the patches consume.
  No patch refresh required. See
  [ADR-0186](docs/adr/0186-vulkan-image-import-impl.md) for the
  series-replay invariant.
- **`psnr_hvs_cuda` async H2D upload + persistent pinned staging
  (T-GPU-OPT-2/3).** Reworks the per-frame upload path of the
  CUDA `psnr_hvs` feature extractor. Previously each plane upload
  allocated and freed a pinned host staging buffer per frame and
  host-blocked between the D2H and the H2D legs; now the staging
  buffers are sized once in `init_fex_cuda` and reused for the
  lifetime of the extractor, and the H2Ds run on a dedicated
  upload stream with a cross-stream `upload_done` event that the
  kernel stream waits on once before the per-plane kernel
  launches. Bit-exactness preserved (cross-backend `places=3`
  gate green for `psnr_hvs_y / cb / cr / psnr_hvs`); fast unit
  tests + Netflix golden gate green (the same 9 pre-existing
  pypsnr / niqe / result-store failures observed on `master`
  are unrelated). Wall-clock impact at the small-frame
  measurement points used in this PR is below noise floor (CPU
  `uint→float` convert remains the inner bottleneck), but the
  no-per-frame-alloc invariant is a prerequisite for the planned
  CUDA graph capture optimisation tracked in the graph-capture
  research digest accompanying PR #308. See
  [`libvmaf/src/cuda/AGENTS.md`](libvmaf/src/cuda/AGENTS.md)
  §Rebase-sensitive invariants for the rebase invariant.
- **U-2-Net `u2netp` saliency replacement deferred (T6-2a-followup' /
  ADR-0265 / Research-0054)** — second blocker following ADR-0257
  (PR #328). [ADR-0257](docs/adr/0257-mobilesal-real-weights-deferred.md)
  recommended swapping the underlying model family from MobileSal to
  U-2-Net's `u2netp` (Apache-2.0, ~4.7 MB, pure RGB). Attempting that
  swap blocks on two independent findings — captured in
  [Research-0054](docs/research/0055-u2netp-saliency-replacement-survey.md):
  (1) upstream [`xuebinqin/U-2-Net`](https://github.com/xuebinqin/U-2-Net)
  carries a clean SPDX `Apache-2.0` `LICENSE`, but `u2netp.pth` is
  distributed only via Google Drive viewer URLs (no GitHub release,
  no pinnable raw URL — same blocker as MobileSal in ADR-0257); and
  (2) U-2-Net's `F.upsample(..., mode='bilinear')` lowers to the
  ONNX `Resize` op which is **not** on the fork's
  `libvmaf/src/dnn/op_allowlist.c`, and bilinear resampling has no
  exact decomposition into the existing allowlist primitives at
  dynamic stride. The smoke-only synthetic placeholder
  (`mobilesal_placeholder_v0`, `smoke: true`) remains shipped
  unchanged; the C-side `feature_mobilesal.c` extractor and its
  smoke test are not touched. Three follow-up rows filed in ADR-0265
  §"Neutral / follow-ups" (`T6-2a-widen-allowlist-resize`,
  `T6-2a-mirror-u2netp-via-release`, `T6-2a-train-saliency-student`).
  Aligns with the task-brief "don't fake it" directive — records the
  real reasons real weights aren't shipping rather than producing a
  graph that would look like real weights but couldn't be. Companion
  to [ADR-0218](docs/adr/0218-mobilesal-saliency-extractor.md) and
  [ADR-0257](docs/adr/0257-mobilesal-real-weights-deferred.md).
- **`ssim_accumulate_avx512` per-lane double reduction vectorised in
  `__m512d` (PR #333 opt #2).** Replaces the 16-iteration scalar
  per-lane loop (and the 384-byte `_Alignas(64) float[16]×6` stack
  spill) with two 8-wide `__m512d` passes that compute `lv`, `cv`,
  `sv`, and `lv*cv*sv` per lane in vector double, then spill to
  `_Alignas(64) double[16]×4` for a final left-to-right scalar
  accumulation into `local_ssim/l/c/s`. The lane-by-lane summation
  order is unchanged — the running-sum associativity invariant from
  ADR-0139 is preserved, so per-frame XML output is byte-for-byte
  identical at `--precision max` against scalar, AVX2, and the prior
  AVX-512 implementation on both Netflix `src01_hrc00/01_576x324`
  and the `checkerboard_1920_1080_10_3_0_0/1_0` pair. AVX2 / NEON
  twins kept on the existing per-lane scalar path (verified
  bit-identical against the new AVX-512). Wall-clock end-to-end
  delta on the 30-frame 1920×1080 `float_ms_ssim` workload sits in
  the noise floor (~±0.3% across 15 reps); the change is a register-
  pressure / stack-spill reduction and the per-lane scalar
  divisions were not the dominant cost on this microbenchmark.

- **`ms_ssim_vulkan.c` submit-side migrated to `vulkan/kernel_template.h`
  submit helpers (T-GPU-DEDUP-26).** The pipeline-side bundles
  (`pl_decimate`, `pl_ssim` — 2 bundles, 4 decimate variants + 9 SSIM
  variants total) were already template-managed; this PR replaces the
  remaining hand-rolled `vkCommandBufferAllocateInfo` /
  `vkAllocateCommandBuffers` / `vkBeginCommandBuffer` /
  `vkFenceCreateInfo` / `vkCreateFence` / `vkQueueSubmit` /
  `vkWaitForFences` / `vkDestroyFence` / `vkFreeCommandBuffers` blocks
  in `extract()` with `vmaf_vulkan_kernel_submit_begin` /
  `_end_and_wait` / `_free` triples (one for the pyramid-decimate
  command buffer, one per scale in the per-scale SSIM loop). Drops the
  `cleanup_cmd:` label and ~50 lines of boilerplate. Numerical contract
  unchanged — `places=4` cross-backend gate validated on
  `testdata/ref_576x324_48f.yuv` ↔ `dis_576x324_48f.yuv` (max abs diff
  1e-06, 0/48 mismatches; see ADR-0214). Netflix CPU goldens untouched.

- **Snapshot `testdata/netflix_benchmark_results.json` regenerated
  against the fork build to reflect upstream `a44e5e61` motion
  edge-mirror fix; `bench_all.sh` default `VMAF` pinned to the in-tree
  fork build (`libvmaf/build/tools/vmaf`) instead of
  `/usr/local/bin/vmaf` (which on most dev hosts is stuck at v3.0.0,
  predating the upstream fix). CPU pooled VMAF on `src01_576x324`
  shifts from 76.668904 → 76.667828 (delta −1.076e-3, motion-driven);
  checker pairs drift ≤3e-6 (well inside `places=4`). cuda / sycl
  rows preserved unchanged (those backends were not regenerated this
  pass — see PR description). See PR #305 for the bisect that
  identified the upstream commit.

- **Repaired 4 wrong-NNNN ADR refs in CHANGELOG / rebase-notes (PR #304
  follow-up).** The 11 cases PR #304 deliberately skipped were re-audited
  against the current `docs/adr/` tree — 7 of them already resolve to
  existing files (the duplicate-NNNN ADRs landed since #304 ran its scan).
  The remaining 4 were genuine slug-drift cases where the cited filename
  did not exist on disk: `0138-iqa-convolve-avx2-bitexact-double.md` →
  `0138-iqa-convolve-avx2-bitexact-double.md`,
  `0140-simd-dx-framework.md` → `0140-simd-dx-framework.md`,
  `0190-float-ms-ssim-cuda.md` → `0190-ms-ssim-vulkan.md`,
  `0178-vulkan-adm-kernel.md` → `0178-vulkan-adm-kernel.md`. All four
  retain the original cited NNNN; only the slug was updated to the
  on-disk filename. No ADR content changed; no NNNN renumbered.

- **`cambi_vulkan.c` migrated to `vulkan/kernel_template.h`
  (T-GPU-DEDUP-25, 5-bundle).** Five distinct push-constant struct
  sizes (one per pipeline stage — preprocess / derivative /
  filter_mode / decimate / mask_dp) force five bundles even though
  every stage uses the same 2-binding SSBO descriptor-set layout
  shape. State drops the legacy quintet
  (`dsl_2bind` + 5× `pl_layout_*` + `shader_modules[CAMBI_PL_COUNT]`
  + shared `desc_pool`) for five `VmafVulkanKernelPipeline` bundles
  (`pl_trivial` / `pl_derivative` / `pl_filter_mode` /
  `pl_decimate` / `pl_mask_dp`), each owning its own descriptor
  pool. The first slot of `pipelines[]` per stage aliases the
  bundle's base pipeline; remaining slots
  (`CAMBI_PL_FILTER_MODE_V`, `CAMBI_PL_MASK_SAT_COL`,
  `CAMBI_PL_MASK_THRESHOLD`) are siblings via
  `vmaf_vulkan_kernel_pipeline_add_variant()`. Validated bit-exact
  against the pre-migration binary on the Netflix-pair smoke
  (576×324×8-bit, `cambi` mean = 0.0 — content-appropriate; the
  pair has no banding artifacts). Net diff −40 LOC (1407 → 1367).
  See [ADR-0221](docs/adr/0246-gpu-kernel-template.md).
- **`vulkan/kernel_template.h` SSBO-binding cap is now a named
  constant.** Replaced the open-coded `8U` upper bound and matching
  `bindings[8]` stack array with `#define
  VMAF_VULKAN_KERNEL_MAX_SSBO_BINDINGS 16U`. Current consumers top
  out at 10 (the SSIM bundle in `ssimulacra2_vulkan.c` uses 8); the
  new cap of 16 absorbs near-future kernels without further edits.
  Vulkan's conformant minimum for `maxDescriptorSetStorageBuffers`
  is 96, so the higher cap remains portable across drivers. No
  behavioural change — same allowed kernel shapes, same scores,
  same public C API.

### Fixed

- **`scripts/ci/deliverables-check.sh` now strips backslashes
  from PR bodies before grepping the deliverable checklist.** The
  script previously stripped backticks, asterisks, and underscores
  only; PRs created via heredoc-quoted body strings in `gh pr
  create` calls escape embedded backticks (the shell emits a
  literal backslash before each backtick), leaving a stray
  backslash in the PR body. After the partial strip the body
  contained "AGENTS.md\ invariant note" (backslash-space), which
  broke the `- [x].*AGENTS.md invariant note` regex and falsely
  flagged the deliverable as missing on otherwise-correct PRs.
  Extending the strip set to also remove backslashes restores the
  regex match.

- **T7-5 Sweeps B+C — fork-added ssimulacra2 SIMD + scalar + tests
  now carry inline ADR citations on every `readability-function-size`
  NOLINT (cite-only, no code splits).** 30 NOLINTs across 6 files:
  `ssimulacra2_avx2.c` (5), `ssimulacra2_avx512.c` (5),
  `ssimulacra2_neon.c` (6), `ssimulacra2_sve2.c` (6) — all cite the
  bit-exactness invariant (ADR-0138/0139, ADR-0141) since splitting
  would perturb register allocation + reduction order vs scalar.
  `ssimulacra2.c` scalar (3) cites the SIMD-parity audit
  (ADR-0141) — splitting would force matching splits in 4 paired
  SIMD files. `test_ssimulacra2_simd.c` (5) cites test scaffolding
  (ADR-0141). Companion to T7-5 Sweep A (PR #82, ADR-0146,
  upstream-mirror files). Pure documentation change — no
  behavioural delta; Netflix golden + cross-backend bit-exactness
  unchanged.
- **`vmaf_tiny_v1.onnx` external-data filename ref repaired.** The
  shipped v1 ONNX referenced an external-data file named
  `mlp_small_final.onnx.data`, which never existed in `model/tiny/`
  — only `vmaf_tiny_v1.onnx.data` was committed. ONNXRuntime fails
  with "External data path validation failed for initializer:
  0.weight" on load, breaking any consumer that loads v1 directly
  (including the `v2-vs-v1` diff path in
  `validate_vmaf_tiny_v2.py`). Rewrites the two `external_data`
  location entries in the ONNX graph to point at the actual
  filename. No tensor data changes; the `.data` file on disk is
  bit-identical.

- **`ssimulacra2_vulkan.c` migrated to `vulkan/kernel_template.h`
  (T-GPU-DEDUP-24, 4-bundle).** Four distinct pipeline shapes (XYB =
  6 SSBO bindings, MUL = 3, BLUR = 2, SSIM = 8) prevent collapsing
  to a single bundle — `_add_variant()` only siblings pipelines
  under the *same* layout. State drops 16 long-lived pipeline-object
  fields (4× `*_dsl` + `*_pl` + `*_shader` + the shared
  `desc_pool`) for four `VmafVulkanKernelPipeline` bundles
  (`pl_xyb` / `pl_mul` / `pl_blur` / `pl_ssim`), each owning its own
  descriptor pool. The first slot of each per-bundle pipeline array
  aliases the bundle's base pipeline; remaining slots
  (per-scale / per-pass variants) are siblings via
  `_add_variant()`. Validated bit-exact against the pre-migration
  binary on the Netflix-pair smoke (`ssimulacra2` mean 24.613842 on
  576×324×8-bit). Net diff −20 LOC (1544 → 1524). See
  [ADR-0246](docs/adr/0246-gpu-kernel-template.md).
- **`psnr_hvs_vulkan.c` migrated to `vulkan/kernel_template.h` +
  `_add_variant()` (T-GPU-DEDUP-18).** First multi-pipeline-via-variant
  consumer landed on top of PR #272 (which adds the
  `vmaf_vulkan_kernel_pipeline_add_variant()` helper). State
  collapses `dsl + pipeline_layout + shader + desc_pool +
  pipeline[3]` to `VmafVulkanKernelPipeline pl + VkPipeline
  pipeline_chroma_u + VkPipeline pipeline_chroma_v`. Plane 0
  (luma) is the template's base pipeline; planes 1+2 (chroma U/V)
  are sibling pipelines via `_add_variant()` — same layout +
  shader + DSL + pool, different `plane` spec-constant.
  Validated against the Netflix-pair smoke (`psnr_hvs` mean 31.33
  / `psnr_hvs_y` 30.58 / `psnr_hvs_cb` 37.26 / `psnr_hvs_cr` 38.20
  across 48 frames) and `meson test test_vulkan_smoke
  test_vulkan_async_pending_fence test_vulkan_pic_preallocation`
  (all green). Numerical contract unchanged.
- **`vif_vulkan.c` migrated to `vulkan/kernel_template.h` +
  `_add_variant()` (T-GPU-DEDUP-19).** State collapses
  `dsl + pipeline_layout + shader + desc_pool + pipelines[4]` to
  `VmafVulkanKernelPipeline pl + VkPipeline scale_variants[3]`.
  Scale 0 is the template's base pipeline; scales 1, 2, 3 are
  sibling pipelines via `_add_variant()` — same layout, shader,
  DSL, pool, different `SCALE` spec-constant. New
  `vif_scale_pipeline()` accessor maps scale index to the right
  `VkPipeline` handle. Validated against the Netflix-pair smoke
  (`integer_vif_scale0..3` means 0.364 / 0.767 / 0.863 / 0.916,
  `integer_vif` mean 0.446 across 48 frames). Numerical contract
- **`float_vif_vulkan.c` migrated to `vulkan/kernel_template.h` +
  `_add_variant()` (T-GPU-DEDUP-20).** Hardest variant case — 7
  pipelines across a 2-D `[mode][scale]` array (mode 0 compute
  scales 0-3, mode 1 decimate scales 1-3). State collapses
  `dsl + pipeline_layout + shader + desc_pool` to
  `VmafVulkanKernelPipeline pl`; the
  `VkPipeline pipelines[2][4]` 2-D lookup is preserved so the
  dispatch path stays clean, but `pipelines[0][0]` aliases
  `s->pl.pipeline` (template's base) and the other 6 are
  variants via `_add_variant()`. Validated against the
  Netflix-pair smoke (`vif_scale0..3` means 0.364, 0.767,
  0.863, 0.916 across 48 frames; matches `integer_vif`
  bit-identically to 4 decimals). Numerical contract
  unchanged.
- **`float_adm_vulkan.c` migrated to `vulkan/kernel_template.h` +
  `_add_variant()` (T-GPU-DEDUP-22).** Twin migration to
  `adm_vulkan` (T-GPU-DEDUP-21); same 16-pipeline 2-D
  `[stage][scale]` shape (4 stages × 4 scales). State collapses
  `dsl + pipeline_layout + shader + desc_pool` to
  `VmafVulkanKernelPipeline pl`; `pipelines[0][0]` aliases
  `s->pl.pipeline` (template's base), the other 15 entries are
  sibling pipelines via `_add_variant()`. Validated by `meson
  test test_vulkan_smoke test_vulkan_async_pending_fence
  test_vulkan_pic_preallocation` (all green) plus clean compile.
  Numerical contract unchanged.
- **`adm_vulkan.c` migrated to `vulkan/kernel_template.h` +
  `_add_variant()` (T-GPU-DEDUP-21).** 16-pipeline 2-D
  `[stage][scale]` array (4 stages × 4 scales). State collapses
  `dsl + pipeline_layout + shader + desc_pool` to
  `VmafVulkanKernelPipeline pl`. The
  `VkPipeline pipelines[4][4]` 2-D lookup is preserved so the
  per-stage dispatch path stays clean, but `pipelines[0][0]`
  aliases `s->pl.pipeline` (template's base) and the other 15
  entries are sibling pipelines via `_add_variant()`. Validated
  by `meson test test_vulkan_smoke
  test_vulkan_async_pending_fence test_vulkan_pic_preallocation`
  (all green) plus clean compile. Numerical contract unchanged.

- **`ms_ssim_vulkan.c` 2-bundle migration to `vulkan/kernel_template.h`
  (T-GPU-DEDUP-23).** The kernel has two distinct pipeline shapes —
  decimate (2 SSBO bindings) and ssim (10 SSBO bindings) — which
  sibling-only `_add_variant()` cannot collapse to one bundle.
  State drops 7 long-lived pipeline-object fields (`decimate_dsl`,
  `decimate_pl`, `decimate_shader`, `ssim_dsl`, `ssim_pl`,
  `ssim_shader`, shared `desc_pool`) for two
  `VmafVulkanKernelPipeline` bundles (`pl_decimate` + `pl_ssim`).
  `decimate_pipelines[0]` aliases `pl_decimate.pipeline` (the
  template's base = scale 0); the other 3 decimate slots are
  siblings via `_add_variant()`. `ssim_pipeline_horiz[0]` aliases
  `pl_ssim.pipeline` (scale 0, pass 0); the other 9 ssim slots
  (4× horiz scales 1..4 + 5× vert scales 0..4) are variants.
  `create_pipelines()` shrinks from ~115 LOC of
  `vkCreateDescriptorSetLayout`/`vkCreatePipelineLayout`/
  `vkCreateShaderModule`/`vkCreateDescriptorPool` boilerplate to
  ~75 LOC of two `_pipeline_create()` + variant loops.
  `close_fex()`'s 8×`vkDestroy*` sweep across two pipeline shapes
  collapses to two `_pipeline_destroy()` calls plus the
  per-variant pipeline destroys. `alloc_descriptor_set()` now
  takes the bundle pointer (per-bundle pool + DSL) instead of the
  shared `s->desc_pool`. **Net −18 LOC** in `ms_ssim_vulkan.c`
  (savings concentrated in `create_pipelines()` and `close_fex()`,
  partly offset by hoisted spec-data fill helpers).
  Bit-exactness preserved —
  Netflix-pair `float_ms_ssim` smoke (576×324×48f) reports mean
  0.963241, identical to pre-migration.
- **`psnr_vulkan.c` migrated to `vulkan/kernel_template.h` (T-GPU-DEDUP-5,
  first consumer).** The dormant `vulkan/kernel_template.h` (410 LOC,
  ADR-0246) shipped with zero consumers; its docstring designated
  `psnr_vulkan.c` as the reference implementation. This commit lands
  that migration. The 5 long-lived pipeline objects (DSL, pipeline
  layout, shader module, compute pipeline, descriptor pool) collapse
  to one `VmafVulkanKernelPipeline pl` bundle. `create_pipeline()`
  shrinks ~104 LOC → ~30 LOC via
  `vmaf_vulkan_kernel_pipeline_create()`. `close_fex()`'s
  `vkDeviceWaitIdle` + 5×`vkDestroy*` sweep collapses to one
  `vmaf_vulkan_kernel_pipeline_destroy()`. Bit-exactness preserved —
  spec-constants, push-constant struct, shader bytecode, dispatch
  grid math, and host-side reduction are byte-identical to the prior
  implementation. **Net −55 LOC.** Pairs with PR #269 (first CUDA
  template consumer) — both dormant kernel templates now have real
  consumers.

- **`moment_vulkan.c` + `ciede_vulkan.c` migrated to
  `vulkan/kernel_template.h` (T-GPU-DEDUP-6).** Second + third
  consumers of the dormant Vulkan kernel template after PR #270
  (psnr_vulkan, the first consumer). Both files follow the
  identical migration pattern: 5 individual pipeline-object fields
  (`dsl`, `pipeline_layout`, `shader`, `pipeline`, `desc_pool`)
  collapse to one `VmafVulkanKernelPipeline pl`; ~100 LOC of
  `create_pipeline()` body collapses to a single
  `vmaf_vulkan_kernel_pipeline_create()` call;
  `close_fex()`'s `vkDeviceWaitIdle` + 5×`vkDestroy*` sweep
  collapses to one `vmaf_vulkan_kernel_pipeline_destroy()`. Net
  **−119 LOC** (moment −60, ciede −59). Bit-exactness preserved —
  spec-constants, push-constant structs, shader bytecodes,
  dispatch grid math, and host-side reductions are byte-identical.
  `motion_vulkan.c` deferred (uses two pipelines sharing a layout;
  needs a multi-pipeline template extension).
- **Tiny-AI test registration macro (`tiny_ai_test_template.h`).**
  The four tiny-AI extractor test files (`test_lpips.c`,
  `test_mobilesal.c`, `test_transnet_v2.c`, `test_fastdvdnet_pre.c`)
  shipped near-identical ~140 LOC of registration smoke tests each.
  New `libvmaf/test/tiny_ai_test_template.h` emits the four standard
  tests via `VMAF_TINY_AI_DEFINE_REGISTRATION_TESTS(ext, feat, env,
  prefix)`; each per-extractor test file now lands in 20-50 LOC.
  Behavior bit-exact preserved (same assertions, same env-var
  save/restore dance, same `_putenv_s` shim for MSVCRT). TransNet V2
  keeps its two extractor-specific extra tests (binary-flag
  round-trip + `provided_features` NULL-termination check). Net
  −286 LOC. Top-leverage non-GPU dedup finding from the
  2026-05-02 whole-codebase audit.
- **Vulkan kernel template — multi-pipeline support + ssim/motion
  migration (T-GPU-DEDUP-7).** Extended
  [`libvmaf/src/vulkan/kernel_template.h`](libvmaf/src/vulkan/kernel_template.h)
  with `vmaf_vulkan_kernel_pipeline_add_variant()` so kernels that
  need multiple `VkPipeline`s sharing one layout / shader / DSL /
  descriptor pool (different spec-constant) can express that
  without re-implementing the boilerplate. Migrated
  [`libvmaf/src/feature/vulkan/motion_vulkan.c`](libvmaf/src/feature/vulkan/motion_vulkan.c)
  and
  [`libvmaf/src/feature/vulkan/ssim_vulkan.c`](libvmaf/src/feature/vulkan/ssim_vulkan.c)
  to the template — net duplicated DSL / pipeline layout / shader /
  pipeline / descriptor-pool create-and-destroy code removed. As
  part of motion's migration, the dormant `pipelines[2]` array
  (kept "for SYCL parity" but functionally identical because
  COMPUTE_SAD is passed via push constants, not spec-constants) was
  collapsed to a single `VkPipeline`. SSIM still owns two pipelines
  (horizontal pass=0 / vertical pass=1 differ in spec constant) —
  pass=0 is the template's base pipeline; pass=1 is attached via
  `_add_variant()`. Validated against the Netflix-pair smoke
  (`float_ssim` mean 0.863, 48 frames) + `meson test
  test_vulkan_smoke test_vulkan_async_pending_fence
  test_vulkan_pic_preallocation` (all green).

- **`integer_psnr_cuda.c` migrated to `cuda/kernel_template.h`
  (T-GPU-DEDUP-4, first consumer).** The dormant
  `cuda/kernel_template.h` (296 LOC, ADR-0246) shipped with zero
  consumers; its docstring designated `integer_psnr_cuda.c` as the
  reference implementation. This commit lands the migration: the
  per-frame async lifecycle (private stream + submit/finished
  event pair) and the device + pinned-host readback pair now come
  from the template helpers
  (`vmaf_cuda_kernel_lifecycle_init/_close`,
  `vmaf_cuda_kernel_readback_alloc/_free`,
  `vmaf_cuda_kernel_submit_pre_launch`,
  `vmaf_cuda_kernel_collect_wait`). `PsnrStateCuda` shrinks
  (3 fields → 1 `VmafCudaKernelLifecycle`; 2 fields → 1
  `VmafCudaKernelReadback`). Bit-exactness preserved — the
  kernel launch, the per-bpc function lookup, the SSE accumulator
  math, and the host-side `log10` score formula are byte-identical
  to the prior implementation. Single-consumer net LOC delta is
  +8 (helpers add per-call boilerplate); the dedup win materialises
  as more CUDA feature kernels migrate one-at-a-time in follow-ups.
- **`integer_ciede_cuda.c` migrated to `cuda/kernel_template.h`
  (T-GPU-DEDUP-11).** Second consumer of the CUDA template (after
  `integer_psnr_cuda` in PR #269). State collapses
  `CUstream str + CUevent event + CUevent finished + VmafCudaBuffer
  *partials + float *partials_host` to
  `VmafCudaKernelLifecycle lc + VmafCudaKernelReadback rb`.
  `init_fex_cuda` calls `vmaf_cuda_kernel_lifecycle_init` and
  `vmaf_cuda_kernel_readback_alloc`; `collect_fex_cuda` calls
  `vmaf_cuda_kernel_collect_wait`; `close_fex_cuda` calls
  `vmaf_cuda_kernel_lifecycle_close` +
  `vmaf_cuda_kernel_readback_free`. `submit_fex_cuda` keeps the
  pre-launch `cuStreamWaitEvent` inline rather than calling
  `vmaf_cuda_kernel_submit_pre_launch` because ciede's per-block
  kernel writes one float without an atomic — the template's
  device-side memset is unnecessary. Numerical contract unchanged
  (`places=4` per ADR-0187).
- **`integer_moment_cuda.c` migrated to `cuda/kernel_template.h`
  (T-GPU-DEDUP-12).** Third consumer of the CUDA template (after
  `integer_psnr_cuda` in PR #269 and `integer_ciede_cuda` in
  PR #277). State collapses `CUstream + CUevent + CUevent +
  VmafCudaBuffer + uint64_t*` to
  `VmafCudaKernelLifecycle lc + VmafCudaKernelReadback rb`.
  Unlike ciede, moment uses atomic-add reduction (4× uint64
  counters), so `submit_fex_cuda` calls
  `vmaf_cuda_kernel_submit_pre_launch` for the mandatory
  device-side memset + dist-ready wait. init / collect / close
  use the matching template helpers. Numerical contract
  unchanged — same atomic accumulators, same
  `sums_host[i] / n_pixels` host computation.
- **`integer_motion_v2_cuda.c` migrated to `cuda/kernel_template.h`
  (T-GPU-DEDUP-13).** Fourth consumer of the CUDA template. State
  collapses `CUstream + CUevent + CUevent + sad device buffer +
  sad host pinned` to
  `VmafCudaKernelLifecycle lc + VmafCudaKernelReadback rb`, with
  the ping-pong of raw ref Y planes (`pix[2]`) kept outside the
  bundle (template models a single device+host pair, not a
  device-only ring). `submit_fex_cuda` keeps the memset on
  `pic_stream` (not `lc.str`) inline because the kernel reads the
  accumulator before the D2H copy on `lc.str`. Numerical contract
  unchanged.
- **`integer_ssim_cuda.c` migrated to `cuda/kernel_template.h`
  (T-GPU-DEDUP-14).** Fifth consumer of the CUDA template (after
  psnr / ciede / moment / motion_v2). State collapses
  `CUstream str + CUevent event + CUevent finished + VmafCudaBuffer
  *partials + float *partials_host` to
  `VmafCudaKernelLifecycle lc + VmafCudaKernelReadback rb`. The
  five intermediate float buffers (`h_ref_mu`, `h_cmp_mu`,
  `h_ref_sq`, `h_cmp_sq`, `h_refcmp`) stay outside the bundle —
  the template models a single device+host pair, not a
  five-buffer pyramid. `submit_fex_cuda` keeps the inline
  `cuStreamWaitEvent + horiz launch + vert launch + DtoH` chain
  rather than calling `vmaf_cuda_kernel_submit_pre_launch` because
  the kernel writes one float per block (no atomic) — the
  template's pre-launch memset is unnecessary. `init_fex_cuda` /
  `collect_fex_cuda` / `close_fex_cuda` use the matching
  template helpers. Numerical contract unchanged
  (`places=4` per the ciede_cuda precision pattern).
- **`integer_ms_ssim_cuda.c` + `integer_psnr_hvs_cuda.c` migrated to
  `cuda/kernel_template.h` (T-GPU-DEDUP-15).** Sixth + seventh
  consumers of the CUDA template. Both kernels have heavy
  multi-buffer state (5-level pyramid + 5 SSIM intermediates +
  3 partials triples for ms_ssim; 3-plane ref/dist/partials triples
  for psnr_hvs) that doesn't fit the template's single-pair
  readback bundle, but the lifecycle (stream + submit/finished
  events) is a clean win on each side. State on each side
  collapses `CUstream str + CUevent event + CUevent finished` to
  a single `VmafCudaKernelLifecycle lc`; init / close call the
  matching template helpers. Submit / collect references updated
  via mechanical sed (`s->str` → `s->lc.str`, `s->event` →
  `s->lc.submit`, `s->finished` → `s->lc.finished`). Numerical
  contracts unchanged.
- **`float_psnr_cuda.c` + `float_ansnr_cuda.c` + `float_motion_cuda.c`
  migrated to `cuda/kernel_template.h` (T-GPU-DEDUP-16).** Eighth,
  ninth, and tenth consumers of the CUDA template. Each kernel
  follows the same pattern: state collapses
  `CUstream + CUevent + CUevent + VmafCudaBuffer *partials +
  float *partials_host` to
  `VmafCudaKernelLifecycle lc + VmafCudaKernelReadback rb`,
  with input upload buffers (ref/dis/blur ping-pong) kept outside.
  `init_fex_cuda` calls `vmaf_cuda_kernel_lifecycle_init` +
  `vmaf_cuda_kernel_readback_alloc`; `collect_fex_cuda` /
  `flush_fex_cuda` call `vmaf_cuda_kernel_collect_wait`;
  `close_fex_cuda` calls `vmaf_cuda_kernel_lifecycle_close` +
  `vmaf_cuda_kernel_readback_free`. The `cuMemsetD8Async`
  pre-launch is kept inline on `pic_stream` rather than calling
  `vmaf_cuda_kernel_submit_pre_launch` because the kernels read
  the accumulator before the D2H copy on `lc.str` (same rationale
  as `motion_v2_cuda` in PR #279). Numerical contracts unchanged.
- **`float_adm_cuda.c` + `float_vif_cuda.c` lifecycle migrated to
  `cuda/kernel_template.h` (T-GPU-DEDUP-17).** Eleventh and twelfth
  consumers of the CUDA template. Both kernels are multi-stage
  pipelines — float_adm runs 16 launches (4 stages × 4 scales)
  with DWT + CSF + decouple intermediates, float_vif runs 4
  compute + 3 decimate launches across a 4-level pyramid — so the
  template's single-pair readback bundle isn't a fit. The
  lifecycle helper still applies cleanly: state collapses
  `CUstream str + CUevent event + CUevent finished` to
  `VmafCudaKernelLifecycle lc`; init / close call the matching
  template helpers; submit / collect references updated via
  mechanical sed (`s->str` → `s->lc.str`, `s->event` →
  `s->lc.submit`, `s->finished` → `s->lc.finished`). Numerical
  contracts unchanged.
- **`float_psnr_vulkan.c` migrated to `vulkan/kernel_template.h`
  (T-GPU-DEDUP-8).** Net **−95 LOC** of duplicated
  DSL / pipeline layout / shader / pipeline / descriptor-pool
  create-and-destroy code. State collapses
  `VkDescriptorSetLayout dsl + VkPipelineLayout pipeline_layout +
  VkShaderModule shader + VkPipeline pipeline + VkDescriptorPool
  desc_pool` to a single `VmafVulkanKernelPipeline pl`. Single
  dispatch, no spec-constant variants — uses the template's base
  `vmaf_vulkan_kernel_pipeline_create` directly. Validated against
  the Netflix-pair smoke (`float_psnr` mean 30.755 dB, 48 frames)
  and `meson test test_vulkan_smoke test_vulkan_async_pending_fence
  test_vulkan_pic_preallocation` (all green). Numerical contract
  unchanged — same shader / spec-constants / push-constants;
  only the Vulkan boilerplate moved to a shared owner.
- **`float_ansnr_vulkan.c` + `motion_v2_vulkan.c` migrated to
  `vulkan/kernel_template.h` (T-GPU-DEDUP-9).** Two single-pipeline
  Vulkan kernels migrated in one PR. State on each side collapses
  the `dsl + pipeline_layout + shader + pipeline + desc_pool`
  quintet to a single `VmafVulkanKernelPipeline pl`;
  `create_pipelines` and `close_fex` shrink to template-driven
  create + destroy. Net diff is dominated by deletions (~190 LOC of
  duplicated boilerplate). No shader / spec-constant /
  push-constant changes. Validated against the Netflix-pair smoke
  (`float_ansnr` mean 23.51 dB, `float_anpsnr` mean 34.16 dB,
  `motion2_v2_score` mean 3.895 across 48 frames) and `meson test
- **`float_motion_vulkan.c` migrated to `vulkan/kernel_template.h`
  (T-GPU-DEDUP-10).** Single-pipeline float-motion kernel migrated;
  state collapses `dsl + pipeline_layout + shader + pipeline +
  desc_pool` to a single `VmafVulkanKernelPipeline pl`;
  `create_pipelines` and `close_fex` shrink to template-driven
  create + destroy. No shader / spec-constant / push-constant
  changes. Validated against the Netflix-pair smoke (`motion` mean
  4.049, `motion2` mean 3.894 across 48 frames) and `meson test
  test_vulkan_smoke test_vulkan_async_pending_fence
  test_vulkan_pic_preallocation` (all green). Numerical contract
  unchanged.

- **`feature_mobilesal.c` + `transnet_v2.c` migrated to `tiny_extractor_template.h`.**
  PR #251 shipped the shared template (`vmaf_tiny_ai_resolve_model_path`,
  `vmaf_tiny_ai_open_session`, `vmaf_tiny_ai_yuv8_to_rgb8_planes`,
  `VMAF_TINY_AI_MODEL_PATH_OPTION`) but left the existing
  `feature_mobilesal.c` + `transnet_v2.c` open-coding the same
  boilerplate. Both files now consume the template, eliminating
  ~98 LOC of duplicated model-path lookup / session-open log /
  YUV→RGB kernel / VmafOption row. Behavior bit-exact preserved
  (the template hoists the literal copies the migrated files
  carried). `feature_lpips.c` and `fastdvdnet_pre.c` were already
  migrated. Closes the AI-template adoption gap noted in the
  2026-05-02 dedup audit. See `dnn/tiny_extractor_template.h`.
- **GPU picture pool dedup — `cuda/ring_buffer.{c,h}` →
  `gpu_picture_pool.{c,h}` (ADR-0239).** The CUDA picture-pool
  primitive is promoted out of `libvmaf/src/cuda/` into the
  backend-agnostic `libvmaf/src/gpu_picture_pool.{c,h}` (Netflix's
  `VmafRingBuffer` was always callback-based and shape-agnostic;
  only the directory and symbol names implied otherwise). Symbols
  rename: `VmafRingBuffer` → `VmafGpuPicturePool`,
  `vmaf_ring_buffer_*` → `vmaf_gpu_picture_pool_*`. SYCL's
  `vmaf_sycl_picture_pool_*` keeps its public-internal API but
  now delegates to the generic pool — the SYCL wrapper just owns
  the `VmafSyclCookie` storage; `std::mutex` drops out. Vulkan's
  `picture_vulkan_pool.c` (added in PR #264) rewrites as a thin
  wrapper around the generic pool with the same pattern. Net
  structural win: ONE round-robin / mutex / unwind implementation
  across all three GPU backends. The `Netflix#1300`
  mutex-destroy-order fix (ADR-0157) travels with the file. Test
  renamed `test_ring_buffer.c` → `test_gpu_picture_pool.c`.

- **GPU backend public-API pattern doc (ADR-0240).** New
  [`docs/development/gpu-backend-template.md`](docs/development/gpu-backend-template.md)
  ships the recipe new GPU backends follow — shared lifecycle
  (`vmaf_<backend>_state_init` / `_import_state` / `_state_free`),
  optional sections (`_available` / `_list_devices` / picture
  preallocation / hwaccel zero-copy import), Doxygen + ABI
  conventions, and the SYCL/Vulkan `NONE / HOST / DEVICE`
  three-method picture-preallocation convention as the
  new-backend default. New
  [`libvmaf/include/libvmaf/AGENTS.md`](libvmaf/include/libvmaf/AGENTS.md)
  pins the public-headers-tree invariant (rebase ordering of the
  four backend headers, ABI-additive rule). PR3 of the GPU dedup
  sequence — doc-only, not codegen, after a 2026-05-02 audit
  measured the four headers at ~20 of ~200 lines truly shared
  (state lifecycle); the rest is genuinely backend-specific
  feature surface. Mirrors the tiny-AI ADR-0250 "recipe doc +
  shared helpers, not codegen" precedent.
- **ADR-0108 deliverables gate now runnable locally (`make pr-check`).**
  The Deep-Dive Deliverables Checklist gate
  (`.github/workflows/rule-enforcement.yml`) previously inlined ~80
  lines of bash that parsed PR bodies + verified ticked file
  references against the PR diff. The check is fundamentally a
  PR-body-vs-diff coherence test, which pre-commit hooks cannot run
  (neither artefact exists at commit time). The bash now lives in
  [`scripts/ci/deliverables-check.sh`](scripts/ci/deliverables-check.sh)
  as the single source of truth, and the workflow calls the script
  in one step. New `make pr-check PR=<num>` (or `make pr-check
  BODY=<file>`) target runs the same gate locally before
  `gh pr create` — saves the typical 60-second CI round-trip when
  a checkbox is ticked instead of opted out (the bug that hit
  PR #260's first run).
- **Top-level docs refresh (post-session-2026-04-29).** `README.md`,
  `CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`, and `docs/principles.md`
  refreshed to current codebase reality: Vulkan backend (T5-1 / T7-36),
  embedded MCP scaffold (T5-2), GPU-parity CI gate (T6-8 / ADR-0214),
  KoNViD-1k tiny-AI corpus, fp64-less SYCL fallback (T7-17), `--tiny-*`
  CLI flags, `enable_mcp` Meson option, hooks-install quickstart step,
  and CLAUDE §12 r12-r14 quality-gate rules (touched-file lint-clean,
  state.md updates, ffmpeg-patches sync). No code or schema changes.
- **User-facing docs refresh (post-session 2026-04-29).** Refreshed
  [`docs/metrics/features.md`](docs/metrics/features.md) (de-duplicated
  the extractor overview table; CAMBI Vulkan, integer-SSIM Vulkan,
  SSIMULACRA 2 CUDA / SYCL twins, PSNR Vulkan chroma WIP all now
  reflected with footnotes) and [`docs/api/gpu.md`](docs/api/gpu.md)
  (Vulkan section flipped from "scaffold only" to "T5-1c full
  default-model coverage"; added pointers to T7-29 image-import API
  and T7-10 HIP scaffold). No code changes; doc-only.
- **AI-tooling docs refresh (`.claude/`).** Skill descriptions and
  one agent description in `.claude/skills/*/SKILL.md` and
  `.claude/agents/vulkan-reviewer.md` have been refreshed to
  match the current state of the fork: `add-gpu-backend` now
  cites the Vulkan T5-1 scaffold ([ADR-0175](docs/adr/0175-vulkan-backend-scaffold.md))
  and image-import ([ADR-0186](docs/adr/0186-vulkan-image-import-impl.md))
  as the canonical recent precedent; `cross-backend-diff` and
  `validate-scores` mention the T6-8 GPU-parity CI gate
  ([ADR-0214](docs/adr/0214-gpu-parity-ci-gate.md)) as the
  contract their local-dev runs mirror, and add `vulkan` to the
  default backend list; `port-upstream-commit` extends the
  SIMD/GPU twin list with the Vulkan kernel + GLSL shader
  surfaces that any upstream port must propagate to. The
  `vulkan-reviewer` agent description and `## Status` block drop
  the stale "forward-looking — backend does not yet exist"
  framing now that `libvmaf/src/vulkan/` and
  `libvmaf/src/feature/vulkan/` are live across all features.
  Root `AGENTS.md` §7 skill table extends to cover the
  AI-tooling skills that existed but were not listed
  (`build-ffmpeg-with-vmaf`, `refresh-ffmpeg-patches`,
  `validate-scores`, `run-netflix-bench`, `bisect-model-quality`,
  `regen-docs`, `format-all`, `lint-all`, the four `dev-llm-*`).
  Hook header comments in `.claude/hooks/*.sh` were audited
  against current behavior and required no edits. No functional
  change to any hook script body. No user-discoverable surface
  changed.
- **Planning + AGENTS.md refresh** (fork-local doc): root
  `AGENTS.md` gains a new §13 "Rebase-sensitive invariants
  (project-wide)" indexing the shipped surfaces from the
  2026-04-29 session — GPU long-tail terminus
  ([ADR-0210](docs/adr/0210-cambi-vulkan-integration.md) closes
  the matrix), Vulkan backend + image import, ssim/ms_ssim
  Vulkan, motion_v2 GPU port, MCP scaffold (ADR-0209), HIP
  scaffold (ADR-0212 placeholder), SVE2 SIMD (ADR-0213
  placeholder), GPU-parity CI gate
  ([ADR-0214](docs/adr/0214-gpu-parity-ci-gate.md)), MobileSal
  (ADR-0218 placeholder), TransNet V2, FastDVDnet (ADR-0215
  placeholder), psnr chroma Vulkan (ADR-0216 placeholder), SYCL
  fp64-less device contract
  ([ADR-0220](docs/adr/0220-sycl-fp64-fallback.md)), model
  registry + Sigstore (ADR-0211 placeholder), and the upstream
  `feature/motion` port from `b949cebf`
  ([PR #197](https://github.com/lusoris/vmaf/pull/197)) +
  `feature/speed` port from `d3647c73`
  ([PR #213](https://github.com/lusoris/vmaf/pull/213)).
  [`libvmaf/AGENTS.md`](libvmaf/AGENTS.md),
  [`libvmaf/src/feature/AGENTS.md`](libvmaf/src/feature/AGENTS.md),
  [`libvmaf/src/sycl/AGENTS.md`](libvmaf/src/sycl/AGENTS.md),
  [`libvmaf/src/dnn/AGENTS.md`](libvmaf/src/dnn/AGENTS.md), and
  the new
  [`libvmaf/src/vulkan/AGENTS.md`](libvmaf/src/vulkan/AGENTS.md)
  pick up the per-subdirectory invariants. No code change.

- **`docs/state.md` post-2026-04-29-session refresh.** Updated
  `_Updated:` stamp to 2026-04-29; rewrote the "Tiny-AI C1 baseline
  `fr_regressor_v1.onnx`" deferral row in the "Deferred (waiting on
  external dataset access)" section to mark its reopen-trigger as
  TRIGGERED — the Netflix Public training corpus that gated C1 is
  now locally available at `.workingdir2/netflix/` (9 ref + 70 dis
  YUVs, ~37 GB, gitignored; provided by lawrence 2026-04-27),
  unblocking BACKLOG T6-1a. Other rows verified accurate vs the
  2026-04-29-session merged PR set (#193–#205, #209) — every PR was
  feature / chore / docs / perf with no bug-status delta to record
  per CLAUDE §12 rule 13. No code changes.
- **SIMD bit-exact test harness (ADR-0245).** New
  [`libvmaf/test/simd_bitexact_test.h`](libvmaf/test/simd_bitexact_test.h)
  centralises the per-test SIMD-parity scaffolding: `xorshift32` PRNG,
  portable POSIX/MinGW/MSVC aligned allocator, x86 AVX2 CPUID gate,
  and `SIMD_BITEXACT_ASSERT_MEMCMP` / `SIMD_BITEXACT_ASSERT_RELATIVE`
  assertion macros. `test_psnr_hvs_avx2.c`, `test_psnr_hvs_neon.c`,
  `test_moment_simd.c`, and `test_motion_v2_simd.c` migrate to the
  harness as proof — net `-106` LOC across the four files. New SIMD
  parity tests now cost ~20 LOC of test-body code instead of ~50–100
  LOC of scaffolding + body. `test_ssimulacra2_simd.c` is intentionally
  unchanged (its `fill_random` FP rounding order is load-bearing for
  input bit patterns; a separate dedup PR with snapshot rerun under
  `/cross-backend-diff` can migrate it). All 41 `meson test` cases
  pass post-refactor; clang-format clean. New
  [`libvmaf/test/AGENTS.md`](libvmaf/test/AGENTS.md) "New SIMD parity
  test" rebase-sensitive invariant row pins the include-order rule
  (`#include "test.h"` MUST precede `#include "simd_bitexact_test.h"`
  because `test.h` lacks a header guard). See
  [ADR-0245](docs/adr/0245-simd-bitexact-test-harness.md).
- **`nr_metric_v1` flips to dynamic-PTQ int8 (T5-3d / ADR-0248).**
  The C2 NR-MOS tiny model joins the dynamic-PTQ family alongside
  `learned_filter_v1` (ADR-0174). Root cause of the previous
  `quantize_dynamic` failure (`Inferred shape and existing shape
  differ in dimension 0: (128) vs (1)`) traced to
  `torch.onnx.export` duplicating every initialiser into
  `graph.value_info` with static-shape annotations that do not
  survive the dynamic-batch axis substitution — same class of bug
  fixed for `vmaf_tiny_v1*.onnx` in PR #174 (T5-3e). Ported the
  `_save_inlined` strip pattern into both
  [`ai/src/vmaf_train/models/exports.py`](ai/src/vmaf_train/models/exports.py)
  (so future fork-trained tiny models are PTQ-clean by
  construction) and
  [`ai/scripts/ptq_dynamic.py`](ai/scripts/ptq_dynamic.py) (so
  pre-existing on-disk ONNX files quantise without re-export).
  Re-saved `model/tiny/nr_metric_v1.onnx` with `value_info`
  duplicates stripped — ORT-CPU produces bit-identical inference
  before vs after on a deterministic 16-sample input set; the
  registry sha256 rolls forward `60c2bd59…` → `75eff676…`. Registry
  entry now carries `quant_mode: "dynamic"`, `int8_sha256:
  "e5ba2086…"`, and `quant_accuracy_budget_plcc: 0.01`.
  `ai/scripts/measure_quant_drop.py` reports `[PASS]
  PLCC=0.992326 drop=0.007674 budget=0.0100`. 2.0× size shrink
  (119 KB → 58 KB). See
  [ADR-0248](docs/adr/0248-nr-metric-v1-ptq.md).
- **Tiny-AI extractor template — shared scaffolding header (ADR-0250).**
  New
  [`libvmaf/src/dnn/tiny_extractor_template.h`](libvmaf/src/dnn/tiny_extractor_template.h)
  ships three `static inline` helpers
  (`vmaf_tiny_ai_resolve_model_path`, `vmaf_tiny_ai_open_session`,
  `vmaf_tiny_ai_yuv8_to_rgb8_planes`) plus one struct-literal-emitting
  macro (`VMAF_TINY_AI_MODEL_PATH_OPTION`) that deduplicate the
  model-path-option-then-env-var resolution + session-open
  log-line pattern + BT.709 limited-range YUV→RGB chroma upsample +
  `model_path` `VmafOption[]` row across the four tiny-AI extractors
  (`feature_lpips.c`, `fastdvdnet_pre.c`, in-flight
  `feature_mobilesal.c` from PR #208, planned `feature_transnet_v2.c`).
  `feature_lpips.c` shrinks 305 → 205 LOC and `fastdvdnet_pre.c`
  341 → 317 LOC (net −100 lines across the two master extractors);
  new tiny-AI extractors target ~30 LOC of extractor-specific tensor
  wiring instead of ~150 LOC where 70 % is plumbing. Bit-exact
  behaviour preserved (YUV→RGB body and option-table layout are
  literal moves; all 40 libvmaf CPU-build tests + the 10 dnn-suite
  smoke tests pass). Power-of-10 friendly — no setjmp/longjmp, no
  recursion, bounded loops, single struct-literal macro (rule 9
  compliant). Recipe doc
  [`docs/ai/extractor-template.md`](docs/ai/extractor-template.md);
  `libvmaf/src/dnn/AGENTS.md` invariant row pins the contract for
  rebase. See
  [ADR-0250](docs/adr/0250-tiny-ai-extractor-template.md).

### Changed

### Changed

- **SYCL fp64-less device init log (T7-17 / ADR-0220).** The init
  message emitted on devices that lack `sycl::aspect::fp64` (Intel
  Arc A-series, most Intel iGPUs, many mobile / embedded GPUs) is
  reworded from the misleading WARNING-level "device lacks fp64
  support — using int64 emulation for gain limiting" to an
  INFO-level "device lacks native fp64 — kernels already use fp32
  + int64 paths, no emulation overhead". An audit confirmed every
  SYCL feature kernel is already fp64-free in its device code:
  ADM gain limiting uses an int64 Q31 split-multiply
  (`gain_limit_to_q31` + `launch_decouple_csf<false>` in
  `libvmaf/src/feature/sycl/integer_adm_sycl.cpp`), VIF gain
  limiting uses fp32 `sycl::fmin`, and accumulators use
  `sycl::plus<int64_t>`. There is no fp64-emulation fallback — the
  previous wording suggested one. New
  [`docs/backends/sycl/overview.md`](docs/backends/sycl/overview.md)
  § "fp64-less device contract (T7-17)" documents the
  no-`double`-in-kernel-lambdas rule + the SPIR-V module-taint
  rationale; new `libvmaf/src/sycl/AGENTS.md` invariant row pins
  the contract on rebase. The originally reported 5–10× Arc A380
  vs Vulkan perf gap has a different root cause (kernel geometry,
  sub-group size, memory pattern) — out of T7-17's scope. See
  [ADR-0220](docs/adr/0220-sycl-fp64-fallback.md).

### Added

- **GPU-parity matrix CI gate (T6-8 / ADR-0214).** New
  [`scripts/ci/cross_backend_parity_gate.py`](scripts/ci/cross_backend_parity_gate.py)
  iterates every `(feature, backend-pair)` cell, diffs per-frame
  metrics with a feature-specific absolute tolerance declared in
  one place (`FEATURE_TOLERANCE`), and emits one JSON record + one
  Markdown row per cell. New CI lane `vulkan-parity-matrix-gate`
  in
  [`.github/workflows/tests-and-quality-gates.yml`](.github/workflows/tests-and-quality-gates.yml)
  runs the gate over CPU ↔ Vulkan/lavapipe on every PR (no GPU
  hardware needed); CUDA / SYCL / hardware-Vulkan are advisory
  until a self-hosted runner is registered. New user-facing doc
  at [`docs/development/cross-backend-gate.md`](docs/development/cross-backend-gate.md);
  `docs/backends/index.md` cross-references it. Generalises and
  is the long-term replacement for the per-feature
  `cross_backend_vif_diff.py` lane (kept for one release cycle).
  See [ADR-0214](docs/adr/0214-gpu-parity-ci-gate.md).
- **FastDVDnet temporal pre-filter (T6-7)** — new feature
  extractor `fastdvdnet_pre` under
  [`libvmaf/src/feature/fastdvdnet_pre.c`](libvmaf/src/feature/fastdvdnet_pre.c)
  registers a 5-frame-sliding-window temporal denoiser backed by
  the public `vmaf_dnn_session_*` API. ONNX I/O contract:
  `frames` float32 NCHW `[1, 5, H, W]` (channel axis stacks
  `[t-2, t-1, t, t+1, t+2]`) → `denoised` float32 NCHW
  `[1, 1, H, W]`. Internal 5-slot ring buffer with replicate-edge
  clamp at clip start/end; per-frame scalar
  `fastdvdnet_pre_l1_residual` appended through the existing
  feature-collector plumbing. Picks up `model_path` from the
  feature option or `VMAF_FASTDVDNET_PRE_MODEL_PATH` env var
  (mirrors LPIPS); declines cleanly with `-EINVAL` when neither
  is set. **Placeholder weights only** —
  `model/tiny/fastdvdnet_pre.onnx` is a smoke-only ~6 KB
  randomly-initialised 3-layer CNN with the right I/O shape;
  real upstream-derived FastDVDnet weights + the FFmpeg
  `vmaf_pre_temporal` filter that consumes the denoised frame
  buffer are tracked as **T6-7b**. New ADR
  [ADR-0215](docs/adr/0215-fastdvdnet-pre-filter.md), user-facing
  doc [`docs/ai/models/fastdvdnet_pre.md`](docs/ai/models/fastdvdnet_pre.md),
  registration smoke test
  [`libvmaf/test/test_fastdvdnet_pre.c`](libvmaf/test/test_fastdvdnet_pre.c)
  mirroring `test_lpips.c`. Closes backlog item T6-7.
- **Vulkan PSNR — chroma extension (T3-15(b))** — `psnr_vulkan`
  now emits `psnr_cb` and `psnr_cr` alongside `psnr_y`. Three
  back-to-back dispatches of the existing plane-agnostic
  `psnr.comp` shader against per-plane SSBOs and per-plane
  `(width, height, num_workgroups_x)` push constants; YUV400
  clamps to luma-only at runtime. Cross-backend gate
  (`scripts/ci/cross_backend_vif_diff.py --feature psnr`)
  extended to assert all three plane scores at `places=4`;
  measured `max_abs_diff = 0.0` across 48 frames at 576×324 on
  lavapipe (deterministic int64 SSE accumulators on both sides).
  See [ADR-0216](docs/adr/0216-vulkan-chroma-psnr.md). Doc at
  [`docs/backends/vulkan/overview.md`](docs/backends/vulkan/overview.md).

- **Embedded MCP server scaffold (T5-2, audit-first)** — new
  public header
  [`libvmaf/include/libvmaf/libvmaf_mcp.h`](libvmaf/include/libvmaf/libvmaf_mcp.h)
  declaring the in-process MCP API (`vmaf_mcp_init` /
  `_start_sse` / `_start_uds` / `_start_stdio` / `_stop` /
  `_close` / `_available` / `_transport_available`); stub TU at
  `libvmaf/src/mcp/mcp.c` returning `-ENOSYS` (or `-EINVAL` on
  bad arguments); new umbrella `enable_mcp` boolean (default
  `false`) plus per-transport sub-flags `enable_mcp_sse`,
  `enable_mcp_uds`, `enable_mcp_stdio`; 12-sub-test smoke at
  `libvmaf/test/test_mcp_smoke.c` pinning the `-ENOSYS` +
  NULL-guard contract; user-facing doc at
  [`docs/mcp/embedded.md`](docs/mcp/embedded.md). **Scaffold
  only** — the T5-2b follow-up PR vendors cJSON + mongoose,
  spawns the dedicated MCP pthread + SPSC ring buffer, and
  fills in the SSE / UDS / stdio transport bodies. Same
  audit-first shape as ADR-0175 (Vulkan T5-1) and ADR-0184
  (T7-29 part 1). See
  [ADR-0209](docs/adr/0209-mcp-embedded-scaffold.md) +
  [ADR-0128](docs/adr/0128-embedded-mcp-in-libvmaf.md) +
  [Research-0005](docs/research/0005-embedded-mcp-transport.md).
- **`cambi_vulkan` feature extractor (T7-36 / ADR-0210)** — closes
  the GPU long-tail matrix terminus. Strategy II hybrid: GPU
  shaders run preprocess, per-pixel derivative, 7×7 spatial-mask
  SAT, 2× decimate, 3-tap separable mode filter; the
  precision-sensitive sliding-histogram `calculate_c_values` + top-K
  pool stay on the host. Bit-exact w.r.t. CPU by construction;
  cross-backend gate runs at `places=4`. New ADR
  [`docs/adr/0210-cambi-vulkan-integration.md`](docs/adr/0210-cambi-vulkan-integration.md)
  + research digest
  [`docs/research/0032-cambi-vulkan-integration.md`](docs/research/0032-cambi-vulkan-integration.md).
- **T6-9: Tiny-model registry schema + Sigstore `--tiny-model-verify`**
  ([ADR-0211](docs/adr/0211-model-registry-sigstore.md)). Formal
  JSON Schema (Draft 2020-12) at
  [`model/tiny/registry.schema.json`](model/tiny/registry.schema.json)
  extended with `license`, `license_url`, and `sigstore_bundle`
  fields per entry; `schema_version` bumped to `1`. New CLI flag
  `--tiny-model-verify` wires `cosign verify-blob` via
  `posix_spawnp(3p)` against the registry's `sigstore_bundle` path,
  failing closed on missing bundle / missing cosign / non-zero exit.
  Public C entry point: `vmaf_dnn_verify_signature()` in
  [`libvmaf/include/libvmaf/dnn.h`](libvmaf/include/libvmaf/dnn.h).
  Python validator at
  [`ai/scripts/validate_model_registry.py`](ai/scripts/validate_model_registry.py)
  (Draft 2020-12 with a structural fallback when `jsonschema` is
  absent) covers schema + cross-file consistency and is a pre-push
  gate. Documentation: new
  [`docs/ai/model-registry.md`](docs/ai/model-registry.md), updated
  [`docs/ai/inference.md`](docs/ai/inference.md) and
  [`docs/ai/security.md`](docs/ai/security.md). Tests:
  `python/test/model_registry_schema_test.py` (10 cases) and
  `libvmaf/test/dnn/test_tiny_model_verify.c` (18 failure-mode
  cases on Unix + 1 NULL-arg case covering malformed JSON,
  default-registry
  derivation, fake-cosign success / non-zero exit, and
  empty / missing PATH branches — drives `model_loader.c`
  coverage to ≥85% per the Coverage Gate critical-file rule;
  ENOSYS smoke on Windows). All five shipped models gain
  license metadata (BSD-3-Clause-Plus-Patent for fork-trained;
  BSD-2-Clause for the upstream LPIPS-Sq export).

### Removed

- **`VMAF_MAX_MODEL_BYTES` env override retired (T7-12)**: the
  historical environment-variable knob that let callers raise (or
  lower, for tests) the tiny-AI ONNX file-size cap has been removed
  from `vmaf_dnn_session_open()` and `vmaf_use_tiny_model()`. Two
  release cycles passed without a shipped model approaching the cap,
  so the testing-hatch is retired in favour of the compile-time
  constant `VMAF_DNN_DEFAULT_MAX_BYTES` (50 MB) as the single source
  of truth. Callers that genuinely need a larger envelope must bump
  the constant in
  [`libvmaf/src/dnn/model_loader.h`](libvmaf/src/dnn/model_loader.h)
  and rebuild. The two env-driven unit tests
  (`test_session_open_respects_max_bytes_env`,
  `test_session_open_ignores_invalid_max_bytes_env`) are removed; all
  other size-cap coverage (oversize fixture rejection, `S_ISREG`
  check, allowlist) stays intact.

### Investigated (deferred)

- **VIF AVX-512 `vpgatherdq` polynomial-replacement attempt — bit-exact
  contract forbids.** The post-merge CPU profile (Research-0053, PR #333)
  attributed 13% of VIF AVX-512 samples to the first `vpgatherdq` log2
  lookup in `vif_statistic_8_avx512`, and proposed swapping the gather
  for an inline 5th-order minimax polynomial (estimated +3–6%
  end-to-end). An empirical bit-equivalence test
  ([`scripts/dev/vif_log2_poly_check.py`](scripts/dev/vif_log2_poly_check.py))
  shows a degree-5 polynomial diverges from the scalar
  `log2_table[i] = round(log2f((float)i) * 2048)` at 428 of 32,768
  inputs by ±1 quantum; even degree 8 misses 2 entries. Bit-equivalence
  first reaches zero mismatches at degree 10, where the FMA cost
  cancels the gather win on Skylake-X / Ice Lake. The optimisation
  cannot ship without violating the SIMD bit-exactness precedent set
  by ADR-0138 / ADR-0145 and would shift Netflix golden scores. Full
  analysis + follow-up paths (smaller L1d-resident table + linear
  interpolation; revisit on Sapphire Rapids / Granite Rapids where
  gather throughput is materially better) in
  [`docs/research/0055-vif-polynomial-log2-attempt.md`](docs/research/0055-vif-polynomial-log2-attempt.md).

### Changed

- **Quarterly upstream-backlog re-audit (T7-4)** (fork-local doc):
  new
  [`docs/upstream-backlog-audit-2026-04-29.md`](docs/upstream-backlog-audit-2026-04-29.md)
  walks the 12 upstream Netflix/vmaf commits landed since the
  fork's last `chore(upstream): port` boundary
  (`798409e3` / `314db130`, PR #181). 8 are already on fork
  (cherry-picked, ported, or covered by an Accepted ADR /
  Research digest); 4 are flagged for fork action and surface
  as 4 recommended new T-rows: port `feature/speed`
  (`d3647c73`), port adm + vif test deltas from `c70debb1`,
  port 32-bit ADM/cpu fallbacks (`8a289703` + `1b6c3886`), and
  schedule the next re-audit for 2026-07-29. No code changes,
  no `docs/state.md` changes (no upstream commit ruled in/out
  a fork bug). Doubles as the ADR-0108 research digest for the
  audit PR.

- **T7-32: backlog-hygiene S-task bundle (3 micro-investigations).**
  One PR closes three S-effort follow-ups identified by the
  2026-04-28 BACKLOG audit. (a) `motion_v2` AVX2 `srlv_epi64`
  audit: new fork-local libvmaf C unit test
  [`libvmaf/test/test_motion_v2_simd.c`](libvmaf/test/test_motion_v2_simd.c)
  exercises adversarial negative-`accum` 16-bit fixtures (10-bit
  and 12-bit, uniform-negative and alternating-mixed-sign) against
  the AVX2 path in
  [`libvmaf/src/feature/x86/motion_v2_avx2.c`](libvmaf/src/feature/x86/motion_v2_avx2.c);
  on the bench host the post-`abs()` aggregation absorbs the
  per-lane logical-vs-arithmetic shift difference and SAD totals
  match scalar — the test stays as a permanent regression guard
  and the
  [`docs/rebase-notes.md`](docs/rebase-notes.md) §0038 placeholder
  follow-up is closed.  (b)
  [`docs/research/0006-tinyai-ptq-accuracy-targets.md`](docs/research/0006-tinyai-ptq-accuracy-targets.md)
  §4 now references the actual shipped `vmaf_tiny_v1_medium.onnx`
  checkpoint (landed by [PR
  #158](https://github.com/lusoris/vmaf/pull/158)) instead of the
  fictional `tiny-vmaf-v2` prototype name; the digest's QAT
  cost/budget framing is unchanged.  (c)
  [`python/vmaf/routine.py`](python/vmaf/routine.py) — both
  `cv_on_dataset` (line ~937) and `explain_model_on_dataset` (line
  ~1109) now mirror `VmafQualityRunner`'s contract: cv reads
  `feature_param.feature_optional_dict` when the param exposes it;
  explain reads `model.model_dict["feature_opts_dicts"]` from the
  serialised model. The two `# FIXME: as set to None, potential bug
  with inconsistent behavior with VmafQualityRunner` comments are
  removed. New regression test
  [`python/test/routine_feature_option_dict_test.py`](python/test/routine_feature_option_dict_test.py)
  covers both `None` and populated dict paths via
  `FeatureAssembler` mock. No behaviour change for callers that did
  not declare per-extractor options.

- **Research-0031: Intel AI-PC NPU/EP applicability digest (T7-9)**
  (fork-local doc): new
  [`docs/research/0031-intel-ai-pc-applicability.md`](docs/research/0031-intel-ai-pc-applicability.md)
  evaluates whether the tiny-AI surface should add first-class
  support for the NPU on Intel Meteor / Lunar / Arrow Lake AI-PC
  platforms. Verdict: **defer the NPU path** until a maintainer
  has hardware to validate int8 + fp16 accuracy gates against
  Research-0006's PTQ pipeline. The integrated Xe / Xe2 GPU
  portion of an AI-PC platform is already reachable today through
  the existing `--tiny-device openvino` path (same code path the
  Arc A380 uses), so the iGPU surface costs the fork zero
  additional code; only the NPU device type is genuinely new
  surface and is the part that's deferred. One forward-pointer
  added to [`docs/ai/inference.md`](docs/ai/inference.md) so
  readers of the EP matrix find the digest. No code change.

- **Research-0024 + AGENTS.md: deliberately diverge from Netflix
  upstream `vif` + `float_adm` option-port chains** (fork-local
  doc): new
  [`docs/research/0024-vif-upstream-divergence.md`](docs/research/0024-vif-upstream-divergence.md)
  is a 5-strategy decision matrix on whether to port the Netflix
  upstream vif chain (`4ad6e0ea` runtime helpers / `8c645ce3`
  prescale options / `41d42c9e` edge-mirror bugfix) and the
  float_adm chain (`4dcc2f7c` 12-parameter `compute_adm`
  signature change + new `score_aim` output). Verdict:
  **Strategy E (skip + document)** for both vif and float_adm
  because (a) the fork's `vif_filter1d_table_s` precomputed
  Gaussian table preserves the ADR-0138/0139/0142/0143 SIMD
  bit-exactness contract that runtime-computed Gaussians would
  break, and (b) threading 12 new ADM parameters through the
  SIMD paths + 3 GPU backends is multi-day work without a
  concrete user demand for the new `aim` feature. **Strategy A
  (verbatim)** stays approved for the motion chain
  (`b949cebf` float_motion-only side) because float_motion has
  no precomputed-table investment to protect. Two new invariants
  added to
  [`libvmaf/src/feature/AGENTS.md`](libvmaf/src/feature/AGENTS.md)
  documenting the vif and adm divergences so future sessions
  don't accidentally re-port the chains. No code change.

- **`docs/benchmarks.md` `TBD` cells filled with measured numbers
  (T7-37)**: first end-to-end fork-bench rerun after the bench-script
  fixes (PR #169 / #170 / #171) and the Vulkan header install (PR
  #175). New per-backend tables for the Netflix 576×324 normal pair
  (CPU 598 fps, CUDA 278 fps, SYCL Arc-A380 315 fps, Vulkan 171 fps),
  the 1920×1080 5-frame pair, and the BBB 4K 200-frame pair (CUDA
  227 fps = 16.4× CPU). CPU SIMD-ISA breakdown shows AVX-512 buys
  6.62× over scalar on Zen 5; AVX2 alone gets 2.96×. `--precision`
  overhead measurement confirms `=max` (`%.17g`) is wall-time-free
  (<1 % delta) but +25.8 % JSON byte-count vs the `%.6f` default.
  Hardware-profile table updated to match the actual bench host
  (`ryzen-4090-arc`: Ryzen 9 9950X3D + RTX 4090 + Arc A380, Linux
  7.0.x CachyOS). Each backend's `frames[0].metrics` key count was
  verified per-row (CPU=15, CUDA=12, SYCL/Vulkan=34) to confirm no
  silent CPU fallback.
- **Tiny-AI PTQ accuracy across Execution Providers measured (T5-3e,
  retires the deferred GPU-EP open question in
  `docs/research/0006-tinyai-ptq-accuracy-targets.md`)**: empirical
  PLCC-drop sweep across `CPUExecutionProvider`,
  `CUDAExecutionProvider` (RTX 4090), and the OpenVINO runtime on
  Intel Arc A380 plus the OpenVINO CPU plugin. CPU EP and CUDA EP
  agree to 6 decimal places on every shipped tiny model
  (`learned_filter_v1`, `vmaf_tiny_v1`, `vmaf_tiny_v1_medium` —
  PLCC drop ≤ 1.2×10⁻⁴, well under the 1×10⁻² registry budget).
  OpenVINO CPU plugin agrees to ~10⁻⁴. Intel Arc through OpenVINO
  2026.1 is currently int8-broken: `Conv`-based int8 graphs fail to
  compile (`No layout format available for convolution: byxf /
  i32`); MLP int8 graphs (`MatMulInteger` + `DynamicQuantizeLinear`)
  compile but emit `inf`/`NaN`. Arc fp32 path is healthy. New
  harness `ai/scripts/measure_quant_drop_per_ep.py` + user doc
  `docs/ai/quant-eps.md` document the reproduction recipe and the
  CUDA-12-ABI `LD_LIBRARY_PATH` shim required on CUDA-13 hosts.
- **Backlog: 9 promote-to-T-NN rows landed from the 2026-04-28
  Section-A audit** (fork-local docs): converts the 12 untracked
  follow-up items captured in
  [`docs/backlog-audit-2026-04-28.md`](docs/backlog-audit-2026-04-28.md)
  §A and the user direction frozen in
  [`.workingdir2/decisions/section-a-decisions-2026-04-28.md`](.workingdir2/decisions/section-a-decisions-2026-04-28.md)
  into actual `.workingdir2/BACKLOG.md` rows. New rows: **T3-17**
  (motion3 GPU coverage on Vulkan + CUDA + SYCL), **T3-18** (GPU
  chroma upload + chroma metrics on Vulkan + CUDA), **T5-3e** (PTQ
  accuracy investigation on CUDA + Intel Arc — the deferral framing
  in `docs/research/0006-tinyai-ptq-accuracy-targets.md` is now
  superseded), **T5-4** (Quantization-Aware Training: implement, do
  not close — `ai/scripts/qat_train.py` remains scaffold until this
  ships), **T7-35** (`enable_lcs` MS-SSIM extra metrics on CUDA +
  Vulkan), **T7-36** (cambi GPU integration PR — replaces ADR-0205
  spike scaffolds with a real lifecycle), **T7-37** (run Netflix
  bench + replace `TBD` cells in `docs/benchmarks.md`), **T7-38**
  (SVE2 SIMD parity for SSIMULACRA 2 PTLR + IIR-blur via
  `qemu-aarch64-static` — no CI hardware required). T6-1a row
  extended to fold in §A.2.2 (DMOS-aligned bisect-cache fixture
  rides along once Netflix Public Dataset access lands). §A.1.2
  (cambi v2 c-values strategy-III) intentionally not opened — gated
  on T7-36 landing first, per the audit decisions doc. §A.4.1
  upstream `libvmaf.c` FIXMEs intentionally not opened — rebase-
  fidelity carve-out applies; first PR that touches the file sweeps
  them per CLAUDE.md §12 r12. ADR-0205 + Research-0020 +
  Research-0006 cross-link the new T-numbers in their respective
  decision / follow-up sections.

### Added

- **HIP (AMD ROCm) compute backend — scaffold-only audit-first PR
  (T7-10, ADR-0212)**: new public header
  [`libvmaf/include/libvmaf/libvmaf_hip.h`](libvmaf/include/libvmaf/libvmaf_hip.h)
  declaring `VmafHipState`, `VmafHipConfiguration`,
  `vmaf_hip_state_init` / `_import_state` / `_state_free`,
  `vmaf_hip_list_devices`, `vmaf_hip_available`. New
  `libvmaf/src/hip/` (`common.{c,h}`, `picture_hip.{c,h}`,
  `dispatch_strategy.{c,h}`) + `libvmaf/src/feature/hip/` (3 kernel
  stubs: `adm_hip.c`, `vif_hip.c`, `motion_hip.c`). All entry points
  return `-ENOSYS` until the runtime PR (T7-10b) lands. New
  `enable_hip` boolean option (default **false**) in
  [`libvmaf/meson_options.txt`](libvmaf/meson_options.txt) with
  conditional `subdir('hip')` in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build). New 9-sub-test
  smoke at
  [`libvmaf/test/test_hip_smoke.c`](libvmaf/test/test_hip_smoke.c)
  pinning the `-ENOSYS` / `-EINVAL` contract for every public
  C-API entry point. New CI matrix row `Build — Ubuntu HIP (T7-10
  scaffold)` in
  [`.github/workflows/libvmaf-build-matrix.yml`](.github/workflows/libvmaf-build-matrix.yml)
  compiling with `-Denable_hip=true` (no ROCm SDK on the runner —
  the scaffold has no SDK requirement). New
  [`docs/backends/hip/overview.md`](docs/backends/hip/overview.md);
  [`docs/backends/index.md`](docs/backends/index.md) flipped from
  "planned" to "scaffold only". New
  [`docs/research/0033-hip-applicability.md`](docs/research/0033-hip-applicability.md)
  digest covering AMD market share + ROCm 6.x Linux maturity.
  Mirrors the Vulkan T5-1 scaffold (ADR-0175); validates the
  abstraction-layer-clean-enough-to-reproduce gating condition for
  T7-10. **Zero hard runtime dependencies** —
  `dependency('hip-lang', required: false)` is silently absent on
  stock Ubuntu runners.

- **SSIMULACRA 2 SVE2 SIMD parity (T7-38, ADR-0213)** (fork-local):
  new aarch64 SVE2 sister TU
  ([`libvmaf/src/feature/arm64/ssimulacra2_sve2.c`](libvmaf/src/feature/arm64/ssimulacra2_sve2.c))
  ports the seven SSIMULACRA 2 SIMD entry points (`multiply_3plane`,
  `linear_rgb_to_xyb`, `downsample_2x2`, `ssim_map`, `edge_diff_map`,
  `blur_plane`, `picture_to_linear_rgb`) under a fixed 4-lane
  `svwhilelt_b32(0, 4)` predicate — bit-identical to the NEON sibling
  irrespective of the runtime vector length, satisfying the
  [ADR-0138](docs/adr/0138-simd-bit-exactness-policy.md) /
  [ADR-0139](docs/adr/0139-ssim-simd-bitexact-double.md) /
  [ADR-0140](docs/adr/0140-ssimulacra2-simd-bitexact.md) byte-exact
  contract. New runtime probe
  [`libvmaf/src/arm/cpu.c`](libvmaf/src/arm/cpu.c) reads
  `getauxval(AT_HWCAP2) & HWCAP2_SVE2`; new build probe in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build) runs
  `cc.compiles(... -march=armv9-a+sve2)` so toolchains without SVE2
  intrinsics gracefully fall back to NEON. The dispatch table in
  [`libvmaf/src/feature/ssimulacra2.c`](libvmaf/src/feature/ssimulacra2.c)
  is purely additive: NEON stays the fallback; SVE2 overrides only
  when the bit is set. Validated under `qemu-aarch64-static -cpu max`:
  dispatch reports `NEON=1 SVE2=1`, all 11 `test_ssimulacra2_simd`
  bit-exactness subtests pass byte-for-byte against the scalar
  reference (37/37 host x86 + 36/36 cross-aarch64 SVE2 suites green).
  Closes the "SVE2 deferred pending CI hardware" footnote in
  [Research-0016](docs/research/0016-ssimulacra2-iir-blur-simd.md) /
  [Research-0017](docs/research/0017-ssimulacra2-ptlr-simd.md) and
  backlog row T7-38.
- **Research-0030 — Phase-3b multi-seed validation (Gate 1 PASSED)**
  (fork-local doc): 5-seed retry of Phase-3b confirms the Subset B
  win is robust and *widens* with more seeds. Aggregate over
  5 seeds × 9 LOSO folds: canonical-6 mean PLCC 0.9633 (seed-mean-std
  0.0150) vs **Subset B mean PLCC 0.9807 (seed-mean-std 0.0019)** —
  **Δ = +0.0175**, 3.5× the Research-0027 stopping-rule threshold of
  +0.005. Subset B is also **8× more stable across seeds** than
  canonical-6 — likely because the consensus-7 feature set carries
  overlapping-but-not-identical signal, acting as an in-network
  regularizer. canonical-6's seed=4 ran to PLCC 0.9381 (3.6 pp below
  best seed); Subset B never strays from `[0.9783, 0.9833]`. **Gate 1
  cleanly passed**; Subset B advances to Gate 2 (KoNViD cross-corpus,
  ~3h extraction) and Gate 3 (Phase-3c lr-sweep on canonical-6).
  9.2σ-equivalent margin on the seed-only std means the headline win
  isn't seed-luck. New `--seeds 0,1,2,3,4` flag on
  [`ai/scripts/phase3_subset_sweep.py`](ai/scripts/phase3_subset_sweep.py)
  with seed-mean-std reporting in the summary.

- **Research-0029 — Phase-3b: StandardScaler retry validates
  broader-feature hypothesis** (fork-local doc): empirical retry
  of the Research-0028 negative result with per-fold StandardScaler.
  **Subset B (consensus-7 with redundancy pruning) clears the
  Research-0027 +0.005 PLCC stopping rule by 2× (+0.0106).** Mean
  LOSO PLCC over 9 folds: canonical-6 = 0.9677, Subset A
  (canonical+ssimulacra2) = 0.9669, **Subset B = 0.9783**, Subset
  C (full-21) = 0.9597. The Phase-3a failure was a preprocessing
  artefact, not a feature-signal artefact — `psnr_*`/`cambi`/
  `ciede2000` (range 0–100) had been dominating gradient updates
  over normalised features (range 0–1). With per-fold
  `(mean, std)` standardisation (statistics fit on train, applied
  to both train and val so no fold-leakage), the Research-0026
  hypothesis is confirmed. Two findings: (1) Subset B's feature
  composition (`adm2`, `adm_scale3`, `vif_scale2`, `motion2`,
  `ssimulacra2`, `psnr_hvs`, `float_ssim`) validates all four
  Research-0027 consensus features and the redundancy pruning
  recommendations; (2) Subset C (full-21) loses even with
  StandardScaler — including all features without pruning hurts
  the tiny `mlp_small` architecture. Three gates before
  `vmaf_tiny_v2.onnx` ships: multi-seed validation
  (`seed ∈ {0..4}`), KoNViD cross-corpus check, and Phase-3c
  `lr`-sweep on canonical-6 to verify the +0.0106 holds under
  matched preprocessing. New `--standardize` flag on
  [`ai/scripts/phase3_subset_sweep.py`](ai/scripts/phase3_subset_sweep.py).
  Driver shared with PR #188 (Research-0028); this PR adds the
  flag + the retry results.

- **Research-0028 — Phase-3 MLP subset sweep (negative-result
  digest)** (fork-local doc): empirical close of Research-0026
  Phase 3. The pre-registered Research-0027 stopping rule fires —
  Subset A (canonical-6 + ssimulacra2) lands LOSO mean PLCC 0.9655
  vs canonical-6 0.9845, a 0.019 *deficit* against the required
  +0.005 to advance. Subsets B (consensus-7) and C (full-21) also
  fail PLCC. **canonical-6 stays the default; no v2 model ships
  from this Phase.** Counterintuitive secondary finding: every
  subset cuts mean RMSE by ~40 % (canonical-6 RMSE 15.20 → A 9.13
  / B 8.91 / C 8.50), strongly suggesting the PLCC drop is a
  feature-scale-variance artefact (raw features fed to mlp_small
  without StandardScaler; psnr / cambi / ciede2000 dominate
  gradient updates by 2 orders of magnitude). Three follow-up
  experiments scoped: Phase-3b (StandardScaler retry), Phase-3c
  (`mlp_medium` / wider epoch sweep), Phase-3d (per-feature
  ablation in Subset C). New driver
  [`ai/scripts/phase3_subset_sweep.py`](ai/scripts/phase3_subset_sweep.py)
  ships with the Phase-3b/c/d follow-ups in mind. No code change
  to the trainer or sidecar; pure results document.

- **Research-0027 — Phase-2 feature correlation, MI, and importance
  results** (fork-local doc): empirical close of Research-0026
  Phase 2 on the full Netflix corpus (11 040 frame rows × 21 features
  extracted via PR #186 over ~118 min wall-clock). **Phase-3 GO
  signal is clear**: consensus top-10 across MI + LASSO + random-forest
  importance methods narrows to **4 features** (`adm2`, `adm_scale3`,
  `ssimulacra2`, `vif_scale2`) — two of which (`adm_scale3`,
  `ssimulacra2`) are NOT in the canonical `vmaf_v0.6.1` 6-tuple.
  11 redundant pairs at `|r| ≥ 0.95` reveal that the motion family
  is internally redundant (motion2 ↔ motion3 r=0.9926), VIF scales
  1/2/3 are pairwise redundant, and `vif_scale1 ↔ ssimulacra2`
  cross-family redundancy at r=0.9807 is the most surprising
  finding. Three Phase-3 candidate subsets recommended:
  **Subset A** (canonical-6 + ssimulacra2, conservative single-feature
  add); **Subset B** (consensus-7 = canonical core + adm_scale3 +
  ssimulacra2 + psnr_hvs + float_ssim, redundant scales dropped);
  **Subset C** (full-21, sanity ceiling). Stopping rules + per-subset
  Pareto criteria documented. Aligns with Research-0023 §5 (data
  axis) + Research-0025 (data resolved) + Research-0026 (feature axis
  framework) — this digest empirically validates the framework.
  No code change; pure results document.

- **Research-0025 — FoxBird outlier resolved via Netflix + KoNViD-1k
  combined training** (fork-local doc): empirical close of
  Research-0023 §5's open question. The canonical combined-trainer
  run (`mlp_small`, 30 epochs, val=Tennis + 10 % KoNViD-holdout) on
  the union of the Netflix Public 9-source corpus (9 690 frames) and
  the KoNViD-1k 1 200-clip parquet (270 051 frames) produces an
  ONNX whose FoxBird PLCC is **0.9936** (vs Netflix-only mlp_small
  baseline `vmaf_tiny_v1.onnx` at 0.9632) — a +3.04 percentage-point
  absolute gain on the canonical outlier. RMSE on FoxBird drops
  17.296 → 3.216 (5.4× lower); SROCC +0.0233. No regression on the
  Netflix-native sources (PLCC ≥ 0.998 on 7/9 clips). Validates
  PR #178 (KoNViD acquisition + loader) + PR #180 (combined trainer
  driver) infrastructure end-to-end. Closes Research-0023 §5
  unblocker question: KoNViD-1k is sufficient — no need to acquire
  BVI-DVC or AOM-CTC for this specific failure mode. Full numbers
  + caveats + next-experiment list in
  [`docs/research/0025-foxbird-resolved-via-konvid.md`](docs/research/0025-foxbird-resolved-via-konvid.md).
  No code change in this PR; docs-only.

- **Tiny-AI combined Netflix + KoNViD-1k trainer driver** (fork-local):
  new [`ai/train/train_combined.py`](ai/train/train_combined.py) feeds
  the union of `NetflixFrameDataset` (Netflix Public 9-source corpus)
  and `KoNViDPairDataset` (KoNViD-1k synthetic-distortion FR pairs)
  into the same `_build_model` + `_train_loop` + `export_onnx`
  pipeline that `ai/train/train.py` uses, so model factory + ONNX
  layout stay identical to the canonical baselines. Five validation
  modes: `netflix-source` (default; mirrors ADR-0203), `konvid-holdout`
  (deterministic 10 % of KoNViD clip keys, whole-clip granularity so
  no frame leakage), `netflix-source-and-konvid-holdout` (union of
  both), and the single-corpus `netflix-only` / `konvid-only`
  fallbacks. Addresses Research-0023 §5 (FoxBird-class outlier needs a
  broader content distribution; KoNViD-1k adds 1 200 UGC clips on top
  of the existing 70 Netflix dis-pairs). Documented in
  [`docs/ai/training.md`](docs/ai/training.md) "Combining KoNViD with
  the Netflix corpus" subsection. New 5-test smoke under
  [`ai/tests/test_train_combined_smoke.py`](ai/tests/test_train_combined_smoke.py)
  verifies the `--epochs 0` initial-ONNX path, deterministic
  KoNViD key splitter, and missing-data fallbacks without touching
  libvmaf or the real corpus.

- **Tiny-AI KoNViD-1k → VMAF-pair acquisition + loader bridge**
  (fork-local): direct follow-up to Research-0023 §5 (FoxBird-class
  variance needs a larger / more diverse corpus). New
  [`ai/scripts/konvid_to_vmaf_pairs.py`](ai/scripts/konvid_to_vmaf_pairs.py)
  takes raw KoNViD-1k `.mp4` sources from
  `$VMAF_DATA_ROOT/konvid-1k/`, generates a synthetic distorted
  variant per clip via libx264 CRF=35 round-trip (same recipe as
  the Netflix dis-pairs), runs libvmaf on each (ref, dis) pair to
  extract the 6 `vmaf_v0.6.1` model features + per-frame VMAF
  teacher score, and dumps to
  `ai/data/konvid_vmaf_pairs.parquet` (gitignored). Per-clip JSON
  caches under `$VMAF_TINY_AI_CACHE/konvid-1k/<key>.json` make
  re-runs idempotent. Smoke (5 clips) takes ~7 s wall; full
  1 200-clip run ~30 min on the `ryzen-4090` profile. New
  [`ai/train/konvid_pair_dataset.py::KoNViDPairDataset`](ai/train/konvid_pair_dataset.py)
  loader bridge mirrors `NetflixFrameDataset`'s interface
  (`feature_dim=6`, `numpy_arrays() → (X, y)`) so the existing
  LOSO trainer can ingest KoNViD pairs unchanged. `keep_keys`
  filter supports LOSO-style holdouts. 5 pytest cases under
  [`ai/tests/test_konvid_pair_dataset.py`](ai/tests/test_konvid_pair_dataset.py)
  cover shape, holdout filter, missing-column error, empty-after-
  filter, and torch tensor item shape — all green. Documented in
  [`docs/ai/training.md`](docs/ai/training.md) §"C1 (KoNViD-1k
  corpus)". Future work: a driver that concatenates Netflix +
  KoNViD `(X, y)` arrays and runs the LOSO sweep on the union.

- **Tiny-AI LOSO evaluation harness for `mlp_small`** (fork-local):
  new `ai/scripts/eval_loso_mlp_small.py` scores each of the 9
  leave-one-source-out fold checkpoints (`mlp_small_final.onnx`)
  on its own held-out clip, plus the two shipped baselines
  (`vmaf_tiny_v1.onnx`, `vmaf_tiny_v1_medium.onnx`) per-clip and
  on the all-clips concatenation. Reports per-fold +
  mean ± std PLCC / SROCC / RMSE to JSON and Markdown. Documented
  in [`docs/ai/loso-eval.md`](docs/ai/loso-eval.md). Numbers from
  the 2026-04-28 sweep on the Netflix corpus (LOSO mean PLCC
  0.9808 ± 0.0214, SROCC 0.9848 ± 0.0176) are captured in
  [Research Digest 0022](docs/research/0022-loso-mlp-small-results.md).
  Mirrors the per-fold accounting that MCP `compare_models` does
  for a single split, but respects the LOSO split structure
  without requiring 9 separate comparison calls.

- **`ssimulacra2_cuda` + `ssimulacra2_sycl` GPU twins
  (ADR-0206)** (fork-local): closes batch 3 part 7 across all
  three GPU backends. CUDA + SYCL extractors are direct ports of
  the [ADR-0201](docs/adr/0201-ssimulacra2-vulkan-kernel.md)
  Vulkan hybrid host/GPU pipeline — host runs YUV→linear-RGB,
  2×2 pyramid downsample, linear-RGB→XYB, and the per-pixel SSIM
  + EdgeDiff combine in double precision (verbatim ports of
  `ssimulacra2.c`); GPU runs the 3-plane elementwise multiply
  (`ssimulacra2_mul3`) and the separable 3-pole IIR Gaussian
  blur (`ssimulacra2_blur_h` / `ssimulacra2_blur_v`). The CUDA
  IIR fatbin is pinned with `-Xcompiler=-ffp-contract=off
  --fmad=false` via a per-kernel `cuda_cu_extra_flags` map in
  `libvmaf/src/meson.build`; SYCL relies on the existing
  `-fp-model=precise` for the same effect. Empirical: Netflix
  normal pair `max_abs_diff = 1.0e-6` on CUDA, both checkerboard
  pairs **bit-exact** (0.0). New extractor names:
  `ssimulacra2_cuda`, `ssimulacra2_sycl` (pair with
  `--backend cuda` / `--backend sycl` for exclusive GPU
  dispatch). New sources:
  `libvmaf/src/feature/cuda/ssimulacra2_cuda.{c,h}`,
  `libvmaf/src/feature/cuda/ssimulacra2/ssimulacra2_blur.cu`,
  `libvmaf/src/feature/cuda/ssimulacra2/ssimulacra2_mul.cu`,
  `libvmaf/src/feature/sycl/ssimulacra2_sycl.cpp`. With Vulkan
  ([ADR-0201](docs/adr/0201-ssimulacra2-vulkan-kernel.md))
  already in master and float_adm twins
  ([ADR-0202](docs/adr/0202-float-adm-cuda-sycl.md)) merging in
  parallel, batch 3 is now feature-complete on every GPU
  backend.

- **Backlog audit — untracked follow-up items (2026-04-28)**
  (fork-local, doc-only): one-shot audit of in-tree
  TODO / FIXME / "deferred" / "scaffold only" / "v2" mentions
  cross-referenced against the canonical backlog tracking
  surfaces (`.workingdir2/OPEN.md`, `.workingdir2/BACKLOG.md`,
  `docs/state.md`, `docs/rebase-notes.md`, ADR Decision /
  Consequences blocks, open GitHub issues / PRs). Output lands
  at [`docs/backlog-audit-2026-04-28.md`](docs/backlog-audit-2026-04-28.md);
  35 distinct clusters across ~1 270 raw hits. Section A lists
  14 untracked items needing decision (cambi v2 GPU c-values
  phase, cambi integration PR, Vulkan motion3 GPU gap,
  `picture_vulkan` luma-only chroma gap, `enable_lcs` GPU stub,
  QAT trainer hook, `docs/benchmarks.md` `TBD` cells, SVE2
  SIMD parity, etc.); Section B lists 5 partially-tracked items
  in ADRs / digests with no T-number (notably
  `iqa_convolve` AVX-512 ADR-0138 follow-up,
  `motion_v2` AVX2 srlv_epi64 audit). Section C lists 4
  resolved-but-stale comments (`libvmaf_vulkan.h` "scaffold
  only", `ssimulacra2.c` "SIMD variants are follow-up PRs",
  `meson.build` Vulkan blurbs) — comment-only fixes for the
  next session that touches each file. No source files
  modified.

- **cambi GPU feasibility spike — hybrid host/GPU verdict + Vulkan
  scaffold (ADR-0205)** (fork-local): closes the spike mandated by
  [ADR-0192](docs/adr/0192-gpu-long-tail-batch-3.md) §Consequences.
  Verdict: cambi is feasible on GPU as a **hybrid host/GPU pipeline**,
  mirroring [ADR-0201](docs/adr/0201-ssimulacra2-vulkan-kernel.md)'s
  precedent. GPU dispatch chain covers preprocessing + per-pixel
  derivative + 7×7 spatial-mask summed-area table + 2× decimate +
  3-tap separable mode filter (all integer + bit-exact w.r.t. CPU);
  the precision-sensitive `calculate_c_values` sliding-histogram
  pass + top-K spatial pooling stay on the host. Because the GPU
  buffers are bit-identical to the CPU's and the c-values phase
  runs the exact CPU code path, the v1 contract tightens to
  **`places=4`** (ADR-0192 carried `places=2` as a planning
  placeholder; ADR-0205 ratchets to the fork's canonical `places=4`
  baseline since the architecture forces ULP=0). Three classical re-formulations
  evaluated in [research digest 0020](docs/research/0020-cambi-gpu-strategies.md):
  (I) single-WG direct port — rejected, ~1/64 GPU utilisation;
  (II) parallel-scan reformulation — rejected for v1, materialises
  17 GiB intermediate at 4K; (III) direct per-pixel histogram —
  deferred to v2 as profile-driven perf polish, ~9× CPU bandwidth.
  Literature surveyed: Blelloch 1990, Sengupta 2007, Merrill &
  Grimshaw 2016. v1 LOC estimate: ~1230 (host glue ~700 + 6 shaders
  ~400 + wiring ~130). This PR ships the architecture sketch
  ([ADR-0205](docs/adr/0205-cambi-gpu-feasibility.md)) + research
  digest + reference shader scaffolds (`cambi_derivative.comp`,
  `cambi_decimate.comp`, `cambi_filter_mode.comp` under
  [`libvmaf/src/feature/vulkan/shaders/`](libvmaf/src/feature/vulkan/shaders/))
  + dormant `cambi_vulkan.c` host skeleton (not yet build-wired,
  matching ssimulacra2 precedent). After the integration follow-up
  PR lands, every registered feature extractor in the fork has at
  least one GPU twin (lpips remains ORT-delegated per
  [ADR-0022](docs/adr/0022-inference-runtime-onnx.md)) and the GPU
  long-tail terminus declared in
  [ADR-0192](docs/adr/0192-gpu-long-tail-batch-3.md) is closed.
- **Tiny-AI Netflix-corpus training prep (ADR-0203)** (fork-local):
  runnable loader + feature extractor + `vmaf_v0.6.1` distillation +
  PyTorch dataset + PLCC/SROCC/KROCC/RMSE eval harness + Lightning-
  style training entry point under `ai/data/` and `ai/train/`.
  Three architectures registered with the entry point: `linear`
  (7 params), `mlp_small` (257 params, default), `mlp_medium`
  (2 561 params). Default validation split holds out the
  `Tennis_24fps` source (1-source-out, content-disjoint). Per-clip
  JSON cache at `$VMAF_TINY_AI_CACHE` (default
  `~/.cache/vmaf-tiny-ai/<source>/<dis-stem>.json`) with atomic
  write-rename. Smoke command `python ai/train/train.py --epochs 0
  --assume-dims 16x16` works without the real corpus or a built
  `vmaf` binary so CI can verify the harness end-to-end. The first canonical training run on the full Netflix corpus
  (mlp_small, 30 epochs, val=Tennis) is documented in ADR-0203
  §"Training results"; final ONNX shipped at
  `model/tiny/vmaf_tiny_v1.onnx` (PLCC 0.9750 / SROCC 0.9792 vs
  vmaf_v0.6.1 distillation target). New
  [`docs/ai/training.md`](docs/ai/training.md) "C1 (Netflix corpus)"
  section + 25 unit tests under [`ai/tests/`](ai/tests/).
  Files: new
  [`ai/data/netflix_loader.py`](ai/data/netflix_loader.py),
  [`ai/data/feature_extractor.py`](ai/data/feature_extractor.py),
  [`ai/data/scores.py`](ai/data/scores.py),
  [`ai/train/dataset.py`](ai/train/dataset.py),
  [`ai/train/eval.py`](ai/train/eval.py),
  [`ai/train/train.py`](ai/train/train.py),
  [`ai/scripts/run_training.sh`](ai/scripts/run_training.sh).

- **GPU long-tail batch 3 part 2 — `float_ansnr_{vulkan,cuda,sycl}`
  extractors (T7-23 / ADR-0192 / ADR-0194)** (fork-local): closes
  the ANSNR matrix gap (was CPU-only float, no GPU twin). Single-
  dispatch GPU kernels apply the CPU's 3x3 ref filter
  ([`ansnr_tools.c::ansnr_filter2d_ref_s`](libvmaf/src/feature/ansnr_tools.c))
  and 5x5 dis filter (Netflix-tuned weights summing to 1.0,
  `/571`) inline from a 20×20 shared / SLM tile, then accumulate
  per-pixel `sig = ref_filtr²` and `noise = (ref_filtr - filtd)²`
  into per-WG float partials. Host reduces in `double` and applies
  the CPU formulas for `float_ansnr` and `float_anpsnr`. Edge-
  replicating mirror (`2*size - idx - 1`) matches CPU
  `ansnr_filter2d_s` — same divergence-from-motion footgun as
  motion_v2 (ADR-0193). Empirical floor on cross-backend gate
  fixture: `max_abs_diff = 6e-6` (8-bit, 48 frames) / `2e-6`
  (10-bit, 3 frames) on **all three backends with identical
  numbers** (Vulkan = CUDA = SYCL — strong evidence the kernel
  logic is correct). Files: new
  [`shaders/float_ansnr.comp`](libvmaf/src/feature/vulkan/shaders/float_ansnr.comp),
  [`float_ansnr_vulkan.c`](libvmaf/src/feature/vulkan/float_ansnr_vulkan.c),
  [`float_ansnr/float_ansnr_score.cu`](libvmaf/src/feature/cuda/float_ansnr/float_ansnr_score.cu),
  [`float_ansnr_cuda.c`](libvmaf/src/feature/cuda/float_ansnr_cuda.c),
  [`float_ansnr_sycl.cpp`](libvmaf/src/feature/sycl/float_ansnr_sycl.cpp).
  New `float_ansnr` lavapipe gate step in
  [`tests-and-quality-gates.yml`](.github/workflows/tests-and-quality-gates.yml)
  + `FEATURE_METRICS` entry in the cross-backend gate.

### Changed

- **Port Netflix upstream `314db130` — remove empty translation unit
  `libvmaf/src/feature/all.c`** (upstream port): the file had been
  reduced to includes + forward declarations + a `MIN` macro with no
  active call sites in the fork (`compute_*` entry points are reached
  via per-extractor TUs, not via `all.c`). Upstream removed it as
  dead code. Drops the file, the `meson.build` line that compiled it,
  and updates the trailing `// NOLINTNEXTLINE` comment in
  `offset.c:22` that listed `all.c` among the per-feature consumers.
  Build + 37 unit tests green after removal.

- **Audit Section C cleanup — refresh stale "scaffold only" / "follow-up
  PR" comments** (fork-local): four code surfaces still advertised work
  that has long since landed. Updated `libvmaf_vulkan.h` (top-level
  header doc-comment + the T7-29 zero-copy import block), the
  `ssimulacra2.c` SIMD blurb (ADR-0161 / 0162 / 0164 + GPU twins
  ADR-0201 / 0206), and the Vulkan blurbs in `libvmaf/src/meson.build`
  + `libvmaf/meson_options.txt`. Comment-only; no behavioural change.
  Closes Section C of `docs/backlog-audit-2026-04-28.md`.

- **Whole-codebase lint sweep — auto-fix subset (52% findings cleared)**
  (fork-local): post-T5-1 + docs-sweep follow-up. Baseline clang-tidy
  whole-codebase scan flagged 1533 unique findings across 84 files.
  This PR clears the auto-fixable categories —
  `readability-isolate-declaration`,
  `readability-braces-around-statements`, `modernize-use-nullptr`,
  `misc-const-correctness`, and `cert-err33-c` — leaving 736 manual
  / NOLINT-with-justification findings (widening-mul on stride math,
  function-size on SIMD reductions per ADR-0138, use-internal-linkage
  on cross-TU dispatch helpers, anonymous-namespace on C++ helpers,
  mt-unsafe `getenv`/`strerror`, etc.) for follow-up sweeps. Build
  green; all 37 meson unit tests pass after each of the four commits;
  clang-format clean on every touched file. Touched 68 files,
  ~1500-line diff.

- **CPU coverage matrix audit — closes 5 stale gaps in one pass
  (no code changes)** (fork-local): post-T7-19 verification
  exposed five matrix entries and backlog rows that were either
  already-shipped work or phantom rows from earlier audit
  snapshots. **T7-22** (`ms_ssim` per-scale SIMD) was already
  shipped via ADR-0138/0139/0140 — verified 3.2× wall-clock
  speedup vs `--cpumask 0xfffffffe`. **CAMBI scalar fallback**
  already exists at
  [`cambi.c:446-460`](libvmaf/src/feature/cambi.c). **motion_v2
  NEON** already exists at
  [`arm64/motion_v2_neon.c`](libvmaf/src/feature/arm64/motion_v2_neon.c).
  **integer `ansnr`** is a phantom row — no extractor is
  registered. **T7-21** (`psnr_hvs` AVX-512) closes as **AVX2
  ceiling** with empirical evidence (1.17× speedup of AVX2 vs
  scalar; AVX-512 widening would force a 2-block host batch
  without measurable payoff). Same verdict for deferred
  float_moment AVX-512. The CPU SIMD column is now closed. See
  [ADR-0180](docs/adr/0180-cpu-coverage-audit.md). Next gap
  surface: GPU long-tail (psnr / ssim / ssimulacra2 / cambi /
  psnr_hvs on CUDA / SYCL / Vulkan).

### Fixed

- **CI: Clang-Tidy job no longer fails on PRs that delete C/C++ files**
  (fork-local CI fix): `.github/workflows/lint-and-format.yml`'s
  `Clang-Tidy (Changed C/C++ Files)` step used `git diff --name-only`
  without `--diff-filter=d`, so a deleted file (e.g.
  `libvmaf/src/feature/all.c` in this PR's upstream port of
  `314db130`) was passed to `clang-tidy`, which then failed with
  `clang-diagnostic-error: no such file or directory`. Added
  `--diff-filter=d` to all three `git diff` invocations in the
  Clang-Tidy step (PR / push / push-with-fallback). No effect on
  Add/Modify paths.

- **Port Netflix upstream `798409e3` — null-deref on `prev_ref` update
  in pure CUDA pipelines** (upstream port, completes fork's earlier
  partial fix): the fork's `read_pictures_update_prev_ref` helper
  (`libvmaf.c:1593`) already carries the `if (ref && ref->ref)` guard
  for the main `vmaf_read_pictures` path, but the same shape was
  missing from `threaded_enqueue_one` (line 1057) and
  `threaded_read_pictures_batch` (line 1105) — both could deref a
  zero-initialised `ref_host` when every registered extractor carries
  `VMAF_FEATURE_EXTRACTOR_CUDA` and `translate_picture_device`
  early-returns without downloading. Patch mirrors lawrence's
  upstream fix (Netflix/vmaf `798409e3`, 2026-04-20). No behavioural
  change on non-CUDA pipelines; preserves the existing ADR-0123
  null-guard rationale across all three call sites.

- **T7-16: NVIDIA-Vulkan + SYCL `adm_scale2` 2.4e-4 boundary drift
  is gone — verified at `places=4` on master** (fork-local doc
  close, sister of T7-15): the cross-backend gate at PR #120
  surfaced a 2.4e-4 score offset on 1/48 frames for `adm_scale2`
  on Vulkan-on-NVIDIA-RTX-4090 (proprietary driver) and SYCL-on-
  Arc. Re-running on master with the same reproducer
  (`python3 scripts/ci/cross_backend_vif_diff.py --feature adm
  --backend vulkan --device 0`) reports `adm_scale2` max_abs_diff
  = 1e-6 (JSON `%f` print floor; ULP=0) on Vulkan device 0
  (RTX 4090, NVIDIA proprietary 595.58.3.0), Vulkan device 1
  (Arc Mesa anv 26.0.5), AND SYCL device 0 (Arc A380). All three
  pass `places=4` at 0/48 mismatches across all 5 ADM metrics.
  Same NVCC / driver / SYCL-runtime upgrade hypothesis as T7-15
  — no `adm_vulkan.c` / `adm_sycl.cpp` commits since PR #120
  (`7c5b63a2`). Verification-only close; the cross-backend gate
  locks the contract going forward.

- **T7-15: `motion_cuda` + `motion_sycl` 2.6e-3 drift vs CPU
  `integer_motion` is gone — verified bit-exact on master**
  (fork-local doc close, PR #172): the cross-backend gate at PR #120
  surfaced a 2.6e-3 score offset on 47/48 frames for both
  `motion_cuda` and `motion_sycl` on the Netflix golden 576×324
  pair. Re-running on master with the same reproducer
  (`python3 scripts/ci/cross_backend_vif_diff.py --feature motion
  --backend cuda`) reports `max_abs_diff=0.0` over all 48 frames
  at `places=8`; SYCL on Arc and Vulkan on Arc Mesa anv both show
  1e-6 (the JSON `%f` print-rounding floor; ULP=0). All three
  pass the existing `places=4` contract and the cross-backend
  gate now locks it going forward. No motion-kernel commits
  landed between PR #120 (`7c5b63a2`) and master, so the
  resolution is most likely the NVCC 13.x / NVIDIA-driver upgrade
  since PR #120 — the kernel source is unchanged but the emitted
  SASS now matches CPU rounding bit-exactly.

- **`libvmaf_vulkan.h` now installs under the prefix when
  `-Denable_vulkan=enabled`** (fork-local): `libvmaf/include/libvmaf/meson.build`
  had install gates for `is_cuda_enabled` and `is_sycl_enabled` but
  none for Vulkan, so `meson install` dropped `libvmaf_cuda.h` and
  `libvmaf_sycl.h` under `<prefix>/include/libvmaf/` but never
  `libvmaf_vulkan.h`. Symptom (lawrence, 2026-04-28): FFmpeg
  `configure` accepts `--enable-libvmaf-vulkan` and reports it as
  enabled, but only `vmaf_pre` and the regular `libvmaf` filter
  end up built — the `libvmaf_vulkan` filter is silently dropped
  because `check_pkg_config libvmaf_vulkan "libvmaf >= 3.0.0"
  libvmaf/libvmaf_vulkan.h vmaf_vulkan_state_init_external` (in
  ffmpeg-patches/0006) can't find the header. Fix: add an
  `is_vulkan_enabled` gate (handles the `feature` option's
  `enabled` and `auto` states), append `libvmaf_vulkan.h` to
  `platform_specific_headers` when active. Verified: a fresh
  `meson install --destdir /tmp/x` now drops the header alongside
  `libvmaf_cuda.h` and `libvmaf_sycl.h`. No CHANGELOG breakage for
  pre-existing CUDA/SYCL consumers — the install set is purely
  additive.

- **`--backend cuda` actually engages CUDA now (was silently CPU)**
  (fork-local): the CLI's `--backend cuda` selector previously set
  `gpumask = 1` intending it as a device pin, but
  `VmafConfiguration::gpumask` is a CUDA-*disable* bitmask per the
  public-header contract — `compute_fex_flags` enables the CUDA
  dispatch slot only when `gpumask == 0`. Net effect: every
  `--backend cuda` run from CLI initialised CUDA but then routed
  the actual feature extractors through the CPU path, producing
  bit-exact CPU-equivalent pools and no GPU speedup. Symptom on
  bench fixtures: identical fps + identical `vmaf` pool across
  `cpu` / `cuda` / `sycl` rows. Fix in
  [`libvmaf/tools/cli_parse.c`](libvmaf/tools/cli_parse.c) — set
  `gpumask = 0` (default) so the runtime engages CUDA after
  `vmaf_cuda_state_init` succeeds. `--gpumask=N --backend cuda`
  combinations preserve the user-supplied N. 5 new regression tests
  in [`libvmaf/test/test_cli_parse.c`](libvmaf/test/test_cli_parse.c)
  cover the four backends + the explicit-gpumask case. End-to-end
  smoke: `--backend cuda` on the Netflix golden 576×324 pair now
  emits 12 feature keys (CUDA extractor set) instead of 15 (CPU
  extractor set). The legacy
  `--gpumask=0 --no_sycl --no_vulkan` invocation continues to work
  as before. Documented in
  [`libvmaf/AGENTS.md`](libvmaf/AGENTS.md) §"Backend-engagement
  foot-guns" — same surface as PR #169.

- **`libvmaf.pc` Cflags leak in static-archive builds (ADR-0200)**
  (fork-local): bug-fix follow-up to ADR-0198. The
  `-include volk_priv_remap.h` flag was attached to
  `volk_dep.compile_args`; on `default_library=static` builds meson
  copies dependency `compile_args` into the generated `libvmaf.pc`
  `Cflags:` so downstream consumers can re-link against transitive
  static deps. Lawrence's BtbN-style fully-static FFmpeg build
  (cross-toolchain glibc-2.28, 2026-04-27) hit:

  ```text
  <command-line>: fatal error: /<libvmaf-build-dir>/subprojects/
    volk-vulkan-sdk-1.4.341.0/volk_priv_remap.h: No such file or directory
  compilation terminated.
  ```

  on FFmpeg's `check_func_headers aom/aom_codec.h` probe — the
  libvmaf-build-dir absolute path no longer existed after libvmaf
  was installed to `/opt/ffbuild/`. Fix: move the `-include` off
  `volk_dep.compile_args` and onto libvmaf's private `c_args`
  via `vmaf_cflags_common += ['-include', volk_priv_remap_h_path]`
  in `libvmaf/src/vulkan/meson.build`, where the path is pulled
  from `subproject('volk').get_variable('volk_priv_remap_h_path')`.
  `c_args:` on a `library()` are private to the target and do
  NOT leak into the generated pkg-config Cflags; the
  symbol-rename behaviour from ADR-0198 stays byte-for-byte
  identical. Post-fix `pkg-config --cflags libvmaf` returns
  `-I${includedir} -I${includedir}/libvmaf -DVK_NO_PROTOTYPES -pthread`
  on both shared and static builds. `nm libvmaf.a` still reports
  0 GLOBAL `vk*` and 719 `vmaf_priv_vk*`. See
  [ADR-0200](docs/adr/0200-volk-priv-remap-pkgconfig-leak-fix.md).

- **Volk / `vk*` symbol clash in fully-static link environments
  (ADR-0198)** (fork-local): follow-up to ADR-0185.
  `-Wl,--exclude-libs,ALL` only takes effect at the
  `gcc -shared` step that produces `libvmaf.so` — when libvmaf
  is built with `default_library=static -Denable_vulkan=enabled`,
  no link step happens and volk's full `vk*` PFN dispatcher
  table stays as STB_GLOBAL inside `libvmaf.a`. BtbN-style
  fully-static FFmpeg builds (lawrence's repro 2026-04-27 on a
  cross-toolchain glibc-2.28 build) that stitched
  `libvmaf.a + libvulkan.a` into one binary hit ~700 GNU-ld
  multi-definition errors:

  ```text
  ld: volk.c.o (symbol from plugin):
      multiple definition of `vkGetInstanceProcAddr';
      libvulkan.a(loader.c.o): first defined here
  ```

  Fixed by renaming volk's `vk*` symbols to `vmaf_priv_vk*`
  at the C preprocessor level via a force-included header
  generated from `volk.h`. The packagefile parses every
  `extern PFN_vkXxx vkXxx;` declaration, emits
  `#define vkXxx vmaf_priv_vkXxx` (784 entries for volk-1.4.341),
  and `-include`s the result on `volk.c` and every libvmaf TU
  pulling in `volk_dep`. Identical behaviour for shared and
  static — no per-build-mode meson branches. Verified: shared
  `nm -D libvmaf.so` reports 0 leaked `vk*` (unchanged from
  ADR-0185); static `nm libvmaf.a` reports 0 GLOBAL `vk*` (was
  ~700) and 719 `vmaf_priv_vk*`; BtbN-style
  `gcc -static main.c libvmaf.a libvulkan-stub.a` link succeeds;
  `test_vulkan_smoke` 10/10 pass on the renamed build (volk
  runtime `dlsym` dispatch still functional). See
  [ADR-0198](docs/adr/0198-volk-priv-remap-static-archive.md).

- **Hide volk / vk* symbols from libvmaf.so's public ABI
  (T7-31, ADR-0185)** (fork-local): when libvmaf is built with
  `-Denable_vulkan=enabled`, the bundled volk Vulkan-loader
  leaked ~30 `volk*` + the full `vk*` API into the .so's
  exported symbols. Static FFmpeg builds (BtbN-style
  cross-toolchain releases, glibc-2.28 environments, etc.)
  that link **both** libvmaf and libvulkan.a got GNU-ld
  multiple-definition errors at the final link:

  ```text
  /opt/ffbuild/lib/libvulkan.a(loader.c.o):
    multiple definition of `vkGetInstanceProcAddr';
  volk.c.o (symbol from plugin): first defined here
  ```

  Fixed by passing `-Wl,--exclude-libs,ALL` on the libvmaf.so
  link command in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build); gated
  off Darwin / Windows where the flag isn't supported (those
  linkers don't auto-export static-archive symbols anyway).
  Verified via `nm -D libvmaf.so` (zero `vk*` / `volk*` post-
  fix); smoke + end-to-end `psnr_vulkan` on Arc A380 unchanged
  (`psnr_y = 34.760779` matches PR #125's bit-exact reference).
  See [ADR-0185](docs/adr/0185-vulkan-hide-volk-symbols.md).

### Added

- **GPU long-tail batch 3 parts 1b + 1c — `motion_v2_cuda` +
  `motion_v2_sycl` extractors (T7-23 / ADR-0192 / ADR-0193)**
  (fork-local): closes batch 3 part 1 (and the integer_motion family
  GPU coverage). CUDA + SYCL twins of the Vulkan motion_v2 kernel
  shipped in PR #146. Both inherit the per-WG int64 SAD partial
  pattern, the raw-pixel ping-pong, and the edge-replicating mirror
  that diverges by one pixel from the motion kernel's mirror.
  - **CUDA**: nested 5x5 filter on the (prev - cur) diff loaded into
    a 20x20 shared int32 tile, warp-reduced via `__shfl_down_sync`,
    `atomicAdd` to a single int64 device buffer. New
    [`integer_motion_v2/motion_v2_score.cu`](libvmaf/src/feature/cuda/integer_motion_v2/motion_v2_score.cu)
    (~180 LOC PTX) +
    [`integer_motion_v2_cuda.c`](libvmaf/src/feature/cuda/integer_motion_v2_cuda.c)
    (~290 LOC host glue with submit/collect async stream pattern).
    Bit-exact vs CPU on 8-bit (48 frames) and 10-bit (3 frames) —
    `max_abs_diff = 0.0` on RTX 4090.
  - **SYCL**: separable V→H 5-tap filter on a 12x36 SLM tile, sub-
    group `reduce_over_group` then SLM cross-subgroup reduction →
    `atomic_ref::fetch_add` to int64. Self-contained (does NOT
    register with `vmaf_sycl_graph_register` because motion_v2
    needs the previous frame's raw ref pixels which the
    `shared_frame` luma buffer doesn't preserve across calls — same
    pattern as ciede_sycl). New
    [`integer_motion_v2_sycl.cpp`](libvmaf/src/feature/sycl/integer_motion_v2_sycl.cpp)
    (~330 LOC). Bit-exact vs CPU on Intel Arc A380 + oneAPI 2025.3.
- **GPU long-tail batch 3 part 1a — `motion_v2_vulkan` extractor
  (T7-23 / ADR-0192 / ADR-0193)** (fork-local): first kernel of
  batch 3, the smallest fork-local Vulkan kernel by far (~280 LOC
  GLSL + ~360 LOC host glue). Single-dispatch design exploits
  convolution linearity:
  `SAD(blur(prev), blur(cur)) == sum(|blur(prev - cur)|)` so the
  kernel reads both `prev_ref` and `cur_ref` planes, computes the
  full V→H separable 5-tap Gaussian over the signed diff in one
  dispatch, and accumulates `|h|` directly into per-WG `int64`
  partials. No blurred-state buffer (vs `motion_vulkan`'s
  ping-pong) — replaced by a smaller raw-pixel ping-pong of
  `ref_buf[2]`. Bit-exact vs CPU on 8-bit and 10-bit (max_abs_diff
  = 0.0 across 48 frames at 576×324, Intel Arc A380 + Mesa anv);
  cross-backend gate runs at `places=4`. Mirror padding
  **diverges** from `motion.comp` — CPU `integer_motion_v2.c`
  uses edge-replicating reflective mirror (`2*size - idx - 1`)
  while `integer_motion.c::edge_8`/`edge_16` use the non-
  replicating variant (`2*(size-1) - idx`); difference is one
  pixel at the boundary and the GLSL must follow the CPU it's
  porting. CUDA + SYCL twins follow as ADR-0192 batch 3 parts
  1b + 1c. New `motion_v2` lavapipe lane step + `FEATURE_METRICS`
  entry in
  [`scripts/ci/cross_backend_vif_diff.py`](scripts/ci/cross_backend_vif_diff.py).
- **GPU long-tail batch 3 scope (T7-23 / ADR-0192)** (fork-local,
  doc-only PR #145): scoping ADR for batch 3 — closes every
  remaining metric gap on the matrix. Group A (no GPU twin yet):
  `integer_motion_v2`, `float_ansnr`, `ssimulacra2`, `cambi`.
  Group B (float twins of int kernels already on GPU):
  `float_psnr` / `float_motion` / `float_vif` / `float_adm`,
  kept native (not aliased to the int kernels — different input
  domains). 21+ PRs to close (7 metrics × 3 backends). After
  batch 3, every registered feature extractor in the fork has at
  least one GPU twin (`lpips` remains ORT-delegated per ADR-0022).
- **GPU long-tail batch 3 part 3 — `float_psnr_{vulkan,cuda,sycl}`
  extractors (T7-23 / ADR-0192 / ADR-0195)** (fork-local): first
  Group B float twin from ADR-0192. Smallest GPU twin in the
  long-tail (~120 LOC GLSL + ~110 LOC PTX + ~150 LOC SYCL). Single-
  dispatch kernels — no halo, no shared tile — compute per-pixel
  `(ref - dis)²` in float, reduce per-WG via sub-group + SLM, host
  accumulates in `double` and applies CPU formula
  `MIN(10·log10(peak² / max(noise / (w·h), 1e-10)), psnr_max)`.
  **Empirically bit-exact** vs CPU on all three backends, both
  8-bit (48 frames) and 10-bit (3 frames) — `max_abs_diff = 0.0e+00`
  everywhere on Intel Arc A380 (Vulkan + SYCL) and NVIDIA RTX 4090
  (CUDA). Float-domain kernel too simple to drift; host-side
  `double` reduction absorbs any per-WG ULP noise. Drive-by docs
  fix: features.md row claimed `float_psnr_y / _cb / _cr` plane
  outputs which were wrong — the CPU extractor only emits a single
  luma `float_psnr` score; corrected in this PR. New
  [`shaders/float_psnr.comp`](libvmaf/src/feature/vulkan/shaders/float_psnr.comp),
  [`float_psnr_vulkan.c`](libvmaf/src/feature/vulkan/float_psnr_vulkan.c),
  [`float_psnr/float_psnr_score.cu`](libvmaf/src/feature/cuda/float_psnr/float_psnr_score.cu),
  [`float_psnr_cuda.c`](libvmaf/src/feature/cuda/float_psnr_cuda.c),
  [`float_psnr_sycl.cpp`](libvmaf/src/feature/sycl/float_psnr_sycl.cpp).
  New `float_psnr` lavapipe gate step + `FEATURE_METRICS` entry.
- **GPU long-tail batch 3 part 4 — `float_motion_{vulkan,cuda,sycl}`
  extractors (T7-23 / ADR-0192 / ADR-0196)** (fork-local): second
  Group B float twin from ADR-0192. Float-domain twin of
  `integer_motion`'s GPU kernels: same V→H 5-tap separable Gaussian
  blur (FILTER_5_s float weights summing to ~1.0), same 2-buffer
  ping-pong of blurred refs, same per-WG float SAD partials + host
  `double` reduction. `motion = sad / (w·h)`,
  `motion2 = min(prev, cur)` emitted at `index - 1` (delayed-by-one
  pattern, matches CPU `float_motion.c::extract`). Mirror padding:
  skip-boundary `2*(sup-1) - idx` matches CPU
  `convolution_internal.h::convolution_edge_s` (NOT motion_v2's
  edge-replicating). **Identical** `max_abs_diff = 3e-6` (8-bit, 48
  frames) / `1e-6` (10-bit, 3 frames) across all three backends —
  strong correctness signal (any algebraic bug would produce
  backend-specific drift). New
  [`shaders/float_motion.comp`](libvmaf/src/feature/vulkan/shaders/float_motion.comp),
  [`float_motion_vulkan.c`](libvmaf/src/feature/vulkan/float_motion_vulkan.c),
  [`float_motion/float_motion_score.cu`](libvmaf/src/feature/cuda/float_motion/float_motion_score.cu),
  [`float_motion_cuda.{c,h}`](libvmaf/src/feature/cuda/float_motion_cuda.c),
  [`float_motion_sycl.cpp`](libvmaf/src/feature/sycl/float_motion_sycl.cpp).
  New `float_motion` lavapipe gate step + `FEATURE_METRICS` entry.
- **GPU long-tail batch 3 part 5 — `float_vif_{vulkan,cuda,sycl}`
  extractors (T7-23 / ADR-0192 / ADR-0197)** (fork-local): third
  Group B float twin from ADR-0192. 4-scale Gaussian pyramid with
  separable `{17, 9, 5, 3}`-tap filters at the default
  `vif_kernelscale = 1.0` (other kernelscale values rejected at
  init for v1 — production uses 1.0 exclusively). 7 dispatches per
  frame (4 compute + 3 decimate). CPU's `VIF_OPT_HANDLE_BORDERS`
  branch: per-scale dims = prev/2 (no border crop), decimation
  samples at `(2*gx, 2*gy)` with mirror padding handling taps near
  the edge. **Mirror-asymmetry fix:** CPU has two H-mirror formulas
  that differ by 1 —
  [`vif_mirror_tap_h`](libvmaf/src/feature/vif_tools.c) returns
  `2 * extent - idx - 1` (scalar fallback only), while
  [`convolution_edge_s`](libvmaf/src/feature/common/convolution_internal.h)
  returns `2 * width - idx - 2` (AVX2 production border path). The
  GPU follows the AVX2 form because that's what production runs;
  using scalar's form drifted `5.46e-4` at scale 1, the AVX2 form
  closes that to `1.4e-5`. **places=4 across all 4 scales,
  identical numbers across all three backends** (`1e-6 / 1.4e-5
  / 1.8e-5 / 3.7e-5` at 8-bit, tighter at 10-bit on Intel Arc A380,
  Mesa anv, NVIDIA RTX 4090, oneAPI 2025.3). New
  [`shaders/float_vif.comp`](libvmaf/src/feature/vulkan/shaders/float_vif.comp),
  [`float_vif_vulkan.c`](libvmaf/src/feature/vulkan/float_vif_vulkan.c),
  [`float_vif/float_vif_score.cu`](libvmaf/src/feature/cuda/float_vif/float_vif_score.cu),
  [`float_vif_cuda.{c,h}`](libvmaf/src/feature/cuda/float_vif_cuda.c),
  [`float_vif_sycl.cpp`](libvmaf/src/feature/sycl/float_vif_sycl.cpp).
  New `float_vif` lavapipe gate step + `FEATURE_METRICS` entry at
  places=4.
- **Tiny-AI training scaffold for the Netflix VMAF corpus (ADR-0242)**
  (fork-local): scaffold-only PR preparing the tiny-AI training pipeline for
  the local Netflix VMAF corpus (9 ref / 70 distorted YUVs at
  `.workingdir2/netflix/`). Ships `docs/ai/training-data.md` with the corpus
  path convention and `--data-root` loader API; `docs/adr/0199-*.md` with
  the architecture-choice space and distillation-vs-from-scratch alternatives
  table; `docs/research/0019-tiny-ai-netflix-training.md` surveying VMAF
  training methodology and distillation literature; and an MCP end-to-end
  smoke test (`mcp-server/vmaf-mcp/tests/test_smoke_e2e.py`) that exercises
  the `vmaf_score` JSON-RPC tool against the Netflix golden fixture. No
  training runs; no Netflix golden assertions modified. Follow-up PR will
  select architecture and run training.
- **GPU long-tail batch 3 parts 6b + 6c — `float_adm_cuda` +
  `float_adm_sycl` extractors (T7-23 / ADR-0192 / ADR-0202)**
  (fork-local): CUDA + SYCL twins of the Vulkan kernel shipped in
  PR #154 / [ADR-0199](docs/adr/0199-float-adm-vulkan.md). Direct
  ports of the four-stage / four-scale Vulkan pipeline: same
  `-1` mirror form, same fused stage 3 with cross-band CM
  threshold, same per-scale 6-slot WG partials reduced on the
  host in `double`. New files:
  [`libvmaf/src/feature/cuda/float_adm/float_adm_score.cu`](libvmaf/src/feature/cuda/float_adm/float_adm_score.cu),
  [`libvmaf/src/feature/cuda/float_adm_cuda.{c,h}`](libvmaf/src/feature/cuda/float_adm_cuda.c),
  [`libvmaf/src/feature/sycl/float_adm_sycl.cpp`](libvmaf/src/feature/sycl/float_adm_sycl.cpp).
  Two precision-critical fixes from bring-up: (1) `--fmad=false`
  on the `float_adm_score` fatbin via a new per-kernel
  `cuda_cu_extra_flags` dict in `meson.build` — NVCC's default
  FMA contraction in the angle-flag dot product cascaded
  through the cube reductions and pushed scale-3 / adm2 past
  `places=4` (3.6e-4 max_abs vs CPU before fix). Scoped to this
  one kernel; integer ADM keeps its existing FMA-on path.
  (2) Parent-LL dimension trap — stage 0 at `scale > 0` clamps
  against the **parent's LL output dims** (`scale_w/h[scale]`),
  NOT the parent's full-resolution image dims
  (`scale_w/h[scale - 1]`). Verified `max_abs_diff ≤ 6e-6`
  across all five outputs (adm2, adm_scale0..3) on the Netflix
  normal pair via `cross_backend_vif_diff.py --backend cuda
  --feature float_adm --places 4` (0/48 mismatches);
  checkerboard 1px is bit-exact. SYCL gates on lavapipe-equivalent
  CI lanes already cover the `float_adm` feature surface from
  PR #154 ([ADR-0199](docs/adr/0199-float-adm-vulkan.md)).
- **GPU long-tail batch 3 part 6 — `float_adm_vulkan` extractor
  (T7-23 / ADR-0192 / ADR-0199)** (fork-local): sixth and final
  Group B float twin. Vulkan compute kernel for the float ADM
  feature extractor. Float twin of `integer_adm_vulkan`
  ([ADR-0178](docs/adr/0178-integer-adm-vulkan.md)) — same 4-stage
  / 4-scale wave-of-stages design (16 pipelines) but with float
  buffers and host-side `double` accumulation. New files:
  [`libvmaf/src/feature/vulkan/float_adm_vulkan.c`](libvmaf/src/feature/vulkan/float_adm_vulkan.c),
  [`libvmaf/src/feature/vulkan/shaders/float_adm.comp`](libvmaf/src/feature/vulkan/shaders/float_adm.comp).
  Mirror-asymmetry status: float_adm has NO trap analogous to
  [ADR-0197](docs/adr/0197-float-vif-gpu.md) — both the scalar
  `adm_dwt2_s` and the AVX2 `float_adm_dwt2_avx2` consume the same
  `dwt2_src_indices_filt_s` index buffer (`2 * sup - idx - 1` for
  both axes); the GPU follows that. `places=4` cross-backend
  contract on the lavapipe lane (new step in
  [`tests-and-quality-gates.yml`](.github/workflows/tests-and-quality-gates.yml)).
  `scripts/ci/cross_backend_vif_diff.py` gains a `float_adm` entry
  in `FEATURE_METRICS`. CUDA + SYCL twins land in a focused
  follow-up PR.
- **GPU long-tail batch 3 part 7 — `ssimulacra2_vulkan` extractor
  (T7-23 / ADR-0192 / ADR-0201)** (fork-local): Vulkan twin of the
  CPU `ssimulacra2` extractor with a hybrid host/GPU pipeline. The
  GPU runs the IIR blur (separable Charalampidis 2016 3-pole, one
  workgroup per row / per column) and the 3-plane elementwise
  product. Host runs YUV → linear-RGB, 2×2 pyramid downsample,
  linear-RGB → XYB (bit-exact port of `linear_rgb_to_xyb`), and
  the per-pixel SSIM + EdgeDiff combine in double precision over
  the GPU-blurred mu/sigma buffers. Empirical CPU-vs-Vulkan on
  Netflix normal pair (576×324, 48 frames): pooled `ssimulacra2`
  `max_abs_diff = 1.81e-7` (mean 3.65e-8, P95 1.56e-7). The
  cross-backend gate runs at `places=4` — matching the rest of
  the Vulkan VIF/MS-SSIM family. ADR-0201 §Precision investigation
  documents the five-tactic measurement chain that drove the
  contract from a `places=1` first-iteration shipping condition
  to `places=4`. CUDA + SYCL twins follow in a separate PR. See
  [ADR-0201](docs/adr/0201-ssimulacra2-vulkan-kernel.md).
- **Port Netflix upstream b949cebf — feature/motion: port several feature
  extractor options** (upstream port, Research-0024 Strategy A):
  Verbatim port of Netflix/vmaf commit b949cebf (Kyle Swanson, 2026-04-27).
  Adds 8 new options to `float_motion` and 3 missing options to
  `integer_motion`: `motion_blend_factor` (default 1.0, no-op),
  `motion_blend_offset` (default 40.0), `motion_fps_weight` (default 1.0,
  no-op), `motion_add_scale1` (default false, no-op), `motion_filter_size`
  (default 5, no-op — preserves FILTER_5_s), `motion_add_uv` (default
  false, no-op), `motion_max_val` (default 10000.0, no-op). Adds
  `VMAF_feature_motion3_score` and `VMAF_integer_feature_motion3_score`
  outputs. Adds `FILTER_3_s`, `FILTER_5_NO_OP_s`, and
  `DEFAULT_MOTION_FILTER_SIZE` to `motion_tools.h`. Adds `motion_decimate`
  parameter to `compute_motion()` and `motion_add_scale1` to
  `vmaf_image_sad_c()`. Also ports `picture_copy()` channel-parameter
  change (from d3647c73) required as prerequisite. All defaults are no-ops:
  integer_motion2 and float_motion2 scores are bit-identical to pre-port
  baseline. Netflix golden assertions unaffected.

- **GPU long-tail batch 2 parts 3b + 3c — `psnr_hvs_cuda` +
  `psnr_hvs_sycl` extractors (T7-23 / ADR-0188 / ADR-0191)**
  (fork-local): closes batch 2 part 3 (and batch 2 entirely).
  CUDA + SYCL twins of the Vulkan psnr_hvs kernel shipped in
  PR #143. Both inherit the per-plane single-dispatch design
  — one work-group per output 8×8 block (step=7), 64 threads
  per WG, cooperative load + thread-0-serial reductions
  matching CPU's exact i,j summation order (locks float
  bit-order to `calc_psnrhvs`). Three dispatches per frame
  (Y / Cb / Cr); host accumulates per-plane partials in float
  matching CPU's `ret` register pattern, then
  `10·log10(1/score)` per plane and combined
  `psnr_hvs = 0.8·Y + 0.1·(Cb + Cr)`.
  - **CUDA** (~270 LOC PTX in
    [`integer_psnr_hvs/psnr_hvs_score.cu`](libvmaf/src/feature/cuda/integer_psnr_hvs/psnr_hvs_score.cu)
    + ~330 LOC host in
    [`integer_psnr_hvs_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_psnr_hvs_cuda.c)):
    picture_copy host-side via `cuMemcpy2DAsync` D2H per
    plane (honours pitched `cuMemAllocPitch` device buffer —
    same fix as ms_ssim_cuda PR #142). Per-plane state arrays
    (`d_ref[3] / d_dist[3] / d_partials[3]`) + pinned host
    staging.
  - **SYCL** (~420 LOC, single TU
    [`integer_psnr_hvs_sycl.cpp`](libvmaf/src/feature/sycl/integer_psnr_hvs_sycl.cpp)):
    self-contained submit/collect (mirrors `ms_ssim_sycl`).
    Host-pinned USM staging carries the picture_copy-
    normalised float planes per plane. Inline picture_copy
    clone (libvmaf's `picture_copy` hardcodes plane 0).
    fp64-free.
  - **Verification**: 48 frames at 576×324 vs CPU scalar.
    **CUDA on NVIDIA RTX 4090** → `max_abs = 8.3e-5` (Y plane,
    same floor as Vulkan), `0/48 places=3 mismatches`.
    **SYCL on Intel Arc A380** → `max_abs = 1.0e-6` across all
    four metrics, `0/48 places=4 mismatches`. SYCL is the
    only backend that hits `places=4` on psnr_hvs — icpx's
    `-fp-model=precise` flag (project-wide SYCL strict-FP
    setting) produces tighter CPU-matching precision than
    nvcc default or glslc default. Investigation in
    [ADR-0191 §"Why not places=4"](docs/adr/0191-psnr-hvs-vulkan.md)
    documents what was tried for the CUDA + Vulkan paths;
    `--fmad=false` was tested for CUDA and didn't help,
    ruling out FMA fusion as the dominant drift source.
  - **v1 limitation**: rejects YUV400P (no chroma) and
    `bpc > 12` (matches CPU). 3 dispatches/frame at 1080p.

- **GPU long-tail batch 2 part 3a — `psnr_hvs_vulkan` extractor
  (T7-23 / ADR-0188 / ADR-0191)** (fork-local): first DCT-based
  GPU kernel in the fork. Vulkan twin of the active CPU
  `psnr_hvs` extractor. Per-plane single-dispatch design — one
  workgroup per output 8×8 block (step=7 sliding window),
  64 threads per workgroup. Cooperative load + per-quadrant
  reductions + scalar Xiph integer DCT (`od_bin_fdct8x8`,
  lifting + RSHIFT, `int32` arithmetic byte-for-byte against
  `third_party/xiph/psnr_hvs.c`) + per-coefficient masking +
  `subgroupAdd` per-block float partial. Host accumulates
  partials in `double` per plane, applies
  `score / pixels / samplemax²` then `10·log10(1/score)` per
  plane. Combined `psnr_hvs = 0.8·Y + 0.1·(Cb + Cr)`. New
  [`libvmaf/src/feature/vulkan/shaders/psnr_hvs.comp`](libvmaf/src/feature/vulkan/shaders/psnr_hvs.comp)
  + [`libvmaf/src/feature/vulkan/psnr_hvs_vulkan.c`](libvmaf/src/feature/vulkan/psnr_hvs_vulkan.c)
  (~540 LOC host). Three pipelines (one per plane, baked-in
  PLANE + BPC specialisation constants); CSF tables per plane
  baked into shader as `const float[64]` arrays. Rejects
  YUV400P (no chroma) and `bpc > 12` (matches CPU). Empirical:
  48 frames at 576×324 on **Intel Arc A380** vs CPU scalar —
  `max_abs = 8.2e-5` across all four metrics
  (`psnr_hvs_y / _cb / _cr / psnr_hvs`), `0/48 places=3
  mismatches`. Gate runs at `places=3` (better than ADR-0188's
  `places=2` floor). New CI step `psnr_hvs cross-backend diff
  (CPU vs Vulkan/lavapipe)` on the lavapipe lane. CUDA + SYCL
  twins follow as batch 2 parts 3b + 3c.

- **GPU long-tail batch 2 parts 2b + 2c — `float_ms_ssim_cuda` +
  `float_ms_ssim_sycl` extractors (T7-23 / ADR-0188 / ADR-0190)**
  (fork-local): closes batch 2 part 2. CUDA + SYCL twins of the
  Vulkan ms_ssim kernel shipped in PR #141. Both inherit the
  three-kernel design — decimate (9-tap 9/7 biorthogonal LPF +
  2× downsample), horiz (11-tap separable Gaussian over five
  SSIM stats), vert+lcs (vertical 11-tap + per-pixel l/c/s +
  per-WG / per-block float partials × 3). Host accumulates
  partials in `double` per scale and applies the Wang weights
  for the `MS-SSIM = ∏_i l[i]^α[i]·c[i]^β[i]·s[i]^γ[i]`
  combine.
  - **CUDA** (~210 LOC PTX in
    [`integer_ms_ssim/ms_ssim_score.cu`](libvmaf/src/feature/cuda/integer_ms_ssim/ms_ssim_score.cu)
    + ~470 LOC host in
    [`integer_ms_ssim_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_ms_ssim_cuda.c)):
    picture_copy normalisation runs on the host (uint → float in
    `[0, 255]`) via `cuMemcpy2DAsync` D2H of the pitched device
    plane into a contiguous pinned host buffer, then H2D upload
    to pyramid level 0. Surfaced one bring-up bug: the device
    plane is allocated by `cuMemAllocPitch` with `stride[0] ≥
    width·bpc` — naïve `cuMemcpyDtoHAsync` of `width·height·bpc`
    bytes mis-copies row N≥1 because it ignores the device
    pitch. Fix: 2D copy honouring `srcPitch = ref_pic->stride[0]`
    + `dstPitch = width·bpc` produces the contiguous host buffer
    `picture_copy` expects.
  - **SYCL** (~510 LOC, single TU
    [`integer_ms_ssim_sycl.cpp`](libvmaf/src/feature/sycl/integer_ms_ssim_sycl.cpp)):
    self-contained submit/collect (does NOT register with
    `vmaf_sycl_graph_register` — same rationale as ssim_sycl).
    Host-pinned USM staging carries the picture_copy-normalised
    float planes; `nd_range<2>` vert+lcs kernel uses
    `sycl::reduce_over_group` × 3 for per-WG partials.
    fp64-free (Intel Arc A380).
  - **Verification**: 48 frames at 576×324 vs CPU scalar —
    `max_abs = 1.0e-6`, `0/48 places=4 mismatches` on **NVIDIA
    RTX 4090** (CUDA) and **Intel Arc A380** (SYCL). Same
    precision floor as `ms_ssim_vulkan` (PR #141).
  - **v1 limitation** (same as ms_ssim_vulkan): no `enable_lcs`
    — 15 extra per-scale metrics deferred to a focused
    follow-up.

- **GPU long-tail batch 2 part 2a — `float_ms_ssim_vulkan`
  extractor (T7-23 / ADR-0188 / ADR-0190)** (fork-local):
  Wang multi-scale SSIM on Vulkan. 5-level pyramid built via
  9-tap 9/7 biorthogonal LPF + 2× downsample
  ([`ms_ssim_decimate.comp`](libvmaf/src/feature/vulkan/shaders/ms_ssim_decimate.comp),
  matches `ms_ssim_decimate_scalar` byte-for-byte). Per-scale
  SSIM compute via a variant of `ssim.comp` that emits **three**
  per-WG partials (`l, c, s`) instead of a single combined
  SSIM
  ([`ms_ssim.comp`](libvmaf/src/feature/vulkan/shaders/ms_ssim.comp)).
  Host accumulates partials in `double` per scale, applies the
  Wang weights `α/β/γ` (matches `ms_ssim.c::g_alphas/g_betas/
  g_gammas` byte-for-byte) for the
  `MS-SSIM = ∏_i l[i]^α[i]·c[i]^β[i]·s[i]^γ[i]` combine on host.
  New
  [`libvmaf/src/feature/vulkan/ms_ssim_vulkan.c`](libvmaf/src/feature/vulkan/ms_ssim_vulkan.c)
  (~700 LOC). Min-dim guard mirrors
  [ADR-0153](docs/adr/0153-float-ms-ssim-min-dim-netflix-1414.md)
  (176×176 minimum). v1 does **not** implement `enable_lcs` (15
  extra per-scale metrics) — deferred to a focused follow-up.
  Surfaced one bring-up bug: `ssim_variance_scalar` clamps
  `ref_sigma_sqd / cmp_sigma_sqd` to `MAX(0, ...)` before the
  sqrt at line 165 of `iqa/ssim_tools.c`; missing this clamp
  produces NaN at scale 0 when float ULP errors push variances
  slightly negative on flat regions. Empirical: 48 frames at
  576×324 on **Intel Arc A380** vs CPU scalar — `max_abs =
  2.0e-6`, `0/48 places=4 mismatches`. New CI step
  `float_ms_ssim cross-backend diff (CPU vs Vulkan/lavapipe)`
  on the lavapipe lane. CUDA + SYCL twins follow as batch 2
  parts 2b + 2c.

- **GPU long-tail batch 2 parts 1b + 1c — `float_ssim_cuda` +
  `float_ssim_sycl` extractors (T7-23 / ADR-0188 / ADR-0189)**
  (fork-local): closes batch 2 part 1. CUDA + SYCL twins of
  the Vulkan ssim kernel shipped in PR #139. Both inherit the
  two-pass design (horizontal 11-tap separable Gaussian → 5
  intermediate float buffers, then vertical 11-tap + per-pixel
  SSIM combine + per-WG / per-block float partial sums; host
  accumulates in `double`).
  - **CUDA** (~210 LOC PTX in
    [`integer_ssim/ssim_score.cu`](libvmaf/src/feature/cuda/integer_ssim/ssim_score.cu)
    + ~340 LOC host in
    [`integer_ssim_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_ssim_cuda.c)):
    `picture_copy` normalisation (uint → float / scaler) inlined
    in the horizontal kernel — no extra host-side conversion
    since picture_cuda already uploaded the raw uint plane.
    Per-block float partials reduced on host in `double`.
  - **SYCL** (~370 LOC, single TU
    [`integer_ssim_sycl.cpp`](libvmaf/src/feature/sycl/integer_ssim_sycl.cpp)):
    self-contained submit/collect (does NOT register with
    `vmaf_sycl_graph_register` — `shared_frame` is luma-only
    packed at uint width and SSIM needs `picture_copy`-normalised
    float planes). Host-pinned USM staging carries the
    normalised ref/cmp; `nd_range<2>` vertical kernel with
    `sycl::reduce_over_group` builds per-WG partials. fp64-free
    (Intel Arc A380 lacks native fp64 — same constraint as
    ciede_sycl).
  - **Verification**: 48 frames at 576×324 vs CPU scalar —
    `max_abs = 1.0e-6`, `0/48 places=4 mismatches` on **NVIDIA
    RTX 4090** (CUDA) and **Intel Arc A380** (SYCL). Same
    precision floor as `ssim_vulkan` (PR #139). Comfortably
    under `places=4` threshold (5e-5).
  - **v1 limitation** (same as ssim_vulkan): GPU paths support
    `scale=1` only — auto-detect rejects `scale > 1` with
    `-EINVAL`. Production 1080p needs
    `--feature float_ssim_{cuda,sycl}:scale=1` pinned (or
    smaller input). GPU-side decimation is a v2 follow-up.
  ms_ssim (batch 2 part 2) follows next.

- **GPU long-tail batch 2 part 1a — `float_ssim_vulkan`
  extractor (T7-23 / ADR-0188 / ADR-0189)** (fork-local):
  Vulkan twin of the active CPU `float_ssim`. **Two-dispatch
  design** — horizontal 11-tap separable Gaussian over
  ref / cmp / ref² / cmp² / ref·cmp into five intermediate
  float buffers, then vertical 11-tap + per-pixel SSIM
  combine + per-WG float partial sums. Host accumulates
  partials in `double` and divides by `(W-10)·(H-10)`
  (matches CPU's `iqa_ssim` valid-region averaging). 11-tap
  Gaussian weights baked into GLSL byte-for-byte from
  `g_gaussian_window_h` in `iqa/ssim_tools.h`. picture_copy
  host-side normalises uint sample → float `[0, 255]` before
  upload (matches `float_ssim.c::extract`). New
  [`libvmaf/src/feature/vulkan/shaders/ssim.comp`](libvmaf/src/feature/vulkan/shaders/ssim.comp)
  + [`libvmaf/src/feature/vulkan/ssim_vulkan.c`](libvmaf/src/feature/vulkan/ssim_vulkan.c)
  (~510 LOC host). **v1 limitation**: GPU path supports
  `scale=1` only — auto-detect rejects `scale > 1` with
  `-EINVAL`; production 1080p needs
  `--feature float_ssim_vulkan:scale=1` pinned (or smaller
  input). Cross-backend gate fixture (576×324) auto-resolves
  to `scale=1`. GPU-side decimation is a v2 follow-up.
  Empirical: 48 frames at 576×324 on **Intel Arc A380** vs
  CPU scalar — `max_abs = 1.0e-6`, `0/48 places=4
  mismatches`. Comfortably under the `places=4` threshold
  (5e-5). New CI step `float_ssim cross-backend diff (CPU vs
  Vulkan/lavapipe)` on the lavapipe lane. CUDA + SYCL twins
  follow as batch 2 parts 1b + 1c.

- **GPU long-tail batch 1c parts 2 + 3 — `ciede_cuda` +
  `ciede_sycl` extractors (T7-23 / ADR-0182 / ADR-0187)**
  (fork-local): closes batch 1c. CUDA + SYCL twins of the
  Vulkan ciede kernel shipped in PR #136. Both emit the
  `ciede2000` metric (logarithmic transform `45 - 20·log10(mean_ΔE)`).
  - **CUDA** (~270 LOC PTX +
    [`integer_ciede_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_ciede_cuda.c)
    ~245 LOC host): per-pixel float ciede2000 (chroma read
    inline at the subsampled position — avoids the host-side
    upscale step), per-block partials reduced on the host in
    `double`. Surfaced a latent `vmaf_cuda_picture_upload_async`
    bug: the bitmask was hardcoded to `0x1` (luma only) in
    `libvmaf.c::translate_picture_host`, leaving chroma device
    buffers uninitialised — fine for every prior CUDA extractor
    (psnr / motion / adm / vif / moment all luma-only) but
    wrong for ciede. Bitmask now picks `0x7` for any pix_fmt
    other than YUV400P; CUDA chroma-aware kernels are unblocked.
  - **SYCL** (~470 LOC, single TU
    [`integer_ciede_sycl.cpp`](libvmaf/src/feature/sycl/integer_ciede_sycl.cpp)):
    self-contained submit/collect (does **not** register with
    `vmaf_sycl_graph_register` — `shared_frame` buffers are
    luma-only). Host-pinned USM staging upscales chroma to
    luma resolution (mirrors `ciede.c::scale_chroma_planes`);
    `nd_range<2>` kernel with `sycl::reduce_over_group` builds
    per-WG float partials; host accumulates in `double`.
    fp64-free (Intel Arc A380 lacks native fp64 — earlier
    `sycl::reduction<double>` attempt threw at runtime).
  - **Verification**: 48 frames at 576×324 vs CPU scalar —
    `max_abs = 1.2e-5`, `0/48 places=4 mismatches` on
    **NVIDIA RTX 4090** (CUDA) and **Intel Arc A380** (SYCL).
    Same precision floor as `ciede_vulkan` (PR #136).
  Closes batch 1c (Vulkan + CUDA + SYCL all done) — every GPU
  long-tail metric in [ADR-0182](docs/adr/0182-gpu-long-tail-batch-1.md)
  now has a working twin on at least one GPU backend.

- **GPU long-tail batch 1c part 1 — `ciede_vulkan` extractor
  (T7-23 / ADR-0187)** (fork-local): Vulkan twin of the CPU
  `ciede` extractor — the first non-bit-exact GPU kernel in
  the fork. Per-pixel ciede2000 ΔE uses ~40 transcendental ops
  (`pow` / `sqrt` / `sin` / `atan2`), so bit-exactness against
  the libm-based CPU is not on the table. New GLSL shader
  ([`libvmaf/src/feature/vulkan/shaders/ciede.comp`](libvmaf/src/feature/vulkan/shaders/ciede.comp))
  emits per-WG `float` partial sums; host accumulates in
  `double`, divides by W·H, and applies the CPU's logarithmic
  transform `45 - 20·log10(mean_ΔE)` for the final `ciede2000`
  metric. 6 storage-buffer bindings (ref + dis Y/U/V at full
  luma resolution); chroma upscaled host-side via the same
  pattern as `ciede.c::scale_chroma_planes`. New
  [`libvmaf/src/feature/vulkan/ciede_vulkan.c`](libvmaf/src/feature/vulkan/ciede_vulkan.c)
  (~480 LOC). Empirical: 48 frames at 576×324 on **Intel Arc
  A380** vs CPU scalar — `max_abs = 1.0e-5`, `0/48 places=4
  mismatches`. Empirical floor lands well under `places=4`
  threshold (≤5e-5), so the cross-backend gate runs at
  `places=4` for parity with the existing kernels. New CI
  step `ciede cross-backend diff (CPU vs Vulkan/lavapipe)` on
  the lavapipe lane. CUDA + SYCL twins follow as batch 1c
  parts 2 + 3 (last GPU long-tail rows).

- **GPU long-tail batch 1d parts 2 + 3 — `float_moment_cuda`
  and `float_moment_sycl` extractors (T7-23 / ADR-0182)**
  (fork-local): closes batch 1d. CUDA + SYCL twins of the
  Vulkan kernel shipped in PR #133 (ADR-0182). Both emit all
  four metrics — `float_moment_ref1st`, `float_moment_dis1st`,
  `float_moment_ref2nd`, `float_moment_dis2nd` — in **one
  kernel pass** via four atomic int64 counters.
  - **CUDA** (~120 LOC PTX +
    [`integer_moment_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_moment_cuda.c)
    ~225 LOC host): warp-shuffle int64 reduction (uint64 via
    two uint32 shuffles, same trick as `psnr_score.cu`) + four
    `atomicAdd(unsigned long long *)`. Same async submit /
    collect model as `psnr_cuda` (PR #129).
  - **SYCL** (~270 LOC, single TU
    [`integer_moment_sycl.cpp`](libvmaf/src/feature/sycl/integer_moment_sycl.cpp)):
    `sycl::atomic_ref<int64_t, ...>` × 4 in a single kernel.
    Rides the existing combined-graph submit / wait machinery
    via `vmaf_sycl_graph_register` (mirrors `psnr_sycl`,
    PR #130).
  - **Verification**: 48 frames at 576×324 vs CPU scalar —
    `max_abs = 0.0`, `0/48 places=4 mismatches` × 4 metrics
    on **NVIDIA RTX 4090** (CUDA) and **Intel Arc A380** (SYCL).
    Byte-exact at JSON precision; `int64` sum is exact on
    integer YUV inputs. `scripts/ci/cross_backend_vif_diff.py
    --feature float_moment --backend {cuda,sycl}`.
  Closes batch 1d (Vulkan + CUDA + SYCL all done); next is
  batch 1c (ciede across 3 backends).

- **Vulkan VkImage zero-copy import — implementation + FFmpeg
  filter (T7-29 parts 2 + 3, ADR-0186)** (fork-local): drops
  the `-ENOSYS` stubs from PR #128 and ships the matching
  FFmpeg-side filter in the same PR. libvmaf side: per-state
  ref/dis staging `VkBuffer` pair (HOST_VISIBLE,
  `DATA_ALIGN`-strided), `vkCmdCopyImageToBuffer` + timeline-
  semaphore wait per frame, no-op-release `VmafPicture` builder
  so `read_imported_pictures` routes through standard
  `vmaf_read_pictures`. New
  [`vmaf_vulkan_state_init_external`](libvmaf/include/libvmaf/libvmaf_vulkan.h)
  adopts the caller's VkInstance/VkDevice (required because
  source VkImage handles are device-bound). FFmpeg side: new
  [`ffmpeg-patches/0006-libvmaf-add-libvmaf-vulkan-filter.patch`](ffmpeg-patches/0006-libvmaf-add-libvmaf-vulkan-filter.patch)
  packages the `libvmaf_vulkan` filter consuming
  `AV_PIX_FMT_VULKAN` frames, pulling `AVVkFrame *` from
  `data[0]`, calling `vmaf_vulkan_state_init_external` with
  the device's compute queue, then `import_image` +
  `read_imported_pictures` per frame. Synchronous v1 design
  (fence-wait inside `import_image`); async pending-fence v2
  deferred. Smoke 10/10 (extends `test_vulkan_smoke.c` with
  five contract tests for the new surface). float_moment
  cross-backend gate clean — confirms the state-struct
  refactor doesn't regress existing kernel paths. Closes T7-29.

- **Fork rule §12 r14 — FFmpeg-patch updates ship in the same
  PR (ADR-0186)** (fork-local, process): every PR that touches
  a libvmaf public surface used by `ffmpeg-patches/` (C-API
  entry points, public headers, CLI flags,
  `meson_options.txt`, symbols probed by the
  `enabled libvmaf*` `check_pkg_config` lines) updates the
  relevant patch in the **same PR**. Pure libvmaf-internal
  refactors, doc-only, and test-only PRs are exempt. Reviewers
  verify with
  `for p in ffmpeg-patches/000*-*.patch; do git -C ffmpeg-8
  apply --check "$p"; done`. Closes a recurring failure mode
  where C-API drift broke the patch stack silently for the
  next rebase.

- **GPU long-tail batch 1d part 1 — `float_moment_vulkan`
  extractor (T7-23 / ADR-0182)** (fork-local): Vulkan twin of
  the CPU `float_moment` extractor. Single GLSL compute kernel
  ([`libvmaf/src/feature/vulkan/shaders/moment.comp`](libvmaf/src/feature/vulkan/shaders/moment.comp))
  emits all four metrics — `float_moment_ref1st`,
  `float_moment_dis1st`, `float_moment_ref2nd`,
  `float_moment_dis2nd` — in one dispatch via four atomic
  `int64` counters, using subgroup int64 reduction
  (`GL_EXT_shader_atomic_int64` +
  `GL_EXT_shader_explicit_arithmetic_types_int64`) into a
  shared array, then a single cross-subgroup
  `atomicAdd` per accumulator. Host divides the four sums by
  `width × height` to recover the raw moments. New
  [`libvmaf/src/feature/vulkan/moment_vulkan.c`](libvmaf/src/feature/vulkan/moment_vulkan.c)
  (~370 LOC) mirrors the `psnr_vulkan` scaffolding (3-binding
  descriptor set, single dispatch per frame, 8/10/12/16 bpc via
  spec constants). Empirical: 48 frames at 576×324 on Intel Arc
  A380 (lavapipe-equivalent) vs CPU scalar — `max_abs = 0.0`,
  `0/48 places=4 mismatches` × 4 metrics via
  `scripts/ci/cross_backend_vif_diff.py --feature float_moment
  --backend vulkan`. CUDA + SYCL twins follow as batch 1d parts
  2 and 3.

- **GPU long-tail batch 1b part 2 — `psnr_sycl` extractor
  (T7-23 / ADR-0182)** (fork-local): SYCL twin of `psnr_cuda`
  (PR #129) and `psnr_vulkan` (PR #125). Per-pixel int64
  squared-error reduction with `sycl::atomic_ref` accumulation
  into a shared device counter. Single kernel per frame, rides
  the existing combined-graph submit/wait machinery via
  `vmaf_sycl_graph_register`. New
  [`libvmaf/src/feature/sycl/integer_psnr_sycl.cpp`](libvmaf/src/feature/sycl/integer_psnr_sycl.cpp)
  (~280 LOC). Empirical: 48 frames at 576×324 on Intel Arc
  A380 vs CPU scalar — `max_abs_diff = 0.0`, `0/48 places=4
  mismatches` via `scripts/ci/cross_backend_vif_diff.py
  --backend sycl`. Closes "psnr on all 3 GPU backends" goal
  from ADR-0182.

- **GPU long-tail batch 1b part 1 — `psnr_cuda` extractor
  (T7-23 / ADR-0182)** (fork-local): CUDA twin of the
  `psnr_vulkan` kernel shipped in PR #125. Per-pixel int64
  squared-error reduction with warp-shuffle + atomicAdd
  (same pattern as `motion_score.cu`'s SAD reduction).
  Single dispatch per frame; emits luma-only `psnr_y` v1.
  New
  [`libvmaf/src/feature/cuda/integer_psnr/psnr_score.cu`](libvmaf/src/feature/cuda/integer_psnr/psnr_score.cu)
  (~120 LOC PTX) +
  [`libvmaf/src/feature/cuda/integer_psnr_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_psnr_cuda.c)
  (~210 LOC host using CUDA's async submit/collect model).
  Empirical: 48 frames at 576×324 on NVIDIA RTX 4090 vs CPU
  scalar — `max_abs_diff = 0.0`, `0/48 places=4 mismatches`
  via `scripts/ci/cross_backend_vif_diff.py --backend cuda`.
  `psnr_sycl` follows in batch 1b part 2.

- **Vulkan VkImage zero-copy import C-API scaffold — T7-29
  part 1 (ADR-0184)** (fork-local): adds three new entry
  points in
  [`libvmaf_vulkan.h`](libvmaf/include/libvmaf/libvmaf_vulkan.h)
  — `vmaf_vulkan_import_image`, `vmaf_vulkan_wait_compute`,
  `vmaf_vulkan_read_imported_pictures` — mirroring the SYCL
  backend's existing import surface. Lets future FFmpeg-side
  filters consume `AVFrame->format == AV_PIX_FMT_VULKAN`
  frames without a `hwdownload,format=yuv420p` round-trip.
  Header purity: Vulkan handles cross the ABI as `uintptr_t`
  to keep the surface usable from translation units that
  don't have `<vulkan/vulkan.h>` in scope (matches the
  libvmaf_cuda.h precedent). **Scaffold only**: every
  function returns `-ENOSYS`, mirroring how the original
  Vulkan backend shipped via ADR-0175. T7-29 part 2 (real
  `vkCmdCopyImageToBuffer` + timeline-semaphore wait) and
  part 3 (FFmpeg-side `libvmaf_vulkan` filter as
  `ffmpeg-patches/0006-*`) follow in subsequent PRs. See
  [ADR-0184](docs/adr/0184-vulkan-image-import-scaffold.md).

- **`libvmaf_sycl` FFmpeg filter — zero-copy QSV/VAAPI import
  (T7-28, ADR-0183)** (fork-local): closes the hwdec ergonomic
  gap exposed by PR #126. New
  [`ffmpeg-patches/0005-libvmaf-add-libvmaf-sycl-filter.patch`](ffmpeg-patches/0005-libvmaf-add-libvmaf-sycl-filter.patch)
  adds a dedicated `libvmaf_sycl` filter that consumes oneVPL
  `mfxFrameSurface1` frames (`AVFrame->data[3]`), extracts the
  underlying VA surface ID, and routes through
  `vmaf_sycl_import_va_surface` for zero-copy DMA-BUF import on
  the Level Zero / SYCL compute queue. Build FFmpeg with
  `--enable-libvmaf-sycl` (in addition to `--enable-libvmaf`).
  Removes the `hwdownload,format=yuv420p` round-trip for the
  common Intel QSV hwdec path. Pairs with the existing
  `0003-libvmaf-wire-sycl-backend-selector.patch` so users have
  two paths: `libvmaf=sycl_device=N` for software frames + SYCL
  compute, `libvmaf_sycl=…` for QSV hwdec + zero-copy SYCL.
  Validated on Intel Arc A380. **T7-29** (Vulkan VkImage import)
  remains open — needs new C-API surface in
  [`libvmaf_vulkan.h`](libvmaf/include/libvmaf/libvmaf_vulkan.h)
  before the FFmpeg-side filter can land. See
  [ADR-0183](docs/adr/0183-ffmpeg-libvmaf-sycl-filter.md).

- **GPU long-tail batch 1a — `psnr_vulkan` extractor (T7-23 /
  ADR-0182)** (fork-local): first kernel of the GPU long-tail
  batch. Per-pixel squared-error reduction on the Vulkan compute
  backend; emits `psnr_y` (luma-only v1; chroma is a focused
  follow-up since `picture_vulkan` upload is luma-only today).
  New
  [`libvmaf/src/feature/vulkan/shaders/psnr.comp`](libvmaf/src/feature/vulkan/shaders/psnr.comp)
  (89 LOC GLSL, 16×8 WG, subgroup-int64 reduction) +
  [`libvmaf/src/feature/vulkan/psnr_vulkan.c`](libvmaf/src/feature/vulkan/psnr_vulkan.c)
  (391 LOC host C, single dispatch/frame, no temporal state).
  Cross-backend gate gains a 4th step ("PSNR cross-backend diff")
  on the lavapipe lane. Empirical: 48 frames at 576×324 on
  Intel Arc A380 / Mesa anv vs CPU scalar — `max_abs_diff = 0.0`,
  `0/48 places=4 mismatches`. Foundation also adds chars
  descriptors to the existing scalar `psnr` / `ciede` /
  `float_moment` registrations ahead of CUDA / SYCL twins
  landing in batches 1b–1d. See
  [ADR-0182](docs/adr/0182-gpu-long-tail-batch-1.md).

- **T7-26 — Global feature-characteristics registry + per-backend
  dispatch-strategy modules** (fork-local): consolidates the
  per-context SYCL graph-replay heuristic into a per-feature
  decision driven by a registry on `VmafFeatureExtractor`. New
  [`libvmaf/src/feature/feature_characteristics.h`](libvmaf/src/feature/feature_characteristics.h)
  exposes the descriptor struct (`n_dispatches_per_frame`,
  `is_reduction_only`, `min_useful_frame_area`,
  `dispatch_hint`). Per-backend glue under
  [`libvmaf/src/{cuda,sycl,vulkan}/dispatch_strategy.{c,h}`](libvmaf/src/sycl/dispatch_strategy.cpp)
  translates the descriptor to backend primitives (SYCL graph
  replay today; CUDA graph capture and Vulkan secondary-cmdbuf
  reuse are stubs that ship the env-override surface for a
  follow-up PR to enable). New env knobs:
  `VMAF_SYCL_DISPATCH=feature:graph,feature:direct,...`,
  `VMAF_CUDA_DISPATCH=...`,
  `VMAF_VULKAN_DISPATCH=feature:reuse,feature:primary,...`.
  Legacy `VMAF_SYCL_USE_GRAPH` / `VMAF_SYCL_NO_GRAPH` kept as
  global aliases. Descriptors seeded for vif (4 dispatches),
  motion (2 dispatches, 1080p area), adm (16 dispatches, 720p
  area). Empirical: ADM at 576×324 within 0.5% of pre-T7-26
  behaviour (registry preserves byte-for-byte AUTO + 720p
  semantics). Foundation for adding the GPU long-tail (14
  metrics × 3 backends = up to 42 future kernels) without
  duplicate dispatch logic. Side-fix: pre-existing GCC LTO
  type-mismatch surfaced by the new `chars` field —
  `null.c` / `feature_lpips.c` / `ssimulacra2.c` were missing
  `#include "config.h"` and saw a smaller `VmafFeatureExtractor`
  struct than `feature_extractor.c`. See
  [ADR-0181](docs/adr/0181-feature-characteristics-registry.md).

- **`float_moment` SIMD parity (AVX2 + NEON) — T7-19, closes
  the only fully-scalar row in the SIMD-coverage matrix**
  (fork-local): new
  [`libvmaf/src/feature/x86/moment_avx2.{c,h}`](libvmaf/src/feature/x86/moment_avx2.c)
  and [`libvmaf/src/feature/arm64/moment_neon.{c,h}`](libvmaf/src/feature/arm64/moment_neon.c)
  implement `compute_1st_moment` / `compute_2nd_moment` 8-wide
  (AVX2) and 4-wide (NEON) following the `ansnr_avx2.c` pattern:
  square in float, accumulate into `double` via scattered-tmp
  (AVX2) or lane-pair widening via `vcvt_f64_f32` (NEON).
  Dispatched from
  [`float_moment.c::init`](libvmaf/src/feature/float_moment.c)
  via function pointers selected from `vmaf_get_cpu_flags()`.
  Tolerance-bounded contract (1e-7 relative — ~500× tighter than
  the production snapshot gate's `places=4`), matching the
  established kernel header documentation. New
  [`test_moment_simd`](libvmaf/test/test_moment_simd.c) runs
  four cases per arch (two random seeds, an aligned width, and a
  tiny edge case to exercise the per-row tail). End-to-end CLI
  output unchanged at JSON `%g` precision. See
  [ADR-0179](docs/adr/0179-float-moment-simd.md).

- **Vulkan ADM kernel + cross-backend gate fixes — T5-1c (closes T5-1c)**
  (fork-local): replaces the 37-line adm_vulkan.c stub with a real
  `VmafFeatureExtractor` (~700 LOC) backed by a new GLSL compute shader
  [`shaders/adm.comp`](libvmaf/src/feature/vulkan/shaders/adm.comp)
  (~660 LOC). Implements 4-scale CDF 9/7 DWT, decouple+CSF fused
  pass, and per-band CSF-denominator + contrast-measure reductions.
  16 pipelines per extractor (one per `(scale, stage)`). Provides the
  standard `integer_adm2`, `integer_adm_scale0..3` outputs.

  This PR ALSO uncovered and fixed three latent bugs that made the
  cross-backend gate land bogus ULP=0 results since PR #118:
  (a) `tools/meson.build` never set `-DHAVE_VULKAN`, so every
  `--vulkan_device` call silently no-op'd; (b)
  `vmaf_use_feature()` skipped `set_fex_vulkan_state()` so the
  imported state never reached the extractor; (c) the script's
  `--feature X` invocation collided with the default model's CPU
  extractors, dropping the GPU writer's scores. Plus a header
  shadowing fix (Vulkan `common.h` → `vulkan_common.h` so
  CUDA+Vulkan can build together) and a new `--backend
  {auto,cpu,cuda,sycl,vulkan}` CLI flag that closes the
  multi-backend dispatcher conflict (first-match-wins favored CUDA
  over Vulkan).

  Real cross-backend numbers (576x324, 48 frames, post-fix gate):

  | backend (device) | vif | motion | adm |
  | --- | --- | --- | --- |
  | CUDA (RTX 4090) | ULP=0 ✓ | **2.6e-3 ❌ 47/48** | ≤1e-6 ✓ |
  | SYCL (Arc, oneAPI) | ≤1e-6 ✓ | **2.6e-3 ❌ 47/48** | scale2 2.4e-4 ❌ 1/48 |
  | Vulkan (RTX) | ≤1e-6 ✓ | ≈1e-6 ✓ | scale2 2.4e-4 ❌ 1/48 |
  | Vulkan (Arc, Mesa anv) | ≤1e-6 ✓ | ≈1e-6 ✓ | ≤3.1e-5 ✓ |

  Three pre-existing kernel-side bugs surfaced by the working gate:
  (1) CUDA motion AND SYCL motion both drift by 2.6e-3 (47/48
  frames) vs CPU — same magnitude, likely shared algorithmic
  inheritance; (2) NVIDIA Vulkan + SYCL `adm_scale2` both drift
  by 2.4e-4 (1/48 frames) — likely shared host-side reduction
  order divergence; (3) SYCL on fp64-less GPUs (e.g. Arc A380)
  uses int64 emulation for gain limiting, causing a 5-10×
  slowdown vs Vulkan on the same hardware. Each tracked as a
  follow-up.

  Vulkan-on-Arc (the path under the lavapipe blocking gate via
  Mesa anv) is the only fully-clean GPU backend in the current
  matrix. Closes T5-1c. See
  [ADR-0178](docs/adr/0178-vulkan-adm-kernel.md).
- **Vulkan motion kernel — T5-1c (motion + motion2)** (fork-local):
  replaces the 37-line motion_vulkan.c stub with a real
  `VmafFeatureExtractor` backed by a new GLSL compute shader
  [`shaders/motion.comp`](libvmaf/src/feature/vulkan/shaders/motion.comp).
  Separable 5-tap Gaussian blur (`{3571, 16004, 26386, 16004, 3571}`,
  sum=65536) + per-WG `int64` SAD reduction; ping-pong blurred-frame
  storage; `integer_motion2` emitted with the standard 1-frame lag.
  `motion3` (5-frame window mode) deliberately deferred. Cross-backend
  diff script generalized: `scripts/ci/cross_backend_vif_diff.py`
  gains `--feature {vif,motion}`. Empirical: ≤1e-6 vs CPU on Arc
  via Vulkan (Mesa anv); 2.6e-3 drift on CUDA/SYCL motion is a
  pre-existing kernel bug surfaced by PR #120's gate fix. See
  [ADR-0177](docs/adr/0177-vulkan-motion-kernel.md). **NOTE**: the
  original "ULP=0" claim in this entry was bogus — the gate was
  comparing CPU-vs-CPU due to the build-system bug PR #120 fixes.
- **Vulkan VIF cross-backend gate + CLI (`--vulkan_device`) — T5-1b-v**
  (fork-local): wires Vulkan into the libvmaf dispatcher and `vmaf`
  CLI. New `--vulkan_device <N>` (auto-pick `-1`, default disabled)
  and `--no_vulkan` flags. Adds `VMAF_FEATURE_EXTRACTOR_VULKAN = 1 << 5`
  and the public state-level API (`vmaf_vulkan_state_init` / `_free` /
  `_available` / `_list_devices`). New
  [`scripts/ci/cross_backend_vif_diff.py`](scripts/ci/cross_backend_vif_diff.py)
  with two CI lanes: `Vulkan VIF Cross-Backend (lavapipe, places=4)` runs
  on every PR via Mesa lavapipe (no GPU runner needed), Arc-A380
  nightly advisory parked behind `if: false` until a self-hosted runner
  with label `vmaf-arc` is registered. **NOTE**: the original
  "ULP=0 vs CPU" claim in this entry was bogus — the meson glue
  for `-DHAVE_VULKAN` in `tools/meson.build` was missing, so
  `--vulkan_device` silently no-op'd. PR #120 fixes the build
  system, the framework state propagation in `vmaf_use_feature()`,
  and the script's invocation pattern. See
  [ADR-0176](docs/adr/0176-vulkan-vif-cross-backend-gate.md).
- **Vulkan VIF math port — T5-1b-iv (4-scale GLSL kernel)** (fork-local):
  full numerical port of the SYCL VIF kernel to a GLSL compute shader.
  `shaders/vif.comp` runs four pipelines (one per `SCALE` specialization
  constant) compiled to SPIR-V via `glslc`, embedded as a byte array,
  dispatched in a single command buffer with pipeline barriers between
  scales. Uses native `int64` accumulators
  (`GL_EXT_shader_explicit_arithmetic_types_int64`) for deterministic
  reductions matching the CPU integer reference. First feature kernel
  to actually run end-to-end on Intel Arc A380.
- **Vulkan runtime bring-up — T5-1b** (fork-local): replaces the T5-1
  scaffold's `-ENOSYS` stubs with a real volk + Vulkan 1.3 + VMA bring-up.
  `vmaf_vulkan_context_new` picks a compute-capable physical device
  (auto: discrete > integrated > virtual > cpu; override via
  `device_index`), creates a dedicated compute queue family, attaches
  a VMA allocator, and exposes a command pool that per-feature dispatch
  wrappers under `libvmaf/src/feature/vulkan/` reuse. New `vma_impl.cpp`
  (C++17 TU isolating the VMA implementation), new `picture_vulkan.{c,h}`
  (VkBuffer alloc / flush / mapped-host pointer accessors). `volk` and
  `VulkanMemoryAllocator` pulled via Meson wrap files (no system install
  required); `glslc` becomes a build-time requirement when
  `-Denable_vulkan=enabled`.
- **Whole-codebase docs sweep — close audit-identified gaps**
  (fork-local): post-T5-1 docs audit identified four undocumented
  user-discoverable surfaces and one stale CLI flag entry. Adds
  [`docs/backends/arm/overview.md`](docs/backends/arm/overview.md)
  for the ARM NEON backend (build, runtime control via `--cpumask`,
  per-feature coverage table, bit-exactness contracts, CI matrix
  pointer). Documents the `motion_v2`, `lpips`, and `float_moment`
  feature extractors in
  [`docs/metrics/features.md`](docs/metrics/features.md) (table
  rows + per-feature sections covering invocation, output metrics,
  output range, input formats, options, backends, limitations).
  Expands the `--no-reference` flag entry in
  [`docs/usage/cli.md`](docs/usage/cli.md) with preconditions
  (`reference_required: false` registry field), failure modes, and
  report-format implications. Updates
  [`docs/api/gpu.md`](docs/api/gpu.md) title to include
  `libvmaf_vulkan.h` and links the new ARM overview from
  [`docs/backends/index.md`](docs/backends/index.md). No code
  changes; touched-file lint cleanup converts standalone
  `**Heading**` lines to `####` H4 (MD036), aligns options
  tables exactly (MD060), and tags bare fenced code blocks with
  `text` (MD040).
- **Vulkan compute backend — scaffold-only audit-first PR**
  (fork-local): closes BACKLOG T5-1 audit half. New public header
  [`libvmaf_vulkan.h`](libvmaf/include/libvmaf/libvmaf_vulkan.h)
  declaring the `VmafVulkanState` API surface (state_init,
  import_state, state_free, list_devices, available). New
  `libvmaf/src/vulkan/` + `libvmaf/src/feature/vulkan/` trees with
  every entry point returning `-ENOSYS`. New `enable_vulkan`
  feature option (default **disabled**) and conditional
  `subdir('vulkan')` in libvmaf's meson. New 4-sub-test smoke
  pinning the stub contract. New CI matrix row compiles with
  `-Denable_vulkan=enabled`. New ffmpeg patch
  [`0004-libvmaf-wire-vulkan-backend-selector.patch`](ffmpeg-patches/0004-libvmaf-wire-vulkan-backend-selector.patch)
  mirroring the SYCL selector — adds a `vulkan_device` libvmaf
  filter option. **Zero runtime dependencies** for the scaffold;
  `dependency('vulkan')` + volk + glslc + VMA land with the
  runtime PR per ADR-0127's "VIF as pathfinder" sequence. See
  [ADR-0175](docs/adr/0175-vulkan-backend-scaffold.md).
- **First per-model PTQ — `learned_filter_v1` flips to dynamic int8**
  (fork-local): closes T5-3 fully (audit half via ADR-0173;
  first-model half + CI gate via this PR). 80 KB → 33 KB (2.4×
  shrink). Drop measurement: PLCC 0.999883 vs fp32 on a 16-sample
  synthetic input set, drop 0.000117 vs the per-model budget 0.01
  (100× margin). Runtime `.int8.onnx` redirect wired in
  `vmaf_dnn_session_open` — when the sidecar declares
  `quant_mode != FP32`, the loader strips trailing `.onnx`,
  appends `.int8.onnx`, re-validates, and passes that path to ORT.
  Fp32 file stays on disk as the regression baseline. New
  `int8_sha256` registry/sidecar field (required when
  `quant_mode != fp32`). New `ai/scripts/measure_quant_drop.py`
  walks the registry and gates each non-fp32 model. New
  `ai-quant-accuracy` step in the `Tiny AI` CI job runs the gate
  on every PR. C2 `nr_metric_v1` stays fp32 — its dynamic-batch
  ONNX export trips ORT's internal shape inference (tracked as
  T5-3c follow-up). See
  [ADR-0174](docs/adr/0174-first-model-quantisation.md). Closes
  BACKLOG T5-3b.
- **PTQ int8 audit harness (audit-first)** (fork-local): scaffolds
  the per-model quantisation pipeline from ADR-0129. Three new
  optional fields in `model/tiny/registry.schema.json`
  (`quant_mode` enum `fp32` / `dynamic` / `static` / `qat`;
  `quant_calibration_set`; `quant_accuracy_budget_plcc` default
  0.01). Three new scripts under `ai/scripts/` (`ptq_dynamic.py`,
  `ptq_static.py`, `qat_train.py` — the last is a CLI scaffold that
  raises `NotImplementedError` until a per-model QAT PR lands the
  trainer hook). New `VmafModelQuantMode` enum + sidecar parser
  branch in `libvmaf/src/dnn/model_loader.{h,c}`; default FP32
  fail-safe on unknown sidecar values. 4 Python smoke tests + 3 C
  sidecar tests. **No shipped model flips its `quant_mode`** in
  this PR — runtime `.int8.onnx` redirect + the `ai-quant-accuracy`
  CI gate land with the first per-model quantisation PR (T5-3b).
  New
  [`docs/ai/quantization.md`](docs/ai/quantization.md) user
  reference. See
  [ADR-0173](docs/adr/0173-ptq-int8-audit-impl.md). Closes
  BACKLOG T5-3 audit half (T5-3b queued for the gate).
- **MCP `describe_worst_frames` tool with VLM fallback**
  (fork-local): new MCP tool that scores a `(ref, dis)` pair, picks
  the N worst-VMAF frames (default 5, capped at 32), extracts each
  as PNG via `ffmpeg`, and runs a vision-language model
  (SmolVLM → Moondream2 cascade) to describe the visible
  artefacts. Returns `{model_id, frames: [{frame_index, vmaf, png,
  description}]}`. Falls back to metadata-only output with a clear
  hint when the new `vlm` optional dependency group isn't
  installed. New `[vlm]` extras (`transformers + torch + Pillow +
  accelerate`); base MCP install stays light. First concrete
  consumer of ADR-0171's bounded-Loop guard — VLM autoregressive
  token generation needs `Loop` nodes. 5 new tests; all 17 MCP
  tests pass. See
  [ADR-0172](docs/adr/0172-mcp-describe-worst-frames.md). Closes
  BACKLOG T6-6.
- **Bounded `Loop.M` trip-count guard** (fork-local): closes the
  follow-up deferred in ADR-0169. Two layers, mirroring the
  ADR-0167 doc-drift enforcement pattern. (1) Python export-time
  `vmaf_train.op_allowlist` traces every `Loop`'s first input back
  to a `Constant` int64 scalar (recurses into subgraphs); rejects
  graph-input M, non-Constant producers, and values outside
  `[0, MAX_LOOP_TRIP_COUNT]` (default 1024, per-call overridable).
  `AllowlistReport.loop_violations` carries actionable diagnostics.
  (2) C wire-format scanner caps total `Loop` nodes per model at
  `VMAF_DNN_MAX_LOOP_NODES = 16` via a counter threaded through
  `scan_graph` / `scan_node` / `scan_attribute`; rejects with
  `-EPERM` and `first_bad="Loop"` on exceedance. C cap is
  intentionally coarser than the Python data-flow check —
  reproducing producer-map lookup would violate the ADR D39
  "no libprotobuf-c" scanner-scope constraint. 5 new Python tests
  plus 1 new C test. See
  [ADR-0171](docs/adr/0171-bounded-loop-trip-count.md). Closes
  BACKLOG T6-5b.
- **`vmaf_pre` ffmpeg filter handles 10/12-bit + optional chroma**
  (fork-local): new public libvmaf API `vmaf_dnn_session_run_plane16`
  accepts packed `uint16` LE single-plane buffers with a `bpc`
  argument (range 9..16). The ffmpeg filter now admits
  `yuv{420,422,444}p1{0,2}le` + `gray{10,12}le` pixel formats and
  dispatches the matching entrypoint by bit-depth. New
  `chroma=0|1` option (default 0 preserves luma-only back-compat)
  re-runs the same session on U/V planes at chroma-subsampled
  dimensions when set. Two new tensor helpers
  (`vmaf_tensor_from_plane16` / `vmaf_tensor_to_plane16`) with 3
  round-trip tests pinning 10-bit identity, bpc bounds, and 12-bit
  clamp behaviour. See
  [ADR-0170](docs/adr/0170-vmaf-pre-10bit-chroma.md). Closes
  BACKLOG T6-4.
- **ONNX op-allowlist admits `Loop` + `If`** (fork-local): unblocks
  MUSIQ / RAFT / small-VLM-class tiny-AI baselines that need
  control-flow ops. The wire-format scanner in
  [`onnx_scan.c`](libvmaf/src/dnn/onnx_scan.c) gains mutually-recursive
  `scan_attribute` / `scan_node` / `scan_graph` helpers that descend
  into `NodeProto.attribute` → `AttributeProto.g` / `.graphs` so a
  forbidden op cannot hide inside a `Loop.body` /
  `If.then_branch` / `If.else_branch` subgraph. Recursion depth-capped
  at `VMAF_DNN_MAX_SUBGRAPH_DEPTH = 8` as a defence-in-depth bound.
  Python `vmaf_train.op_allowlist` mirrors the recursion via a new
  `_collect_op_types` helper so the export-time check and the runtime
  load-time check stay in lockstep. `Scan` stays off the allowlist
  (variant-typed input/output binding makes static bound-checking
  impractical). The bounded-iteration guard for `Loop.M → Constant ≤
  MAX_LOOP_ITERATIONS` is **explicitly deferred** to a follow-up ADR
  (T6-5b). 4 existing tests flipped + 4 new subgraph-recursion tests
  added (2 C, 2 Python). See
  [ADR-0169](docs/adr/0169-onnx-allowlist-loop-if.md). Closes
  BACKLOG T6-5.
- **Tiny-AI Wave 1 baselines C2 + C3** (fork-local): trained ONNX
  checkpoints `nr_metric_v1.onnx` (NR MobileNet, ~19K params,
  224×224 grayscale → MOS) and `learned_filter_v1.onnx` (residual
  CNN for ffmpeg `vmaf_pre`, ~19K params, denoise-style residual
  filter) shipped in `model/tiny/`. Both trained on KoNViD-1k:
  C2 supervised on its MOS labels, C3 self-supervised on
  synthetic gaussian + JPEG degradation pairs derived from the same
  middle-frames. Op-allowlist + ORT roundtrip atol 1e-4 both pass.
  Four new scripts under `ai/scripts/` (`fetch_konvid_1k.py` /
  `extract_konvid_frames.py` / `train_konvid.py` /
  `export_tiny_models.py`); two new datamodule classes
  (`FrameMOSDataset`, `PairedFrameDataset`) in
  `vmaf_train.data.frame_dataset`. Registry schema +
  `VmafModelKind` enum extended with `kind: "filter"` to accommodate
  C3 (registry trust-root for filter models — NOT loaded by
  libvmaf's scoring path). KoNViD-1k MOS values are not
  redistributed; populated manifest stays gitignored. **C1
  (`fr_regressor_v1.onnx`) is deferred** — Netflix Public Dataset
  is access-gated (Google Drive, manual approval) and cannot be
  downloaded programmatically; tracked in
  [`docs/state.md`](docs/state.md). See
  [ADR-0168](docs/adr/0168-tinyai-konvid-baselines.md). Closes
  BACKLOG T6-1 partially (2 of 3).
- **Path-mapped doc-drift enforcement** (fork-local): closes the
  gap surfaced by the 2026-04-25 docs audit. New project hook
  [`.claude/hooks/docs-drift-warn.sh`](.claude/hooks/docs-drift-warn.sh)
  emits an informational `NOTICE` when an Edit/Write touches a
  user-discoverable surface (libvmaf headers / feature extractors /
  SIMD twins / CLI / MCP / tiny-AI CLI / ffmpeg patches) but no
  matching `docs/<topic>/` file is touched. CI counterpart in
  [`rule-enforcement.yml`](.github/workflows/rule-enforcement.yml)
  promoted from advisory to blocking + rewritten to use a
  path-mapped surface→docs check; ADR additions no longer satisfy
  it (ADRs are decisions, not user docs). Per-PR opt-out
  `no docs needed: REASON` for genuine internal-refactor PRs.
  See [ADR-0167](docs/adr/0167-doc-drift-enforcement.md).
- **Documentation refresh covering 16 recent PRs** (fork-local): in
  the same PR as ADR-0167, the audit-flagged gaps are filled —
  `vmaf_cuda_state_free()` API documented in
  [`docs/api/gpu.md`](docs/api/gpu.md); `-EAGAIN` semantics +
  `vmaf_read_pictures` monotonic-index requirement in
  [`docs/api/index.md`](docs/api/index.md); SSIMULACRA 2 + PSNR-HVS
  SIMD coverage matrix and `float_ms_ssim` <176×176 minimum
  documented in [`docs/metrics/features.md`](docs/metrics/features.md).
- **Tracked `docs/state.md` + bug-status hygiene rule** (fork-local):
  closes [Issue #20](https://github.com/lusoris/vmaf/issues/20) and
  backlog item T7-1. New tracked file [`docs/state.md`](docs/state.md)
  is the canonical in-tree register of bug status (Open / Recently
  closed / Confirmed not-affected / Deferred). New CLAUDE.md §12
  rule 13 mandates a same-PR update on every bug close / open /
  rule-out; the PR template carries a checkbox. ADRs cover decisions,
  this file covers bug status — distinct artifacts. See
  [ADR-0165](docs/adr/0165-state-md-bug-tracking.md).
- **MCP server release artifact channel — PyPI + GitHub release
  attachment + Sigstore** (fork-local): closes backlog item T7-2.
  [`supply-chain.yml`](.github/workflows/supply-chain.yml) extended
  with new `mcp-build` / `mcp-sign` / `mcp-publish-pypi` jobs.
  After this lands, `pip install vmaf-mcp` works (PyPI Trusted
  Publishing via OIDC, no token); the same wheel + sdist also
  attach to the GitHub release with a Sigstore keyless `.bundle` +
  PEP 740 attestation + SLSA L3 provenance. One-time PyPI
  Trusted-Publisher binding required (operational note in the ADR).
  See [ADR-0166](docs/adr/0166-mcp-server-release-channel.md).
- **Self-hosted GPU runner enrollment guide** (fork-local): closes
  backlog item T7-3. New
  [`docs/development/self-hosted-runner.md`](docs/development/self-hosted-runner.md)
  pins the registration steps so the next operator (or the user's
  local dev box, per popup 2026-04-25) can stand a runner up in
  ~10 minutes. The fine-grained label scheme (`gpu-cuda`,
  `gpu-intel`, `avx512`) is documented for future job targeting.
- **`motion_v2` NEON SIMD** (fork-local): aarch64 users now get a
  NEON fast path for the `motion_v2` feature. Scalar + AVX2 + AVX-512
  variants already existed; this closes the ISA-parity gap (backlog
  T3-4). The NEON impl uses arithmetic right-shift throughout
  (`vshrq_n_s64`, `vshlq_s64(v, -bpc)`) to match the scalar C `>>`
  semantics byte-for-byte — deliberately diverging from the fork's
  AVX2 variant, which uses logical `_mm256_srlv_epi64` and can
  diverge on negative-diff pixels; an AVX2 re-audit is queued as
  follow-up. Five small `static inline` helpers keep every function
  under ADR-0141's 60-line budget; zero clang-tidy warnings, no
  NOLINT. Verified bit-exact under QEMU user-mode on the Netflix
  `src01_hrc00/01_576x324` pair. See
  [ADR-0145](docs/adr/0145-motion-v2-neon-bitexact.md).

### Fixed

- **`float_ms_ssim` rejects input below 176×176 at init**
  (Netflix upstream issue
  [#1414](https://github.com/Netflix/vmaf/issues/1414)). The
  5-level 11-tap MS-SSIM pyramid walks off the kernel footprint
  at a mid-level scale for inputs below 176×176 (QCIF and
  smaller), previously producing a confusing mid-run `error:
  scale below 1x1!` + cascading `problem reading pictures` /
  `problem flushing context`. The fix checks `w < GAUSSIAN_LEN
  << (SCALES - 1)` at init and returns `-EINVAL` with a helpful
  error that names the input resolution, the required minimum
  (176×176), and the upstream issue. Minimum is derived from
  the existing filter constants so it stays in sync if those
  ever change. Visible behaviour: init now fails immediately
  instead of mid-stream; zero impact on inputs ≥176×176. New
  3-subtest reducer in `test_float_ms_ssim_min_dim.c` verified
  to fail pre-fix and pass post-fix. Closes backlog item T1-4.
  See [ADR-0153](docs/adr/0153-float-ms-ssim-min-dim-netflix-1414.md).

### Added

- **SSIMULACRA 2 regression gate** (fork-local, backlog T3-3). New
  `python/test/ssimulacra2_test.py` invokes `vmaf --feature ssimulacra2`
  on the canonical `src01_hrc00/01_576x324` pair + the small 160×90
  derived fixture and pins the per-frame + pooled output scores
  against reference values with 4-place tolerance. Catches unintended
  drift in the extractor output — complements the kernel-level SIMD
  bit-exact unit tests with an end-to-end integration gate. Closes
  ADR-0130's T3-3 deferral; backlog T3-1 + T3-3 both close now.
  See [ADR-0164](docs/adr/0164-ssimulacra2-snapshot-gate.md).

- **SSIMULACRA 2 `picture_to_linear_rgb` SIMD (AVX2 + AVX-512 + NEON)**
  (fork-local, backlog T3-1 phase 3 — closes T3-1 in full). The last
  scalar hot path in the SSIMULACRA 2 extractor is now vectorised on
  all 3 ISAs. YUV → linear RGB with BT.709/BT.601 matmul + sRGB
  EOTF, handling all pixel formats: BT.709/BT.601 × limited/full,
  any chroma subsampling ratio (420/422/444/irregular), 8-16 bpc.
  Strategy: per-lane scalar pixel reads fill an aligned scratch
  (handles all chroma ratios + bit depths uniformly); SIMD matmul +
  normalise + clamp; per-lane scalar `powf` for the sRGB EOTF branch
  (mirrors the phase-1 `cbrtf` pattern). Byte-for-byte bit-exact to
  scalar under `FLT_EVAL_METHOD == 0`. New shared header
  `ssimulacra2_simd_common.h` with `simd_plane_t` decouples SIMD
  TUs from `VmafPicture`. Five new test subtests
  (420-8bit/420-10bit/444-8bit/444-10bit/422-8bit) — 11/11 pass on
  AVX-512 host + 11/11 under `qemu-aarch64-static` (NEON).
  SSIMULACRA 2 now has **zero scalar hot paths**. See
  [ADR-0163](docs/adr/0163-ssimulacra2-ptlr-simd.md).

- **SSIMULACRA 2 FastGaussian IIR blur SIMD (AVX2 + AVX-512 + NEON)**
  (fork-local, backlog T3-1 phase 2). `blur_plane` — the single
  largest wall-clock cost in the SSIMULACRA 2 extractor (30 calls
  per frame across 5 blur-combinations × 6 scales) — now runs on
  SIMD. Horizontal pass batches N rows with `_mm256_i32gather_ps` /
  `_mm512_i32gather_ps` (AVX2 N=8, AVX-512 N=16) or 4 explicit
  `vsetq_lane_f32` calls (NEON N=4, no native gather on aarch64).
  Vertical pass uses column-SIMD loads/stores over the per-column
  IIR state arrays. Byte-for-byte bit-exact to scalar under
  `FLT_EVAL_METHOD == 0` — verified via new `test_blur` subtest
  (6/6 on AVX-512 host, 6/6 under `qemu-aarch64-static` for NEON).
  Dispatched via a new `blur_fn` function pointer in `Ssimu2State`
  assigned in `init_simd_dispatch()`. Only `picture_to_linear_rgb`
  remains scalar — deferred to follow-up. Closes backlog T3-1
  phase 2. See
  [ADR-0162](docs/adr/0162-ssimulacra2-iir-blur-simd.md).

- **SSIMULACRA 2 SIMD fast paths (AVX2 + AVX-512 + NEON)** (fork-local,
  backlog T3-1 + T3-2). Five of the eight hot kernels in the
  SSIMULACRA 2 pipeline now run on SIMD: `multiply_3plane`,
  `linear_rgb_to_xyb` (per-lane scalar `cbrtf` preserves bit-exactness),
  `downsample_2x2`, `ssim_map`, `edge_diff_map`. All 15 kernels
  (5 × 3 ISAs) produce **byte-for-byte identical output to scalar**
  under `FLT_EVAL_METHOD == 0` — verified via new unit test
  `test_ssimulacra2_simd.c` (5/5 pass on AVX-512 host; 5/5 under
  `qemu-aarch64-static` on NEON). Scalar summation order preserved
  left-to-right throughout to avoid IEEE-754 non-associativity drift
  (caught pre-merge by the bit-exact test). Reductions on
  `ssim_map` / `edge_diff_map` use the ADR-0139 per-lane `double`
  scalar tail. Runtime dispatch via function pointers in
  `Ssimu2State` with AVX-512 > AVX2 > NEON > scalar precedence.
  **Deferred to follow-up PRs**: IIR blur (`fast_gaussian_1d` / `blur_plane`,
  serial recurrence + per-column state) and `picture_to_linear_rgb`
  (`powf` EOTF) — see ADR-0161 §Alternatives. Closes backlog T3-1 + T3-2
  partially. See
  [ADR-0161](docs/adr/0161-ssimulacra2-simd-bitexact.md).

- **`psnr_hvs` NEON SIMD path** (fork-local, backlog T3-5-neon).
  Sister port to the AVX2 variant; aarch64 users now get the same
  byte-identical vectorized Xiph/Daala 8×8 integer DCT. NEON's
  4-wide `int32x4_t` means each 8-column row splits into
  `lo` (cols 0-3) + `hi` (cols 4-7); the 30-butterfly runs twice
  per DCT pass and the 8×8 transpose decomposes into four 4×4
  `vtrn1q_s32` / `vtrn2q_s32` / `vtrn1q_s64` / `vtrn2q_s64`
  stages plus a top-right ↔ bottom-left block swap. Float
  accumulators stay scalar per ADR-0139/0159 bit-exactness rule;
  `accumulate_error()` threads the outer `ret` by pointer
  (ADR-0159 summation-order lesson inherited). New unit test
  `test_psnr_hvs_neon.c`: 5/5 DCT subtests pass under
  `qemu-aarch64-static`. 576×324 8-bit Netflix golden pair
  scalar-vs-NEON diff: byte-identical `psnr_hvs_{y,cb,cr}` scores.
  1080p 10-bit pairs deferred to native-aarch64 CI (QEMU segfaults
  on heavy 10-bit threadpool allocations — known emulator limit,
  not a defect in the port). Runtime dispatch gated by
  `VMAF_ARM_CPU_FLAG_NEON`. ISA-parity matrix for psnr_hvs now
  closes: scalar + AVX2 + NEON. See
  [ADR-0160](docs/adr/0160-psnr-hvs-neon-bitexact.md).

- **`psnr_hvs` AVX2 SIMD path** (fork-local, backlog T3-5). x86_64
  users with AVX2 now get a vectorized Xiph/Daala 8×8 integer DCT
  (the hot inner kernel of psnr_hvs). Scalar + AVX2 paths are
  **byte-identical** on every Netflix golden pair — verified per-
  frame via `VMAF_CPU_MASK=0` vs default. **3.58× DCT speedup** on
  a microbenchmark (11.0 → 39.3 Mblocks/s at `-O3 -mavx2 -mfma`);
  real-world speedup scales with resolution (at 1080p × 3 planes
  the DCT is the dominant cost). Butterfly network vectorized 8
  rows in parallel via `__m256i` registers + matrix transpose
  between row and column passes. Float accumulators (means /
  variances / mask / error) kept scalar by construction for
  bit-exactness (ADR-0139 precedent). Includes new unit test
  `test_psnr_hvs_avx2.c` pinning the bit-exactness contract on 5
  reproducible inputs. NEON sister port landed as
  [ADR-0160](docs/adr/0160-psnr-hvs-neon-bitexact.md). See
  [ADR-0159](docs/adr/0159-psnr-hvs-avx2-bitexact.md).

- **`vmaf_cuda_state_free()` public API** (Netflix upstream issue
  [#1300](https://github.com/Netflix/vmaf/issues/1300)). New
  symbol in [`libvmaf/include/libvmaf/libvmaf_cuda.h`](libvmaf/include/libvmaf/libvmaf_cuda.h)
  that frees a `VmafCudaState` allocated by `vmaf_cuda_state_init()`.
  Must be called AFTER `vmaf_close()` on any VmafContext that
  imported the state. Mirrors the SYCL backend's
  `vmaf_sycl_state_free()` ownership pattern — caller allocates,
  framework imports by-value, caller frees after close. Safe to
  pass NULL. Closes the per-cycle host-memory leak where users
  had no public way to free the struct. See
  [ADR-0157](docs/adr/0157-cuda-preallocation-leak-netflix-1300.md).

### Fixed

- **CUDA preallocation memory leak** (Netflix upstream issue
  [#1300](https://github.com/Netflix/vmaf/issues/1300)). Users
  running CUDA-accelerated VMAF in init/preallocate/fetch/close
  loops saw GPU memory rise monotonically across cycles. Four
  framework-side leaks confirmed by ASan and fixed in this PR:
  (1) `VmafCudaState` heap allocation had no public free (fixed by
  the new `vmaf_cuda_state_free()` API above); (2)
  `vmaf_cuda_release()` destroyed the CUDA stream + context but
  never called `cuda_free_functions()` to release the dlopen'd
  driver function-pointer table — fixed by adding the free call
  after the existing `memset`, via a saved pointer; (3)
  `vmaf_ring_buffer_close()` locked `pthread_mutex` + freed the
  buffer but never unlocked or destroyed the mutex (POSIX UB) —
  fixed by adding `pthread_mutex_unlock` + `pthread_mutex_destroy`
  before the `free` calls; (4) adjacent cold-start leak in
  `init_with_primary_context()` where a retained CUDA primary
  context wasn't released if `cuStreamCreateWithPriority()` failed
  — fixed in the same commit. New GPU-gated reducer
  `test_cuda_preallocation_leak.c` does 10x init/preallocate/fetch
  /close cycles and reports zero framework-side leaked bytes under
  ASan (183 bytes remain in `libcuda.so.1`'s internal
  process-lifetime driver cache, matching SYCL behaviour).
  **Visible behaviour change**: every CUDA caller must now call
  `vmaf_cuda_state_free(cu_state)` AFTER `vmaf_close(vmaf)` —
  callers relying on informal `free(cu_state)` will double-free.
  Preserves ADR-0122 / ADR-0123 null-guards and ADR-0156
  `CHECK_CUDA_GOTO` cleanup paths verbatim. Closes backlog item
  T1-7. See [ADR-0157](docs/adr/0157-cuda-preallocation-leak-netflix-1300.md).

- **CUDA backend: graceful error propagation on `cuMemAlloc`
  OOM and all other CUDA failures** (Netflix upstream issue
  [#1420](https://github.com/Netflix/vmaf/issues/1420)). The
  `CHECK_CUDA` macro previously fired `assert(0)` on every
  CUDA error, which aborted the process — two concurrent
  VMAF-CUDA analyses crashed the second one immediately when
  it OOMed on `cuMemAlloc`. Wholesale refactor: replaced all
  178 `CHECK_CUDA(...)` call sites across 7 CUDA TUs
  (`common.c`, `picture_cuda.c`, `libvmaf.c`,
  `integer_motion_cuda.c`, `integer_vif_cuda.c`,
  `integer_adm_cuda.c`, `cuda_helper.cuh`) with two new
  macros — `CHECK_CUDA_GOTO(label)` (cleanup-aware) and
  `CHECK_CUDA_RETURN` (immediate-return) — that map `CUresult`
  to `-errno` via `vmaf_cuda_result_to_errno` and propagate
  the error through cleanup labels. `cuMemAlloc` OOM now
  returns `-ENOMEM`; resource exhaustion on
  `cuStreamCreate` / `cuEventCreate` returns `-EIO`;
  context / device-loss errors return `-ENODEV`; invalid
  handle / value / context errors return `-EINVAL`. Twelve
  `static` helper functions promoted from `void → int` to
  carry errors upward. New GPU-gated reducer in
  `test_cuda_buffer_alloc_oom.c` verifies `cuMemAlloc(1 TiB)`
  now returns `-ENOMEM` (was: `assert(0)`). Fixes the NDEBUG
  footgun (`assert(0)` was a no-op in release builds →
  silent continue into segfault). Preserves ADR-0122 /
  ADR-0123 null-guards on public entry points verbatim.
  Closes backlog item T1-6. See
  [ADR-0156](docs/adr/0156-cuda-graceful-error-propagation-netflix-1420.md).

### Documented (not fixed)

- **ADM `i4_adm_cm` int32 rounding overflow** (Netflix upstream
  issue [#955](https://github.com/Netflix/vmaf/issues/955)) is
  deliberately preserved. `add_bef_shift_flt[idx] = (1u <<
  (shift_flt[idx] - 1))` in `libvmaf/src/feature/integer_adm.c`
  scales 1–3 overflows `int32_t` (`1u << 31 = 0x80000000` wraps
  to `-2147483648`), so every `(prod + add_bef_shift) >> 32`
  subtracts 2^31 instead of adding it — ADM scales 1–3 biased
  low by ≈1 LSB per summed term. The buggy arithmetic is encoded
  in the Netflix golden assertions (project hard rule #1 /
  [ADR-0024](docs/adr/0024-netflix-golden-preserved.md)); fixing
  it unilaterally would diverge from every published VMAF number
  calibrated on these outputs. In-file warning comments, a
  rebase-notes invariant, and `AGENTS.md` pin the decision.
  Closes backlog item T1-8 as "verified present, deliberately
  preserved". See
  [ADR-0155](docs/adr/0155-adm-i4-rounding-deferred-netflix-955.md).

### Changed

- **`vmaf_score_pooled` returns `-EAGAIN` for pending features**
  (Netflix upstream issue
  [#755](https://github.com/Netflix/vmaf/issues/755)). Several
  extractors (integer_motion's motion2/motion3) write frame N's
  score retroactively when frame N+1 is extracted — or on flush
  for the tail. Previously `vmaf_score_pooled(vmaf, ..., i, i)`
  called immediately after `vmaf_read_pictures(vmaf, ref, dist,
  i)` returned `-EINVAL`, indistinguishable from programmer
  error. Now: `-EAGAIN` for the transient "valid but not yet
  written" case; `-EINVAL` stays reserved for genuine misuse
  (bad pointer, out-of-range, feature-name typo). Inline
  `vmaf_feature_vector_get_score` previously returned a literal
  `-1` for both cases; now splits the same way. **Visible
  behaviour change** for callers that want to distinguish
  transient from fatal — they can now branch on `-EAGAIN` and
  retry after one more read or after flush. Callers that treat
  any non-zero as fatal are unchanged. Drive-by: reserved
  `__VMAF_FEATURE_COLLECTOR_H__` header guard renamed to
  `VMAF_FEATURE_COLLECTOR_INCLUDED` (ADR-0141). 4-subtest
  reducer in `test_score_pooled_eagain.c` verified to fail
  pre-fix and pass post-fix. Closes backlog item T1-1. See
  [ADR-0154](docs/adr/0154-score-pooled-eagain-netflix-755.md).

- **`vmaf_read_pictures` now rejects non-monotonic indices with
  `-EINVAL`** (Netflix upstream issue
  [#910](https://github.com/Netflix/vmaf/issues/910)). The
  `integer_motion` / motion2 / motion3 extractors keep sliding-
  window state keyed by `index % N`, so submitting frames out of
  order or with duplicate indices silently corrupts their
  ring-buffers. The reported symptom was a missing
  `integer_motion2_score` on the last frame whenever submission
  order didn't match frame order. The fix is a monotonic-index
  guard at the API boundary (new `last_index` + `have_last_index`
  fields on `VmafContext`, checked inside the existing
  `read_pictures_validate_and_prep` helper from ADR-0146): strictly
  increasing indices are accepted (gaps fine); duplicates and
  regressions return `-EINVAL`. **Visible behaviour change**:
  duplicates / out-of-order submissions that previously produced
  silent-wrong-answer now fail with `-EINVAL` — well-defined
  rejection replaces ill-defined corruption. Zero impact on
  in-tree callers (vmaf CLI + test suite already iterate strictly
  increasing); downstream integrations that deliberately submit
  non-monotonic indices need to either track the next-index
  themselves or reset the context. 3-subtest reducer in
  `test_read_pictures_monotonic.c` verified to fail pre-fix and
  pass post-fix. Closes backlog item T1-2. See
  [ADR-0152](docs/adr/0152-vmaf-read-pictures-monotonic-index.md).

### Added

- **i686 (32-bit x86) build-only CI job** (reproduces Netflix
  upstream issue [#1481](https://github.com/Netflix/vmaf/issues/1481)).
  New matrix row in `.github/workflows/libvmaf-build-matrix.yml`
  (`Build — Ubuntu i686 gcc (CPU, no-asm)`) invokes
  `meson setup libvmaf libvmaf/build --cross-file=build-aux/i686-linux-gnu.ini -Denable_asm=false`,
  pinning the workaround documented in upstream's bug report.
  New cross-file `build-aux/i686-linux-gnu.ini` (gcc + `-m32`,
  `cpu_family = 'x86'`, `cpu = 'i686'`) + new install-deps step
  installing `gcc-multilib` + `g++-multilib`. Test + tox steps
  skipped for the i686 leg because meson marks cross-built tests
  as `SKIP 77` (the host can run i686 binaries natively but meson
  doesn't know that). Fixing the underlying AVX2
  `_mm256_extract_epi64` compile failure (24 call sites in
  `adm_avx2.c`) is **explicitly out of scope** — this entry adds
  the CI gate only. Closes backlog item T4-8. See
  [ADR-0151](docs/adr/0151-i686-ci-netflix-1481.md).

- **Windows MSYS2/MinGW CUDA build support** (port of Netflix
  upstream PR [#1472](https://github.com/Netflix/vmaf/pull/1472),
  birkdev, 2026-03-16, OPEN). Enables
  `-Denable_cuda=true -Denable_nvcc=true` on Windows with MSYS2 +
  MinGW-GCC host compiler + MSVC Build Tools + CUDA toolkit.
  Source-portability guards in CUDA headers + `.cu` files: drop
  `<pthread.h>` from `cuda/common.h`; DEVICE_CODE guards on
  `<ffnvcodec/*>` vs `<cuda.h>` in `cuda_helper.cuh` +
  `picture.h`; `#ifndef DEVICE_CODE` around `feature_collector.h`
  in 5 ADM `.cu` files. Meson build plumbing: `vswhere`-based
  `cl.exe` discovery (without adding it to PATH, which would
  break MinGW-GCC CPU build), Windows SDK + MSVC include path
  injection via `-I` flags to nvcc, CUDA version detection via
  `nvcc --version` (replaces `meson.get_compiler('cuda')` which
  needs MSVC as default C compiler). Fork carve-outs: keep
  positional (not `#ifndef __CUDACC__`) initializers in
  `integer_adm.h`; keep `pthread_dependency` on `cuda_static_lib`
  because `ring_buffer.c` still uses pthread directly; merge
  fork's ADR-0122 gencode coverage block with upstream's new
  nvcc-detect block. Drive-by: rename reserved `__VMAF_SRC_*_H__`
  header guards to `VMAF_SRC_*_INCLUDED` per ADR-0141. Linux
  CPU build 32/32 + Linux CUDA build 35/35 pass; Windows CUDA
  build not yet CI-validated (tracked as T7-3 — self-hosted
  Windows+CUDA runner enrollment). Closes backlog item T4-2.
  See [ADR-0150](docs/adr/0150-port-netflix-1472-cuda-windows.md).

### Fixed

- **FIFO-mode workfile/procfile opens no longer race-hang on slow
  systems** (port of Netflix upstream PR
  [#1376](https://github.com/Netflix/vmaf/pull/1376)). The Python
  harness under `python/vmaf/core/executor.py` +
  `python/vmaf/core/raw_extractor.py` previously waited for child
  processes to create named pipes via a 1-second `os.path.exists()`
  polling loop, which could time out on loaded CI / virtualised
  hosts. Replaced with `multiprocessing.Semaphore(0)` signalled
  by the child processes after `os.mkfifo(...)`; parent acquires
  with 5-second soft-timeout warning then blocks indefinitely.
  Applied to both the base `Executor` class and the
  `ExternalVmafExecutor`-style subclass. Fork carve-outs:
  upstream's `__version__ = "3.0.0" → "4.0.0"` bump is **not**
  applied (fork tracks its own versioning per ADR-0025); unused
  `from time import sleep` imports removed per ADR-0141.
  Closes backlog item T4-7. See
  [ADR-0149](docs/adr/0149-port-netflix-1376-fifo-semaphore.md).

### Changed

- **IQA reserved-identifier rename + touched-file lint cascade
  cleanup** (refactor, fork-local). Rename every `_iqa_*` /
  `struct _kernel` / `_ssim_int` / `_map_reduce` / `_map` /
  `_reduce` / `_context` / `_ms_ssim_map` / `_ssim_map` /
  `_ms_ssim_reduce` / `_ssim_reduce` / `_alloc_buffers` /
  `_free_buffers` symbol and the underscore-prefixed header
  guards (`_CONVOLVE_H_`, `_DECIMATE_H_`, `_SSIM_TOOLS_H_`,
  `__VMAF_MS_SSIM_DECIMATE_H__`) to their non-reserved
  spellings across the IQA tree (21 files). Sweeps the
  ADR-0141 touched-file lint cascade that surfaced
  (~40 pre-existing warnings across `ssim.c`, `ms_ssim.c`,
  `integer_ssim.c`, `iqa/*.{c,h}`, `convolve_*.{c,h}`,
  `test_iqa_convolve.c`): `static` / cross-TU NOLINT for
  `misc-use-internal-linkage`, `size_t` casts for
  `bugprone-implicit-widening-of-multiplication-result`,
  multi-decl splits, function-size refactors of `calc_ssim` /
  `compute_ssim` / `compute_ms_ssim` / `run_gauss_tests` via
  small named `static` helpers, `(void)` casts for unused
  feature-extractor lifecycle parameters, and scoped
  NOLINTBEGIN/END for `clang-analyzer-security.ArrayBound`
  false positives on the kernel-offset clamps and for
  `clang-analyzer-unix.Malloc` on test-helper allocations
  that leak by design at process exit. Bit-identical VMAF
  score on Netflix golden pair `src01_hrc00/01_576x324`
  (scalar vs SIMD, with `--feature float_ssim --feature
  float_ms_ssim` and the full `vmaf_v0.6.1` model). Closes
  backlog item T7-6. See
  [ADR-0148](docs/adr/0148-iqa-rename-and-cleanup.md).

- **Thread-pool job-object recycling** (perf, fork-local port of
  Netflix upstream PR [#1464](https://github.com/Netflix/vmaf/pull/1464),
  thread-pool portion only). `libvmaf/src/thread_pool.c` now recycles
  `VmafThreadPoolJob` slots via a mutex-protected free list rather
  than `malloc`/`free` on every enqueue, and stores payloads ≤ 64
  bytes inline in the job struct (`char inline_data[64]`) so the
  common-case enqueue path avoids a second allocation entirely.
  Adapted to the fork's `void (*func)(void *data, void **thread_data)`
  signature and per-worker `VmafThreadPoolWorker` data path (which
  upstream lacks). **~1.8–2.6× enqueue throughput** on a 500 000-job
  4-thread micro-benchmark; bit-identical VMAF scores between
  `--threads 4` and serial, and between `VMAF_CPU_MASK=0` and `=255`
  under `--threads 4`. Closes the thread-pool half of backlog T3-6
  (the AVX2 PSNR half was already covered by fork commit `81fcd42e`).
  See [ADR-0147](docs/adr/0147-thread-pool-job-pool.md).

- **Function-size NOLINT sweep** — refactored every
  `readability-function-size` NOLINT suppression in `libvmaf/src/` (20
  sites across 12 files: `dict.c`, `picture.c`, `picture_pool.c`,
  `predict.c`, `libvmaf.c`, `output.c`, `read_json_model.c`,
  `feature/feature_extractor.c`, `feature/feature_collector.c`,
  `feature/iqa/convolve.c`, `feature/iqa/ssim_tools.c`,
  `feature/x86/vif_statistic_avx2.c`) into small named `static`
  helpers. IQA / SIMD files use `static inline` helpers threaded
  through an explicit `struct vif_simd8_lane` to preserve the
  ADR-0138 / ADR-0139 bit-exactness invariants (per-lane scalar-float
  reduction, single-rounded float-mul → widen → double-add).
  Netflix-golden-pair VMAF score remains bit-identical between
  `VMAF_CPU_MASK=0` and `VMAF_CPU_MASK=255`. Zero new NOLINTs
  introduced. Drive-by fixes: TU-static `_calc_scale` →
  `iqa_calc_scale` for `bugprone-reserved-identifier`; tightened
  `calloc(w * h, ...)` widening; separated multi-declaration forms;
  `model_collection_parse_loop` now writes directly through
  `cfg_name` instead of the aliased `c->name` (drops the last
  `readability-non-const-parameter` NOLINT). See
  [ADR-0146](docs/adr/0146-nolint-sweep-function-size.md).

- **VIF AVX2 convolve: generalised for arbitrary filter widths** (port of
  Netflix upstream [`f3a628b4`](https://github.com/Netflix/vmaf/commit/f3a628b4),
  Kyle Swanson, 2026-04-21). `libvmaf/src/feature/common/convolution_avx.c`
  drops from 2,747 LoC of branch-unrolled kernels specialised to
  `fwidth ∈ {3, 5, 9, 17}` down to 247 LoC of a single parametric 1-D
  scanline pair. New `MAX_FWIDTH_AVX_CONV` ceiling in `convolution.h`
  lets the VIF AVX2 dispatch in `vif_tools.c` drop its hard-coded
  fwidth whitelist. Fork cleanup per ADR-0141: four helpers now
  `static`, strides widened to `ptrdiff_t` to eliminate
  `bugprone-implicit-widening-of-multiplication-result` on every
  pointer-offset site. Paired with a 10× loosening of the Netflix
  golden tolerance on two full-VMAF assertions
  (`VMAF_score`, `VMAFEXEC_score`: `places=2` → `places=1`),
  matching Netflix's own upstream test update. The generalised
  kernel's accumulation order differs at ULP scale vs the
  specialised ones; drift is orders of magnitude below perceptual
  discriminability. See
  [ADR-0143](docs/adr/0143-port-netflix-f3a628b4-generalized-avx-convolve.md).

### Added

- **VIF: configurable `vif_sigma_nsq` feature parameter**: port of
  Netflix upstream [`18e8f1c5`](https://github.com/Netflix/vmaf/commit/18e8f1c5)
  (Kyle Swanson, 2026-04-20) promoting VIF's hard-coded neural-noise
  variance `static const float sigma_nsq = 2` into a runtime-configurable
  double parameter `vif_sigma_nsq` (range `[0.0, 5.0]`, alias `snsq`,
  default `2.0`). Threaded through `compute_vif` → `vif_statistic_s` and
  the fork-local `vif_statistic_s_avx2` AVX2 variant (which upstream does
  not ship; its signature was extended in lockstep so both paths agree on
  the new 14-argument contract). Default-path scores bit-identical to
  pre-port master. Use via CLI:
  `vmaf --feature float_vif:snsq=2.5 ...` or per-model. See
  [ADR-0142](docs/adr/0142-port-netflix-18e8f1c5-vif-sigma-nsq.md).
- **Governance — Q2 2026 modernization ADRs**: four Proposed ADRs +
  four research digests scoping the next modernization workstreams
  (no implementation yet):
  - [ADR-0126](docs/adr/0126-ssimulacra2-extractor.md) /
    [Research-0003](docs/research/0003-ssimulacra2-port-sourcing.md):
    SSIMULACRA 2 feature extractor (port libjxl C++ reference).
  - [ADR-0127](docs/adr/0127-vulkan-compute-backend.md) /
    [Research-0004](docs/research/0004-vulkan-backend-design.md):
    Vulkan compute backend (volk + GLSL→SPIR-V + VMA, VIF
    pathfinder).
  - [ADR-0128](docs/adr/0128-embedded-mcp-in-libvmaf.md) /
    [Research-0005](docs/research/0005-embedded-mcp-transport.md):
    Embedded MCP server in libvmaf (SSE + UDS + stdio, flag-gated).
  - [ADR-0129](docs/adr/0129-tinyai-ptq-quantization.md) /
    [Research-0006](docs/research/0006-tinyai-ptq-accuracy-targets.md):
    Tiny-AI PTQ int8 (static + dynamic + QAT per-model via
    `model/registry.json`).
- **SIMD DX framework — `simd_dx.h` + upgraded `/add-simd-path` skill**:
  fork-internal header
  ([`libvmaf/src/feature/simd_dx.h`](libvmaf/src/feature/simd_dx.h))
  that codifies the ADR-0138 (widen-then-add) and ADR-0139 (per-lane
  scalar-double reduce) patterns as ISA-suffixed macros
  (`SIMD_WIDEN_ADD_F32_F64_AVX2_4L` / `_AVX512_8L` / `_NEON_4L`,
  `SIMD_ALIGNED_F32_BUF_*`, `SIMD_LANES_*`). Zero runtime overhead —
  each macro documents its scalar C equivalent and is guarded by the
  matching `__AVX2__` / `__AVX512F__` / `__ARM_NEON` ifdef. The
  `/add-simd-path` skill
  ([`.claude/skills/add-simd-path/SKILL.md`](.claude/skills/add-simd-path/SKILL.md))
  gained `--kernel-spec=widen-add-f32-f64|per-lane-scalar-double|none`,
  `--lanes=N`, and `--tail=scalar|masked` flags so new SIMD TUs
  scaffold from a short declaration instead of a cold copy-paste.
  Demonstrated on two real kernels in the same PR: a new bit-exact
  `iqa_convolve_neon`
  ([`libvmaf/src/feature/arm64/convolve_neon.c`](libvmaf/src/feature/arm64/convolve_neon.c))
  and a bit-exactness fix for `ssim_accumulate_neon` that mirrors the
  ADR-0139 x86 fix. Together they complete the SSIM / MS-SSIM SIMD
  coverage on aarch64. See
  [ADR-0140](docs/adr/0140-simd-dx-framework.md) +
  [research digest 0013](docs/research/0013-simd-dx-framework.md).
- **aarch64 cross-compile lane**:
  [`build-aux/aarch64-linux-gnu.ini`](build-aux/aarch64-linux-gnu.ini)
  meson cross-file for `aarch64-linux-gnu-gcc` +
  `qemu-aarch64-static`. The `test_iqa_convolve` meson target now
  covers `arm64` / `aarch64` alongside `x86_64` / `x86` so future NEON
  convolve changes gate on the same bit-exactness contract as the x86
  variants.
- **I18N / thread-safety**: `thread_locale.h/.c` cross-platform thread-local
  locale abstraction ported from upstream PR
  [Netflix/vmaf#1430](https://github.com/Netflix/vmaf/pull/1430) (Diego Nieto,
  Fluendo). `vmaf_write_output_{xml,json,csv,sub}`, `svm_save_model`,
  `vmaf_read_json_model`, and both SVM model parsers now switch the calling
  thread's locale to `"C"` for numeric I/O instead of using the
  process-global `setlocale` bracket. POSIX.1-2008 `uselocale` +
  `newlocale(LC_ALL_MASK)` on Linux/macOS/BSD; `_configthreadlocale` +
  per-thread `setlocale` on Windows; graceful no-op fallback elsewhere.
  Fixes a latent data-race under multi-threaded hosts (ffmpeg filter graphs
  with multiple VMAF instances, MCP server worker pools) where one thread's
  `setlocale(LC_ALL, "C")` bracket would clobber another thread's active
  locale mid-call. See
  [ADR-0137](docs/adr/0137-thread-local-locale-for-numeric-io.md).
- **Public API**: `vmaf_model_version_next(prev, &version)` iterator for
  enumerating the built-in VMAF model versions compiled into the
  library. Opaque-handle cursor — NULL to start, NULL-return to stop.
  Ports [Netflix#1424](https://github.com/Netflix/vmaf/pull/1424) with
  three correctness corrections (NULL-pointer arithmetic UB,
  off-by-one returning the sentinel, const-qualifier mismatches in the
  test); see [ADR-0135](docs/adr/0135-port-netflix-1424-expose-builtin-model-versions.md).
- **Build**: libvmaf now exports `libvmaf_dep` via `declare_dependency`
  and registers an `override_dependency('libvmaf', ...)` in
  `libvmaf/src/meson.build`, so the fork is consumable as a meson
  subproject with the standard `dependency('libvmaf')` idiom. Ports
  [Netflix#1451](https://github.com/Netflix/vmaf/pull/1451); see
  [ADR-0134](docs/adr/0134-port-netflix-1451-meson-declare-dependency.md).
- **Metric**: SSIMULACRA 2 scalar feature extractor
  ([`libvmaf/src/feature/ssimulacra2.c`](libvmaf/src/feature/ssimulacra2.c))
  — port of libjxl's perceptual similarity metric on top of the fork's
  YUV pipeline. Ingests YUV 4:2:0/4:2:2/4:4:4 at 8/10/12 bpc with a
  configurable YUV→RGB matrix (`yuv_matrix` option, BT.709 limited
  default), converts through linear RGB → XYB → 6-scale pyramid with
  SSIMMap + EdgeDiffMap + canonical 108-weight polynomial pool.
  Pyramid blur is a bit-close C port of libjxl's `FastGaussian`
  3-pole recursive IIR (`lib/jxl/gauss_blur.cc`,
  Charalampidis 2016 truncated-cosine approximation, k={1,3,5},
  zero-pad boundaries). Registered as feature `ssimulacra2` — one
  scalar per frame in `[0, 100]`, identity inputs return exactly
  `100.000000`. Scalar only; AVX2/AVX-512/NEON follow-ups are
  separate PRs. See
  [ADR-0130](docs/adr/0130-ssimulacra2-scalar-implementation.md) +
  [Research-0007](docs/research/0007-ssimulacra2-scalar-port.md).
- **CLI**: `--precision $spec` flag for score output formatting.
  - `N` (1..17) → `printf "%.<N>g"`
  - `max` / `full` → `"%.17g"` (round-trip lossless, opt-in)
  - `legacy` → `"%.6f"` (synonym for the default)
  - default (no flag) → `"%.6f"` (Netflix-compatible per ADR-0119;
    supersedes ADR-0006's original `%.17g` default)
- **Public API**: `vmaf_write_output_with_format()` accepts a `score_format`
  string; old `vmaf_write_output()` routes through the new function with
  `"%.6f"` default.
- **GPU backends**: SYCL/oneAPI backend (Lusoris + Claude); CUDA backend
  optimizations (decoupled buffer elimination, VIF rd_stride, ADM inline
  decouple).
- **Numerical correctness**: float ADM `sum_cube` and `csf_den_scale` use
  double-precision accumulation in scalar/AVX2/AVX512 paths to eliminate
  ~8e-5 drift between scalar and SIMD reductions.
- **MS-SSIM SIMD**: separable scalar-FMA decimate with AVX2 (8-wide),
  AVX-512 (16-wide), and NEON (4-wide) variants for the 9-tap 9/7
  biorthogonal wavelet LPF used by the MS-SSIM scale pyramid. Per-lane
  `_mm{256,512}_fmadd_ps` (x86) / `vfmaq_n_f32` (aarch64) with
  broadcast coefficients produces output byte-identical to the scalar
  reference; stride-2 horizontal deinterleave via
  `_mm256_shuffle_ps`+`_mm256_permute4x64_pd` (AVX2),
  `_mm512_permutex2var_ps` (AVX-512), and `vld2q_f32` (NEON). Runtime
  dispatch prefers AVX-512 > AVX2 > scalar on x86 and NEON > scalar on
  arm64. Netflix MS-SSIM golden passes at places=4 through every
  dispatched path; 10 synthetic `memcmp` cases (1x1 border, odd
  dimensions, 1920x1080) verify strict byte-equality in
  [`libvmaf/test/test_ms_ssim_decimate.c`](libvmaf/test/test_ms_ssim_decimate.c).
  See [ADR-0125](docs/adr/0125-ms-ssim-decimate-simd.md).
- **AI-agent scaffolding**: `.claude/` directory with 7 specialized review
  agents (c-, cuda-, sycl-, vulkan-, simd-, meson-reviewer, perf-profiler),
  18 task skills, hooks for unsafe-bash blocking and auto-format,
  `CLAUDE.md` + `AGENTS.md` onboarding, `docs/principles.md` (Power-of-10 +
  JPL-C-STD + CERT + MISRA).
- **Quality gates**: GitHub Actions workflows for CI (Netflix golden gate
  D24, sanitizers, cross-backend ULP), lint (clang-tidy, cppcheck,
  pre-commit), security (semgrep, CodeQL, gitleaks, dependency-review),
  supply-chain (SBOM, Sigstore keyless signing, SLSA L3 provenance).
- **Tiny AI**: nightly `bisect-model-quality` workflow
  ([`.github/workflows/nightly-bisect.yml`](.github/workflows/nightly-bisect.yml))
  runs `vmaf-train bisect-model-quality` against a deterministic
  synthetic placeholder cache
  ([`ai/testdata/bisect/`](ai/testdata/bisect/),
  reproducible from
  [`ai/scripts/build_bisect_cache.py`](ai/scripts/build_bisect_cache.py))
  and posts the verdict + per-step PLCC/SROCC/RMSE table to sticky
  tracker issue #40. Real DMOS-aligned cache swaps in via a follow-up;
  see [ADR-0109](docs/adr/0109-nightly-bisect-model-quality.md) +
  [Research-0001](docs/research/0001-bisect-model-quality-cache.md).
  Closes #4.
- **CI**: three DNN-enabled matrix legs in
  [`.github/workflows/libvmaf-build-matrix.yml`](.github/workflows/libvmaf-build-matrix.yml)
  — `Build — Ubuntu gcc (CPU) + DNN`, `Build — Ubuntu clang (CPU) + DNN`,
  `Build — macOS clang (CPU) + DNN`. Each leg installs ONNX Runtime
  (Linux: MS tarball pinned to 1.22.0; macOS: Homebrew) and runs the
  meson `dnn` test suite plus full `ninja test`. The two Linux legs
  are pinned to required status checks on `master`; the macOS leg
  stays `experimental: true` because Homebrew ORT floats. See
  [ADR-0120](docs/adr/0120-ai-enabled-ci-matrix-legs.md) +
  [`docs/rebase-notes.md` entry 0021](docs/rebase-notes.md).
- **CI**: two Windows GPU build-only matrix legs in
  [`.github/workflows/libvmaf-build-matrix.yml`](.github/workflows/libvmaf-build-matrix.yml)
  — `Build — Windows MSVC + CUDA (build only)` and
  `Build — Windows MSVC + oneAPI SYCL (build only)`. Both gate the
  MSVC build-portability of the CUDA host code and SYCL `vmaf_sycl_*`
  C-API entry points, respectively. No test step (windows-latest has
  no GPU). Both legs are pinned to required status checks on `master`.
  See [ADR-0121](docs/adr/0121-windows-gpu-build-only-legs.md) +
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: Win32 `pthread.h` compat shim at
  [`libvmaf/src/compat/win32/pthread.h`](libvmaf/src/compat/win32/pthread.h)
  — header-only, maps the in-use pthread subset (mutex / cond / thread
  create+join+detach + `PTHREAD_MUTEX_INITIALIZER` /
  `PTHREAD_COND_INITIALIZER`) onto Win32 SRWLOCK + CONDITION_VARIABLE +
  `_beginthreadex`. Wired in via a new `pthread_dependency` in
  `libvmaf/meson.build`, gated on `cc.check_header('pthread.h')`
  failing — POSIX and MinGW (winpthreads) builds are untouched. Lets
  the Windows MSVC GPU legs from ADR-0121 actually compile the libvmaf
  core (~14 TUs `#include <pthread.h>` unconditionally). Pattern
  mirrors the long-standing `compat/gcc/stdatomic.h` shim. nvcc fatbin
  and icpx SYCL `custom_target`s additionally thread the shim include
  path through `cuda_extra_includes` / `sycl_inc_flags` on Windows
  (custom targets bypass meson's `dependencies:` plumbing).
- **Build**: SYCL Windows host-arg handling in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build) — `icpx-cl`
  on Windows targets `x86_64-pc-windows-msvc` and rejects `-fPIC`.
  `sycl_common_args` / `sycl_feature_args` now route the flag through
  `sycl_pic_arg = host_machine.system() != 'windows' ? ['-fPIC'] : []`
  instead of hard-coding it. PIC is the default for Windows DLLs, so
  dropping the flag is the correct build-system fix, not a workaround.
- **Build**: SYCL Windows source portability — four MSVC C++
  blockers fixed so `icpx-cl` compiles the SYCL TUs.
  (1) [`libvmaf/src/ref.h`](libvmaf/src/ref.h) +
  [`libvmaf/src/feature/feature_extractor.h`](libvmaf/src/feature/feature_extractor.h)
  (UPSTREAM) gained an `#if defined(__cplusplus) && defined(_MSC_VER)`
  branch that pulls `atomic_int` via `using std::atomic_int;` —
  MSVC's `<stdatomic.h>` only surfaces the C11 typedefs in
  `namespace std::` under C++, while gcc/clang expose them globally
  via a GNU extension. POSIX paths fall through to the original
  `<stdatomic.h>` line; ABI unchanged. (2)
  [`libvmaf/src/sycl/d3d11_import.cpp`](libvmaf/src/sycl/d3d11_import.cpp)
  switched `<libvmaf/log.h>` (non-existent) to `"log.h"` (the actual
  internal header). (3)
  [`libvmaf/src/sycl/dmabuf_import.cpp`](libvmaf/src/sycl/dmabuf_import.cpp)
  moved `<unistd.h>` inside `#if HAVE_SYCL_DMABUF` — POSIX `close()`
  is only used in the VA-API path, so non-DMA-BUF hosts (Windows
  MSVC, macOS) no longer fail with `'unistd.h' file not found`. (4)
  [`libvmaf/src/sycl/common.cpp`](libvmaf/src/sycl/common.cpp)
  replaced POSIX `clock_gettime(CLOCK_MONOTONIC)` with
  `std::chrono::steady_clock` — guaranteed monotonic by the C++
  standard and portable on every supported host. All four preserve
  POSIX/Linux behaviour bit-identically. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: CUDA Windows source portability — fifth MSVC blocker
  fixed on the CUDA leg's CPU SIMD compile path.
  [`libvmaf/src/feature/x86/motion_avx2.c`](libvmaf/src/feature/x86/motion_avx2.c)
  (UPSTREAM) line 529 indexed an `__m256i` directly
  (`final_accum[0] + ... + final_accum[3]`) — gcc/clang allow this
  via the GNU vector extension, MSVC rejects it with `C2088:
  built-in operator '[' cannot be applied to an operand of type
  '__m256i'`. Replaced with four `_mm256_extract_epi64` calls,
  summed — bit-exact lane sum on every compiler. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: x86 SIMD Windows source portability — sweep that
  finishes the MSVC compile of the libvmaf CPU SIMD layer.
  Round-19 surfaced the same vector-extension pattern at 19 more
  call sites plus 6 GCC-style `(__m256i)x` casts.
  [`libvmaf/src/feature/x86/adm_avx2.c`](libvmaf/src/feature/x86/adm_avx2.c)
  (UPSTREAM) had 6 lines using
  `(__m256i)(_mm256_cmp_ps(...))` casts (replaced with
  `_mm256_castps_si256(...)`) and 12 sites of `__m128i[N]`
  lane-extract reductions (replaced with `_mm_extract_epi64`).
  [`libvmaf/src/feature/x86/adm_avx512.c`](libvmaf/src/feature/x86/adm_avx512.c)
  (UPSTREAM) had 6 sister lane-extract reductions on the
  AVX-512 paths.
  [`libvmaf/src/feature/x86/motion_avx512.c`](libvmaf/src/feature/x86/motion_avx512.c)
  (UPSTREAM, ported from PR #1486) had one final lane-extract
  reduction. All 19 + 6 fixes are bit-exact rewrites — gcc/clang
  emit identical vextract+padd sequences either way.
  Additionally
  [`libvmaf/src/sycl/d3d11_import.cpp`](libvmaf/src/sycl/d3d11_import.cpp)
  switched from C-style COBJMACROS helpers
  (`ID3D11Device_CreateTexture2D`, etc.) to C++ method-call syntax
  (`device->CreateTexture2D`) because d3d11.h gates COBJMACROS
  behind `!defined(__cplusplus)` and the TU compiles as C++
  under icpx-cl. ABI-equivalent. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: x86 SIMD alignment specifier — round-20 swap from
  GCC trailing `__attribute__((aligned(N)))` to C11-standard
  leading `_Alignas(N)` across 17 scratch-buffer sites in
  `vif_statistic_avx2.c` (UPSTREAM), `ansnr_avx{2,512}.c`
  (UPSTREAM), `float_adm_avx{2,512}.c` (UPSTREAM),
  `float_psnr_avx{2,512}.c` (UPSTREAM) and `ssim_avx{2,512}.c`
  (UPSTREAM). Same alignment guarantee, MSVC-portable
  (`/std:c11`). The pre-existing portable `ALIGNED(x)` macro in
  `vif_avx{2,512}.c` was already MSVC-clean and remains untouched.
- **Build**: `mkdirp` Windows portability —
  [`libvmaf/src/feature/mkdirp.c`](libvmaf/src/feature/mkdirp.c)
  and
  [`libvmaf/src/feature/mkdirp.h`](libvmaf/src/feature/mkdirp.h)
  (third-party MIT-licensed micro-library) gate `<unistd.h>` to
  non-Windows, add `<direct.h>` + `_mkdir` on MSVC, and provide a
  local `mode_t` typedef (MSVC's `<sys/types.h>` doesn't declare
  it). The `mode` argument is silently ignored on the Windows
  path — same behaviour as before for POSIX callers. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: round-21 MSVC mop-up —
  [`libvmaf/src/feature/x86/adm_avx512.c`](libvmaf/src/feature/x86/adm_avx512.c)
  (UPSTREAM) adds six more `_mm_extract_epi64` rewrites at lines
  2128 / 2135 / 2142 / 2589 / 2595 / 2601 that the round-19 sweep
  missed (bit-exact).
  [`libvmaf/src/log.c`](libvmaf/src/log.c) (UPSTREAM) gates
  `<unistd.h>` to non-Windows and pulls `_isatty` / `_fileno` from
  `<io.h>` on MSVC via macro redirection; the single `isatty(fileno
  (stderr))` call site compiles unchanged on every platform.
  See [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **CI**: `.github/workflows/lint-and-format.yml` pre-commit job
  now checks out with `lfs: true`. Without it `model/tiny/*.onnx`
  lands as LFS pointer stubs and pre-commit's "changes made by
  hooks" reporter flags the stubs as pre-commit-induced
  modifications against HEAD's resolved blobs, failing the job
  even though no hook touched them. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: round-21e MSVC mop-up — the Windows MSVC legs now
  build the full tree (CLI tools, unit tests, `libvmaf.dll`)
  instead of the earlier short cut of skipping tools / tests.
  Source changes:
  (i) eight C99 variable-length arrays converted to compile-time
  constants or heap allocations —
  [`libvmaf/src/predict.c:385,453`](libvmaf/src/predict.c),
  [`libvmaf/src/libvmaf.c:1741`](libvmaf/src/libvmaf.c),
  [`libvmaf/src/read_json_model.c:517,520`](libvmaf/src/read_json_model.c),
  [`libvmaf/test/test_feature_extractor.c:56`](libvmaf/test/test_feature_extractor.c),
  [`libvmaf/test/test_cambi.c:254`](libvmaf/test/test_cambi.c),
  [`libvmaf/test/test_pic_preallocation.c:382,506`](libvmaf/test/test_pic_preallocation.c);
  (ii) fork-added POSIX/GNU `getopt_long` shim at
  [`libvmaf/tools/compat/win32/`](libvmaf/tools/compat/win32/)
  (header + ~260-line companion source) declared via a single
  `getopt_dependency` in
  [`libvmaf/meson.build`](libvmaf/meson.build) that
  auto-propagates the .c into the `vmaf` CLI and
  `test_cli_parse`;
  (iii) `pthread_dependency` threaded through the eleven test
  targets in
  [`libvmaf/test/meson.build`](libvmaf/test/meson.build)
  that transitively include `<pthread.h>` via
  `feature_collector.h`;
  (iv) `<unistd.h>` → `<io.h>` redirection
  (`isatty`/`fileno` → `_isatty`/`_fileno`) added to
  [`libvmaf/tools/vmaf.c`](libvmaf/tools/vmaf.c);
  (v) `<unistd.h>` → `<windows.h>` + `Sleep` macros
  added to
  [`libvmaf/test/test_ring_buffer.c`](libvmaf/test/test_ring_buffer.c)
  and
  [`libvmaf/test/test_pic_preallocation.c`](libvmaf/test/test_pic_preallocation.c)
  for `usleep` / `sleep`;
  (vi) `__builtin_clz` / `__builtin_clzll` MSVC fallback via
  `__lzcnt` / `__lzcnt64` extracted into
  [`libvmaf/src/feature/compat_builtin.h`](libvmaf/src/feature/compat_builtin.h)
  and included from the three TUs that use the builtin
  (`integer_adm.c`, `x86/adm_avx2.c`, `x86/adm_avx512.c`);
  (vii) `extern "C"` wrap added around
  `#include "log.h"` in
  [`libvmaf/src/sycl/d3d11_import.cpp`](libvmaf/src/sycl/d3d11_import.cpp)
  so `vmaf_log` resolves against the C-linkage symbol
  produced by `log.c` when this .cpp TU gets pulled into
  a SYCL-enabled test executable by icpx-cl. Upstream
  `log.h` has no `__cplusplus` guard; the wrap keeps the
  fork-local fix inside the fork-added .cpp instead of
  touching the shared header.
  Workflow change: both Windows MSVC matrix legs now pass
  `--default-library=static` in `meson_extra` because libvmaf's
  public API carries no `__declspec(dllexport)` — a vanilla
  MSVC shared build produces an empty import lib and tools
  fail with `LNK1181`. Mirrors the MinGW leg's static-link
  choice. Both MSVC CUDA and MSVC SYCL legs validated
  locally end-to-end on a Windows Server 2022 VM with
  CUDA 13.0, oneAPI 2025.3, and Level Zero loader v1.18.5
  prior to push.
  See [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md)
  paragraphs (h)–(p).
- **CUDA**: out-of-the-box GPU coverage for Ampere `sm_86` (RTX 30xx)
  and Ada `sm_89` (RTX 40xx). The gencode array in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build) now
  unconditionally emits cubins for `sm_75` / `sm_80` / `sm_86` /
  `sm_89` plus a `compute_80` PTX backward-JIT fallback, independent
  of host `nvcc` version. Upstream Netflix only shipped cubins at Txx
  major boundaries, so Ampere-`sm_86` / Ada-`sm_89` ran on a
  `compute_90` PTX that cannot JIT backward and fell over at
  kernel-load time on every consumer 3080/3090/4070/4090. See
  [ADR-0122](docs/adr/0122-cuda-gencode-coverage-and-init-hardening.md)
  and [`docs/rebase-notes.md` entry 0023](docs/rebase-notes.md).
- **CUDA**: actionable init-failure logging in
  [`libvmaf/src/cuda/common.c`](libvmaf/src/cuda/common.c). When
  `cuda_load_functions()` (the `nv-codec-headers` dlopen wrapper
  around `libcuda.so.1`) fails, `vmaf_cuda_state_init()` now emits a
  multi-line message naming the missing library, the loader-path
  check command (`ldconfig -p | grep libcuda`), and the docs section
  at
  [`docs/backends/cuda/overview.md#runtime-requirements`](docs/backends/cuda/overview.md#runtime-requirements).
  A parallel message on `cuInit(0)` failure distinguishes
  driver-load failure from userspace/kernel version skew. Also fixes
  a pre-existing leak on both error paths (`cuda_free_functions()` +
  `free(c)` + `*cu_state = NULL`). See
  [ADR-0122](docs/adr/0122-cuda-gencode-coverage-and-init-hardening.md).
- **Automated rule-enforcement for four process ADRs**: new workflow
  [`.github/workflows/rule-enforcement.yml`](.github/workflows/rule-enforcement.yml)
  plus a pre-commit `check-copyright` hook
  ([`scripts/ci/check-copyright.sh`](scripts/ci/check-copyright.sh)) close
  the "rule-without-a-check" gap on
  [ADR-0100](docs/adr/0100-project-wide-doc-substance-rule.md),
  [ADR-0105](docs/adr/0105-copyright-handling-dual-notice.md),
  [ADR-0106](docs/adr/0106-adr-maintenance-rule.md), and
  [ADR-0108](docs/adr/0108-deep-dive-deliverables-rule.md). The
  ADR-0108 six-deliverable checklist is **blocking**; the other
  three are advisory comments because their predicates require
  human judgement (pure-refactor exemption, ADR-triviality call,
  copyright-template choice). Upstream-port PRs (`port:` title /
  `port/` branch) are exempt. Reviewer documentation at
  [`docs/development/automated-rule-enforcement.md`](docs/development/automated-rule-enforcement.md).
  First `--all-files` pass also backfilled 18 pre-existing missing
  headers (13 upstream C files Netflix 2016–2026, 4 fork-authored
  NEON sources + `python/compat/config.h` Lusoris+Claude 2026);
  `libvmaf/src/pdjson.{c,h}` (vendored JSON parser) and
  `python/vmaf/matlab/` (upstream MATLAB MEX) are excluded from
  the hook rather than receiving synthetic headers. See
  [ADR-0124](docs/adr/0124-automated-rule-enforcement.md) and
  [Research-0002](docs/research/0002-automated-rule-enforcement.md).

### Changed

- **Upstream port — ADM** (Netflix `966be8d5`, fork PR #44, merged
  `d06dd6cf`): integer ADM kernels + AVX2/AVX-512 SIMD paths +
  `barten_csf_tools.h` ported wholesale; `i4_adm_cm` signature extended
  from 8 to 13 args. Netflix golden VMAF mean unchanged at
  `76.66890` (places=4 OK). See
  [`docs/rebase-notes.md` entry 0012](docs/rebase-notes.md).
- **Upstream port — motion** (Netflix PR #1486 head `2aab9ef1`, sister
  to ADM port): integer motion + AVX2/AVX-512 paths +
  `motion_blend_tools.h` ported wholesale; new `integer_motion3`
  sub-feature appears in the default VMAF model output. Golden mean
  shifts `76.66890` → `76.66783` (within `places=2` tolerance the
  upstream PR loosened to). See
  [`docs/rebase-notes.md` entry 0013](docs/rebase-notes.md).
- Python diagnostic output (`Result._get_perframe_score_str`) now emits
  scores at `%.17g` instead of `%.6f` for round-trip reproducibility.
- Copyright headers across Netflix-authored sources updated `2016-2020` →
  `2016-2026`.
- **CI hygiene — Node 24 stragglers**: finish the `@v7` bump left over
  after the rename sweep (rebase-notes 0019/0020). `scorecard.yml`
  SHA-pinned `actions/upload-artifact@330a01c4... # v5` →
  `@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1`, and
  `nightly-bisect.yml` `actions/setup-python@v5` → `@v6`. Resolves the
  last `Node.js 20 actions are deprecated` runner warnings ahead of
  the **2026-06-02** forced-Node-24 cutover (full Node-20 removal
  2026-09-16). Every other workflow was already bumped on master.
- **Engineering process**: every fork-local PR now ships the six
  deep-dive deliverables (research digest, decision matrix in the
  ADR, `AGENTS.md` invariant note, reproducer command, fork-changelog
  entry, rebase note) in the same PR. New scaffolding:
  [`docs/research/`](docs/research/),
  [`docs/rebase-notes.md`](docs/rebase-notes.md), updated
  [`PR template`](.github/PULL_REQUEST_TEMPLATE.md). See
  [ADR-0108](docs/adr/0108-deep-dive-deliverables-rule.md). Existing
  fork-local PRs have a one-shot rebase-notes backfill (10 grouped
  workstream entries) so the next upstream sync starts from a
  populated ledger. Closes #38.
- **Coverage gate**: end-to-end overhaul. (1) Build with
  `-fprofile-update=atomic` (CPU + GPU jobs) so parallel meson tests
  stop racing the `.gcda` counters on instrumented SIMD inner loops —
  eliminates the "Unexpected negative count for vif_avx2.c:673"
  geninfo hard-fail. (2) Run `meson test --num-processes 1` in the
  coverage steps so multiple test binaries don't concurrently merge
  their counters into the same `.gcda` files for the shared
  `libvmaf.so` (the on-exit merge is a multi-PROCESS race the atomic
  flag does not cover). (3) Switch `lcov` → `gcovr`: `gcovr`
  deduplicates `.gcno` files belonging to the same source compiled
  into multiple targets, fixing the `dnn_api.c — 1176%` over-count
  that surfaced after (1)+(2) on the first attempt. (4) Install
  ONNX Runtime in the coverage job and build with
  `-Denable_dnn=enabled` so `libvmaf/src/dnn/*.c` contribute real
  coverage instead of stubs (the 85% per-critical-file gate was
  previously unmeasurable). Coverage artifact is now
  `coverage.{xml,json,txt}` (Cobertura + gcovr JSON summary +
  human-readable text). (5) Carve `vmaf_use_tiny_model` out of
  `libvmaf/src/dnn/dnn_api.c` into a new
  `libvmaf/src/dnn/dnn_attach_api.c` so the unit-test binaries —
  which pull in `dnn_sources` for `feature_lpips.c` but never link
  `libvmaf.c` — don't end up with an undefined reference to
  `vmaf_ctx_dnn_attach` once `enable_dnn=enabled` activates the real
  bodies. The new TU is wired into `libvmaf.so` only via a separate
  `dnn_libvmaf_only_sources` list.
  See [ADR-0110](docs/adr/0110-coverage-gate-fprofile-update-atomic.md)
  (race fixes) and
  [ADR-0111](docs/adr/0111-coverage-gate-gcovr-with-ort.md)
  (gcovr + ORT) and
  [`docs/rebase-notes.md` entry 0014](docs/rebase-notes.md).
- **Lint scope**: upstream-mirror Python tests under `python/test/*.py`
  are now linted at the same standard as fork-added code. Mechanical
  Black + isort reformat applied to the four Netflix golden test
  files (`feature_extractor_test.py`, `quality_runner_test.py`,
  `vmafexec_test.py`, `vmafexec_feature_extractor_test.py`) — no
  assertion values changed; imports regrouped, line wrapping
  normalised. `python/test/resource/` (binary fixtures) remains
  excluded. Per user direction "don't skip linting on upstream
  things": `/sync-upstream` and `/port-upstream-commit` will
  re-trigger lint failures whenever upstream rewrites these files,
  and the fix is another in-tree reformat pass — never an exclusion.
  See [`docs/rebase-notes.md` entry 0014](docs/rebase-notes.md).
- **Coverage Gate annotations cleanup**: `actions/upload-artifact@v5|@v6
  → @v7` (and `actions/download-artifact@v5 → @v7` on supply-chain.yml)
  across every workflow under `.github/workflows/`, ahead of GitHub's
  2026-06-02 forced-Node-24 cutoff that turns the current Node 20
  deprecation banner into a hard error. Coverage Gate gcovr
  invocations also pipe stderr through `grep -vE 'Ignoring
  (suspicious|negative) hits' ... || true` so the chatty annotation
  for legitimately-large hit counts on tight inner loops (e.g.
  `ansnr_tools.c:207` at ~4.93 G hits across an HD multi-frame
  coverage suite) is dropped without losing the underlying data —
  `--gcov-ignore-parse-errors=suspicious_hits.warn` still tells
  gcovr to accept the count, only the annotation is filtered. The
  filter regex is anchored to gcov's exact warning prefix, so any
  *other* gcovr warning still surfaces. See
  [ADR-0117](docs/adr/0117-coverage-gate-warning-noise-suppression.md)
  and [`docs/rebase-notes.md` entry 0015](docs/rebase-notes.md).

### Fixed

- **SSIM / MS-SSIM NEON bit-exactness to scalar**: fork-local
  `ssim_accumulate_neon`
  ([`libvmaf/src/feature/arm64/ssim_neon.c`](libvmaf/src/feature/arm64/ssim_neon.c))
  previously carried the same ~0.13 float-ULP drift on
  `float_ms_ssim` / ~6 × 10⁻⁸ drift on `float_ssim` that ADR-0139
  fixed for AVX2 / AVX-512 — it was never surfaced because CI has no
  aarch64 runner. The NEON accumulator now computes the float-valued
  intermediates in vector float (`float32x4_t`) and spills to
  `SIMD_ALIGNED_F32_BUF_NEON(4)` buffers so the
  `2.0 * mu1 * mu2 + C1` numerator + division + `l*c*s` triple
  product run per-lane in scalar double, matching the x86 fix. Also
  plugged the aarch64 `iqa_convolve` gap — there was no NEON convolve
  at all before this PR; the VIF / ADM features used the scalar path
  on aarch64 while x86 ran AVX2 / AVX-512. Verified bit-identical
  under `qemu-aarch64-static` on both Netflix `src01_hrc00/01_576x324`
  and `checkerboard_1920_1080_10_3` pairs at `--precision max`. See
  [ADR-0140](docs/adr/0140-simd-dx-framework.md) +
  [research digest 0013](docs/research/0013-simd-dx-framework.md).
- **SSIM / MS-SSIM AVX2 + AVX-512 bit-exactness to scalar**: fork-local
  `ssim_accumulate_avx2` / `ssim_accumulate_avx512`
  ([`libvmaf/src/feature/x86/ssim_avx2.c`](libvmaf/src/feature/x86/ssim_avx2.c),
  [`libvmaf/src/feature/x86/ssim_avx512.c`](libvmaf/src/feature/x86/ssim_avx512.c))
  previously computed the `l`, `c`, `s` factors as vector float and
  produced the `l * c * s` triple product in float before accumulating
  to double — that diverged from the scalar reference by ~0.13 float
  ULPs (8th decimal) on `float_ms_ssim`, because scalar evaluates
  `2.0 * mu1 * mu2 + C1` and `2.0 * srsc + C2` in double (the literal
  `2.0` is a C `double`) and runs `lv * cv * sv` as three double
  multiplies. The SIMD accumulators now compute the float-valued
  intermediates (`srsc`, denominators, `sv`) in vector float and do
  the double-precision numerator + division + triple product per-lane
  in scalar double inside an 8/16-wide inner loop, matching scalar
  byte-for-byte. Verified: scalar = AVX2 = AVX-512 bit-identical at
  `--precision max` on both Netflix `src01_hrc00/01_576x324` and
  `checkerboard_1920_1080_10_3` pairs. `ssim_precompute_*` and
  `ssim_variance_*` were already bit-exact (pure elementwise float
  ops). Companion fix to the new bit-exact `_iqa_convolve_avx2/512`
  dispatch. See
  [ADR-0139](docs/adr/0139-ssim-simd-bitexact-double.md) +
  [ADR-0138](docs/adr/0138-iqa-convolve-avx2-bitexact-double.md).
- **CUDA multi-session `vmaf_cuda_picture_free` assertion-0 crash**:
  two or more concurrent CUDA sessions freeing pictures tripped
  `Assertion 0 failed` inside the driver because
  `cuMemFreeAsync(ptr, stream)` enqueued the free on a stream that
  was destroyed two statements later. The fork swaps the async call
  for synchronous `cuMemFree` at
  [`libvmaf/src/cuda/picture_cuda.c:247`](libvmaf/src/cuda/picture_cuda.c#L247);
  the preceding `cuStreamSynchronize` already removed any async
  overlap so perf is unchanged. Ports
  [Netflix#1382](https://github.com/Netflix/vmaf/pull/1382)
  (tracking [Netflix#1381](https://github.com/Netflix/vmaf/issues/1381));
  see [ADR-0131](docs/adr/0131-port-netflix-1382-cumemfree.md).
- **`vmaf_feature_collector_mount_model` list-corruption on ≥3
  models**: the upstream body advanced the `*head` pointer-to-pointer
  instead of walking a local cursor, overwriting the head element
  with its own successor and losing every entry past the second.
  Fork rewrites mount/unmount in
  [`libvmaf/src/feature/feature_collector.c`](libvmaf/src/feature/feature_collector.c)
  with a correct traversal, and `unmount_model` now returns
  `-ENOENT` (not `-EINVAL`) when the requested model isn't mounted
  so callers can distinguish misuse from not-found. Test coverage
  extended to a 3-element mount / unmount sequence. Ports
  [Netflix#1406](https://github.com/Netflix/vmaf/pull/1406);
  see [ADR-0132](docs/adr/0132-port-netflix-1406-feature-collector-model-list.md).
- **`KBND_SYMMETRIC` sub-kernel-radius out-of-bounds reflection**:
  upstream's 2-D symmetric boundary extension reflected the index a
  single time, which leaves out-of-bounds values whenever the input
  dimension is smaller than the kernel half-width (for the 9-tap
  MS-SSIM LPF, `n ≤ 3`). The fork rewrites `KBND_SYMMETRIC` in
  [`libvmaf/src/feature/iqa/convolve.c`](libvmaf/src/feature/iqa/convolve.c)
  and the scalar / AVX2 / AVX-512 / NEON `ms_ssim_decimate_mirror`
  helpers into the period-based form (`period = 2*n`) that bounces
  correctly for any offset. Netflix golden outputs are unchanged
  because 576×324 and 1920×1080 inputs never exercise the
  sub-kernel-radius regime. See
  [`docs/development/known-upstream-bugs.md`](docs/development/known-upstream-bugs.md)
  and [ADR-0125](docs/adr/0125-ms-ssim-decimate-simd.md).
- **`adm_decouple_s123_avx512` LTO+release SEGV**: the stack array
  `int64_t angle_flag[16]` is read via two `_mm512_loadu_si512`
  calls. Under `--buildtype=release -Db_lto=true`, link-time
  alignment inference promotes them to `vmovdqa64`, which faults
  because the C default stack alignment for `int64_t[16]` is 8
  bytes. Annotating the array with `_Alignas(64)` at
  [`libvmaf/src/feature/x86/adm_avx512.c:1317`](libvmaf/src/feature/x86/adm_avx512.c#L1317)
  keeps both the unaligned source form and the LTO-promoted aligned
  form correct. Debug / no-LTO builds, and every CI sanitizer job,
  are unaffected.
- **`test_pic_preallocation` VmafModel leaks**:
  `test_picture_pool_basic` / `_small` / `_yuv444` loaded a
  `VmafModel` via `vmaf_model_load` and never freed it, so
  LeakSanitizer reported 208 B direct + 23 KiB indirect per test.
  Paired each load with `vmaf_model_destroy(model)` in
  [`libvmaf/test/test_pic_preallocation.c`](libvmaf/test/test_pic_preallocation.c).
- **`libvmaf_cuda` ffmpeg filter segfault on first frame**: external
  reporter (2026-04-19) hit a SIGSEGV in `vmaf_ref_fetch_increment` on
  every invocation of ffmpeg's `libvmaf_cuda` filter against the fork's
  master build. Root cause is a three-commit composition: upstream
  `32b115df` (2026-04-07) added the experimental `VMAF_PICTURE_POOL`
  with an always-live `vmaf->prev_ref` slot; upstream `f740276a`
  (2026-04-09) moved the `vmaf_picture_ref(&vmaf->prev_ref, ref)` tail
  onto the non-threaded path without guarding against `ref->ref ==
  NULL`; fork commit `65460e3a` ([ADR-0104](docs/adr/0104-picture-pool-always-on.md))
  dropped the `VMAF_PICTURE_POOL` meson gate for ABI stability
  (+10 fps CPU gain), exposing the unguarded deref to every default
  build. On the CUDA-device-only extractor set that the ffmpeg filter
  registers, `rfe_hw_flags` returns `HW_FLAG_DEVICE` only,
  `translate_picture_device` early-returns without downloading, and
  `ref_host` stays zero-initialised — the subsequent
  `vmaf_picture_ref(&prev_ref, &ref_host)` deref'd `NULL`. Fix is a
  narrow null-guard at `libvmaf/src/libvmaf.c:1428`
  (`if (ref && ref->ref) vmaf_picture_ref(...)`). Semantically correct,
  not merely defensive: the only `VMAF_FEATURE_EXTRACTOR_PREV_REF`
  consumer is CPU `integer_motion_v2`, which is never registered
  alongside a pure-CUDA set. SYCL is unaffected (`vmaf_read_pictures_sycl`
  does not touch `prev_ref`). Always-on picture pool stays. See
  [ADR-0123](docs/adr/0123-cuda-post-cubin-load-regression-32b115df.md);
  follow-up item to port the null-guard upstream to Netflix/vmaf.
- **VIF `init()` fail-path leak**: `libvmaf/src/feature/integer_vif.c`'s
  `init()` carves one `aligned_malloc` into the VifBuffer sub-pointers by
  walking a `uint8_t *data` cursor forward through the allocation. When
  `vmaf_feature_name_dict_from_provided_features` returned NULL, the
  fail-path called `aligned_free(data)` on the *advanced* cursor — not a
  valid `aligned_malloc` return — leaking the whole block and passing a
  garbage pointer to `free`. Fail path now frees `s->public.buf.data`,
  the saved base pointer. Ported from Netflix upstream PR
  [#1476](https://github.com/Netflix/vmaf/pull/1476); the companion
  void*→uint8_t* UB portability fix from that PR is already on master
  (commit `b0a4ac3a`, rebase-notes 0022 §e).
- **CLI precision default reverted to `%.6f` (Netflix-compat)**: ADR-0006
  shipped `%.17g` as the default for round-trip-lossless output, but
  several Netflix golden tests in `python/test/command_line_test.py`,
  `vmafexec_test.py` etc. do *exact-string* matches against XML output
  (not `assertAlmostEqual`), so the wider default broke the gate. Default
  now matches upstream Netflix byte-for-byte; `--precision=max` (alias
  `full`) is the explicit opt-in for `%.17g`. `--precision=legacy` is
  preserved as a synonym for the (new) default. Library
  `vmaf_write_output_with_format(..., NULL)` and `python/vmaf/core/result.py`
  formatters revert in lockstep. See
  [ADR-0119](docs/adr/0119-cli-precision-default-revert.md) (supersedes
  [ADR-0006](docs/adr/0006-cli-precision-17g-default.md)). Latent on
  master 2026-04-15 → 2026-04-19; surfaced by ADR-0115's CI consolidation
  routing tox through master-targeting PRs.
- **`--frame_skip_ref` / `--frame_skip_dist` hang**: the skip loops in
  `libvmaf/tools/vmaf.c` fetched pictures from the preallocated picture
  pool (now always-on per ADR-0104) but never `vmaf_picture_unref`'d
  them, exhausting the pool after N skips and blocking the next fetch
  indefinitely. Each skipped picture is now unref'd immediately after
  fetch. Surfaced by `test_run_vmafexec_with_frame_skipping{,_unequal}`
  hanging locally (timeout 60 s, no output written) once tox started
  exercising both flags on master-targeting PRs.
- **CI tox doctest collection**: `pytest --doctest-modules` errored on five
  upstream files under `python/vmaf/resource/` (parameter / dataset / example
  config files; `vmaf_v7.2_bootstrap.py` and friends — dots in the stem make
  them unimportable as Python modules). Tox commands now pass
  `--ignore=vmaf/resource` so doctest collection skips that subtree. The
  files carry no doctests to begin with, so this is correctness, not a
  workaround. Surfaced by ADR-0115's CI trigger consolidation, which finally
  ran tox on PRs to master.

- **SYCL build with non-icpx host CXX**: `libvmaf/src/meson.build`
  unconditionally added `-fsycl` to the libvmaf shared-library link args
  whenever SYCL was enabled, even when the project's C++ compiler was
  gcc / clang / msvc. The host link driver does not understand `-fsycl`
  and failed with `g++: error: unrecognized command-line option '-fsycl'`
  at the `libvmaf.so` link step. The arg is now gated on
  `meson.get_compiler('cpp').get_id() == 'intel-llvm'`. The runtime
  libraries (libsycl + libsvml + libirc + libze_loader) declared as link
  dependencies already cover the gcc/clang link path, matching the
  documented "host C++ + sidecar icpx" project mode. Surfaced by
  ADR-0115's CI consolidation, which added an Ubuntu SYCL job that
  exercises this configuration on PRs to master.

- **FFmpeg patch series application**: `Dockerfile` and
  `.github/workflows/ffmpeg.yml` now walk `ffmpeg-patches/series.txt`
  and apply each patch in order via `git apply` with a `patch -p1`
  fallback. The Dockerfile previously `COPY`'d only patch 0003 (which
  fails to apply standalone because it references `LIBVMAFContext`
  fields added by patch 0001), and `ffmpeg.yml` referenced a stale
  `../patches/ffmpeg-libvmaf-sycl.patch` that no longer existed.
  Patches `0001-libvmaf-add-tiny-model-option.patch`,
  `0002-add-vmaf_pre-filter.patch`, and
  `0003-libvmaf-wire-sycl-backend-selector.patch` were also
  regenerated via real `git format-patch -3` so they carry valid
  `index <sha>..<sha> <mode>` header lines (the originals were
  hand-stubbed with placeholder SHAs and `git apply` choked on them).
  Docker images and CI FFmpeg-SYCL builds now exercise the full
  fork-added FFmpeg surface (tiny-AI + `vmaf_pre` + SYCL selector),
  not just SYCL. Also drops the bogus `--enable-libvmaf-sycl`
  configure flag (patch 0003 wires SYCL via `check_pkg_config`
  auto-detection — there is no such configure switch) and splits
  the Dockerfile's nvcc flags into a libvmaf set
  (`NVCC_FLAGS`, retains the four `-gencode` lines plus
  `--extended-lambda` and the `--expt-*` flags for Thrust/CUB) and
  an FFmpeg set (`FFMPEG_NVCC_FLAGS`, single-arch
  `compute_75,sm_75` matching FFmpeg's own modern-nvcc default —
  PTX is forward-compatible via driver JIT) so FFmpeg's
  `check_nvcc -ptx` probe stops failing with `nvcc fatal: Option
  '--ptx (-ptx)' is not allowed when compiling for multiple GPU
  architectures`. Also drops `--enable-libnpp` from FFmpeg
  configure — FFmpeg n8.1 explicitly `die`s if libnpp >= 13.0
  (configure:7335-7336 `"libnpp support is deprecated, version
  13.0 and up are not supported"`), and we don't actually use
  scale_npp / transpose_npp filters in VMAF workflows; cuvid +
  nvdec + nvenc + libvmaf-cuda are what we exercise. Patch 0002
  also gained a missing `#include "libavutil/imgutils.h"` for
  `av_image_copy_plane` (caught by the local docker build —
  upstream FFmpeg builds with `-Werror=implicit-function-declaration`).
  See ADR-0118 and entry 0018.

- **CI workflow naming**: renamed all six core `.github/workflows/*.yml`
  files to purpose-descriptive kebab-case (e.g. `ci.yml` →
  `tests-and-quality-gates.yml`, `libvmaf.yml` →
  `libvmaf-build-matrix.yml`) and normalised every workflow `name:` and
  job `name:` to Title Case. Required-status-check contexts in
  `master` branch protection re-pinned in the same merge window. See
  [ADR-0116](docs/adr/0116-ci-workflow-naming-convention.md) +
  [`docs/rebase-notes.md` entry 0020](docs/rebase-notes.md).

### Re-attributed

- 11 SYCL files in `libvmaf/{include,src,test}/.../sycl/` from
  `Netflix, Inc.` to `Lusoris and Claude (Anthropic)` — these files were
  authored entirely by the fork.
### Added

- **SpEED-QA feasibility digest + Proposed ADR (research-0051 / ADR-0253).**
  Closes the user's 2026-04-21 deep-research queued track on SpEED-QA as a
  candidate full-reference metric. Recommends DEFER over GO / SCAFFOLD-ONLY:
  the fork keeps the existing `speed_chroma` / `speed_temporal` research-stage
  extractors (PR #213, port of upstream `d3647c73`) and does not add a
  `speed_qa` reduction. Three findings drive the call —
  (1) SpEED-QA's GSM-entropy backbone overlaps `vif` substantially with no new
  perceptual axis; (2) the "10–40× faster than VIF" headline inverts on the
  fork's AVX-512 / CUDA / SYCL VIF stack; (3) the assumed-but-missing
  `model/speed_4_v0.6.0.json` upstream binary the brief referenced does not
  exist anywhere in `upstream/master`, `upstream/speed_ported`, or any open
  Netflix PR. Decision is reversible on three named triggers (see ADR-0253
  *Consequences → Follow-ups*). Docs-only PR — no code, no model registry
  change, no CLI flag, no behavioural delta. See
  [ADR-0253](docs/adr/0253-speed-qa-extractor.md) +
  [`docs/research/0051-speed-qa-feasibility.md`](docs/research/0051-speed-qa-feasibility.md).


- **`vmaf-tune compare` — codec-comparison mode (research-0061
  Bucket #7, ADR-0237 Phase A follow-up).** Given a single source and
  a target VMAF, `vmaf-tune compare --src REF.yuv --target-vmaf 92
  --encoders libx264,libx265,libsvtav1,libaom,libvvenc` runs each
  codec's recommend predicate in a thread pool and emits a ranked
  `(codec, best_crf, bitrate_kbps, encode_time_ms, vmaf_score)` table
  sorted by smallest file. Supports `--format markdown|json|csv` and
  `--output PATH`. Until Phase B's recommend backend lands, point
  `--predicate-module MODULE:CALLABLE` at any importable
  `(codec, src, target_vmaf) -> RecommendResult` callable to drive the
  ranking from a shim. Default `--encoders` resolves to every adapter
  currently registered in `codec_adapters/` — Phase A wires `libx264`
  only, so the canonical four / five codec invocation only ranks
  codecs whose adapters have already merged. New module
  `tools/vmaf-tune/src/vmaftune/compare.py` (predicate-driven
  orchestration + markdown / JSON / CSV renderers); 13 mocked smoke
  tests under `tools/vmaf-tune/tests/test_compare.py` (no `ffmpeg`,
  no built `vmaf` required). Schema exported as
  `vmaftune.compare.COMPARE_ROW_KEYS`. User docs:
  [`docs/usage/vmaf-tune.md`](../docs/usage/vmaf-tune.md) §"Codec
  comparison".

- **`tools/vmaf-tune/` Phase A — quality-aware encode automation scaffold
  (ADR-0237 Phase A Accepted, Research-0044).** New Python tool that
  drives FFmpeg over a `(preset, crf)` grid against `libx264`, scores
  each encode with the libvmaf CLI, and emits a JSONL corpus of
  `(source, encoder, params, bitrate, vmaf)` rows. Schema versioned via
  `vmaftune.SCHEMA_VERSION = 1` and exported as `CORPUS_ROW_KEYS`; the
  schema is the API contract that Phase B (target-VMAF bisect) and
  Phase C (per-title CRF predictor) will consume. Codec adapter
  registry (`codec_adapters/`) is multi-codec from day one — Phase A
  wires `libx264` only; subsequent codecs (`libx265`, `libsvtav1`,
  `libvpx-vp9`, `libvvenc`, neural extras) are one-file additions
  without touching the search loop. Subprocess-mocked smoke tests
  under `tools/vmaf-tune/tests/` (13 cases) cover command shape,
  version parsing, JSONL round-trip, encode-failure handling, and the
  schema contract — no `ffmpeg` or built `vmaf` binary required.
  User docs: [`docs/usage/vmaf-tune.md`](../docs/usage/vmaf-tune.md).
  Phases B–F remain Proposed under ADR-0237; this PR ships only the
  Phase A corpus scaffold.


- **Tiny-AI / saliency**: Added `saliency_student_v1` — a fork-trained
  tiny U-Net (~113 K parameters, ONNX opset 17, BSD-3-Clause-Plus-Patent)
  trained from scratch on the DUTS-TR public saliency-detection corpus
  (Wang et al. 2017). Replaces the smoke-only
  `mobilesal_placeholder_v0` as the recommended weights for the
  `mobilesal` feature extractor. The C-side `feature_mobilesal.c`
  extractor is unchanged (same `input` / `saliency_map` tensor names,
  same NCHW shapes); the new model is a true drop-in. The decoder uses
  `ConvTranspose` for stride-2 upsampling so every op in the graph is
  on `libvmaf/src/dnn/op_allowlist.c` without an allowlist patch in
  the same PR. DUTS images are not redistributed in-tree; only the
  trained weights are. The placeholder remains in the registry with
  `smoke: true` for legacy reasons. New model card at
  [`docs/ai/models/saliency_student_v1.md`](docs/ai/models/saliency_student_v1.md);
  decision in
  [ADR-0286](docs/adr/0286-saliency-student-fork-trained-on-duts.md);
  digest in
  [Research-0054](docs/research/0062-saliency-student-from-scratch-on-duts.md).
  Trainer at `ai/scripts/train_saliency_student.py`.
### Changed

- **Per-GPU-generation ULP calibration table for the cross-backend
  parity gate (T-GPU-ULP / ADR-0234).** New
  [`scripts/ci/gpu_ulp_calibration.yaml`](scripts/ci/gpu_ulp_calibration.yaml)
  maps a runtime GPU identifier (Research-0041 schema:
  `vulkan:0xVVVV:0xDDDD` / `cuda:M.m` / `sycl:0xVVVV:DRIVER`) to a
  per-feature absolute tolerance. Both
  [`scripts/ci/cross_backend_vif_diff.py`](scripts/ci/cross_backend_vif_diff.py)
  and
  [`scripts/ci/cross_backend_parity_gate.py`](scripts/ci/cross_backend_parity_gate.py)
  now accept `--gpu-id <runtime_id>` and `--calibration-table <path>`;
  when omitted, behaviour is identical to before (per-feature
  `FEATURE_TOLERANCE` defaults remain authoritative). Lookup picks
  the most-specific glob match (`vulkan:0x10005:*` for lavapipe;
  trailing `*` is supported). The hosted-CI lavapipe lane in
  [`.github/workflows/tests-and-quality-gates.yml`](.github/workflows/tests-and-quality-gates.yml)
  passes `--gpu-id "vulkan:0x10005:0x0"` so the gate's tolerance
  decisions are now per-arch annotated in the parity report's JSON
  + Markdown artefacts. Initial coverage: 1 calibrated row (Mesa
  lavapipe — tolerances match the gate's pre-existing
  `FEATURE_TOLERANCE` defaults so behaviour is unchanged) plus 11
  placeholder rows (NVIDIA Ampere / Turing / Ada / Hopper, AMD
  RDNA2 / RDNA3, Intel Arc Alchemist / Battlemage, generic Intel
  SYCL); placeholders are functional no-ops until a real-hardware
  corpus replaces their `features:` block. New unit test
  [`scripts/ci/test_calibration.py`](scripts/ci/test_calibration.py)
  (19 cases) covers the loader, glob semantics, specificity ranking,
  and the shipped-table round-trip. The ONNX calibration head and
  `--gpu-calibrated` CLI flag from ADR-0234's "Decision" §
  remain deferred to the follow-up PR `feat(ai):
  T7-GPU-ULP-CAL — calibration-head v0`. See
  [ADR-0234](docs/adr/0234-gpu-gen-ulp-calibration.md) (now
  Accepted),
  [Research-0041](docs/research/0041-gpu-gen-ulp-calibration.md),
  and the rebase-notes entry under
  [`docs/rebase-notes.md`](docs/rebase-notes.md).


- **MobileSal real-weights swap deferred (T6-2a-followup, ADR-0257)** —
  the original plan to swap the smoke-only `mobilesal_placeholder_v0`
  ONNX in `model/tiny/registry.json` for real upstream MobileSal
  weights (mirroring PR #326 / ADR-0253 for FastDVDnet) is deferred
  indefinitely. Survey in
  [`docs/research/0053-mobilesal-real-weights-blocker.md`](docs/research/0053-mobilesal-real-weights-blocker.md)
  shows three independent blockers: (1) upstream
  [`yuhuan-wu/MobileSal`](https://github.com/yuhuan-wu/MobileSal) is
  **CC BY-NC-SA 4.0** (incompatible with the fork's
  BSD-3-Clause-Plus-Patent — both the Non-Commercial and Share-Alike
  clauses bind), (2) trained checkpoints are distributed only via
  Google Drive viewer URLs (no GitHub release; no raw-download URL the
  export script can pin by SHA), and (3) MobileSal is RGB-D while the
  C-side contract is RGB-only. ADR-0218's claim that upstream MobileSal
  is "MIT-licensed" was inaccurate; corrected here and in
  [ADR-0257](docs/adr/0257-mobilesal-real-weights-deferred.md). The
  smoke-only placeholder remains shipped; the C-side
  `feature_mobilesal.c` extractor and its I/O contract are unchanged.
  `docs/ai/models/mobilesal.md` updated with the corrected upstream
  licence and the blocker pointer. Recommended replacement is to swap
  the underlying model family from MobileSal to U-2-Net's `u2netp`
  variant (Apache-2.0, 4.7 MB, pure RGB), tracked as new backlog row
  T6-2a-replace-with-u2netp; that scope shift is deliberately not
  bundled into this docs-only PR.
- **SHA-pin every GitHub Actions reference in `.github/workflows/*.yml`
  (OSSF Scorecard `Pinned-Dependencies` remediation).** Every
  `uses: <owner>/<repo>@<tag>` reference in the 13 fork workflows is
  now resolved to a 40-char commit SHA with the original semver
  preserved as a trailing `# vN.M.K` comment, mirroring the pattern
  already established for `ossf/scorecard-action`,
  `sigstore/cosign-installer`, `softprops/action-gh-release`, and
  `anchore/sbom-action`. 97 first-party and third-party action
  references converted across `docker-image.yml`, `docs.yml`,
  `ffmpeg-integration.yml`, `libvmaf-build-matrix.yml`,
  `lint-and-format.yml`, `nightly-bisect.yml`, `nightly.yml`,
  `release-please.yml`, `rule-enforcement.yml`, `scorecard.yml`,
  `security-scans.yml`, `supply-chain.yml`, and
  `tests-and-quality-gates.yml`. **Single documented holdout**: the
  `slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v2.0.0`
  reusable-workflow ref in `supply-chain.yml` keeps its `vX.Y.Z` form
  per the SLSA generator maintainers' published guidance — GitHub
  Actions consumers cannot currently SHA-pin reusable-workflow refs in
  every code path, and the existing inline comment in
  `supply-chain.yml` already calls this out. **Why this matters**: the
  `vN` floating tag is an attacker-rotatable handle (a compromised
  upstream maintainer or a tag-overwrite supply-chain incident silently
  swaps the executed code under us); SHA pinning fixes the executed
  bytes and lets Dependabot surface bumps as reviewable diffs rather
  than as silent rotations. The change is a pure ref substitution — no
  action versions are bumped — so workflow behaviour is unchanged. See
  [ADR-0263](docs/adr/0263-ossf-scorecard-policy.md) (created by
  PR #337) and the OSSF Scorecard
  [Pinned-Dependencies check documentation](https://github.com/ossf/scorecard/blob/main/docs/checks.md#pinned-dependencies).


- **Per-GPU-generation ULP calibration table for the cross-backend
  parity gate (T-GPU-ULP / ADR-0234).** New
  [`scripts/ci/gpu_ulp_calibration.yaml`](scripts/ci/gpu_ulp_calibration.yaml)
  maps a runtime GPU identifier (Research-0041 schema:
  `vulkan:0xVVVV:0xDDDD` / `cuda:M.m` / `sycl:0xVVVV:DRIVER`) to a
  per-feature absolute tolerance. Both
  [`scripts/ci/cross_backend_vif_diff.py`](scripts/ci/cross_backend_vif_diff.py)
  and
  [`scripts/ci/cross_backend_parity_gate.py`](scripts/ci/cross_backend_parity_gate.py)
  now accept `--gpu-id <runtime_id>` and `--calibration-table <path>`;
  when omitted, behaviour is identical to before (per-feature
  `FEATURE_TOLERANCE` defaults remain authoritative). Lookup picks
  the most-specific glob match (`vulkan:0x10005:*` for lavapipe;
  trailing `*` is supported). The hosted-CI lavapipe lane in
  [`.github/workflows/tests-and-quality-gates.yml`](.github/workflows/tests-and-quality-gates.yml)
  passes `--gpu-id "vulkan:0x10005:0x0"` so the gate's tolerance
  decisions are now per-arch annotated in the parity report's JSON
  + Markdown artefacts. Initial coverage: 1 calibrated row (Mesa
  lavapipe — tolerances match the gate's pre-existing
  `FEATURE_TOLERANCE` defaults so behaviour is unchanged) plus 11
  placeholder rows (NVIDIA Ampere / Turing / Ada / Hopper, AMD
  RDNA2 / RDNA3, Intel Arc Alchemist / Battlemage, generic Intel
  SYCL); placeholders are functional no-ops until a real-hardware
  corpus replaces their `features:` block. New unit test
  [`scripts/ci/test_calibration.py`](scripts/ci/test_calibration.py)
  (19 cases) covers the loader, glob semantics, specificity ranking,
  and the shipped-table round-trip. The ONNX calibration head and
  `--gpu-calibrated` CLI flag from ADR-0234's "Decision" §
  remain deferred to the follow-up PR `feat(ai):
  T7-GPU-ULP-CAL — calibration-head v0`. See
  [ADR-0234](docs/adr/0234-gpu-gen-ulp-calibration.md) (now
  Accepted),
  [Research-0041](docs/research/0041-gpu-gen-ulp-calibration.md),
  and the rebase-notes entry under
  [`docs/rebase-notes.md`](docs/rebase-notes.md).


- **AI / DNN:** Replaced the `fastdvdnet_pre` smoke-only placeholder
  ONNX with real upstream FastDVDnet weights (Tassano, Delon, Veit
  2020; MIT license) pinned at `m-tassano/fastdvdnet` commit `c8fdf61`.
  The new graph wraps upstream's RGB+noise-map model in a `LumaAdapter`
  that preserves the C-side `[1, 5, H, W]` luma I/O contract from
  ADR-0215: `Y → [Y, Y, Y]` tiling for the upstream 15-channel input,
  a constant `sigma = 25/255` noise map, and BT.601 RGB→Y collapse on
  the output. Upstream `nn.PixelShuffle` is swapped at export time for
  an allowlist-safe `Reshape`/`Transpose`/`Reshape` decomposition
  (`DepthToSpace` is deliberately not on the ONNX op allowlist).
  Registry row `model/tiny/registry.json` flips `smoke: false` with
  the new MIT license, upstream commit pin, and refreshed sha256.
  9.5 MiB ONNX, opset 17. New exporter
  `ai/scripts/export_fastdvdnet_pre.py`. See ADR-0253 and
  `docs/ai/models/fastdvdnet_pre.md`.


- **AI / DNN:** Replaced the `fastdvdnet_pre` smoke-only placeholder
  ONNX with real upstream FastDVDnet weights (Tassano, Delon, Veit
  2020; MIT license) pinned at `m-tassano/fastdvdnet` commit `c8fdf61`.
  The new graph wraps upstream's RGB+noise-map model in a `LumaAdapter`
  that preserves the C-side `[1, 5, H, W]` luma I/O contract from
  ADR-0215: `Y → [Y, Y, Y]` tiling for the upstream 15-channel input,
  a constant `sigma = 25/255` noise map, and BT.601 RGB→Y collapse on
  the output. Upstream `nn.PixelShuffle` is swapped at export time for
  an allowlist-safe `Reshape`/`Transpose`/`Reshape` decomposition
  (`DepthToSpace` is deliberately not on the ONNX op allowlist).
  Registry row `model/tiny/registry.json` flips `smoke: false` with
  the new MIT license, upstream commit pin, and refreshed sha256.
  9.5 MiB ONNX, opset 17. New exporter
  `ai/scripts/export_fastdvdnet_pre.py`. See ADR-0253 and
  `docs/ai/models/fastdvdnet_pre.md`.


- **AI / DNN:** Replaced the `fastdvdnet_pre` smoke-only placeholder
  ONNX with real upstream FastDVDnet weights (Tassano, Delon, Veit
  2020; MIT license) pinned at `m-tassano/fastdvdnet` commit `c8fdf61`.
  The new graph wraps upstream's RGB+noise-map model in a `LumaAdapter`
  that preserves the C-side `[1, 5, H, W]` luma I/O contract from
  ADR-0215: `Y → [Y, Y, Y]` tiling for the upstream 15-channel input,
  a constant `sigma = 25/255` noise map, and BT.601 RGB→Y collapse on
  the output. Upstream `nn.PixelShuffle` is swapped at export time for
  an allowlist-safe `Reshape`/`Transpose`/`Reshape` decomposition
  (`DepthToSpace` is deliberately not on the ONNX op allowlist).
  Registry row `model/tiny/registry.json` flips `smoke: false` with
  the new MIT license, upstream commit pin, and refreshed sha256.
  9.5 MiB ONNX, opset 17. New exporter
  `ai/scripts/export_fastdvdnet_pre.py`. See ADR-0253 and
  `docs/ai/models/fastdvdnet_pre.md`.


- **CHANGELOG + ADR-index fragment files (T7-39 / ADR-0221)** — every PR
  in flight before this change fought merge conflicts in
  [`CHANGELOG.md`](CHANGELOG.md) and
  [`docs/adr/README.md`](docs/adr/README.md) (each PR adds a row, every
  other PR's row collides). PRs now drop a single fragment file under
  `changelog.d/<section>/<topic>.md` (Keep-a-Changelog sections: `added`,
  `changed`, `deprecated`, `removed`, `fixed`, `security`) and one row
  fragment under `docs/adr/_index_fragments/NNNN-slug.md`. Two new in-tree
  shell scripts —
  [`scripts/release/concat-changelog-fragments.sh`](scripts/release/concat-changelog-fragments.sh)
  and [`scripts/docs/concat-adr-index.sh`](scripts/docs/concat-adr-index.sh)
  — render `CHANGELOG.md`'s Unreleased block and `docs/adr/README.md`
  from the fragment trees; both ship `--check` (CI) and `--write`
  (release-please / local) modes. Migration is content-preserving: the
  existing 3119-line Unreleased body is archived verbatim under
  `changelog.d/_pre_fragment_legacy.md`, and 159 ADR rows are split into
  per-slug fragment files driven by a frozen `_order.txt` manifest that
  preserves the existing commit-merge order. New PRs append one fragment
  file (and one line to `_order.txt`) instead of editing the consolidated
  files. Doc-Substance Gate (ADR-0167) recognises a new
  `changelog.d/<section>/<row>.md` as a CHANGELOG entry. See
  [ADR-0221](docs/adr/0221-changelog-adr-fragment-pattern.md) +
  [`docs/research/0034-changelog-fragment-pattern.md`](docs/research/0034-changelog-fragment-pattern.md).


- T7-5 — readability-function-size NOLINT sweep. Refactored
  `float_adm.c::extract` (debug-feature appends extracted into
  `append_debug_features` helper) and `tools/vmaf.c::main` (eight
  helpers extracted: `open_input_videos`, `init_gpu_backends`,
  `allocate_model_arrays`, `model_label`, `load_model_collection_entry`,
  `load_one_model_entry`, `configure_tiny_model`, `resolve_tiny_device`,
  `skip_initial_frames`, `run_frame_loop`, `report_pooled_scores`).
  Two pre-2026-04-21 historical-debt NOLINTs removed; remaining NOLINTs
  in `tools/vmaf.c` (`copy_picture_data`, `init_gpu_backends`, `main`)
  carry inline justification per ADR-0141 §2 — load-bearing CLI
  cleanup-ownership chain and conditional-compilation backend stanzas
  that further extraction would obscure. Netflix CPU golden assertions
  byte-exact (90/90 + 57/57 VMAF-specific tests pass; pre-existing
  pypsnr/niqe Python-3.14 failures unchanged). Closes T7-5.


- **Dedup duplicate-NNNN ADRs (bookkeeping).** Renumbered ten ADR files
  that violated the `docs/adr/README.md` "IDs assigned in commit order
  and never reused" rule (5 NNNN values had 2–7 sharing files;
  earliest-committed file at each colliding NNNN kept its number, the
  rest moved into the next free range 0242–0251). Filenames, H1
  headings, in-tree citations (`docs/`, `libvmaf/src/`, `ai/`,
  `scripts/`, `model/`, `mcp-server/`), and `docs/adr/README.md` index
  rows updated; ADR body prose is unchanged. Mappings recorded in
  [`docs/adr/README.md`](docs/adr/README.md) Conventions section under
  "2026-05-02 dedup sweep". Fork-private planning dossiers may still
  cite old NNNNs — consult the mapping table when resolving
  pre-sweep references.

## (2022-04-11) [v2.3.1]

This is a minor release with some CAMBI extensions and speed-ups and adding it
to AOM CTC v3, as well as a few minor fixes/cleanups.

- CAMBI extensions: full reference, PQ eotf, up to 16 bit-depth support,
  max_log_contrast parameter.
- CAMBI: option to output heatmaps.

## (2021-10-16) [v2.3.0]

New release to add CAMBI (Contrast Aware Multiscale Banding Index).

- Python library: add encode width and height to Asset.
- libvmaf: add pixel format VMAF_PIX_FMT_YUV400P.
- Add cambi; add tests.
- Improve documentation. (#912)

## (2021-09-20) [v2.2.1]

This is another minor release to address a few last minute items for the AOM CTC
v2, as well as a few minor fixes/cleanups.

- Fix a race condition in vmaf_thread_pool_wait(). (#894)
- Avoid chroma resampling for 420mpeg2 y4m input (#906)

## (2021-07-02) [v2.2.0]

This is a minor release to address a few items for the AOM CTC v2, as well as a
few minor fixes/cleanups.

- Fixes a CIEDE-2000 precision issue, where cross-platform mismatches were seen.
  (#878)
- Adds libvmaf API function vmaf_feature_dictionary_free(). (#879)

## (2021-01-13) [v2.1.1]

This is a minor release to address a few last minute items for the initial AOM CTC.

**New features:**

- Fixes a SSIM/MS-SSIM precision bug where a lossless comparison did not always
  result in a perfect 1.0 score. (#796).
- Adds feature extractor options to clip the dB scores for both PSNR/SSIM.
  --aom_ctc v1.0 has been updated to use these clipping options according to the
  AOM CTC. (#802).

## (2020-12-30) [v2.1.0]

This is a minor release for the initial AOM CTC. Support has been added for
templated feature names. While this is a general purpose software feature,
templated feature names are immediately useful for simultaneous computation of
VMAF and VMAF NEG since the two metrics rely on slightly different VIF/ADM
variations. Global feature overrides via the `--feature` flag are no longer
supported, instead individual models can have their features overloaded
individually, the syntax for which is as follows:

 ```sh
--model version=vmaf_v0.6.1:vif.vif_enhn_gain_limit=1.0:adm.adm_enhn_gain_limit=1.0
```

**New features:**

- Per-model feature overloading via new API `vmaf_model_feature_overload()`.
- Multiple unique configurations of the same feature extractor may be registered
  run at the same time.
- `--aom_ctc v1.0` preset, encompassing all metrics specified by the AOM CTC.

## (2020-12-4) [2.0.0]

**New features:**

- Add PSNR-HVS and CIEDE2000 metrics.
- ci/actions: upload linux/macos artifacts (#738)
- libvmaf/feature: deprecate daala_ssim (#735)
- libvmaf: remove support for pkl models
- libvmaf/psnr: rewrite using integer types, 2x speedup
- vmaf: if no model is specified, enable v0.6.1 by default (#730)
- libvmaf/x86: add AVX2/AVX-512 optimizations for adm, vif and motion
- ci/actions: add xxd to build dependencies for Windows
- libvmaf: add support for built-in models
- libvmaf/integer_vif: use symmetrical mirroring on edges
- Fix log2 by replacing log2f_approx with log2f
- libvmaf_rc: provide a backwards compatible compute_vmaf(), link vmafossexec with
  libvmaf
- libvmaf: add framework support for json models
- libvmaf/libsvm: update libsvm to version 324
- libvmaf/motion: add motion_force_zero to motion fex
- return sha1 if Asset string is longer than 255
- Add CID/iCID Matlab source code
- build: unbreak x86 builds (Fixes: #374)
- Add 12bit and 16bit support for python YUV reader; add tests.
- Add PypsnrFeatureExtractor
- Add processes to FeatureAssembler. (#662)

**Fixed bugs:**

- fix motion flush for single frame input
- Fixing the perf_metric for a single entry list input

## (2020-8-24) [1.5.3]

(Updates since 1.5.1)

**Fixed bugs:**

- Fix inverted height and width in integer_motion in vmaf_rc (#650).

**New features:**

- libvmaf: add support for CSV and JSON logging
- Python: Add an (optional) step in Executor class to do python-based processing
  to ref/dis files (#523).
- Restructure python project and documentation (#544).
- Move test resource to Netflix/vmaf_resource repo (#552).
- Add Github CI (#558).
- Add vmaf_float_v0.6.1neg model; add vif_enhn_gain_limit and adm_enhn_gain_limit
  options to vmaf_rc.
- Update documentation for FFmpeg+libvmaf.
- Improvements to AucPerfMetric (#643).
- Add motion_force_zero option to vmaf_rc.

## (2020-6-30) [1.5.2]

**Fixed bugs:**

- Fix pkgconfig version sync issue (#572)

**New features:**

- libvmaf_rc general improvements

## (2020-2-27) [1.5.1]

**New features:**

- `libvmaf` has been relocated, and now has its own self-enclosed source tree
  (`./libvmaf/`) and build system (`meson`).
- Update license to BSD+Patent.
- Migrate the build system from makefile to meson.
- Introduce a new release candidate API with the associated library `libvmaf_rc`
  and executable `vmaf_rc` under `./libvmaf/build`.
- Add SI and TI feature extractor python classes.
- Add fixed-point SSIM implementation.
- Migrate to python3.

## (2019-9-8) [1.3.15]

**Fixed bugs:**

- Fix a case when CPU cores > 128(MAX_NUM_THREADS) / 3 (#319).
- Avoid dis-filtering ref when not needed, fix return type (#325).
- Update name of file for failed dis_path fopen (#334).
- A few compilation fixes (warnings and errors) (#326).
- Bump up g++ version to 9 for travis (#352).
- Use stat struct instead of ftell to retrieve the file size (#350).

**New features:**

- Write aggregate scores, exec FPS to json output.
- Add support for python3 (#332).
- Print progress in vmafossexec (#337).
- Add VMAF logo.
- Add link to report VMAF bad cases.

## (2019-3-1) [1.3.14]

**Fixed bugs:**

- Fix VMAF value mismatch on 160x90 videos after optimization (#315).
- Fix w10 error with using uninitialized offset_flag variable (#302).

**New features:**

- Add automated Windows builds with AddVeyor (#313).
- Report aggregate CI scores and fix empty model name in log (#304).

## (2019-1-31) [1.3.13]

**New features:**

- Optimized C code for speed. Running in multithreading mode, `vmafossexec`
  achieves ~40% run time reduction compared to the previous version.
- Printed out individual vmaf bootstrap scores in text file from `vmafossexec`.
- refactored windows solution (#283) (#284) (#285) (#291) (#298).

## (2018-12-17) [1.3.11]

**New features:**

- Revise number of bootstrap models definition:
  model/vmaf_rb_v0.6.3/vmaf_rb_v0.6.3.pkl has 21 models (20 bootstrap models and
  one using the full data). From these 21 models, the 20 of them are same as
  v0.6.2, only added an additional bootstrap model.
- Output the per bootstrap model predictions from wrapper/vmafossexec.
- Print bootstrap individual scores in xml and json.
- Add BD-rate calculator and update documentation.
- Report aggregate PSNR, SSIM, and MS-SSIM scores.
- Add sklearn linear regression class to TrainTestModel.
- Enable BRISQUE feature in VMAF training with bootstrapping.
- Add --save-plot option to command line tools.
- Add ST-RREDOpt (time optimized), ST-MAD feature extractors, quality runners and
  unittestts. Refactor ST-RRED feature extractor. (#216)

**Fixed bugs:**

- Bug fixed. When start vmaf in multi-thread at the same time. (#239)
- Fix name of min function in vmaf.h and vmaf.cpp. (#227)
- Fix implicit declaration of functions (#225)

## (2018-9-13) [1.3.10]

**New features:**

- Remove sureal as a submodule to vmaf. sureal is now available through pip install.

## (2018-8-7) [1.3.9]

**Fixed bugs:**

- libvmaf: fix case where user defined read_frame() callback was being ignored.

## (2018-6-21) [1.3.8]

**Fixed bugs:**

- Fix compute_vmaf boolean type issue (#178).

## (2018-6-12) [1.3.7]

**New features:**

- Add the --ci option to calculate confidence intervals to predicted VMAF scores
  (run_vmaf, run_vmaf_in_batch, ffmpeg2vmaf, vmafossexec).
- Update libvmaf version to 1.3.7 after compute_vmaf() interface change (added
  enable_conf_interval option).
- Add new models: 1) model/vmaf_4k_v0.6.1.pkl for 4KTV viewing at distance 1.5H,
  2) model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl for VMAF prediction with a confidence
  interval, 3) model/vmaf_4k_rb_v0.6.2/vmaf_4k_rb_v0.6.2.pkl for 4KTV viewing at
  distance 1.5H, with a confidence interval.

## (2018-6-4) [1.3.6]

**New features:**

- Update libvmaf version to 1.3.6 (to make consistent with VDK version from now
  on) after compute_vmaf() interface change (added thread and subsample options).
- Add the option to set the number of threads to use in vmafossexec.
- Add the option to subsample frames to save computation in vmafossexec.

## (2018-5-23) [1.3.5]

**New features:**

- Add multi-threading to vmafossexec.

## (2018-5-8) [1.3.4]

**Refactoring:**

- Refactor mos out of vmaf repo; rename to sureal as submodule.
- Refactor TrainTestModel to make predict() to output dictionary.
- Refactor TrainTestModel.
- Rename KFLK metric to AUC (Area Under the Curve) for better interpretability.

**New features:**

- Add bootstrapping to VMAF. Add two new classes BootstrapVmafQualityRunner and
  BaggingVmafQualityRunner
- Add Resolving Power Performance Metric.
- Add BRISQUE and NIQE feature extractors. Added two new classes
  BrisqueNorefFeatureExtractor and NiqeNorefFeatureExtractor. Add
  NiqeQualityRunner.

**Fixed bugs:**

- Add .gitattributes (#127). Force .pkl and .model files to retain LF line-ending.
  Required for use on Windows where model files would otherwise be checked out as
  CRLF which VMAF's parser doesn't handle.
- Allow MinGW compilation of ptools (#133). ptools doesn't build on MinGW as *nix
  socket headers are included. This patch selects Windows headers for MinGW
  builds.
- Update compute vmaf interface (#138). Update VMAF version in libvmaf.pc and etc.
  Catch logic error (resulted from wrong model file format) in compute_vmaf(). Use
  custom error code.

## (2017-12-3) [1.3.3]

**Fixed bugs:**

- Update VMAF version to 0.6.2 after compute_vmaf() interface change (#124).

## (2017-12-3) [1.3.2]

**Refactoring:**

- Lift check for exec existence during program load.
- Refactor psnr, ssim, ms_ssim and vmaf_feature to call ExternalProgramCaller.
- Refactor feature/Makefile to make executables depend on libvmaf.a.
- Refactor wrapper/Makefile to include additional objs in libvmaf.a but exclude
  main.o.
- Remove ar -d command after removing main.o from libvmaf.a.

**New features:**

- Generalize read_dataset.
- Update default Asset resampling method to bicubic (#116).
- Extend ffmpeg2vmaf script to allow ref/dis input to be YUV (#118).
- Improve README.md (#121).

**Fixed bugs:**

- Temporary fix Visual Studio builds (#112).
- Avoid unnecessary dependency on matplotlib in run_vmaf (#114).
- Remove unneeded dependencies in Dockerfile, fixes #115 (#117).
- MinGW support (#123).
- Change compute_vmaf() interface to return an error code instead of throw an
  error #124 (#126).

## (2017-8-12) [1.3.1]

**Refactoring:**

- Refactor NorefExecutorMixin to eliminate repeated codes.
- Refactor C code: get rid of unused double functions; uniformly use read_frame
  callback function to void repeated code;
- Add strip option to Makefile.

**New features:**

- Update Asset class: add copy functions to Asset; add ref/dis_yuv_type; deprecate
  yuv_type; add ref/dis_start_sec;
- Update subjective models: add confidence interval to subjective model
  parameters; refactor MLE model and make subclasses; add run_subj command line.
- Recommend pip, add ffmpeg2vmaf info and reorganize prerequisite installation (#88).
- Reduce sleep time in parallel_map.
- Add library interface for VMAF (#90).
- Add VisualStudio2015 support (#92).
- Add example of image dataset notyuv.
- Add pkgconfig file and changed Makefile.
- Add VmafPhoneQualityRunner class.
- Add DMOS_MLE_CO subjective model.

**Fixed bugs:**

- Update RegressionMixin to handle AUC exception for dicitonary-style dataset.
- Fix Makefile fedora libptools issue. (#98)

## (2017-4-13) [1.2.4]

**Refactoring:**

- Deprecate run_executors_in_parallel.
- Refactor NorefFeatureExtractor into NorefExecutorMixin so that it can be used
  for all executors.
- Add abstract methods to some base classes.

**New features:**

- Add ST-RRED runner (StrredQualityRunner), based on "Video Quality Assessment by
  Reduced Reference Spatio-Temporal Entropic Differencing", by R. Soundararaajan,
  A. Bovik.
- Add start/end frame support for Executor.

## (2017-3-8) [1.2.3]

**New features:**

- Refactor to replace config.ROOT with config.VmafConfig.

## (2017-3-1) [1.2.2]

**New features:**

- Generalize Result and FileSystemResultStore to allow None values.

## (2017-2-27) [1.2.1]

**Tasks:**

- Refactor to prepare for pypi packaging.

## (2017-2-20) [1.2.0]

**New features:**

- Updated VMAF model to version v0.6.1. Changes include: 1) added a custom model
  for cellular phone screen viewing; 2) trained using new dataset, covering more
  difficult content; 3) elementary metric fixes: ADM behavior at near-black
  frames, motion behavior at scene boundaries; 4) compressed quality score range
  by 20% to accommodate higher dynamic range; 5) Use MLE instead of DMOS as
  subjective model.

## (2017-1-24) [1.1.23]

**Fixed bugs:**

- Replace subprocess.call with run_process (checking return value).

## (2017-1-22) [1.1.22]

**New features:**

- Add command line ffmpeg2vmaf, which takes encoded videos as input.

## (2017-1-18) [1.1.21]

**New features:**

- Allow processing non-YUV input videos.

## (2016-12-20) [1.1.20]

**New features:**

- Add STRRED runner.

## (2016-12-19) [1.1.19]

**New features:**

- Allow specifying crop and pad parameter in dataset files.

## (2016-12-8) [1.1.18]

**Fixed bugs:**

- Replace pathos with custom function for parallel executor running.

## (2016-12-8) [1.1.17]

**Fixed bugs:**

- Fix command line run_testing issue. Add command line test cases.

## (2016-12-5) [1.1.16]

**New features:**

- Speed up VMAF convolution operation by AVX.

## (2016-11-30) [1.1.15]

**Fixed bugs:**

- Fix vmafossexec memory leakage.

## (2016-11-28) [1.1.14]

**New features:**

- Add enable_transform_score option to VmafQualityRunner, VmafossExecQualityRunner.

## (2016-11-18) [1.1.13]

**Fixed bugs:**

- Fix a bug in DatasetReader.to_aggregated_dataset_file.

## (2016-11-15) [1.1.12]

**New features:**

- Add Travis continuous integration.

## (2016-11-11) [1.1.11]

**New features:**

- Add implementation of AUC (Area Under the Curve) - quality metric evaluation
  method based on AUC. Refer to: L. Krasula, K. Fliegel, P. Le Callet, M.Klima,
  "On the accuracy of objective image and video quality models: New methodology
  for performance evaluation", QoMEX 2016.

## (2016-11-07) [1.1.10]

**New features:**

- Add options to use custom subjective models in run_vmaf_training and run_testing
  commands.

## (2016-11-02) [1.1.9]

**New features:**

- Add DatasetReader and subclasses; add SubjectiveModel and subclasses.

## (2016-10-19) [1.1.8]

**New features:**

- Add quality runners for each individual VMAF elementary metrics.

## (2016-10-14) [1.1.7]

**Fixed bugs:**

- Issue #36: SSIM and MS-SSIM sometimes get negative values.

## (2016-10-10) [1.1.6]

**New features:**

- Add Xcode project support.
- Add more pooling options (median, percx) to CLIs.

## (2016-10-8) [1.1.5]

**New features:**

- Add support for docker usage (#30).

## (2016-10-7) [1.1.4]

**Fixed bugs:**

- Issue #29: Make ptools build under Fedora.

## (2016-10-6) [1.1.3]

**New features:**

- Generalize dataset format to allow per-content YUV format.

## (2016-10-5) [1.1.2]

**Fixed bugs:**

- Make ptools work under Mac OS.
- Update SklearnRandomForestTrainTestModel test with sklearn 0.18.

## (2016-09-29) [1.1.1]

**New features:**

- Update command lines run_vmaf, run_psnr, run_vmaf_in_batch, run_cleaning_cache,
  run_vmaf_training and run_testing.

## (2016-09-28) [1.1.0]

**New features:**

- Update wrapper/vmafossexec: 1) it now takes pkl model file as input, so that
  slopes/intercepts are no longer hard-coded; 2) it now takes multiple YUV input
  formats; 3) add flag to enable/disable VMAF score clipping at 0/100; 4) allow
  customly running PSNR/SSIM/MS-SSIM; 5) allow customly outputing XML/JSON
- Add SSIM/MS-SSIM option in run_testing.

## (2016-09-09) [1.0.9]

**Fixed bugs:**

- Move VmafQualityRunnerWithLocalExplainer to quality_runner_adhoc to resolve
  multiple instances of VMAF found when calling QualityRunner.find_subclass.

**New features:**

- Add custom_clip_0to1 to TrainTestModel.

## (2016-09-07) [1.0.8]

**New features:**

- Generalize read_dataset to allow specifying width, height and resampling method
  on which to calculate quality.
- Add bicubic to SUPPORTED_RESAMPLING_TYPES for Asset.
- Update Asset rule with resampling_type in **str** to avoid duplicates in data
  store.

## (2016-08-20) [1.0.7]

**New features:**

- Update VmafFeatureExtractor to 0.2.2b with scaled ADM features exposed (adm_scale0-3).

## (2016-08-20) [1.0.6]

**New features:**

- Add DisYUVRawVideoExtractor and related classes.
- Add NeuralNetworkTrainTestModel base class that integrates TensorFlow.
- Add example class ToddNoiseClassifierTrainTestModel.

## (2016-08-20) [1.0.5]

**New features:**

- Add LocalExplainer class.
- Add show_local_explanation option to run_vmaf script.

## (2016-07-21) [1.0.4]

**Fixed bugs:**

- Fix a series of numerical issues in VMAF features, increment
  VmafFeatureExtractor version number.
- Retrain VmafQualityRunner after feature update, increment version number.

## (2016-07-20) [1.0.3]

**New features:**

- Add base class NorefFeatureExtractor for any feature extractor that do not
  use a reference video.
- Add MomentNorefFeatureExtractor subclassing NorefFeatureExtractor as an example
  implementation.

## (2016-06-16) [1.0.2]

**New features:**

- Refactor feature code to expose ssim/ms-ssim, speed up ssim/ms-ssim.

## (2016-06-10) [1.0.1]

**Fixed bugs:**

- Fix feature while looping by moving feof to after read_image.
- Fix issue #2 use hashed string for log filename and result filename to avoid
  file names getting too long.

**New features:**

- Add SsimFeatureExtractor and MsSsimFeatureExtractor with intermediate features
  (luminence, contrast, structure).
