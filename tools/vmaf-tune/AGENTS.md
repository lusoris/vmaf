# `tools/vmaf-tune/` — agent notes

Quality-aware encode automation harness. See
[`docs/adr/0237-quality-aware-encode-automation.md`](../../docs/adr/0237-quality-aware-encode-automation.md)
for the umbrella spec and
[`docs/research/0044-quality-aware-encode-automation.md`](../../docs/research/0044-quality-aware-encode-automation.md)
for the option-space digest.

## Rebase-sensitive invariants

- **The Phase A JSONL corpus row schema is the API contract for Phase
  B / C.** Phase B (target-VMAF bisect) and Phase C (per-title CRF
  predictor) read corpora produced by this tool. Adding optional keys
  with a default is fine; renaming or removing keys, or changing their
  type/semantics, requires bumping `vmaftune.SCHEMA_VERSION` and
  updating every downstream consumer in the same PR. The canonical
  key list lives in `src/vmaftune/__init__.py` (`CORPUS_ROW_KEYS`)
  and is asserted on every emitted row by `corpus._row_for`.
- **The `vmaf_model` JSONL field is now per-row, not per-job.** Since
  ADR-0289 (resolution-aware model selection), `corpus._row_for`
  populates `vmaf_model` from `score_res.request.model`, which in
  turn comes from `resolution.select_vmaf_model_version(width, height)`
  when `CorpusOptions.resolution_aware` is True. Mixed-ladder corpora
  legitimately contain multiple distinct `vmaf_model` values across
  rows. Downstream consumers (Phase B/C/D) must group/filter by
  `vmaf_model` rather than assuming a constant.
- **`resolution.py` decision rule is height-only.** `height >= 2160`
  picks `vmaf_4k_v0.6.1`; everything else picks `vmaf_v0.6.1`. Width
  is accepted in the API for symmetry but ignored in the body. Do not
  add per-codec / per-pixel-count branches without an ADR-0289
  follow-up — the rule mirrors Netflix's published guidance and is
  the only defensible default until the fork ships its own
  intermediate models.
- **The codec-adapter contract is multi-codec from day one.** Phase A
  wires `libx264` end-to-end; `libaom-av1`
  ([ADR-0279](../../docs/adr/0279-vmaf-tune-codec-adapter-libaom.md))
  joins as a metadata-and-argv-helper adapter (its argv shape uses
  `-cpu-used`, not `-preset`, so the encode driver gains a second
  argv path when the codec-pluggable encode wiring lands).
  `codec_adapters/__init__.py` exposes a registry the search loop
  must use uniformly. Do not branch on codec name in `corpus.py` /
  `encode.py` / `score.py`; route via the adapter. New codecs are
  one-file additions under `codec_adapters/`.
- **Adapter preset vocabulary is the cross-codec sweep axis.** The
  ten-name preset tuple (`placebo, slowest, slower, slow, medium,
  fast, faster, veryfast, superfast, ultrafast`) is shared across
  AV1-family adapters so a single `--preset` axis covers x264 / x265
  / svtav1 / libaom-av1 in one sweep. Each adapter maps the name
  onto its codec-specific knob (cpu-used, preset enum, ...). Do not
  introduce per-adapter preset names; if the codec needs a knob the
  shared vocabulary cannot express, route it through `extra_params`
  rather than splitting the preset axis.
- **The codec-adapter contract is multi-codec from day one.**
  `codec_adapters/__init__.py` exposes a registry the search loop must
  use uniformly. Do not branch on codec name in `corpus.py` /
  `encode.py` / `score.py`; route via the adapter. New codecs are
  one-file additions under `codec_adapters/`. Wired today: `libx264`
  (Phase A scaffold) and `libx265` (ADR-0288). One narrow exception
  lives in `encode.parse_versions(stderr, encoder=…)` — the per-codec
  banner regex (x264's `x264 - core <N>` vs x265's
  `x265 [info]: HEVC encoder version <V>`) cannot be expressed as a
  single pattern, so the function dispatches on the encoder name. This
  branch is allowed; the corpus emitter and the search loop must still
  go through the registry.
  wires `libx264` plus the NVENC family (`h264_nvenc`,
  `hevc_nvenc`, `av1_nvenc` — see
  [ADR-0290](../../docs/adr/0290-vmaf-tune-nvenc-adapters.md)).
  `codec_adapters/__init__.py` exposes a registry the search loop
  must use uniformly. Do not branch on codec name in `corpus.py` /
  `encode.py` / `score.py`; route via the adapter. New codecs are
  one-file additions under `codec_adapters/`. Hardware-encoder
  families share private helpers (e.g. `_nvenc_common.py`) — keep
  the mnemonic preset map and CQ window in one place per family so
  the per-codec files stay thin.
- **Subprocess boundary is the test seam.** `encode.run_encode` and
  `score.run_score` accept a `runner` argument that defaults to
  `subprocess.run`. Tests inject a fake; production callers leave it
  default. Do not reach for `os.system` / `popen` shortcuts —
  `tests/test_corpus.py` will silently stop covering the path.
- **Fast-path is opt-in; the grid stays canonical
  ([ADR-0276](../../docs/adr/0276-vmaf-tune-fast-path.md)).** The
  `fast` subcommand under `src/vmaftune/fast.py` accelerates the
  *recommendation* use case via proxy + Bayesian + GPU-verify, but
  must never automatically replace the Phase A grid path. The grid
  is the ground-truth corpus generator that Phase B/C/D consume;
  removing or re-routing it breaks the Phase A.5 → Phase A
  fallback contract for proxy-OOD sources. The `fast` subcommand
  surfaces its smoke vs production mode in the CLI output's
  `notes` field — keep that visibility when extending the loop.
- **Optuna is an optional runtime dep.** Importing it at module
  scope outside `src/vmaftune/fast.py` (or its tests) is a bug —
  the core install path stays zero-dep so corpus generation works
  on hosts that never run the fast path. The lazy-import guard in
  `fast.py` is the only correct entry point; tests that exercise
  `fast.py` use `pytest.importorskip("optuna")`.
- **`recommend` is a pure consumer of the corpus schema.** The
  `recommend` subcommand reads `vmaf_score`, `bitrate_kbps`, `crf`,
  `preset`, `encoder`, `exit_status` directly from rows produced by
  `corpus.py` (or loaded via `--from-corpus` from a previous run).
  No new schema, no parallel data path. If `SCHEMA_VERSION` bumps,
  `recommend.py`'s row-reader is one of the downstream consumers
  that must be updated in the same PR — the contract is checked by
  `test_recommend.py` against `CORPUS_ROW_KEYS`.
- **Predicate semantics are part of the user-visible contract.**
  `--target-vmaf T` returns the *smallest CRF* whose `vmaf_score >=
  T` (falling back to closest-miss when nothing clears, marked
  `(UNMET)`). `--target-bitrate KBPS` returns the row with minimum
  `|bitrate_kbps - KBPS|`, ties broken by smaller CRF. The two
  flags are mutually exclusive at the argparse layer (exit code 2
  when both are passed). Changing any of these defaults is a
  user-visible behaviour change requiring an ADR.
- **AMF preset compression is fixed (ADR-0282).** The 7-into-3 preset
  table in `codec_adapters/_amf_common.py` (`_PRESET_TO_AMF`) is the
  cross-codec axis Phase B / C consumers depend on. Do not extend
  `presets` beyond the canonical 7 names without amending ADR-0282 —
  the registry uniformity that lets the search loop ignore codec
  identity rests on every codec accepting the same preset vocabulary.
  AV1 (`av1_amf`) is RDNA3+ only; `ensure_amf_available` is the
  runtime gate.

## Phase scope

Phase A (this scaffold): grid sweep + JSONL emit, x264 only.
Phase A.5 (this PR): opt-in `fast` subcommand scaffold (proxy +
Bayesian + GPU-verify, smoke-mode validated; production loop
deferred to follow-up). Phases B–F per ADR-0237 are explicitly out
of scope here; do not add bisect / predictor / ladder / MCP code
into this tree without an ADR-0237 follow-up promoting the
corresponding phase.
Phase A (this scaffold): grid sweep + JSONL emit. Wired codecs:
`libx264` (initial scaffold) and `libx265` (ADR-0288). Further codecs
(`libsvtav1`, `libvpx-vp9`, `libvvenc`, `libaom`, neural-codec extras)
are one-file adapter additions under `codec_adapters/` per ADR-0237.
Phases B–F per ADR-0237 are explicitly out of scope here; do not add
bisect / predictor / ladder / MCP code into this tree without an
ADR-0237 follow-up promoting the corresponding phase.
  wired `libx264`; ADR-0281 widened the registry with the three
  Intel QSV adapters (`h264_qsv`, `hevc_qsv`, `av1_qsv`). The
  search loop must use the registry uniformly. Do not branch on
  codec name in `corpus.py` / `encode.py` / `score.py`; route via
  the adapter. New codecs are one-file additions under
  `codec_adapters/`.
- **The QSV adapters share `_qsv_common.py`.** Three encoders with
  identical parameter shape (preset vocabulary, ICQ
  `global_quality` window) is a deliberate exception to the
  "one file per codec, nothing shared" Phase A convention. Per
  ADR-0281, future codec families that share parameter shape
  (NVENC's three encoders, AMF's three encoders) follow the same
  pattern: one `_<family>_common.py` private module, three thin
  dataclass adapters. Single-codec families stay flat.
- **The encode pipeline (`encode.py`) is still x264-CRF-tied.**
  ADR-0281 added the QSV adapter classes but did not widen
  `build_ffmpeg_command` to dispatch on `adapter.quality_knob`.
  Until that follow-up lands, the QSV adapters validate
  `(preset, global_quality)` correctly but the harness will not
  yet successfully drive a QSV encode end-to-end.
- **Subprocess boundary is the test seam.** `encode.run_encode`,
  `score.run_score`, and the QSV `ffmpeg_supports_encoder` probe
  accept a `runner` argument that defaults to `subprocess.run`.
  Tests inject a fake; production callers leave it default. Do
  not reach for `os.system` / `popen` shortcuts —
  `tests/test_corpus.py` and `tests/test_codec_adapter_qsv.py`
  will silently stop covering the path.

## Phase scope

Phase A (the original scaffold): grid sweep + JSONL emit, x264
only. ADR-0281 added the three QSV codec adapters as a one-file
extension off the registry; the encode-pipeline widening that
makes them functional is itself a separate Phase A follow-up.
Phases B–F per ADR-0237 (bisect / predictor / ladder / MCP) remain
explicitly out of scope here; do not add that code into this tree
without an ADR-0237 follow-up promoting the corresponding phase.
