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
- **The codec-adapter contract is multi-codec from day one.** Phase A
  only wires `libx264`, but `codec_adapters/__init__.py` exposes a
  registry the search loop must use uniformly. Do not branch on codec
  name in `corpus.py` / `encode.py` / `score.py`; route via the
  adapter. New codecs are one-file additions under
  `codec_adapters/`.
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

## Phase scope

Phase A (this scaffold): grid sweep + JSONL emit, x264 only.
Phase A.5 (this PR): opt-in `fast` subcommand scaffold (proxy +
Bayesian + GPU-verify, smoke-mode validated; production loop
deferred to follow-up). Phases B–F per ADR-0237 are explicitly out
of scope here; do not add bisect / predictor / ladder / MCP code
into this tree without an ADR-0237 follow-up promoting the
corresponding phase.
