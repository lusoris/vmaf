# AGENTS.md — libvmaf/tools

Orientation for agents working on the CLI binaries. Parent:
[../AGENTS.md](../AGENTS.md).

## Scope

Three C binaries built by libvmaf's Meson tree:

- `vmaf` — the end-user scoring CLI
- `vmaf_bench` — micro-benchmark harness for extractors and backends
- `vmaf-perShot` — per-shot CRF predictor sidecar (T6-3b / ADR-0222)

```text
tools/
  vmaf.c              # main() + option dispatch for the vmaf CLI
  vmaf_bench.c        # main() + benchmark harness
  vmaf_per_shot.c     # main() + scan/predict for the perShot sidecar
  cli_parse.c/.h      # shared option parser (--precision, --tiny-model, …)
```

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md)).
- **Default numeric precision is `%.6f`** (Netflix-compatible — required by
  CLAUDE.md §8 golden gate). `--precision=max` (alias `full`) opts in to
  `%.17g` (IEEE-754 round-trip lossless). `--precision=N` overrides with
  `"%.<N>g"`; `--precision=legacy` is preserved as a synonym for the default.
  See [ADR-0119](../../docs/adr/0119-cli-precision-default-revert.md)
  (supersedes [ADR-0006](../../docs/adr/0006-cli-precision-17g-default.md)).
  This applies to both stderr and file outputs (XML / JSON / CSV / sub-XML).
- **`--tiny-model PATH`** loads an ONNX checkpoint via
  [src/dnn/](../src/dnn/AGENTS.md). Path is resolved via `realpath` inside
  the loader; the CLI passes the string through unchanged. See
  [ADR-0023](../../docs/adr/0023-tinyai-user-surfaces.md).
- **No new hard dependencies** — the CLI must still build when `enable_dnn=disabled`.
- **`--frame_skip_ref` / `--frame_skip_dist`** pre-loops in
  [vmaf.c](vmaf.c) MUST `vmaf_picture_unref()` each fetched picture
  immediately. The picture pool is always-on (see ADR-0104 below) and
  fixed-size; without unref the pool exhausts after N skips and the next
  fetch blocks indefinitely. Re-test with
  `python -m pytest python/test/command_line_test.py
  ::VmafexecCommandLineTest::test_run_vmafexec_with_frame_skipping` — if
  it hangs (timeout, no output), the unref is missing or wrong.

## Governing ADRs

- [ADR-0119](../../docs/adr/0119-cli-precision-default-revert.md) — `%.6f`
  default (Netflix-compat) + `--precision=max` for round-trip lossless.
  Supersedes ADR-0006.
- [ADR-0006](../../docs/adr/0006-cli-precision-17g-default.md) — *Superseded.*
  Original `%.17g`-default decision; kept for history.
- [ADR-0023](../../docs/adr/0023-tinyai-user-surfaces.md) — `--tiny-model`
  as one of four tiny-AI surfaces.
- [ADR-0222](../../docs/adr/0222-vmaf-per-shot-tool.md) — `vmaf-perShot`
  per-shot CRF predictor sidecar (T6-3b).
  - **Sidecar invariant**: this binary is **standalone** —
    it does not link the libvmaf metric path; its output is
    an encoder hint, not a quality score. Any future
    integration must keep the per-shot prediction outside
    `vmaf_score_*` to preserve roadmap §2.4's separation.
  - **Schema invariant**: CSV / JSON columns
    (`shot_id`, `start_frame`, `end_frame`, `frames`,
    `mean_complexity`, `mean_motion`, `predicted_crf`)
    are stable across v1; v2's trained MLP must reuse
    them to avoid downstream encoder churn.
- [ADR-0104](../../docs/adr/0104-picture-pool-always-on.md) — picture
  pool is always compiled in and sized for the live-picture set; this
  is what makes the `--frame_skip_*` unref invariant load-bearing.
