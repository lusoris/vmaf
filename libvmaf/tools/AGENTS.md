# AGENTS.md — libvmaf/tools

Orientation for agents working on the CLI binaries. Parent:
[../AGENTS.md](../AGENTS.md).

## Scope

Two C binaries built by libvmaf's Meson tree:

- `vmaf` — the end-user scoring CLI
- `vmaf_bench` — micro-benchmark harness for extractors and backends

```
tools/
  vmaf.c              # main() + option dispatch for the vmaf CLI
  vmaf_bench.c        # main() + benchmark harness
  cli_parse.c/.h      # shared option parser (--precision, --tiny-model, …)
```

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md)).
- **Default numeric precision is `%.17g`** (IEEE-754 round-trip lossless).
  `--precision=N` overrides; `--precision=max` aliases `%.17g`. See
  [ADR-0006](../../docs/adr/0006-cli-precision-17g-default.md). This
  applies to both stderr and file outputs (XML / JSON / CSV / sub-XML).
- **`--tiny-model PATH`** loads an ONNX checkpoint via
  [src/dnn/](../src/dnn/AGENTS.md). Path is resolved via `realpath` inside
  the loader; the CLI passes the string through unchanged. See
  [ADR-0023](../../docs/adr/0023-tinyai-user-surfaces.md).
- **No new hard dependencies** — the CLI must still build when `enable_dnn=disabled`.

## Governing ADRs

- [ADR-0006](../../docs/adr/0006-cli-precision-17g-default.md) — `%.17g` default + `--precision` flag.
- [ADR-0023](../../docs/adr/0023-tinyai-user-surfaces.md) — `--tiny-model` as one of four tiny-AI surfaces.
