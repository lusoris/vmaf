# vmaf-tune (Phase A scaffold)

Quality-aware encode automation harness for the lusoris vmaf fork.

Phase A scope (this PR): drives `ffmpeg`/`libx264` over a parameter grid,
scores each encode with the `vmaf` CLI, emits a JSONL corpus.

Phase B (target-VMAF bisect) and Phase C (per-title CRF predictor) are
NOT implemented here — see [ADR-0237](../../docs/adr/0237-quality-aware-encode-automation.md)
for the roadmap and [Research-0044](../../docs/research/0044-quality-aware-encode-automation.md)
for the option-space digest.

User documentation: [`docs/usage/vmaf-tune.md`](../../docs/usage/vmaf-tune.md).

## Layout

```
tools/vmaf-tune/
  pyproject.toml
  vmaf-tune                       # console entry-point shim (thin wrapper)
  src/vmaftune/
    __init__.py                   # version + public API
    cli.py                        # argparse wiring
    encode.py                     # ffmpeg/x264 driver (subprocess)
    score.py                      # vmaf binary driver (subprocess)
    corpus.py                     # grid sweep orchestrator + JSONL writer
    codec_adapters/
      __init__.py
      x264.py                     # CodecAdapter for libx264 (Phase A only)
  tests/
    test_corpus.py                # smoke test (mocks subprocess)
```

## Quick start

```bash
# from repo root
pip install -e tools/vmaf-tune
vmaf-tune corpus \
    --source path/to/ref.yuv --width 1920 --height 1080 --pix-fmt yuv420p \
    --preset medium --preset slow \
    --crf 22 --crf 28 --crf 34 \
    --output corpus.jsonl
```

Each emitted row has the schema documented in [`docs/usage/vmaf-tune.md`](../../docs/usage/vmaf-tune.md).
The schema is the API contract that Phase B/C will consume; do not
change it without bumping `SCHEMA_VERSION` in `src/vmaftune/__init__.py`.

## Tests

```bash
pytest tools/vmaf-tune/tests/
```

The shipped smoke mocks `subprocess.run` so it requires neither `ffmpeg`
nor a built `vmaf` binary.
