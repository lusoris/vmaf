# Bisect-model-quality fixture cache

Deterministic placeholder cache used by [`.github/workflows/nightly-bisect.yml`](../../../.github/workflows/nightly-bisect.yml).
Regenerable from [`ai/scripts/build_bisect_cache.py`](../../scripts/build_bisect_cache.py).

## Layout

```
features.parquet      256 rows × 6 default features + mos column
models/model_NN.onnx  8 linear FR models (input "input", output "score")
```

Total committed size: ~16 KB.

## Why these bytes are committed

CI re-runs `build_bisect_cache.py --check` before exercising the bisect.
The check regenerates everything from fixed seeds and asserts byte-equality
against the committed tree. This guards against silent drift in
`pandas` / `pyarrow` / `onnx` serialisers between Python versions and
catches accidental edits.

## Why the timeline is regression-free

All 8 committed models are tiny perturbations of the optimal weights
(`weights = ones / N_FEATURES + N(0, 1e-4)`), so `bisect-model-quality
--min-plcc 0.85` returns `first_bad_index = None`. A red nightly means
the wiring broke (parquet drift, onnx drift, runtime crash), not that
a real quality model regressed.

The synthetic-regression case — "introducing a bad ONNX trips the
alert" — is covered by
[`ai/tests/test_bisect_model_quality.py::test_bisect_localises_first_bad`](../../tests/test_bisect_model_quality.py),
which builds a `bad_from=5` timeline at runtime.

## Why this is a placeholder, not a real golden cache

A real cache would draw features from a frozen YUV subset of NFLX-public
/ LIVE / KonIQ with attached DMOS labels, run through `libvmaf`. That
requires (a) the source datasets in tree or fetched from a stable
mirror, (b) DMOS labels for each clip, (c) a frozen feature-extractor
build. None of those exist yet for the fork. See
[Research-0001](../../../docs/research/0001-bisect-model-quality-cache.md)
for the design space and the swap path.

## Regeneration

```bash
python ai/scripts/build_bisect_cache.py            # rewrite in place
python ai/scripts/build_bisect_cache.py --check    # CI-style drift check
```

After regenerating, commit the resulting parquet + ONNX bytes.
