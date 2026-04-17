# `--precision` — score output precision

`--precision` controls how VMAF scores are formatted in XML / JSON / CSV / SUB
output and on stderr. Fork-added per
[ADR-0006](../adr/0006-cli-precision-17g-default.md); the default changed from
upstream's `%.6f` to `%.17g` (IEEE-754 round-trip lossless).

## Grammar

```
--precision N          # integer 1..17 → printf "%.<N>g"
--precision max        # alias for "%.17g" (the default)
--precision full       # alias for "%.17g"
--precision legacy     # "%.6f" — pre-fork Netflix compatibility
```

Defaults (no `--precision`): `%.17g`.

## When to pick each

| Mode | Use when |
| --- | --- |
| `max` / `full` / no flag | **Default, always correct.** Scores re-parse to the exact same double. Required for any archival report or cross-backend diff. |
| `legacy` | Reproducing pre-fork Netflix output byte-for-byte (e.g. validating a migration). Never for new data. |
| `N = 3` to `N = 6` | Human reading on a terminal where you want short scores. Do not pipe this into a tool that compares scores. |
| `N = 7` to `N = 16` | Intermediate — no common reason; usually you want `max`. |
| `N = 17` | Equivalent to `max` explicitly. |

## Why `%.17g` is the default

IEEE-754 double precision holds ~15.95 significant decimal digits. `%.17g` is
the *minimum* printf format that guarantees:

```
parse(print(x)) == x    for every finite double x
```

Anything shorter (`%.15g`, `%.6f`) can round-trip-corrupt scores that differ by
one ULP — exactly the resolution at which cross-backend diffs matter. See
[../benchmarks.md](../benchmarks.md) and
[../backends/index.md](../backends/index.md) for the cross-backend ULP
budget.

## Effect on all output channels

The same format applies uniformly to:

- Pooled stderr line (`vmaf_v0.6.1: 76.66890501970558`)
- XML per-frame attribute values
- JSON per-frame + pooled numbers
- CSV cells
- SubRip `<subtitle>` text

There is no way to pick different precisions for different outputs in one
invocation — this is intentional so that `output.xml`, `output.json`, and the
stderr line always agree.

## Example — round-trip check

```shell
./build/libvmaf/tools/vmaf \
  --reference  src01_hrc00_576x324.yuv \
  --distorted  src01_hrc01_576x324.yuv \
  --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
  --output scores.json --json
```

Default output (lossless):

```json
{"vmaf": {"mean": 76.66890501970558, ...}}
```

Same run with `--precision legacy`:

```json
{"vmaf": {"mean": 76.668905, ...}}
```

The legacy form drops ~9 significant digits. Across the 3 Netflix CPU golden
pairs, the pre-fork default and the new `%.17g` default agree to 6 decimals —
but that is the exact margin at which SIMD / GPU backends deviate, so archival
reports must use the default.

## Interaction with goldens

The 3 Netflix CPU golden tests
([ADR-0024](../adr/0024-netflix-golden-preserved.md)) still pass bit-for-bit
with the default `%.17g` — the hardcoded Python `assertAlmostEqual` tolerances
are round-trip-safe margins, not exact-string compares.

## Related

- [cli.md](cli.md) — full CLI reference, `--precision` summary row.
- [ADR-0006](../adr/0006-cli-precision-17g-default.md) — rationale for the
  default change.
- [../benchmarks.md](../benchmarks.md) — fork-added benchmark numbers, all
  reported at `%.17g`.
