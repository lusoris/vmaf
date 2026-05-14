# CHUG UGC-HDR ingestion

`ai/scripts/chug_to_corpus_jsonl.py` ingests the CHUG UGC-HDR dataset
into the fork's MOS-corpus JSONL shape.

## What CHUG Is

CHUG is the Crowdsourced User-Generated HDR Video Quality Dataset:
5,992 bitrate-ladder HDR videos derived from 856 UGC-HDR references,
with AMT subjective ratings and portrait/landscape coverage.

Repository: <https://github.com/shreshthsaini/CHUG>
Paper DOI: <https://doi.org/10.1109/ICIP55913.2025.11084488>

## License Posture

The CHUG README badge says CC BY-NC 4.0, while `license.txt` contains
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 text. Treat
CHUG as non-commercial/share-alike research data until that mismatch is
clarified. Do not commit CHUG CSVs, videos, JSONL rows, feature caches,
or local trained heads.

## Quick Start

```bash
mkdir -p .workingdir2/chug
curl -L https://raw.githubusercontent.com/shreshthsaini/CHUG/master/chug.csv \
  -o .workingdir2/chug/manifest.csv
curl -L https://raw.githubusercontent.com/shreshthsaini/CHUG/master/chug-video.txt \
  -o .workingdir2/chug/chug-video.txt

PYTHONPATH=ai/src python ai/scripts/chug_to_corpus_jsonl.py --chug-dir .workingdir2/chug
```

The default run caps at `--max-rows 500`. Use `--full` for all 5,992
manifest rows:

```bash
PYTHONPATH=ai/src python ai/scripts/chug_to_corpus_jsonl.py \
  --chug-dir .workingdir2/chug \
  --output .workingdir2/chug/chug.jsonl \
  --full \
  --verbose
```

The run is resumable through `.workingdir2/chug/.download-progress.json`.
The adapter downloads each MP4 via `curl`, probes it with `ffprobe`,
deduplicates by SHA-256, and appends JSONL rows.

## Output Schema

The common MOS-corpus fields match [mos-corpora.md](mos-corpora.md):
`src`, `src_sha256`, geometry, `mos`, `mos_std_dev`, `corpus`,
`corpus_version`, and ingest timestamp.

CHUG-specific fields are also preserved:

| Field | Meaning |
|---|---|
| `mos_raw_0_100` | Source `mos_j` value from CHUG's 0-100 MOS axis. |
| `chug_video_id` | Hashed video ID used in the S3 URL. |
| `chug_ref` | Reference flag from the CHUG CSV. |
| `chug_bitladder` | Bitrate-ladder label such as `360p_0.2M_`. |
| `chug_resolution` | Manifest resolution label. |
| `chug_bitrate_label` | Manifest bitrate label. |
| `chug_orientation` | Portrait / landscape label. |
| `chug_framerate_manifest` | Manifest framerate before ffprobe correction. |
| `chug_content_name` | Source content name from the CHUG CSV. |
| `chug_height_manifest`, `chug_width_manifest` | Manifest geometry. |

CHUG's source MOS is 0-100. The adapter maps trainer-facing `mos` onto
`[1, 5]` with:

```text
mos = 1 + 4 * mos_raw_0_100 / 100
```

The raw value remains available for future aggregation paths that use a
0-100 MOS axis directly.

## Local Baseline Training

Once `chug.jsonl` exists, a local baseline MOS-head train can be launched
without committing outputs:

```bash
python ai/scripts/train_konvid_mos_head.py \
  --konvid-1k .workingdir2/chug/chug.jsonl \
  --konvid-150k .workingdir2/chug/no-konvid-150k.jsonl \
  --out-onnx .workingdir2/chug/chug_mos_head.onnx \
  --out-card .workingdir2/chug/chug_mos_head_card.md \
  --out-manifest .workingdir2/chug/chug_mos_head.json
```

This is a baseline unlock, not a final HDR model. CHUG rows do not yet
carry extracted canonical-6, saliency, or shot features, so the next
quality step is a CHUG feature-extraction pass before production claims.
