# vmaf-tune saliency-aware ROI encoding

`vmaf-tune recommend-saliency` runs the fork-trained
`saliency_student_v1` ONNX model
([ADR-0286](../adr/0286-saliency-student-fork-trained-on-duts.md)) over
a source clip and produces a single encode whose bit allocation is
biased toward salient regions (faces, focal subjects, action). The
high-level surface and graceful-fallback contract are described in
[`vmaf-tune.md` §Saliency-aware encoding](vmaf-tune.md#saliency-aware-encoding-recommend---saliency-aware);
this page covers the codec-by-codec ROI surface — what each encoder
actually accepts on disk and how the saliency map is reduced onto it.

The codec-extension landed in the PR that bumps `qpfile_format` from
the x264-only stub to a four-codec dispatch table. See
[ADR-0293](../adr/0293-vmaf-tune-saliency-aware.md) §Status update
2026-05-08 for the decision history.

## Per-codec ROI surfaces

| Codec | `qpfile_format` | Encoder surface | Block size | Granularity |
| --- | --- | --- | --- | --- |
| `libx264` | `x264-mb` | `-x264-params qpfile=<path>` | 16×16 (MB) | per-MB |
| `libx265` | `x265-zones` | `-x265-params zones=<f0>,<f1>,q=<qp>` | 64×64 (CTU) | clip-level mean |
| `libsvtav1` | `svtav1-roi` | `-svtav1-params roi-map-file=<path>` | 64×64 (SB) | per-SB |
| `libvvenc` | `vvenc-qp-delta` | `-vvenc-params QpaperROIFile=<path>` | 128×128 (CTU) | per-CTU |
| `h264_nvenc`, `hevc_nvenc`, `av1_nvenc` | `none` | — | — | not exposed |
| `*_amf`, `*_qsv`, `*_videotoolbox`, `libaom-av1` | `none` | — | — | not exposed |

Codecs whose `qpfile_format` is `none` do not expose a portable ROI
surface through FFmpeg. `vmaf-tune recommend-saliency` detects that
case at dispatch time and runs a plain encode while logging a single
warning:

```text
saliency-aware: h264_nvenc does not expose a portable ROI surface; running plain encode
```

This matches the
[`vmaf-roi`](vmaf-roi.md) C sidecar's posture — the saliency bias is
opportunistic, the encode always succeeds.

## Format details

### libx264 — per-MB ASCII qpfile

x264 accepts a per-MB QP-offset table via the `--qpfile` extension
honoured since r2390. Format pinned by `saliency.write_x264_qpfile`:

```text
0 I 0
<row 0 offsets, space-separated>
<row 1 offsets, space-separated>
…
1 P 0
<row 0 offsets>
…
```

One header line per encode-frame followed by one row of MB offsets per
MB row. The saliency map is reduced onto a 16×16 grid by
[`reduce_qp_map_to_blocks`](../../tools/vmaf-tune/src/vmaftune/saliency.py),
which row/column-trims fewer-than-16-pixel edges to keep the grid
integer-aligned. References: [x264 wiki — qpfile][x264-qpfile].

### libx265 — clip-level mean zone

x265 does not expose a per-CTU sidecar through ffmpeg's libx265
wrapper; what *is* portable is the
[`zones=`][x265-zones] syntax that x265 documents:

```text
zones=<startFrame>,<endFrame>,q=<qp>/…
```

Zones are *temporal* slices, not spatial — each zone overrides the QP
for a frame range. To carry the saliency-driven *spatial* signal
through this surface, `saliency.write_x265_zones`:

1. Reduces the saliency mask onto the 64×64 CTU grid.
2. Computes the *clip-level mean* QP offset and rounds to int.
3. Emits a single zone covering `[0, duration_frames)` with absolute
   QP equal to `baseline_qp + mean_offset`, clamped to `[0, 51]`.

This is a deliberate granularity loss compared with x264's per-MB
qpfile — the x265 zones surface is the FFmpeg-only path. Users who
need true per-CTU granularity should drive x265 via the C-side
[`vmaf-roi`](vmaf-roi.md) sidecar (ADR-0247), which emits the x265
`--qpfile`-style ROI form.

References: [x265 documentation — zones][x265-zones],
[`vmaf-roi.c::emit_x265`](../../libvmaf/tools/vmaf_roi.c).

### libsvtav1 — binary signed-int8 ROI map

SVT-AV1 reads a binary `--roi-map-file` sidecar: one signed `int8_t`
byte per superblock, row-major, no header. `saliency.write_svtav1_roi_map`
emits exactly this format keyed to the 64×64 SB grid; bit-for-bit
parity with the C-side `vmaf-roi` `emit_svtav1` helper is enforced by
[`tests/test_saliency_roi.py::test_write_svtav1_roi_map_binary_format`](../../tools/vmaf-tune/tests/test_saliency_roi.py).

For multi-frame clips the per-frame block is repeated `duration_frames`
times — matching the per-clip aggregate the saliency model emits.

References: [SVT-AV1 README — ROI maps][svtav1-roi],
[`vmaf-roi.c::emit_svtav1`](../../libvmaf/tools/vmaf_roi.c).

### libvvenc — per-CTU ASCII QP-delta

VVenC accepts a per-CTU QP-delta sidecar via the `QpaperROIFile`
config-key on `-vvenc-params`. ASCII format: one signed integer per
CTU (128×128), space-separated, one row per CTU row, terminated by
`\n`. Multi-frame maps repeat the per-frame block separated by a blank
line so the encoder can advance frame-by-frame. This matches the
example configs that ship under `cfg/qpaper_roi*.cfg` in the VVenC
source distribution.

VVenC also ships a coarser 4-tier saliency-tier mode that maps every
pixel to one of `{background, low, medium, high}` and applies a fixed
QP delta per tier. The fork ships only the per-CTU form — the 4-tier
form is too coarse to carry the saliency model's signal usefully.

References: [VVenC manual — QP-paper ROI][vvenc-roi].

## Block-size mismatches

The saliency mask is computed at the source resolution. Each emitter
reduces it onto the codec's block grid via the shared
`_saliency_to_block_offsets` helper, which:

1. Converts the per-pixel mask to per-pixel QP offsets through
   `saliency_to_qp_map` (linear blend, ±`qp_offset`).
2. Reduces by *block-mean* to the codec's block size.
3. Trims the right/bottom edge if the source dimensions are not a
   multiple of the block size — the dropped strip is at most
   `block_side − 1` pixels wide (e.g. ≤ 127 px for VVenC's 128×128
   CTU).

Sources smaller than one block on either axis raise `ValueError`
rather than emitting an empty sidecar. In practice this only triggers
for VVenC at sub-128 px sources, which is well below the smallest
sample clip the harness ingests.

## Codec adapter contract

The dispatcher in `saliency.saliency_aware_encode` consumes two fields
from the codec adapter (see
[`codec_adapters/__init__.py`](../../tools/vmaf-tune/src/vmaftune/codec_adapters/__init__.py)):

* `supports_qpfile: bool` — gate flag. Adapters with `False` are
  skipped without invoking the saliency model.
* `qpfile_format: str` — one of `"x264-mb"`, `"x265-zones"`,
  `"svtav1-roi"`, `"vvenc-qp-delta"`, `"none"`. Drives emitter and
  augment-helper selection.

Adding a new codec is one file under `codec_adapters/` plus a
registry entry; the saliency dispatcher itself never branches on
codec identity.

## Smoke test

```shell
pytest tools/vmaf-tune/tests/test_saliency_roi.py -v
```

The test suite mocks the ONNX session and the encode runner so it
runs without ffmpeg, x265, SVT-AV1, VVenC, or onnxruntime installed —
the on-disk format pins are the source of truth.

[x264-qpfile]: https://en.wikibooks.org/wiki/MeGUI/x264_Settings#--qpfile
[x265-zones]: https://x265.readthedocs.io/en/master/cli.html#cmdoption-zones
[svtav1-roi]: https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/Docs/Parameters.md
[vvenc-roi]: https://github.com/fraunhoferhhi/vvenc/blob/master/doc/software-manual.md
