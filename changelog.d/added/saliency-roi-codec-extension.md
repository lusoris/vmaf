- **vmaf-tune saliency-aware ROI: x265 / SVT-AV1 / libvvenc support
  ([ADR-0293](../docs/adr/0293-vmaf-tune-saliency-aware.md) status
  update 2026-05-08).**
  Extends `tools/vmaf-tune/src/vmaftune/saliency.py` from x264-only to
  drive HEVC / AV1 / VVC encodes with the same saliency QP-offset
  map. Adds `write_x265_zones()` (clip-level mean offset emitted as a
  `zones=` token), `write_svtav1_roi_map()` (binary signed-int8 grid
  matching SVT-AV1's `--roi-map-file` and the C-side `vmaf-roi`
  `emit_svtav1` helper byte-for-byte), and `write_vvenc_qp_delta()`
  (ASCII per-CTU QP-delta in VVenC's `QpaperROIFile` format).
  `saliency_aware_encode()` now dispatches via a new `qpfile_format`
  field on the codec-adapter Protocol (`"x264-mb"`, `"x265-zones"`,
  `"svtav1-roi"`, `"vvenc-qp-delta"`, or `"none"`); HW codecs
  (NVENC / AMF / QSV / VideoToolbox) and `libaom-av1` advertise
  `"none"` and degrade to plain encodes with a single warning line.
  User docs:
  [`docs/usage/vmaf-tune-saliency.md`](../docs/usage/vmaf-tune-saliency.md).
  Tests under `tools/vmaf-tune/tests/test_saliency_roi.py` pin every
  emitter's on-disk format and assert the dispatcher injects the
  right `-{x264,x265,svtav1,vvenc}-params` for each codec; HW codecs
  are tested for the warn-and-fallback path. No upstream
  Netflix/vmaf surface is touched.
