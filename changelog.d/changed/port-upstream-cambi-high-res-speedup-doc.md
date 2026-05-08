- **Port upstream `721569bc` — `cambi_high_res_speedup` parameter doc +
  motion2 score refresh.** Cherry-picked Netflix/vmaf commit
  [`721569bc`](https://github.com/Netflix/vmaf/commit/721569bc1b18847a396c86575f4698e372454c04)
  ("resource/doc: add cambi_high_res_speedup parameter and update
  motion2 score") and remapped it onto the fork's docs tree:
  - `docs/metrics/cambi.md`: documented the existing
    `cambi_high_res_speedup` extractor option (downsamples post
    spatial mask for resolutions ≥ 1080p; possible min resolutions
    `[1080, 1440, 3840, 0]`; default `0`).
  - `docs/metrics/confidence-interval.md` and `docs/usage/python.md`:
    refreshed the `VMAF_feature_motion2_score` sample value from
    `3.8953518541666665` to `3.8943597291666667` to track upstream's
    current canonical number for the `src01_hrc00 ↔ src01_hrc01`
    576x324 example. Documentation prose only; no engine code, no
    test golden, no CLI flag changes.
