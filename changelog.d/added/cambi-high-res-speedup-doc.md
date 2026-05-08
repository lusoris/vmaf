- **CAMBI: `cambi_high_res_speedup` parameter documented**
  ([`docs/metrics/cambi.md`](../docs/metrics/cambi.md)). Ported from
  upstream Netflix/vmaf `721569bc` (christosb, 2026-05-06). The
  option speeds up CAMBI on >= 1080p inputs by downsampling after
  the spatial mask, trading a small loss of accuracy for throughput.
  Allowed minimum-resolution thresholds are `[1080, 1440, 3840, 0]`;
  default `0` disables the speed-up. The same upstream commit also
  refreshes the `VMAF_feature_motion2_score` example value in
  [`confidence-interval.md`](../docs/metrics/confidence-interval.md)
  and [`python.md`](../docs/usage/python.md) to track the
  upstream `motion_v2` mirroring fix
  ([`856d3835`](https://github.com/Netflix/vmaf/commit/856d3835)).
