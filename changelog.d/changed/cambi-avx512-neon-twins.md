- **Cambi `calculate_c_values_row` AVX-512 + NEON twins
  ([ADR-0328 status update 2026-05-08](../docs/adr/0328-cambi-cluster-port-skip-shared-header-rename.md#status-update-2026-05-08-simd-twins-completed)).**
  Closes the perf follow-ups left open by [PR #463](https://github.com/lusoris/vmaf/pull/463):
  - `calculate_c_values_row_avx512` (16-lane gather + mmask predicate)
    in `libvmaf/src/feature/x86/cambi_avx512.c`.
  - `calculate_c_values_row_neon` (4-lane vectorised arithmetic +
    scalar histogram gathers) in `libvmaf/src/feature/arm64/cambi_neon.c`.

  Both variants are bit-identical to the scalar reference at full
  IEEE-754 precision (gated by the new `test_cambi_simd` parity test
  at four widths × every available ISA on the host). Netflix golden
  scores are byte-identical; CLI / public-API surface is unchanged.

  AVX-512 throughput on AMD Zen 5 is ~1.15× over AVX-2 on a synthetic
  width=1920 row microbench (gather-bound; Intel parts expected closer
  to 1.5–2× per the cambi.md SIMD section). NEON throughput is
  host-dependent; the test asserts only correctness.

  See [`docs/metrics/cambi.md` § CPU SIMD paths](../docs/metrics/cambi.md#cpu-simd-paths)
  for the per-variant table and the bit-exactness gate description.
