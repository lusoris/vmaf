# Research-0086 — Contributor-pack web-data expansion (2026-05-08)

- **Status**: Active — feeds the TransNet-V2 shot-metadata + HDR-VMAF
  model-slot dual delivery on branch
  `feat/transnet-shot-meta-and-hdr-vmaf-model`.
- **Workstream**:
  [ADR-0223](../adr/0223-transnet-v2-shot-detector.md) (TransNet-V2
  integration),
  [ADR-0300](../adr/0300-vmaf-tune-hdr-aware.md) (HDR-aware corpus,
  model-port follow-up),
  [ADR-0237](../adr/0237-quality-aware-encode-automation.md)
  (vmaf-tune Phase A umbrella).
- **Last updated**: 2026-05-08.

## Summary

Two "free signal" expansions to the `vmaf-tune` corpus surface
identified during the 2026-05-08 contributor-pack audit:

1. **TransNet-V2 shot-metadata trio.** TransNet-V2 (Apache-2.0
   weights, ADR-0223) is already wired into the harness as the shot
   detector behind the per-shot CRF tuner. The shot count and per-shot
   duration distribution per source is currently *computed* (in the
   `Shot` ranges returned by `detect_shots()`) but *discarded* at the
   corpus-row boundary. Surfacing it as three columns
   (`shot_count`, `shot_avg_duration_sec`, `shot_duration_std_sec`)
   gives Phase B / C predictors a content-class proxy at zero
   additional wall-time cost. Animation tends toward short shots with
   low variance, live-action drama toward longer shots with higher
   variance — the std column is the discriminator.

2. **HDR VMAF model-port slot.** ADR-0300 § Consequences flagged the
   HDR-trained VMAF model port as a follow-up. Verification against
   `upstream/master` (`git ls-tree upstream/master -- model/`,
   2026-05-08) and the rendered GitHub UI confirms that the canonical
   Netflix filename `vmaf_hdr_v0.6.1.json` is **not** shipped in the
   upstream public `model/` tree. Netflix maintains the artefact in
   a separate research bundle outside the public repository.
   `select_hdr_vmaf_model()` is therefore upgraded with transfer-aware
   routing and a "drop the file in `model/` and the harness picks it
   up" deployment story — a verbatim copy of the JSON is not landed
   in this PR pending the license review tracked by the ADR-0300
   follow-up backlog row.

## Decision matrix — TransNet metadata column choice

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| Surface raw `shot_boundaries` array per row | Maximum downstream flexibility | Bloats every row by O(N) shots × 8 bytes; redundant across (preset, crf) cells of the same source | Aggregates carry the discriminative signal at constant size |
| Just `shot_count` (single int) | Smallest schema delta | Discards the duration distribution that distinguishes news (many short shots, low std) from animation (many short shots, low std) from live-action drama (few long shots, high std) | Trio is the smallest set that separates the three content classes |
| `shot_count` + percentile summary (p10 / p50 / p90) | Captures bimodal shot distributions | 4 columns vs 3; population std covers the same proxy at lower complexity | Kept the simpler trio; percentile summary is a future extension |

## Decision matrix — HDR model port shape

| Option | Pros | Cons | Why not chosen |
| --- | --- | --- | --- |
| Verbatim binary port of `vmaf_hdr_v0.6.1.json` from a Netflix research mirror | Closes ADR-0300's open consequence | Netflix-license review not done; the artefact lives outside the public `model/` tree and "ports" without a clear redistribution clause | Rejected without license clarification — see "no guessing" rule (memory `feedback_no_guessing`) |
| Train a fork-local HDR-tuned VMAF model from scratch | Clean licensing | Out of scope of this PR; needs HDR DMOS dataset + multi-week training | Future Phase F backlog item |
| Slot-only: register the canonical name + transfer routing, no JSON shipped | Operator drops a licensed copy and the harness picks it up; honest about the gap | HDR scoring still falls back to the SDR model in the fork's default state | **Chosen** — preserves the "scores trend low for HDR, user is warned" semantic from ADR-0300 |

## References

- TransNet-V2 weights, Apache-2.0:
  <https://github.com/soCzech/TransNetV2>.
- ADR-0223 ports the weights into the fork (`libvmaf/src/dnn/`) and
  ADR-0222 wires the `vmaf-perShot` CLI binary on top.
- Upstream `Netflix/vmaf` master `model/` tree, verified empty of
  HDR-tagged JSONs 2026-05-08 (WebFetch +
  `git ls-tree upstream/master -- model/`).
- ADR-0300's original Context section: "Netflix maintains an
  HDR-trained VMAF model (`vmaf_hdr_v0.6.1.json`) in a separate
  research artifact; it has not been ported into this fork."
