# ADR-0315: Vendor-neutral VVC encode strategy — tiered Tier-1-now / Tier-2-backlog / Tier-3-revisit

- **Status**: Proposed
- **Date**: 2026-05-05
- **Deciders**: Lusoris, lawrence
- **Tags**: codecs, vvc, h266, gpu, hip, sycl, vulkan-video, vmaf-tune, nn-vc, fork-local

## Context

The fork ships VVC (H.266) encode today through a CPU-only adapter
([`tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py`](../../tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py))
that wraps Fraunhofer HHI's reference encoder. NVIDIA's Video Codec
SDK 13.0 added hardware H.266 encode on Ada-Lovelace silicon and
newer; AMD AMF and Intel QSV expose VVC **decode** but not encode.
The user (lawrence, 2026-05-05) asked whether a GPU-accelerated
VVC encode path that does not require NVIDIA hardware is feasible.

Research-0085 surveyed the landscape and identified four practical
candidate paths: (a) ship NVENC h266 anyway (closed silicon, helps
RTX-40 users only), (b) HIP-port VVenC's hot kernels (open-source,
3–6 eng-months, vendor-coverage AMD), (c) wait for Vulkan Video
`VK_KHR_video_encode_h266` ratification + driver landing
(~24+ months out), or (d) document NN-VC + the existing Vulkan
scoring backend as the de-facto vendor-neutral story today.

Forces:

- **Power of 10 / scope discipline** — the fork has limited engineering
  bandwidth; a 6-month HIP-port commitment crowds out tiny-AI and
  Vulkan-scoring backlog items.
- **Demand-pull principle** — fork-local efforts only ship when a
  user reports the gap as binding (precedent:
  [ADR-0009](0009-batch-a-upstream-port-strategy.md)).
- **Hardware availability** — the fork's CI matrix does not include
  RDNA 3/4 or Intel PVC nodes; a HIP port without a CI gate cannot
  satisfy [ADR-0214](0214-cross-backend-numerical-parity.md)'s
  GPU-parity rule.
- **Vendor-neutrality value** — non-NVIDIA users today have *zero*
  GPU acceleration on the VVC encode side; the gap is real, not
  theoretical.

## Decision

We will adopt a three-tier strategy. **Tier 1 ships today**:
document NN-VC as the fork's vendor-neutral H.266 GPU story (it
already runs on any ONNXRuntime EP) and wire the existing Vulkan
backend to `vmaf-tune` for GPU-accelerated scoring of CPU-encoded
VVC bitstreams (sibling [ADR-0314](0314-vmaf-tune-vulkan-score-quick-win.md),
scoped separately). **Tier 2 stays in the backlog**: a HIP port of
VVenC's motion-estimation, transform, and loop-filter kernels, gated
on three triggers (user-reported throughput pain on a real corpus,
Tier-1 NN-VC docs adopted in production, RDNA 3/4 or PVC CI machine
available). **Tier 3 is revisited quarterly**: a
`VK_KHR_video_encode_h266`-based libvmaf-side encode adapter, gated on
Khronos ratification and at least one shipping driver implementation.

Research-0085 is the authoritative source survey; this ADR is the
decision artifact that points back to it.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| **A. Ship CUDA-VVC NVENC adapter only** (mirror [ADR-0290](0290-vmaf-tune-nvenc-adapters.md) for h266) | Trivial effort (~0.25 eng-month); reuses existing NVENC adapter ladder; helps real RTX-40 users | Does **not** answer the user's question — non-NVIDIA users get nothing; reinforces the NVIDIA dependency the user flagged | Will land as a separate small PR; rejected as the *answer* to the vendor-neutrality question. |
| **B. Wait for `VK_KHR_video_encode_h266` ratification** | Zero effort; eventually delivers vendor-neutral hardware encode on every Vulkan 1.4+ driver; reuses the fork's existing Vulkan loader / queue / DMABUF plumbing | Spec is unratified at write time; AV1 precedent suggests 24+ month spec-to-driver lag; no Mesa MR exists yet; abandons users who need a non-NVIDIA path **today** | Rejected as a standalone strategy. Reframed as Tier 3 (revisit quarterly) inside the chosen tiered approach. |
| **C. HIP port of VVenC hot kernels (immediate)** | Genuinely vendor-neutral on AMD silicon; precedent for fork-side GPU ports exists ([ADR-0033](0033-hip-applicability.md)) | 3–6 eng-months; rebase burden against vvenc's quarterly cadence; no CI hardware available; no demand signal yet beyond the question itself | Rejected as immediate Tier 1; queued as Tier 2 with explicit demand-pull triggers. |
| **D. Tiered approach (chosen)** | Ships *something* vendor-neutral today (NN-VC docs + Vulkan scoring); preserves option value on the HIP port without burning engineering hours speculatively; revisits Vulkan Video on a calendar | Communicates "we don't have a hardware encoder for AMD yet"; relies on NN-VC + CPU encode for the near-term throughput story | **Chosen.** Matches the fork's demand-pull pattern; gives users an honest answer + a non-zero GPU contribution (NN-VC + scoring) immediately; preserves room for Tier 2 / Tier 3 to arrive when the triggers fire. |
| **E. ZLUDA-hosted hypothetical CUDA-VVC** | In theory runs CUDA codecs on AMD/Intel | No open-source CUDA VVC encoder exists; ZLUDA is experimental and uncovered for codec workloads; reviewers would reasonably reject the production posture | Rejected as not actionable. |

## Consequences

- **Positive**:
  - User question answered honestly today: NN-VC + Vulkan scoring
    deliver the vendor-neutral GPU contribution that exists in 2026.
  - Engineering bandwidth preserved for tiny-AI / Vulkan-scoring
    backlog rather than committed to a 6-month HIP port without a
    demand signal.
  - Tier 2 / Tier 3 stay tracked in the backlog with explicit
    triggers, preventing them from being silently forgotten.
- **Negative**:
  - Non-NVIDIA users do not get *hardware* VVC encode acceleration
    in the near term; the only real lever for them is NN-VC quality
    and CPU SIMD throughput.
  - The fork's "vendor-neutral codec story" is partial — strong on
    AV1 and HEVC (NVENC + AMF + QSV adapters all exist), weaker on
    H.266 (NVENC only at the hardware tier).
- **Neutral / follow-ups**:
  - A separate small PR may land an `h266_nvenc` adapter for
    RTX-40-class users (analogue of [ADR-0290](0290-vmaf-tune-nvenc-adapters.md));
    it is **not** part of this strategy ADR.
  - Tier 2 trigger conditions live in Research-0085 §"Tier-1 → Tier-2
    transition gate"; reviewers monitor user reports for the binding
    signal.
  - Tier 3 quarterly revisit owner is whoever runs the next
    `/sync-upstream` cycle after each Khronos Working Group public
    cadence event.
  - The fork's `docs/codecs/vvc.md` (forthcoming) will document the
    user-facing matrix: which codec on which hardware on which day.

## References

- Research-0085: [Vendor-neutral VVC GPU encode landscape](../research/0085-vendor-neutral-vvc-encode-landscape.md)
  — source survey, citations, cost/risk/value matrix.
- [ADR-0290](0290-vmaf-tune-nvenc-adapters.md) — NVENC adapter ladder
  pattern that an h266_nvenc adapter would mirror.
- [ADR-0314](0314-vmaf-tune-vulkan-score-quick-win.md) — Tier-1
  sibling, scoped separately, wires Vulkan scoring through
  `vmaf-tune`.
- [ADR-0033](0033-hip-applicability.md) — prior HIP applicability
  survey for libvmaf features.
- [ADR-0127](0127-vulkan-compute-backend.md) — Vulkan compute backend
  (scoring).
- [ADR-0009](0009-batch-a-upstream-port-strategy.md) — fork's
  demand-pull pattern for fork-local engineering effort.
- [ADR-0214](0214-cross-backend-numerical-parity.md) — GPU-parity
  rule that Tier 2 has to satisfy on landing.
- Source: `req` — paraphrased user request: "investigate whether the
  fork can offer GPU-accelerated VVC encode without NVIDIA hardware,
  and recommend a tiered path that ships something today" (lawrence,
  2026-05-05).
