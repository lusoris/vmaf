# Research-0085 — Vendor-neutral VVC GPU encode landscape (skeleton)

- **Status**: SKELETON — most factual claims need verification before
  this becomes load-bearing for any decision. See "Verification status"
  immediately below.
- **Workstream**: [ADR-0315](../adr/0315-vendor-neutral-vvc-encode-strategy.md),
  [ADR-0314](../adr/0314-vmaf-tune-vulkan-score-quick-win.md) (sibling,
  scoped separately)
- **Last updated**: 2026-05-05

## Verification status — read before citing

The first revision of this digest (subagent-authored, 2026-05-05) carried
`[verified]` tags on several sources that were **not actually verified
against the linked documentation**. Specifically: it asserted that NVENC
in NVIDIA Video Codec SDK 13.0 supports H.266/VVC encode on Ada-Lovelace
silicon; a direct WebFetch of NVIDIA's NVENC Application Note for SDK
13.0 confirms NVENC supports only **H.264, HEVC 8-bit, HEVC 10-bit,
AV1 8-bit, AV1 10-bit** — there is no H.266 encode support documented.

The post-hoc revision (this version) keeps the structure but downgrades
every unverified claim to `[UNVERIFIED]` and strips the cost-matrix /
findings rows that depended on the fabricated NVENC-encode premise.

What is actually verified at write time, with linked source:

| Claim | Source | Verified by |
| --- | --- | --- |
| NVENC supports H.264 / HEVC / AV1 only; **no H.266 encode** | <https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html> | WebFetch 2026-05-05 |
| Intel was first to ship hardware H.266/VVC **decode** (Lunar Lake / Xe2) | guru3D coverage of Intel announcement | WebSearch 2026-05-05 |
| VVenC v1.14 released January 2026; 20–2400× VTM speedup depending on preset | WebSearch result citing release | WebSearch 2026-05-05 |
| Fork's CPU adapter exists at `tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py`; shells out to CPU `vvenc` binary; exposes NN-VC toggles | Fork tree, `vvenc.py` | Direct read |

Everything else below is `[UNVERIFIED]` or an open question.

## Question

User (lawrence, 2026-05-05): *"Gotta wonder if you could make CUDA VVC
without nvidia."*

Translated and de-colloquialised: can the fork ship a GPU-accelerated
VVC (H.266) encode path that does **not** require NVIDIA hardware?
Today the fork's only VVC adapter shells out to Fraunhofer HHI's
**CPU** `vvenc` reference encoder. The question is whether a
vendor-neutral GPU encode story exists today, near-term, or only on a
multi-year horizon — and what the cost/value ladder looks like.

This skeleton scopes the survey questions; concrete verified findings
will replace the `[UNVERIFIED]` tags as sources are checked.

## Sources

`[verified]` = directly confirmed against the linked source by a
WebFetch / direct file read at write time. `[UNVERIFIED]` = claim
present in the original draft but not re-verified; treat as an
open question, not a fact.

- `[verified]` Fork-internal:
  - `tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py` — the
    existing CPU adapter, with NN-VC toggle.
  - `tools/vmaf-tune/AGENTS.md` — describes the NN-VC integration
    surface and Phase A scope boundary.
  - [ADR-0290](../adr/0290-vmaf-tune-nvenc-adapters.md) — current
    NVENC adapter ladder (h264 / hevc / av1 only at write time).
  - [ADR-0127](../adr/0127-vulkan-compute-backend.md) — fork's
    Vulkan compute backend (scoring, not encoding).
  - [ADR-0033](../adr/0033-hip-applicability.md) — prior HIP-port
    feasibility note (focused on VMAF features, not codecs).
- `[verified]` NVIDIA Video Codec SDK 13.0 NVENC Application Note
  (<https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html>) —
  NVENC supports H.264, HEVC 8-bit, HEVC 10-bit, AV1 8-bit, AV1 10-bit.
  **No H.266/VVC encode.** Newest codec is AV1, on Ada and Blackwell.
- `[UNVERIFIED]` Fraunhofer HHI VVenC repository GPU-port roadmap.
  Open question: does an upstream HIP / SYCL / CUDA port exist or is
  one planned? Need direct repo / issue-tracker check.
- `[UNVERIFIED]` AMD AMF SDK headers — claim was that
  `VideoEncoderVVC.h` is absent in shipping 1.4.x branches and that
  RDNA 4 / Ryzen-AI launch briefings did not name VVC encode. Needs
  direct check of the published AMF SDK + AMD slide decks.
- `[UNVERIFIED]` AMD VCN media engine VVC decode support — claim was
  RDNA 4 ASIC has decode-only ASIC support; vendor-doc check needed.
- `[UNVERIFIED]` Intel Battlemage / Lunar Lake VVC decode — partially
  verified (Lunar Lake was first chipmaker with VVC hardware decode,
  per WebSearch 2026-05-05). The specific claim that VVC encode is
  not in oneVPL 2.10's `mfx_status.h` capability flags needs direct
  header check.
- `[UNVERIFIED]` Khronos Vulkan Video extensions:
  `VK_KHR_video_encode_h264` / `_h265` / `_av1` ratification dates
  and `VK_KHR_video_encode_h266` provisional status. Needs Khronos
  registry / blog check.
- `[UNVERIFIED]` Mesa 24.x / 25.x release notes — claimed RADV
  ships AV1 encode on RDNA 3 / RDNA 4; ANV ships AV1 encode on
  Battlemage; H.266 encode absent on every Mesa driver path.
  Needs direct Mesa release-notes check.
- `[UNVERIFIED]` SVT-AV1 / x265 GPU-port status — claim was no
  upstreamed HIP / CUDA / SYCL fork exists. Needs repo issue-tracker
  check.
- `[UNVERIFIED]` ZLUDA project status — claim was AMD-funded phase
  paused 2024, community reactivation 2025, "experimental" posture,
  partial CUDA Driver API and cuBLAS / cuFFT coverage. Needs repo
  check.
- `[UNVERIFIED]` ONNX Runtime EP matrix — claim was CUDA, ROCm,
  DirectML, OpenVINO, CoreML, WebGPU, QNN supported. Likely accurate
  but needs ORT docs check.

## Findings (under construction)

### F0. NO vendor ships hardware VVC encode silicon as of 2026-05-05

NVIDIA NVENC SDK 13.0 documentation (verified) supports only H.264 /
HEVC / AV1 — **no H.266 encode**. AMD AMF and Intel QSV claims are
`[UNVERIFIED]` but indicative search suggests the same: no shipping
silicon ships VVC encode regardless of vendor. The "vendor-neutral
GPU VVC encode" question therefore has an unusually direct answer
today: there's no GPU VVC encode path on **any** vendor.

### F1. Vulkan-side scoring is shipped today; Vulkan-side encoding is not

The fork has a Vulkan compute backend ([ADR-0127](../adr/0127-vulkan-compute-backend.md))
for VMAF scoring. This is the basis for the sibling
[ADR-0314](../adr/0314-vmaf-tune-vulkan-score-quick-win.md) (separate
PR) which wires `vmaf-tune --score-backend=vulkan`. That gives
non-NVIDIA users vendor-neutral GPU **scoring** of CPU-encoded VVC
bitstreams — partial answer to the user's question.

There is no Vulkan-side **encode** path today (none in libvmaf, and
`[UNVERIFIED]` whether Mesa / proprietary drivers ship any
`VK_KHR_video_encode_h266` implementation).

### F2. NN-VC is the de-facto vendor-neutral GPU contribution today

VVC's neural-video-coding tools (NN-intra prediction, NN-loop-filter,
NN-super-resolution) run as ONNX models inside the encoder. The
fork's `vvenc.py` adapter exposes them via `--nnvc-intra`,
`--nnvc-loop-filter`, `--nnvc-sr` (verified by reading the file).
ONNX Runtime EP coverage for non-NVIDIA hardware is `[UNVERIFIED]`
in detail but the fork already integrates ORT for tiny-AI on multiple
EPs, so vendor-neutrality of the *neural portion* is plausible.

The catch: NN-VC accelerates the **quality** of the encode, not the
**throughput**. The CPU encoder loops still run on CPU; the GPU is
only burning the NN-intra inference. Quality-lift magnitude
(`[UNVERIFIED]`) — the original draft claimed 1–3% bitrate at
iso-VMAF; needs benchmark.

### F3. HIP / SYCL port of vvenc hot kernels — feasibility skeleton

The original draft proposed a HIP port of vvenc's motion-estimation,
transform, and loop-filter kernels as Tier 2. The skeleton is
plausible but every concrete number — % of CPU time per kernel,
3–5× wall-clock speedup projection, eng-month estimate — is
`[UNVERIFIED]`. A real Tier-2 ADR would need:

1. A profile run of CPU vvenc on a representative corpus to confirm
   the per-kernel CPU-time distribution.
2. A check of upstream vvenc's GPU-port roadmap (don't reinvent
   what upstream may ship).
3. Hardware availability for ongoing CI per ADR-0214's GPU-parity
   rule.

### F4. ZLUDA is technically interesting but not actionable

ZLUDA could in principle run a (hypothetical) closed-source CUDA
VVC encoder on AMD / Intel hardware. In practice no such CUDA VVC
encoder exists in the open-source ecosystem, and ZLUDA's coverage
is `[UNVERIFIED]` for codec workloads. Not a near-term path.

## Cost / risk / value matrix (skeleton)

The original draft's matrix included a row "A. NVENC h266 adapter
(0.25 eng-months)" that depended on the fabricated NVENC-encode
premise; that row is **removed**.

The remaining rows are kept, with effort / speedup numbers downgraded
to `[UNVERIFIED]`:

| Path | Effort `[UNVERIFIED]` | License risk | User value | Verdict |
| --- | --- | --- | --- | --- |
| **Vulkan-scoring quick-win** ([ADR-0314](../adr/0314-vmaf-tune-vulkan-score-quick-win.md), sibling) | Small | None | Non-NVIDIA users get GPU scoring for VVC encodes (encode stays on CPU) | **Tier 1**: ship now via the sibling PR. |
| **NN-VC documentation + corpus integration** | Small | Mixed (NN-VC weights are LGPL-derived; tooling is Apache-2.0) — `[UNVERIFIED]` | Any-GPU users; quality lift `[UNVERIFIED]` | **Tier 1**: bundle with Vulkan-scoring rollout. |
| **HIP port of vvenc hot kernels** | Medium-large; numbers `[UNVERIFIED]` | Apache-2.0 OK; Fraunhofer patent licence still applies | RDNA 3 / 4 / CDNA users; speedup `[UNVERIFIED]` | **Tier 2**: gated on Tier 1 success and corpus profile. |
| **SYCL port of same kernels** | Incremental over HIP | Apache-2.0 OK | Adds Intel PVC / Xe2 + cross-vendor via Codeplay plugins | **Tier 2.5**: deferred. |
| **Vulkan Video VVC encode adapter** | Driver-side does the work; libvmaf-side small | None | All vendors once silicon + drivers ship | **Tier 3**: revisit quarterly; gated on `VK_KHR_video_encode_h266` ratification + at least one driver shipping. |
| **ZLUDA-hosted CUDA-VVC** | n/a | Risky | Zero today (no open-source CUDA VVC encoder exists) | **Rejected**. |
| **Wait for AMD/Intel hardware VVC encode** | 0 (passive) | None | None until silicon ships | Reflected in Tier 3's "revisit". |

## Recommendations (priority order)

### Tier 1 — ship today (small total effort)

1. **Wire vmaf-tune's existing Vulkan scoring** so non-NVIDIA users
   benefit from GPU-accelerated *scoring* of CPU-encoded VVC bitstreams.
   Scoped separately as **ADR-0314** (sibling PR — this digest does
   not implement it).
2. **Document NN-VC as the vendor-neutral GPU contribution today**
   in the user-facing docs. Be explicit that:
   - Hardware VVC encode does **not** exist on any GPU vendor
     (verified for NVIDIA SDK 13.0; `[UNVERIFIED]` but indicative
     for AMD / Intel).
   - VVenC is the open-source CPU encoder; encoder loops stay on
     CPU but neural tools accelerate on GPU via ONNXRuntime EPs.
   - Vulkan-side scoring closes the GPU loop on the consumption side.

### Tier 2 — backlog, demand-pulled

**HIP port of vvenc's motion-estimation + transform + loop-filter
kernels.** Triggered only when **all three** are true:

- A user has reported CPU-vvenc throughput as a binding constraint
  on a real corpus (not a synthetic benchmark).
- Tier-1 Vulkan-scoring + NN-VC docs have landed and at least one
  downstream consumer is using NN-VC tools through `vmaf-tune` in
  production.
- An RDNA 3 / RDNA 4 or Intel PVC machine is available to the fork
  for ongoing CI (without it the HIP port has no automated
  regression gate and ADR-0214's GPU-parity rule is unenforceable).

If any of the three is false, Tier 2 stays in the backlog. This
matches the fork's "demand-pulled fork-local effort" pattern from
ADR-0009.

### Tier 3 — speculative, revisit quarterly

**Vulkan Video VVC encode adapter.** Triggered by:

- `VK_KHR_video_encode_h266` ratification (`[UNVERIFIED]` whether
  even provisional today; needs Khronos check).
- At least one driver shipping the extension (Mesa RADV, NVIDIA
  proprietary, or ANV — any one suffices to start).
- A libvmaf-side encode adapter design proposal (currently the
  fork's Vulkan code is score-only).

## Reproducer / verification commands

```bash
# 1. Confirm what NVENC actually supports (verified 2026-05-05):
nvidia-smi --query-gpu=name,driver_version --format=csv
ffmpeg -hide_banner -encoders 2>&1 | grep -iE 'nvenc|h266|vvc'
# expected: h264_nvenc / hevc_nvenc / av1_nvenc; no h266_nvenc.

# 2. Confirm AMF / QSV do NOT expose VVC encode (UNVERIFIED — run to confirm):
ffmpeg -hide_banner -encoders 2>&1 | grep -iE 'h266|vvc'
# expected per indicative search: only libvvenc enumerates;
# no h266_amf / h266_qsv / h266_nvenc.

# 3. Vulkan-Video extension presence (UNVERIFIED — run to confirm):
vulkaninfo | grep -iE 'video_encode_(h264|h265|av1|h266)'
# expected per indicative search: h264 / h265 / av1 may be present
# on RDNA 4 / Battlemage; h266 absent.

# 4. Fork's CPU adapter still works for vendor-neutral encode:
python -m vmaftune encode --codec libvvenc --preset slow \
    --input testdata/yuv/akiyo_qcif.yuv --qp 32 --output /tmp/akiyo.266

# 5. NN-VC tools on whatever GPU is present (ONNXRuntime auto-EP):
python -m vmaftune encode --codec libvvenc --preset medium \
    --nnvc-intra --input testdata/yuv/akiyo_qcif.yuv --qp 32 \
    --output /tmp/akiyo_nnvc.266

# 6. Docs build clean:
mkdocs build --strict 2>&1 | grep -E "(WARNING|ERROR)" || echo "docs build clean"
```

## Open questions (substantive — block promotion to "findings")

1. **AMD AMF VVC encode** — does any shipping AMF SDK version expose
   a VVC encoder enum or capability? Direct AMF header check needed.
2. **Intel oneVPL VVC encode** — does any shipping oneVPL version
   expose VVC encode caps in `mfx_status.h`? Direct header check
   needed.
3. **Intel Lunar Lake / Battlemage VVC decode** — partially verified
   (Intel was first chipmaker with VVC hardware decode); confirm the
   exact silicon generations and oneVPL / VAAPI plumbing.
4. **Khronos `VK_KHR_video_encode_h266`** — provisional? proposal?
   ratified? Direct registry check needed.
5. **Mesa AV1 encode** — which RADV / ANV release shipped which
   silicon-generation support? Direct release-notes check needed.
6. **Mesa VVC decode** — which release / driver landed VVC decode
   support? Direct release-notes check needed.
7. **Fraunhofer HHI VVenC GPU-port roadmap** — does an upstream HIP /
   SYCL / CUDA port exist or is one planned? Repo + issues check.
8. **NN-VC quality-lift magnitude** — what's the actual bitrate-at-
   iso-VMAF lift on representative corpora? Real benchmark needed.
9. **HIP-portability of vvenc kernels** — what's the actual CPU-time
   distribution per kernel on a real corpus, and what fraction is
   GPU-portable? Real profile needed.
10. **ZLUDA codec-workload coverage** — does ZLUDA's CUDA Driver API +
    cuBLAS / cuFFT subset cover what a hypothetical CUDA VVC encoder
    would need? Repo + issues check.

## Related

- [ADR-0315](../adr/0315-vendor-neutral-vvc-encode-strategy.md) —
  the decision this skeleton feeds. ADR-0315 is also being downgraded
  to a skeleton pending verification of the underlying claims.
- [ADR-0314](../adr/0314-vmaf-tune-vulkan-score-quick-win.md) —
  Tier-1 sibling, scoped separately, wires Vulkan scoring through
  `vmaf-tune`. This is the only concrete deliverable in the skeleton's
  Tier 1 today.
- [ADR-0290](../adr/0290-vmaf-tune-nvenc-adapters.md) — NVENC
  adapter ladder (h264 / hevc / av1 only). No h266 NVENC adapter
  exists or is planned (NVENC silicon does not ship H.266 encode).
- [ADR-0127](../adr/0127-vulkan-compute-backend.md) — Vulkan compute
  backend for scoring.
- [`tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py`](../../tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py)
  — the existing CPU adapter, with NN-VC toggle.
- VVenC upstream: <https://github.com/fraunhoferhhi/vvenc>
- NVIDIA Video Codec SDK 13.0 NVENC Application Note (verified):
  <https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html>
