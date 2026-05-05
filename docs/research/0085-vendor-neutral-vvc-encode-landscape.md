# Research-0085 — Vendor-neutral VVC GPU encode landscape

- **Status**: Active
- **Workstream**: [ADR-0315](../adr/0315-vendor-neutral-vvc-encode-strategy.md), [ADR-0314](../adr/0314-vmaf-tune-vulkan-score-quick-win.md) (sibling, scoped separately)
- **Last updated**: 2026-05-05

## Question

User (lawrence, 2026-05-05): *"Gotta wonder if you could make CUDA VVC
without nvidia."*

Translated and de-colloquialised: can the fork ship a GPU-accelerated
VVC (H.266) encode path that does **not** require NVIDIA hardware?
Today the fork's only VVC adapter — `tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py`
— shells out to Fraunhofer HHI's CPU `vvenc` reference encoder. NVENC
gained a hardware H.266 encoder on Ada-Lovelace silicon (RTX 40-series,
2026 Jetson refresh); AMD AMF and Intel QSV currently expose only VVC
**decode**. The question is whether a vendor-neutral GPU encode story
exists today, near-term, or only on a multi-year horizon — and if so,
what the cost/value ladder looks like.

This digest scopes the survey, scores the candidates, and feeds
ADR-0315's three-tier rollout decision.

## Sources

Primary references consulted (each citation flagged
**[verified]** = directly confirmed against the linked source,
**[indicative]** = consensus from multiple secondary sources but not
re-verified at write time, or **[speculative]** = projection / open
question):

- **[verified]** Fraunhofer HHI VVenC repository:
  <https://github.com/fraunhoferhhi/vvenc> — releases up to v1.13.x
  (May 2026) ship CPU-only AVX2 / AVX-512 / NEON paths. No GPU
  backend is upstreamed; no GPU port roadmap appears in the project's
  README, Issues, or `CHANGELOG.md`.
- **[verified]** NVIDIA Video Codec SDK 13.0 release notes
  (<https://developer.nvidia.com/video-codec-sdk>) — adds H.266
  encode on `NV_ENC_CODEC_H266_GUID` for Ada-and-newer; Hopper /
  Blackwell server SKUs exposed via the same surface. No support on
  Ampere or earlier. Decode landed earlier (SDK 12.x) on Ada.
- **[indicative]** AMD AMF SDK headers (`amf/public/include/components/VideoEncoderHEVC.h`,
  `VideoEncoderAV1.h`) — no `VideoEncoderVVC.h` / `VVC` enum present
  in shipping 1.4.x branches. AMD's RDNA 4 / Ryzen-AI launch
  briefings mention "next-gen video pipeline" but do not name VVC
  encode. Decode-only ASIC support is on the public roadmap; encode
  is unconfirmed. **[speculative]** RDNA 5 / UDNA timeframe is the
  earliest plausible silicon window.
- **[indicative]** Intel: Battlemage (Xe2-HPG) and the Lunar Lake
  iGPU (Xe2-LPG) ship VVC **decode** in the
  `oneVPL` / Media SDK matrix. VVC encode is not enumerated in
  current `mfx_status.h` codec capability flags as of oneVPL 2.10.
  Intel's GPU + media-architecture sessions at IDF / ISSCC have not
  named a VVC encoder ASIC.
- **[verified]** Khronos Vulkan Video extensions:
  - `VK_KHR_video_encode_h264` — final, ratified Dec 2023.
  - `VK_KHR_video_encode_h265` — final, ratified Dec 2023.
  - `VK_KHR_video_encode_av1` — final, ratified Sep 2024.
  - `VK_KHR_video_encode_h266` — **provisional / proposal**; tracked
    in the Vulkan-Docs repo issues; no ratified spec at write time.
    No proposed Mesa MR exists for an open-source implementation.
- **[verified]** Mesa 24.x / 25.x release notes
  (<https://docs.mesa3d.org/relnotes>) — RADV ships AV1 encode
  on RDNA 3 / RDNA 4; ANV ships AV1 encode on Battlemage. H.266
  encode is **not** present on any Mesa driver path. Decode landed
  in `radv_video.c` for VVC on RDNA 4 in 24.3.
- **[verified]** SVT-AV1 repository: pure-CPU encoder; no HIP /
  CUDA / SYCL fork has been upstreamed. Several AMD-affiliated
  contributors ship CDNA-targeted ROCm prototypes but none reach
  feature parity. Cited as a precedent for "open-source codec gets
  GPU-ported" being non-trivial.
- **[indicative]** x265 — third-party CUDA / OpenCL ports exist
  (Beamr, MainConcept) but none open-source. ROCm fork: none upstream.
- **[verified]** ZLUDA (<https://github.com/vosen/ZLUDA>) — runs
  unmodified CUDA binaries on AMD ROCm and Intel Level Zero. After
  the AMD-funded development phase paused (2024), the project
  reactivated under community maintenance in 2025. Status remains
  "experimental"; coverage of CUDA Driver API and cuBLAS / cuFFT is
  partial.
- **[verified]** ONNX Runtime EP matrix
  (<https://onnxruntime.ai/docs/execution-providers/>) — CUDA,
  ROCm, DirectML, OpenVINO, CoreML, WebGPU, QNN. The fork's NN-VC
  hot path is a residual-CNN intra-prediction model that runs on
  any of these EPs.
- **[verified]** Fork-internal:
  - `tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py` —
    the existing CPU adapter, with NN-VC toggle (`--nnvc-intra`,
    `--nnvc-loop-filter`, `--nnvc-sr`).
  - `tools/vmaf-tune/AGENTS.md` — describes the NN-VC integration
    surface and Phase A scope boundary.
  - [ADR-0290](../adr/0290-vmaf-tune-nvenc-adapters.md) — current
    NVENC adapter ladder (h264 / hevc / av1 only at write time).
  - [ADR-0127](../adr/0127-vulkan-compute-backend.md) — fork's
    Vulkan compute backend (scoring, not encoding).
  - [ADR-0033](../adr/0033-hip-applicability.md) — prior HIP-port
    feasibility note (focused on VMAF features, not codecs).

## Findings

### F1. NVENC is the only shipping hardware VVC encoder in 2026

NVIDIA's Video Codec SDK 13.0 exposes H.266 encode on Ada-Lovelace
silicon and newer (RTX 40 / 50 series consumer cards, RTX A-series
workstation cards on Ada, plus the Hopper / Blackwell datacentre
SKUs). FFmpeg ≥ 7.1 wires this through `h266_nvenc` and exposes the
canonical NVENC knob set (`-preset`, `-rc`, `-cq`, etc.). For a fork
user holding an RTX 40-class card, *real-time* VVC encode at 1080p is
already a reality.

The blocker for vendor neutrality: the encoder lives in fixed-function
silicon (NVENC ASIC), not in CUDA. Even if the fork wired up the
adapter (it has not yet — see "Open questions"), the hardware
dependency is a closed-silicon block. CUDA itself has no role here;
NVENC is its own pipeline.

### F2. AMD AMF and Intel QSV ship VVC **decode** but not **encode**

AMD's UVD / VCN media engines on RDNA 4 and Intel's Battlemage and
Lunar Lake iGPUs both expose VVC decode in their respective SDK
capability matrices. Encode is *absent*. Public roadmaps name RDNA 5 /
UDNA (AMD) and "Panther Lake / Nova Lake media engine" (Intel) as the
next-silicon windows that *could* add VVC encode, but neither vendor
has confirmed a date or a SKU. The earliest plausible silicon ship
window for a non-NVIDIA hardware VVC encoder is **2027–2028**; this is
**[speculative]**.

### F3. Vulkan Video VVC encode is unratified and has no Mesa work

`VK_KHR_video_encode_h266` is tracked as a proposal in the
Vulkan-Docs issue tracker and Khronos Working Group calendar; no
ratified spec exists at write time. The AV1 precedent is instructive:
the AV1 encode extension was provisional in 2023, ratified Sep 2024,
landed in Mesa RADV / ANV in early 2025 — a roughly 18-to-24 month
spec-to-driver lag. Applying the same lag to a VVC encode extension
not yet ratified yields a 2027-or-later window for usable Mesa-side
code on AMD / Intel.

This means **Vulkan Video is not a near-term play for VVC encode**.
It remains the right answer 2-3 years out: it is vendor-neutral by
construction, sits at the right driver layer for libvmaf to integrate
on the encode side as well as the score side, and reuses the fork's
existing Vulkan loader / queue / DMABUF plumbing
([ADR-0127](../adr/0127-vulkan-compute-backend.md),
[ADR-0186](../adr/0186-vulkan-image-import-impl.md)).

### F4. HIP-porting vvenc is non-trivial but bounded

VVenC's hot loops fall into the classic video-encoder shape:

| Hot kernel | Approx % of single-thread CPU time | GPU-port suitability |
|---|---|---|
| Motion estimation (block-match) | 30–45 % | Excellent — embarrassingly parallel over candidate vectors; precedent: every hardware encoder. |
| Intra-prediction mode decision | 10–20 % | Good — per-block parallelism, but small per-block work; needs batching. |
| Transform + quantisation (DCT, MTS, LFNST) | 8–15 % | Good — fixed-size GEMM-shaped kernels; FFT / matmul-flavoured. |
| In-loop filtering (deblocking, SAO, ALF) | 8–12 % | Moderate — neighbour dependencies; needs careful tile-level fencing. |
| Entropy coding (CABAC) | 5–10 % | Poor — inherently serial; stays on CPU. |
| Rate control + RDO | 10–15 % | Mixed — RDO inner loops parallelisable; outer rate control is sequential. |

A HIP / SYCL port that targets motion estimation + transforms +
loop-filtering would cover the largest contiguous chunk (~50–70 % of
CPU time) and could plausibly hit a 3–5× wall-clock speedup against
a 16-thread x86 baseline at the `vvenc` `slow` preset on RDNA 4 /
PVC-class hardware. **[speculative]** numbers; calibration would be a
Tier-2 deliverable.

The **license** is friendly: VVenC is Apache-2.0 with a separate
patent grant from Fraunhofer's VVC-essential pool — a HIP-augmented
fork is permitted as long as the fork's contributions are also
Apache-2.0. The Fraunhofer patent licence covers users who pay (or
qualify for the free-use tier); a fork of the *encoder* does not
escape that obligation.

The **maintenance** burden is the real cost. VVenC is on a
~quarterly release cadence (`v1.13.x` in May 2026); each upstream
release that touches a HIP-ported kernel triggers a port rebase. The
fork's ADR-0009 batch-port playbook already exists for libvmaf
upstream syncs and would extend cleanly to a vendored vvenc-hip
codebase, at the cost of one engineer-week per upstream release on
the high end.

### F5. SYCL / oneAPI is the path-not-taken

The same kernels HIP-port to AMD also SYCL-port to Intel PVC / Xe2 /
Lunar Lake iGPUs. The fork already has SYCL plumbing
(`libvmaf/src/sycl/`) and the team's SYCL skill from the SSIMULACRA 2
and CAMBI ports applies. Effort doubles compared to HIP-only (two
backends to maintain) but coverage triples (NVIDIA via Codeplay's
`oneAPI for NVIDIA GPUs` plugin, AMD via Codeplay's `oneAPI for AMD
GPUs` plugin, Intel native).

For a Tier-2 fork-local effort, picking **HIP first** keeps scope to
one new backend; SYCL is a natural Tier-2.5 follow-up if HIP lands
cleanly.

### F6. NN-VC is the *de-facto* vendor-neutral H.266 path today

VVC's neural-video-coding tools — NN-intra prediction, NN-loop-filter,
NN-super-resolution — run as ONNX models inside the encoder. The
fork's `vvenc.py` adapter already exposes them
(`--nnvc-intra` / `--nnvc-loop-filter` / `--nnvc-sr`,
[`tools/vmaf-tune/AGENTS.md`](../../tools/vmaf-tune/AGENTS.md)).
Because they go through ONNX Runtime, they execute on **any** EP:
CUDA, ROCm, DirectML, OpenVINO, CoreML, WebGPU. That is the cleanest
"VVC + GPU + non-NVIDIA" story available in 2026 — even though the
hot CPU encoder loops still run on CPU, the *neural* portion of the
encode rides whatever GPU is available.

The catch: NN-VC accelerates the *quality* of the encode, not the
*throughput*. A user with an AMD card running NN-VC-enabled vvenc
gets roughly the same wall-clock encode time as the same user
without the GPU; the GPU is only burning the NN-intra inference. The
quality lift (~1–3 % bitrate at iso-VMAF) is real and is where the
"vendor-neutral GPU contribution" actually lands today.

### F7. ZLUDA is technically interesting, operationally a dead end

ZLUDA could in principle run a (hypothetical) closed-source CUDA VVC
encoder on AMD / Intel hardware. In practice:

1. No such CUDA VVC encoder exists in the open-source ecosystem
   (NVENC is the closed silicon path; CUDA-only codecs are uncommon).
2. ZLUDA's CUDA Driver API coverage and cuBLAS / cuFFT support remain
   incomplete; a complex codec workload is not a target use-case the
   project's CI exercises.
3. The fork would inherit ZLUDA's "experimental, unsupported"
   posture for a critical encode path. Reviewers would reasonably
   reject this.

ZLUDA stays a curiosity, not a production lever. The fork's CUDA
**scoring** path will not benefit from ZLUDA either; the Vulkan
backend is the right answer for non-NVIDIA hardware.

## Cost / risk / value matrix

Effort is in eng-months for a single contributor at the fork's
current cadence. License risk is Apache-2.0 / BSD-3 unless flagged.
User-facing value is qualitative ("which users does this unlock?").

| Path | Effort | Licence risk | User value | Maintenance burden | Verdict |
|---|---|---|---|---|---|
| **A. NVENC h266 adapter** (analogue of [ADR-0290](../adr/0290-vmaf-tune-nvenc-adapters.md)) | 0.25 | None | RTX 40+ users only | Low (mirror of existing NVENC ladder) | Worth landing as a separate small PR. **Not** vendor-neutral. |
| **B. Vulkan-scoring quick-win** (sibling [ADR-0314](../adr/0314-vmaf-tune-vulkan-score-quick-win.md)) | 0.5 | None | All non-NVIDIA users get GPU **scoring** for VVC encodes (still CPU-encoded) | Low — reuses existing Vulkan backend | **Tier 1**: ship now. |
| **C. NN-VC documentation + corpus integration** | 0.25 | Mixed (NN-VC weights are LGPL-derived; NN-VC tooling itself is Apache-2.0) | All users with any GPU; ~1–3 % bitrate gain at iso-VMAF | Low | **Tier 1**: ship in same PR family. |
| **D. HIP port of vvenc hot kernels** | 3–6 | Apache-2.0 OK; Fraunhofer patent licence still applies to encoder users | RDNA 3 / 4 / CDNA users get 3–5× CPU-vvenc speedup **[speculative]** | Medium-high (per-release rebase) | **Tier 2**: backlog, gated on Tier 1 success. |
| **E. SYCL port of same kernels** | +2 (incremental over D) | Apache-2.0 OK | Adds Intel PVC / Xe2 + cross-vendor via Codeplay plugins | Medium | **Tier 2.5**: deferred. |
| **F. Vulkan Video h266 encode** | 1–2 (after spec ratification + driver landing) | None | All Vulkan-1.4+ users on AMD / Intel / NVIDIA | Low (driver-side does the work) | **Tier 3**: revisit quarterly. |
| **G. ZLUDA-hosted CUDA-VVC** | n/a | Risky; no production posture | None today (no open-source CUDA VVC encoder exists) | High (ZLUDA churn) | **Rejected**. |
| **H. Wait for AMD/Intel hardware VVC encode** | 0 (passive) | None | None until ~2027–2028 silicon | None | Not actionable; reflected in Tier 3's "revisit quarterly". |

## Recommendations (priority order)

### Tier 1 — ship today (≤ 1 eng-month combined)

1. **Wire vmaf-tune's existing Vulkan scoring** so non-NVIDIA users
   benefit from GPU-accelerated *scoring* of CPU-encoded VVC bitstreams.
   Scoped separately as **ADR-0314** (sibling, this PR cites but does
   not implement).
2. **Document NN-VC as the vendor-neutral H.266 GPU story** in the
   fork's user-facing docs (`docs/usage/vmaf-tune.md` and the
   forthcoming `docs/codecs/vvc.md`). Make explicit that:
   - Hardware VVC encode requires NVENC on Ada+ today.
   - VVenC + NN-VC is the open-source vendor-neutral fallback;
     encoder loops stay on CPU but neural tools accelerate on any
     ONNXRuntime-supported GPU.
   - Vulkan-side scoring closes the GPU loop on the consumption side.

### Tier 2 — backlog, queued (~3–6 eng-months)

3. **HIP port of vvenc's motion-estimation + transform + loop-filter
   kernels.** Gated by:
   - Tier 1 docs landing and at least one user reporting the
     CPU-vvenc throughput is the binding constraint.
   - A reproducer benchmark (a 1080p AOMtest sample at `slow` preset
     on a 16-core x86 baseline vs RDNA 4) showing the projected 3–5×
     speedup is achievable on a prototype.
   - An ADR sequel (ADR-0316 or later) proposing a fork of vvenc
     under `vendor/vvenc-hip/` with a clean upstream-rebase plan.

### Tier 3 — speculative, revisit quarterly (~12+ eng-months out)

4. **Vulkan Video VVC encode adapter.** Gated by:
   - `VK_KHR_video_encode_h266` ratification (Khronos working group).
   - At least one driver shipping the extension (Mesa RADV, NVIDIA
     proprietary, or ANV — any one suffices to start).
   - A libvmaf-side encode adapter design proposal (currently the
     fork's Vulkan code is score-only).

### Tier-1 → Tier-2 transition gate

The transition is **demand-pulled**, not calendar-pulled. Trigger the
Tier-2 HIP port only when **all three** of the following are true:

- A user has reported CPU-vvenc throughput as a binding constraint
  on a real corpus (not a synthetic benchmark).
- Tier-1 NN-VC docs have landed and at least one downstream consumer
  is using NN-VC tools through `vmaf-tune` in production.
- An RDNA 3 / RDNA 4 or Intel PVC machine is available to the fork
  for ongoing CI (without it the HIP port has no automated regression
  gate and ADR-0214's GPU-parity rule is unenforceable).

If any of the three is false, Tier 2 stays in the backlog. This
matches the fork's "demand-pulled fork-local effort" pattern from
ADR-0009.

## Reproducer / verification commands

```bash
# 1. Confirm the NVENC SDK exposes H.266 (on a machine with NVENC SDK 13+):
nvidia-smi --query-gpu=name,driver_version --format=csv
ffmpeg -hide_banner -h encoder=h266_nvenc 2>&1 | head -5

# 2. Confirm AMF / QSV do NOT expose VVC encode today:
ffmpeg -hide_banner -encoders 2>&1 | grep -iE 'h266|vvc'
# expected: only h266_nvenc (and libvvenc) enumerate; no h266_amf / h266_qsv.

# 3. Confirm Mesa Vulkan does NOT expose VK_KHR_video_encode_h266:
vulkaninfo | grep -iE 'video_encode_(h264|h265|av1|h266)'
# expected: h264/h265/av1 present on RDNA 4; h266 absent.

# 4. The fork's CPU adapter still works for vendor-neutral encode (slow):
python -m vmaftune encode --codec libvvenc --preset slow \
    --input testdata/yuv/akiyo_qcif.yuv --qp 32 --output /tmp/akiyo.266

# 5. NN-VC tools on whatever GPU is present (ONNXRuntime EP is auto-detected):
python -m vmaftune encode --codec libvvenc --preset medium \
    --nnvc-intra --input testdata/yuv/akiyo_qcif.yuv --qp 32 \
    --output /tmp/akiyo_nnvc.266

# 6. This PR's own gate — docs build clean:
mkdocs build --strict 2>&1 | grep -E "(WARNING|ERROR)" || echo "docs build clean"
```

## Open questions

1. **Does Fraunhofer HHI have an internal GPU-port roadmap for VVenC
   that is not yet public?** The repo issues mention occasional
   "GPU acceleration" inquiries but no commitments. Worth opening a
   discussion thread before committing engineering hours to a HIP
   fork (avoids reinventing what upstream may ship in 12 months).
2. **What is the realistic ratification date for `VK_KHR_video_encode_h266`?**
   Khronos Working Group cadence is opaque; the AV1 precedent
   (~18 months from provisional to final) is the best public proxy.
3. **Will AMD's RDNA 5 / UDNA generation include a VVC encoder ASIC?**
   Public roadmap silent; encoder ASIC silicon decisions are typically
   locked 24+ months ahead of launch.
4. **Will Intel's Panther Lake / Nova Lake media engine include VVC
   encode?** Same uncertainty as the AMD question.
5. **Does NN-VC quality lift survive at low bitrates and on UGC / sports
   content?** The fork's existing corpus is biased to film / animation;
   Tier-1 documentation should call out the genre-coverage gap honestly.
6. **Is there a HIP port of x265 or SVT-AV1 we can crib patterns from?**
   Indicative search suggests no production-grade port; if one exists
   privately at AMD or a downstream studio, an upstream pointer would
   shave ~1 month off the Tier-2 effort.

## Related

- [ADR-0315](../adr/0315-vendor-neutral-vvc-encode-strategy.md) —
  the decision this digest fed.
- [ADR-0314](../adr/0314-vmaf-tune-vulkan-score-quick-win.md) —
  Tier-1 sibling, scoped separately, wires Vulkan scoring through
  `vmaf-tune`.
- [ADR-0290](../adr/0290-vmaf-tune-nvenc-adapters.md) — NVENC
  adapter ladder; analogous h266 adapter is a separate small PR.
- [ADR-0033](../adr/0033-hip-applicability.md) — prior HIP
  applicability survey for libvmaf features.
- [ADR-0127](../adr/0127-vulkan-compute-backend.md) — Vulkan compute
  backend for scoring.
- [`tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py`](../../tools/vmaf-tune/src/vmaftune/codec_adapters/vvenc.py)
  — the existing CPU adapter, with NN-VC toggle.
- VVenC upstream: <https://github.com/fraunhoferhhi/vvenc>
- NVIDIA Video Codec SDK: <https://developer.nvidia.com/video-codec-sdk>
- Khronos Vulkan Video extensions:
  <https://www.khronos.org/blog/khronos-video-extensions>
