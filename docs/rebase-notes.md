# Rebase notes

Single ledger of fork-local changes that need attention when this fork
syncs from `upstream/master` (Netflix/vmaf). Required by
[ADR-0108](adr/0108-deep-dive-deliverables-rule.md): every fork-local
PR that touches upstream-shared paths or establishes a rebase-sensitive
invariant adds an entry here. PRs with no rebase impact state "no
rebase impact" in the PR description and skip the entry.

The intended reader is whoever runs the next `/sync-upstream` (see
[ADR-0002](adr/0002-merge-path-master-default.md) and
`.claude/skills/sync-upstream/`). Read top-to-bottom before resolving
conflicts.

## Format

Each entry is a `### NNNN — short title` heading with three fields:

- **Touches**: paths likely to conflict on upstream merge.
- **Invariant**: what the fork relies on that an upstream change could
  silently drop.
- **Re-test**: the command(s) to run after the merge to confirm the
  invariant survived. Reproducer-style — no surrounding prose required.

IDs are assigned in commit order and never reused. A single entry may
cover several PRs in one workstream; cross-link from the ID heading.

## Entries (backfilled 2026-04-18 per ADR-0108 adoption)

### 0228 — Vulkan 1.4 bump deferred (ADR-0264, docs-only)

- **Touches**: none (docs-only PR). Future Step A of T-VK-1.4-BUMP
  will touch `libvmaf/src/feature/vulkan/shaders/vif.comp` and
  `libvmaf/src/feature/vulkan/shaders/ciede.comp`; Step B will touch
  the three `apiVersion` sites in `libvmaf/src/vulkan/common.c` (lines
  54, 264, 374) and the `VMA_VULKAN_VERSION` define in
  `libvmaf/src/vulkan/vma_impl.cpp` (line 22).
- **Invariant**: `master` stays on `VK_API_VERSION_1_3` and
  `VMA_VULKAN_VERSION = 1003000`. Lifting the constant in any future
  upstream sync (Netflix doesn't ship a Vulkan backend, so the
  conflict is improbable) without first auditing `precise` /
  `OpDecorate ... NoContraction` decoration on `vif.comp` and
  `ciede.comp` will reintroduce the NVIDIA-driver regression captured
  in [research-0053](research/0053-vulkan-1-4-nvidia-fp-contraction-regression.md).
  The `psnr_hvs_strict_shaders` `-O0` list in
  [`libvmaf/src/vulkan/meson.build`](../libvmaf/src/vulkan/meson.build)
  is the existing precedent for shader-side bit-exactness mitigations
  and should be the place a 1.4-era audit lands its results
  (potentially expanding to cover `vif.comp` + `ciede.comp` if the
  `precise` audit decides the optimizer is the right place to gate).
- **Re-test**: when Step B lands, the gate is
  `python3 scripts/ci/cross_backend_vif_diff.py --feature vif --backend vulkan`
  and the same with `--feature ciede` against NVIDIA + RADV +
  lavapipe; max abs diff must stay ≤ `5.0e-05` (`places=4`) on all
  three.
### 0229 — HIP fifth-consumer kernel `float_ansnr_hip` (ADR-0266)
### 0228 — `y4m_convert_411_422jpeg` 1-byte heap-buffer-overflow fix

- **Touches**:
  - `libvmaf/tools/y4m_input.c` — upstream-mirrored Daala-derived Y4M
    parser. The fix sits inside the 4:1:1 → 4:2:2-jpeg chroma upsample
    routine `y4m_convert_411_422jpeg`, lines ~500–530 in the function's
    three sub-loops. Upstream Netflix/vmaf carries the same shape; if
    upstream lands its own fix during a sync, prefer the upstream
    version and drop ours.
  - `libvmaf/test/test_y4m_411_oob.c` (new, fork-local) — drives the
    minimal W=2 H=4 4:1:1 stream through `video_input_open` +
    `video_input_fetch_frame`. Wholly fork-added; no upstream collision.
  - `libvmaf/test/meson.build` — adds `test_y4m_411_oob` executable +
    `test()` registration.
- **Invariant**: the **first two** sub-loops of `y4m_convert_411_422jpeg`
  must guard `_dst[(x << 1) | 1]` writes with `(x << 1 | 1) < dst_c_w`,
  matching the third sub-loop's existing guard. Without the guard a
  4:1:1 stream of width 2 (`dst_c_w == 1`) writes one byte past the
  destination chroma row.
- **Re-test**:
  - `cd libvmaf && meson setup ../build-asan --buildtype=debug
    -Db_sanitize=address -Db_lundef=false -Denable_cuda=false
    -Denable_sycl=false -Denable_vulkan=disabled`
  - `ninja -C build-asan test/test_y4m_411_oob`
  - `ASAN_OPTIONS=detect_leaks=0 ./build-asan/test/test_y4m_411_oob`
    — must report `1 tests run, 1 passed`. Pre-fix the binary aborts
    with `AddressSanitizer: heap-buffer-overflow … WRITE of size 1`
    at `y4m_input.c:507`.

### 0227 — `tools/vmaf-tune/` Phase A scaffold (ADR-0237 Phase A)

- **Touches**:
  - `libvmaf/src/feature/hip/float_ansnr_hip.{c,h}` (new) — fifth
    consumer of `libvmaf/src/hip/kernel_template.h`. Mirrors
    `libvmaf/src/feature/cuda/float_ansnr_cuda.c`
    call-graph-for-call-graph; `init/submit/collect/close` invoke
    the kernel-template helpers in the same order; the submit
    body intentionally bypasses
    `vmaf_hip_kernel_submit_pre_launch` (no atomic, kernel writes
    per-block (sig, noise) interleaved float partials directly).
  - `libvmaf/src/hip/meson.build` — adds the new TU to
    `hip_sources`.
  - `libvmaf/src/feature/feature_extractor.c` — adds the
    `extern VmafFeatureExtractor vmaf_fex_float_ansnr_hip;`
    declaration and the registry row under `#if HAVE_HIP`.
  - `libvmaf/test/test_hip_smoke.c` — adds
    `test_float_ansnr_hip_extractor_registered` sub-test pinning
    the lookup contract.
- **Invariant — the `submit_pre_launch` bypass is load-bearing**.
  The CUDA twin makes the same choice for the same reason. If a
  future PR adds a `submit_pre_launch` call to
  `float_ansnr_cuda.c`'s submit path, the HIP twin must follow in
  the same PR. Likewise the readback shape
  (`wg_count * 2u * sizeof(float)`) and the bpc table
  (peak/psnr_max for 8/10/12/16-bit) mirror the CUDA twin
  verbatim — keep aligned on rebase.
- **Re-test on rebase**:

  ```bash
  cd libvmaf
  meson setup build -Denable_hip=true -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build  # 48/48 green (47 CPU + HIP smoke)
  ```

### 0230 — HIP sixth-consumer kernel `motion_v2_hip` (ADR-0267)

- **Touches**:
  - `libvmaf/src/feature/hip/integer_motion_v2_hip.{c,h}` (new) —
    sixth consumer of `libvmaf/src/hip/kernel_template.h`. Mirrors
    `libvmaf/src/feature/cuda/integer_motion_v2_cuda.c`
    call-graph-for-call-graph; carries the
    `VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag and a `flush()`
    callback. The state struct has a `uintptr_t pix[2]` ping-pong
    slot pair tracked outside the kernel-template (the template
    models a single device+host pair only).
  - `libvmaf/src/hip/meson.build` — adds the new TU to
    `hip_sources`.
  - `libvmaf/src/feature/feature_extractor.c` — adds the
    `extern VmafFeatureExtractor vmaf_fex_integer_motion_v2_hip;`
    declaration and the registry row under `#if HAVE_HIP`.
  - `libvmaf/test/test_hip_smoke.c` — adds
    `test_motion_v2_hip_extractor_registered` sub-test pinning
    the lookup contract (extractor name is `motion_v2_hip`,
    matching the CUDA twin's `motion_v2_cuda` naming).
- **Invariant — temporal-extractor + ping-pong shape**. The
  `VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag bit, the `flush()`
  callback registration, and the `uintptr_t pix[2]` slot pair are
  load-bearing for the runtime PR (T7-10b). The runtime PR will
  swap `uintptr_t pix[2]` for a real device-buffer handle pair
  matching the CUDA twin's `VmafCudaBuffer *pix[2]`. On rebase:
  if the CUDA twin's flush-pass shape changes (currently
  `min(score[i], score[i+1])`), update the HIP twin's
  `flush_fex_hip` body in the same PR.
- **Re-test on rebase**: same as 0229 — `meson test -C build`
  with `enable_hip=true` exercises the smoke contract.

### 0227 — ms_ssim_vulkan submit-side migrated to kernel_template (T-GPU-DEDUP-26)

- **Touches**:
  - `libvmaf/src/feature/vulkan/ms_ssim_vulkan.c` — `extract()`'s
    raw `VkCommandBuffer` / `VkFence` / `vkAllocateCommandBuffers` /
    `vkBeginCommandBuffer` / `vkCreateFence` / `vkQueueSubmit` /
    `vkWaitForFences` / `vkDestroyFence` / `vkFreeCommandBuffers`
    blocks become `VmafVulkanKernelSubmit` triples
    (`vmaf_vulkan_kernel_submit_begin` /
    `_submit_end_and_wait` / `_submit_free`). One triple covers the
    decimate-pyramid command buffer; one triple per scale covers the
    per-scale SSIM submit. The pipeline-side bundles
    (`pl_decimate` 2-binding 4-variant + `pl_ssim` 10-binding
    9-variant) and their `_add_variant()` chains are unchanged from
    the prior migration.
- **Invariant**: any future submit-side template change (timeline
  semaphores, deferred fence release, queue-family parameterisation)
  must keep the helpers' synchronous-wait + per-frame fence + per-frame
  command-buffer contract intact, since `ms_ssim_vulkan.c` does host
  readback of the `l_partials` / `c_partials` / `s_partials` buffers
  immediately after `_submit_end_and_wait` returns. The submit-side
  contract is the same one already documented in
  `libvmaf/src/vulkan/AGENTS.md`'s "Rebase-sensitive invariants"
  section for `kernel_template.h`.
- **Re-test**:

  ```bash
  cd libvmaf && meson test -C build
  python scripts/ci/cross_backend_vif_diff.py \
      --vmaf-binary libvmaf/build/tools/vmaf \
      --reference testdata/ref_576x324_48f.yuv \
      --distorted testdata/dis_576x324_48f.yuv \
      --width 576 --height 324 \
      --feature float_ms_ssim --backend vulkan --places 4
### 0231 — SHA-pin GitHub Actions (OSSF `Pinned-Dependencies`)

- **Touches**: every workflow file under `.github/workflows/`.
  All 13 fork workflows (`docker-image.yml`, `docs.yml`,
  `ffmpeg-integration.yml`, `libvmaf-build-matrix.yml`,
  `lint-and-format.yml`, `nightly-bisect.yml`, `nightly.yml`,
  `release-please.yml`, `rule-enforcement.yml`, `scorecard.yml`,
  `security-scans.yml`, `supply-chain.yml`,
  `tests-and-quality-gates.yml`) had their `uses:` directives
  rewritten from `<owner>/<repo>@vN[.M.K]` to
  `<owner>/<repo>@<40-char-sha>  # vN.M.K`. 97 references converted;
  the SLSA reusable-workflow ref in `supply-chain.yml` is the single
  documented holdout (see `Invariant` below).
- **Invariant — SHA-pin policy for `uses:`**. Every action reference
  in `.github/workflows/*.yml` MUST be a 40-char commit SHA with the
  semver tag preserved as a trailing `# vN.M.K` comment. The OSSF
  Scorecard `Pinned-Dependencies` check parses both forms and a
  floating tag (`@vN`) is treated as unpinned and counts against the
  aggregate score. **Single permitted exception**: the SLSA generator
  reusable workflow
  (`slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml`)
  must keep its `vX.Y.Z` tag form because GitHub Actions consumers
  cannot SHA-pin reusable-workflow refs in every code path; the
  exception is documented inline in `supply-chain.yml` and survives
  on each rebase. **Why this matters on upstream sync**: Netflix
  upstream does not ship the fork's CI tree, so a `/sync-upstream`
  run that drags new workflow content (e.g. via repository templates
  or bot-authored bumps) into `.github/workflows/` can re-introduce
  floating-tag references unnoticed. The post-rebase check below is
  the standing gate — anything that lights up needs to be re-pinned
  before merging the sync.
- **Re-test on rebase**:

  ```bash
  # Anything that prints is a regression — every uses: must be either
  # already SHA-pinned (40 hex) or, for the documented SLSA exception,
  # the slsa-github-generator reusable-workflow ref.
  grep -hnE '^\s*(- )?uses:\s+[^@]+@[^ #]+\s*$' .github/workflows/*.yml \
    | grep -vE '@[a-f0-9]{40}' \
    | grep -v 'slsa-framework/slsa-github-generator/.github/workflows/'
  # SHA-resolution sanity for any new pin (per-action):
  gh api repos/<owner>/<repo>/git/ref/tags/<vN.M.K> --jq '.object.sha'
  # If the result is a "tag" object (annotated tag), deref:
  gh api repos/<owner>/<repo>/git/tags/<sha-from-prev> --jq '.object.sha'
  ```

### 0226 — CUDA drain-batch engine-loop opt (T-GPU-OPT-1)

- **Touches**:
  - `libvmaf/src/cuda/drain_batch.{h,c}` (new) — TLS drain-batch
    table + shared drain stream + `_open()/register/_flush()/_close()`
    API.
  - `libvmaf/src/libvmaf.c` — engine-side per-frame loop now wraps
    submit/collect with `_open()` + `_flush()` so all CUDA
    extractor `finished` events are waited on a single shared
    drain stream.
  - All 12 CUDA feature kernels (`libvmaf/src/feature/cuda/*.c`)
    register their `finished` event + `drained` flag with the
    drain batch on submit; collect skips its private
    `cuStreamSynchronize` when `drained` is true.
- **Invariant — drained-flag contract**. Every CUDA extractor's
  collect path **must** check the per-frame `drained` flag and
  skip its own `cuStreamSynchronize` when set; otherwise the
  drain batching is a no-op. The flag is reset to `false` per
  frame inside `vmaf_cuda_drain_batch_register()`.
- **Re-test on rebase**:

  ```bash
  cd libvmaf
  meson setup build -Denable_cuda=true -Denable_sycl=false
  ninja -C build
  meson test -C build --suite=fast cuda
  ```

  Expected: all CUDA tests green; bench shows ≥5% wall-clock
  gain on a 7-extractor VMAF model (model.json with all
  feature extractors enabled).

### 0225 — Netflix bench snapshot regen (upstream a44e5e61 motion fix)

- **Touches**:
  - `testdata/netflix_benchmark_results.json` — fork-added snapshot.
    CPU rows now reflect the post-fix motion feature; cuda / sycl
    rows from the previous regen are preserved unchanged because
    those backends were not exercised on this rerun (host-environment
    tooling — wrong renderD path, `libvmaf_cuda` not enabled in the
    local FFmpeg build). Future full regens should include cuda /
    sycl.
  - `testdata/bench_all.sh` — default `VMAF=` no longer points at
    `/usr/local/bin/vmaf` (which on most dev hosts is stuck at the
    pre-upstream-`a44e5e61` v3.0.0); now defaults to the in-tree
    fork build at `libvmaf/build/tools/vmaf`.
  - `testdata/benchmark_netflix.py` — `FFMPEG`, `YUVDIR` and the
    hardcoded `LD_LIBRARY_PATH=/usr/local/lib` are now overridable
    via `VMAF_FFMPEG`, `VMAF_YUVDIR` and any caller-set
    `LD_LIBRARY_PATH`.
- **Invariant**: the snapshot's CPU pooled VMAF for
  `src01_576x324` is 76.667828 (post-fix), not 76.668904 (the
  upstream-buggy mirror). If `/sync-upstream` ever re-pulls a
  Netflix change that touches `motion.c` mirror-handling, this
  number is the reference.
- **Re-test**:

  ```bash
  cd libvmaf
  meson setup build -Denable_cuda=true -Denable_sycl=false
  ninja -C build
  LD_LIBRARY_PATH=$(pwd)/build/src python3 \
      ../testdata/benchmark_netflix.py
  ```

  Expected CPU pooled rows: 76.667828, 35.068672, 7.985899.

### 0224 — CUDA graph capture feasibility (research-0047, DEFER)

- **Touches**: none — investigation-only; no code lands. The research
  digest [`docs/research/0047-cuda-graph-capture-feasibility.md`](research/0047-cuda-graph-capture-feasibility.md)
  documents why a CUDA graph capture path on the per-frame submit chain
  is **deferred** rather than shipped (realised wall-clock gain capped
  at ~1-3% vs. the predicted 10-20%, with a 4-slot picture-pool
  rotation that defeats single-graph capture and forces per-frame
  `cuGraphExecKernelNodeSetParams` rebinding for `(ref, dis)` device
  pointers).
- **Invariant**: the `kernel_template.h` docstring keeps naming
  `VmafCudaKernelLifecycle.finished` as a graph-capture hook point.
  Don't prune that comment on rebase — leaving the door open in the
  template is free, and the digest's "what needs to be true for a
  future GO" section depends on the hook still being there.
- **Re-test on rebase**:

  ```bash
  # Confirm the docstring still references graph capture as the hook
  # point — wording change is fine, removal is not.
  grep -q "graph capture" libvmaf/src/cuda/kernel_template.h
  ```

### 0223 — ADR slug-drift repair in CHANGELOG / rebase-notes (PR #304 follow-up)

- **Touches**: `CHANGELOG.md`, `docs/rebase-notes.md`. No code; no
  upstream-shared path; no public-API surface.
- **Invariant**: every `[ADR-NNNN](docs/adr/NNNN-slug.md)` link in the
  fork's tracked docs resolves to an actual on-disk file under
  `docs/adr/`. Repaired 4 broken slugs that did not exist on disk
  (`0138-iqa-convolve-avx2-bitexact-double` → `0138-iqa-convolve-avx2-bitexact-double`,
  `0140-simd-dx-framework` → `0140-simd-dx-framework`,
  `0190-ms-ssim-vulkan` → `0190-ms-ssim-vulkan`,
  `0178-vulkan-adm-kernel` → `0178-vulkan-adm-kernel`). All retained
  their cited NNNN per ADR-0028 (NNNN is immutable once Accepted).
- **Re-test on rebase**: from repo root, the following must print no
  lines:

  ```bash
  for ref in $(grep -ohE 'docs/adr/[0-9]{4}-[a-z0-9-]+\.md' \
      CHANGELOG.md docs/rebase-notes.md AGENTS.md docs/state.md \
      | sort -u); do
    test -f "$ref" || echo "MISSING: $ref"
  done
  ```

### 0125 — cambi_vulkan migrated to kernel_template (T-GPU-DEDUP-25, 5-bundle)

- **Touches**:
  - `libvmaf/src/feature/vulkan/cambi_vulkan.c` — state's quintet
    (`dsl_2bind` + 5× `pl_layout_*` + `shader_modules[CAMBI_PL_COUNT]`
    + shared `desc_pool`) collapses to five
    `VmafVulkanKernelPipeline` bundles (`pl_trivial`,
    `pl_derivative`, `pl_filter_mode`, `pl_decimate`, `pl_mask_dp`),
    each owning its own descriptor pool. The first slot of
    `pipelines[]` per stage aliases the bundle's base pipeline;
    `CAMBI_PL_FILTER_MODE_V`, `CAMBI_PL_MASK_SAT_COL`, and
    `CAMBI_PL_MASK_THRESHOLD` are sibling variants built via
    `vmaf_vulkan_kernel_pipeline_add_variant()`.
  - `cambi_vk_alloc_set` takes a bundle pointer
    (`->desc_pool` / `->dsl`) — every dispatch site picks the
    bundle that matches its push-constant struct.
  - The `cambi_vk_make_dsl` / `cambi_vk_make_pl` /
    `cambi_vk_create_shader` / `cambi_vk_build_pipeline` helpers
    are dropped — the template subsumes them.
- **Invariant — variants destroyed before bundle, base alias must
  be skipped**. Five distinct push-constant struct sizes
  (`CambiVkPushTrivial` / `CambiVkPushDerivative` /
  `CambiVkPushFilterMode` / `CambiVkPushDecimate` /
  `CambiVkPushMaskDp`) force five bundles even though every
  stage's DSL is 2-binding SSBO; `_add_variant()` only siblings
  pipelines under the *same* layout. `close_fex` must
  `vkDestroyPipeline()` the variant slots
  (`CAMBI_PL_FILTER_MODE_V`, `CAMBI_PL_MASK_SAT_COL`,
  `CAMBI_PL_MASK_THRESHOLD`) *before* calling
  `vmaf_vulkan_kernel_pipeline_destroy()` on each bundle.
- **Numerical contract**: bit-exact preserved. Same shaders +
  spec-constants + push-constants as before; only the Vulkan
  pipeline-bundle scaffolding moved to the template. Validated
  on the Netflix-pair smoke (576×324×8-bit): `cambi` mean = 0.0,
  identical to pre-migration (the pair has no banding artifacts).
- **Rebase impact**: low. Builds on top of PR #272's
  `_add_variant()` helper. Upstream Netflix/vmaf has no Vulkan
  backend, so there is nothing to merge against.

### 0124 — ssimulacra2_vulkan migrated to kernel_template (T-GPU-DEDUP-24, 4-bundle)

- **Touches**:
  - `libvmaf/src/feature/vulkan/ssimulacra2_vulkan.c` — state's
    16 long-lived pipeline-object fields (4×
    `*_dsl + *_pl + *_shader` + the shared `desc_pool`) collapse
    to four `VmafVulkanKernelPipeline` bundles (`pl_xyb`, `pl_mul`,
    `pl_blur`, `pl_ssim`), each owning its own descriptor pool.
    The first slot of each per-bundle pipeline array
    (`xyb_pipelines[0]`, `mul_pipelines[0]`,
    `blur_pipelines_h[0]`, `ssim_pipelines[0]`) aliases the
    bundle's base `VkPipeline`; remaining per-scale / per-pass
    slots are siblings via
    `vmaf_vulkan_kernel_pipeline_add_variant()`.
  - `ss2v_build_pipeline_int3` reroutes through `_add_variant()`
    instead of calling `vkCreateComputePipelines` directly;
    `ss2v_alloc_set` takes a bundle pointer (`->desc_pool` /
    `->dsl`) instead of a separate DSL argument; descriptor-set
    free sites at the tail of `ss2v_run_scale` route to each
    bundle's pool.
  - The `ss2v_make_dsl` / `ss2v_make_pl` / `ss2v_create_shader`
    helpers are dropped — the template subsumes them.
- **Invariant — variants destroyed before bundle, slot 0 alias
  must be skipped**. Four distinct DSL shapes (XYB = 6 SSBOs,
  MUL = 3, BLUR = 2, SSIM = 8) prevent collapsing to one bundle:
  `_add_variant()` only siblings pipelines under the same layout.
  `close_fex` must `vkDestroyPipeline()` the variant slots in
  `xyb_pipelines[1..N-1]`, `mul_pipelines[1..N-1]`,
  `ssim_pipelines[1..N-1]`, `blur_pipelines_h[1..N-1]`, and
  every slot of `blur_pipelines_v[]` *before* calling
  `vmaf_vulkan_kernel_pipeline_destroy()` on each bundle, and
  must skip slot 0 of the first three arrays + `blur_pipelines_h`
  to avoid double-freeing the aliased base.
- **Numerical contract**: bit-exact preserved. Same shaders +
  spec-constants + push-constants as before; only the Vulkan
  pipeline-bundle scaffolding moved to the template. Validated
  on the Netflix-pair smoke (576×324×8-bit): `ssimulacra2`
  mean = 24.613842, identical to pre-migration.
- **Rebase impact**: low. Builds on top of PR #272's
  `_add_variant()` helper. Upstream Netflix/vmaf has no
  ssimulacra2 extractor and no Vulkan backend, so there is
  nothing to merge against.

### 0118 — psnr_hvs_vulkan migrated to kernel_template + `_add_variant` (T-GPU-DEDUP-18)

- **Touches**:
  - `libvmaf/src/feature/vulkan/psnr_hvs_vulkan.c` — state's
    `dsl + pipeline_layout + shader + desc_pool + pipeline[3]`
    collapses to `VmafVulkanKernelPipeline pl +
    VkPipeline pipeline_chroma_u + VkPipeline pipeline_chroma_v`.
    Plane 0 is the template's base pipeline; planes 1+2 are
    siblings via `vmaf_vulkan_kernel_pipeline_add_variant()`.
  - New `psnr_hvs_plane_pipeline()` accessor maps plane index to
    the right `VkPipeline` handle.
- **Invariant — variants destroyed before bundle**. `close_fex`
  must `vkDestroyPipeline()` the chroma U/V variants before
  calling `vmaf_vulkan_kernel_pipeline_destroy(&s->pl)` — same
  rule as ssim_vulkan in T-GPU-DEDUP-7.
- **Numerical contract**: unchanged. Same shaders + spec-constants
  + push-constants as before; only the Vulkan pipeline-bundle
  scaffolding moved to the template.
- **Rebase impact**: low. Builds on top of PR #272's
  `_add_variant()` helper.

### 0119 — vif_vulkan migrated to kernel_template + `_add_variant` (T-GPU-DEDUP-19)

- **Touches**:
  - `libvmaf/src/feature/vulkan/vif_vulkan.c` — state's
    `dsl + pipeline_layout + shader + desc_pool + pipelines[4]`
    collapses to `VmafVulkanKernelPipeline pl +
    VkPipeline scale_variants[3]`. Scale 0 is the template's
    base pipeline; scales 1, 2, 3 are siblings via
    `vmaf_vulkan_kernel_pipeline_add_variant()`.
  - New `vif_scale_pipeline()` accessor maps scale index to the
    right `VkPipeline` handle (replaces `s->pipelines[scale]`).
- **Invariant — variants destroyed before bundle**. `close_fex`
  must `vkDestroyPipeline()` the 3 scale variants before calling
  `vmaf_vulkan_kernel_pipeline_destroy(&s->pl)` — same rule as
  ssim_vulkan in T-GPU-DEDUP-7 and psnr_hvs_vulkan in
  T-GPU-DEDUP-18.
- **Numerical contract**: unchanged. Same shaders, same
  spec-constants, same push-constants as before; only the Vulkan
  pipeline-bundle scaffolding moved to the template.
- **Rebase impact**: low. Builds on top of PR #272's
  `_add_variant()` helper.

### 0120 — float_vif_vulkan migrated to kernel_template + `_add_variant` (T-GPU-DEDUP-20)

- **Touches**:
  - `libvmaf/src/feature/vulkan/float_vif_vulkan.c` — state
    collapses `dsl + pipeline_layout + shader + desc_pool` to
    `VmafVulkanKernelPipeline pl`; the
    `VkPipeline pipelines[2][4]` 2-D lookup table is preserved so
    the existing `[mode][scale]` dispatch path stays clean, but
    `pipelines[0][0]` aliases `s->pl.pipeline` (the template's
    base). The other 6 entries are sibling pipelines created via
    `vmaf_vulkan_kernel_pipeline_add_variant()`.
- **Invariant — variants destroyed before bundle**. `close_fex`
  must `vkDestroyPipeline()` the 6 sibling variants (every
  `(mode, scale)` except `(0, 0)`) before calling
  `vmaf_vulkan_kernel_pipeline_destroy(&s->pl)` — same rule as
  ssim_vulkan / psnr_hvs_vulkan / vif_vulkan.
- **Invariant — `pipelines[0][0]` aliasing**. The base pipeline
  handle is owned by `s->pl.pipeline`; we copy it into
  `pipelines[0][0]` after `_create()` so the dispatch path can
  use a uniform 2-D lookup. The destroy loop must skip
  `(mode=0, scale=0)` to avoid double-freeing the template's
  pipeline.
- **Numerical contract**: unchanged. Same shaders, spec-constants
  (`mode` + `scale`), push-constants. Netflix-pair smoke matches
  `integer_vif` bit-identically to 4 decimals.
- **Rebase impact**: low. Builds on top of PR #272's
  `_add_variant()` helper.
### 0122 — float_adm_vulkan migrated to kernel_template + `_add_variant` (T-GPU-DEDUP-22)

- **Touches**:
  - `libvmaf/src/feature/vulkan/float_adm_vulkan.c` — twin to
    adm_vulkan (T-GPU-DEDUP-21); 16-pipeline 2-D
    `[stage][scale]` array. State collapses
    `dsl + pipeline_layout + shader + desc_pool` to
    `VmafVulkanKernelPipeline pl`. `pipelines[0][0]` aliases
    `s->pl.pipeline`; the other 15 entries are siblings via
    `vmaf_vulkan_kernel_pipeline_add_variant()`.
- **Invariants**:
  - Variants destroyed before bundle.
  - `pipelines[0][0]` aliasing — destroy loop must skip
    `(stage=0, scale=0)`.
- **Numerical contract**: unchanged. Same float (`_s` suffix)
  primitives from `adm_tools.c`; same 5-element spec-constant
  tuple; same float partial accumulation reduced in double on
  the host.
### 0121 — adm_vulkan migrated to kernel_template + `_add_variant` (T-GPU-DEDUP-21)

- **Touches**:
  - `libvmaf/src/feature/vulkan/adm_vulkan.c` — state collapses
    `dsl + pipeline_layout + shader + desc_pool` to
    `VmafVulkanKernelPipeline pl`; the
    `VkPipeline pipelines[4][4]` 2-D lookup is preserved so the
    per-stage dispatch path stays clean. `pipelines[0][0]`
    aliases `s->pl.pipeline` (the template's base); the other
    15 entries are sibling pipelines via
    `vmaf_vulkan_kernel_pipeline_add_variant()`.
- **Invariants**:
  - Variants destroyed before bundle (same rule as
    ssim_vulkan / psnr_hvs / vif / float_vif).
  - `pipelines[0][0]` aliasing — destroy loop must skip
    `(stage=0, scale=0)` to avoid double-freeing the template's
    pipeline.
- **Numerical contract**: unchanged. Same shaders + 5-element
  spec-constant tuple (width, height, bpc, scale, stage) +
  push-constants.
- **Rebase impact**: low. Builds on top of PR #272.
### 0123 — `ms_ssim_vulkan` 2-bundle migration (T-GPU-DEDUP-23)

- **Touches**:
  - `libvmaf/src/feature/vulkan/ms_ssim_vulkan.c` — state collapses
    `decimate_dsl + decimate_pl + decimate_shader + ssim_dsl +
    ssim_pl + ssim_shader + desc_pool` (7 fields) to two bundles
    `VmafVulkanKernelPipeline pl_decimate` + `pl_ssim`. Each bundle
    owns its own descriptor pool. The kernel has two distinct
    pipeline shapes (decimate = 2 SSBO bindings, ssim = 10
    bindings), so two bundles is the minimum — `_add_variant()`
    only siblings pipelines under the *same* layout.
  - `decimate_pipelines[0]` aliases `pl_decimate.pipeline` (the
    template's base = scale 0). The remaining
    `MS_SSIM_SCALES - 2` decimate variants (scales 1..3) are
    siblings via `_add_variant()`.
  - `ssim_pipeline_horiz[0]` aliases `pl_ssim.pipeline` (base =
    scale 0, pass 0). The other 9 entries (4× `ssim_pipeline_horiz`
    for scales 1..4, plus 5× `ssim_pipeline_vert` for scales 0..4)
    are variants.
- **Invariant — variants destroyed before bundle**. Same rule as
  ADR-0106 entry 0106: `close_fex` must destroy
  `decimate_pipelines[1..3]` and `ssim_pipeline_horiz[1..4]` +
  `ssim_pipeline_vert[0..4]` *before* calling
  `vmaf_vulkan_kernel_pipeline_destroy()` on `pl_decimate` /
  `pl_ssim`.
- **Invariant — `[0]` aliasing destroy-skip**. `decimate_pipelines[0]`
  and `ssim_pipeline_horiz[0]` *must not* be passed to
  `vkDestroyPipeline` in `close_fex` — `_destroy()` already releases
  them via `pl_decimate.pipeline` / `pl_ssim.pipeline`. Double-free
  is UB. The destroy loops in `close_fex` start at `i = 1` for
  decimate and skip `i == 0` for ssim_horiz.
- **Invariant — per-bundle descriptor pool**. The shared
  `s->desc_pool` is gone; `alloc_descriptor_set` now takes a
  `const VmafVulkanKernelPipeline *bundle` and uses
  `bundle->desc_pool` + `bundle->dsl`. Per-frame
  `vkFreeDescriptorSets` calls must target the matching pool
  (`pl_decimate.desc_pool` for decimate sets, `pl_ssim.desc_pool`
  for ssim sets) — mixing them is undefined behavior.
- **Numerical contract**: unchanged. Same shaders, spec constants,
  push constants, and dispatch order as before. `float_ms_ssim`
  Netflix-pair smoke (576×324×48f) reports mean 0.963241; ssim
  pyramid intermediate values bit-identical to pre-migration run.
- **Rebase impact**: low. Upstream Netflix has no Vulkan backend.
  Conflicts only against the parallel `T-GPU-DEDUP-{18..22}` PRs
  (#284–#288) on `CHANGELOG.md` / `docs/rebase-notes.md` —
  auto-resolve keeps both halves.

### 0106 — Vulkan kernel template multi-pipeline + ssim/motion migration (T-GPU-DEDUP-7)

- **Touches**:
  - `libvmaf/src/vulkan/kernel_template.h` — new
    `vmaf_vulkan_kernel_pipeline_add_variant()` helper. Takes the
    base pipeline bundle (DSL / pipeline layout / shader / pool
    owned by `vmaf_vulkan_kernel_pipeline_create`) plus a partial
    `VkComputePipelineCreateInfo` and produces a sibling
    `VkPipeline` re-using the same layout / shader. The base
    `_create` and `_destroy` entry points are unchanged; existing
    consumers (psnr, moment, ciede) keep working.
  - `libvmaf/src/feature/vulkan/motion_vulkan.c` — state collapses
    `VkPipeline pipelines[2]` (kept "for SYCL parity" but
    functionally identical because COMPUTE_SAD goes through push
    constants, not spec-constants) to a single
    `VmafVulkanKernelPipeline pl`. `create_pipelines` /
    `close_fex` shrink to template-driven create + destroy.
  - `libvmaf/src/feature/vulkan/ssim_vulkan.c` — state becomes
    `VmafVulkanKernelPipeline pl + VkPipeline pipeline_vert`. Pass
    0 (horizontal) is the template's base pipeline; pass 1
    (vertical) is created via `_add_variant()`. `close_fex`
    destroys the variant first, then calls
    `vmaf_vulkan_kernel_pipeline_destroy()` on the bundle.
- **Invariant — no spec-constant drift between base and variant**.
  `_add_variant()` overwrites `sType` / `stage.sType` /
  `stage.stage` / `stage.module` / `layout` of the caller's
  `VkComputePipelineCreateInfo` so the variant is guaranteed to
  share the base's shader and layout. Callers control the
  variant's spec-constant via `pSpecializationInfo`. Reordering
  these overwrites lets a consumer accidentally bind a different
  shader module under the same layout — UB at descriptor-set time.
- **Invariant — variant destroyed before bundle**. `close_fex` in
  ssim must `vkDestroyPipeline(s->pipeline_vert)` before
  `vmaf_vulkan_kernel_pipeline_destroy(&s->pl)` — the bundle's
  `_destroy` releases the descriptor pool, which the
  `vkAllocateDescriptorSets` issued against the variant pipeline's
  layout cleanly drops only when the variant pipeline is already
  gone.
- **Numerical contract**: unchanged. Both kernels run identical
  shaders + spec-constants + push-constants as before; only the
  Vulkan boilerplate that creates / destroys the pipeline
  scaffolding moved to a shared owner. Cross-backend parity gate
  at `places=4` holds — Netflix-pair `float_ssim` smoke
  (576×324×48f) reports mean 0.863, identical to pre-migration.
- **Rebase impact**: low. The base pipeline-bundle helpers
  predate this change (PR #270 / #271); the new
  `_add_variant` is additive. Upstream Netflix has no Vulkan
  backend to conflict with.
### 0111 — integer_ciede_cuda migrated to kernel_template (T-GPU-DEDUP-11)

- **Touches**:
  - `libvmaf/src/feature/cuda/integer_ciede_cuda.c` — state's
    `CUstream + CUevent + CUevent + VmafCudaBuffer + host-pinned
    float*` quintet collapses to
    `VmafCudaKernelLifecycle lc + VmafCudaKernelReadback rb`.
    init / collect / close call the template's
    `lifecycle_init`/`readback_alloc`/`collect_wait`/
    `lifecycle_close`/`readback_free` helpers. submit keeps the
    pre-launch wait inline (intentional — ciede has no atomic, so
    the template's pre-launch memset is unnecessary).
- **Numerical contract**: unchanged. Pure CUDA-boilerplate
  consolidation. The host-side reduction in collect still uses
  the same `double` accumulator over per-block float partials —
  `places=4` (ADR-0187) holds.
### 0112 — integer_moment_cuda migrated to kernel_template (T-GPU-DEDUP-12)

- **Touches**:
  - `libvmaf/src/feature/cuda/integer_moment_cuda.c` — state's
    stream/event/device-buffer/host-pinned quintet collapses to
    `VmafCudaKernelLifecycle lc + VmafCudaKernelReadback rb`.
    submit calls `vmaf_cuda_kernel_submit_pre_launch` (atomic
    counters require the device-side memset). init / collect /
    close call the matching template helpers.
- **Numerical contract**: unchanged. Same per-frame atomic
  accumulators (4× uint64), same `sums_host[i] / n_pixels`
  host division.
- **Rebase impact**: low. Upstream Netflix has no equivalent
  template; this consolidation is fork-local.
### 0113 — integer_motion_v2_cuda migrated to kernel_template (T-GPU-DEDUP-13)

- **Touches**:
  - `libvmaf/src/feature/cuda/integer_motion_v2_cuda.c` —
    stream/event pair + sad device+host quintet collapses to
    `lc + rb`. Raw-pixel ping-pong `pix[2]` stays outside the
    bundle. submit keeps the memset on `pic_stream` inline
    rather than calling `submit_pre_launch` (the helper would
    move the memset to `lc.str`, which races with the kernel
    reading the accumulator). init / collect / close call the
    matching template helpers.
- **Numerical contract**: unchanged. Same D2D copy, same
  conditional kernel launch on frame ≥ 1, same
  host-side `min(score[i], score[i+1])` flush.
### 0114 — integer_ssim_cuda migrated to kernel_template (T-GPU-DEDUP-14)

- **Touches**:
  - `libvmaf/src/feature/cuda/integer_ssim_cuda.c` —
    stream/event/partials device+host quintet collapses to
    `lc + rb`. Five intermediate float buffers (`h_ref_mu`,
    `h_cmp_mu`, `h_ref_sq`, `h_cmp_sq`, `h_refcmp`) stay
    outside the bundle. submit keeps the
    `cuStreamWaitEvent + horiz + vert + DtoH` chain inline —
    SSIM writes one float per block (no atomic), so the
    template's `submit_pre_launch` memset is unnecessary. init /
    collect / close use the matching template helpers.
- **Numerical contract**: unchanged. Same horiz-then-vert
  two-pass pipeline, same per-block float partial reduction in
  double on the host. `places=4` (matching the ciede_cuda
  precision pattern) holds.
- **Rebase impact**: low. Upstream Netflix has no equivalent;
  this is fork-added.
### 0115 — ms_ssim_cuda + psnr_hvs_cuda lifecycle migration (T-GPU-DEDUP-15)

- **Touches**:
  - `libvmaf/src/feature/cuda/integer_ms_ssim_cuda.c` —
    stream + 2-event lifecycle replaced with
    `VmafCudaKernelLifecycle lc`; multi-level pyramid + SSIM
    intermediate + 3-partials buffers stay outside the template's
    single-pair readback bundle.
  - `libvmaf/src/feature/cuda/integer_psnr_hvs_cuda.c` — same
    shape; 3-plane ref/dist/partials triples remain inline.
- **Numerical contract**: unchanged. The migration only affects
  init / close boilerplate; submit / collect dispatch and host
  reduction paths are untouched apart from the
  `s->str` → `s->lc.str` / `s->event` → `s->lc.submit` /
  `s->finished` → `s->lc.finished` field renames.
### 0116 — float_psnr/ansnr/motion cuda → kernel_template (T-GPU-DEDUP-16)

- **Touches**:
  - `libvmaf/src/feature/cuda/float_psnr_cuda.c` —
    stream/event/partials quintet → `lc + rb`; input upload
    buffers `ref_in` / `dis_in` stay outside the bundle.
  - `libvmaf/src/feature/cuda/float_ansnr_cuda.c` — same shape;
    rb wraps the (sig, noise) interleaved partials.
  - `libvmaf/src/feature/cuda/float_motion_cuda.c` — same shape;
    rb wraps the SAD partials, `blur[2]` ping-pong stays outside.
- **Numerical contract**: unchanged. Same dispatch geometry, same
  reduction order. Cross-backend parity gate at the kernels'
  contracted precision (places=3 per ADR-0192) holds.
### 0117 — float_adm + float_vif cuda lifecycle migration (T-GPU-DEDUP-17)

- **Touches**:
  - `libvmaf/src/feature/cuda/float_adm_cuda.c` —
    stream + 2-event lifecycle replaced with
    `VmafCudaKernelLifecycle lc`; multi-stage DWT + CSF
    pipeline state stays outside the template's single-pair
    readback bundle.
  - `libvmaf/src/feature/cuda/float_vif_cuda.c` — same shape;
    4-level pyramid + per-scale (num, den) pairs remain inline.
- **Numerical contract**: unchanged. The migration only affects
  init / close stream-event boilerplate; submit / collect
  dispatch and host reduction paths are untouched apart from
  the field renames.
- **Rebase impact**: low. Upstream Netflix has no equivalent
  template; this is fork-added.
### 0107 — float_psnr_vulkan migrated to kernel_template (T-GPU-DEDUP-8)

- **Touches**:
  - `libvmaf/src/feature/vulkan/float_psnr_vulkan.c` — state's
    `dsl + pipeline_layout + shader + pipeline + desc_pool` quintet
    is collapsed into a single `VmafVulkanKernelPipeline pl`;
    `create_pipelines` and `close_fex` shrink to template-driven
    create + destroy. No shader changes, no spec-constant changes,
    no push-constant changes.
- **Numerical contract**: unchanged. The migration is a pure
  Vulkan-boilerplate consolidation. Cross-backend parity gate
  at `places=4` holds — Netflix-pair smoke reports `float_psnr`
  mean 30.755 dB, identical to pre-migration.
### 0109 — float_ansnr_vulkan + motion_v2_vulkan migrated to kernel_template (T-GPU-DEDUP-9)

- **Touches**:
  - `libvmaf/src/feature/vulkan/float_ansnr_vulkan.c` —
    single-pipeline state collapses to
    `VmafVulkanKernelPipeline pl`; `create_pipelines` and
    `close_fex` shrink to template-driven create + destroy.
  - `libvmaf/src/feature/vulkan/motion_v2_vulkan.c` — same shape.
- **Numerical contract**: unchanged. Pure Vulkan-boilerplate
  consolidation. Cross-backend parity gate at the kernel's
  contracted precision holds — Netflix-pair smoke reports
  `float_ansnr` mean 23.51 dB and `motion2_v2_score` mean 3.895,
  identical to pre-migration.
### 0110 — float_motion_vulkan migrated to kernel_template (T-GPU-DEDUP-10)

- **Touches**:
  - `libvmaf/src/feature/vulkan/float_motion_vulkan.c` —
    single-pipeline state collapses to
    `VmafVulkanKernelPipeline pl`; `create_pipelines` and
    `close_fex` shrink to template-driven create + destroy.
- **Numerical contract**: unchanged. Pure Vulkan-boilerplate
  consolidation. Netflix-pair smoke reports `motion` mean
  4.049 / `motion2` mean 3.894, identical to pre-migration.
- **Rebase impact**: low. Upstream Netflix has no Vulkan backend.
### 0108 — Bristol VI-Lab feasibility digest + BVI-CC ingest ADR (Draft)

- **Touches**:
  - `docs/research/0046-bristol-vi-lab-feasibility.md` (new) —
    nine-dataset survey + use-case fit + effort estimate.
  - `docs/adr/0241-bristol-bvi-cc-ingest.md` (new, Status: Draft)
    — proposal to ingest BVI-CC as the second tiny-AI corpus.
  - `docs/adr/README.md` — index row for ADR-0241.
  - `CHANGELOG.md` — Added entry.
- **Numerical contract**: not applicable (docs-only).
- **Rebase impact**: none. Pure research deliverables; upstream
  Netflix has no equivalent surface.

### 0094 — Vulkan VkImage import v2 async pending-fence (T7-29 part 4 / ADR-0251)

- **ADR**: [ADR-0251](adr/0251-vulkan-async-pending-fence.md);
  predecessor [ADR-0186](adr/0186-vulkan-image-import-impl.md).
- **Touches**:
  - `libvmaf/src/vulkan/import.c` — full rewrite of the
    submission path. Single-fence `submit_and_wait` becomes
    per-slot `submit_to_slot` + `drain_slot_fence`; the new
    `slot_alloc` / `slot_release` helpers materialise / tear
    down a ring slot (staging-pair + cmd buffer + fence).
    `vmaf_vulkan_import_image` indexes into the ring by
    `frame_index % ring_size`; `vmaf_vulkan_wait_compute`
    drains every outstanding fence.
    `vmaf_vulkan_state_build_pictures` waits the slot's fence
    before exposing the host pointer. Public-API signatures
    are unchanged.
  - `libvmaf/src/vulkan/vulkan_internal.h` — new `struct
    VmafVulkanImportSlot`; `VmafVulkanImportSlots` becomes a
    fixed-capacity `VmafVulkanImportSlot ring[VMAF_VULKAN_RING_MAX]`
    plus geometry + `ring_size`. Two new defines —
    `VMAF_VULKAN_RING_DEFAULT` (4) and `VMAF_VULKAN_RING_MAX`
    (8). `VmafVulkanState` gains `requested_ring_size`.
  - `libvmaf/src/vulkan/common.c` — `vmaf_vulkan_state_init` and
    `_state_init_external` set
    `requested_ring_size = VMAF_VULKAN_RING_DEFAULT`.
  - `libvmaf/test/test_vulkan_async_pending_fence.c` (new,
    contract smoke for the v1 → v2 swap).
  - `libvmaf/test/meson.build` — registers the new test under
    the existing `enable_vulkan` guard.
  - `libvmaf/src/vulkan/AGENTS.md` (new) — pins the three
    rebase-sensitive ring invariants.
  - `docs/adr/0251-vulkan-async-pending-fence.md` (new),
    `docs/research/0042-vulkan-async-pending-fence.md` (new),
    `docs/api/gpu.md`, `docs/backends/vulkan/overview.md`,
    `CHANGELOG.md`, `docs/rebase-notes.md`.
  - `ffmpeg-patches/0006-libvmaf-add-libvmaf-vulkan-filter.patch`
    — **unchanged**. The v2 ring is fully internal to
    `VmafVulkanState`; the public ABI stays byte-identical
    so the filter consumes the new path transparently.
- **Invariant 1 — fixed ring depth at first import**.
  `lazy_alloc_ring` is the only place that materialises the
  ring; once allocated the depth never changes for the
  lifetime of the `VmafVulkanState`. Any caller that needs
  a different depth has to free + re-init. The geometry
  pinning contract from v1 (ADR-0186) is preserved verbatim.
- **Invariant 2 — `vkResetFences` only after `VK_SUCCESS`
  from `vkWaitForFences`**. Sole reset path lives in
  `drain_slot_fence`; `fence_in_flight` flips back to 0 only
  after the wait succeeds. A `-EIO` from the wait propagates
  up without resetting (so a retry would correctly re-wait
  rather than silently move on).
- **Invariant 3 — `state_free` drains before destroying**.
  `vmaf_vulkan_import_slots_free` walks the ring and calls
  `drain_slot_fence` on every in-flight slot, then issues
  one `vkQueueWaitIdle` belt-and-braces (any feature kernel
  that submitted on the same queue may still be running).
  Reordering this triggers validation-layer
  "destroying in-use object" errors.
- **Numerical contract**: unchanged. Async submission only
  changes when the host can read the staging buffer, not
  which bytes the GPU writes. Cross-backend parity gate
  (`scripts/ci/cross_backend_parity_gate.py`, `places=4`) holds.
- **Memory delta**: staging arena scales `1 → ring_size`
  per direction. At default depth and 1080p 8-bit Y, the
  per-state host-visible footprint grows from ~4 MiB to
  ~16 MiB. Documented in
  [ADR-0251 §Consequences](adr/0251-vulkan-async-pending-fence.md#consequences).

### 0090 — `cambi_vulkan` extractor (T7-36 / ADR-0210)

- **ADR**: [ADR-0210](adr/0210-cambi-vulkan-integration.md);
  predecessor [ADR-0205](adr/0205-cambi-gpu-feasibility.md).
- **Touches**:
  - `libvmaf/src/feature/vulkan/cambi_vulkan.c` (replaces the
    spike scaffold's `init_stub`/`extract_stub`/`close_stub`
    triple with the full Vulkan-aware lifecycle).
  - `libvmaf/src/feature/vulkan/shaders/cambi_preprocess.comp`
    (new), `cambi_mask_dp.comp` (new — unified row-SAT / col-SAT /
    threshold-compare via `PASS=0/1/2` spec const).
  - `libvmaf/src/feature/cambi.c` — appends a small block of
    public trampolines (`vmaf_cambi_*`) at the bottom of the file
    that thinly wrap the file-static helpers. **No upstream
    function-static code is renamed or moved**; the entire
    upstream body of cambi.c above the trampolines stays
    byte-identical, which keeps Netflix sync straightforward.
  - `libvmaf/src/feature/cambi_internal.h` (new) — internal-only
    header exposing `vmaf_cambi_calculate_c_values`,
    `vmaf_cambi_get_spatial_mask`, etc., to the GPU twin.
  - `libvmaf/src/vulkan/meson.build` — registers the 5 cambi
    shaders in `vulkan_shader_sources[]` and `cambi_vulkan.c` in
    `vulkan_sources`.
  - `libvmaf/src/feature/feature_extractor.c` — adds the extern
    decl + registry entry for `vmaf_fex_cambi_vulkan` under
    `#if HAVE_VULKAN`.
  - `scripts/ci/cross_backend_vif_diff.py` — `cambi` row in
    `FEATURE_METRICS` so the cross-backend gate runs at
    `places=4` against the CPU baseline.
  - `docs/adr/0210-cambi-vulkan-integration.md`,
    `docs/research/0032-cambi-vulkan-integration.md`,
    `docs/backends/vulkan.md`, `CHANGELOG.md`.
- **Invariant 1 — bit-exactness by construction**. Every GPU
  phase is integer arithmetic (`uint16` derivative, `int32` SAT,
  `>` compare, stride-2 gather, 3-element `mode3` lookup). The
  readback into the host `VmafPicture` pair is byte-identical to
  what the CPU would have written; the host residual then runs
  the unmodified CPU `calculate_c_values` + spatial pooling on
  those buffers. Any rebase that introduces float arithmetic into
  one of these GPU phases — e.g., a future Netflix change to the
  derivative kernel that adds a bilinear interpolation step —
  will silently break `places=4` and must be caught at the
  cross-backend gate.
- **Invariant 2 — `cambi_internal.h` signatures must stay in
  lock-step with cambi.c's file-static helpers**. The Vulkan
  twin calls `vmaf_cambi_calculate_c_values`, which trampolines
  to the file-static `calculate_c_values`. Any signature change
  to the latter (extra parameters, type changes) must update the
  trampoline + header in the same PR or the GPU build breaks.
- **On upstream sync**: cambi.c's file-static helpers are
  sometimes renamed by upstream (e.g., `decimate` →
  `cambi_decimate` would happen during a Netflix tidy-up). When
  rebasing, search cambi.c's tail for the trampoline block — its
  five `static` calls (`get_spatial_mask`, `decimate`,
  `filter_mode`, `calculate_c_values`, `spatial_pooling`,
  `weight_scores_per_scale`, `get_pixels_in_window`,
  `increment_range`, `decrement_range`,
  `get_derivative_data_for_row`, `cambi_preprocessing`) need to
  match the upstream symbol names. Update the trampoline body if
  upstream renames; signatures should not need to change because
  the trampoline already takes the function-pointer-typedef form
  (`VmafRangeUpdater` etc.).
- **Re-test on rebase**:
  `python3 scripts/ci/cross_backend_vif_diff.py --backend vulkan
  --feature cambi --ref testdata/ref_576x324_48f.yuv --dist
  testdata/dis_576x324_48f.yuv --width 576 --height 324
  --pixel-format 420 --bitdepth 8 --frames 48`. Should emit
  `places=4 PASS` with `max_abs_diff = 0.0`. If it diverges,
  bisect the GPU phases by reading back individual buffers
  (`image_buf` / `mask_buf` / `deriv_buf`) and comparing against
  the CPU's in-place `pic` plane after the equivalent stage.

The pre-ADR-0108 fork-local PRs are summarised by workstream rather
than per-PR. Future PRs add entries individually.

### 0085 — Upstream `c70debb1` partial port (adm_csf + barten_csf tests)

- **No ADR.** Pure upstream cherry-pick per
  [ADR-0108](adr/0108-deep-dive-deliverables-rule.md) carve-out
  ("pure upstream syncs and `port-upstream-commit` PRs are exempt").
- **Upstream source**: `c70debb1` (Kyle Swanson, 2026-04-28):
  "libvmaf/test: port new adm/vif/speed tests". The audit row that
  flagged the gap is T-NEW-2 in the
  [2026-04-29 quarterly upstream-backlog re-audit (PR #205)](upstream-backlog-audit-2026-04-29.md).
- **Touches** (additive only):
  - `libvmaf/src/feature/adm_csf_tools.h` — new header (verbatim
    from upstream); declares the inline `adm_native_csf` helper
    (DLM-paper CSF) used by the new `test_adm_csf` unit.
  - `libvmaf/test/test_adm_csf.c` — new unit (verbatim from
    upstream); 2 `mu_assert` cases on `adm_native_csf(3, 3.0,
    1080, {0, 45})`.
  - `libvmaf/test/test_barten_csf.c` — new unit (verbatim from
    upstream); 23 `mu_assert` cases over `barten_rod_cone_sens`,
    `barten_mtf`, `barten_csf`, `linear_interpolate`,
    `barten_watson_blend_csf` (all symbols already on the fork).
  - `libvmaf/test/meson.build` — registers the two new
    executables + adds `test('test_adm_csf', ...)` and
    `test('test_barten_csf', ...)`.
  - `CHANGELOG.md` Unreleased § Changed.
- **Deliberate scope cuts** (the upstream commit's other halves
  are not portable verbatim):
  1. `test_vif_tools.c` — depends on upstream symbols
     `NUM_KERNELSCALES`, the 21-entry `valid_kernelscales` table,
     `vif_validate_kernelscale`, `vif_get_filter_size`,
     `vif_get_filter`, `speed_get_antialias_filter`, and a
     `[NUM_KERNELSCALES][5][65]` filter table that the fork's
     `vif_filter1d_table_s [11][4][65]` does not match. Per
     [Research-0024](research/0024-vif-upstream-divergence.md)
     Strategy E, the fork deliberately diverges from the
     upstream `vif` runtime-helper chain to preserve the
     ADR-0138 / 0139 / 0142 / 0143 SIMD bit-exactness contract.
     Porting this test requires porting the runtime helpers
     first.
  2. `test_speed_chroma.c` — `#include`s `feature/speed.c`
     directly; the fork has no SpEED extractor (`feature/speed.c`
     does not exist). Pairs with audit row T-NEW-1 (port the
     SpEED extractor wholesale, or absorb it into the tiny-AI
     speed metric).
- **Invariants** (rebase-relevant):
  1. The new `adm_csf_tools.h` header is wholly additive and
     does not conflict with the existing fork `adm_csf_s`
     non-inline helper in `adm_tools.h` (different signature,
     different translation units).
  2. The two new tests do not depend on Netflix golden YUVs —
     they evaluate the closed-form CSF math directly. No
     golden-data interaction.
- **On upstream sync**: a future port of the upstream `vif`
  runtime-helper chain (Research-0024 Strategy A reversal) or the
  SpEED extractor (T-NEW-1) unlocks the deferred halves of this
  commit. Until then, fork-side `test_vif_tools.c` /
  `test_speed_chroma.c` stay absent.
- **Re-test on rebase**:

  ```bash
  meson setup build-cpu libvmaf -Denable_cuda=false -Denable_sycl=false
  ninja -C build-cpu test_adm_csf test_barten_csf
  meson test -C build-cpu test_adm_csf test_barten_csf
  ```

### 0084 — Embedded MCP server scaffold (T5-2, ADR-0209)

- **ADR**: [ADR-0209](adr/0209-mcp-embedded-scaffold.md) (audit-first
  scaffold) on top of the [ADR-0128](adr/0128-embedded-mcp-in-libvmaf.md)
  governance + [Research-0005](research/0005-embedded-mcp-transport.md)
  design.
- **Upstream source**: fork-local. Netflix/vmaf has no embedded
  MCP server (and no plans to add one — the workflow is
  agent-tooling-specific, well outside upstream's library scope).
- **Touches**:
  - `libvmaf/include/libvmaf/libvmaf_mcp.h` — new public header.
  - `libvmaf/include/libvmaf/meson.build` — new
    `if get_option('enable_mcp')` install branch.
  - `libvmaf/src/mcp/` — new directory: `mcp.c` (stub TU) +
    `meson.build` (exposes `mcp_sources` + `mcp_defines`).
  - `libvmaf/src/meson.build` — new `is_mcp_enabled` guard +
    `subdir('mcp')` block; `mcp_sources` threaded into the
    `library('vmaf', ...)` source list alongside `dnn_sources`.
  - `libvmaf/test/meson.build` — new `if get_option('enable_mcp')`
    block wiring `test_mcp_smoke`.
  - `libvmaf/test/test_mcp_smoke.c` — new 12-sub-test smoke.
  - `libvmaf/meson_options.txt` — new `enable_mcp` umbrella +
    three sub-flags (all default `false`).
- **Invariant**: every public entry point in `libvmaf_mcp.h`
  (`vmaf_mcp_init` / `_start_sse` / `_start_uds` / `_start_stdio` /
  `_stop` / `_close`) returns `-ENOSYS` (or `-EINVAL` on bad
  arguments) until the T5-2b runtime PR lands. The smoke pins
  this contract — a runtime PR that flips a return code without
  flipping the smoke expectation regresses the gate.
- **On upstream sync**: zero interaction with upstream files.
  Wholly additive directory + boolean build flags. The
  `subdir('mcp')` insertion in `libvmaf/src/meson.build` lives
  next to the existing `subdir('dnn')` / Vulkan blocks; an
  upstream conflict in that area would be confined to those few
  lines and is mechanical to resolve.
- **Re-test on rebase**:

  ```bash
  meson setup build-cpu libvmaf -Denable_cuda=false -Denable_sycl=false -Denable_mcp=false
  ninja -C build-cpu && meson test -C build-cpu  # baseline still green

  meson setup --reconfigure build-cpu libvmaf -Denable_mcp=true \
              -Denable_mcp_sse=true -Denable_mcp_uds=true -Denable_mcp_stdio=true
  ninja -C build-cpu
  meson test -C build-cpu test_mcp_smoke  # 12/12 sub-tests pass
  ```

### 0065 — T7-37 Netflix bench rerun + `docs/benchmarks.md` `TBD` fill

- **No ADR.** Empirical fill of pre-existing `TBD` cells; no new
  decision. The bench *script* fixes that this rerun depends on
  shipped earlier under PR #169 (libvmaf/AGENTS.md backend-engagement
  foot-guns), PR #170 (`--backend cuda` actually engages CUDA), and
  PR #171 (`testdata/bench_all.sh` uses correct flags). Vulkan header
  install for SDK consumers is PR #175.
- **Touches** (additive only): `docs/benchmarks.md` (every `TBD`
  cell replaced with measured numbers; hardware-profile table updated
  to the `ryzen-4090-arc` host the rerun was performed on; "How to
  reproduce" section now documents fixture acquisition for the
  gitignored BBB 4K 200-frame pair). `CHANGELOG.md` Unreleased §
  Changed entry.
- **Invariants** (rebase-relevant): none. The numbers are tied to
  fork commit `41301496` and the `ryzen-4090-arc` profile; an
  upstream rebase that changes feature pipelines would invalidate
  the table but not break parsing.
- **On upstream sync**: zero interaction. Pure docs.
- **Re-test on rebase**: `bash testdata/bench_all.sh` (after a fresh
  fork build) — confirms the bench script still drives all four
  backends and that the per-row metrics-key counts (CPU=15, CUDA=12,
  SYCL/Vulkan=34) still distinguish them. If they collapse to one
  count, the new upstream broke a backend dispatcher silently.

### 0050 — `float_adm_cuda` + `float_adm_sycl` extractors (ADR-0202)

- **ADR**: [ADR-0202](adr/0202-float-adm-cuda-sycl.md)
- **Touches**:
  - `libvmaf/src/feature/cuda/float_adm/float_adm_score.cu` (new)
  - `libvmaf/src/feature/cuda/float_adm_cuda.{c,h}` (new)
  - `libvmaf/src/feature/sycl/float_adm_sycl.cpp` (new)
  - `libvmaf/src/meson.build` — three changes: (1) new
    `float_adm_score` entry in `cuda_cu_sources`, (2) new
    `cuda_cu_extra_flags` dict that threads `--fmad=false` +
    `-Xcompiler=-ffp-contract=off` into the `float_adm_score`
    fatbin only, (3) new SYCL source in `sycl_feature_sources`.
  - `libvmaf/src/feature/feature_extractor.c` (extern decls +
    list entries for `vmaf_fex_float_adm_cuda` /
    `vmaf_fex_float_adm_sycl` under `#if HAVE_CUDA` /
    `#if HAVE_SYCL`).
- **Invariant 1 — `--fmad=false` for the float_adm fatbin only**:
  the angle-flag dot product
  (`ot_dp = oh*th + ov*tv`) and the cube reductions
  (`xa*xa*xa`, `csf_o*csf_o*csf_o`) require IEEE-754 add/mul
  ordering to match the GLSL `precise` qualifier in
  `float_adm.comp`. NVCC's default `-fmad=true` fuses these and
  drifts past `places=4` at scale 3 / adm2. The integer ADM
  kernels share `cuda_flags` but use `int64` accumulators where
  FMA is irrelevant — keep the FMA-on default for them.
- **Invariant 2 — parent-LL dimension trap**: stage 0 at
  `scale > 0` reads the parent's LL band; the mirror/clamp
  bounds are `scale_w/h[scale]` (= parent's LL output dims =
  current scale's input dims), NOT `scale_w/h[scale - 1]`
  (= parent's full image dims). Both `float_adm_cuda.c` and
  `float_adm_sycl.cpp` cite this inline. Do not "simplify" by
  using the off-by-one neighbour.
- **Re-test**:

  ```bash
  CXX=icpx CC=icx meson setup build-cs -Denable_cuda=true \
       -Denable_sycl=true -Denable_vulkan=enabled \
       -Denable_float=true \
       -Dsycl_compiler=/opt/intel/oneapi/compiler/latest/bin/icpx
  ninja -C build-cs
  python3 scripts/ci/cross_backend_vif_diff.py \
    --vmaf-binary build-cs/tools/vmaf \
    --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
    --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
    --width 576 --height 324 --feature float_adm \
    --backend cuda --places 4
  # Same with --backend sycl on a host with an SYCL device.
  # Both must report 0/N mismatches at places=4.
  ```

### 0049 — `float_adm_vulkan` extractor (ADR-0199)

- **ADR**: [ADR-0199](adr/0199-float-adm-vulkan.md)
- **Touches**:
  - `libvmaf/src/feature/vulkan/float_adm_vulkan.c` (new)
  - `libvmaf/src/feature/vulkan/shaders/float_adm.comp` (new)
  - `libvmaf/src/vulkan/meson.build` (adds the .comp shader and
    the new .c source)
  - `libvmaf/src/feature/feature_extractor.c` (extern decl + list
    entry under `#if HAVE_VULKAN`)
  - `scripts/ci/cross_backend_vif_diff.py` (`float_adm` entry in
    `FEATURE_METRICS`)
  - `.github/workflows/tests-and-quality-gates.yml` (lavapipe
    `float_adm` step at `places=4`)
- **Invariant**: float_adm GPU port uses the `2 * sup - idx - 1`
  mirror form on both axes — matches both the scalar `adm_dwt2_s`
  and the AVX2 `float_adm_dwt2_avx2`, which both consume the same
  `dwt2_src_indices_filt_s` index buffer. **This is intentionally
  different from float_vif's GPU mirror (ADR-0197), which uses
  `-2` because float_vif's AVX2 path takes a different code branch.**
  Do not "fix" the asymmetry by analogy with float_vif.
- **Re-test**:

  ```bash
  meson setup build-vk -Denable_vulkan=enabled -Denable_cuda=false \
                       -Denable_sycl=false
  ninja -C build-vk
  meson test -C build-vk
  VK_LOADER_DRIVERS_SELECT='*lvp*' python3 \
    scripts/ci/cross_backend_vif_diff.py \
    --vmaf-binary build-vk/tools/vmaf \
    --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
    --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
    --width 576 --height 324 --feature float_adm --places 4
  ```
### 0083 — SSIMULACRA 2 Vulkan kernel (ADR-0201)

- **ADR**: [ADR-0201](adr/0201-ssimulacra2-vulkan-kernel.md)
- **Upstream source**: fork-local. No SSIMULACRA 2 extractor in
  upstream Netflix/vmaf — fully fork-local feature.
- **Touches**:
  - [`libvmaf/src/feature/vulkan/ssimulacra2_vulkan.c`](../libvmaf/src/feature/vulkan/ssimulacra2_vulkan.c)
    (new file).
  - [`libvmaf/src/feature/vulkan/shaders/ssimulacra2_xyb.comp`](../libvmaf/src/feature/vulkan/shaders/ssimulacra2_xyb.comp),
    `ssimulacra2_blur.comp`, `ssimulacra2_mul.comp`,
    `ssimulacra2_ssim.comp` (4 new shader files).
  - [`libvmaf/src/vulkan/meson.build`](../libvmaf/src/vulkan/meson.build)
    — added 4 shaders to `vulkan_shader_sources` and 1 source to
    `vulkan_sources`; added all 4 ssimulacra2 shaders to
    `psnr_hvs_strict_shaders` (the `-O0` strict-mode list, kept its
    legacy name).
  - [`libvmaf/src/feature/feature_extractor.c`](../libvmaf/src/feature/feature_extractor.c)
    — registered `vmaf_fex_ssimulacra2_vulkan` in the Vulkan branch
    of the extractor list (between `psnr_hvs_vulkan` and the CUDA
    block).
  - [`scripts/ci/cross_backend_vif_diff.py`](../scripts/ci/cross_backend_vif_diff.py)
    — added `ssimulacra2` to `FEATURE_METRICS`.
- **Rebase impact**: low — fully additive, no upstream-shared
  files modified beyond `feature_extractor.c`'s registry array
  (which always grows on every new extractor and is not a rebase
  pain point).
- **Verification command**:

  ```bash
  meson setup libvmaf/build-vk-ss2 \
    -Denable_vulkan=enabled -Denable_cuda=false -Denable_sycl=false \
    libvmaf
  ninja -C libvmaf/build-vk-ss2 tools/vmaf
  python3 scripts/ci/cross_backend_vif_diff.py \
    --vmaf-binary libvmaf/build-vk-ss2/tools/vmaf \
    --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
    --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
    --width 576 --height 324 \
    --feature ssimulacra2 --backend vulkan --places 1
  # expected: max_abs_diff ≈ 1.59e-2, 0/48 mismatches at places=1
  ```
- **Follow-ups**:
  - CUDA + SYCL twins (batch 3 parts 7b + 7c per ADR-0192).
  - Performance follow-up: re-bin multiple rows / columns per WG
    in the IIR blur (currently `local_size = 1`, one row/col per WG
    for correctness).
  - Optional: rename `psnr_hvs_strict_shaders` to `strict_shaders`
    in `libvmaf/src/vulkan/meson.build` (cosmetic — out of scope
    for this PR).

### 0001 — SIMD bit-identical reductions for float ADM

- **Workstream PRs**: #18, commits `24c88a32`, `f082cfd3`.
- **Touches**:
  `libvmaf/src/feature/integer_adm.c`,
  `libvmaf/src/feature/float_adm.c`,
  `libvmaf/src/feature/x86/adm_avx2.c`,
  `libvmaf/src/feature/x86/adm_avx512.c`,
  `libvmaf/src/feature/arm64/adm_neon.c`,
  upstream `python/test/feature_extractor_test.py` test expectations.
- **Invariant**: `sum_cube` and `csf_den_scale` accumulate cubed
  values in **double precision** (via `_mm256_cvtps_pd` /
  `_mm512_cvtps_pd`) in scalar, AVX2, AVX-512, and NEON. Upstream
  accumulates in float, which produces ~8e-5 drift between scalar and
  SIMD. Test expectations were tightened to match the double-precision
  path; an upstream-side accumulator change would re-introduce the
  drift and break the tightened assertions.
- **Re-test**: `meson test -C build --suite=fast && python -m pytest
  python/test/feature_extractor_test.py -k adm`.

### 0002 — CUDA ADM decouple-inline buffer elimination

- **Workstream PRs**: commit `787e3382`.
- **Touches**:
  `libvmaf/src/feature/cuda/integer_adm_cuda.cu`,
  `libvmaf/src/feature/cuda/adm_decouple_inline.cuh` (new),
  `libvmaf/src/feature/cuda/meson.build`. Upstream's
  `adm_decouple.cu` is no longer compiled in the fork.
- **Invariant**: CSF and CM CUDA kernels read `ref` / `dis` DWT2
  buffers directly and compute `decouple_r` / `decouple_a` inline via
  `__device__` helpers in `adm_decouple_inline.cuh`. The 6 intermediate
  buffers (`decouple_r`, `decouple_a`, `csf_a` × {scale-0 int16,
  scales 1-3 int32}) and the standalone `adm_decouple.cu` source are
  intentionally removed. ~107 MB GPU memory savings at 4K. An upstream
  change to `adm_decouple.cu` will look orphaned and a literal merge
  would re-introduce the buffer allocations.
- **Re-test**: `meson setup build -Denable_cuda=true && ninja -C build
  && meson test -C build --suite=cuda`.

### 0003 — SYCL backend (USM pool / D3D11 import / `vmaf_sycl_*` API)

- **Workstream PRs**: #33, #35, #5 (initial scaffolding), and the
  picture-pool deadlock fix that landed via #32.
- **Touches**:
  `libvmaf/include/libvmaf/libvmaf_sycl.h`,
  `libvmaf/src/sycl/`,
  `libvmaf/src/feature/sycl/`,
  `libvmaf/src/libvmaf.c` (SYCL public-API entry points),
  `meson_options.txt` (`enable_sycl`).
- **Invariant**: `vmaf_sycl_preallocate_pictures` constructs a real
  `VmafSyclPicturePool` honoring `VmafSyclPicturePreallocationMethod`
  (`NONE` / `DEVICE` / `HOST`); `vmaf_sycl_picture_fetch` dispatches
  to the pool when configured. The whole SYCL tree is fork-local and
  has no upstream counterpart — upstream changes to
  `libvmaf/src/libvmaf.c` near the SYCL entry-point block are likely
  to conflict. Picture-pool error paths in `vmaf_read_pictures`
  (libvmaf.c) must `goto cleanup;` rather than `return err;` to avoid
  leaking ref/dist pictures into the live-picture set (closes the
  always-on-pool deadlock fixed in #32 — see ADR-0104). See
  [ADR-0101](adr/0101-sycl-usm-picture-pool.md),
  [ADR-0103](adr/0103-sycl-d3d11-surface-import.md),
  [ADR-0104](adr/0104-picture-pool-always-on.md).
- **Re-test**: `meson setup build -Denable_sycl=true && ninja -C build
  && meson test -C build --suite=sycl` (requires oneAPI / icpx).

### 0004 — DNN runtime + tiny-AI surfaces

- **Workstream PRs**: #5, #8, #21, #22, #23, #31, #34, plus the
  pre-numbered DNN feat commits (`9b985946`, `1e5336d3`, `d122b721`).
- **Touches**:
  `libvmaf/include/libvmaf/dnn.h`,
  `libvmaf/src/dnn/`,
  `libvmaf/src/feature/feature_lpips.c`,
  `model/tiny/`,
  `meson_options.txt` (`enable_onnxruntime`).
- **Invariant**: ordered EP selection (CUDA → DML → CPU) with graceful
  fallback (ADR-0102); `fp16_io` does host-side fp32↔fp16 cast on the
  scoring path; `VMAF_TINY_MODEL_DIR` enforces a path jail on model
  load (PR #31); the runtime op-allowlist (PR #21) walks the ONNX
  graph and rejects unknown ops + bounds Loop/If `trip_count` at 1024
  (ADR-0036/0107). DNN tree is fork-local; upstream has no DNN code
  yet, so conflicts here are unlikely but the
  `meson_options.txt` and `libvmaf/src/meson.build` blocks
  near the DNN flag may collide.
- **Re-test**: `meson setup build -Denable_onnxruntime=true && ninja
  -C build && meson test -C build --suite=dnn`.

### 0005 — `--precision` CLI flag (IEEE-754 round-trip lossless)

- **Workstream PRs**: commit `c989fbd9`.
- **Touches**:
  `libvmaf/tools/vmaf.c`,
  `libvmaf/tools/cli_parse.c`,
  `libvmaf/include/libvmaf/libvmaf.h` (added
  `vmaf_write_output_with_format`),
  `libvmaf/src/output.c`.
- **Invariant**: default `--precision` is `%.17g` (round-trip
  lossless); `legacy` opts back into upstream's `%.6f`; the public C
  API gained `vmaf_write_output_with_format` and the old
  `vmaf_write_output` routes through it with the `%.17g` default.
  ABI-breaking only if upstream adds a same-named function with a
  different signature. See [ADR-0006](adr/0006-cli-precision-17g-default.md).
- **Re-test**: `vmaf -r ref.yuv -d dis.yuv ... --precision=full` and
  diff against `--precision=legacy`.

### 0006 — Netflix golden tests preserved verbatim as required gate

- **Workstream PRs**: across the fork's life; codified in
  [ADR-0024](adr/0024-netflix-golden-preserved.md).
- **Touches**: `python/test/quality_runner_test.py`,
  `python/test/vmafexec_test.py`,
  `python/test/vmafexec_feature_extractor_test.py`,
  `python/test/feature_extractor_test.py`,
  `python/test/result_test.py`,
  `python/test/resource/yuv/`.
- **Invariant**: `assertAlmostEqual(...)` golden values in the five
  upstream Python test files are **never modified by this fork**.
  Fork-added tests live in separate files (e.g.
  `python/test/test_precision_flag.py`). The CI gate "Netflix CPU
  golden tests (D24)" is required and blocks merge. Upstream changes
  to these files are accepted unless they relax the assertions.
- **Re-test**: `make test-netflix-golden`.

### 0007 — Build system (CUDA 13.2, oneAPI 2025.3, MkDocs migration)

- **Workstream PRs**: #7, #17, commit `8a995cb0`.
- **Touches**: `meson.build`, `meson_options.txt`, top-level
  `Makefile`, `docs/` (Sphinx → MkDocs Material migration —
  `docs/conf.py` removed, `mkdocs.yml` added),
  `docs/requirements.txt`,
  `Dockerfile.*`, distro install scripts under `scripts/`.
- **Invariant**: image pins are non-conservative
  ([ADR-0027](adr/0027-non-conservative-image-pins.md)) — CUDA 13.2,
  oneAPI 2025.3, clang-format 22, black 26 — and ship experimental
  toolchain flags (`--expt-relaxed-constexpr`, etc.) deliberately. An
  upstream sync that pulls in a Dockerfile change targeted at older
  CUDA or older oneAPI must not relax the pins.
- **Re-test**: `meson setup build -Denable_cuda=true
  -Denable_sycl=true && ninja -C build && mkdocs build --strict`.

### 0008 — Workspace / docs / MATLAB / resource-tree relocations

- **Workstream PRs**: codified across [ADR-0026](adr/0026-workspace-relocated-under-python.md),
  [ADR-0029](adr/0029-resource-tree-relocated.md),
  [ADR-0030](adr/0030-matlab-sources-relocated.md),
  [ADR-0031](adr/0031-fork-docs-moved-under-docs.md),
  [ADR-0032](adr/0032-unittest-script-moved-to-scripts.md),
  [ADR-0033](adr/0033-codeql-config-moved-to-github.md),
  [ADR-0034](adr/0034-single-patches-directory.md),
  [ADR-0038](adr/0038-purge-upstream-matlab-mex-binaries.md).
- **Touches**: any path-walk in upstream's CI / scripts / docs that
  assumes the upstream layout (root-level `workspace/`,
  `resource/`, `matlab/`, root `unittest` script, root `patches/`).
- **Invariant**: the fork's layout is `python/vmaf/workspace/`,
  `python/vmaf/resource/`, `python/vmaf/matlab/`, `scripts/unittest`,
  `ffmpeg-patches/` only, `.github/codeql-config.yml`. Upstream moves
  to a different sub-tree (e.g. a hypothetical
  `tools/workspace/`) need to either be applied via a corresponding
  fork-side relocation or rejected with a rebase note.
- **Re-test**: `python -m pytest python/test/ -k golden` (verifies the
  resource-tree path works); `make test-netflix-golden`.

### 0009 — License headers (Lusoris/Claude on wholly-new files

2016–2026 on Netflix files)

- **Workstream PRs**: commits `c159761d`, `a185f8ef`, `0e98c949`, codified
  in [ADR-0025](adr/0025-copyright-handling-dual-notice.md) /
  [ADR-0105](adr/0105-copyright-handling-dual-notice.md).
- **Touches**: every wholly-new fork file (notably the SYCL tree and
  `libvmaf/src/dnn/`) and every Netflix-touched file (year range
  `2016 → 2016–2026`).
- **Invariant**: wholly-new fork files carry
  `Copyright 2026 Lusoris and Claude (Anthropic)` under the same
  BSD-3-Clause-Plus-Patent license; mixed files use a dual-copyright
  notice. An upstream commit that resets a Netflix file's year range
  (e.g. back to `2016–2020`) must be partially rejected — keep the
  fork's `2016–2026`.
- **Re-test**: grep that wholly-new fork files retain the Lusoris/Claude
  header (`grep -L "Copyright 2026 Lusoris" libvmaf/src/sycl/*.cpp` —
  expected to match nothing).

### 0010 — `.claude/` agent scaffolding + ADR tree + AGENTS.md / CLAUDE.md

- **Workstream PRs**: #14, #24, #37, plus continuous additions.
- **Touches**: `.claude/`, `AGENTS.md`, `CLAUDE.md`, `docs/adr/`,
  `.github/PULL_REQUEST_TEMPLATE.md`.
- **Invariant**: this whole tree is fork-local and has no upstream
  counterpart. Upstream additions to `.github/` (issue templates,
  workflows) need to merge cleanly with the fork's existing files
  rather than replacing them. The ADR tree's IDs ≤ 0099 are
  *backfills*; new decisions start at 0100
  ([ADR-0028](adr/0028-adr-maintenance-rule.md) /
  [ADR-0106](adr/0106-adr-maintenance-rule.md)).
- **Re-test**: visual review of `.github/` and `docs/adr/README.md`
  after the merge.

---

*Pre-ADR-0108 entries above are the result of a one-shot backfill
sweep on 2026-04-18; subsequent fork-local PRs add their own entries
inline.*

### 0011 — Nightly bisect-model-quality + fixture cache

- **Workstream PRs**: closes #4; sticky tracker issue #40.
- **Touches**:
  `.github/workflows/nightly-bisect.yml`,
  `ai/scripts/build_bisect_cache.py`,
  `ai/testdata/bisect/{features.parquet, models/*.onnx, README.md}`,
  `scripts/ci/post-bisect-comment.py`,
  `docs/ai/bisect-model-quality.md`,
  `docs/adr/0109-nightly-bisect-model-quality.md`,
  `docs/research/0001-bisect-model-quality-cache.md`,
  `mkdocs.yml` (nav).
- **Invariant**: the committed parquet + ONNX bytes under
  `ai/testdata/bisect/` must regenerate **byte-identically** from
  `ai/scripts/build_bisect_cache.py` with seeds `FEATURE_SEED=20260418`
  and `MODEL_SEED=20260419`. The CI `--check` step asserts this before
  every bisect run, so any upstream pull that bumps `pandas` /
  `pyarrow` / `onnx` enough to change the serialiser bytes will fail
  the workflow until the cache is regenerated and committed.
- **Re-test**:

  ```bash
  python ai/scripts/build_bisect_cache.py --check
  vmaf-train bisect-model-quality \
      ai/testdata/bisect/models/model_*.onnx \
      --features ai/testdata/bisect/features.parquet \
      --min-plcc 0.85 --input-name input
  # Expected: "no regression in this range"; first_bad_index None.
  ```

  Pure upstream code is not touched, so no Netflix-side conflict
  vector. Only fork-local files; risk is toolchain drift, not merge
  conflict.

### 0012 — Upstream ADM port (Netflix `966be8d5`)

- **Workstream PRs**: this PR; ports a single upstream commit.
- **Touches**:
  `libvmaf/src/feature/integer_adm.{c,h}`,
  `libvmaf/src/feature/x86/adm_avx2.{c,h}`,
  `libvmaf/src/feature/x86/adm_avx512.{c,h}`,
  `libvmaf/src/feature/alias.c`,
  `libvmaf/src/feature/barten_csf_tools.h` (new upstream file).
- **Invariant**: the eight ADM files now mirror upstream's content
  byte-for-byte (modulo our clang-format-22 pass and the Netflix
  copyright-year bump on the new header). Future `/sync-upstream`
  runs can take new upstream ADM commits cleanly. **Do not** revert
  to a pre-`966be8d5` ADM kernel without also reverting the call-site
  signatures in `integer_compute_adm` — upstream extended
  `i4_adm_cm` from 8 to 13 args.
- **Re-test**:

  ```bash
  ninja -C libvmaf/build && meson test -C libvmaf/build
  libvmaf/build/tools/vmaf -r python/test/resource/yuv/src01_hrc00_576x324.yuv \
      -d python/test/resource/yuv/src01_hrc01_576x324.yuv \
      -w 576 -h 324 -p 420 -b 8 \
      --model version=vmaf_v0.6.1 -o /tmp/vmaf-port.json
  grep '<metric name="vmaf"' /tmp/vmaf-port.json
  # Expected: mean ≈ 76.66890 (golden 76.66890519623612, places=4 OK).
  ```

### 0013 — Upstream motion port (Netflix PR #1486 head `2aab9ef1`)

- **Workstream PRs**: this PR; ports upstream PR #1486 (4 commits on top
  of `966be8d5` ADM base, head `2aab9ef1`). Sister to entry 0012.
- **Touches**:
  `libvmaf/src/feature/integer_motion.{c,h}`,
  `libvmaf/src/feature/motion_blend_tools.h` (new upstream file),
  `libvmaf/src/feature/x86/motion_avx2.c`,
  `libvmaf/src/feature/x86/motion_avx512.c`,
  `libvmaf/src/feature/alias.c` (additive: `integer_motion3` row),
  `python/test/{quality_runner,vmafexec,feature_extractor,vmafexec_feature_extractor}_test.py`
  (golden tolerance updates: `places=4` → `places=2` on motion-affected
  asserts; expected values unchanged).
- **Invariant**: motion files mirror upstream byte-for-byte (modulo our
  clang-format-22 pass). The `alias.c` row for `integer_motion3` was
  inserted surgically to avoid clobbering the AVX-512 ADM registration
  added by entry 0012; new motion3 metric appears in default VMAF model
  output but is not standalone-loadable via `--feature integer_motion3`
  (sub-feature only). Netflix golden VMAF mean shifts
  `76.668904824` → `76.667830213` (well within `places=2` tolerance the
  upstream PR loosened to). **Do not** revert `places=4` on
  motion-touching assertions without also reverting the motion code.
- **Re-test**:

  ```bash
  ninja -C libvmaf/build && meson test -C libvmaf/build
  libvmaf/build/tools/vmaf -r python/test/resource/yuv/src01_hrc00_576x324.yuv \
      -d python/test/resource/yuv/src01_hrc01_576x324.yuv \
      -w 576 -h 324 -p 420 -b 8 \
      --model version=vmaf_v0.6.1 -o /tmp/vmaf-motion-port.json
  grep -E '<metric name="vmaf"|integer_motion3' /tmp/vmaf-motion-port.json
  # Expected: vmaf mean ≈ 76.66783; integer_motion3 mean ≈ 3.98976.
  ```

### 0014 — Coverage gate overhaul + upstream `python/test/` reformat

- **Workstream PRs**: this PR (coverage-gate overhaul + in-tree reformat
  of upstream-mirror Python tests).
- **Touches**:
  `.github/workflows/ci.yml` (CPU + GPU coverage jobs:
  `-Dc_args=-fprofile-update=atomic` / `-Dcpp_args=-fprofile-update=atomic`,
  `meson test --num-processes 1`, `-Denable_dnn=enabled`, ORT install
  step on the CPU coverage job, `lcov`/`geninfo` replaced by `gcovr`
  with `--json-summary` / `--xml` / `--txt` output, artifact rename
  `coverage-lcov-{cpu,gpu}` → `coverage-{cpu,gpu}`),
  `scripts/ci/coverage-check.sh` (rewritten to parse gcovr JSON via
  `python3 -c` — same CLI signature),
  `libvmaf/src/dnn/dnn_api.c` + new `libvmaf/src/dnn/dnn_attach_api.c`
  (`vmaf_use_tiny_model` carved out into its own TU so the unit-test
  binaries — which pull in `dnn_sources` for `feature_lpips.c` but
  never link `libvmaf.c` — don't end up with an undefined reference
  to `vmaf_ctx_dnn_attach` once `enable_dnn=enabled` activates the
  real bodies),
  `libvmaf/src/dnn/meson.build` + `libvmaf/src/meson.build`
  (new `dnn_libvmaf_only_sources` list wired into `libvmaf.so` only),
  `python/test/{feature_extractor,quality_runner,vmafexec,vmafexec_feature_extractor}_test.py`
  (mechanical Black + isort reformat — no assertion values changed,
  imports regrouped, line wrapping normalised).
- **Invariant**: coverage CI must keep all five pieces in lockstep —
  (a) `-fprofile-update=atomic` closes the intra-process counter race
  on SIMD inner loops (`vif_avx2.c:673`, `motion_avx2`, etc.) →
  negative counts → `geninfo`/gcovr abort; (b) `--num-processes 1`
  closes the inter-process race where multiple parallel test binaries
  merge their counters into the same `.gcda` files for the shared
  `libvmaf.so` at process exit (per-thread atomicity does not cover
  this); (c) `gcovr` deduplicates `.gcno` files belonging to the
  same source compiled into multiple targets — without dedup, lcov
  sums hits across compilation units and yields impossible
  >100% values (`dnn_api.c — 1176%` was the smoking gun on the first
  attempt that had only (a)+(b)); (d) ORT install + `enable_dnn=enabled`
  in the coverage job is what makes `libvmaf/src/dnn/*.c` measurable
  in the first place — without ORT, the DNN tree compiles in stub
  branches and the 85% per-critical-file gate is meaningless;
  (e) `vmaf_use_tiny_model` lives in `dnn_attach_api.c` and is added
  to `libvmaf.so` only via `dnn_libvmaf_only_sources` — moving it
  back into `dnn_api.c` reintroduces the `vmaf_ctx_dnn_attach`
  undefined-reference link error in `test_feature_extractor` /
  `test_lpips` whenever `enable_dnn=enabled`, since those test
  binaries pull in `dnn_sources` for `feature_lpips.c` but never
  link `libvmaf.c`. Lint
  scope: upstream-mirror Python tests are linted at the same standard
  as fork-added code; we accept that `/sync-upstream` and
  `/port-upstream-commit` will re-trigger Black/isort failures
  whenever upstream rewrites these files, and the fix is another
  in-tree reformat pass — never an exclusion. The fork's
  `pyproject.toml` and `.pre-commit-config.yaml` keep
  `python/test/resource/` (binary fixtures only) excluded;
  `python/test/*.py` is in scope. See
  [ADR-0110](adr/0110-coverage-gate-fprofile-update-atomic.md) (race
  fixes, superseded) and
  [ADR-0111](adr/0111-coverage-gate-gcovr-with-ort.md) (gcovr + ORT
  layer).
- **Re-test**:

  ```bash
  # Reproduce coverage path locally (requires gcc + python3-pip):
  pip install --user 'gcovr>=8.0'
  cd libvmaf
  meson setup build-cov-test --buildtype=debug -Db_coverage=true \
      -Denable_avx512=true -Denable_float=true -Denable_dnn=disabled \
      -Dc_args=-fprofile-update=atomic -Dcpp_args=-fprofile-update=atomic
  ninja -C build-cov-test
  meson test -C build-cov-test --print-errorlogs --num-processes 1
  ~/.local/bin/gcovr --root .. \
      --filter 'src/.*' \
      --exclude '.*/test/.*' --exclude '.*/tests/.*' \
      --exclude '.*/subprojects/.*' \
      --gcov-ignore-parse-errors=negative_hits.warn \
      --gcov-ignore-parse-errors=suspicious_hits.warn \
      --print-summary --txt build-cov-test/coverage.txt \
      --json-summary build-cov-test/coverage.json \
      build-cov-test
  grep -E 'dnn_api|model_loader' build-cov-test/coverage.txt
  # Expected: gcovr completes without "Unexpected negative count" AND no
  # per-file percentages exceed 100% (drop --num-processes 1 to reproduce
  # the multi-process .gcda merge race; switch back to lcov to reproduce
  # the dnn_api.c — 1176% over-count from compilation-unit summation).

  # Lint smoke test for upstream-mirror tree:
  pre-commit run --files python/test/quality_runner_test.py
  # Expected: Black/isort/Ruff all PASS — files are reformatted in-tree
  # to fork style and stay clean until the next upstream sync.
  ```

### 0015 — Tox doctest collection skips `vmaf/resource/`

- **Workstream PRs**: this PR (`fix(ci): skip pytest doctest collection
  of vmaf/resource/ data files`). Surfaced once ADR-0115 consolidated
  CI triggers to `master` and tox actually started running on PRs.
- **Touches**: `python/tox.ini` (single-line `--ignore=vmaf/resource`
  added to the pytest invocation, plus an explanatory comment block).
  Pure fork-local; no upstream Python file changes.
- **Invariant**: `pytest --doctest-modules` must not attempt to import
  files under `python/vmaf/resource/`. Those are parameter / dataset /
  example-config `.py` files; several have dots in their stems (e.g.
  `vmaf_v7.2_bootstrap.py`) that make them unimportable as Python
  modules. None carry doctests, so the ignore is correctness rather
  than a workaround. **Do not** drop the `--ignore=vmaf/resource`
  flag without first verifying every file under that directory has
  been renamed to a dot-free stem and is importable.
- **Re-test**:

  ```bash
  cd python && tox -e py311 -- --collect-only --doctest-modules \
      --ignore=vmaf/resource 2>&1 | grep -c "ERROR collecting vmaf/resource"
  # Expected: 0 (was 5 before the fix).
  ```

  Pure upstream code is not touched, so no Netflix-side conflict
  vector. Risk is upstream renaming or removing files under
  `python/vmaf/resource/` such that the directory disappears, in
  which case the `--ignore` becomes a harmless no-op.

### 0016 — SYCL `-fsycl` link-arg gated on icpx CXX

- **Workstream PRs**: this PR (`fix(libvmaf): gate -fsycl link arg on
  icpx CXX, allow gcc/clang host linker`). Surfaced once
  ADR-0115's CI consolidation added an Ubuntu SYCL job to PR-time CI
  that uses `CXX=g++` (host linker) with sidecar icpx for SYCL .cpp
  compilation.
- **Touches**: `libvmaf/src/meson.build` (the `vmaf_link_args` block
  immediately after the `is_sycl_enabled` flag handling — currently
  ~lines 696-712). Pure fork-local; no upstream Meson file changes
  expected.
- **Invariant**: `-fsycl` is appended to `vmaf_link_args` **only** when
  `meson.get_compiler('cpp').get_id() == 'intel-llvm'` (icpx).
  Rationale: the documented project mode (see comment near `is_sycl_enabled`
  block at top of `src/meson.build`) compiles SYCL `.cpp` files via
  `custom_target` with icpx, while the project's CXX driver may be gcc /
  clang / msvc; in that mode the SPIR-V device code is already embedded
  in the icpx-compiled `.o` files at compile time, and the runtime
  libraries (`libsycl` + `libsvml` + `libirc` + `libze_loader`) declared
  as link dependencies resolve every symbol. Passing `-fsycl` to a
  non-icpx linker is a hard error
  (`g++: error: unrecognized command-line option '-fsycl'`). **Do not**
  remove the `cpp.get_id() == 'intel-llvm'` guard without first verifying
  every CI matrix leg uses icpx as the project CXX.
- **Re-test**:

  ```bash
  meson setup build -Denable_sycl=true \
      -Dcpp_link_args=-Wl,--no-undefined
  ninja -C build src/libvmaf.so.3
  # Expected: link succeeds; no `-fsycl` errors with gcc/clang host CXX.
  ```

  Pure fork-local guard; no Netflix-side conflict vector.

### 0017 — CLI precision default `%.6f` (Netflix-compat) + frame-skip unref

- **Workstream PRs**: this PR (`fix(cli): revert precision default to
  %.6f and unref skipped frames`). Reverts the default flipped by
  commit `c989fbd9` (ADR-0006) per ADR-0119. Companion fix in
  `libvmaf/tools/vmaf.c` resolves the picture-pool exhaustion in the
  `--frame_skip_ref/dist` loops surfaced once the always-on picture
  pool (ADR-0104) made unref'ing skipped pictures mandatory.
- **Touches**:
  - `libvmaf/tools/cli_parse.c` (`VMAF_DEFAULT_PRECISION_FMT` +
    `VMAF_LOSSLESS_PRECISION_FMT` macros, `resolve_precision_fmt()`
    body, `--help` text)
  - `libvmaf/tools/cli_parse.h` (field comments only; struct shape
    unchanged)
  - `libvmaf/src/output.c` (`DEFAULT_SCORE_FORMAT` macro)
  - `libvmaf/tools/vmaf.c` (skip loop bodies at the
    `c.frame_skip_ref` / `c.frame_skip_dist` for-loops)
  - `python/vmaf/core/result.py` (per-frame and aggregate `:.6f`
    formatters)
  - `python/test/command_line_test.py` is unmodified — Netflix golden
    assertions stay frozen per CLAUDE.md §8; the binary's output
    format adapts to them, not the other way around.
- **Invariant**: `vmaf` CLI default score-output format is `%.6f`
  (matches upstream Netflix byte-for-byte). `--precision=max|full`
  selects `%.17g` (IEEE-754 round-trip lossless). `--precision=legacy`
  is a synonym for the default. The library default for
  `vmaf_write_output_with_format(..., score_format=NULL)` matches.
  Skipped frames in the `--frame_skip_ref` / `--frame_skip_dist`
  pre-loops are `vmaf_picture_unref`'d immediately after fetch so the
  preallocated picture pool is not exhausted before the main scoring
  loop runs. **Do not** flip the macros back to `%.17g` or remove the
  unrefs without a superseding ADR — both are golden-gate-load-bearing.
- **Re-test**:

  ```bash
  ninja -C libvmaf/build
  python -m pytest python/test/command_line_test.py \
      ::VmafexecCommandLineTest::test_run_vmafexec \
      ::VmafexecCommandLineTest::test_run_vmafexec_with_frame_skipping \
      ::VmafexecCommandLineTest::test_run_vmafexec_with_frame_skipping_unequal \
      -v
  # Expected: all three PASS in <1 s combined.
  ```

  Pure fork-local; no Netflix-side conflict vector. If upstream ever
  changes the default format string, treat their value as the new
  baseline and reconfirm the golden assertions before adopting.

### 0018 — FFmpeg patches ship as ordered series.txt

- **Workstream PRs**: this PR (`fix(ci): drop dead sycl trigger +
  consolidate windows.yml into libvmaf.yml (ADR-0115)`). Surfaced
  once ADR-0115's consolidation routed the docker / FFmpeg-SYCL
  jobs through the master-targeting CI gate for the first time on
  this branch — the standalone `0003-…sycl…` apply broke because
  it referenced struct fields added by `0001-…tiny-model…`, the
  Dockerfile only `COPY`'d 0003, and `ffmpeg.yml` referenced a
  stale `../patches/` path.
- **Touches**: `Dockerfile` (lines ~86-95 — the FFmpeg patch-apply
  block), `.github/workflows/ffmpeg.yml` (the `Build FFmpeg with
  SYCL patch series` step), `ffmpeg-patches/000{1,2,3}-*.patch`
  (regenerated via real `git format-patch -3` so they carry valid
  `index <sha>..<sha> <mode>` lines and committable SHAs). Pure
  fork-local; no upstream FFmpeg or Netflix file changes.
- **Invariant**: both the Dockerfile and `ffmpeg.yml` walk
  `ffmpeg-patches/series.txt` line-by-line and apply each patch
  via `git apply` with a `patch -p1` fallback. **Do not** ship a
  new patch without appending it to `series.txt`, and **do not**
  reorder existing entries — patch 0003 references LIBVMAFContext
  fields added by patch 0001, so any out-of-order apply breaks
  the build at hunk 2 of vf_libvmaf.c.
- **Two flag-side fixes bundled in the same PR**:
  1. `--enable-libvmaf-sycl` is **not** a valid FFmpeg configure
     option. Patch 0003 uses `check_pkg_config libvmaf_sycl …`
     auto-detection (matching how `libvmaf_cuda` is wired) — it
     never registers the switch. Both Dockerfile and ffmpeg.yml
     used to pass the flag and configure rejected it with
     `Unknown option "--enable-libvmaf-sycl"`. SYCL support is
     now controlled solely by `-Denable_sycl=true` at libvmaf
     build time; FFmpeg picks it up automatically when
     `libvmaf-sycl.pc` is on `PKG_CONFIG_PATH`.
  2. The Dockerfile now carries **two** nvcc-flag ARGs.
     `NVCC_FLAGS` (libvmaf) keeps four `-gencode` lines plus the
     experimental `--extended-lambda` /
     `--expt-relaxed-constexpr` / `--expt-extended-lambda` flags
     needed for Thrust/CUB host+device code. `FFMPEG_NVCC_FLAGS`
     (FFmpeg) carries a single `-gencode arch=compute_75,code=sm_75
     -O2` — FFmpeg's `check_nvcc` runs `nvcc -ptx`, which fails with
     `nvcc fatal: Option '--ptx (-ptx)' is not allowed when
     compiling for multiple GPU architectures` on multi-arch input,
     and `--extended-lambda` requires host+device compilation.
     compute_75 PTX is forward-compatible with all newer GPUs via
     driver JIT.
  3. `--enable-libnpp` is no longer passed to FFmpeg's configure.
     FFmpeg n8.1's libnpp probe carries an explicit
     `die "ERROR: libnpp support is deprecated, version 13.0 and
     up are not supported"` (configure:7335-7336) that fires on
     the base image's CUDA 13.2 libnpp. We don't use scale_npp /
     transpose_npp / sharpen_npp in any VMAF workflow; cuvid +
     nvdec + nvenc + libvmaf-cuda is the actual GPU path. Revisit
     once we move to an FFmpeg release that supports CUDA 13
     libnpp upstream.
  4. Patch 0002 (`add-vmaf_pre-filter`) gained a missing
     `#include "libavutil/imgutils.h"` for `av_image_copy_plane()`.
     FFmpeg's libavfilter Makefile builds with
     `-Werror=implicit-function-declaration` so this fired during
     the actual compile (not configure). Caught by a local
     `docker build` rather than waiting for GitHub Actions —
     much faster iteration loop.
- **Re-test**:

  ```bash
  cd /tmp && rm -rf ffmpeg-test && \
      git clone -q --depth 1 -b n8.1 \
          https://git.ffmpeg.org/ffmpeg.git ffmpeg-test && \
      cd ffmpeg-test && \
      while IFS= read -r line; do \
          case "$line" in ''|\#*) continue ;; esac; \
          git apply "/path/to/vmaf/ffmpeg-patches/$line" \
              || patch -p1 < "/path/to/vmaf/ffmpeg-patches/$line"; \
      done < /path/to/vmaf/ffmpeg-patches/series.txt
  # Expected: all three patches apply with no rejects; the resulting
  # tree compiles with --enable-libvmaf. SYCL is auto-detected via
  # check_pkg_config (patch 0003), so no explicit configure flag is
  # required when libvmaf-sycl.pc is on PKG_CONFIG_PATH.
  ```

  Pure fork-local series; no Netflix-side conflict vector. See ADR-0118.

### 0019 — Coverage Gate annotations: upload-artifact v7 + gcovr filter

- **Workstream PRs**: this PR.
- **Touches**:
  `.github/workflows/ci.yml` (CPU + GPU coverage steps: gcovr stderr
  piped through `grep -vE 'Ignoring (suspicious|negative) hits' ... ||
  true`),
  `.github/workflows/{ci,lint,nightly,nightly-bisect,supply-chain,libvmaf}.yml`
  (`actions/upload-artifact@v5|@v6 → @v7`,
  `actions/download-artifact@v5 → @v7` in `supply-chain.yml`). Note:
  `windows.yml` was consolidated into `libvmaf.yml` by ADR-0115 / PR #50,
  so the windows-side bump now lives in `libvmaf.yml`'s
  `build (MINGW64, …)` job.
- **Invariant**: Coverage Gate Annotations panel must finish empty on a
  clean run. The two pieces are coordinated — (a) `@v7` for upload /
  download artifact actions silences GitHub's Node-20 deprecation banner
  ahead of the 2026-06-02 forced-Node-24 cutoff; (b) the gcovr stderr
  filter swallows the `Ignoring (suspicious|negative) hits` warnings
  that gcovr 8 emits for the legitimately-large hit counts in tight
  ANSNR / VIF / motion inner loops (e.g. `ansnr_tools.c:207` at ~4.93 G
  hits across an HD multi-frame coverage suite — real, not gcov bug).
  The filter is regex-narrow and anchored to gcov's exact warning
  prefix; any *other* gcovr warning still surfaces. Upstream
  (Netflix/vmaf) does not maintain these CI files; rebase impact is
  limited to the unlikely case that an upstream sync touches the
  shared `.github/workflows/` tree, which it currently does not. See
  [ADR-0117](adr/0117-coverage-gate-warning-noise-suppression.md).
- **Re-test**:

  ```bash
  # Verify gcovr filter locally (after a coverage build per entry 0014):
  ~/.local/bin/gcovr --root .. \
      --filter 'src/.*' \
      --exclude '.*/test/.*' --exclude '.*/tests/.*' \
      --exclude '.*/subprojects/.*' \
      --gcov-ignore-parse-errors=negative_hits.warn \
      --gcov-ignore-parse-errors=suspicious_hits.warn \
      --print-summary --txt build-cov-test/coverage.txt \
      build-cov-test \
    2> >(grep -vE 'Ignoring (suspicious|negative) hits' >&2 || true)
  # Expected: stderr contains the gcovr summary block but NO
  # "Ignoring (suspicious|negative) hits" lines. coverage.txt unchanged.

  # Verify all upload/download-artifact instances are on @v7:
  grep -rE 'actions/(upload|download)-artifact@v[0-6]' .github/workflows/
  # Expected: empty output.
  ```

### 0020 — CI workflow file + display-name renames (Title Case sweep)

- **Workstream PRs**: this PR; renames all six core
  `.github/workflows/*.yml` files to purpose-descriptive kebab-case and
  normalises every workflow `name:` and job `name:` to Title Case. See
  [ADR-0116](adr/0116-ci-workflow-naming-convention.md).
- **Touches**:
  `.github/workflows/{ci,lint,security,libvmaf,ffmpeg,docker}.yml`
  (renamed via `git mv` to
  `tests-and-quality-gates.yml`, `lint-and-format.yml`,
  `security-scans.yml`, `libvmaf-build-matrix.yml`,
  `ffmpeg-integration.yml`, `docker-image.yml`),
  `README.md` (5 badge URLs + labels), `docs/principles.md` (line 5
  workflow-tuple update), `.claude/skills/add-gpu-backend/SKILL.md` +
  `scaffold.sh` (filename refs), `docs/adr/0116-*.md` (new),
  `docs/adr/README.md` (index row), `CHANGELOG.md`.
- **Invariant**: workflow files are *purpose-named*; their `name:`
  fields are Title Case sentences with em-dash axis tags; job-level
  `name:` strings are Title Case sentences (Build — / Pre-Commit /
  Coverage Gate / etc.). Required-status-check contexts in `master`
  branch protection are bound to job-level names — when renaming any
  job, re-pin via
  `gh api --method PUT repos/lusoris/vmaf/branches/master/protection`.
  The 19 required gates' *semantics* are unchanged from
  [ADR-0037](adr/0037-master-branch-protection.md); only their display
  strings move.
- **Re-test**:

  ```bash
  # Validate every workflow file parses and lists the expected job names.
  cd .github/workflows
  for f in tests-and-quality-gates.yml lint-and-format.yml security-scans.yml \
           libvmaf-build-matrix.yml ffmpeg-integration.yml docker-image.yml; do
      yq '.name, .jobs.[].name' "$f" || echo "PARSE FAIL: $f"
  done
  # Expected: each workflow prints its Title Case workflow name + job names;
  # no PARSE FAIL lines.
  ```

### 0021 — DNN-enabled CI matrix legs (gcc + clang + macOS)

- **Workstream PRs**: this PR; adds three new entries to the
  `libvmaf-build` matrix in
  [`.github/workflows/libvmaf-build-matrix.yml`](../.github/workflows/libvmaf-build-matrix.yml)
  covering `-Denable_dnn=enabled` across Ubuntu/gcc, Ubuntu/clang, and
  macOS/clang. See
  [ADR-0120](adr/0120-ai-enabled-ci-matrix-legs.md).
- **Touches**:
  `.github/workflows/libvmaf-build-matrix.yml` (3 new matrix entries +
  ORT install steps + dedicated dnn-suite test step),
  `docs/adr/0120-ai-enabled-ci-matrix-legs.md` (new),
  `docs/adr/README.md` (index row),
  `CHANGELOG.md` (Added entry).
- **Invariant**: the DNN matrix legs install ONNX Runtime via the same
  pinned source as the dedicated Tiny AI job
  ([tests-and-quality-gates.yml](../.github/workflows/tests-and-quality-gates.yml))
  — Linux: MS tarball at the version pinned by `ORT_VERSION`; macOS:
  Homebrew. When the Tiny AI job's pin changes, the matrix legs'
  `ORT_VERSION` env in their `Install ONNX Runtime (linux, DNN leg)`
  step must change to match; otherwise compiler/portability coverage
  drifts away from the gating leg's actual ABI.
- **Re-test**:

  ```bash
  # Local sanity: the matrix file parses and the new job names exist.
  yq '.jobs.libvmaf-build.strategy.matrix.include[] | select(.dnn==true) | .name' \
      .github/workflows/libvmaf-build-matrix.yml
  # Expected output (3 lines):
  #   Build — Ubuntu gcc (CPU) + DNN
  #   Build — Ubuntu clang (CPU) + DNN
  #   Build — macOS clang (CPU) + DNN

  # Local DNN build sanity (matches what each leg will run):
  meson setup libvmaf libvmaf/build --buildtype release \
      --prefix $PWD/install -Denable_float=true -Denable_dnn=enabled
  ninja -vC libvmaf/build install
  meson test -C libvmaf/build --suite=dnn --print-errorlogs
  ```

- **Branch protection**: the two Linux DNN legs are pinned as required
  status checks on `master` immediately after this PR's merge (19 → 21
  contexts). The macOS leg stays informational (`experimental: true`)
  because Homebrew ORT floats. Re-pin command:

  ```bash
  gh api --method PUT repos/lusoris/vmaf/branches/master/protection \
      --input /tmp/protection-update.json
  ```

### 0022 — Windows GPU build-only matrix legs (MSVC + CUDA, MSVC + oneAPI SYCL)

- **Workstream PRs**: this PR; adds a new top-level `windows-gpu-build`
  job to
  [`.github/workflows/libvmaf-build-matrix.yml`](../.github/workflows/libvmaf-build-matrix.yml)
  with two matrix entries (CUDA, SYCL). See
  [ADR-0121](adr/0121-windows-gpu-build-only-legs.md).
- **Touches**:
  `.github/workflows/libvmaf-build-matrix.yml` (new
  `windows-gpu-build` job),
  `docs/adr/0121-windows-gpu-build-only-legs.md` (new),
  `docs/adr/README.md` (index row),
  `CHANGELOG.md` (Added entry),
  `libvmaf/src/compat/win32/pthread.h` (new — Win32 pthread shim
  for MSVC; mirrors `compat/gcc/stdatomic.h` pattern),
  `libvmaf/src/feature/integer_adm.h` (UPSTREAM — converted
  the `dwt_7_9_YCbCr_threshold[3]` designated initializer to
  positional form so MSVC/nvcc-on-Windows accepts the C++
  parse; semantically identical, no behavioural change),
  `libvmaf/src/ref.h` and
  `libvmaf/src/feature/feature_extractor.h` (UPSTREAM —
  added `#if defined(__cplusplus) && defined(_MSC_VER)`
  branch around `#include <stdatomic.h>` so MSVC C++ TUs
  pull `atomic_int` via `using std::atomic_int;`; POSIX
  paths unchanged),
  `libvmaf/src/sycl/d3d11_import.cpp` (fix non-existent
  `<libvmaf/log.h>` → `"log.h"`),
  `libvmaf/src/sycl/dmabuf_import.cpp` (move `<unistd.h>`
  inside `#if HAVE_SYCL_DMABUF` guard for non-VA-API
  hosts),
  `libvmaf/src/sycl/common.cpp` (replace POSIX
  `clock_gettime(CLOCK_MONOTONIC)` with portable
  `std::chrono::steady_clock`),
  `libvmaf/src/feature/x86/motion_avx2.c` (UPSTREAM —
  replace GCC vector-extension `__m256i[N]` indexing at
  line 529 with `_mm256_extract_epi64`; bit-exact),
  `libvmaf/src/feature/x86/adm_avx2.c` (UPSTREAM —
  replace 6 `(__m256i)(_mm256_cmp_ps(...))` casts with
  `_mm256_castps_si256(...)` and 12 `__m128i[N]`
  reductions with `_mm_extract_epi64`; bit-exact),
  `libvmaf/src/feature/x86/adm_avx512.c` (UPSTREAM —
  replace 12 `__m128i[N]` reductions with
  `_mm_extract_epi64`; bit-exact),
  `libvmaf/src/log.c` (UPSTREAM — gate `<unistd.h>`
  behind `!_WIN32`, include `<io.h>` + redirect
  `isatty`/`fileno` to `_isatty`/`_fileno` for MSVC),
  `libvmaf/src/feature/integer_vif.c` (UPSTREAM —
  switch the `aligned_malloc` cursor from `void *`
  to `uint8_t *` with explicit typed-pointer casts
  so MSVC accepts the byte-wise pointer arithmetic),
  `libvmaf/src/feature/cuda/integer_adm_cuda.c`
  (UPSTREAM — drop unused `<unistd.h>` include),
  `libvmaf/src/dnn/model_loader.c` (fork-added —
  Windows fallback definitions for POSIX `S_ISDIR`
  / `S_ISREG` path-classification macros),
  `.github/workflows/lint-and-format.yml` (fork-added —
  set `lfs: true` on the pre-commit job's checkout so
  LFS-stored ONNX blobs resolve and don't appear as
  phantom pre-commit-induced diffs),
  `libvmaf/src/feature/x86/motion_avx512.c` (UPSTREAM —
  replace 1 `__m128i[N]` reduction with
  `_mm_extract_epi64`; bit-exact),
  `libvmaf/src/feature/x86/{vif_statistic_avx2,ansnr_avx2,ansnr_avx512,float_adm_avx2,float_adm_avx512,float_psnr_avx2,float_psnr_avx512,ssim_avx2,ssim_avx512}.c`
  (UPSTREAM — convert 17 sites of trailing
  `__attribute__((aligned(N)))` to leading C11
  `_Alignas(N)`; same alignment, MSVC-portable),
  `libvmaf/src/feature/mkdirp.c` and
  `libvmaf/src/feature/mkdirp.h` (UPSTREAM third-party
  MIT-licensed micro-library — gate `<unistd.h>` to
  non-Windows, add `<direct.h>` + `_mkdir` for Windows,
  add `mode_t` typedef for MSVC),
  `libvmaf/meson.build` (new `pthread_dependency` gated on
  `cc.check_header('pthread.h')` failing),
  `libvmaf/src/meson.build` and `libvmaf/test/meson.build` (thread
  `pthread_dependency` into every target compiling pthread-using TUs).
- **Invariant**: Windows GPU legs are pinned to the same toolchain
  versions as the corresponding Linux GPU legs (CUDA 13.0.0, oneAPI
  BaseKit 2025.3.0.372) so a Linux-vs-Windows divergence implies an
  MSVC ABI issue, not a tooling-version delta. When either Linux GPU
  leg bumps its toolchain, the Windows leg must move in lockstep —
  the Intel installer URL on Windows hard-codes the per-release
  directory id and the version string, so the bump is two-line
  edits in the SYCL `Install Intel oneAPI (windows)` step (the
  `WINDOWS_BASEKIT_URL` env var). Both legs additionally inject
  `/experimental:c11atomics` into `CFLAGS` / `CXXFLAGS` because
  libvmaf uses C11 atomics that MSVC's `<stdatomic.h>` rejects
  without that opt-in flag — when MSVC ships full C11 atomics
  support, the flag becomes unconditional and can be dropped.
  Two Windows-only dependency steps round out the parity:
  the CUDA leg's `Jimver/cuda-toolkit` sub-package list includes
  both `crt` (CUDA Runtime Library compile-time headers, ships
  `crt/host_config.h`; `cuda_cccl` is not a valid Windows
  sub-package name — installer rejects it) and `nvvm` (ships
  `nvvm/bin/cicc.exe` + `nvvm/libdevice/libdevice.*.bc`; without
  it, nvcc's `.cu → PTX` stage fails with `The system cannot
  find the path specified.` — on Linux apt pulls NVVM in
  transitively with `cuda-nvcc-XY`, Windows requires it
  explicitly); the SYCL leg builds
  the Level Zero
  loader from source (`oneapi-src/level-zero` v1.18.5 →
  `cmake --build … --target install`) because Windows oneAPI
  BaseKit ships the SYCL runtime but not `ze_loader.lib`, and
  libvmaf's meson `cc.find_library('ze_loader')` needs both the
  header and the import library. When the Linux apt
  `level-zero-dev` version moves, bump the L0 git tag to match.
  `libvmaf/src/meson.build` guards the explicit `svml` / `irc`
  `cc.find_library` calls behind `host_machine.system() !=
  'windows'` — those calls exist for the gcc/g++ + icpx Linux
  flow where the host linker is non-Intel; on Windows the host
  compiler is icx-cl itself and auto-injects the Intel runtime.
  Round-10 surfaced an additional Windows-only gap: ~14 libvmaf
  TUs `#include <pthread.h>` unconditionally, but MSVC and
  clang-cl ship no pthread (MinGW does, via winpthreads). The
  fork now ships a header-only Win32 shim at
  `libvmaf/src/compat/win32/pthread.h` mapping the in-use
  pthread subset (mutex / cond / thread create+join+detach)
  onto SRWLOCK + CONDITION_VARIABLE + `_beginthreadex`. The
  shim is wired in via `pthread_dependency` in
  `libvmaf/meson.build`, declared only when
  `cc.check_header('pthread.h')` fails — so MinGW and POSIX
  paths stay untouched. When upstream Netflix/vmaf adds new
  pthread surface (e.g., `pthread_rwlock_*`), extend
  `compat/win32/pthread.h` to cover it. Both nvcc fatbin
  `custom_target`s (CUDA) and icpx `custom_target`s (SYCL
  `common.cpp` / `picture_sycl.cpp` / `dmabuf_import.cpp`,
  plus the SYCL feature kernels) bypass meson's
  `dependencies:` plumbing and hand-roll their own `-I`
  lists, so the shim
  path must be threaded into both `cuda_extra_includes` and
  `sycl_inc_flags` explicitly on Windows. icpx-cl on
  Windows additionally rejects `-fPIC` (`unsupported option
  for target 'x86_64-pc-windows-msvc'`) — so
  `sycl_common_args` and `sycl_feature_args` route their
  `-fPIC` token through `sycl_pic_arg = host_machine.system()
  != 'windows' ? ['-fPIC'] : []`. PIC is the default for
  Windows DLLs, so dropping the flag is the correct fix
  rather than a workaround. Round-14 surfaced a third
  Windows-only blocker: `libvmaf/src/feature/integer_adm.h`
  (an upstream Netflix file, last touched by upstream port
  d06dd6cf) initialises `dwt_7_9_YCbCr_threshold[3]` with
  C99 designated initializers (`{.a = ..., .k = ..., .f0 =
  ..., .g = {...}}`). The header is included from both
  `integer_adm.c` (C TU) and `cuda/integer_adm/*.cu` (C++
  TU via nvcc); MSVC's C++ frontend (and nvcc's cudafe++
  on Windows) rejects C99 designated initializers without
  `/std:c++20`. Converted to positional initialization in
  the same struct-member order (a / k / f0 / g[4]) — the
  conversion is provably semantically identical and works
  in every C/C++ standard, so it costs nothing on the
  upstream-merge side beyond a trivial conflict marker if
  upstream Netflix later edits the same lines. Restore
  designated form post-merge if upstream has it.
  Round-17 surfaced four more Windows/MSVC-only SYCL
  blockers, two of which touch upstream-shared headers.
  (a) `libvmaf/src/ref.h` and
  `libvmaf/src/feature/feature_extractor.h` (UPSTREAM)
  unconditionally `#include <stdatomic.h>` and use the
  `atomic_int` typedef in struct definitions. MSVC's
  `<stdatomic.h>` (added in 19.34) only declares the C11
  symbols inside the global namespace under C; in C++
  compilation (icpx-cl drives the SYCL TUs as C++) MSVC
  surfaces them only inside `namespace std::`. gcc/clang
  expose both via a GNU extension, so the upstream code
  works on every other platform. The fork now wraps both
  headers' `#include <stdatomic.h>` in
  `#if defined(__cplusplus) && defined(_MSC_VER)` →
  `#include <atomic>` + `using std::atomic_int;`, falling
  through to the original `<stdatomic.h>` line on every
  other configuration. ABI is unchanged — `atomic_int`
  resolves to the same underlying type. If upstream
  Netflix adds further C11 atomic typedefs in these
  headers (e.g., `atomic_uint`, `atomic_size_t`), extend
  the `using std::` lines to cover them. (b)
  `libvmaf/src/sycl/d3d11_import.cpp` (fork-added)
  used `<libvmaf/log.h>` which doesn't exist — `log.h`
  lives at `libvmaf/src/log.h` and is internal. Switched
  to `"log.h"`; the icpx invocation already supplies the
  src-relative `-I`. (c) `libvmaf/src/sycl/dmabuf_import.cpp`
  (fork-added) included `<unistd.h>` at file scope, but
  POSIX `close()` is only used inside the
  `#if HAVE_SYCL_DMABUF` VA-API block. Moved the
  `<unistd.h>` include inside that guard so non-DMA-BUF
  builds (Windows MSVC, macOS) compile cleanly. (d)
  `libvmaf/src/sycl/common.cpp` (fork-added) called
  `clock_gettime(CLOCK_MONOTONIC)`, which doesn't exist
  on Windows. Replaced with `std::chrono::steady_clock`
  (guaranteed monotonic by the C++ standard, portable on
  every supported host). All four fixes preserve
  POSIX/Linux behaviour bit-identically and only change
  the Windows MSVC build path.
  Round-18 surfaced a fifth Windows blocker on the CUDA
  leg's CPU SIMD compile path:
  `libvmaf/src/feature/x86/motion_avx2.c:529` (UPSTREAM,
  ported in commit 9371a0aa from Netflix PR #1486)
  computed `final_accum[0] + final_accum[1] +
  final_accum[2] + final_accum[3]` to extract the four
  int64 lanes from an `__m256i`. gcc/clang allow this
  via the GNU vector-extension treatment of `__m256i`
  (it carries `__attribute__((vector_size(32)))`); MSVC
  rejects it with `C2088: built-in operator '[' cannot
  be applied to an operand of type '__m256i'`. Replaced
  with `_mm256_extract_epi64(final_accum, N)` for
  N ∈ {0..3}, summed — bit-exact lane sum on every
  compiler. Restore the index form post-merge if
  upstream Netflix later edits the same lines and your
  toolchain matrix doesn't include MSVC.
  Round-19 surfaced the same MSVC pattern at 19 more
  call sites across the AVX2/AVX-512 ADM and motion
  files plus six GCC-style vector casts.
  `libvmaf/src/feature/x86/adm_avx2.c` (UPSTREAM):
  6 lines (915-920) used `(__m256i)(_mm256_cmp_ps(...))`
  C-style casts that gcc/clang accept via the GNU
  vector extension; replaced with the dedicated
  `_mm256_castps_si256(...)` bit-cast intrinsic. 12
  lane-extract sites (`r2_h[0]+r2_h[1]`, etc. at lines
  2420 / 2425 / 2430 / 2893 / 2897 / 2901 / 4079 /
  4084 / 4089 / 4627 / 4631 / 4635) replaced with
  `_mm_extract_epi64(r2_X, N)` summed pair.
  `libvmaf/src/feature/x86/adm_avx512.c` (UPSTREAM):
  6 sister lane-extract sites (lines 4470 / 4477 /
  4484 / 4625 / 4631 / 4637) — same fix. The AVX-512
  paths reduce a `__m512i` down to `__m128i` first
  (via `_mm512_extracti64x4_epi64` →
  `_mm256_extracti64x2_epi64`) before the index, so
  only the final `__m128i[N]` step needed changing.
  `libvmaf/src/feature/x86/motion_avx512.c` (UPSTREAM,
  ported in 9371a0aa from PR #1486): one final
  `r2[0]+r2[1]` reduction (line 448), same fix. All
  19 lane-extract fixes plus the 6 cast fixes are
  bit-exact rewrites and only change the source-level
  syntax to MSVC-portable form. Restore the original
  forms post-merge if upstream Netflix later edits
  the same lines and your toolchain matrix doesn't
  include MSVC. Additionally
  `libvmaf/src/sycl/d3d11_import.cpp` (fork-added)
  switched from C-style COBJMACROS helpers
  (`ID3D11Device_CreateTexture2D`, `…_Release`, etc.)
  to C++ method-call syntax (`device->CreateTexture2D`,
  `tex->Release`) — d3d11.h gates COBJMACROS behind
  `!defined(__cplusplus)`, so the C-style helpers
  aren't visible in this `.cpp` TU. The two forms are
  ABI-equivalent (both dispatch through the COM vtable);
  the choice is purely lexical and POSIX builds aren't
  affected (the whole TU is `#ifdef _WIN32`).
  Round-20 surfaced two more Windows-only blockers.
  (a) 17 sites across the x86 SIMD layer used GCC's
  `float tmp[N] __attribute__((aligned(M)));` form to
  align scratch buffers for `_mm{256,512}_store_ps`.
  MSVC rejects the trailing-attribute syntax with
  `C2146: syntax error: missing ';' before identifier
  '__attribute__'`. Replaced with the C11-standard
  `_Alignas(M) float tmp[N];` (alignment specifier
  before the type) — works in gcc, clang and MSVC
  with `/std:c11`. Files touched (all UPSTREAM):
  `vif_statistic_avx2.c` (×2), `ansnr_avx2.c` (×2),
  `ansnr_avx512.c` (×2), `float_adm_avx2.c` (×2),
  `float_adm_avx512.c` (×2), `float_psnr_avx2.c` (×1),
  `float_psnr_avx512.c` (×1), `ssim_avx2.c` (×4),
  `ssim_avx512.c` (×4). The pre-existing
  `vif_avx2.c` / `vif_avx512.c` already define a
  portable `ALIGNED(x)` macro at file scope and
  position the attribute before the type, so they
  compile cleanly under MSVC and were not touched.
  (b) `libvmaf/src/feature/mkdirp.c` (UPSTREAM,
  third-party MIT-licensed copy of Stephen Mathieson's
  micro-library) included `<unistd.h>` unconditionally
  but never used POSIX `unistd` symbols (only `mkdir`
  via `<sys/stat.h>`/`<direct.h>`). Gated `<unistd.h>`
  to non-Windows and added `<direct.h>` for Windows;
  switched `mkdir(pathname)` → `_mkdir(pathname)` (the
  non-deprecated MSVC name). `libvmaf/src/feature/mkdirp.h`
  added a `mode_t` typedef under MSVC since neither
  `<sys/types.h>` nor `<sys/stat.h>` declare it on
  Windows; `mode` is ignored on the Windows path
  anyway.
  Round-21 surfaced two more blockers (the round-19
  `__m128i[N]` sweep missed six sites) plus a
  pre-commit workflow checkout gap.
  (a) `libvmaf/src/feature/x86/adm_avx512.c` (UPSTREAM)
  had six further `r2_X[0] + r2_X[1]` reductions at
  lines 2128 / 2135 / 2142 / 2589 / 2595 / 2601 that
  reduce a `__m512i` accumulator down to `__m128i`
  before the lane index. Replaced with the same
  `_mm_extract_epi64(r2_X, N)` summed-pair pattern
  used in round 19 — bit-exact, MSVC-portable.
  (b) `libvmaf/src/log.c` (UPSTREAM) included
  `<unistd.h>` unconditionally to pick up POSIX
  `isatty` / `fileno`. On MSVC both live in
  `<io.h>` as `_isatty` / `_fileno`; gated the
  include and macro-redirected the names so the
  one call site at line 34 compiles on both sides
  without touching the POSIX path.
  (c) `.github/workflows/lint-and-format.yml`
  (fork-added) checks out without `lfs: true`, so
  the `model/tiny/*.onnx` files land as LFS
  pointer stubs. pre-commit's "changes made by
  hooks" reporter then diffs the stubs against
  HEAD's real blobs and fails the job even though
  no hook touched them. Added `lfs: true` to the
  pre-commit job's checkout.
  (d) `libvmaf/src/meson.build` —
  `cuda_common_vmaf_lib` static library had no
  `dependencies:` list, so the Win32 pthread shim
  (wired in via `pthread_dependency` in
  [libvmaf/meson.build](../libvmaf/meson.build))
  wasn't on its include path; `cuda/common.h`
  unconditionally `#include <pthread.h>` and
  MSVC failed with C1083. Added
  `dependencies : [pthread_dependency]` — no-op
  on POSIX (empty list), routes the shim path in
  on Windows.
  (e) `libvmaf/src/feature/integer_vif.c`
  (UPSTREAM) walked one big `aligned_malloc`
  result as `void *data` and did
  `data += pad_size` / `data += h *
  stride_16` etc. to carve the buffer into
  typed sub-pointers. gcc/clang accept
  pointer arithmetic on `void *` as a GNU
  extension (treating `sizeof(void) == 1`);
  MSVC rejects it with `C2036: 'void *':
  unknown size`. Replaced the cursor type
  with `uint8_t *` and added explicit casts
  at assignment sites that take a typed
  pointer (`uint16_t *mu1`, `uint32_t
  *mu1_32`, etc.). Byte offsets are
  identical, layout unchanged, bit-exact.
  If upstream Netflix edits the same loop,
  reabsorb the walk and re-apply the
  cursor-type + cast pattern.
  (f) `libvmaf/src/feature/cuda/integer_adm_cuda.c`
  (UPSTREAM) included `<unistd.h>` at line 33
  but used no POSIX symbols from it; MSVC
  failed with C1083. Dropped the unused
  include outright — simplest fix, no runtime
  change on any platform.
  (g) `libvmaf/src/dnn/model_loader.c`
  (fork-added) uses `S_ISDIR` / `S_ISREG` to
  classify resolved paths. MSVC ships the
  underlying `S_IFMT` / `S_IFDIR` / `S_IFREG`
  bit masks in `<sys/stat.h>` but not the
  POSIX classification macros. Added a
  Windows-only fallback (`#ifndef S_ISDIR
  #define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
  #endif`, same for S_ISREG) guarded by
  `#ifdef _WIN32`. Semantically identical
  to the POSIX macro on Linux/macOS.
  Round-21e surfaced the final source-portability
  blockers once the DLL build passed preprocessing.
  (h) `libvmaf/src/predict.c`, `libvmaf/src/libvmaf.c`
  and `libvmaf/src/read_json_model.c` (all UPSTREAM)
  used C99 variable-length arrays —
  `double scores[cnt]` at
  [predict.c:385](../libvmaf/src/predict.c#L385),
  `char name[name_sz]` at
  [predict.c:453](../libvmaf/src/predict.c#L453)
  and [libvmaf.c:1741](../libvmaf/src/libvmaf.c#L1741),
  plus [`cfg_name[cfg_name_sz]`](../libvmaf/src/read_json_model.c#L517)
  and [`generated_key[generated_key_sz]`](../libvmaf/src/read_json_model.c#L520)
  in the `.json` model-collection parser. gcc/clang
  accept VLAs as a C11 optional feature; MSVC
  (even with `/std:c11`) rejects them outright with
  `C2057: expected constant expression` (plus C2466
  and C2133 on the `const size_t` sized arrays —
  MSVC treats `const` as runtime-bounded, not a
  constant expression, even when the initialiser is
  literal like `4 + 1`). Replaced each runtime-sized
  buffer with a small `malloc` + explicit `free` on
  every exit path (in predict.c and read_json_model.c
  a `goto out;` cleanup arm was introduced because
  the loops error-exit mid-function). The
  `generated_key` buffer in read_json_model.c uses
  the narrower fix — `char generated_key[5];` —
  since its size (four decimal digits of the
  bootstrap sub-model index plus NUL) is a true
  compile-time constant. Buffers are a handful of
  bytes each (`name_sz` is the model-collection name
  length plus the fixed `_ci_p95_lo` suffix, `scores`
  holds ~20 doubles, `cfg_name` is the name plus
  `_0000` suffix), so the heap round-trip is not
  performance-relevant; the new `-ENOMEM` failure
  mode is handled uniformly by existing callers.
  The read_json_model.c refactor also plugs a
  pre-existing leak of the `name` buffer on the
  early `return -EINVAL` when a JSON object key
  isn't a string — the `goto out;` path frees
  `name` + `cfg_name` on every exit.
  [`libvmaf/test/test_feature_extractor.c:56`](../libvmaf/test/test_feature_extractor.c#L56)
  (UPSTREAM) declared `const unsigned n_threads = 8;`
  and used it as the extent of `VmafFeatureExtractorContext
  *fex_ctx[n_threads];`. Converted to
  `enum { n_threads = 8 };` so MSVC sees a
  constant-expression; every other compiler
  accepts enum constants identically. Re-absorb
  if upstream Netflix later edits the same loops
  and your toolchain matrix omits MSVC.
  (i) The Windows MSVC build-only legs now build
  the full tree — CLI tools, unit tests and
  libvmaf.dll — rather than the previous short
  cut of disabling `-Denable_tools` /
  `-Denable_tests`. Per user direction
  ("fix the code ffs"), the tree polyfills the
  remaining POSIX surfaces on MSVC instead:
  ([`libvmaf/tools/compat/win32/getopt.h`](../libvmaf/tools/compat/win32/getopt.h)
  +
  [`libvmaf/tools/compat/win32/getopt.c`](../libvmaf/tools/compat/win32/getopt.c))
  a from-scratch POSIX/GNU-compatible
  `getopt_long` shim (short / long options,
  `no_argument` / `required_argument` /
  `optional_argument`, argv permutation for
  non-option operands, `--` explicit stop,
  `=`-embedded values). The shim is fork-added
  (BSD-3-Clause-Plus-Patent, Copyright 2026
  Lusoris and Claude) and declared via a single
  `getopt_dependency` in
  [`libvmaf/meson.build`](../libvmaf/meson.build),
  gated on `cc.check_header('getopt.h')` failing.
  The dependency auto-propagates the shim
  `.c` into any consuming target via meson's
  `sources:` keyword, so both the `vmaf` CLI
  (`libvmaf/tools/meson.build`) and the
  `test_cli_parse` unit test
  (`libvmaf/test/meson.build`) pick it up
  uniformly. MinGW ships `<getopt.h>` via
  mingw-w64-crt, so `check_header` succeeds
  there and the shim stays out of the TU list.
  (j) Eleven test executables
  (`test_log`, `test_dict`, `test_opt`,
  `test_cpu`, `test_ref`, `test_feature`,
  `test_ciede`, `test_luminance_tools`,
  `test_cli_parse`, `test_sycl`,
  `test_sycl_pic_preallocation`) were missing
  `pthread_dependency` in their `dependencies:`
  lists at [`libvmaf/test/meson.build`](../libvmaf/test/meson.build).
  On POSIX `pthread_dependency` is an empty
  list so the omission was invisible; on MSVC
  those TUs transitively include
  `feature_collector.h` → `<pthread.h>` and
  fail with C1083. Threaded the dependency
  through all eleven targets. `test_cli_parse`
  additionally lists `getopt_dependency` to pick
  up the shim.
  (k) Three additional VLA sites surfaced once
  the test harness built on MSVC:
  [`test_cambi.c:254`](../libvmaf/test/test_cambi.c#L254)
  had `unsigned w = 5, h = 5;
  uint16_t buffer[3 * w];`; converted to
  `enum { w = 5, h = 5 };` so the array extent is
  a constant expression.
  [`test_pic_preallocation.c:382`](../libvmaf/test/test_pic_preallocation.c#L382)
  and
  [`test_pic_preallocation.c:506`](../libvmaf/test/test_pic_preallocation.c#L506)
  had `const int num_threads = N; pthread_t
  threads[num_threads];` — MSVC rejects `const int`
  as non-constant-expression. Converted to
  `enum { num_threads = N, fetches_per_thread = M };`.
  (l) [`test_ring_buffer.c:23`](../libvmaf/test/test_ring_buffer.c#L23)
  and
  [`test_pic_preallocation.c:26`](../libvmaf/test/test_pic_preallocation.c#L26)
  included `<unistd.h>` for `usleep` / `sleep`.
  Gated behind `!_WIN32` with a Win32 fallback
  via `<windows.h>` + `#define usleep(us)
  Sleep(((us) + 999) / 1000)` / `#define sleep(s)
  Sleep((s) * 1000)`. The conversion rounds
  sub-millisecond `usleep` inputs up, which is
  safe for these test paths (they use 100 µs
  jitter and 1 s waits).
  (m) [`libvmaf/tools/vmaf.c`](../libvmaf/tools/vmaf.c)
  included `<unistd.h>` for `isatty` / `fileno`.
  Applied the same gating pattern used in
  [`log.c`](../libvmaf/src/log.c#L34) in round-21(b) —
  include `<io.h>` on MSVC and redirect
  `isatty` / `fileno` to `_isatty` / `_fileno`
  via `#define`.
  (n) `__builtin_clz` / `__builtin_clzll` are
  GCC intrinsics; MSVC ships `__lzcnt` /
  `__lzcnt64` via `<intrin.h>` instead. The
  shim already lived in
  [`libvmaf/src/feature/integer_vif.h`](../libvmaf/src/feature/integer_vif.h)
  but
  [`integer_adm.c:939`](../libvmaf/src/feature/integer_adm.c#L939),
  [`x86/adm_avx2.c:1425`](../libvmaf/src/feature/x86/adm_avx2.c#L1425)
  and
  [`x86/adm_avx512.c:1217`](../libvmaf/src/feature/x86/adm_avx512.c#L1217)
  don't include that header. Extracted the shim
  into a dedicated
  [`libvmaf/src/feature/compat_builtin.h`](../libvmaf/src/feature/compat_builtin.h)
  (fork-added) and included it from all four TUs.
  The guard is `defined(_MSC_VER) && !defined(__clang__)`,
  so clang-cl / icx-cl (which provide the GCC
  intrinsics natively) skip the shim.
  (o) The SYCL leg's D3D11 import TU
  [`libvmaf/src/sycl/d3d11_import.cpp`](../libvmaf/src/sycl/d3d11_import.cpp)
  is C++ (icpx-cl drives it as C++ on Windows)
  but included the internal C header
  [`log.h`](../libvmaf/src/log.h) without an
  `extern "C"` wrap. `log.h` is an upstream
  Netflix header with no `__cplusplus` guard,
  so `vmaf_log` got C++ name-mangled in the
  .cpp TU and failed to resolve against the
  C-linkage symbol produced by `log.c` at
  link time (`LNK2019` from every test target
  that pulls in the SYCL static lib). Wrapped
  the `#include "log.h"` with
  `extern "C" { ... }` inside the fork-added
  .cpp rather than touching the upstream
  header — keeps `log.h` identical to upstream
  on every `/sync-upstream`.
  (p) The Windows MSVC legs build with
  `--default-library=static`. libvmaf's public
  API has no `__declspec(dllexport)` attributes
  (upstream Netflix is POSIX-shaped), so a
  vanilla MSVC shared build produces
  `src/vmaf-3.dll` with no exported symbols and
  the toolchain therefore never emits the
  companion `vmaf.lib` import library. Downstream
  tool targets then fail with
  `LNK1181: cannot open input file 'src\vmaf.lib'`.
  The MinGW matrix leg has used
  `--default-library static` since day one for
  the same reason
  ([line 387](../.github/workflows/libvmaf-build-matrix.yml#L387));
  the MSVC legs now mirror that choice via
  `matrix.include[].meson_extra`. Downstream
  consumers that want a DLL can either add
  `__declspec(dllexport)` decorations to the
  public API or use a `.def` file; that is a
  separate decision and out of scope for the
  build-only gate.
- **Re-test**:

  ```bash
  # Local sanity: the matrix file parses and the new job names exist.
  yq '.jobs.windows-gpu-build.strategy.matrix.include[].name' \
      .github/workflows/libvmaf-build-matrix.yml
  # Expected output (2 lines):
  #   Build — Windows MSVC + CUDA (build only)
  #   Build — Windows MSVC + oneAPI SYCL (build only)
  ```

- **Branch protection**: the two Windows GPU legs are pinned as
  required status checks on `master` immediately after this PR's
  merge. After ADR-0120's two Linux DNN legs the count moves
  21 → 23. Re-pin via:

  ```bash
  gh api --method PUT repos/lusoris/vmaf/branches/master/protection \
      --input /tmp/protection-update.json
  ```

### 0023 — CUDA gencode coverage (sm_86/sm_89/compute_80 PTX) + init hardening

- **Workstream PRs**: the ADR-0122 PR (gencode + init hardening) and
  the ADR-0123 follow-up for the `32b115df` post-cubin-load
  regression.
- **Touches**:
  - `libvmaf/src/meson.build` — the `gencode` array in the
    `if get_option('enable_nvcc')` branch.
  - `libvmaf/src/cuda/common.c` — `vmaf_cuda_state_init()` error
    paths (multi-line actionable log, `cuda_free_functions()` +
    `free(c)` + `*cu_state = NULL` cleanup).
  - `docs/backends/cuda/overview.md` — `## Runtime requirements`
    section and `### GPU architecture coverage` table.
- **Invariant**: the `gencode` array unconditionally emits cubins for
  `sm_75` / `sm_80` / `sm_86` / `sm_89` plus a `compute_80` PTX,
  independent of host `nvcc` version. Upstream Netflix's gencode
  only ships cubins at Txx major boundaries (`sm_75` / `sm_80` /
  `sm_90` / `sm_100` / `sm_120`); a literal merge that replaces our
  array with upstream's would re-open the Ampere-`sm_86` /
  Ada-`sm_89` coverage hole. The `sm_90` / `sm_100` / `sm_120`
  entries are still version-gated and should be preserved verbatim
  if upstream adds new gates. The init-path error messages are
  fork-local strings; upstream's terse `"Error: failed to load CUDA
  functions"` must NOT win a merge.
- **Re-test**:

  ```bash
  meson setup build -Denable_cuda=true -Denable_nvcc=true
  ninja -C build 2>&1 | grep -E 'compute_(80|86|89)'
  # Expect at least -gencode=arch=compute_86,code=sm_86 and
  #                -gencode=arch=compute_89,code=sm_89 and
  #                -gencode=arch=compute_80,code=compute_80

  # Actionable init message (run without CUDA driver on the loader path):
  LD_LIBRARY_PATH= ./build/tools/vmaf --help 2>&1 | grep -qi 'libcuda.so.1' || \
      echo "init log regressed"
  ```

### 0024 — `vmaf_read_pictures` null-guard for CUDA device-only path

- **Workstream PRs**: the ADR-0123 follow-up landed atop the ADR-0122
  gencode/init-hardening work.
- **Touches**:
  - `libvmaf/src/libvmaf.c` — the non-threaded tail of
    `vmaf_read_pictures` at the `prev_ref` update site (line ~1428 in
    the fork; upstream equivalent is the tail added by
    `f740276a`).
- **Invariant**: the `prev_ref` update is guarded by
  `if (ref && ref->ref)` so pure-CUDA extractor sets (where `ref =
  &ref_host` but `ref_host` was never populated by
  `translate_picture_device`) do not deref a NULL refcount. Upstream
  currently has the same unguarded tail; the bug is masked upstream
  only because the experimental `VMAF_PICTURE_POOL` gate from
  `32b115df` is still in place. A literal upstream merge that removes
  our null-guard while upstream's experimental gate is still holding
  would pass tests but re-open the `libvmaf_cuda` ffmpeg crash the
  moment the gate flips default-on (which the fork did in
  `65460e3a`, [ADR-0104](adr/0104-picture-pool-always-on.md)). Keep
  the guard until the upstream null-guard port lands.
- **Re-test**:

  ```bash
  # Unit tests cover the non-regression on the library side:
  meson test -C build

  # End-to-end regression: ffmpeg libvmaf_cuda must exit 0 on a
  # CUDA-device-only extractor set (full recipe in ADR-0123).
  ./ffmpeg -init_hw_device cuda=cu:0 -filter_hw_device cu \
    -i /tmp/ref.mp4 -i /tmp/dis.mp4 \
    -lavfi "[0:v]format=yuv420p,hwupload_cuda[r];\
            [1:v]format=yuv420p,hwupload_cuda[d];\
            [r][d]libvmaf_cuda=log_path=/tmp/out.json:log_fmt=json" \
    -f null -
  ```

### 0025 — VIF `init()` fail-path frees advanced byte-cursor

- **Workstream PRs**: PR #47 (rewritten to leak-fix-only after master
  absorbed the void*→uint8_t* half via commit `b0a4ac3a`, entry 0022
  §e). Ports the leak-fix half of upstream Netflix PR
  [#1476](https://github.com/Netflix/vmaf/pull/1476).
- **Touches**: `libvmaf/src/feature/integer_vif.c` (UPSTREAM —
  2-line fix in the `init()` `fail:` handler).
- **Invariant**: `init()` walks `uint8_t *data` forward through
  `aligned_malloc`'s one allocation, advancing past each
  sub-pointer assignment. If
  `vmaf_feature_name_dict_from_provided_features` returns NULL
  the fail path must free the *base* pointer `s->public.buf.data`,
  never the advanced cursor `data`. Upstream master still has
  `aligned_free(data)` there — same bug — so this entry is the
  reminder to not let an upstream sync re-introduce the
  advanced-cursor form. If upstream lands PR #1476 or an
  equivalent, the sync can drop this entry.
- **Re-test**:

  ```bash
  meson test -C build --suite=fast
  # Static check: ripgrep the pattern that must NOT return.
  rg -n "aligned_free\(data\)" libvmaf/src/feature/integer_vif.c && \
      echo 'REGRESSED' || echo 'ok'
  ```

### 0026 — Automated rule-enforcement workflow + copyright pre-commit hook

- **Workstream PRs**: this PR (ADR-0124 adoption). Closes the
  "rule-without-a-check" gap on ADR-0100 / 0105 / 0106 / 0108.
- **Touches** (all FORK-ADDED — no upstream overlap):
  `.github/workflows/rule-enforcement.yml` (new),
  `scripts/ci/check-copyright.sh` (new),
  `.pre-commit-config.yaml` (appended local hook).
- **Invariant**: the `deep-dive-checklist` job is **blocking** on
  every PR that is not an upstream port (exempt via `port:` title
  prefix or `port/` branch). The other three gates
  (`doc-substance-check`, `adr-backfill-check`, copyright
  pre-commit) are advisory or pre-commit, never CI-blocking; this
  split is the whole point of ADR-0124 and an upstream sync must
  not move them into the required-status-check set without a
  follow-up ADR. The opt-out parser matches
  `/^-?\s*no .* (?:needed|impact|rebase-sensitive)/` per
  ADR-0108 §Opt-out-lines — if upstream ever changes PR-template
  phrasing (unlikely; this is fork-local), the regex and the
  template must move together.
- **Re-test**:

  ```bash
  # Lint the workflow + hook locally.
  pre-commit run --files \
    .github/workflows/rule-enforcement.yml \
    scripts/ci/check-copyright.sh \
    .pre-commit-config.yaml

  # Dry-run the copyright hook against a staged source file.
  scripts/ci/check-copyright.sh libvmaf/src/libvmaf.c && echo ok

  # Synthetic PR body that violates ADR-0108 should fail the parser;
  # see docs/research/0002-automated-rule-enforcement.md §Verification
  # plan for the three test cases.
  ```

### 0027 — SSIMULACRA 2 scalar extractor (libjxl FastGaussian IIR blur)

- **Workstream PRs**: this PR (`feat/ssimulacra2-scalar`); proposal
  ADR in PR #67.
- **Touches**:
  `libvmaf/src/feature/ssimulacra2.c` (fork-local, new),
  `libvmaf/src/meson.build`,
  `libvmaf/src/feature/feature_extractor.c`.
- **Invariant**: the extractor embeds several tables that must
  track libjxl upstream — opsin absorbance matrix, `MakePositiveXYB`
  offsets, 108 pooling weights, polynomial-transform coefficients,
  and the FastGaussian coefficient-derivation formulas
  (radius = `3.2795·σ + 0.2546`, Cramer's 3×3 solve for β, n2/d1
  assignment per Charalampidis 2016 (33)). If libjxl ever changes
  any of these, update `ssimulacra2.c` in the same PR that syncs
  upstream. Self-consistency must stay at exactly `100.000000` for
  identical ref/dist inputs — this is the cheapest regression check.
- **Re-test**:

  ```bash
  meson test -C build --suite=fast
  ./build/tools/vmaf \
    --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
    --distorted python/test/resource/yuv/src01_hrc00_576x324.yuv \
    -w 576 -h 324 -p 420 -b 8 --feature ssimulacra2 -o /tmp/self.xml \
    && grep -q 'ssimulacra2="100.000000"' /tmp/self.xml \
    && echo "ok: self-consistency 100.0"
  ```

### 0028 — MS-SSIM separable decimate + AVX2/AVX-512/NEON SIMD

- **Workstream PRs**: `feat/ms-ssim-decimate-simd-v2` (supersedes the
  rebase-incompatible `feat/ms-ssim-decimate-simd`; AVX2/AVX-512,
  commits `7de8cd7f` scalar separable, `5f93c864` AVX2, `73436438`
  AVX-512); `feat/ms-ssim-decimate-neon-v2` (NEON follow-up, stacked).
- **Touches**:
  `libvmaf/src/feature/ms_ssim_decimate.{c,h}` (NEW),
  `libvmaf/src/feature/x86/ms_ssim_decimate_avx2.{c,h}` (NEW),
  `libvmaf/src/feature/x86/ms_ssim_decimate_avx512.{c,h}` (NEW),
  `libvmaf/src/feature/arm64/ms_ssim_decimate_neon.{c,h}` (NEW),
  `libvmaf/src/feature/ms_ssim.c` (call-site change),
  `libvmaf/src/meson.build` (register new SIMD TUs),
  `libvmaf/test/test_ms_ssim_decimate.c` (NEW),
  `libvmaf/test/meson.build` (arm64 gating).
- **Invariant**: the 9-tap 9/7 biorthogonal wavelet LPF
  coefficients (`ms_ssim_lpf_h` / `ms_ssim_lpf_v`) are duplicated
  verbatim in five TUs for bit-identity: the scalar
  `ms_ssim_decimate.c`, the AVX2 variant, the AVX-512 variant, the
  NEON variant, and upstream's `g_lpf_h` / `g_lpf_v` in
  `ms_ssim.c`. Any upstream change to the coefficient values or the
  `KBND_SYMMETRIC` mirror branch in `iqa/convolve.c` must be
  mirrored to all five. If not mirrored, SIMD paths and scalar
  diverge silently and the bit-equality `memcmp` in
  `test_ms_ssim_decimate` catches it — but only when that test
  runs, so diff the five files first.
- **Re-test** (on each supported host arch):

  ```bash
  # x86_64 host — native build.
  meson test -C build
  ./build/test/test_ms_ssim_decimate

  # aarch64 host OR aarch64 cross under qemu — see /tmp/aarch64-cross.txt.
  meson setup build-arm64 libvmaf --cross-file /tmp/aarch64-cross.txt \
      -Denable_cuda=false -Denable_sycl=false
  ninja -C build-arm64
  qemu-aarch64-static -L /usr/aarch64-linux-gnu \
      build-arm64/test/test_ms_ssim_decimate

  # Netflix MS-SSIM golden — places=4 must still pass through SIMD.
  .venv/bin/python -m pytest \
      python/test/feature_extractor_test.py::FeatureExtractorTest::test_run_ms_ssim_fextractor
  ```

### 0029 — `KBND_SYMMETRIC` period-based reflection in `iqa/convolve.c`

- **Workstream PRs**: `feat/ms-ssim-decimate-simd-v2` follow-up
  (CI triage on PR #69, 2026-04-20).
- **Touches**: `libvmaf/src/feature/iqa/convolve.c` (upstream
  file, rewritten `KBND_SYMMETRIC`).
- **Invariant**: `KBND_SYMMETRIC(img, w, h, x, y, _)` must use the
  period-based form (`period = 2*w`, `period = 2*h`) so that offsets
  with `|x| > w` or `|y| > h` still land in bounds. Upstream's
  single-reflect form was out-of-bounds whenever `w < kernel_half`
  or `h < kernel_half`; the latent bug did not reproduce in Netflix
  golden tests because MS-SSIM pyramids never decimate below
  ~60×34. Any upstream change that reverts to the single-reflect
  form must be rejected or re-ported.
- **Re-test**:

  ```bash
  ./build/test/test_ms_ssim_decimate        # test_1x1 border case
  .venv/bin/python -m pytest \
      python/test/feature_extractor_test.py::FeatureExtractorTest::test_run_ms_ssim_fextractor
  ```

### 0030 — `adm_decouple_s123_avx512` stack-array 64-byte alignment

- **Workstream PRs**: `feat/ms-ssim-decimate-simd-v2` follow-up
  (CI triage on PR #69, 2026-04-20).
- **Touches**: `libvmaf/src/feature/x86/adm_avx512.c` (upstream
  file, one-line `_Alignas(64)` on `int64_t angle_flag[16]` at
  line 1317). `libvmaf/test/test_pic_preallocation.c` (upstream
  file, three `vmaf_model_destroy(model)` calls pairing the
  `vmaf_model_load` in `test_picture_pool_basic` / `_small` /
  `_yuv444`).
- **Invariant**: the stack slot for `angle_flag` must be 64-byte
  aligned because two `_mm512_loadu_si512(&angle_flag[0/8])`
  loads in the same scope may be promoted to aligned `vmovdqa64`
  by LTO. Dropping the `_Alignas(64)` annotation re-introduces
  the SEGV under `--buildtype=release -Db_lto=true
  -Db_sanitize=address`. Debug / no-LTO builds keep
  `vmovdqu64` and cannot flag the regression. See
  `docs/development/known-upstream-bugs.md`.
- **Re-test**:

  ```bash
  meson setup build-asan-lto libvmaf \
      -Denable_cuda=false -Denable_sycl=false \
      -Db_sanitize=address --buildtype=release -Db_lto=true
  ninja -C build-asan-lto test/test_pic_preallocation
  ASAN_OPTIONS=detect_leaks=1 \
      ./build-asan-lto/test/test_pic_preallocation
  ```

### 0031 — Batch-A upstream-port small-fix sweep (ports of unmerged PRs)

- **Workstream PRs**: `feat/batch-a-upstream-small-fix-sweep` — commits
  `546a40ee` (T0-1), `8fed8ad1` (T4-4), `83a1db46` (T4-5),
  `34425dee` (T4-6). ADRs 0131, 0132, 0134, 0135.
- **Touches**:
  - `libvmaf/src/cuda/picture_cuda.c` (one-line `cuMemFree` port of
    [Netflix#1382][pr1382])
  - `libvmaf/src/feature/feature_collector.c` +
    `libvmaf/test/test_feature_collector.c` (mount/unmount bugfix
    port of [Netflix#1406][pr1406] + shared-helper test refactor)
  - `libvmaf/src/meson.build` (`declare_dependency` +
    `override_dependency` port of [Netflix#1451][pr1451])
  - `libvmaf/include/libvmaf/model.h`, `libvmaf/src/model.c`,
    `libvmaf/test/test_model.c`, `docs/api/index.md` (built-in
    model iterator port of [Netflix#1424][pr1424])
- **Invariant**: each of the four upstream PRs is OPEN (unmerged) on
  the port date; when Netflix merges any of them, the fork's
  version is correction-bearing (T4-4 test refactor, T4-6 three
  defect fixes + Doxygen doc expansion), not line-identical.
  Resolution on upstream merge is **always "keep fork version"**
  because the fork's version already satisfies the PR's intent
  and additionally fixes the defects.
  - Netflix#1406 conflict will land in `test_feature_collector.c`
    — fork uses `load_three_test_models()` helper vs upstream's
    inline per-model `VmafModel *m0, *m1, *m2;` duplication.
  - Netflix#1424 conflict will land in `libvmaf/src/model.c` and
    `libvmaf/test/test_model.c` — fork uses `else if` guard +
    `idx + 1 < CNT` + const-qualified test types.
  - Netflix#1382 and Netflix#1451 are line-identical in substance;
    merge should be clean aside from trailing-comma style drift.
- **Re-test**:

  ```bash
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false
  ninja -C build test/test_feature_collector test/test_model
  build/test/test_feature_collector
  build/test/test_model
  # Expected: 6/6 pass in test_feature_collector (mount/unmount
  # 3-model sequences); 39/39 pass in test_model (includes
  # test_version_next full-iteration invariant).
  ```

[pr1382]: https://github.com/Netflix/vmaf/pull/1382
[pr1406]: https://github.com/Netflix/vmaf/pull/1406
[pr1424]: https://github.com/Netflix/vmaf/pull/1424
[pr1451]: https://github.com/Netflix/vmaf/pull/1451

### 0032 — Thread-local locale handling for numeric I/O (port of Netflix/vmaf#1430)

- **Workstream PRs**: `port/netflix-1430-thread-locale`
  (T4-3 from the "Batch-A follow-up" sweep, 2026-04-20).
- **Touches**: `libvmaf/src/thread_locale.h` / `libvmaf/src/thread_locale.c`
  (new, upstream-authored); `libvmaf/src/meson.build` (two
  `cdata.set('HAVE_USELOCALE'/'HAVE_XLOCALE_H')` probes +
  `src_dir + 'thread_locale.c'` in `libvmaf_sources`);
  `libvmaf/src/output.c` (four writers gain
  `push_c()` + `pop()` bracket, preserving fork's
  `ferror(outfile) ? -EIO : 0` return contract from
  [ADR-0119](adr/0119-cli-precision-default-revert.md));
  `libvmaf/src/svm.cpp` (drop `<locale.h>` include; replace
  `setlocale/strdup/setlocale` bracket with
  `vmaf_thread_locale_push_c/pop`; add
  `buffer.imbue(std::locale::classic())` to both SVM parser ctors
  with fork's K&R + 4-space style);
  `libvmaf/src/read_json_model.c` (bracket `model_parse` with
  push/pop); `libvmaf/test/meson.build` (new
  `test_locale_handling` target + test registration);
  `libvmaf/test/test_locale_handling.c` (new, upstream-authored
  with three fork corrections for the `score_format` parameter).
- **Invariant**: fork's output writers return
  `ferror(outfile) ? -EIO : 0` — this must survive any upstream
  refactor of the writer bodies. The `push_c()` call MUST be
  paired with a `pop()` on every return path (writer bodies have
  a single tail return, so the pattern is locally
  `push → body → pop → return ferror-check`). Dropping
  `pop()` leaks a `locale_t` on POSIX and leaves the thread
  locked to "C" on Windows.
- **Re-test**:

  ```bash
  meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build test_locale_handling
  # Repro the user-visible failure without the fix:
  LC_ALL=de_DE.UTF-8 build/tools/vmaf --reference ref.yuv \
      --distorted dis.yuv --width 1920 --height 1080 \
      --pixel_format 420 --bitdepth 8 --output result.json \
      --json
  # Assert output contains period decimals, not comma.
  python -c "import json; d=json.load(open('result.json')); \
      assert all('.' in repr(v) for v in \
      [f['metrics']['vmaf'] for f in d['frames']])"
  ```

- **On upstream sync**: when Netflix merges PR #1430, the
  `(cherry picked from commit 054a97ed…)` trailer in
  `git log port/netflix-1430-thread-locale` lets the next
  `/sync-upstream` skip this commit. If the upstream diff
  drifts, redo the three fork corrections listed in
  [ADR-0137](adr/0137-thread-local-locale-for-numeric-io.md)
  §Decision.

### 0033 — SSIM / MS-SSIM SIMD bit-exact to scalar via per-lane scalar double

- **Workstream PRs**: `feat/ms-ssim-decimate-neon` (this PR —
  companion to the ADR-0138 convolve fast path).
- **Touches**:
  [`libvmaf/src/feature/x86/ssim_avx2.c`](../libvmaf/src/feature/x86/ssim_avx2.c)
  and
  [`libvmaf/src/feature/x86/ssim_avx512.c`](../libvmaf/src/feature/x86/ssim_avx512.c)
  — `ssim_accumulate_*` rewritten. `ssim_precompute_*` and
  `ssim_variance_*` unchanged (they were already bit-exact).
  Plus the new bit-exact
  [`convolve_avx2.c`](../libvmaf/src/feature/x86/convolve_avx2.c) /
  [`convolve_avx512.c`](../libvmaf/src/feature/x86/convolve_avx512.c)
  and the upstream h-pass OOB fix at
  [`iqa/convolve.c:159`](../libvmaf/src/feature/iqa/convolve.c#L159).
- **Invariants** (see
  [ADR-0139](adr/0139-ssim-simd-bitexact-double.md) §Decision):
  1. **Convolve taps** — *single-rounded `float*float` → widen →
     `double` add*, NO FMA. Mirrors scalar `sum += img[i]*k[j]` in
     [`iqa/convolve.c`](../libvmaf/src/feature/iqa/convolve.c).
  2. **SSIM accumulate** — scalar's `2.0 *` literal
     (`2.0 * ref_mu[i] * cmp_mu[i] + C1` and `2.0 * srsc + C2`)
     is a C `double` literal. Both SIMD accumulators do the `2.0 *`
     numerator + division + final `l*c*s` product per-lane in
     scalar double to match scalar type promotions byte-for-byte.
  3. **H-pass outer-loop bound** — `y < dst_h + vc - kh_even` (not
     `y < dst_h + vc`); the `- kh_even` is load-bearing because the
     last cache row on even-tap kernels (e.g. box-8) is never read
     by the v-pass but was previously written OOB when image height
     equals kernel height.

  Fork-local SSIM SIMD is NOT upstream. If upstream ever adds their
  own SSIM AVX2/AVX-512, **keep the fork's version on conflict** —
  it's the only variant verified bit-exact to scalar at
  `--precision max`.
- **Re-test**:

  ```bash
  meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build test_iqa_convolve test_ms_ssim_decimate
  # Bit-exactness check across dispatch backends:
  FIX=python/test/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv
  DIS=python/test/resource/yuv/checkerboard_1920_1080_10_3_1_0.yuv
  for m in 255 16 0; do
    build/tools/vmaf --cpumask $m --reference $FIX --distorted $DIS \
        --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
        --feature float_ssim --feature float_ms_ssim \
        --output /tmp/ssim_$m.xml --precision max
  done
  diff <(grep -v '<fyi fps' /tmp/ssim_255.xml) \
       <(grep -v '<fyi fps' /tmp/ssim_16.xml)    # expect empty
  diff <(grep -v '<fyi fps' /tmp/ssim_255.xml) \
       <(grep -v '<fyi fps' /tmp/ssim_0.xml)     # expect empty
  ```

- **On upstream sync**: the AVX2/AVX-512 SSIM surface is entirely
  fork-local (upstream has VIF/ADM/motion/CAMBI SIMD but no SSIM).
  If upstream ever introduces SSIM SIMD, their kernel bodies will
  almost certainly compute `l*c*s` in vector float for throughput —
  **do not adopt**. The fork's per-lane-scalar-double reduction is
  required for the bit-exactness claim. Same applies to
  `convolve_avx2/512` — they are fork-only; dispatch sits in
  `ssim_tools.c` via `_iqa_convolve_set_dispatch`.

### 0034 — SIMD DX framework + NEON SSIM/convolve bit-exact port

- **Workstream PRs**: `feat/simd-dx-framework` (this PR, PR #A);
  ships the two demos on top of which PR #B will consume the
  framework (ssimulacra2, motion_v2, vif_statistic, ...).
- **Touches**:
  [`libvmaf/src/feature/simd_dx.h`](../libvmaf/src/feature/simd_dx.h)
  (new header),
  [`libvmaf/src/feature/arm64/convolve_neon.c`](../libvmaf/src/feature/arm64/convolve_neon.c)
  +
  [`convolve_neon.h`](../libvmaf/src/feature/arm64/convolve_neon.h)
  (new NEON port),
  [`libvmaf/src/feature/arm64/ssim_neon.c`](../libvmaf/src/feature/arm64/ssim_neon.c)
  (`ssim_accumulate_neon` rewritten for ADR-0139 bit-exactness;
  `precompute` + `variance` unchanged),
  [`libvmaf/src/feature/float_ssim.c`](../libvmaf/src/feature/float_ssim.c)
  +
  [`libvmaf/src/feature/float_ms_ssim.c`](../libvmaf/src/feature/float_ms_ssim.c)
  (wire `iqa_convolve_neon` into the aarch64 dispatch setters),
  [`libvmaf/src/meson.build`](../libvmaf/src/meson.build)
  (`arm64_sources` += convolve_neon.c),
  [`libvmaf/test/meson.build`](../libvmaf/test/meson.build)
  (`test_iqa_convolve` arch filter extended to `arm64` / `aarch64`),
  [`libvmaf/test/test_iqa_convolve.c`](../libvmaf/test/test_iqa_convolve.c)
  (NEON variant check + aarch64 CPU flag detection),
  [`libvmaf/test/dnn/meson.build`](../libvmaf/test/dnn/meson.build)
  (`test_cli.sh` gated on `not meson.is_cross_build()` — bash
  invokes `$VMAF_BIN` directly so meson's exe_wrapper isn't
  applied), new
  [`build-aux/aarch64-linux-gnu.ini`](../build-aux/aarch64-linux-gnu.ini)
  meson cross-file,
  [`.claude/skills/add-simd-path/SKILL.md`](../.claude/skills/add-simd-path/SKILL.md)
  (upgraded kernel-spec flags).
- **Invariants** (see [ADR-0140](adr/0140-simd-dx-framework.md)
  §Decision):
  1. `simd_dx.h` is fork-local. Keep the fork's version on upstream
     conflict. Macro names are ISA-suffixed (`_AVX2_4L`,
     `_AVX512_8L`, `_NEON_4L`) — do not collapse into a cross-ISA
     abstraction; the fork's SIMD policy
     (user-memory `feedback_simd_dx_scope.md`) rules out
     Highway / simde / xsimd.
  2. The ADR-0138 widen-then-add rule (single-rounded
     `float * float` → widen → `double` add, NO FMA) applies to
     NEON exactly as to AVX2 / AVX-512. The NEON form uses paired
     `float64x2_t` accumulators (lo / hi) because NEON has no
     `float64x4_t`.
  3. The ADR-0139 per-lane scalar-double reduction rule applies to
     `ssim_accumulate_neon` exactly as to the AVX2 / AVX-512
     variants. The NEON implementation uses
     `SIMD_ALIGNED_F32_BUF_NEON` (`_Alignas(16) float name[4]`) +
     a 4-iteration scalar loop.
- **Re-test** (requires `aarch64-linux-gnu-gcc` +
  `qemu-user-static` + aarch64 sysroot at `/usr/aarch64-linux-gnu`):

  ```bash
  cd libvmaf
  meson setup ../build-aarch64 \
    --cross-file ../build-aux/aarch64-linux-gnu.ini \
    -Denable_cuda=false -Denable_sycl=false -Denable_dnn=disabled
  cd ..
  ninja -C build-aarch64
  meson test -C build-aarch64                       # expect 31/31 OK
  # Bit-exactness check scalar vs NEON under QEMU:
  REF=python/test/resource/yuv/src01_hrc00_576x324.yuv
  DIS=python/test/resource/yuv/src01_hrc01_576x324.yuv
  for m in 255 0; do
    LD_LIBRARY_PATH=$PWD/build-aarch64/src qemu-aarch64-static \
      -L /usr/aarch64-linux-gnu build-aarch64/tools/vmaf \
      --cpumask $m --reference $REF --distorted $DIS \
      --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
      --feature float_ssim --feature float_ms_ssim \
      --output /tmp/ssim_$m.xml --precision max
  done
  diff <(grep -v '<fyi fps' /tmp/ssim_255.xml) \
       <(grep -v '<fyi fps' /tmp/ssim_0.xml)     # expect empty
  ```

- **On upstream sync**: upstream has no NEON SSIM and no NEON
  convolve for IQA. If they ever add one, **keep the fork's
  version on conflict** — the fork's NEON path is the only variant
  verified bit-exact to scalar at `--precision max`. The
  `build-aux/aarch64-linux-gnu.ini` cross-file has no upstream
  equivalent. The `/add-simd-path` skill is fork-only; upstream
  doesn't ship `.claude/skills/`.

### 0036 — Port Netflix generalised AVX convolve + ADR-0141 cleanup

- **Workstream PRs**: `port/upstream-f3a628b4-generalized-avx-convolve` (this PR).
- **Upstream commit**: [`f3a628b4`](https://github.com/Netflix/vmaf/commit/f3a628b4)
  "feature/common: generalize avx convolution for arbitrary filter widths"
  (Kyle Swanson, 2026-04-21).
- **Touches**:
  - [convolution.h](../libvmaf/src/feature/common/convolution.h)
    — upstream-tracking: adds `#define MAX_FWIDTH_AVX_CONV 17`.
  - [convolution_avx.c](../libvmaf/src/feature/common/convolution_avx.c)
    — upstream-tracking (2,500 LoC deletion) **plus fork-delta cleanup**
    per ADR-0141: four scanline helpers `convolution_f32_avx_s_1d_*`
    changed from external linkage to `static` (no other TU uses them
    after the specialised-path removal); stride parameters widened
    from `int` to `ptrdiff_t` in the helpers, with `(ptrdiff_t)` casts
    at public-function multiplication sites; `#include <stddef.h>`
    added for the type.
  - [`libvmaf/src/feature/vif_tools.c`](../libvmaf/src/feature/vif_tools.c) —
    upstream-tracking: three AVX dispatch sites drop the
    `fwidth == 17 || ... == 3` whitelist in favour of
    `fwidth <= MAX_FWIDTH_AVX_CONV`.
  - [`python/test/quality_runner_test.py`](../python/test/quality_runner_test.py),
    [`python/test/vmafexec_test.py`](../python/test/vmafexec_test.py) —
    upstream-authored loosening of two full-VMAF-score assertions
    from `places=2` (±0.005) to `places=1` (±0.05). Adopted per the
    ADR-0142 Netflix-authority precedent (project rule #1 addresses
    fork drift, not upstream-authored test updates the fork must
    track).
- **Invariants** (see
  [ADR-0143](adr/0143-port-netflix-f3a628b4-generalized-avx-convolve.md)
  §Decision):
  1. **Static linkage on scanline helpers** — upstream leaves the four
     `convolution_f32_avx_s_1d_*_scanline` helpers with external
     linkage out of habit; the fork narrows them to `static`. On
     upstream sync: if upstream ever externs them from another TU,
     that's a flag to re-audit; keep the fork's `static` unless the
     reference is real.
  2. **`ptrdiff_t` strides inside helpers** — the public
     `convolution_f32_avx_*_s` wrappers keep `int` strides (matching
     the upstream interface + `convolution.h` declarations). Helpers
     take `ptrdiff_t` to silence `bugprone-implicit-widening-of-
     multiplication-result`. If upstream changes the public interface
     to `ptrdiff_t`, drop the fork's wrapper-level casts.
  3. **`MAX_FWIDTH_AVX_CONV = 17`** — the ceiling is upstream's; if
     upstream bumps it, the fork must rebuild + re-run the VIF golden
     test pair.
- **Re-test**:

  ```bash
  meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build            # expect 32/32 OK
  clang-tidy -p build libvmaf/src/feature/common/convolution_avx.c
  # Zero warnings expected on the touched file.
  ```

  Netflix CPU golden CI leg exercises the two loosened assertions;
  confirmed locally under meson test.
- **On upstream sync**: upstream is the source of truth for
  `convolution_avx.c`, `convolution.h`, `vif_tools.c` dispatch, and
  the two python golden tolerances. On a rebase, prefer upstream for
  those files **except**:
  - Keep the fork's `static` on the four scanline helpers.
  - Keep the fork's `ptrdiff_t` helper signatures + multiplication-
    site casts (unless upstream adopts them too, in which case
    converge).
  - Keep the fork's `#include <stddef.h>`.
  If upstream re-introduces a specialised fast path for common
  widths, evaluate on a per-fwidth perf profile — the fork's
  `/profile-hotpath` skill covers this.

### 0038 — `motion_v2` NEON SIMD (fork-local)

- **Workstream PR**: `port/motion-bundle-neon-and-updates` (this PR).
- **Upstream**: none — aarch64 NEON for `motion_v2` is fork-local.
  Upstream scalar + AVX2 + AVX-512 variants exist; this PR adds the
  missing NEON fourth path. Scalar is the bit-exactness ground truth.
- **Touches** (fork-local):
  - [motion_v2_neon.c](../libvmaf/src/feature/arm64/motion_v2_neon.c)
    — new TU, ~300 LoC. 4-wide int32 SIMD over the 5-tap Gaussian
    pipeline. Five `static inline` helpers keep every function
    under the ADR-0141 60-line budget.
  - [motion_v2_neon.h](../libvmaf/src/feature/arm64/motion_v2_neon.h)
    — new header declaring the two public entry points.
  - [integer_motion_v2.c](../libvmaf/src/feature/integer_motion_v2.c)
    — dispatch update: adds an `#if ARCH_AARCH64` block in `init`
    that selects the NEON variant when `VMAF_ARM_CPU_FLAG_NEON` is
    present, mirroring the existing x86 dispatch blocks.
  - [`libvmaf/src/meson.build`](../libvmaf/src/meson.build) — add
    `arm64/motion_v2_neon.c` to the `arm64_sources` list.
- **Invariants** (see
  [ADR-0145](adr/0145-motion-v2-neon-bitexact.md) §Decision):
  1. **Arithmetic right-shift throughout**. The fork's AVX2 path
     uses `_mm256_srlv_epi64` (*logical*) which can diverge from
     scalar on negative-diff pixels. The NEON port uses
     `vshrq_n_s64(v, 16)` for the known Phase-2 shift and
     `vshlq_s64(v, -(int64_t)bpc)` for the variable Phase-1
     shift — both arithmetic, matching scalar C `>>` on signed
     integer. On rebase: keep the arithmetic forms; do NOT adopt
     `vshrq_n_u64` or a logical emulation even if it runs faster.
  2. **4-lane stride + mirror tails**. SIMD stride = 4; scalar
     tails cover the remainder. The Phase-2 helper
     `x_conv_row_sad_neon` hands 4 lanes to `x_conv_block4_neon`
     and drops to scalar for both left/right edges (`j < 2` and
     `j + 6 > w`). On rebase: preserve the 4-lane stride and the
     two-sided scalar tail.
  3. **Signature parity with AVX2**. Both pipeline entry points
     match the AVX2 + AVX-512 variants' `(const uint8_t *prev,
     ptrdiff_t, const uint8_t *cur, ptrdiff_t, int32_t *y_row,
     unsigned w, unsigned h, unsigned bpc)` signature. On rebase:
     if upstream changes the signature, mirror the change here
     AND in the x86 variants in lockstep.
- **Re-test**:

  ```bash
  meson setup build-aarch64 libvmaf \
    --cross-file build-aux/aarch64-linux-gnu.ini \
    -Denable_cuda=false -Denable_sycl=false
  ninja -C build-aarch64
  meson test -C build-aarch64 --no-rebuild   # expect 31/31 OK
  clang-tidy -p build-aarch64 \
    libvmaf/src/feature/arm64/motion_v2_neon.c
  # Zero warnings expected on the touched file.

  # NEON-vs-scalar bit-exact diff under QEMU:
  YUV=python/test/resource/yuv
  for mask in 0 255; do
    LD_LIBRARY_PATH=build-aarch64/src \
      qemu-aarch64-static -L /usr/aarch64-linux-gnu \
      build-aarch64/tools/vmaf \
      -r $YUV/src01_hrc00_576x324.yuv \
      -d $YUV/src01_hrc01_576x324.yuv \
      -w 576 -h 324 -p 420 -b 8 -n --feature motion_v2 \
      --cpumask $mask -o /tmp/mv2_$mask.xml --precision max
  done
  diff <(grep -v 'fps=' /tmp/mv2_0.xml) \
       <(grep -v 'fps=' /tmp/mv2_255.xml)  # expect empty
  ```

- **On upstream sync**: upstream has no NEON `motion_v2` and has not
  signalled plans to add one. If they ever do, diff their NEON
  against the fork's: on logical-vs-arithmetic shift, keep the
  fork's arithmetic form (matches scalar). On the function
  decomposition (the five helpers), adopt upstream's if it's
  smaller; the fork's layout is ADR-0141-driven, not a semantic
  contract.
- **Follow-up T7-32 (closed 2026-04-29)**: AVX2 `motion_v2`
  audit against scalar on a negative-diff corpus is in tree as
  [`libvmaf/test/test_motion_v2_simd.c`](../libvmaf/test/test_motion_v2_simd.c).
  The test exercises four adversarial 16-bit fixtures
  (uniform-negative diffs at bpc 10 and 12, alternating-mixed-sign
  at bpc 10 and 12) and `memcmp`-equivalent compares the AVX2
  Phase-1 + Phase-2 SAD against a line-for-line scalar reference
  duplicated from `integer_motion_v2.c`. On the bench host the
  AVX2 path agrees with scalar on every fixture: the post-`abs()`
  Phase-2 aggregation absorbs the per-lane logical-vs-arithmetic
  shift difference for these inputs. The test stays as a permanent
  guard — if a future fixture, microarch, or compiler change
  surfaces a delta, the byte-equal compare will catch it and the
  divergence becomes a correctness PR. The rebase-time invariant
  (NEON keeps arithmetic shift, AVX2 keeps logical shift unless a
  delta surfaces) is unchanged.

### 0039 — `readability-function-size` NOLINT sweep (ADR-0146)

- **ADR**: [ADR-0146](adr/0146-nolint-sweep-function-size.md)
- **Touches**:
  - `libvmaf/src/dict.c`
  - `libvmaf/src/picture.c`
  - `libvmaf/src/picture_pool.c`
  - `libvmaf/src/predict.c`
  - `libvmaf/src/libvmaf.c`
  - `libvmaf/src/output.c`
  - `libvmaf/src/read_json_model.c`
  - `libvmaf/src/feature/feature_extractor.c`
  - `libvmaf/src/feature/feature_collector.c`
  - `libvmaf/src/feature/iqa/convolve.c`
  - `libvmaf/src/feature/iqa/ssim_tools.c`
  - `libvmaf/src/feature/x86/vif_statistic_avx2.c`
- **Invariant**: every `readability-function-size` NOLINT suppression
  has been replaced by a set of small `static` (or `static inline`,
  for the SIMD / IQA files) helpers. The helper names are stable
  interfaces the surrounding code depends on (e.g.
  `iqa_convolve_1d_separable`, `iqa_convolve_2d`,
  `ssim_compute_stats`, `ssim_workspace_alloc` / `_free`,
  `vif_stat_simd8_compute` / `_reduce`, `struct vif_simd8_lane`,
  `read_pictures_extractor_loop`, `read_pictures_post_extractor`,
  `read_pictures_validate_and_prep`,
  `read_pictures_update_prev_ref`). Upstream Netflix has no
  equivalent helpers; rebases touching any of these files will
  conflict against the fork's split shape.
- **On upstream sync**:
  - If upstream lands a different decomposition of `_iqa_convolve`
    or `_iqa_ssim`, prefer upstream's shape only if it keeps the
    ADR-0138 / ADR-0139 bit-exactness invariants (single-rounded
    float mul → widen to double → double add; per-lane scalar-float
    reduction through aligned temp buffer). Otherwise keep the
    fork's split and re-document the divergence here.
  - The fork renamed `_calc_scale` → `iqa_calc_scale` to clear the
    `bugprone-reserved-identifier` check. If upstream modifies
    `_calc_scale`, keep the fork's name and port the behavioural
    change.
  - `model_collection_parse_loop` writes directly to `cfg_name`
    rather than through `c->name` — if upstream ever rewrites
    `model_collection_parse`, preserve the direct write (it's what
    lets the param stay non-const without a NOLINT).
- **Re-test on rebase** (x86, any libsvm-less host):

  ```bash
  ninja -C build && meson test -C build
  for mask in 0 255; do
    VMAF_CPU_MASK=$mask ./build/tools/vmaf \
      --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
      --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
      --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
      -m version=vmaf_v0.6.1 -o /tmp/vmaf_$mask.xml
  done
  diff <(grep -v fyi /tmp/vmaf_0.xml) <(grep -v fyi /tmp/vmaf_255.xml)
  # expect exit 0 (Netflix-golden-pair VMAF bit-identical scalar vs SIMD)
  ```

  Also run `clang-tidy -p build` on every file in **Touches**;
  expect zero warnings.
- **Follow-up T7-6**: decide whether to rename the `_iqa_*` API
  surface (convolve / ssim / decimate / img_filter / filter_pixel /
  get_pixel) across all callers to clear the remaining
  `bugprone-reserved-identifier` suppressions in `ssim.c`,
  `ms_ssim.c`, `float_ms_ssim.c`. Out of scope here.

### 0040 — Thread-pool job recycling + inline data buffer (ADR-0147)

- **ADR**: [ADR-0147](adr/0147-thread-pool-job-pool.md)
- **Touches**: [`libvmaf/src/thread_pool.c`](../libvmaf/src/thread_pool.c)
- **Invariants**:
  1. `VmafThreadPoolJob` carries a fixed-size `char inline_data[64]`
     buffer. Payloads ≤ 64 bytes go through
     `memcpy(job->inline_data, data, data_sz)` +
     `job->data = job->inline_data`; payloads > 64 bytes take the
     legacy `malloc` path. The cleanup path MUST distinguish the
     two via `job->data != job->inline_data` — a naive
     `free(job->data)` would corrupt the slot. Enforced in
     `vmaf_thread_pool_job_clear_data`.
  2. `free_jobs` list is protected by the existing `queue.lock`;
     enqueue pops from it before `malloc`ing, runner recycles onto
     it after running a job. `vmaf_thread_pool_destroy` walks the
     list after `vmaf_thread_pool_wait` returns (all workers have
     exited → no lock needed). Any reorder that frees the queue
     lock before the `free_jobs` walk is a leak on shutdown.
  3. Fork's `void (*func)(void *data, void **thread_data)`
     signature + per-worker `VmafThreadPoolWorker` are fork-local;
     upstream Netflix #1464 has `func(void *data)`. Keep the fork's
     signature on any rebase — callers
     (`src/libvmaf.c:threaded_enqueue_one` etc.) depend on the
     two-arg form.
- **On upstream sync**: Netflix PR #1464 is CLOSED (not merged) and
  bundles twelve unrelated optimizations. Only the thread-pool
  portion is ported here. If upstream ever reopens and merges #1464
  (or a successor), cherry-pick **only** the pool mechanics; reject
  the payload-signature changes, the ADM / VIF / predict.c pieces
  (they conflict with ADR-0138 / 0139 / 0142 bit-exactness and with
  T7-5 predict.c refactor), and the feature-collector capacity bump
  (fork already capped at 8 for a reason — see
  [`src/feature/feature_collector.c`][fc-src]).

[fc-src]: ../libvmaf/src/feature/feature_collector.c

- **Re-test on rebase** (x86, any libsvm-less host):

  ```bash
  ninja -C build && meson test -C build
  for threads in 1 4; do
    for mask in 0 255; do
      VMAF_CPU_MASK=$mask ./build/tools/vmaf \
        --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
        --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
        --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
        -m version=vmaf_v0.6.1 --threads $threads -o /tmp/vmaf_${threads}_${mask}.xml
    done
  done
  # Expect bit-identical scores (attribute order may differ across
  # --threads 1 vs --threads 4 because feature-collector emits in
  # insertion order; the numeric values match).
  diff <(grep -v fyi /tmp/vmaf_4_0.xml) <(grep -v fyi /tmp/vmaf_4_255.xml)
  # expect exit 0 (scalar vs SIMD threaded)
  ```

  Also run `clang-tidy -p build libvmaf/src/thread_pool.c` — expect
  zero warnings. Re-run the 500 000-job micro-benchmark from
  ADR-0147 §Decision if performance is under investigation.

### 0041 — IQA reserved-identifier rename + cleanup (ADR-0148)

- **ADR**: [ADR-0148](adr/0148-iqa-rename-and-cleanup.md)
- **Touches**: 21 files across `libvmaf/src/feature/`
  (`iqa/{convolve,decimate,ssim_tools}.{c,h}`,
  `iqa/ssim_simd.h`, `ssim.c`, `integer_ssim.c`,
  `ms_ssim.c`, `ms_ssim_decimate.h`, `float_ssim.c`,
  `float_ms_ssim.c`, `x86/convolve_avx2.{c,h}`,
  `x86/convolve_avx512.{c,h}`, `arm64/convolve_neon.{c,h}`,
  `AGENTS.md`) plus `libvmaf/test/test_iqa_convolve.c`.
- **Invariants**:
  1. Every `_iqa_*` / `_kernel` / `_ssim_int` / `_map_reduce`
     / `_map` / `_reduce` / `_context` / `_ms_ssim_*` /
     `_ssim_*` / `_alloc_buffers` / `_free_buffers` symbol
     and the four underscore-prefixed header guards
     (`_CONVOLVE_H_`, `_DECIMATE_H_`, `_SSIM_TOOLS_H_`,
     `__VMAF_MS_SSIM_DECIMATE_H__`) is renamed to its
     non-reserved spelling. The fork's IQA surface no longer
     uses C's reserved-identifier name space.
  2. The `clang-analyzer-security.ArrayBound` NOLINT bracket
     in `ssim_accumulate_row` and `ssim_reduce_row_range`
     (integer_ssim.c) is load-bearing — the inner kernel-loop
     `k_min` / `k_max` clamping is provably correct
     (`k_min = max(0, hkernel_offs - x)`,
     `k_max = min(hkernel_sz, hkernel_sz - (x + hkernel_offs - w + 1))`)
     but the analyzer can't follow it across helper
     boundaries. Do not collapse the bracket.
  3. The `clang-analyzer-unix.Malloc` NOLINT bracket in
     `test_iqa_convolve.c` (`check_simd_variant`,
     `check_case`) is intentional — test exits process on
     failure path; small allocations leak by design at test
     end. Do not refactor to free-on-exit.
  4. The cross-TU NOLINT pattern on `compute_ssim`
     (ssim.c) and `compute_ms_ssim` (ms_ssim.c) — clang-tidy
     `misc-use-internal-linkage` runs per-TU and can't see
     the header bridge to `float_ssim.c` / `float_ms_ssim.c`.
     Keep the inline justification comment.
- **On upstream sync**:
  - The Netflix upstream IQA library (`tjdistler/iqa`) has
    been effectively abandoned (last meaningful commit
    pre-2020). Future rebases will conflict on every renamed
    symbol; drop the underscore-prefix on each conflict and
    mirror the fork's `iqa_*` naming.
  - If upstream Netflix/vmaf ever reincorporates the IQA
    naming wholesale, prefer the fork's spellings — this PR
    is a one-shot mechanical rename with no semantic content.
- **Re-test on rebase**:

  ```bash
  ninja -C build && meson test -C build
  for mask in 0 255; do
    VMAF_CPU_MASK=$mask ./build/tools/vmaf \
      --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
      --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
      --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
      -m version=vmaf_v0.6.1 \
      --feature float_ssim --feature float_ms_ssim \
      -o /tmp/iqa_$mask.xml
  done
  diff <(grep -v fyi /tmp/iqa_0.xml) <(grep -v fyi /tmp/iqa_255.xml)
  # expect exit 0 (bit-identical scalar vs SIMD on float_ssim/ms_ssim)
  ```

  Also run `clang-tidy -p build` on every touched file
  (excluding `arm64/`); expect zero warnings.

### 0042 — Port Netflix #1376 — FIFO-hang fix via Semaphore (ADR-0149)

- **ADR**: [ADR-0149](adr/0149-port-netflix-1376-fifo-semaphore.md)
- **Upstream commit**: Netflix PR
  [#1376](https://github.com/Netflix/vmaf/pull/1376),
  head `1c06ca4f1bb5da38b54db075a27c35ba8ea9d7b7` (OPEN upstream
  as of 2026-04-24).
- **Touches**:
  - [`python/vmaf/core/executor.py`](../python/vmaf/core/executor.py)
    — base `Executor` class + `ExternalVmafExecutor`-style
    subclass; delete `_wait_for_workfiles` / `_wait_for_procfiles`
    polling loops; rewrite `_open_{work,proc}files_in_fifo_mode`
    around `multiprocessing.Semaphore(0)`; add `open_sem=None`
    kwarg to every `_open_{ref,dis}_{work,proc}file` and to the
    `_open_workfile` staticmethod; drop unused
    `from time import sleep`.
  - [`python/vmaf/core/raw_extractor.py`](../python/vmaf/core/raw_extractor.py)
    — `AssetExtractor` + `DisYUVRawVideoExtractor`; add
    `open_sem=None` to `_open_{ref,dis}_workfile` overrides
    (release on entry since these are no-ops); delete
    `_wait_for_workfiles` overrides; drop unused
    `from time import sleep`.
- **Fork carve-outs** (load-bearing on rebase):
  1. **`python/vmaf/__init__.py:__version__`** stays `"3.0.0"` —
     do NOT port upstream's bump to `"4.0.0"`. The fork tracks
     its own versioning (`v3.x.y-lusoris.N`) per
     [ADR-0025](adr/0025-copyright-handling-dual-notice.md).
  2. **`from time import sleep` is dropped from both files** —
     upstream leaves the import in place (unused after their
     patch); the fork removes it because ADR-0141 touched-file
     rule requires ruff F401 clean.
  3. **Upstream typo preserved**: the subclass warning message
     contains "to be created to be created". Comments note the
     typo inline; do not silently fix on rebase — it's upstream-
     authored and project policy is verbatim port.
- **On upstream sync**: upstream PR #1376 is still OPEN. When it
  merges, re-diff against the merged form; the touched hunks
  should be conflict-free because the fork now carries the same
  shape. Re-check whether upstream fixed the "to be created to
  be created" typo; if so, adopt the fix (it becomes a simple
  string update).
- **Re-test**:

  ```bash
  python3 -m py_compile python/vmaf/core/executor.py \
                         python/vmaf/core/raw_extractor.py
  ruff check python/vmaf/core/executor.py python/vmaf/core/raw_extractor.py
  black --check python/vmaf/core/executor.py python/vmaf/core/raw_extractor.py
  # all silent

  # No FIFO-mode unit test in the tree; end-to-end harness
  # exercise (needs libsvm + ffmpeg + fixtures) goes via
  #   make test-netflix-golden
  # which doesn't exercise fifo_mode path but does verify the
  # refactor didn't break executor.py imports.
  ```

### 0043 — Port Netflix #1472 — CUDA on Windows MSYS2/MinGW (ADR-0150)

- **ADR**: [ADR-0150](adr/0150-port-netflix-1472-cuda-windows.md)
- **Upstream commits**: Netflix PR
  [#1472](https://github.com/Netflix/vmaf/pull/1472) —
  `15745cdf` (portability) + `b7b65e64` (meson plumbing). Both
  OPEN upstream as of 2026-04-24.
- **Touches**:
  - [`libvmaf/src/cuda/common.h`](../libvmaf/src/cuda/common.h)
    — drop `<pthread.h>` include; rename reserved header guard
    `__VMAF_SRC_CUDA_COMMON_H__` → `VMAF_SRC_CUDA_COMMON_INCLUDED`.
  - [`libvmaf/src/cuda/cuda_helper.cuh`](../libvmaf/src/cuda/cuda_helper.cuh)
    — `#ifdef DEVICE_CODE` guard around `<cuda.h>` vs
    `<ffnvcodec/dynlink_loader.h>`.
  - [`libvmaf/src/picture.h`](../libvmaf/src/picture.h) —
    `#ifdef DEVICE_CODE` guard around `<cuda.h>` +
    forward-declare `VmafCudaState` vs `<ffnvcodec/*>` + full
    `libvmaf_cuda.h`; rename reserved header guard.
  - [`libvmaf/src/feature/integer_adm.h`](../libvmaf/src/feature/integer_adm.h)
    — updated comment above `dwt_7_9_YCbCr_threshold` table
    noting the fork's positional-initializer shape vs upstream's
    `#ifndef __CUDACC__` shape (see §Fork carve-outs).
  - `libvmaf/src/feature/cuda/integer_adm/{adm_cm,adm_csf,adm_csf_den,adm_decouple,adm_dwt2}.cu`
    — `#ifndef DEVICE_CODE` guard around
    `#include "feature_collector.h"`.
  - [`libvmaf/src/meson.build`](../libvmaf/src/meson.build) —
    Windows nvcc plumbing (+70 LoC under
    `host_machine.system() == 'windows'`): `vswhere`-based
    `cl.exe` discovery, MSVC + Windows SDK include path
    injection, CUDA version detection via `nvcc --version`,
    `nvcc_ccbin_flags` + `nvcc_host_includes` threaded through
    every `custom_target` that invokes nvcc.
- **Fork carve-outs** (load-bearing on rebase):
  1. **`integer_adm.h` uses positional initializers**, NOT
     upstream's `#ifndef __CUDACC__` wrap. Both shapes resolve
     the MSVC/nvcc C++-designated-initializer issue; the
     positional form is C++-portable and keeps the table
     available to future `.cu` consumers. Keep the fork's form
     on rebase.
  2. **`cuda_static_lib` keeps `dependencies : [pthread_dependency]`**.
     Upstream drops it; the fork needs it because `ring_buffer.c`
     (built as part of `cuda_static_lib`) `#include`s
     `<pthread.h>` directly. On rebase: keep the fork's version.
  3. **`meson.build` gencode coverage block**: the fork's
     ADR-0122 explicit cubin list (sm_75/80/86/89 + compute_80
     PTX) sits after the new upstream nvcc-detect block. On
     rebase, re-assemble the same merged order: nvcc-detect
     first, then gencode coverage (both host-independent).
  4. **Header guards**: `_INCLUDED` spellings are fork-local
     (ADR-0148 precedent). Upstream keeps reserved
     `__VMAF_SRC_*_H__` spellings. On rebase, keep `_INCLUDED`.
- **On upstream sync**: PR #1472 is still OPEN. When merged,
  re-diff the three conflict-resolved hunks against upstream's
  final form. Keep fork's version on the four carve-outs above
  unless upstream meaningfully reshapes those regions.
- **Re-test on rebase** (Linux host with CUDA toolkit):

  ```bash
  meson setup libvmaf libvmaf/build-cuda \
      -Denable_cuda=true -Denable_nvcc=true -Denable_sycl=false
  ninja -C libvmaf/build-cuda && meson test -C libvmaf/build-cuda
  # Expect 6 .fatbin files generated + CLI linked + 35/35 tests pass.
  ```

  Windows validation is operator-driven — CI does not yet have a
  Windows + MSYS2 + MinGW + MSVC BuildTools + CUDA runner
  (tracked as T7-3 in `.workingdir2/OPEN.md`).
- **Prerequisites note** (Windows only): `nv-codec-headers` must
  be built from git master commit `876af32` or later. The
  release tag `n13.0.19.0` is missing `cuMemFreeHost`,
  `cuStreamCreateWithPriority`, `cuLaunchHostFunc`, and other
  `CudaFunctions` members libvmaf uses. Pre-existing issue, not
  scope of this port.

### 0058 — `libvmaf.pc` Cflags leak fix (ADR-0200)

- **ADR**: [ADR-0200](adr/0200-volk-priv-remap-pkgconfig-leak-fix.md);
  bug-fix follow-up to entry 0057.
- **Upstream source**: fork-local. Netflix has no Vulkan backend.
- **Touches**:
  - [`libvmaf/subprojects/packagefiles/volk/meson.build`](../libvmaf/subprojects/packagefiles/volk/meson.build)
    — drops `-include volk_priv_remap.h` from `volk_dep.compile_args`;
    keeps `-DVK_NO_PROTOTYPES`.
  - [`libvmaf/src/vulkan/meson.build`](../libvmaf/src/vulkan/meson.build)
    — pulls `volk_priv_remap_h_path` from the volk subproject and
    appends `['-include', <path>]` to `vmaf_cflags_common` (private
    `c_args:` on libvmaf's `library()` call).
- **Invariants** (load-bearing):
  1. **`-include` MUST stay off `volk_dep.compile_args`** — otherwise
     it leaks into static `libvmaf.pc` Cflags. Test on rebase:
     `meson setup ... -Ddefault_library=static -Denable_vulkan=enabled`,
     then `grep Cflags meson-private/libvmaf.pc` — must NOT contain
     `volk_priv_remap` or any build-dir absolute path.
  2. **`-include` MUST be applied to libvmaf's compile** — every
     libvmaf TU that calls volk's `vk*` API needs the rename macros
     active. The `vmaf_cflags_common` injection covers this for
     all libvmaf sub-libraries (libvmaf_feature, libvmaf_cpu, etc.).
  3. **The path comes from `subproject('volk').get_variable(...)`,
     not from a hardcoded string** — survives volk wrap version bumps.
- **On upstream sync**: zero upstream interaction.
- **Re-test on rebase / volk wrap bump**:

  ```bash
  meson setup build-vk-static-test libvmaf -Denable_vulkan=enabled \
      -Denable_cuda=false -Denable_sycl=false -Ddefault_library=static
  ninja -C build-vk-static-test src/libvmaf.a
  grep Cflags build-vk-static-test/meson-private/libvmaf.pc
  # Expected: no `volk_priv_remap` substring, no build-dir absolute path
  ```

### 0057 — Volk `vk*` priv-remap for static-archive builds (ADR-0198)

- **ADR**: [ADR-0198](adr/0198-volk-priv-remap-static-archive.md);
  follow-up to [ADR-0185](adr/0185-vulkan-hide-volk-symbols.md).
- **Upstream source**: fork-local. Netflix/vmaf has no Vulkan
  backend.
- **Touches**:
  - [`libvmaf/subprojects/packagefiles/volk/meson.build`](../libvmaf/subprojects/packagefiles/volk/meson.build)
    — overlay applied on top of the upstream volk wrap. Adds a
    `custom_target` that runs `gen_priv_remap.py` to produce
    `volk_priv_remap.h` from the upstream `volk.h`, and wires
    `-include` of the generated header into `volk.c`'s `c_args`
    and `volk_dep`'s `compile_args`.
  - [`libvmaf/subprojects/packagefiles/volk/gen_priv_remap.py`](../libvmaf/subprojects/packagefiles/volk/gen_priv_remap.py)
    — fork-added generator script (regex against
    `extern PFN_vkXxx vkXxx;` declarations).
- **Invariants** (load-bearing):
  1. **Force-include must propagate to every libvmaf TU pulling in
     `volk_dep`** — verified via meson dep graph. Removing the
     `-include` from `compile_args` re-introduces the static-link
     multi-def cascade.
  2. **Generator regex matches every `vk*` PFN declaration in
     `volk.h`** — confirmed for volk-1.4.341 (`784` declarations,
     `784` remaps). Bumping the volk wrap version: re-run the
     generator (it's a configure-time custom target, so it's
     automatic) and confirm the rename count printed to stdout
     matches the count of `^extern PFN_vk` lines in the new
     `volk.h`.
  3. **The renamed symbols use the `vmaf_priv_` prefix** — chosen
     to match no upstream Netflix or Vulkan SDK identifier. Don't
     rename to `_vk*` (collides with reserved-identifier C
     namespace) or `vkv_*` etc.
- **On upstream sync**: zero upstream interaction. The volk wrap
  is a libvmaf-managed subproject; Netflix doesn't ship a Vulkan
  backend.
- **Re-test on rebase / after any volk wrap bump**:

  ```bash
  meson setup build-vk-static libvmaf -Denable_vulkan=enabled \
      -Denable_cuda=false -Denable_sycl=false \
      -Ddefault_library=static
  ninja -C build-vk-static src/libvmaf.a
  test "$(nm build-vk-static/src/libvmaf.a 2>/dev/null \
            | grep -cE '^[0-9a-f]* (T|D|B|R) vk[A-Z]')" = "0" \
      && echo OK
  ```

  (Followed by the BtbN-style link reproducer in the ADR
  References section.)

### 0056 — SSIMULACRA 2 snapshot gate + fp-contract-off split (ADR-0164)

- **ADR**: [ADR-0164](adr/0164-ssimulacra2-snapshot-gate.md)
- **Upstream source**: fork-local. Netflix/vmaf has no SSIMULACRA 2.
- **Touches**:
  - [python/test/ssimulacra2_test.py](../python/test/ssimulacra2_test.py)
    — new fork-added Python test. Uses `subprocess.call` against
    `ExternalProgram.vmafexec` with `--feature ssimulacra2`; parses
    the `--json` output; asserts pooled + per-frame scores.
- **Invariants** (load-bearing):
  1. **Pinned values are CPU-only** — generated on master HEAD
     after PR #100 merge. Re-generate if the scalar or any SIMD
     path changes semantically (which per ADR-0161/0162/0163's
     bit-exactness contract, it shouldn't — any bit-exact refactor
     leaves pinned values unchanged).
  2. **Tolerance is 4 decimal places (`places=4`)** — matches
     1e-4. The CPU paths are bit-exact so actual drift should be
     0; the tolerance is defensive.
  3. **`-ffp-contract=off` everywhere in the ssimulacra2 pipeline**:
     `libvmaf_ssimulacra2_static_lib` (scalar extractor),
     `x86_ssimulacra2_avx2_lib`, `x86_ssimulacra2_avx512_lib`, and
     `arm64_ssimulacra2_lib` (from ADR-0161). All four split out
     of their umbrella libs so other extractors keep upstream's
     default FMA policy. **Without this** the CI GCC/clang hosts
     drifted ~2e-4 from my AVX-512 authoring host — GCC 10+
     defaults `-ffp-contract=fast` on x86 with `-mfma` and on
     aarch64, fusing `a*b+c` in scalar glue around the SIMD
     calls. Do NOT remove any of these carve-outs on rebase.
  4. **Fixtures are already-checked-in** — `src01_hrc00/01_576x324`
     is also the primary Netflix golden fixture; the 160×90
     derived one stresses the sub-176 pyramid-termination path.
  5. **Do NOT modify the Netflix golden assertions in
     quality_runner_test.py et al.** — those are upstream-pinned.
     This test is a SEPARATE file that adds fork-specific scores.
- **On upstream sync**: no upstream interaction. If Netflix adopts
  SSIMULACRA 2 in the future, cross-reference against their
  pinning if they add one.
- **Re-test on rebase / after any ssimulacra2 change**:

  ```bash
  cd python && python -m pytest test/ssimulacra2_test.py -v   # 2/2
  ```

- **Follow-ups**:
  - Cross-reference gate against libjxl `tools/ssimulacra2` when
    `ssimulacra2_rs` cargo install is fixed.
  - Expand fixture coverage if new YUV test assets land.

### 0055 — SSIMULACRA 2 `picture_to_linear_rgb` SIMD (ADR-0163)

- **ADR**: [ADR-0163](adr/0163-ssimulacra2-ptlr-simd.md)
- **Upstream source**: fork-local. Netflix/vmaf has no SSIMULACRA 2.
- **Touches**:
  - [ssimulacra2_avx2.{c,h}](../libvmaf/src/feature/x86/ssimulacra2_avx2.c)
    — new `ssimulacra2_picture_to_linear_rgb_avx2` + helpers
    (`read_plane_scalar_s2`, `srgb_to_linear_lane_avx2`,
    `compute_matrix_coefs`).
  - [ssimulacra2_avx512.{c,h}](../libvmaf/src/feature/x86/ssimulacra2_avx512.c)
    — 16-wide AVX-512 port.
  - [ssimulacra2_neon.{c,h}](../libvmaf/src/feature/arm64/ssimulacra2_neon.c)
    — 4-wide aarch64 port.
  - [ssimulacra2.c](../libvmaf/src/feature/ssimulacra2.c) — new
    `ptlr_fn` field in `Ssimu2State`; dispatch wrapper
    `convert_picture_to_linear_rgb` unpacks `VmafPicture` into
    `simd_plane_t[3]`; init assigns AVX2/AVX-512/NEON pointers.
  - [ssimulacra2_simd_common.h](../libvmaf/src/feature/ssimulacra2_simd_common.h)
    — new shared header declaring `simd_plane_t`. Decouples SIMD
    TUs from `VmafPicture` type.
  - [test_ssimulacra2_simd.c](../libvmaf/test/test_ssimulacra2_simd.c)
    — new `test_ptlr_420_8`, `test_ptlr_420_10`, `test_ptlr_444_8`,
    `test_ptlr_444_10`, `test_ptlr_422_8` subtests + scalar
    references `ref_read_plane`, `ref_srgb_to_linear`,
    `ref_picture_to_linear_rgb`.
- **Invariants** (load-bearing):
  1. **Scalar-order matmul** — `G = Yn + cb_g * Un + cr_g * Vn`
     chained left-to-right in all three SIMD TUs. Regression test
     catches reordering drift (~1 ulp).
  2. **Per-lane scalar `powf`** — vector polynomial approximation
     would drift scalar bit-exactness. Do not replace the lane
     spill/reload pattern with a vector libm.
  3. **`simd_plane_t` layout** — `{data, stride, w, h}` ordering
     assumed by all three SIMD TUs. The dispatch wrapper builds
     this from `VmafPicture` fields; layout must match.
  4. **Bounds clamping in `read_plane_scalar_*`** mirrors scalar
     reference verbatim (`if (sx < 0) sx = 0; if (sx >= pw) sx =
     pw-1;` etc.). Do not simplify — removes per-lane safety at
     plane edges.
  5. **Arbitrary chroma ratios** fall through to the `int64_t`
     multiplication branch. Don't remove it — SSIMULACRA 2 is
     supposed to accept non-standard ratios gracefully.
- **On upstream sync**: no upstream interaction. If Netflix adopts
  SSIMULACRA 2 in the future and provides a SIMD YUV→RGB path,
  diff against the fork's — preserve the bit-exactness contract
  unless ADR-0142 Netflix-authority carve-out opens.
- **Re-test on rebase**:

  ```bash
  ninja -C build && build/test/test_ssimulacra2_simd     # 11/11
  ninja -C build-aarch64 && \
    qemu-aarch64-static -L /usr/aarch64-linux-gnu/ \
      build-aarch64/test/test_ssimulacra2_simd            # 11/11
  ```

- **Follow-ups**:
  - T3-3 SSIMULACRA 2 snapshot-JSON regression test — still
    pending (gated on `tools/ssimulacra2` availability).
  - SSIMULACRA 2 now has **zero scalar hot paths**. T3-1 closes in
    full with phases 1+2+3 (ADR-0161, 0162, 0163).

### 0054 — SSIMULACRA 2 FastGaussian IIR blur SIMD (ADR-0162)

- **ADR**: [ADR-0162](adr/0162-ssimulacra2-iir-blur-simd.md)
- **Upstream source**: fork-local. No SSIMULACRA 2 extractor in
  upstream Netflix/vmaf.
- **Touches**:
  - [ssimulacra2_avx2.{c,h}](../libvmaf/src/feature/x86/ssimulacra2_avx2.c)
    — new `ssimulacra2_blur_plane_avx2` + 2 helpers (`hblur_8rows_avx2`,
    `vblur_simd_8cols_avx2`).
  - [ssimulacra2_avx512.{c,h}](../libvmaf/src/feature/x86/ssimulacra2_avx512.c)
    — 16-wide port.
  - [ssimulacra2_neon.{c,h}](../libvmaf/src/feature/arm64/ssimulacra2_neon.c)
    — 4-wide aarch64 port, uses `vsetq_lane_f32` in place of gather.
  - [ssimulacra2.c](../libvmaf/src/feature/ssimulacra2.c) — adds
    `blur_fn` function pointer to `Ssimu2State`, dispatch in
    `init_simd_dispatch()`, call-site in `blur_3plane`.
  - [test_ssimulacra2_simd.c](../libvmaf/test/test_ssimulacra2_simd.c)
    — new `test_blur` + scalar reference (`ref_blur_plane`,
    `ref_fast_gaussian_1d`).
- **Invariants** (load-bearing):
  1. **Row-batching lane layout** — horizontal pass lane `i`
     MUST hold row `(y_base + i)`. Gather index vector entries
     are `(y_base + i) * w` (stride-w). Changing this breaks
     bit-exactness vs scalar.
  2. **Scalar left-to-right summation order** — `n2_k * sum -
     d1_k * prev1_k - prev2_k` chained sequentially; `o0 + o1 + o2`
     at output time is `(o0 + o1) + o2`. Changing to
     `(o0 + o2) + o1` or `o0 + (o1 + o2)` will drift ~1 ulp and
     the regression test catches it.
  3. **`col_state` is 6 * w contiguous floats** — layout is
     `[prev1_0 | prev1_1 | prev1_2 | prev2_0 | prev2_1 | prev2_2]`.
     SIMD loads assume this layout; changing field order requires
     updating all three SIMD TUs in lockstep with `blur_plane`.
  4. **NEON lane-set pattern** — aarch64 has no gather intrinsic;
     4 explicit `vsetq_lane_f32` calls per input vector. Do not
     replace with a `ld1 {v.s}[lane]`-style pseudo-gather without
     re-verifying bit-exactness.
  5. **Scalar tail in vertical pass** matches scalar reference
     body verbatim. Any deviation breaks `memcmp` equality on
     widths that aren't multiples of the SIMD width.
- **On upstream sync**: no upstream interaction. If Netflix adopts
  SSIMULACRA 2 in the future and provides their own IIR blur SIMD,
  diff against the fork's and preserve the bit-exactness contract
  unless an ADR-0142 Netflix-authority carve-out is opened.
- **Re-test on rebase**:

  ```bash
  meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build test_ssimulacra2_simd  # 6/6
  # aarch64:
  ninja -C build-aarch64
  qemu-aarch64-static -L /usr/aarch64-linux-gnu/ \
    build-aarch64/test/test_ssimulacra2_simd  # 6/6
  ```

- **Follow-ups**:
  - `picture_to_linear_rgb` SIMD — last scalar hot path in the
    extractor. 2 calls / frame. Low ROI but mechanical.
  - T3-3 SSIMULACRA 2 snapshot-JSON regression test — still pending.

### 0053 — SSIMULACRA 2 SIMD bit-exact ports (ADR-0161)

- **ADR**: [ADR-0161](adr/0161-ssimulacra2-simd-bitexact.md)
- **Upstream source**: fork-local. Upstream Netflix/vmaf has no
  SSIMULACRA 2 extractor at all (fork-added in ADR-0130).
- **Touches**:
  - [ssimulacra2_avx2.c](../libvmaf/src/feature/x86/ssimulacra2_avx2.c)
    / [.h](../libvmaf/src/feature/x86/ssimulacra2_avx2.h) — 5 AVX2
    kernels + per-lane `cbrtf` helper.
  - [ssimulacra2_avx512.c](../libvmaf/src/feature/x86/ssimulacra2_avx512.c)
    / [.h](../libvmaf/src/feature/x86/ssimulacra2_avx512.h) — 5
    AVX-512 kernels; mechanical 16-wide widening of the AVX2 path.
  - [ssimulacra2_neon.c](../libvmaf/src/feature/arm64/ssimulacra2_neon.c)
    / [.h](../libvmaf/src/feature/arm64/ssimulacra2_neon.h) — 5
    NEON kernels; 4-wide aarch64 mirror.
  - [ssimulacra2.c](../libvmaf/src/feature/ssimulacra2.c) — adds
    function-pointer dispatch fields to `Ssimu2State` +
    `init_simd_dispatch()` helper, calls go through the pointers.
  - [meson.build](../libvmaf/src/meson.build) — registers the
    three SIMD TUs in `x86_avx2_sources` / `x86_avx512_sources` /
    `arm64_sources`.
  - [test_ssimulacra2_simd.c](../libvmaf/test/test_ssimulacra2_simd.c)
    and `test/meson.build` — new bit-exact test harness.
- **Invariants** (load-bearing):
  1. **Byte-for-byte bit-exactness to scalar** on all 5 vectorised
     kernels under `FLT_EVAL_METHOD == 0`. Regression caught pre-
     merge: naïve pairing `(a+b)+(c+d)` vs scalar `((a+b)+c)+d`
     drifts by 1 ULP. Keep sequential scalar-order chains in all
     three SIMD TUs on rebase.
  2. **`cbrtf` is per-lane scalar libm**, not a polynomial. Any
     replacement with a vector cbrt would drift the ssimulacra2
     score and break the regression test. Keep the spill/reload
     pattern.
  3. **`ssim_map` / `edge_diff_map` reductions use the ADR-0139
     per-lane `double` scalar tail**. Do NOT SIMD-reduce float
     lanes then lift to double — summation order changes.
  4. **`downsample_2x2` deinterleave** uses ISA-appropriate ops:
     AVX2 `vshufps+vpermpd`, AVX-512 `vpermt2ps`, NEON
     `vuzp1q_f32`+`vuzp2q_f32`. After deinterleave, sum order is
     `((r0e+r0o)+r1e)+r1o` matching scalar.
  5. **`#pragma STDC FP_CONTRACT OFF`** at every TU header.
     Ignored by aarch64 GCC (non-fatal `-Wunknown-pragmas`); kept
     for portability (clang, MSVC).
  6. **IIR blur + `picture_to_linear_rgb` stay scalar** in this PR.
     Follow-up PRs target these; when they land, re-verify
     bit-exactness via `test_ssimulacra2_simd` expansion.
  7. Runtime dispatch order: AVX-512 > AVX2 on x86; NEON on
     aarch64; scalar fallback. Preserve on rebase.
- **On upstream sync**:
  - Upstream has no SSIMULACRA 2 extractor; nothing to merge.
  - If Netflix adopts SSIMULACRA 2 in the future, diff their
    implementation against the fork's scalar + SIMD TUs; keep
    the fork's bit-exactness contract absent a specific
    Netflix-authority carve-out ADR.
- **Re-test on rebase**:

  ```bash
  meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build test_ssimulacra2_simd   # 5/5
  clang-tidy -p build libvmaf/src/feature/x86/ssimulacra2_avx2.c \
                       libvmaf/src/feature/x86/ssimulacra2_avx512.c
  # aarch64:
  ninja -C build-aarch64
  qemu-aarch64-static -L /usr/aarch64-linux-gnu/ \
    build-aarch64/test/test_ssimulacra2_simd   # 5/5
  clang-tidy -p build-aarch64 \
    libvmaf/src/feature/arm64/ssimulacra2_neon.c
  ```

- **Follow-ups**:
  - **IIR blur vectorisation** (`blur_plane` vertical-pass
    column batching) — the biggest frame-level wallclock win.
  - **`picture_to_linear_rgb` per-lane `powf`** — lower ROI but
    mechanical.
  - **T3-3 SSIMULACRA 2 snapshot-JSON regression test** —
    ADR-0130 deferred; still pending.

### 0052 — `psnr_hvs` SIMD bit-exact ports (ADR-0159 AVX2, ADR-0160 NEON)

- **ADRs**: [ADR-0159](adr/0159-psnr-hvs-avx2-bitexact.md) (AVX2),
  [ADR-0160](adr/0160-psnr-hvs-neon-bitexact.md) (NEON sister port).
- **Upstream source**: fork-local. Upstream Netflix/vmaf has no
  psnr_hvs SIMD path.
- **Touches**:
  - [`libvmaf/src/feature/x86/psnr_hvs_avx2.c`](../libvmaf/src/feature/x86/psnr_hvs_avx2.c)
    — AVX2 TU.
  - [`libvmaf/src/feature/x86/psnr_hvs_avx2.h`](../libvmaf/src/feature/x86/psnr_hvs_avx2.h)
    — AVX2 header.
  - [`libvmaf/src/feature/arm64/psnr_hvs_neon.c`](../libvmaf/src/feature/arm64/psnr_hvs_neon.c)
    — NEON TU (sister port, ADR-0160).
  - [`libvmaf/src/feature/arm64/psnr_hvs_neon.h`](../libvmaf/src/feature/arm64/psnr_hvs_neon.h)
    — NEON header.
  - [`libvmaf/src/feature/third_party/xiph/psnr_hvs.c`](../libvmaf/src/feature/third_party/xiph/psnr_hvs.c)
    — add `PsnrHvsState` + runtime dispatch in `init()` (AVX2
    under `ARCH_X86`, NEON under `ARCH_AARCH64`) + scoped
    NOLINTBEGIN/END around the upstream Xiph scalar block (kept
    verbatim as the bit-exact reference).
  - [`libvmaf/src/meson.build`](../libvmaf/src/meson.build) — add
    `x86/psnr_hvs_avx2.c` to `x86_avx2_sources` and
    `arm64/psnr_hvs_neon.c` to `arm64_sources`.
  - [`libvmaf/test/test_psnr_hvs_avx2.c`](../libvmaf/test/test_psnr_hvs_avx2.c),
    [`libvmaf/test/test_psnr_hvs_neon.c`](../libvmaf/test/test_psnr_hvs_neon.c)
    — bit-exact unit tests (x86 and aarch64 respectively).
  - [`libvmaf/test/meson.build`](../libvmaf/test/meson.build) —
    register both tests under `enable_asm`, arch-gated.
- **Invariants** (load-bearing):
  1. **Bit-exactness to scalar**: every `od_coeff` (int32) and
     every final `psnr_hvs_{y,cb,cr,psnr_hvs}` value the AVX2
     path emits must be byte-identical to the scalar reference
     on the Netflix golden pairs. If a rebase introduces any
     pattern that breaks this (e.g. a floating-point horizontal
     reduce in the mask accumulator), the unit test
     `test_psnr_hvs_avx2` will fail — don't relax the
     assertions; fix the SIMD path.
  2. **DCT butterfly layout**: `butterfly → transpose →
     butterfly → transpose`. The transpose lives inside
     `od_bin_fdct8x8_avx2`. Do not move it.
  3. **Float accumulators stay scalar**: means / variances /
     mask / error accumulation in `calc_psnrhvs_avx2` use the
     same per-block scalar loop as scalar psnr_hvs — bit-exact
     by construction. Do not vectorize these with horizontal
     reductions without replicating ADR-0139's per-lane
     scalar-float reduction pattern. The cross-block error
     accumulator `ret` is threaded through `accumulate_error()`
     **by pointer**, not returned-then-summed: each of the 64
     per-coefficient contributions per block must hit the outer
     `ret` directly, matching scalar's inline `ret += ...` at
     `third_party/xiph/psnr_hvs.c` line 355. IEEE-754 float
     add is non-associative — summing into a local float and
     then adding the per-block total to `ret` changes the
     summation tree and drifts the Netflix golden by ~5.5e-5.
  4. **`#pragma STDC FP_CONTRACT OFF`** at the TU header
     disables FMA formation. Required: `fmaf(a, b, c)` can
     differ from `(a*b)+c` by 1 ulp, breaking bit-exactness.
     Do not remove the pragma; do not add `-ffp-contract=fast`
     to the build flags for this TU.
  5. **NOLINT suppressions are load-bearing** — each cites
     ADR-0141 inline (bit-exactness scalar-diff auditability
     for the 30-butterfly function, scalar float→double
     promotion for `sqrt`, extractor-registry extern linkage
     for `vmaf_fex_psnr_hvs`, upstream-Xiph scoped block for
     rebase parity).
- **On upstream sync**:
  - Upstream has no psnr_hvs SIMD as of 2026-04-24. Keep fork's
    version on conflict.
  - If upstream ever touches `psnr_hvs.c` for non-SIMD reasons
    (e.g. a masking-table update), rebase the AVX2 TU to match
    line-for-line and re-run `test_psnr_hvs_avx2` to confirm
    bit-exactness survives.
  - NEON follow-up PR is a sister port; its `arm64/psnr_hvs_neon.c`
    will mirror this ADR's invariants. On rebase, the two SIMD TUs
    must stay in lock-step with the scalar reference.
- **Re-test on rebase**:

  ```bash
  ninja -C build
  meson test -C build test_psnr_hvs_avx2
  # Expect: 5/5 subtests pass (DCT bit-exact on 3 random seeds +
  # delta + constant input).

  # CLI-level bit-exactness on Netflix golden (requires the YUV
  # fixtures in python/test/resource/yuv/):
  # VMAF_CPU_MASK=0    (scalar)
  # VMAF_CPU_MASK=255  (AVX2 enabled)
  # Diff per-frame psnr_hvs_{y,cb,cr,psnr_hvs} XML fields; expect
  # byte-identical across all 3 golden pairs.
  ```

### 0051 — Netflix#1486 motion updates verified present (ADR-0158)

- **ADR**: [ADR-0158](adr/0158-netflix-1486-motion-updates-verified-present.md)
- **Upstream source**: Netflix upstream PR
  [#1486](https://github.com/Netflix/vmaf/pull/1486) ("Port motion
  updates"), MERGED 2026-04-20 as commits `a44e5e6` (code) + `62f47d5`
  (Netflix golden updates).
- **Touches**: documentation-only; the actual code changes this
  ADR documents are already in the fork's master via earlier
  incremental motion3 / blend / five-frame-window commits.
- **Invariants** (load-bearing for future `/sync-upstream`):
  1. The `edge_8` mirror fix (`i_tap = height - (i_tap - height + 2)`)
     is present at `integer_motion.c:240`,
     `x86/motion_avx2.c:147`, `x86/motion_avx512.c:147`. If
     upstream's mirror line ever diverges again, this is the
     hunk to watch.
  2. The `motion_max_val` feature option is at
     `integer_motion.c:57,118-120` with default 10000.0 and
     `FEATURE_PARAM` flag. Upstream's default = fork's default;
     don't drift.
  3. `VMAF_integer_feature_motion3_score` output plumbing is in
     `integer_motion.c` + `alias.c`.
  4. Fork-local motion extensions (five-frame-window,
     moving-average, blend, fps_weight) are ADDITIONS on top of
     Netflix#1486. They are not upstream. Upstream changes to
     motion extractor internals may conflict with them — diff
     against `libvmaf/src/feature/integer_motion.c` on every
     rebase and check that the fork's `MIN(s->score *
     s->motion_fps_weight, s->motion_max_val)` invocations are
     preserved (lines ~409, ~503).
- **On upstream sync**: nothing to port from Netflix#1486 — it's
  absorbed. If a future upstream PR touches the same code paths,
  prefer upstream's version for the scalar/edge handling and the
  fork's version for the five-frame-window / blend extensions.
- **Re-test on rebase**:

  ```bash
  ninja -C build
  meson test -C build
  # Expect: 35/35 pass.

  # Verify the upstream markers are still in place after rebase:
  grep -n "height - (i_tap - height + 2)\|motion_max_val\|VMAF_integer_feature_motion3_score" \
      libvmaf/src/feature/integer_motion.c \
      libvmaf/src/feature/alias.c \
      libvmaf/src/feature/x86/motion_avx2.c \
      libvmaf/src/feature/x86/motion_avx512.c
  # Expect: matches at all 4 files. If any missing, the rebase
  # silently dropped the Netflix#1486 content — investigate.
  ```

### 0050 — CUDA preallocation memory leak fix + `vmaf_cuda_state_free` (ADR-0157)

- **ADR**: [ADR-0157](adr/0157-cuda-preallocation-leak-netflix-1300.md)
- **Upstream source**: Netflix upstream issue
  [#1300](https://github.com/Netflix/vmaf/issues/1300) (OPEN since
  2024; no maintainer fix as of 2026-04-24). User reports GPU
  memory rises monotonically across init/preallocate/fetch/close
  cycles.
- **Touches**:
  - [`libvmaf/include/libvmaf/libvmaf_cuda.h`](../libvmaf/include/libvmaf/libvmaf_cuda.h)
    — new public `vmaf_cuda_state_free()` API declaration.
  - [`libvmaf/src/cuda/common.c`](../libvmaf/src/cuda/common.c)
    — new `vmaf_cuda_state_free()` implementation;
    `vmaf_cuda_release()` now calls `cuda_free_functions()`;
    `vmaf_cuda_state_init()` gets an outer failure unwind;
    `init_with_primary_context()` releases the retained primary
    context on `fail_after_pop`.
  - [`libvmaf/src/cuda/ring_buffer.c`](../libvmaf/src/cuda/ring_buffer.c)
    — `vmaf_ring_buffer_close()` now unlocks + destroys the mutex
    before freeing.
  - [`libvmaf/test/test_cuda_preallocation_leak.c`](../libvmaf/test/test_cuda_preallocation_leak.c)
    — new GPU-gated reducer (10-cycle loop with full cleanup).
  - [`libvmaf/test/test_cuda_pic_preallocation.c`](../libvmaf/test/test_cuda_pic_preallocation.c),
    [`libvmaf/test/test_cuda_buffer_alloc_oom.c`](../libvmaf/test/test_cuda_buffer_alloc_oom.c)
    — add missing `vmaf_cuda_state_free()` + `vmaf_model_destroy()`
    calls after `vmaf_close()` in every test that allocates these.
  - [`libvmaf/test/meson.build`](../libvmaf/test/meson.build)
    — register the new reducer under `enable_cuda` guard.
- **Invariants** (load-bearing):
  1. Public contract: every caller of `vmaf_cuda_state_init()`
     MUST call `vmaf_cuda_state_free()` AFTER `vmaf_close()` on
     any VmafContext that imported the state. Informal
     `free(cu_state)` is a silent double-free hazard AFTER close
     (vmaf_close's vmaf_cuda_release already memset's + frees
     CudaFunctions internals; vmaf_cuda_state_free only frees the
     heap allocation itself).
  2. `vmaf_cuda_release()` frees `CudaFunctions` via a saved
     pointer AFTER the `memset`. Order matters — `memset` first
     so `cu_state->f` is zeroed in the caller's struct, then
     free via the saved local. Do not re-order.
  3. `vmaf_ring_buffer_close()` unlocks BEFORE destroying the
     mutex (POSIX requires the mutex be unlocked for destroy).
  4. The cold-start unwind in `init_with_primary_context` releases
     `cuDevicePrimaryCtxRetain`'s retained context if
     `cuStreamCreateWithPriority` fails.
  5. The ADR-0122 / ADR-0123 `is_cudastate_empty()` null-guards at
     the top of every public `vmaf_cuda_*` entry must continue to
     compose with the new `vmaf_cuda_state_free()` (which accepts
     NULL directly and doesn't call through to the CUDA API).
  6. The new free call order in callers is:
     `vmaf_close(vmaf)` → `vmaf_cuda_state_free(cu_state)` →
     `vmaf_model_destroy(model)`. Reversing the first two
     produces a use-after-free.
- **On upstream sync**:
  - Upstream has no `vmaf_cuda_state_free()` as of 2026-04-24.
    Keep the fork's version on any conflict. If upstream eventually
    lands the same API with a different spelling, prefer
    upstream's spelling and add a compat alias — but do not break
    the fork's ABI.
  - `vmaf_cuda_release()`'s `cuda_free_functions()` call is
    fork-local. On rebase, keep it.
  - The ring-buffer `pthread_mutex_unlock` + `pthread_mutex_destroy`
    pair is fork-local. On rebase, keep it.
  - If upstream refactors `VmafCudaState` ownership semantics
    (unlikely — their pattern has been "leaked state in a long-
    lived process is acceptable" historically), re-audit this
    ADR and the new public API.
- **Re-test on rebase**:

  ```bash
  ninja -C libvmaf/build-cuda
  meson test -C libvmaf/build-cuda
  # Expect: 40/40 pass including test_cuda_preallocation_leak.

  # ASan leak-check:
  cd libvmaf && meson setup build-asan-cuda \
      -Db_sanitize=address -Denable_cuda=true -Denable_sycl=false \
      --buildtype=debug
  ninja -C build-asan-cuda
  ASAN_OPTIONS='detect_leaks=1:leak_check_at_exit=1' \
      build-asan-cuda/test/test_cuda_preallocation_leak
  # Expect: 0 bytes leaked from libvmaf/src/* frames.
  # (~180 bytes in libcuda.so.1 is expected — driver's process-
  #  lifetime cuInit cache, does not grow per cycle.)
  ```

### 0049 — CUDA graceful error propagation (ADR-0156)

- **ADR**: [ADR-0156](adr/0156-cuda-graceful-error-propagation-netflix-1420.md)
- **Upstream source**: Netflix upstream issue
  [#1420](https://github.com/Netflix/vmaf/issues/1420) (OPEN as
  of 2026-04-24). Reports that two concurrent VMAF-CUDA
  processes crash the second one at `vmaf_cuda_buffer_alloc`
  due to `CHECK_CUDA(cuMemAlloc)` → `assert(0)` on OOM.
- **Touches**:
  - [`libvmaf/src/cuda/cuda_helper.cuh`](../libvmaf/src/cuda/cuda_helper.cuh)
    — redefined `CHECK_CUDA` family. New macros
    `CHECK_CUDA_GOTO` + `CHECK_CUDA_RETURN` + helper
    `vmaf_cuda_result_to_errno`. Old `assert(0)` semantics
    removed entirely.
  - [`libvmaf/src/cuda/common.c`](../libvmaf/src/cuda/common.c),
    [`libvmaf/src/cuda/picture_cuda.c`](../libvmaf/src/cuda/picture_cuda.c),
    [`libvmaf/src/libvmaf.c`](../libvmaf/src/libvmaf.c)
    — all `CHECK_CUDA(...)` sites converted; cleanup labels
    added where contexts / buffers were pushed / allocated.
  - [`libvmaf/src/feature/cuda/integer_motion_cuda.c`](../libvmaf/src/feature/cuda/integer_motion_cuda.c),
    [`integer_vif_cuda.c`](../libvmaf/src/feature/cuda/integer_vif_cuda.c),
    [`integer_adm_cuda.c`](../libvmaf/src/feature/cuda/integer_adm_cuda.c)
    — same conversion; 12 `static` helpers promoted `void → int`.
  - [`libvmaf/test/test_cuda_buffer_alloc_oom.c`](../libvmaf/test/test_cuda_buffer_alloc_oom.c)
    — new GPU-gated reducer.
  - [`libvmaf/test/meson.build`](../libvmaf/test/meson.build)
    — register new test under `enable_cuda` guard.
- **Invariants** (load-bearing):
  1. `CHECK_CUDA_GOTO` / `CHECK_CUDA_RETURN` must **never**
     call `assert(0)` or `abort()` on a CUDA error. Any
     regression back to the upstream abort-on-error
     semantics re-introduces Netflix#1420 and the NDEBUG
     footgun.
  2. Every `CHECK_CUDA_GOTO` target label must pop any
     previously-pushed CUDA context and free any
     partially-constructed buffers before returning the
     errno. The graceful path must not leak resources.
  3. `vmaf_cuda_result_to_errno` uses numeric `CUresult`
     values directly (0 / 1 / 2 / 3 / 4 / 101 / 201 / 400)
     so host TUs that don't include `<cuda.h>` can
     transitively consume the mapping via the inline
     function. If upstream renumbers `CUresult` enum
     values (historically stable — they've been fixed
     since CUDA 1.0), re-audit the switch.
  4. ADR-0122 / ADR-0123 `is_cudastate_empty(...)` guards
     at the top of every public `vmaf_cuda_*` entry point
     must stay — they run before the CUDA API is touched
     and compose cleanly with the new error propagation.
  5. Twelve `static` helper signatures in the feature
     extractors are `int`-returning (was `void`): any
     upstream-port that restores the `void` return silently
     regresses the error path.
- **On upstream sync**:
  - Upstream Netflix still uses `assert(0)` in
    `CHECK_CUDA` as of 2026-04-24. Keep the fork's macro
    definitions in `cuda_helper.cuh` on any upstream
    conflict — this file is fork-local behaviour.
  - If upstream eventually lands Netflix#1420 with a
    similar refactor, prefer the fork's version unless
    upstream's has identical semantics (no `assert(0)` /
    no `abort()` / translates `CUresult` to `-errno`).
    Re-verify `test_cuda_buffer_alloc_oom` after rebase.
  - If upstream adds new `CHECK_CUDA(...)` sites in a
    port, rewrite them to `CHECK_CUDA_GOTO` /
    `CHECK_CUDA_RETURN` as part of the port commit.
  - If upstream changes any of the 12 `static` helper
    signatures back to `void`, re-promote them to `int`
    during the merge.
- **Re-test on rebase**:

  ```bash
  ninja -C libvmaf/build-cuda
  meson test -C libvmaf/build-cuda
  # Expect: 39/39 pass including test_cuda_buffer_alloc_oom.

  # Reducer check — verify the OOM-to-errno path is live:
  meson test -C libvmaf/build-cuda test_cuda_buffer_alloc_oom -v
  # Expect subtests: request 1 TiB → -ENOMEM; request 0 bytes → 0.

  clang-tidy -p libvmaf/build-cuda --quiet \
      libvmaf/src/cuda/common.c \
      libvmaf/src/cuda/picture_cuda.c \
      libvmaf/src/feature/cuda/integer_motion_cuda.c \
      libvmaf/src/feature/cuda/integer_vif_cuda.c \
      libvmaf/src/feature/cuda/integer_adm_cuda.c \
      libvmaf/src/libvmaf.c
  # Expect exit 0 on every file.
  ```

### 0049 — `compute_motion` / `picture_copy` signature changes (b949cebf upstream port)

- **Upstream commit**: Netflix/vmaf b949cebf (feature/motion: port several feature extractor options)
- **Prerequisite commit**: Netflix/vmaf d3647c73 (picture_copy: add channel parameter)
- **PR**: upstream/port-b949cebf-motion

**Rebase-sensitive invariants:**

1. **`compute_motion` signature change** — `compute_motion()` in
   `libvmaf/src/feature/motion.c` / `motion.h` now takes an extra
   `int motion_decimate` parameter (the `motion_add_scale1` flag).
   Any new caller added in the fork that calls `compute_motion()` must
   pass this parameter. The SIMD integer motion callers
   (`motion_avx2.c`, `motion_avx512.c`) do NOT call `compute_motion()`
   — they use the SAD/convolution dispatch table directly and are
   unaffected.

2. **`vmaf_image_sad_c` signature change** — similarly gains
   `int motion_add_scale1`. Any caller in the fork must be updated.
   Currently only called from `compute_motion()` internally.

3. **`picture_copy` signature change** — gains `int channel` as the
   last parameter (0=Y, 1=U, 2=V). Every caller in the tree has been
   updated to pass `0` (luma). When adding new callers that need UV
   planes, pass `1` or `2`. The fork's CUDA/SYCL/Vulkan callers have
   been updated in this PR.

4. **Default behavior preserved** — all new options default to no-op
   values. `motion_add_scale1=false`, `motion_add_uv=false`,
   `motion_blend_factor=1.0`, `motion_fps_weight=1.0`,
   `motion_filter_size=5` (= DEFAULT_MOTION_FILTER_SIZE). Integer and
   float motion2 scores are bit-identical to pre-port baseline.

5. **`vif_scale_frame_s` dependency avoided** — the upstream b949cebf
   motion.c imports `vif_scale_frame_s` from vif_tools.h. The fork
   does not have this function yet (vif options chain is deferred,
   Research-0024 Strategy E). The bilinear downscaler for
   `motion_add_scale1` is implemented as local static functions in
   `motion.c` (`motion_scale_bilinear`, `motion_bilinear_interp`,
   `motion_mirror_f`). When upstream's vif options chain is eventually
   ported, reconcile by replacing these local functions with
   `vif_scale_frame_s`.

**Reproducer:**
```bash
# verify bit-exactness (default options, scores must be identical):
./libvmaf/build/tools/vmaf \
  --reference testdata/ref_576x324_48f.yuv \
  --distorted testdata/dis_576x324_48f.yuv \
  --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
  --model path=model/vmaf_v0.6.1.json \
  --feature motion --no_prediction --json --output /tmp/motion.json
# integer_motion2 scores must match pre-port baseline at 6 decimal places.
```

### 0048 — `i4_adm_cm` int32 rounding overflow deliberately preserved (ADR-0155)

- **ADR**: [ADR-0155](adr/0155-adm-i4-rounding-deferred-netflix-955.md)
- **Upstream source**: Netflix upstream issue
  [#955](https://github.com/Netflix/vmaf/issues/955) (OPEN since
  2020; no maintainer response as of 2026-04-24). Reports that
  `add_bef_shift_flt[idx] = (1u << (shift_flt[idx] - 1))` in
  `libvmaf/src/feature/integer_adm.c` scales 1–3 overflows
  `int32_t` (`1u << 31 = 0x80000000` wraps to `-2147483648`).
  Rounding term is sign-negated; ADM scales 1–3 biased low by
  ≈1 LSB per summed term.
- **Touches** (documentation-only):
  - [`docs/adr/0155-adm-i4-rounding-deferred-netflix-955.md`](adr/0155-adm-i4-rounding-deferred-netflix-955.md)
    — new ADR (this entry's anchor).
  - [`libvmaf/src/feature/integer_adm.c`](../libvmaf/src/feature/integer_adm.c)
    — in-file warning comment above the overflow site
    (`add_bef_shift_flt[]` initialiser loop around line 1277).
    **No code change.**
  - [`libvmaf/src/feature/AGENTS.md`](../libvmaf/src/feature/AGENTS.md)
    — invariant note under "Rebase-sensitive invariants".
- **Invariants** (load-bearing — do NOT silently "fix"):
  1. `integer_adm.c` keeps `int32_t add_bef_shift_flt[3]` with
     the overflowing `1u << 31` assignment. The Netflix golden
     assertions (`python/test/quality_runner_test.py`,
     `vmafexec_test.py`, `feature_extractor_test.py`) encode the
     buggy ADM output. Project hard rule #1
     ([ADR-0024](adr/0024-netflix-golden-preserved.md)) prohibits
     changing those assertions.
  2. Any "fix" that changes ADM numerical output must land
     together with a coordinated Netflix-authored golden-number
     update (the [ADR-0142](adr/0142-port-netflix-18e8f1c5-vif-sigma-nsq.md)
     Netflix-authority carve-out). Until Netflix#955 closes
     upstream, there is no authority to track.
- **On upstream sync**:
  - If Netflix finally lands a fix for #955 (widening the
    rounding term to `uint32_t` or `int64_t`), sync the C-side
    fix AND the updated `assertAlmostEqual` values in the same
    merge. Re-run `make test-netflix-golden` and
    `/cross-backend-diff` on the golden pairs to verify the new
    numbers are consistent across CPU / CUDA / SYCL.
  - Remove the in-file warning comment above the `add_bef_shift_flt`
    initialiser loop, flip ADR-0155 to `Superseded by ADR-NNNN`,
    and drop this rebase-notes entry.
  - If upstream instead closes #955 as wont-fix, keep this entry
    verbatim and update the ADR status to note upstream's
    closure.
- **Re-test on rebase** (gates the invariant by confirming the
  golden numbers are unchanged):

  ```bash
  ninja -C build
  make test-netflix-golden
  # Expect: VMAF mean 76.66890… on src01_hrc00/01_576x324 golden
  # pair — bit-identical to pre-rebase.
  ```

### 0047 — `vmaf_score_pooled` -EAGAIN for pending features (ADR-0154)

- **ADR**: [ADR-0154](adr/0154-score-pooled-eagain-netflix-755.md)
- **Upstream source**: Netflix upstream issue
  [#755](https://github.com/Netflix/vmaf/issues/755) (OPEN as of
  2026-04-24). Upstream maintainer closed the door on the
  streaming use case in 2020 ("you cannot call
  vmaf_score_pooled() in a loop"); fork reopens it via error-code
  semantics without changing the retroactive-write design.
- **Touches**:
  - [`libvmaf/src/feature/feature_collector.c`](../libvmaf/src/feature/feature_collector.c)
    — `vmaf_feature_collector_get_score` returns `-EAGAIN`
    (was `-EINVAL`) when the requested index is valid but not
    yet written.
  - [`libvmaf/src/feature/feature_collector.h`](../libvmaf/src/feature/feature_collector.h)
    — inline `vmaf_feature_vector_get_score` now returns
    `-EINVAL` for null/out-of-range and `-EAGAIN` for
    not-written (was `-1` for both). Added `#include <errno.h>`.
    Rename reserved `__VMAF_FEATURE_COLLECTOR_H__` guard to
    `VMAF_FEATURE_COLLECTOR_INCLUDED`.
  - [`libvmaf/test/test_score_pooled_eagain.c`](../libvmaf/test/test_score_pooled_eagain.c)
    — new 4-subtest reducer.
  - [`libvmaf/test/meson.build`](../libvmaf/test/meson.build)
    — register the new test.
- **Invariants** (load-bearing, enforced by the reducer):
  1. `vmaf_feature_collector_get_score(fc, name, &score, i)`
     returns `-EAGAIN` iff the feature `name` is registered and
     `i` is in range but `score[i].written == false`.
  2. The return stays `-EINVAL` for (a) null pointers, (b)
     `i >= feature_vector->capacity`, (c) unknown feature name.
  3. The inline fast-path `vmaf_feature_vector_get_score` uses
     the same split.
- **On upstream sync**: upstream has not changed the error
  semantics since 2020. If they do (unlikely), keep the fork's
  `-EAGAIN` — it is strictly more informative and downstream
  code depending on the split would regress.
- **Re-test on rebase**:

  ```bash
  ninja -C build && meson test -C build test_score_pooled_eagain
  # Expect: 4/4 subtests pass.

  # Reducer check:
  git stash push libvmaf/src/feature/feature_collector.c libvmaf/src/feature/feature_collector.h
  ninja -C build && meson test -C build test_score_pooled_eagain
  # Expect: Fail: 1 (tests fail without -EAGAIN split).
  git stash pop
  ```

### 0046 — `float_ms_ssim` min-dim guard (ADR-0153)

- **ADR**: [ADR-0153](adr/0153-float-ms-ssim-min-dim-netflix-1414.md)
- **Upstream source**: Netflix upstream issue
  [#1414](https://github.com/Netflix/vmaf/issues/1414) (OPEN as
  of 2026-04-24). No upstream fix has landed; fork adds the
  guard independently.
- **Touches**:
  - [`libvmaf/src/feature/float_ms_ssim.c`](../libvmaf/src/feature/float_ms_ssim.c)
    — add `#include "log.h"` + `#include "iqa/ssim_tools.h"` +
    a `min_dim = GAUSSIAN_LEN << (SCALES - 1)` check at the
    start of `init`; extract SIMD dispatch into a new
    `ms_ssim_init_simd_dispatch` helper to keep `init` within
    the ADR-0141 60-line budget.
  - [`libvmaf/test/test_float_ms_ssim_min_dim.c`](../libvmaf/test/test_float_ms_ssim_min_dim.c)
    — new 3-subtest reducer.
  - [`libvmaf/test/meson.build`](../libvmaf/test/meson.build)
    — register the new test executable.
- **Invariant** (load-bearing, enforced by the reducer):
  `float_ms_ssim.init` returns `-EINVAL` when
  `w < 176 || h < 176`, where 176 is computed dynamically from
  the filter constants. The magic number is not hardcoded —
  changing `SCALES` or `GAUSSIAN_LEN` upstream will auto-update
  the minimum.
- **On upstream sync**: if Netflix upstream lands a similar
  init-time guard, keep the fork's version — the helper name
  `ms_ssim_init_simd_dispatch` is fork-local (introduced to
  satisfy ADR-0141) and upstream's patch won't match. Both
  guards should be compatible; re-verify the reducer after
  rebase.
- **Re-test on rebase**:

  ```bash
  ninja -C build && meson test -C build test_float_ms_ssim_min_dim
  # Expect: 3/3 subtests pass.

  # Reducer check (confirms the guard is load-bearing):
  git stash push libvmaf/src/feature/float_ms_ssim.c
  ninja -C build && meson test -C build test_float_ms_ssim_min_dim
  # Expect: Fail: 1 (tests fail without the guard).
  git stash pop
  ```

### 0045 — `vmaf_read_pictures` monotonic-index guard (ADR-0152)

- **ADR**: [ADR-0152](adr/0152-vmaf-read-pictures-monotonic-index.md)
- **Upstream source**: Netflix upstream issue
  [#910](https://github.com/Netflix/vmaf/issues/910) (OPEN as of
  2026-04-24). No upstream fix has landed; the fork adds the
  guard independently, per the 2021-10-14 maintainer comment
  that recommended exactly this shape.
- **Touches**:
  - [`libvmaf/src/libvmaf.c`](../libvmaf/src/libvmaf.c) — add
    `unsigned last_index` + `bool have_last_index` fields to
    `VmafContext`; prepend a monotonic-index check inside
    `read_pictures_validate_and_prep` (returns `-EINVAL` on
    duplicates / regressions); update the two new fields at
    the tail of the same helper on success.
  - [`libvmaf/test/test_read_pictures_monotonic.c`](../libvmaf/test/test_read_pictures_monotonic.c)
    — new 3-subtest reducer covering the Netflix#910
    sequence and the two classes of rejection (duplicate,
    out-of-order).
  - [`libvmaf/test/meson.build`](../libvmaf/test/meson.build)
    — register the new test executable.
- **Invariant** (load-bearing, enforced by the reducer):
  `vmaf_read_pictures(vmaf, ref, dist, index)` returns
  `-EINVAL` when `have_last_index && index <= last_index`.
  Flush (`vmaf_read_pictures(vmaf, NULL, NULL, 0)`) routes to
  `flush_context` *before* the guard runs — flushing remains
  always-available independent of the last accepted index.
- **On upstream sync**:
  - If Netflix upstream eventually lands a similar guard at the
    API boundary, keep the fork's version — the helper function
    name (`read_pictures_validate_and_prep`) is fork-local
    (ADR-0146), upstream's patch will target a different
    insertion point. Both guards should be compatible; re-verify
    the reducer after rebase.
  - If upstream instead lands an internal reordering mechanism
    (buffer-and-sort frames before dispatch), revisit this
    decision — the fork's API-level contract is stricter and may
    need to relax to match. Open a new ADR if so.
- **Re-test on rebase**:

  ```bash
  ninja -C build && meson test -C build test_read_pictures_monotonic
  # Expect: 3/3 subtests pass.

  # Reducer check (confirms the guard is load-bearing):
  git stash push libvmaf/src/libvmaf.c
  ninja -C build && meson test -C build test_read_pictures_monotonic
  # Expect: Fail: 1 (the test rejects the un-guarded behaviour).
  git stash pop
  ```

### 0044 — i686 (32-bit x86) build-only CI job (ADR-0151)

- **ADR**: [ADR-0151](adr/0151-i686-ci-netflix-1481.md)
- **Upstream source**: Netflix upstream issue
  [#1481](https://github.com/Netflix/vmaf/issues/1481) (OPEN as
  of 2026-04-24). Reports i686 compile failure on
  `_mm256_extract_epi64`. Workaround documented in the issue:
  `-Denable_asm=false`.
- **Touches**:
  - [`build-aux/i686-linux-gnu.ini`](../build-aux/i686-linux-gnu.ini)
    — new cross-file; gcc + `-m32` + `cpu_family = 'x86'` /
    `cpu = 'i686'`. No `exe_wrapper`.
  - [`.github/workflows/libvmaf-build-matrix.yml`](../.github/workflows/libvmaf-build-matrix.yml)
    — new matrix row with `i686: true` flag + new install-deps
    step for `gcc-multilib` + `g++-multilib`; existing "Run
    tests" + "Run tox tests (ubuntu)" steps widened with
    `&& !matrix.i686` guards.
- **Invariants**:
  1. The i686 matrix row pins `-Denable_asm=false` — this is the
     upstream-documented workaround for
     `_mm256_extract_epi64`'s missing declaration on 32-bit x86
     targets. Do NOT remove the flag without first gating every
     `_mm256_extract_epi64` call site in
     `libvmaf/src/feature/x86/adm_avx2.c` +
     `motion_avx2.c` + `adm_avx512.c` on `__x86_64__`. Removing
     the flag naively will re-break the build.
  2. No `exe_wrapper` in the cross-file: meson marks tests as
     `SKIP 77` even though the host can run i686 binaries
     natively. Build-only gate by design.
- **On upstream sync**:
  - If upstream Netflix fixes #1481 at source (by gating the
    intrinsic calls on `__x86_64__` or by emulating via two
    `_mm256_extract_epi32` halves), sync the fix and **re-enable
    ASM on the i686 row** (drop `-Denable_asm=false` from
    `meson_extra`). Re-verify bit-exactness via
    `/cross-backend-diff` on the x86_64 golden pair.
  - If upstream marks i686 unsupported in meson (e.g. via a
    hard error), the fork's i686 row should be removed or
    downgraded to `continue-on-error: true`.
- **Re-test on rebase** (Ubuntu host with `gcc-multilib`):

  ```bash
  meson setup libvmaf libvmaf/build-i686 \
      --cross-file=build-aux/i686-linux-gnu.ini \
      -Denable_asm=false \
      -Denable_cuda=false -Denable_sycl=false
  ninja -C libvmaf/build-i686
  file libvmaf/build-i686/tools/vmaf
  # Expect: ELF 32-bit LSB pie executable, Intel i386
  ```

  CI runs this same sequence via the new matrix row.

### 0058 — Tiny-AI Netflix corpus training scaffold (ADR-0252)

- **ADR**: [ADR-0252](adr/0242-tiny-ai-netflix-training-corpus.md).
- **Upstream source**: fork-local. Netflix/vmaf has no tiny-AI training
  harness or MCP server.
- **Touches**:
  - [`ai/`](../ai/) — training harness; `NflxLocalDataset` loader reads
    from `--data-root` (never from a hardcoded path).
  - [`docs/ai/training-data.md`](ai/training-data.md) — corpus path
    convention and loader API docs; purely additive.
  - [`mcp-server/vmaf-mcp/tests/test_smoke_e2e.py`](../mcp-server/vmaf-mcp/tests/test_smoke_e2e.py)
    — new e2e smoke test; references only committed golden fixtures.
- **Invariants** (load-bearing):
  1. **Data path is local-only.** `.workingdir2/netflix/` is gitignored;
     no YUV from this corpus is ever committed. The `--data-root` CLI
     flag must remain the sole mechanism for locating the corpus.
  2. **Smoke test uses only committed fixtures.** `test_smoke_e2e.py`
     references `python/test/resource/yuv/src01_hrc00_576x324.yuv` (a
     committed golden file), never the local corpus path. On upstream
     sync the golden YUV path must stay stable.
  3. **No Netflix golden assertion is modified.** The `places=4` tolerance
     in `test_smoke_e2e.py` asserts against the `vmaf_v0.6.1` CPU
     reference; it is not a golden assertion and may be adjusted by
     `/regen-snapshots` with justification.
- **On upstream sync**: zero interaction with Netflix upstream. The `ai/`
  subtree and `mcp-server/` are wholly fork-local; upstream merges are
  conflict-free here. If Netflix ever ships a training harness, reconcile
  separately.
- **Re-test on rebase**:

  ```bash
  cd mcp-server/vmaf-mcp && python -m pytest tests/test_smoke_e2e.py -v
  # Requires: meson compile -C build (vmaf binary)
  # Skips automatically if binary or golden YUV is absent.
  ```

### 0085 — Research-0030 Phase-3b multi-seed validation (Gate 1 passed)

- **No ADR.** Empirical research digest closing Gate 1 of the
  3-gate v2 validation chain. Architecture decision unchanged.
- **Upstream source**: fork-local. Netflix has no multi-seed
  validation surface for tiny-AI training.
- **Touches** (additive only):
  - `docs/research/0030-phase3b-multiseed-validation.md` —
    per-seed PLCC tables + stability analysis + Gate 2/3 plan.
  - `ai/scripts/phase3_subset_sweep.py` — adds `--seeds` flag
    (comma-separated list) + per-seed result aggregation.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **The +0.0175 Δ is multi-seed mean PLCC**, not seed-0
     PLCC. Don't cite the +0.0106 from Research-0029 once
     Research-0030 lands; the multi-seed number is more
     trustworthy.
  2. **Subset B is more stable than canonical-6 across seeds**.
     Don't ship a v2 model citing single-seed numbers — always
     report multi-seed mean ± seed-mean-std for any tiny-AI
     metric in a future digest.
  3. **The `--seeds` flag aggregates by flattening (seed × fold)
     pairs**. The reported `mean_plcc` is the mean of all
     `n_seeds × n_folds` measurements; `seed_mean_plcc_std`
     is the std *across per-seed means*, which is the right
     number for "is the result seed-stable".
- **On upstream sync**: zero interaction. Fork-only research.
- **Re-test on rebase**: documentation-only PR; the runs/ files
  reproduce from the canonical command.

### 0084 — Research-0029 Phase-3b StandardScaler retry (positive result)

- **No ADR.** Empirical research digest; revives the Research-0026
  hypothesis after the Research-0028 negative result. The
  architectural decision (ship `vmaf_tiny_v2`) is gated on three
  validation steps documented in the digest §"Required before
  shipping".
- **Upstream source**: fork-local. Netflix has no tiny-AI
  preprocessing-sensitivity analysis surface.
- **Touches** (additive only):
  - `docs/research/0029-phase3b-standardscaler-results.md` —
    per-fold tables + apples-to-apples comparison + 3-gate
    pre-shipping checklist.
  - `ai/scripts/phase3_subset_sweep.py` — adds `--standardize`
    flag + `_standardize_inplace` helper.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **StandardScaler statistics MUST be fit per-fold on the
     train split only.** Fitting on the full data would leak
     held-out information into LOSO; the `_standardize_inplace`
     helper enforces this by taking only the train slice as input.
  2. **A shipped `vmaf_tiny_v2.onnx` MUST bundle its scaler
     `(mean, std)`** in the sidecar JSON per ADR-0049 — otherwise
     inference applies different normalisation than training and
     the win evaporates. Currently UN-implemented; tracked as a
     §"Caveats" #5 follow-up.
  3. **Subset B's feature list is the load-bearing finding**:
     `adm2`, `adm_scale3`, `vif_scale2`, `motion2`, `ssimulacra2`,
     `psnr_hvs`, `float_ssim`. Phase-3c experiments may shift
     the optimal arch / lr / epochs but should keep this set.
- **On upstream sync**: zero interaction. Fork-only research.
- **Re-test on rebase**: documentation-only PR; the runs/ files
  are reproducible from the `--standardize` invocation in
  §"Reproducer".

### 0082 — Research-0028 Phase-3 subset sweep (negative-result digest)

- **No ADR.** Empirical research digest. The architectural decision
  (no v2 model ships from this Phase) is governed by Research-0027's
  pre-registered stopping rule.
- **Upstream source**: fork-local. Netflix has no tiny-AI subset-
  sweep surface.
- **Touches** (additive only):
  - `docs/research/0028-phase3-subset-sweep.md` — per-fold tables
    + headline + standardisation caveat + Phase-3b/c/d follow-ups.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **canonical-6 stays the default** until Phase-3b lands a
     ≥ 0.005 PLCC win (per Research-0027 stopping rule).
  2. **The PLCC drop is most likely a feature-scale issue**, not
     evidence the new features lack signal. Don't cite this digest
     to retire `ssimulacra2` / `adm_scale3` from the candidate pool;
     re-test with `StandardScaler` first.
  3. **Phase-3 results are seed=0 only.** Any v2-shipping decision
     needs 3-seed mean±std and KoNViD cross-check.
- **On upstream sync**: zero interaction. Fork-only research.
- **Re-test on rebase**: documentation-only PR; runs/ files are
  reproducible from the canonical command in §"Reproducer".

### 0081 — Research-0027 Phase-2 feature importance results

- **No ADR.** Empirical research digest closing Research-0026
  Phase 2; the architectural decision (Subset A / B / C) is
  deferred to Phase-3 results in a future digest.
- **Upstream source**: fork-local. Netflix has no cross-metric
  feature-importance analysis surface.
- **Touches** (additive only):
  - `docs/research/0027-phase2-feature-importance.md` — per-method
    top-10 + consensus + redundancy + Phase-3 subset
    recommendations.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **Consensus top-10 is the load-bearing finding**:
     `adm2`, `adm_scale3`, `ssimulacra2`, `vif_scale2`. Phase-3
     candidate subsets MUST include all four.
  2. **The 11-pair redundancy table is corpus-specific** —
     measurements on Netflix Public 9-source. KoNViD-1k cross-
     check is a Phase-3 prerequisite if Subsets B/C advance.
  3. **`runs/full_features_netflix.parquet` and
     `runs/full_features_correlation.json` stay gitignored.**
     Reproducer in §"Reproducer" regenerates both.
- **On upstream sync**: zero interaction. Fork-only research.
- **Re-test on rebase**: documentation-only PR; the `runs/` files
  are reproducible from the canonical commands.

### 0080 — Phase-2 analysis scripts (Research-0026 Phase 2 prep)

- **No ADR.** Pure analysis scaffolding; the architectural
  decision (which features to ship in v2) is gated on Phase 2's
  numerical output via Research-0027.
- **Upstream source**: fork-local. Netflix has no tiny-AI
  training nor cross-metric correlation tooling.
- **Touches** (additive only):
  - `ai/scripts/extract_full_features.py` — parquet extractor
    over Netflix corpus with `FULL_FEATURES`. Per-clip JSON
    cache at `$XDG_CACHE_HOME/vmaf-tiny-ai-full/<source>/<dis_stem>.json`.
  - `ai/scripts/feature_correlation.py` — Pearson + MI + LASSO
    + RF + consensus top-K analyser; outputs JSON.
  - `ai/tests/test_feature_correlation.py` — 5 pytest cases
    against synthetic parquet (no libvmaf dependency).
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **The per-clip JSON cache and the `FULL_FEATURES` tuple
     must stay in lock-step.** If the tuple grows (or shrinks),
     pre-existing cache files become stale and silently
     misalign their stored `per_frame` columns with the new
     tuple. The extractor MUST be re-run with a cleared cache
     when `FULL_FEATURES` changes. Regression hint:
     `test_default_features_unchanged` in
     `test_feature_sets.py` already guards the canonical 6;
     extend coverage to `FULL_FEATURES` if rebases touch it.
  2. **`motion3` resolves to extractor `motion_v2`** in
     `_METRIC_TO_EXTRACTOR`, not `motion3` (the upstream-canonical
     extractor name in the integer_motion_v2 module). The CLI
     `--feature motion3` does NOT exist. The JSON output key is
     `integer_motion3` which `_lookup` finds via the `integer_`
     fallback.
  3. **`adm` and `vif` aggregates are NOT in `FULL_FEATURES`.**
     The integer extractor emits `integer_adm2` and
     `integer_vif_scale0..3` but no bare `adm`/`vif`. Listing
     them produced all-NaN columns in v1 — fixed in PR #185
     amend.
- **On upstream sync**: zero interaction. Pure fork-side
  analysis tooling.
- **Re-test on rebase**:

  ```bash
  pytest ai/tests/test_feature_correlation.py ai/tests/test_feature_sets.py -v
  # Expect: 14 passed in <1 s.
  ```

### 0079 — Tiny-AI feature-set registry (Research-0026 Phase 1)

- **No ADR.** Pure additive extension of an existing module; the
  architectural decision (which features, which model) lives in
  Research-0026's go/no-go gate after Phase 2.
- **Upstream source**: fork-local. Netflix/vmaf has no tiny-AI
  training pipeline.
- **Touches** (additive only):
  - `ai/data/feature_extractor.py` — adds `FULL_FEATURES` (21
    entries), `FEATURE_SETS` registry, `resolve_feature_set()`
    helper. `_METRIC_TO_EXTRACTOR` grew 11 → 25 entries.
  - `ai/tests/test_feature_sets.py` — new 9-test smoke suite.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant — these are load-bearing):
  1. **`DEFAULT_FEATURES` stays the canonical 6-tuple** matching
     `vmaf_v0.6.1`'s SVR input layout. Test
     `test_default_features_unchanged` is the regression guard;
     any quiet broadening would invalidate every shipped
     tiny-AI ONNX (input-dim baked into the model). If a future
     change must broaden the default, ship a paired model swap
     under ADR-0049 sidecar policy.
  2. **`FULL_FEATURES` excludes `lpips` and `float_moment`** per
     Research-0026 §"Open questions" Q1. Test
     `test_full_features_excludes_lpips_and_moment` enforces.
     Adding either would re-classify the experiment from "tiny
     model on classical features" to "ensemble of DNNs".
  3. **Every entry in `FULL_FEATURES` MUST have an entry in
     `_METRIC_TO_EXTRACTOR`**. Test
     `test_every_full_feature_has_extractor_mapping` is the
     guard — without the mapping the libvmaf CLI silently emits
     NaN columns for the missing metric.
- **On upstream sync**: zero interaction. Fork-only training surface.
- **Re-test on rebase**:

  ```bash
  pytest ai/tests/test_feature_sets.py -v
  # Expect: 9 passed in <1 s.
  ```

### 0078 — Research-0026 cross-metric feature fusion plan

- **No ADR.** Pure research-plan digest; the architectural
  decision (which features to add) is deferred to Research-0027
  follow-up after Phase 2 numbers land.
- **Upstream source**: fork-local. Netflix/vmaf has no tiny-AI
  training and no broader-feature-set hypothesis under
  investigation.
- **Touches** (additive only):
  - `docs/research/0026-cross-metric-feature-fusion.md` — 4-phase
    experimental plan + cost estimate + go/no-go criteria.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **The 6-feature canonical baseline (`adm2`, `vif_scale0..3`,
     `motion2`) stays the default.** Any v2 model is opt-in via a
     new `feature_set` field in the sidecar JSON; existing
     `vmaf_tiny_v1.onnx` users get the same numbers.
  2. **`lpips` is OUT of the candidate pool** (Phase 1/2). It's
     DNN-based and would blur the line between "tiny model on
     classical features" and "ensemble of DNNs". Revisit only if
     classical features can't close the gap.
- **On upstream sync**: zero interaction. Pure fork-side
  research planning.
- **Re-test on rebase**: documentation-only; no test surface.

### 0077 — Research-0025 FoxBird outlier resolved via KoNViD combined training

- **No ADR.** Empirical research digest closing the open question
  in Research-0023 §5; no architecture or policy decision. Pure
  documentation of an empirical result.
- **Upstream source**: fork-local. Netflix/vmaf has no tiny-AI
  training, no KoNViD-1k integration, and no LOSO eval surface.
- **Touches** (additive only):
  - `docs/research/0025-foxbird-resolved-via-konvid.md` —
    per-clip table + comparison to Netflix-only baselines +
    interpretation + caveats + next-experiment list.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **The training-fit per-clip numbers in §"Per-clip result"
     are NOT held-out generalisation metrics** — FoxBird is in
     the training set. The proper validation is the LOSO sweep
     on the combined corpus (§"Next experiments" #1). Don't cite
     the 0.9936 FoxBird PLCC as a generalisation number; cite it
     as "training-fit on combined corpus, 5.4× RMSE improvement
     vs Netflix-only".
  2. **Combined trainer command line is canonical.** The
     reproduction recipe in §"Setup" includes `--seed 0`,
     `--konvid-val-fraction 0.1`, `--val-source Tennis`,
     `--val-mode netflix-source-and-konvid-holdout`. Changing
     any knob invalidates the per-clip numbers.
  3. **`runs/tiny_combined_canonical/` stays gitignored.** The
     final ONNX is reproducible from the parquet + Netflix
     corpus + the canonical CLI; the durable record is the
     digest's table.
- **On upstream sync**: zero interaction. Research digest is
  fork-only.
- **Re-test on rebase**:

  ```bash
  python ai/train/train_combined.py \
    --netflix-root .workingdir2/netflix \
    --konvid-parquet ai/data/konvid_vmaf_pairs.parquet \
    --model-arch mlp_small --epochs 30 --batch-size 256 --lr 1e-3 \
    --val-mode netflix-source-and-konvid-holdout \
    --val-source Tennis --konvid-val-fraction 0.1 --seed 0 \
    --out-dir runs/tiny_combined_canonical
  # Expect: FoxBird PLCC ≈ 0.9936 ± 1e-3 (numerical-noise floor),
  # mean PLCC ≥ 0.9983 across 9 Netflix clips.
  ```

### 0076 — Research-0024 vif/adm upstream-divergence digest (Strategy E doc)

- **No ADR.** Pure documentation digest; the divergence
  decisions it ratifies are already governed by ADR-0138 /
  0139 / 0142 / 0143 (vif SIMD bit-exactness contract) and
  ADR-0024 (Netflix golden-data immutability). The digest
  itself fits the per-PR research-digest deliverable bar from
  ADR-0108.
- **Upstream source**: forward-looking — pre-emptively documents
  the fork's *non-port* of Netflix `4ad6e0ea` / `41d42c9e` /
  `bc744aa3` / `8c645ce3` (vif chain) and `4dcc2f7c` (float_adm
  chain). Strategy A on `b949cebf` motion chain stays approved.
- **Touches** (additive only):
  - `docs/research/0024-vif-upstream-divergence.md` — 5-strategy
    decision matrix + numerical-risk analysis for each chain.
  - `libvmaf/src/feature/AGENTS.md` — two new "rebase-sensitive
    invariants" entries pinning the vif and adm divergences.
  - `CHANGELOG.md` Unreleased § Changed.
- **Invariants** (rebase-relevant — these are the *whole point*):
  1. **Do not port `4ad6e0ea` (vif runtime helpers) or
     `8c645ce3` (vif prescale options) verbatim.** They replace
     the precomputed `vif_filter1d_table_s` table whose frozen
     `const float` Gaussians make AVX2 == AVX-512 == NEON ==
     scalar bit-for-bit. A future opt-in second-path port
     (Strategy C, runtime helpers behind `--vif-prescale != 1`)
     is allowed but must not touch the default code path.
  2. **Do not port `4dcc2f7c` float_adm options chain.** The
     12-parameter `compute_adm` signature change cascades
     through SIMD (avx2 / avx512 / neon) **and** 3 GPU backends
     (vulkan / cuda / sycl). The new `aim` feature has no fork-
     side golden values; defer until concrete user demand.
  3. **Mirror bugfix `41d42c9e` is a separate decision.** Must
     come paired with `places=4 → places=3` golden loosening per
     ADR-0142 Netflix-authority precedent. Not part of Strategy
     E; eligible for a focused single-purpose PR if any shipped
     model drifts more than `places=3` because of the missing
     fix.
  4. **`b949cebf` motion chain port stays APPROVED** under
     Strategy A (verbatim, float_motion-side only). Float_motion
     has no precomputed-table investment to protect; existing
     fork integer_motion already has 6/9 of these options; cheap
     to mirror onto float_motion.
- **On upstream sync**: zero conflict — pure additions to
  research/ and AGENTS.md.
- **Re-test on rebase**: documentation-only PR; rendered markdown
  is the only verification surface.

  ```bash
  # Re-run the diff scan that produced the digest (catches new
  # upstream commits since 9dac0a59):
  git fetch upstream && git log --pretty=format:'%h %s' \
    upstream/master ^origin/master --since="2026-01-01" \
    -- libvmaf/src/feature/{float_,integer_,}{vif,motion,adm,cambi}*.{c,h} \
       libvmaf/src/feature/{vif,motion,adm,cambi}_options.h \
    | head -30
  # If new vif / adm option ports appear, update Research-0024 §"Same
  # divergence test for motion + float_adm" before deciding to port.
  ```

### 0075 — Upstream `798409e3` + `314db130` ports (CUDA null-deref + remove `all.c`)

- **No ADR.** Pure upstream cherry-picks per
  [ADR-0108](adr/0108-deep-dive-deliverables-rule.md) carve-out
  ("pure upstream syncs and `port-upstream-commit` PRs are exempt").
- **Upstream source**:
  - `798409e3` (Lawrence Curtis, 2026-04-20):
    "Fix null deref crash on prev_ref update in pure CUDA pipelines"
  - `314db130` (Kyle Swanson, 2026-04-28):
    "libvmaf/feature: remove empty translation unit all.c"
- **Touches** (additive / removal only):
  - `libvmaf/src/libvmaf.c` — adds `if (ref && ref->ref)` guard
    before `vmaf_picture_ref(&vmaf->prev_ref, ref)` at the two
    threaded paths (`threaded_enqueue_one` line 1057 and
    `threaded_read_pictures_batch` line 1105). Main path at
    line 1597 already has the guard.
  - `libvmaf/src/feature/all.c` — file deleted.
  - `libvmaf/src/meson.build` — drops the `feature_src_dir + 'all.c'`
    line.
  - `libvmaf/src/feature/offset.c` — updates the `// NOLINTNEXTLINE`
    comment to drop `all.c` from the list of per-feature consumers.
  - `CHANGELOG.md` Unreleased § Fixed (798409e3) + § Changed (314db130).
- **Invariants** (rebase-relevant):
  1. **The fork has THREE prev_ref update sites; all need the
     `if (ref && ref->ref)` guard.** The main `vmaf_read_pictures`
     path already had it (via `read_pictures_update_prev_ref` helper);
     the threaded paths (`#ifdef VMAF_BATCH_THREADING`) inherited
     the unguarded shape from upstream's old code. Future upstream
     rebases must preserve all three guards even if Netflix
     refactors the threaded paths.
  2. **`all.c` deletion is symbol-safe.** All `compute_*` functions
     it forward-declared are reached via per-extractor TUs that
     `#include` the relevant `<feature>.h`. No external linker
     dependency on `all.c`'s symbols.
- **On upstream sync**: zero conflict expected — fork now matches
  upstream tip on these two surfaces.
- **Re-test on rebase**:

  ```bash
  meson setup build-cpu libvmaf -Denable_cuda=false -Denable_sycl=false \
    -Denable_vulkan=disabled
  ninja -C build-cpu
  meson test -C build-cpu  # 37 tests, all pass.
  ```

### 0074 — Combined Netflix + KoNViD-1k trainer driver

- **No ADR.** Pure engineering follow-up; the architecture rationale
  is fully covered by ADR-0203 (training-prep architecture) and
  Research-0023 §5 (FoxBird-class outlier needs broader corpus).
- **Upstream source**: fork-local. Netflix/vmaf has no tiny-AI
  trainer.
- **Stacks on** the KoNViD-1k loader bridge (PR #178 / rebase-note
  0073). Rebase order: land 0073 first.
- **Touches** (additive only):
  - `ai/train/train_combined.py` — concatenating trainer that reuses
    `_build_model` / `_train_loop` / `export_onnx` from
    `ai/train/train.py`.
  - `ai/tests/test_train_combined_smoke.py` — 5 pytest cases
    (key splitter + `--epochs 0` paths, no libvmaf or real corpus
    required).
  - `docs/ai/training.md` — "Combining KoNViD with the Netflix
    corpus" subsection rewritten from "follow-up" to runnable.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **Reuse the canonical training-loop helpers.** Don't fork
     `_build_model` / `_train_loop` / `export_onnx` into this file.
     Both trainers must share the model factory so a future change
     (e.g. adding `mlp_large`) lands in one place.
  2. **KoNViD train/val splits hold out *whole clip keys*, not
     random frames.** A frame-level split would let frames from the
     same clip leak across train/val and inflate PLCC by 5-10 pp
     (well-known VQA pitfall — same reasoning as ADR-0203's Netflix
     1-source-out split).
  3. **Missing data falls back, not errors.** Missing
     `--konvid-parquet` → Netflix-only path. Missing
     `--netflix-root` → KoNViD-only path. Both missing → initial-
     weights ONNX export + `rc=0` so the smoke command always
     produces a deterministic artefact.
- **On upstream sync**: zero interaction; pure fork-local trainer.
- **Re-test on rebase**:

  ```bash
  pytest ai/tests/test_train_combined_smoke.py -v
  # Expect: 5 passed (under ~3 s, no libvmaf required).
  python ai/train/train_combined.py --epochs 0 \
    --netflix-root /tmp/missing --konvid-parquet /tmp/missing.parquet \
    --out-dir /tmp/combined_smoke
  # Expect: <out-dir>/mlp_small_combined_final.onnx written, rc=0.
  ```

### 0073 — KoNViD-1k → VMAF-pair acquisition + loader bridge

- **No ADR.** Acquisition + loader pieces are pure additions; the
  methodology fits inside ADR-0203 / Research-0019.
- **Upstream source**: fork-local. KoNViD-1k integration is a
  fork-only training-data play.
- **Touches** (additive only):
  - `ai/scripts/konvid_to_vmaf_pairs.py` — acquisition pipeline.
  - `ai/train/konvid_pair_dataset.py` — `KoNViDPairDataset`
    class mirroring `NetflixFrameDataset`'s interface.
  - `ai/tests/test_konvid_pair_dataset.py` — 5 pytest cases.
  - `docs/ai/training.md` — new "C1 (KoNViD-1k corpus)" section.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **`KoNViDPairDataset` mirrors `NetflixFrameDataset` shape.**
     `feature_dim == 6`, `numpy_arrays() → (X, y)` returns
     `(n_frames, 6)` + `(n_frames,)`. If `NetflixFrameDataset`'s
     feature order changes, mirror it here.
  2. **Acquisition parquet schema is fixed.** Required columns:
     `key`, `frame_index`, `vif_scale0..3`, `adm2`, `motion2`,
     `vmaf`. Add freely; do NOT rename / drop those.
  3. **`ai/data/konvid_vmaf_pairs.parquet` and
     `$VMAF_TINY_AI_CACHE/konvid-1k/` stay gitignored.** They
     regenerate from raw KoNViD `.mp4` sources.
- **On upstream sync**: zero interaction.
- **Re-test on rebase**:

  ```bash
  pytest ai/tests/test_konvid_pair_dataset.py -v
  # Expect: 5 passed
  python ai/scripts/konvid_to_vmaf_pairs.py --max-clips 5
  # Expect: ~7 s wall, ai/data/konvid_vmaf_pairs.parquet with
  #         5 unique keys × ~200 frames each.
  ```

### 0072 — Tiny-AI 3-arch LOSO eval harness + Research-0023

- **No ADR.** Methodology fits inside Research-0023; ADR-0203
  already covers the training-prep architecture and the three-arch
  sweep concept.
- **Research digest**:
  [`docs/research/0023-loso-3arch-results.md`](research/0023-loso-3arch-results.md).
- **Upstream source**: fork-local. Netflix/vmaf has no LOSO eval
  surface.
- **Touches** (additive only):
  - `ai/scripts/eval_loso_3arch.py` — new harness; reuses the
    `_load_session` + `_load_clip` + `CLIPS` helpers from
    `eval_loso_mlp_small.py` (PR #165).
  - `docs/research/0023-loso-3arch-results.md` — methodology +
    per-fold tables for `mlp_small` / `mlp_medium` / `linear`.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **Reuse the PR #165 helpers.** Don't fork the
     `_load_session` external-data workaround into a copy — both
     scripts must keep using the same import. If a follow-up
     re-exports the shipped baselines with corrected
     `external_data.location`, both scripts deprecate the
     workaround simultaneously.
  2. **`runs/` and `model/tiny/training_runs/` stay gitignored.**
     The harness writes `runs/loso_eval/loso_3arch_eval.{json,md}`;
     the durable record is the table in Research-0023 §2 + the
     per-fold tables in §3. Regenerate via the loop in §6 of the
     digest.
- **On upstream sync**: zero interaction. Pure fork-local
  evaluation harness.
- **Re-test on rebase**:

  ```bash
  python ai/scripts/eval_loso_3arch.py
  diff <(jq -r '.archs.mlp_small.aggregate.mean_plcc' runs/loso_eval/loso_3arch_eval.json) <(echo 0.9808)
  diff <(jq -r '.archs.mlp_medium.aggregate.mean_plcc' runs/loso_eval/loso_3arch_eval.json) <(echo 0.9727)
  diff <(jq -r '.archs.linear.aggregate.mean_plcc' runs/loso_eval/loso_3arch_eval.json) <(echo 0.3679)
  # Expect: identical lines on a populated cache + identical fold ONNX.
  ```

### 0071 — T7-16 ADM Vulkan/SYCL drift verified-resolved (doc close)

- **No ADR.** Verification-only close, sister of T7-15.
- **Upstream source**: fork-local. ADM cross-backend gate is a
  fork-only test surface; Netflix/vmaf has no Vulkan or SYCL
  backend.
- **Touches** (additive only):
  - `docs/state.md` — new "Recently closed" row for T7-16.
  - `.workingdir2/BACKLOG.md` — T7-16 row marked closed (local-
    only planning dossier; gitignored).
  - `CHANGELOG.md` Unreleased § Fixed.
- **Invariants** (rebase-relevant):
  1. **`places=4` cross-backend ADM contract.** Empirical
     `adm_scale2` max_abs_diff is now 1e-6 (print floor; ULP=0)
     on Vulkan device 0 (NVIDIA), device 1 (Mesa anv on Arc),
     and SYCL device 0 (Arc); residual `adm_scale1 ≈ 3.1e-5`
     and `adm2 ≈ 5e-6` on 1/48 frames pass `places=4` (5e-5
     tolerance) but fail `places=5`. Hold the gate at `places=4`.
  2. **No ADM kernel source change.** Fix is environmental
     (NVCC + driver + SYCL runtime).
- **On upstream sync**: zero interaction.
- **Re-test on rebase**:

  ```bash
  python3 scripts/ci/cross_backend_vif_diff.py \
    --vmaf-binary libvmaf/build/tools/vmaf \
    --feature adm --backend vulkan --device 0 --places 4 \
    --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
    --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
    --width 576 --height 324
  # Expect: 0/48 mismatches across all 5 ADM metrics.
  ```

### 0070 — T7-15 motion CUDA/SYCL drift verified-resolved (doc close)

- **No ADR.** Verification-only close; no code change in PR #172.
- **Upstream source**: fork-local. Cross-backend gate is a
  fork-only test surface; not in Netflix/vmaf.
- **Touches** (additive only):
  - `docs/state.md` — "Recently closed" row for T7-15.
  - `.workingdir2/BACKLOG.md` — T7-15 row marked closed (local-
    only planning dossier; gitignored).
  - `CHANGELOG.md` Unreleased § Fixed.
- **Invariants** (rebase-relevant):
  1. **The `places=4` cross-backend gate stays at `places=4`.**
     Empirical max_abs_diff is currently 0.0 (CUDA) or 1e-6 (SYCL/
     Vulkan, JSON `%f` rounding floor); tightening to `places=5`
     could be tempting but the 1e-6 print-floor would then make
     the SYCL + Vulkan rows fail. Hold at `places=4` until
     `--precision=max` is wired into the diff tool.
  2. **No motion-kernel source change.** PR #172 didn't modify
     `libvmaf/src/feature/cuda/integer_motion/*.cu` or
     `libvmaf/src/feature/sycl/integer_motion_sycl.cpp`. The fix
     is environmental (NVCC + driver), so the next CI run on a
     fresh image needs to be re-verified against the gate.
- **On upstream sync**: zero interaction.
- **Re-test on rebase**:

  ```bash
  python3 scripts/ci/cross_backend_vif_diff.py \
    --vmaf-binary libvmaf/build/tools/vmaf \
    --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
    --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
    --width 576 --height 324 --feature motion --backend cuda \
    --places 4
  # Expect: 0/48 mismatches, max_abs_diff = 0.0
  ```

### 0069 — `libvmaf_vulkan.h` installed under prefix (build bug)

- **No ADR.** Build-system bug fix; matches existing CUDA / SYCL
  install conditions.
- **Upstream source**: fork-local. Vulkan backend is fork-only;
  Netflix/vmaf has no `libvmaf_vulkan.h`.
- **Touches**:
  - `libvmaf/include/libvmaf/meson.build` — adds an
    `is_vulkan_enabled` gate that handles the `feature` option's
    `enabled` / `auto` states; appends `libvmaf_vulkan.h` to
    `platform_specific_headers` when active.
  - `CHANGELOG.md` Unreleased § Fixed.
- **Invariants** (rebase-relevant):
  1. **Install rule mirrors the CUDA / SYCL pattern but uses the
     feature-option API.** The
     `is_cuda_enabled = get_option('enable_cuda') == true` boolean
     idiom doesn't apply to `enable_vulkan` because that's a
     feature option, not a boolean. Use `.enabled() or .auto()`.
     Don't "simplify" to `== true` — that would silently drop the
     install in the `auto` state.
  2. **Pairs with
     `ffmpeg-patches/0006-libvmaf-add-libvmaf-vulkan-filter.patch`**
     which probes for the header via
     `check_pkg_config libvmaf_vulkan "libvmaf >= 3.0.0"
     libvmaf/libvmaf_vulkan.h vmaf_vulkan_state_init_external`.
     Removing the install rule re-introduces lawrence's
     2026-04-28 symptom: FFmpeg silently drops the
     `libvmaf_vulkan` filter despite `--enable-libvmaf-vulkan`.
- **On upstream sync**: zero interaction; Vulkan backend is
  fork-only.
- **Re-test on rebase**:

  ```bash
  cd libvmaf
  CC=icx CXX=icpx meson setup build -Denable_vulkan=enabled \
    -Denable_cuda=true -Denable_sycl=true -Db_lto=false
  ninja -C build
  meson install -C build --destdir /tmp/libvmaf-install
  ls /tmp/libvmaf-install/usr/local/include/libvmaf/libvmaf_vulkan.h
  # Expect: file exists.
  ```

### 0066 — `--backend cuda` inverted-gpumask fix (CLI bug)

- **No ADR.** Bug fix; behaviour now matches the public-header
  `VmafConfiguration::gpumask` contract.
- **Upstream source**: fork-local. The `--backend` CLI selector was
  added by the fork (Netflix/vmaf has no exclusive-backend selector).
- **Touches** (additive + 1-line behavioural fix):
  - `libvmaf/tools/cli_parse.c::parse_cli_args` — `--backend cuda`
    branch sets `gpumask = 0` (was `gpumask = 1`).
  - `libvmaf/test/test_cli_parse.c` — 5 new regression tests
    (`test_backend_{cpu,cuda_engages_cuda,cuda_preserves_explicit_gpumask,sycl,vulkan}`)
    plus `run_aom_ctc_tests` / `run_backend_tests` helper split to
    keep `run_tests` under the function-size budget.
  - `CHANGELOG.md` Unreleased § Fixed.
- **Invariants** (rebase-relevant):
  1. **`VmafConfiguration::gpumask` semantics: `if gpumask: disable
     CUDA`.** `compute_fex_flags` in
     [`src/libvmaf.c`](../libvmaf/src/libvmaf.c) routes CUDA only
     when `gpumask == 0`. Any code path that sets a non-zero
     `gpumask` to "request CUDA" silently disables it. The CLI's
     `--backend cuda` branch must set `gpumask = 0` and rely on
     `use_gpumask = true` to trigger `vmaf_cuda_state_init`. Do not
     "fix" this back to `gpumask = 1` — it's the bug being fixed.
  2. **Explicit `--gpumask=N --backend cuda` preserves N.** A user
     who passes `--gpumask=2` already has `use_gpumask = true`, so
     the `--backend cuda` branch's defaulting block (gated on
     `!settings->use_gpumask`) is skipped. The
     `test_backend_cuda_preserves_explicit_gpumask` regression
     locks this in.
- **On upstream sync**: zero interaction; `--backend` is fork-only.
- **Re-test on rebase**:

  ```bash
  ./build/test/test_cli_parse | grep -E 'backend_'
  # Expect: 5 backend tests pass.
  build/tools/vmaf -r REF -d DIS -w 576 -h 324 -p 420 -b 8 \
    --model "path=model/vmaf_v0.6.1.json" --threads 1 \
    --backend cuda --output cuda.json --json -q
  python3 -c "import json; d=json.load(open('cuda.json')); \
    assert len(d['frames'][0]['metrics']) == 12, 'CUDA not engaged'"
  ```

### 0067 — Tiny-AI PTQ accuracy across Execution Providers (T5-3e)

- **No ADR.** Investigation/measurement PR; ADR-0129 already
  governs the PTQ workstream. Findings update
  [`docs/research/0006-tinyai-ptq-accuracy-targets.md`](research/0006-tinyai-ptq-accuracy-targets.md)
  §"GPU-EP quantisation" — that section was previously a
  deferred-open-question; it is now the empirical landing spot.
- **Research digest**: same file (Research-0006).
- **Upstream source**: fork-local. Netflix/vmaf does not ship a
  PTQ harness or any tiny-AI ONNX path.
- **Touches** (additive only):
  - `ai/scripts/measure_quant_drop_per_ep.py` — new sibling of
    `measure_quant_drop.py`. CPU+CUDA via ORT;
    Arc / OpenVINO-CPU via the native `openvino` Python runtime
    (no `onnxruntime-openvino` because no cp314 wheel exists).
    Reuses the `_load_session` rename workaround from PR #165 +
    a `value_info`-strip fix so dynamic-PTQ doesn't choke on
    the shipped MLP ONNX.
  - `docs/ai/quant-eps.md` — new user doc; linked from
    `docs/ai/index.md`.
  - `docs/research/0006-tinyai-ptq-accuracy-targets.md` —
    refreshed header, replaced "GPU-EP open question" with the
    measurement table, fixed pre-existing MD040/MD060 lints
    surfaced on the touched file.
  - `docs/ai/index.md` — added the quant-eps row, rewrapped to
    80 cols.
  - `CHANGELOG.md` Unreleased § Changed.
- **Invariants** (rebase-relevant):
  1. **`measure_quant_drop.py` (the CI gate) is unchanged.**
     The new script is purely additive. Any rebase that
     conflates the two scripts must keep the CI gate
     CPU-only — Arc int8 is broken, so a per-EP gate would
     red-light every PR.
  2. **`value_info` strip is required for `vmaf_tiny_v1*`
     dynamic PTQ.** The shipped MLP ONNX duplicate weight
     tensors in `value_info`, which makes
     `quantize_dynamic` raise
     `Inferred shape and existing shape differ`. The fix
     is in `_save_inlined`. Don't remove it during a refactor
     unless the underlying ONNX is regenerated.
  3. **CUDA-12 ABI shim.** ORT-GPU 1.25 wheels link
     `libcublasLt.so.12` even on CUDA-13 hosts. The
     reproduction recipe pins the
     `nvidia-*-cu12` wheels and prepends them to
     `LD_LIBRARY_PATH`. If a future ORT wheel drops the cu12
     ABI we can cut the shim, but the script tolerates either
     since it doesn't import any CUDA symbol itself.
- **On upstream sync**: zero interaction; entirely fork-local.
- **Re-test on rebase**:

  ```bash
  SP=$VIRTUAL_ENV/lib/python3.14/site-packages/nvidia
  export LD_LIBRARY_PATH="$SP/cublas/lib:$SP/cudnn/lib:$SP/cuda_nvrtc/lib:$SP/cuda_runtime/lib:$SP/cufft/lib:$SP/curand/lib:$SP/cusolver/lib:$SP/cusparse/lib:$SP/cuda_cupti/lib:$SP/nvtx/lib:$SP/nvjitlink/lib"
  python ai/scripts/measure_quant_drop_per_ep.py \
      --eps cpu cuda openvino \
      --extra-fp32 vmaf_tiny_v1.onnx vmaf_tiny_v1_medium.onnx \
      --out runs/quant-eps-$(date +%Y-%m-%d)
  # Expected: CPU + CUDA PASS (drop ≤ 1.2e-4); OpenVINO Arc ERR
  # (compile failure for Conv-int8) or NaN (MatMul-int8) until a
  # newer intel_gpu plugin lands.
  ```

### 0065 — `testdata/bench_all.sh` correct backend-engagement flags

- **No ADR.** Bug fix; no behavioural surface change beyond
  "the bench actually engages the backends it claims to now."
- **Upstream source**: fork-local. `testdata/bench_all.sh` is a
  fork-only bench harness; not in Netflix/vmaf.
- **Touches** (additive only):
  - `testdata/bench_all.sh` — switched per-row flag pattern from the
    disable-only singletons (`--no_sycl` for "CUDA", etc.) to the
    correct engagement form (`--gpumask=0 --no_sycl --no_vulkan` for
    CUDA, `--sycl_device=0 --no_cuda --no_vulkan` for SYCL,
    `--vulkan_device=0 --no_cuda --no_sycl` for Vulkan, and
    `--no_cuda --no_sycl --no_vulkan` for CPU). Added a 4th column
    (Vulkan) to the comparator. Honours `$VMAF_BIN` for the binary
    path and `$VMAF_ONEAPI_SETVARS` for the oneAPI install location.
  - `CHANGELOG.md` Unreleased § Fixed.
- **Invariants** (rebase-relevant):
  1. **Disable-only singletons don't engage a backend.** `--no_sycl`
     alone leaves CUDA available *but unrequested*. `--no_cuda` alone
     leaves SYCL available but unrequested. The CLI inits CUDA only
     when `c.use_gpumask` is set; SYCL only when `c.sycl_device >= 0`
     or `c.use_gpumask`; Vulkan only when `c.vulkan_device >= 0`. Any
     change to those gates that drops one of the per-row flags will
     re-introduce the silent CPU fallback. Verify after a rebase by
     inspecting JSON `frames[0].metrics` key counts (CPU 14-15,
     CUDA 11-12, Vulkan ~34) — see
     [`libvmaf/AGENTS.md`](../libvmaf/AGENTS.md) §"Backend-engagement
     foot-guns".
  2. **`gpumask` semantics are inverted from intuition.** `gpumask=0`
     enables CUDA dispatch; `gpumask=1` disables it. The per-row
     CUDA flag is `--gpumask=0`, not `--gpumask=1`. Don't "fix" it
     to `--gpumask=1` for symmetry with sycl_device/vulkan_device —
     that's the bug being fixed (parallel to PR #170).
- **On upstream sync**: zero interaction; `testdata/bench_all.sh`
  is fork-only.
- **Re-test on rebase**:

  ```bash
  bash testdata/bench_all.sh    # smoke
  # Verify each row's JSON keys match the expected per-backend count:
  jq '.frames[0].metrics | keys | length' testdata/bbb/results/t1_cpu.json
  jq '.frames[0].metrics | keys | length' testdata/bbb/results/t1_cuda.json
  jq '.frames[0].metrics | keys | length' testdata/bbb/results/t1_vulkan.json
  ```

### 0063 — Tiny-AI LOSO eval harness for `mlp_small`

- **No ADR.** The methodology fits inside Research Digest 0022;
  ADR-0203 already covers the training-prep architecture.
- **Research digest**:
  [`docs/research/0022-loso-mlp-small-results.md`](research/0022-loso-mlp-small-results.md).
- **Upstream source**: fork-local. Netflix/vmaf has no LOSO eval
  surface.
- **Touches** (additive only):
  - `ai/scripts/eval_loso_mlp_small.py` — new evaluation harness.
  - `docs/ai/loso-eval.md` — usage doc.
  - `docs/research/0022-loso-mlp-small-results.md` — methodology +
    results.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **`_load_session` workaround for renamed-baseline ONNX.** The
     shipped baselines `model/tiny/vmaf_tiny_v1*.onnx` reference
     their pre-rename `external_data.location` values. The
     workaround in `_load_session` rewrites the entries before
     handing the proto to ORT. Removing the workaround breaks the
     baseline phase. The proper fix (re-export with matching
     names) is tracked as a follow-up; until then this code path
     is load-bearing.
  2. **`runs/` and `model/tiny/training_runs/` stay gitignored.**
     The harness writes to `runs/loso_eval/` by default; do NOT
     promote any of those outputs into the tree. The 9 fold ONNX
     and the per-clip JSON cache regenerate from the corpus +
     trainer + libvmaf CLI.
- **On upstream sync**: zero interaction. Pure fork-local
  evaluation harness.
- **Re-test on rebase**:

  ```bash
  python ai/scripts/eval_loso_mlp_small.py
  diff <(jq -r '.loso_aggregate.mean_plcc' runs/loso_eval/loso_mlp_small_eval.json) <(echo 0.9808)
  # Expect: identical line on a populated cache + identical fold ONNX.
  ```

### 0064 — Section-A audit: 9 backlog rows + ADR cross-links

- **No ADR.** Process / docs PR; rows trace back to the
  individually-cited ADRs / research digests in their own
  References columns.
- **Decision dossier**:
  [`.workingdir2/decisions/section-a-decisions-2026-04-28.md`](../.workingdir2/decisions/section-a-decisions-2026-04-28.md).
- **Source audit**:
  [`docs/backlog-audit-2026-04-28.md`](backlog-audit-2026-04-28.md).
- **Upstream source**: fork-local. Pure backlog hygiene PR; no
  Netflix code touched.
- **Touches** (additive only):
  - `.workingdir2/BACKLOG.md` — 9 new rows: T3-17, T3-18, T5-3e,
    T5-4, T7-35, T7-36, T7-37, T7-38; T6-1a row extended with the
    bisect-cache fixture sub-bullet.
  - `docs/research/0006-tinyai-ptq-accuracy-targets.md` — drops
    the "defer until first user" framing on the GPU-EP
    quantisation open question per user direction; cross-links
    T5-3e.
  - `docs/research/0020-cambi-gpu-strategies.md` — v2 follow-up
    section now cites T7-36 as the gate for opening the v2 row.
  - `docs/adr/0205-cambi-gpu-feasibility.md` — Decision section's
    "follow-up integration PR" now cites T7-36.
  - `CHANGELOG.md` Unreleased § Changed.
- **Invariants** (rebase-relevant): none. Pure backlog text.
  Rebase-conflict risk is limited to the same `BACKLOG.md` table
  rows that any future row addition would touch; trivial to
  re-resolve.
- **On upstream sync**: zero interaction.
- **Re-test on rebase**: none — docs-only.

### 0062 — ssimulacra2 CUDA + SYCL twins (ADR-0206)

- **ADR**: [ADR-0206](adr/0206-ssimulacra2-cuda-sycl.md).
- **Upstream source**: fork-local. Netflix/vmaf has no SSIMULACRA 2
  GPU implementation; this PR adds the CUDA + SYCL twins of the
  fork's [ADR-0201](adr/0201-ssimulacra2-vulkan-kernel.md) Vulkan
  kernel.
- **Touches** (additive + small wiring edits):
  - `docs/adr/0206-ssimulacra2-cuda-sycl.md` and the index row in
    `docs/adr/README.md`.
  - `libvmaf/src/feature/cuda/ssimulacra2_cuda.{c,h}` — new CUDA
    dispatch.
  - `libvmaf/src/feature/cuda/ssimulacra2/ssimulacra2_blur.cu` and
    `ssimulacra2_mul.cu` — new CUDA fatbins.
  - `libvmaf/src/feature/sycl/ssimulacra2_sycl.cpp` — new SYCL
    extractor.
  - `libvmaf/src/feature/feature_extractor.c` — two new extern
    declarations + two new entries in `feature_extractor_list[]`.
  - `libvmaf/src/meson.build` — adds `ssimulacra2_blur` +
    `ssimulacra2_mul` to `cuda_cu_sources`, introduces (or
    extends, if PR #157 / [ADR-0202](adr/0202-float-adm-cuda-sycl.md)
    landed first) the `cuda_cu_extra_flags` map with a
    `ssimulacra2_blur` entry, threads `per_kernel_flags` into the
    fatbin custom-target, and lists the two new C / CPP TUs.
  - `libvmaf/src/cuda/AGENTS.md` and `libvmaf/src/sycl/AGENTS.md` —
    rebase invariant notes for the per-kernel `--fmad=false` flag
    and the `-fp-model=precise` SYCL build flag.
  - `docs/backends/cuda/overview.md`,
    `docs/backends/sycl/overview.md`,
    `docs/metrics/features.md` — coverage matrix updates.
  - `CHANGELOG.md` Unreleased § Added.
- **Invariants** (load-bearing on rebase):
  1. **Per-kernel `--fmad=false` for `ssimulacra2_blur`.** The
     IIR's `o = n2 * sum - d1 * prev1 - prev2` must NOT fuse
     into FMAs — without the flag the recursive Gaussian's
     per-step rounding compounds across the 6-scale pyramid past
     `places=4`.
  2. **`-fp-model=precise` on the SYCL feature build line.**
     Removing it drifts `ssimulacra2_sycl` past `places=2`
     through the IIR.
  3. **Hybrid host/GPU split mirrors Vulkan.** Host runs YUV→RGB,
     XYB, downsample, and SSIM/EdgeDiff combine in double; GPU
     runs only mul + IIR blur. Any future PR that ports XYB or
     YUV→RGB onto the GPU MUST land alongside an updated
     [ADR-0206](adr/0206-ssimulacra2-cuda-sycl.md) and
     re-validate `places=4` on every Netflix CPU pair.
  4. **CUDA fex uses `.extract` (synchronous), not
     `.submit`/`.collect`.** Per-frame raw YUV is D2H-copied
     from `picture_cuda`'s device-side `VmafPicture.data[]` into
     pinned host scratch via `cuMemcpy2DAsync`. Skipping the
     copy segfaults — direct host reads on a `CUdeviceptr` are
     the failure mode the prior agent's WIP hit.
- **On upstream sync**: zero interaction with Netflix. The GPU
  coverage matrix for `ssimulacra2` is wholly fork-local.
- **Re-test on rebase**:

  ```bash
  meson setup build_cuda libvmaf -Denable_cuda=true -Denable_sycl=false
  ninja -C build_cuda

  python3 scripts/ci/cross_backend_vif_diff.py \
    --vmaf-binary ./build_cuda/tools/vmaf \
    --feature ssimulacra2 --backend cuda --places 4 \
    --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
    --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
    --width 576 --height 324 --pixel-format 420 --bitdepth 8
  # Expect: 0/48 mismatches, max_abs_diff ~1e-6.
  ```

### 0061 — cambi GPU feasibility spike (ADR-0205)

- **ADR**: [ADR-0205](adr/0205-cambi-gpu-feasibility.md).
- **Research digest**: [`docs/research/0020-cambi-gpu-strategies.md`](research/0020-cambi-gpu-strategies.md).
- **Upstream source**: fork-local. Netflix/vmaf has no Vulkan backend.
- **Touches** (additive only):
  - `docs/adr/0205-cambi-gpu-feasibility.md`, `docs/research/0020-cambi-gpu-strategies.md`, `docs/adr/README.md` index row.
  - `libvmaf/src/feature/vulkan/cambi_vulkan.c` — new dormant scaffold (not yet in `vulkan_sources`, not yet registered).
  - `libvmaf/src/feature/vulkan/shaders/cambi_{derivative,decimate,filter_mode}.comp` — new reference GLSL shaders, not yet in the build's `shaders` list.
  - `libvmaf/src/feature/AGENTS.md` invariants + `CHANGELOG.md` bullet.
- **Invariants** (rebase-relevant):
  1. **Hybrid host/GPU port by decision.** If Netflix upstream tightens the c-value formula or histogram update protocol, the host residual call site in the eventual `cambi_vulkan.c::cambi_vulkan_extract` must be updated alongside `cambi.c::calculate_c_values` — the same code is reused. Do NOT translate the c-values phase to GPU during any upstream-port PR; that optimisation belongs to the v2 strategy-III PR (deferred).
  2. **Scaffolds dormant in the spike PR.** The `cambi_vulkan.c` extractor returns `-ENOSYS` from `cambi_vulkan_init_stub` until the integration follow-up wires it in. Do NOT register `vmaf_fex_cambi_vulkan_scaffold` in `feature_extractor.c`'s list.
  3. **Shaders not in the build's shader list.** Adding them to `libvmaf/src/vulkan/meson.build`'s `vulkan_shaders` list before the integration PR produces orphaned `*_spv.h` headers. Leave them alone in this spike PR.
- **On upstream sync**: zero interaction. cambi.c itself is upstream-mirrored — Netflix changes flow through `port-upstream-commit`; only the integration PR's host residual call site needs paired attention.
- **Re-test on rebase**:

  ```bash
  meson setup build -Denable_vulkan=enabled -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build
### 0059 — Tiny-AI Netflix corpus training prep (ADR-0203)

- **ADR**: [ADR-0203](adr/0203-tiny-ai-training-prep-impl.md).
- **Upstream source**: fork-local. Netflix/vmaf has no equivalent
  training surface.
- **Touches**:
  - [`ai/data/`](../ai/data/) — Netflix loader, libvmaf-CLI feature
    extractor, distillation scoring.
  - [`ai/train/`](../ai/train/) — PyTorch dataset, eval harness,
    Lightning-style training entry point.
  - [`ai/scripts/run_training.sh`](../ai/scripts/run_training.sh) —
    convenience wrapper.
  - [`ai/tests/`](../ai/tests/) — five new pytest modules
    (`test_netflix_loader.py`, `test_dataset.py`, `test_eval.py`,
    `test_train_smoke.py`, plus `conftest.py`).
  - [`docs/ai/training.md`](ai/training.md) — new "C1 (Netflix
    corpus)" section; existing sections untouched.
  - [`ai/AGENTS.md`](../ai/AGENTS.md) — invariants section added.
- **Invariants** (load-bearing):
  1. **Filename ladder regex is fork-specific.**
     `<source>_<quality>_<height>_<bitrate>.yuv` (dis) +
     `<source>_<fps>fps.yuv` (ref). Upstream may publish a different
     naming convention later; do NOT merge them — keep this loader
     scoped to the Netflix corpus, add a sibling loader for any
     upstream alternative.
  2. **Per-clip cache schema is consumed by both dataset and any
     downstream tooling.** Schema is
     `{features:{feature_names, per_frame, n_frames},
       scores:{per_frame, pooled}}`. Any change must invalidate
     `$VMAF_TINY_AI_CACHE` (delete or version-tag the directory).
  3. **Smoke command stays runnable without a built `vmaf` binary.**
     The `_make_zero_payload` helper in `ai.train.dataset` injects
     a fake payload for `--epochs 0` so CI gates don't drag a libvmaf
     build into the Python test surface.
  4. **YUV size probe never silently guesses.** `probe_yuv_dims`
     either matches the 1920x1080 default, returns ffprobe's answer,
     or raises. Tests pass `assume_dims=(16, 16)` explicitly for
     synthetic fixtures.
- **On upstream sync**: no interaction with upstream. The `ai/`
  subtree is wholly fork-local.
- **Re-test on rebase**:

  ```bash
  python -m pytest ai/tests/test_netflix_loader.py \
      ai/tests/test_dataset.py ai/tests/test_eval.py \
      ai/tests/test_train_smoke.py -v
  python ai/train/train.py --epochs 0 --data-root /tmp/mock_corpus \
      --assume-dims 16x16 --val-source BetaSrc --out-dir /tmp/out
  ```

### 0073 — Tiny-AI QAT trainer + first per-model QAT pass (T5-4)

- **ADR**: [ADR-0207](adr/0207-tinyai-qat-design.md) (design),
  [ADR-0208](adr/0208-learned-filter-v1-qat-impl.md) (per-model impl).
- **Touches**: `ai/train/qat.py` (new), `ai/scripts/qat_train.py`
  (rewrite from `NotImplementedError` scaffold),
  `ai/configs/learned_filter_v1_qat.yaml` (new),
  `ai/tests/test_qat_smoke.py` (new), `docs/ai/quantization.md`
  (QAT tier added). All paths are wholly fork-local; no upstream
  Netflix/vmaf interaction.
- **Invariants**:
  1. **Two-step pipeline (PyTorch QAT → fp32 ONNX → ORT
     static-quantize) is load-bearing.** Both the legacy ONNX
     exporter (`quantized::conv2d`) and the new TorchDynamo
     exporter (`Conv2dPackedParamsBase.__obj_flatten__`) refuse
     to consume `convert_fx` output on PyTorch 2.11. The bridge
     (state-dict diff to a fresh fp32 module + ORT static-quantize)
     is the only path that yields a QDQ ONNX. Do NOT collapse to
     a single-step `convert_fx → torch.onnx.export` until both
     PyTorch issues are fixed; re-check both exporters on each
     PyTorch upgrade.
  2. **State-dict transfer matches by submodule name + shape.**
     `_copy_qat_weights_into_fp32` walks `fp32_state` keys, finds
     the same key in the FX-prepared module, copies the tensor.
     Tiny-AI models today have stable submodule names (`entry`,
     `body.*`, `exit`); a model architecture that uses
     top-level `nn.Sequential` would break this because
     `prepare_qat_fx` renames Sequential children to numeric
     indices. The `RuntimeError("0 tensors copied")` guard catches
     the silent failure mode.
  3. **FX preparation runs on CPU.** PyTorch 2.11's FX symbolic
     tracer is flaky on CUDA buffers; the trainer migrates the
     model to CPU before `prepare_qat_fx` and back to the
     accelerator for the fine-tune phase. The smoke test
     deliberately exercises the CPU path so this stays covered.
  4. **`torch.ao.quantization` deprecation will hard-fail in
     PyTorch 2.10**. Migration target is
     `torchao.quantization.pt2e` (`prepare_pt2e` /
     `convert_pt2e`); the two-step pipeline is mostly
     pt2e-compatible — only the FX-prep call changes.
- **On upstream sync**: no interaction with upstream. The `ai/`
  subtree is fully fork-local.
- **Re-test on rebase**:

  ```bash
  python -m pytest ai/tests/test_qat_smoke.py -v
  python ai/scripts/qat_train.py \
      --config ai/configs/learned_filter_v1_qat.yaml \
      --output /tmp/qat_smoke.int8.onnx --smoke
  ```

### 0074 — GPU-parity matrix CI gate (T6-8 / ADR-0214)

- **Touched surfaces (fork-local)**:
  `scripts/ci/cross_backend_parity_gate.py` (new),
  `.github/workflows/tests-and-quality-gates.yml` (new
  `vulkan-parity-matrix-gate` job),
  `docs/development/cross-backend-gate.md` (new),
  `docs/backends/index.md` (cross-backend section),
  `libvmaf/AGENTS.md` (rebase-sensitive invariant note).
- **Why this matters on rebase**: the CI lane and the matrix-gate
  script are entirely fork-local. Upstream Netflix/vmaf has no
  comparable gate; conflicts on rebase are restricted to the CI
  workflow file when upstream rearranges its own jobs. The gate's
  Python script lives outside `libvmaf/src/` so the upstream-sync
  path doesn't see it.
- **Invariants the gate enforces**:
  1. Per-feature absolute tolerance is declared in one place
     (`FEATURE_TOLERANCE` in
     [`scripts/ci/cross_backend_parity_gate.py`](../scripts/ci/cross_backend_parity_gate.py)).
     Tightening a tolerance requires a measurement-driven
     follow-up ADR; loosening requires a justification ADR
     (CLAUDE.md §12 r1).
  2. The legacy single-feature gate
     [`scripts/ci/cross_backend_vif_diff.py`](../scripts/ci/cross_backend_vif_diff.py)
     stays for one release cycle. Sister PRs in this session add
     to it; the T6-8b cleanup PR deletes it once the matrix gate
     has soaked.
  3. CUDA / SYCL / hardware-Vulkan are advisory until a
     self-hosted runner is registered. The script supports them
     via `--backends`; flipping the CI lane to required is a
     follow-up wiring change, not a code change.
- **On upstream sync**: no interaction with upstream
  `tests-and-quality-gates.yml` (the gate job is fork-added);
  rebase conflicts limited to insertion-order in the workflow
  file.
- **Re-test on rebase**:

  ```bash
  cd libvmaf && meson setup build \
      -Denable_cuda=false -Denable_sycl=false \
      -Denable_vulkan=enabled -Denable_float=true \
      --buildtype=release && ninja -C build
  cd ..
  python3 scripts/ci/cross_backend_parity_gate.py \
      --vmaf-binary libvmaf/build/tools/vmaf \
      --reference testdata/ref_576x324_48f.yuv \
      --distorted testdata/dis_576x324_48f.yuv \
      --width 576 --height 324 --backends cpu vulkan \
      --json-out /tmp/parity.json --md-out /tmp/parity.md
  ```

### 0220 — SYCL feature kernels are unconditionally fp64-free (T7-17)

- **Touches**: `libvmaf/src/sycl/common.cpp` (init log line),
  `libvmaf/src/sycl/AGENTS.md` (new invariant row), all SYCL
  feature kernels under `libvmaf/src/feature/sycl/` (no diff
  today, but the contract pins their shape going forward).
- **Invariant**: every SYCL feature-kernel lambda captures and
  operates on `float` / integer types only. No `double` operand
  inside a `parallel_for` body, no `sycl::reduction<double>`, no
  `sycl::plus<double>`. A single fp64 instruction in the TU's
  SPIR-V module causes the Level Zero runtime to reject the
  entire module on Intel Arc A-series and other fp64-less
  devices, even when the offending kernel is never submitted.
  Host-side `double` (in `extract` / `flush` post-processing,
  score aggregation, log10 normalisation) remains fine.
  Concrete patterns in tree: ADM gain limiting via int64 Q31
  (`gain_limit_to_q31` + `launch_decouple_csf<false>` in
  `integer_adm_sycl.cpp`); VIF gain limiting via fp32
  `sycl::fmin`; CIEDE / SSIM accumulators via
  `sycl::reduction<int64_t>` / `sycl::plus<int64_t>`.
- **On upstream sync**: Netflix/vmaf has no SYCL backend
  upstream; conflicts cannot enter via `git merge`. The risk is
  a fork-local cherry-pick (e.g. a SYCL twin of a new CUDA
  kernel) bringing a `double` into a kernel lambda. Audit the
  lambda capture list and any `sycl::reduce*` calls against
  this invariant before merging.
- **Re-test on rebase**:

  ```bash
  # Build SYCL backend
  meson setup build-sycl libvmaf -Denable_sycl=true CC=icx CXX=icpx
  ninja -C build-sycl

  # On an fp64-less device (e.g. Intel Arc A380), confirm the
  # init log line is INFO-level and reads "device lacks native
  # fp64 — kernels already use fp32 + int64 paths, no emulation
  # overhead". The SYCL kernels must launch successfully (no
  # SPIR-V module rejection from the Level Zero runtime).
  build-sycl/tools/vmaf --reference testdata/ref_576x324_48f.yuv \
      --distorted testdata/dis_576x324_48f.yuv \
      --width 576 --height 324 --backend sycl \
      --feature integer_vif --feature integer_adm \
      --output /tmp/sycl-fp64less.json --json
  ```

### 0091 — T6-9 model registry schema + `--tiny-model-verify` (ADR-0211)


- **No rebase impact: 100% fork-local surface.** The registry
  (`model/tiny/registry.json`), its JSON Schema
  (`model/tiny/registry.schema.json`), the `--tiny-model-verify`
  CLI flag, and the `vmaf_dnn_verify_signature()` C entry point
  are entirely fork-local — none of these paths exist in upstream
  Netflix/vmaf. Listed here for completeness so a future
  `/sync-upstream` run sees the surface area was acknowledged.
- **Touches** (additive only): `model/tiny/registry.json`,
  `model/tiny/registry.schema.json`,
  `ai/scripts/validate_model_registry.py`,
  `libvmaf/src/dnn/model_loader.{c,h}` (added
  `vmaf_dnn_verify_signature()`),
  `libvmaf/include/libvmaf/dnn.h` (public declaration),
  `libvmaf/tools/cli_parse.{c,h}` (`ARG_TINY_MODEL_VERIFY` +
  `tiny_model_verify` field), `libvmaf/tools/vmaf.c` (call site),
  `libvmaf/test/dnn/test_tiny_model_verify.c`,
  `python/test/model_registry_schema_test.py`,
  `docs/ai/model-registry.md`, `docs/ai/inference.md`,
  `docs/ai/security.md`, `docs/adr/0209-...md`,
  `docs/adr/README.md` (index row), `CHANGELOG.md`,
  `libvmaf/src/dnn/AGENTS.md`.
- **Invariants** (rebase-relevant):
  1. **Schema is the contract.** New registry fields land in
     `registry.schema.json` *first*, then in `registry.json`,
     then in any consumers (the C-side parser, the Python
     validator, the MCP). Reverse order causes mismatch.
  2. **`schema_version` is bounded.** The schema accepts only
     `{0, 1}`; bump the enum *and* the loader's check together
     when adding `2`.
  3. **Banned-function rule applies.** The `cosign` invocation
     uses `posix_spawnp(3p)` with an explicit argv array. Do not
     replace with `system(3)` / `popen(3)` — both shell-parse the
     command and would re-introduce injection risk.
  4. **Bundle-file absence is fail-closed.** When
     `sigstore_bundle` points at a not-yet-existing file
     (pre-release state), `vmaf_dnn_verify_signature()` returns
     `-ENOENT`. The CLI surfaces this as a load failure; do not
     "soften" to a warning without an explicit ADR.
- **Re-test on rebase**:

  ```bash
  python3 ai/scripts/validate_model_registry.py
  python3 -m pytest python/test/model_registry_schema_test.py -v
  meson test -C build-cpu --suite=dnn
  ```

### 0074 — HIP (AMD ROCm) backend scaffold (T7-10)

- **ADR**: [ADR-0212](adr/0212-hip-backend-scaffold.md).
- **Upstream source**: fork-local. HIP backend is fork-only;
  Netflix/vmaf has no `libvmaf_hip.h` and no `enable_hip` meson
  option.
- **Touches**:
  - `libvmaf/include/libvmaf/libvmaf_hip.h` (new).
  - `libvmaf/include/libvmaf/meson.build` — adds the
    `is_hip_enabled` install gate, mirroring `is_cuda_enabled` /
    `is_sycl_enabled` boolean idioms.
  - `libvmaf/meson_options.txt` — new `enable_hip` boolean option
    (default false).
  - `libvmaf/src/meson.build` — new `is_hip_enabled` flag,
    conditional `subdir('hip')`, `hip_sources` + `hip_deps`
    threaded through `libvmaf_feature_static_lib` (alongside the
    existing CUDA / SYCL / Vulkan aggregations) and the top-level
    `library('vmaf', ...)` `dependencies` list.
  - `libvmaf/src/hip/` (new directory: `common.{c,h}`,
    `picture_hip.{c,h}`, `dispatch_strategy.{c,h}`, `meson.build`).
  - `libvmaf/src/feature/hip/` (new directory: `adm_hip.c`,
    `vif_hip.c`, `motion_hip.c`).
  - `libvmaf/test/test_hip_smoke.c` (new).
  - `libvmaf/test/meson.build` — registers the smoke test under
    `if get_option('enable_hip') == true`.
  - `.github/workflows/libvmaf-build-matrix.yml` — adds
    `Build — Ubuntu HIP (T7-10 scaffold)` row.
  - `docs/backends/hip/overview.md` (new),
    `docs/backends/index.md` (planned → scaffold row),
    `docs/research/0033-hip-applicability.md` (new),
    `docs/adr/0212-hip-backend-scaffold.md` (new),
    `docs/adr/README.md` (new index row).
  - `libvmaf/AGENTS.md` — new "HIP backend scaffold contract"
    rebase-sensitive invariant entry.
  - `CHANGELOG.md` — Unreleased § Added.
- **Invariants** (rebase-relevant):
  1. **`enable_hip` is a `boolean` option, not a `feature`.**
     Mirrors `enable_cuda` / `enable_sycl`; do not "harmonise" with
     `enable_vulkan`'s `feature` / `disabled` form without an ADR
     amendment per ADR-0212 § "Decision".
  2. **Public C-API entry points return `-ENOSYS` for the
     scaffold.** The smoke test
     [libvmaf/test/test_hip_smoke.c](../libvmaf/test/test_hip_smoke.c)
     pins this. A rebase that "succeeds" by accidentally enabling a
     code path (e.g. a refactor that early-returns 0 from
     `vmaf_hip_state_init`) breaks the smoke and the runtime PR's
     contract baseline.
  3. **`hip_sources` is added to
     `libvmaf_feature_static_lib`, NOT directly to the top-level
     `library('vmaf', ...)`.** The static lib is extracted into
     libvmaf via `objects: [..., libvmaf_feature_static_lib.extract_all_objects(recursive: true), ...]`
     at the bottom of `libvmaf/src/meson.build`. Adding `hip_sources`
     to the top library() too would double-link.
  4. **`hip_deps` IS added to the top library() `dependencies:`
     list.** The runtime PR will populate `hip_deps` with the real
     `dependency('hip-lang')` linkage; threading it through the top
     library() ensures consumers see the transitive dependency.
  5. **Header purity**: `libvmaf_hip.h` does not include
     `<hip/hip_runtime.h>`. HIP runtime types cross the public ABI
     as `uintptr_t` (matches the CUDA / Vulkan precedent;
     ADR-0212). Don't add `<hip/...>` includes to the public header
     during a rebase / runtime-PR bring-up.
- **No FFmpeg patch**: the fork's `ffmpeg-patches/` series does
  not currently consume the HIP API surface. CLAUDE §12 r14 only
  requires patch updates when an existing patch consumes the
  surface; the runtime PR (T7-10b) will add the `hip_device`
  filter option and the corresponding patch.
- **On upstream sync**: zero interaction; HIP backend is
  fork-only.
- **Re-test on rebase**:

  ```bash
  cd libvmaf
  meson setup build-hip -Denable_cuda=false -Denable_sycl=false \
                        -Denable_hip=true
  ninja -C build-hip
  meson test -C build-hip test_hip_smoke
  # Expect: 9/9 pass.

  # Default no-HIP build still works:
  meson setup build-cpu -Denable_cuda=false -Denable_sycl=false
  ninja -C build-cpu
  meson test -C build-cpu --suite=fast
  ```

### 0074 — SSIMULACRA 2 SVE2 SIMD parity (T7-38)

- **ADR**: [ADR-0213](adr/0213-ssimulacra2-sve2.md).
- **Touches**: `libvmaf/src/feature/arm64/ssimulacra2_sve2.{c,h}`
  (new), `libvmaf/src/feature/ssimulacra2.c` (dispatch table override
  in `init_simd_dispatch`), `libvmaf/src/arm/cpu.{c,h}` (HWCAP2_SVE2
  probe + new `VMAF_ARM_CPU_FLAG_SVE2` enum value),
  `libvmaf/src/meson.build` (cc.compiles probe + optional
  `arm64_ssimulacra2_sve2` static library),
  `libvmaf/test/test_ssimulacra2_simd.c` (SVE2 picker overrides on the
  arm64 path + dispatch diagnostic), `build-aux/aarch64-linux-gnu-sve2.ini`
  (new cross-file pinning `qemu-aarch64-static -cpu max`). All paths
  are wholly fork-local; no upstream Netflix/vmaf code is modified.
- **Invariants**:
  1. **Fixed 4-lane SVE2 predicate.** Every kernel uses
     `svwhilelt_b32(0, 4)` so SIMD arithmetic order is identical to
     the NEON sibling regardless of the runtime vector length. This
     keeps the [ADR-0138](adr/0138-iqa-convolve-avx2-bitexact-double.md) /
     [ADR-0139](adr/0139-ssim-simd-bitexact-double.md) /
     [ADR-0140](adr/0140-simd-dx-framework.md) byte-exact
     contract intact. Do NOT widen the predicate to `svptrue_b32()`
     without a separate ADR + snapshot regen — variable-length lane
     reductions perturb the per-step rounding order.
  2. **NEON stays the fallback.** SVE2 is purely additive; the
     dispatch table assigns NEON first and only overrides on
     `VMAF_ARM_CPU_FLAG_SVE2`. A toolchain that fails the
     `cc.compiles(... -march=armv9-a+sve2)` probe leaves
     `HAVE_SVE2` unset and the legacy NEON-only build is unchanged.
  3. **`-ffp-contract=off` mirrors the NEON sibling.** Without it
     GCC fuses the per-lane scalar tail's `a*b+c` patterns into
     `fmla`, drifting against the SIMD path by ~1 ulp. The
     `arm64_ssimulacra2_sve2` static library carries the flag like
     its NEON counterpart.
- **On upstream sync**: no interaction with upstream — `arm64/`
  feature TUs and the `arm/cpu.{c,h}` flag enum are fork-local.
  An upstream sync that rewrites `init_simd_dispatch` in
  `libvmaf/src/feature/ssimulacra2.c` would also need the SVE2 cases
  preserved.
- **Re-test on rebase**:

  ```bash
  meson setup build-arm64-sve2 libvmaf \
      --cross-file=build-aux/aarch64-linux-gnu-sve2.ini -Denable_asm=true
  ninja -C build-arm64-sve2 test/test_ssimulacra2_simd
  meson test -C build-arm64-sve2 test_ssimulacra2_simd
  # stderr should report `ssimulacra2 simd dispatch: NEON=1 SVE2=1`
  # and 11/11 tests should pass.
  ```

### 0075 — `enable_lcs` MS-SSIM extras on CUDA + Vulkan (T7-35 / ADR-0243)

- **Touched surfaces (fork-local)**:
  [`libvmaf/src/feature/cuda/integer_ms_ssim_cuda.c`](../libvmaf/src/feature/cuda/integer_ms_ssim_cuda.c)
  (added `enable_lcs` to `MsSsimStateCuda` + `options[]` + 15 host-side
  `vmaf_feature_collector_append` calls gated on the bool),
  [`libvmaf/src/feature/vulkan/ms_ssim_vulkan.c`](../libvmaf/src/feature/vulkan/ms_ssim_vulkan.c)
  (rewrote `enable_lcs` help text + added `emit_lcs_metrics` helper +
  gated 15 `vmaf_feature_collector_append` calls),
  [`scripts/ci/cross_backend_vif_diff.py`](../scripts/ci/cross_backend_vif_diff.py)
  + [`scripts/ci/cross_backend_parity_gate.py`](../scripts/ci/cross_backend_parity_gate.py)
  (new `float_ms_ssim_lcs` pseudo-feature + `FEATURE_ALIASES` map
  + `places=4` tolerance row).
- **Why this matters on rebase**: the GPU MS-SSIM extractors are
  fork-local (Netflix upstream has no Vulkan or CUDA MS-SSIM kernel
  today). The `enable_lcs` semantic and the metric names
  (`float_ms_ssim_{l,c,s}_scale{0..4}`) **must match** the upstream
  CPU reference at
  [`libvmaf/src/feature/float_ms_ssim.c:189-221`](../libvmaf/src/feature/float_ms_ssim.c#L189-L221).
  If upstream ever renames or reorders those metrics, mirror the
  change on the GPU side in the same merge — public-API contract.
- **Invariants the contract enforces**:
  1. Default-path output (`enable_lcs=false`) stays bit-identical
     to the pre-T7-35 binary: only the host-side appends are gated;
     no kernel / shader / device-buffer changes.
  2. Metric ordering is metric-wise (all `l_scale*` first, then
     `c_*`, then `s_*`) — matches the CPU emission order.
  3. `places=4` cross-backend tolerance per
     [ADR-0190](adr/0190-ms-ssim-vulkan.md); enforced by the new
     `float_ms_ssim_lcs` cell in the parity matrix gate
     ([ADR-0214](adr/0214-gpu-parity-ci-gate.md)).
- **On upstream sync**: zero interaction; the GPU twins do not
  exist upstream. The CPU `float_ms_ssim.c` is shared with upstream
  but `enable_lcs` is upstream-stable since v3.0.0.
- **Re-test on rebase**:

  ```bash
  cd libvmaf && meson setup build-vulkan \
      -Denable_cuda=false -Denable_sycl=false \
      -Denable_vulkan=enabled -Denable_float=true \
      --buildtype=release && ninja -C build-vulkan
  cd ..
  python3 scripts/ci/cross_backend_vif_diff.py \
      --vmaf-binary libvmaf/build-vulkan/tools/vmaf \
      --reference testdata/ref_576x324_48f.yuv \
      --distorted testdata/dis_576x324_48f.yuv \
      --width 576 --height 324 \
      --feature float_ms_ssim_lcs --backend vulkan --places 4
  ```

### 0075 — 32-bit ADM/cpu fallbacks port (T-NEW-3)

- **Touched surfaces (upstream-mirror)**:
  `libvmaf/src/feature/x86/adm_avx2.c`,
  `libvmaf/src/feature/x86/adm_avx512.c`,
  `libvmaf/src/x86/cpu.c`. Cherry-picks of upstream
  `8a289703` (Christopher Degawa, "adm: add fallback for
  extract_epi64 for 32-bit") and `1b6c3886` ("x86/cpu:
  remove limit of avx+ on 32-bit").
- **Why this matters on rebase**: trivially conflict-free with
  any future upstream `extract_epi64` work because we land
  upstream's exact `extract_epi64` macro/inline-fn pair.
  The conflict surface is the fork's clang-format-100col
  layout in `adm_avx2.c` / `adm_avx512.c` and the
  `_Alignas(64)` LTO-correctness slot in `adm_avx512.c`
  (`docs/development/known-upstream-bugs.md`); both are
  preserved verbatim.
- **Invariants the port preserves**:
  1. `_Alignas(64) int64_t angle_flag[16]` in
     `adm_decouple_s123_avx512` stays — without it, LTO
     can promote the unaligned load to `vmovdqa64` and
     fault under `--buildtype=release -Db_lto=true`.
  2. The `extract_epi64` symbol must remain resolved on
     both `__x86_64__` (macro to `_mm256_extract_epi64`)
     and 32-bit (fallback inline). If a future upstream
     change inlines the helper differently, keep the
     conditional definition.
- **On upstream sync**: if Netflix ships further 32-bit
  fallbacks (motion / psnr — not in this port), expect a
  parallel `extract_epi64`-style helper at the top of each
  affected SIMD file. The fork should mirror those verbatim
  into the same files.
- **Re-test on rebase**:

  ```bash
  meson setup build-i686 libvmaf \
      --cross-file=build-aux/i686-linux-gnu.ini \
      -Denable_asm=false
  ninja -C build-i686
  meson setup build-cpu libvmaf -Denable_avx512=true
  ninja -C build-cpu
  meson test -C build-cpu
  ```

### 0076 — codec-aware FR regressor surface (T7-CODEC-AWARE / ADR-0235)

- **Touches**: `ai/src/vmaf_train/codec.py` (new),
  `ai/src/vmaf_train/models/fr_regressor.py` (extended),
  `ai/scripts/bvi_dvc_to_full_features.py`,
  `ai/scripts/extract_full_features.py`. No upstream-shared paths.
- **Invariant**: `CODEC_VOCAB` in `ai/src/vmaf_train/codec.py` is
  closed and **order-stable** — the index of each codec is the
  one-hot column index baked into trained ONNX. Adding a codec
  appends to the tuple and bumps `CODEC_VOCAB_VERSION`; reordering
  silently invalidates every shipped `fr_regressor_v2_*.onnx`.
  `FRRegressor(num_codecs=0)` must remain the v1 single-input
  contract — flipping the default would break every existing
  `model/tiny/fr_regressor_v1.onnx` consumer.
- **Re-test**: `pytest ai/tests/test_codec_aware_fr.py -v` (8 sub-tests
  covering vocabulary contract + alias table + back-compat). Pure
  fork-local addition; no upstream rebase impact for the next
  `/sync-upstream`.

### 0075 — feature/speed extractors (T-NEW-1, upstream port d3647c73)

- **Touches**: `libvmaf/src/feature/speed.c` (new),
  `libvmaf/src/feature/picture_copy.{c,h}` (signature change —
  added `int channel` parameter), `libvmaf/src/feature/float_*.c`
  call sites updated to pass `channel=0`,
  `libvmaf/src/feature/feature_extractor.c` registry block,
  `libvmaf/src/feature/alias.c`, `libvmaf/src/meson.build`,
  `libvmaf/src/feature/vif_tools.{c,h}` (helper-function port from
  upstream `4ad6e0ea`).
- **Upstream source**: verbatim cherry-pick of Netflix/vmaf
  `d3647c73` ("feature/speed: port speed_chroma and speed_temporal
  extractors") with its dependency `4ad6e0ea` ("feature/vif: port
  helper functions"). Both are pre-existing on Netflix master and
  enter the fork as part of the T7-4 audit catch-up.
- **Invariant**: `picture_copy()` now takes a `channel` argument —
  every fork-local extractor that calls it (CUDA `integer_ms_ssim`,
  Vulkan `ssim` / `ms_ssim`) passes `channel=0`. If upstream later
  evolves the signature again (e.g. adds bit-depth or stride
  validation), update those fork-local call sites in lockstep.
  Speed extractors only register when `VMAF_FLOAT_FEATURES=1`
  (build with `-Denable_float=true`).
- **On upstream sync**: future Netflix commits in
  `libvmaf/src/feature/speed.c` apply cleanly because the file is
  now a verbatim mirror; conflict potential is limited to the
  registry block in `feature_extractor.c` (interleave with the
  fork's Vulkan / SYCL / CUDA blocks) and to any further
  `picture_copy` signature evolution.
- **Re-test on rebase**:

  ```bash
  meson setup build-cpu libvmaf -Denable_cuda=false \
      -Denable_sycl=false -Denable_float=true
  ninja -C build-cpu
  meson test -C build-cpu test_speed
  meson test -C build-cpu              # full meson suite
  make test-netflix-golden             # 3 CPU canonical pairs
### 0221 — CHANGELOG + ADR-index fragment-file pattern (T7-39 / ADR-0221)

- **What changed**: the fork stopped editing `CHANGELOG.md` and
  `docs/adr/README.md` directly. Both files are now rendered from
  fragment trees:
  - `changelog.d/<section>/<topic>.md` (Keep-a-Changelog sections),
    plus the migration archive `changelog.d/_pre_fragment_legacy.md`.
  - `docs/adr/_index_fragments/<NNNN-slug>.md`, plus
    `docs/adr/_index_fragments/_order.txt` (frozen commit-merge order
    manifest) and `docs/adr/_index_fragments/_header.md` (table prelude).
  Two scripts render the consolidated outputs:
  - `scripts/release/concat-changelog-fragments.sh --check|--write`
  - `scripts/docs/concat-adr-index.sh --check|--write`
- **On upstream sync**: zero interaction — `CHANGELOG.md` is a
  fork-local Markdown surface (Netflix upstream doesn't ship a
  Keep-a-Changelog file in this format), and `docs/adr/` is entirely
  fork-local. A `/sync-upstream` run will not touch the fragment trees.
- **Re-test on rebase**:

  ```bash
  bash scripts/release/concat-changelog-fragments.sh --check
  bash scripts/docs/concat-adr-index.sh --check
  # both must exit 0; otherwise run --write and re-stage.
  ```

### 0077 — DISTS extractor proposal (T7-DISTS / ADR-0236)

- **What landed**: ADR-0236 (Proposed) + Research-0043 design digest
  + ADR README index row + CHANGELOG entry.
- **Rebase impact**: pure fork-local proposal-stage docs; no code,
  no Netflix-mirror file touched, no ffmpeg-patches change, no
  public C-API surface change.
- **Reproducer (when implementation lands as T7-DISTS)**:

  ```sh
  vmaf --feature dists_sq=model_path=model/tiny/dists_sq.onnx \
       --reference ref.yuv --distorted dist.yuv \
       --width 1920 --height 1080 --pix_fmt yuv420p
### 0076 — GPU-gen ULP calibration head (proposal-stage, T7-GPU-ULP-CAL / ADR-0234)

- **What landed**: ADR-0234 (Proposed), Research-0041, data-collection
  scaffold at `ai/scripts/collect_gpu_calibration_data.py`, forward-pointer
  in `docs/usage/cli.md` for the future `--gpu-calibrated` flag.
- **Rebase impact**: pure fork-local (proposal docs + Python script);
  no upstream Netflix/vmaf code touched, no public C-API changes, no
  ffmpeg-patches changes.
- **Reproducer**:

  ```sh
  python3 ai/scripts/collect_gpu_calibration_data.py --smoke
### 0095 — Per-backend GPU kernel scaffolding templates (CUDA + Vulkan, ADR-0246)

- **ADR**: [ADR-0246](adr/0246-gpu-kernel-template.md).
- **Touches**:
  - `libvmaf/src/cuda/kernel_template.h` (new, header-only).
  - `libvmaf/src/vulkan/kernel_template.h` (new, header-only).
  - `libvmaf/src/cuda/AGENTS.md` (new invariant row + dir listing).
  - `libvmaf/src/vulkan/AGENTS.md` (new file).
  - `docs/backends/kernel-scaffolding.md` (new).
  - `docs/adr/0246-gpu-kernel-template.md` (new).
  - `CHANGELOG.md`, `docs/adr/README.md`.
  All paths are wholly fork-local. Upstream Netflix/vmaf has no
  Vulkan backend at all today and the CUDA backend uses different
  per-kernel scaffolding shapes; nothing here can collide on a
  pure upstream sync.
- **Invariants**:
  1. **Templates are unused at PR-merge time.** `kernel_template.h`
     in both `libvmaf/src/cuda/` and `libvmaf/src/vulkan/` lands
     with zero call-sites. Each future kernel migration is its
     own gated PR (`places=4` cross-backend-diff per
     [ADR-0214](adr/0214-gpu-parity-ci-gate.md)). Do not bulk-port
     existing kernels onto the templates in a single sync — that
     would short-circuit the per-kernel gate.
  2. **Per-backend, not cross-backend.** Resist the urge to merge
     the two templates into a unified `gpu/kernel_template.h`.
     CUDA async-stream + event vs Vulkan command-buffer + fence +
     descriptor-pool share no concrete shape; a unified API would
     be lowest-common-denominator.
  3. **Helper functions, not macros.** The header bodies are
     `static inline` functions for cuda-gdb / Nsight / RenderDoc
     step-debugging. The `CHECK_CUDA_GOTO` / `CHECK_CUDA_RETURN`
     macros in `cuda_helper.cuh` stay where they pay off (textual
     `goto label`), and the templates use them internally.
- **On upstream sync**: no interaction with upstream paths. An
  upstream sync that touches `libvmaf/src/cuda/common.h` or
  `picture_cuda.h` may shift the helper signatures the template
  consumes (`vmaf_cuda_buffer_alloc`, `vmaf_cuda_picture_get_stream`,
  …); update the template if so.
- **Re-test on rebase**:

  ```bash
  # CUDA build (configure inside libvmaf/ — see CLAUDE.md §2 note).
  meson setup libvmaf/build-cuda libvmaf \
      -Denable_cuda=true -Denable_nvcc=true \
      -Denable_vulkan=disabled -Denable_sycl=false
  ninja -C libvmaf/build-cuda
  meson test -C libvmaf/build-cuda

  # Vulkan build.
  meson setup libvmaf/build-vulkan libvmaf \
      -Denable_vulkan=enabled -Denable_cuda=false -Denable_sycl=false
  ninja -C libvmaf/build-vulkan
  meson test -C libvmaf/build-vulkan
### 0222 — `vmaf-perShot` per-shot CRF predictor sidecar (T6-3b)

- **Touches**: `libvmaf/tools/meson.build` (new executable +
  test wiring), `libvmaf/tools/vmaf_per_shot.c` (new file —
  fork-local, no upstream sibling), `libvmaf/tools/test/meson.build`
  (test row), `libvmaf/tools/test/test_vmaf_per_shot.sh` (new
  smoke test), `libvmaf/tools/AGENTS.md` (sidecar invariants),
  `docs/usage/cli.md` (cross-link), `docs/usage/vmaf-perShot.md`
  (new user doc), `docs/ai/roadmap.md` (T6-3b row update).
- **Invariant**: the sidecar must stay **standalone** — it does
  not link the libvmaf metric path. Any upstream patch that tries
  to fold per-shot CRF prediction into `vmaf_score_*` would
  collapse the encoder-hint vs. quality-score separation
  recorded in roadmap §2.4 and ADR-0222 §Decision. The
  CSV / JSON column set (`shot_id`, `start_frame`, `end_frame`,
  `frames`, `mean_complexity`, `mean_motion`, `predicted_crf`)
  is the public schema; downstream encoders consume it directly.
- **Conflict expectation on `/sync-upstream`**: low. Upstream
  Netflix has no per-shot CRF predictor in tree, so there is no
  natural collision point — `tools/meson.build` is the only
  mutually-edited file and the new `executable('vmaf-perShot', …)`
  block is appended after `vmaf_bench_deps`, well clear of
  upstream's likely additions.
- **Reproducer**:

  ```bash
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false \
      -Denable_vulkan=disabled
  ninja -C build
  meson test -C build test_vmaf_per_shot --print-errorlogs
  ./build/tools/vmaf-perShot \
      --reference testdata/ref_576x324_48f.yuv \
      --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
      --output /tmp/plan.csv
  cat /tmp/plan.csv
### 0075 — `vmaf-roi` sidecar binary (T6-2b / ADR-0247)

- **Touches**:
  - `libvmaf/tools/meson.build` — adds the `vmaf_roi` executable
    target (after the existing `vmaf` target, before `vmaf_bench`).
    Append-only; no upstream-shared lines moved or removed.
  - `libvmaf/test/meson.build` — adds the `test_vmaf_roi`
    executable + `test()` registration. Append-only.
  - `libvmaf/tools/vmaf_roi.c` — wholly new, fork-local.
  - `libvmaf/tools/vmaf_roi_core.h` — wholly new, fork-local.
  - `libvmaf/test/test_vmaf_roi.c` — wholly new, fork-local.
- **Invariant**: the `vmaf-roi` sidecar emits two byte-exact formats
  that downstream encoder drivers (x265 `--qpfile`, SVT-AV1
  `--roi-map-file`) will hard-depend on:
  1. **x265 ASCII grid** — two `#`-prefixed header lines
     (`# vmaf-roi qpfile (x265, --qpfile-style)` and
     `# frame=N ctu=S cols=C rows=R strength=F.FFF`),
     space-separated signed integers, one row per CTU row, `\n`
     terminator.
  2. **SVT-AV1 raw binary** — exactly `cols * rows` bytes of
     `int8_t`, row-major, **no header**.
  3. **QP-offset clamp** — `+-12` (`VMAF_ROI_CORE_QP_OFFSET_MAX`).
  4. **Reduction** — per-CTU **mean** (not max). Switching to max or
     a percentile changes every downstream encoder result and
     requires its own ADR.
  5. **Pure helpers in `vmaf_roi_core.h`** — the per-CTU mean reducer
     and saliency-to-QP mapper are `static inline` in a header so
     `test_vmaf_roi` compiles them without dragging the libvmaf
     link surface in. Moving them into a `.c` TU breaks the test
     wiring.
- **On upstream sync**: no interaction with upstream — `tools/` is a
  fork-local surface from upstream's perspective (upstream ships
  `vmaf.c` only). An upstream sync that rewrites
  `libvmaf/tools/meson.build` should preserve the `vmaf_roi`
  executable block.
- **Re-test on rebase**:

  ```bash
  meson setup build-cpu libvmaf \
      -Denable_cuda=false -Denable_sycl=false -Denable_tools=true
  ninja -C build-cpu tools/vmaf_roi test/test_vmaf_roi
  meson test -C build-cpu test_vmaf_roi
  ./build-cpu/tools/vmaf_roi \
      --reference testdata/ref_576x324_48f.yuv \
      --width 576 --height 324 --frame 0 --output - \
      --encoder x265 --ctu-size 64 --strength 6.0 | head -3
  # First two lines are the # comment header; row 1 of the grid
  # should be "4 2 1 -1 -1 -1 1 2 4" (placeholder radial map).
### 0219 — motion3 GPU coverage on Vulkan + CUDA + SYCL (T3-15(c) / ADR-0219)

- **What changed**: The `motion` GPU twins
  (`libvmaf/src/feature/vulkan/motion_vulkan.c`,
  `libvmaf/src/feature/cuda/integer_motion_cuda.c`,
  `libvmaf/src/feature/sycl/integer_motion_sycl.cpp`) now emit
  `VMAF_integer_feature_motion3_score` in 3-frame window mode
  (default). Cross-backend gates extended (`scripts/ci/cross_backend_*.py`
  `FEATURE_METRICS["motion"]`).
- **Invariants**:
  1. **`motion3 = host-side scalar post-process of motion2`.** No
     device-side state changes; motion3 is computed on the host in
     `extract()` / `collect()` / `flush()` after the existing SAD
     reduction. The post-processing function
     (`motion3_postprocess_*`) mirrors CPU `integer_motion.c`
     lines 510-560 byte-for-byte:
     `clip(motion_blend(motion2 * fps_weight, blend_factor,
     blend_offset), max_val)` with optional moving-average against
     the *unaveraged* prior blended value.
  2. **`motion_five_frame_window=true` returns `-ENOTSUP` at
     `init()`** on all three GPU backends. The 5-deep blur ring +
     second SAD-pair dispatch remain deferred. Do NOT silently
     fall back to the 3-frame path when the user enables the flag —
     fail loud per CERT C / CLAUDE.md §12 r4.
  3. **CPU motion3 algorithm is the source of truth.** Any port of
     an upstream Netflix change to `integer_motion.c` that touches
     `motion_blend(...)`, the `motion_max_val` clip, or the
     moving-average rule MUST be mirrored in
     `motion3_postprocess_*` across all three GPU files in the
     same PR. The cross-backend gate at `places=4` will catch
     drift, but only after a full GPU run.
- **On upstream sync**: Pure fork-local additions to GPU TUs.
  Upstream Netflix has no GPU motion extractor. The
  `motion_blend_tools.h` header is upstream-mirrored — if a sync
  rewrites the `motion_blend()` formula, regenerate the GPU
  snapshot and re-run the cross-backend gate.
- **Re-test on rebase**:

  ```bash
  # CPU sanity (motion3 emission unchanged)
  ./libvmaf/build/tools/vmaf \
      --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
      --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
      --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
      --feature motion --output /tmp/motion.json --json
  python -c "import json; d=json.load(open('/tmp/motion.json')); \
      print('motion3 frames:', sum(1 for f in d['frames'] \
      if 'integer_motion3' in f.get('metrics', {})))"
  # Expect 49 (one motion3 per frame).

  # Cross-backend gate (Vulkan/lavapipe lane works on every host):
  python scripts/ci/cross_backend_vif_diff.py \
      --feature motion --backend vulkan \
      --ref python/test/resource/yuv/src01_hrc00_576x324.yuv \
      --dis python/test/resource/yuv/src01_hrc01_576x324.yuv \
      --width 576 --height 324 --bitdepth 8 \
      --vmaf-bin libvmaf/build/tools/vmaf
  # Expect: integer_motion / integer_motion2 / integer_motion3 all OK at places=4.
### 0216 — vmaf_tiny_v2 (Phase-3-validated tiny VMAF MLP)

- **Touches**: `model/tiny/registry.json`, `model/tiny/vmaf_tiny_v2.{onnx,json}`,
  `ai/scripts/{train,export,validate}_vmaf_tiny_v2.py`, `ai/AGENTS.md`,
  `libvmaf/test/dnn/{test_vmaf_tiny_v2.py,meson.build}`,
  `docs/ai/{models/vmaf_tiny_v2.md,inference.md,roadmap.md}`,
  `docs/adr/{0244-vmaf-tiny-v2.md,README.md}`, `CHANGELOG.md`. All
  paths are wholly fork-local; no upstream Netflix/vmaf code is
  modified.
- **Invariants**:
  1. **Bundled scaler stats are part of the trust root.** The shipped
     ONNX bakes `(input - mean) / std` as Constant `Sub` + `Div`
     nodes that run before the MLP. Re-exporting **must** go through
     `ai/scripts/export_vmaf_tiny_v2.py`, which pulls `mean` / `std`
     from the trainer checkpoint and writes them as graph
     initialisers. Adding an out-of-band scaler step at runtime
     (e.g., a sidecar JSON consumed by the loader) is forbidden
     without a follow-up ADR — it splits the trust root and
     invalidates the registry sha256 contract.
  2. **Feature column order is fixed.** The graph reads
     `(adm2, vif_scale0, vif_scale1, vif_scale2, vif_scale3, motion2)`
     in exactly this order; reordering breaks the bundled `mean` /
     `std` constants. Any change to the feature set requires a fresh
     Phase-3 chain (Research-0027 → 0028 → 0029 → 0030).
  3. **opset 17.** Matches the sister tiny-AI models
     (`learned_filter_v1`, `nr_metric_v1`, `fastdvdnet_pre`) and the
     ORT op-allowlist baseline. Upgrading requires re-validating the
     `Sub` / `Div` / `Gemm` / `Relu` / `Squeeze` ops against
     `op_allowlist.c`.
- **On upstream sync**: zero interaction. Netflix/vmaf has no
  equivalent surface; an upstream sync that touches
  `libvmaf/src/dnn/` (op-allowlist or model-loader changes) needs to
  preserve `Sub` / `Div` / `Gemm` / `Relu` / `Squeeze` in the
  allowlist for opset 17.
- **Re-test on rebase**:

  ```bash
  bash libvmaf/test/dnn/test_registry.sh
  python3 libvmaf/test/dnn/test_vmaf_tiny_v2.py
  python3 ai/scripts/validate_vmaf_tiny_v2.py \
      --onnx model/tiny/vmaf_tiny_v2.onnx \
      --parquet runs/full_features_netflix.parquet \
      --rows 100 --min-plcc 0.97
  meson test -C build-cpu --suite=dnn
### 0094 — Tiny-AI extractor template (ADR-0250)

- **Touches**: `libvmaf/src/dnn/tiny_extractor_template.h` (new),
  `libvmaf/src/feature/feature_lpips.c`,
  `libvmaf/src/feature/fastdvdnet_pre.c`,
  `libvmaf/src/dnn/AGENTS.md`, `docs/ai/extractor-template.md` (new),
  `docs/adr/0250-tiny-ai-extractor-template.md` (new).
- **Invariants**:
  1. **Helper signatures are wire-format-stable.**
     `vmaf_tiny_ai_resolve_model_path(name, option, env_var)` and
     `vmaf_tiny_ai_open_session(name, path, &out)` produce the
     user-facing log lines `<name>: no model path …` and
     `<name>: vmaf_dnn_session_open(<path>) failed: <rc>` —
     downstream tooling greps these. Don't rename or reorder the
     parameters without bumping every extractor + the recipe doc.
  2. **YUV→RGB is bit-exact.** The shared
     `vmaf_tiny_ai_yuv8_to_rgb8_planes` is a literal move of the
     pre-existing `feature_lpips.c` body (BT.709 limited-range,
     nearest-neighbour chroma upsample). LPIPS / saliency / future
     colour-sensitive tiny-AI scores depend on byte-exact equality
     with the prior ad-hoc copies. Any change to the conversion
     constants or the rounding rule needs a separate ADR + a
     coordinated snapshot regen — `model/tiny/` weights aren't
     re-trained against new colour math casually.
  3. **Option-table macro is plain text substitution.** The
     `VMAF_TINY_AI_MODEL_PATH_OPTION(state_t, help)` macro emits a
     single struct literal — no control flow, no recursion, no
     variadic shenanigans (Power-of-10 rule 1 / rule 9). Don't
     extend it into a multi-option emitter without a fresh ADR.
- **On upstream sync**: zero interaction with upstream —
  `feature_lpips.c` and `fastdvdnet_pre.c` are fork-only files,
  and the new `dnn/tiny_extractor_template.h` lives entirely under
  fork-introduced `libvmaf/src/dnn/`. An upstream sync that
  rewrites unrelated `feature_*.c` files won't conflict.
- **Re-test on rebase**:

  ```bash
  cd libvmaf
  meson setup build-cpu -Denable_cuda=false -Denable_sycl=false
  ninja -C build-cpu
  meson test -C build-cpu --suite=dnn
  meson test -C build-cpu test_lpips test_fastdvdnet_pre
  # All 10 dnn-suite + both extractor tests must pass.
  ```


### 0095 — Vulkan ring-depth tunable (ADR-0251 follow-up #3)

- **PR**: feat/t7-29-followup3-ring-tunable.
- **What rebases need to know**: `VmafVulkanConfiguration` grew an
  additive `unsigned max_outstanding_frames` field. Existing
  zero-initialised configs continue to receive the canonical default
  (`0 → VMAF_VULKAN_RING_DEFAULT == 4`). The clamp helper
  `vmaf_vulkan_clamp_ring_size` moved from `import.c` (file-local
  static) to `vulkan_internal.h` (static inline) so `state_init` and
  `lazy_alloc_ring` share one definition; an upstream sync that
  re-introduces the static in `import.c` would shadow the header
  helper — drop the duplicate, keep the inline.
- **New public symbol**:
  `vmaf_vulkan_state_max_outstanding_frames(const VmafVulkanState *)`
  — read-side accessor for the clamped value. Pure additive surface;
  no upstream collision.
- **On upstream sync**: zero interaction. The ring is wholly
  fork-introduced (ADR-0251); upstream Netflix has no Vulkan backend.
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false \
      -Denable_vulkan=enabled
  ninja -C build
  meson test -C build test_vulkan_async_pending_fence
  # All 8 cases must pass: 4 v2-contract + 4 ring-tunable.
### 0096 — `tools/vmaf-tune/` automation umbrella spec (ADR-0237 / Research-0044)

- **PR**: feat/vmaf-tune-spec.
- **What rebases need to know**: this PR ships only an umbrella ADR
  + research digest under `docs/`. No tracked source code, no
  `tools/vmaf-tune/` directory yet, no Meson changes. An upstream
  sync touching ffmpeg-patches or `libvmaf/` cannot collide with
  this PR.
- **On upstream sync**: zero interaction. Spec-only PR.
- **Re-test on rebase**:

  ```bash
  # No build/test impact — verify the docs render and links are alive:
  ls docs/adr/0237-quality-aware-encode-automation.md \
     docs/research/0044-quality-aware-encode-automation.md
  grep -c '\[ADR-0237\]' docs/adr/README.md
  ```

### 0097 — `test_speed` gated on `enable_float` (fix default-build failure)

- **PR**: fix/test-speed-chroma-registration.
- **What rebases need to know**: `libvmaf/test/meson.build` now wraps
  the `test_speed` executable + `test()` registration in
  `if get_option('enable_float')`. The `speed_chroma` /
  `speed_temporal` extractors live in `speed.c`, which is only
  compiled when `enable_float=true` (the entries in
  `feature_extractor.c` are wrapped in `#if VMAF_FLOAT_FEATURES`),
  so the test's `vmaf_get_feature_extractor_by_name("speed_chroma")`
  returned NULL on a default build (`enable_float=false`).
- **On upstream sync**: zero interaction. `test_speed.c` was added
  fork-side via the Netflix port commit `d3647c73`. The gating
  pattern matches `test_vulkan_*` (`if get_option('enable_vulkan').enabled()`).
- **Re-test on rebase**:

  ```bash
  # default (enable_float=false): test_speed must NOT be in the suite
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false --reconfigure
  ninja -C build
  meson test -C build  # expect: NO test_speed in the run

  # CI shape (enable_float=true): test_speed must run + pass
  meson setup build libvmaf -Denable_float=true --reconfigure
  ninja -C build
  meson test -C build test_speed  # expect: 5/5 pass
  ```

### 0098 — Vulkan picture preallocation surface (ADR-0238)

- **PR**: feat/vulkan-picture-preallocation.
- **What rebases need to know**: ABI grows additively. New public
  surface in `libvmaf/include/libvmaf/libvmaf_vulkan.h`:
  `enum VmafVulkanPicturePreallocationMethod`,
  `VmafVulkanPictureConfiguration`,
  `vmaf_vulkan_preallocate_pictures`, `vmaf_vulkan_picture_fetch`.
  New enumerator
  `VMAF_PICTURE_BUFFER_TYPE_VULKAN_DEVICE` in
  `libvmaf/src/picture.h::VmafPictureBufferType`. New TU
  `libvmaf/src/vulkan/picture_vulkan_pool.c` (~180 LOC); registered
  in `libvmaf/src/vulkan/meson.build`. Fork-internal accessor
  `vmaf_vulkan_state_context()` (declared in `vulkan_internal.h`)
  exposes the imported state's VkInstance/VkDevice to the pool —
  used only by `libvmaf.c::vmaf_vulkan_preallocate_pictures`.
- **`VmafContext` field added**:
  `vmaf->vulkan.pool` next to `vmaf->vulkan.state`. The
  `vmaf_close()` teardown closes the pool before clearing the
  state pointer (matches SYCL).
- **On upstream sync**: zero interaction. Vulkan backend is
  fork-only; upstream Netflix has no Vulkan integration.
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false \
      -Denable_vulkan=enabled
  ninja -C build
  meson test -C build test_vulkan_pic_preallocation
  # All 6 cases must pass under ASan/UBSan:
  #   test_method_none_is_a_no_op
  #   test_method_host_allocates_round_robins
  #   test_method_device_allocates_round_robins
  #   test_fetch_without_preallocate_falls_back
  #   test_unknown_method_rejected
  #   test_null_args_rejected
### 0099 — feature_mobilesal.c + transnet_v2.c migrated to tiny_extractor_template.h

- **PR**: refactor/migrate-ai-to-template.
- **What rebases need to know**: `feature_mobilesal.c` and
  `transnet_v2.c` previously open-coded the model-path resolution
  (`getenv` + log block), the YUV→RGB kernel (mobilesal only), the
  `vmaf_dnn_session_open` + log boilerplate, and the
  `VmafOption[].model_path` row. They now use the helpers from
  `dnn/tiny_extractor_template.h` (PR #251) — the same template
  `feature_lpips.c` and `fastdvdnet_pre.c` already consume. Net
  −98 LOC of identical boilerplate.
- **Behavior preserved**: bit-exact YUV→RGB conversion (mobilesal
  used the literal copy of `feature_lpips.c`'s body that the
  template hoisted), identical error-log strings, identical
  option-table flag/type/offset shape. The migrated
  `mobilesal_options` macro expands to the same struct literal the
  hand-rolled version produced.
- **On upstream sync**: zero interaction. Both files are
  fork-introduced; upstream Netflix has neither extractor.
### 0100 — `cuda/ring_buffer.{c,h}` → `gpu_picture_pool.{c,h}` (ADR-0239)

- **PR**: refactor/gpu-picture-pool-extract.
- **What rebases need to know**:
  `libvmaf/src/cuda/ring_buffer.c` and `ring_buffer.h` are removed.
  The same callback-based round-robin pool lives at
  `libvmaf/src/gpu_picture_pool.{c,h}` under renamed symbols
  (`VmafRingBuffer` → `VmafGpuPicturePool`,
  `vmaf_ring_buffer_*` → `vmaf_gpu_picture_pool_*`,
  `_fetch_next_picture` → `_fetch`). All call sites in
  `libvmaf.c` migrated. `libvmaf/test/test_ring_buffer.c` renamed
  to `test_gpu_picture_pool.c` with the corresponding meson update.
- **Netflix-upstream interaction**: minimal — Netflix's
  `cuda/ring_buffer.{c,h}` last touched in commit `cb1d49c6`. An
  upstream sync that resurrects the old names should be redirected
  to the new ones; the file move is purely fork-local.
- **`Netflix#1300` mutex-destroy-order fix preserved** (ADR-0157)
  — moved verbatim to the new file; the fix remains attached to
  `vmaf_gpu_picture_pool_close`.
- **SYCL pool migration**: `vmaf_sycl_picture_pool_*` keeps its
  public-internal API but now delegates to the generic pool. The
  SYCL wrapper struct (`VmafSyclPicturePool`) just owns the
  `VmafSyclCookie` storage. `std::mutex` drops out.
- **Vulkan pool migration**: bundled into this PR after #264
  merged. `picture_vulkan_pool.c` rewrites as a thin wrapper
  around the generic pool — wrapper struct owns per-pool state
  for the alloc/free callbacks; the generic pool owns the
  round-robin slots / mutex / unwind. Same pattern as the SYCL
  migration above.
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build --suite=dnn
  meson test -C build test_lpips test_mobilesal test_transnet_v2 test_fastdvdnet_pre
  # All 11 dnn-suite + 4 extractor smoke tests must pass.
  meson test -C build  # 47/47 pass under ASan/UBSan

  # CUDA build (CI-only; pre-existing local nvcc include-path quirk):
  meson setup build-cuda libvmaf -Denable_cuda=true
  ninja -C build-cuda
  meson test -C build-cuda test_gpu_picture_pool

  # SYCL build:
  meson setup build-sycl libvmaf -Denable_sycl=true
  ninja -C build-sycl
  meson test -C build-sycl
  ```

### 0104 — `psnr_vulkan.c` migrated to `vulkan/kernel_template.h`

- **PR**: refactor/migrate-psnr-vulkan-to-template.
- **What rebases need to know**: `vulkan/kernel_template.h` (410 LOC,
  ADR-0246, PR #251) shipped with zero consumers. Its docstring
  designated `psnr_vulkan.c` as the reference implementation. This
  PR lands the migration as the **first consumer** of the Vulkan
  template — paired with PR #269 (the first CUDA template
  consumer). The 5 long-lived pipeline objects (descriptor-set
  layout, pipeline layout, shader module, compute pipeline,
  descriptor pool) collapse from individual struct fields to one
  `VmafVulkanKernelPipeline pl` bundle. `create_pipeline()`
  (~104 LOC) collapses to a single
  `vmaf_vulkan_kernel_pipeline_create()` call (~30 LOC) — the
  template owns the descriptor-set layout creation, pipeline
  layout, shader module, compute pipeline, and descriptor-pool
  sizing. `close_fex()`'s `vkDeviceWaitIdle` + 5×`vkDestroy*` sweep
  collapses to one `vmaf_vulkan_kernel_pipeline_destroy()` call.
- **Net LOC delta**: −55 LOC on `psnr_vulkan.c` directly. Unlike
  the CUDA template (where helper-call boilerplate roughly
  matches the inline savings), the Vulkan template's pipeline
  creation is dramatic enough that even the first consumer wins.
- **Bit-exactness gates**: spec-constants, push-constant struct,
  shader bytecode, dispatch grid math, and host-side reduction
  are byte-identical to the prior implementation. The template
  only owns descriptor-set layout / pipeline layout / shader
  module / compute pipeline creation / descriptor pool sizing —
  none of which affects the kernel's mathematical behaviour.
  Cross-backend parity gate (places=4) re-runs unchanged.
- **On upstream sync**: zero interaction. `psnr_vulkan.c` is
  fork-introduced (T7-23 / ADR-0182 / ADR-0216).
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false \
      -Denable_vulkan=enabled
  ninja -C build
  meson test -C build  # 50/50 pass on lavapipe
  # Cross-backend parity gate (places=4):
  python scripts/ci/cross_backend_parity_gate.py --feature psnr_y --places 4
  ```

### 0105 — `moment_vulkan.c` + `ciede_vulkan.c` migrated to `vulkan/kernel_template.h`

- **PR**: refactor/migrate-motion-vulkan-to-template (note: the
  branch name reflects the original intent; motion's two-pipeline
  shape didn't fit the template's single-pipeline contract, so this
  PR migrates moment + ciede instead).
- **What rebases need to know**: second + third consumers of
  `vulkan/kernel_template.h` (after PR #270 = psnr_vulkan, the
  first consumer). Both files follow the identical migration pattern:
  - Replace 5 individual pipeline-object fields (`dsl`,
    `pipeline_layout`, `shader`, `pipeline`, `desc_pool`) with one
    `VmafVulkanKernelPipeline pl` bundle.
  - Replace ~100 LOC of `create_pipeline()` body (descriptor-set
    layout + pipeline layout + shader module + compute pipeline +
    descriptor pool boilerplate) with a single
    `vmaf_vulkan_kernel_pipeline_create()` call.
  - Replace `close_fex()`'s `vkDeviceWaitIdle` + 5×`vkDestroy*`
    sweep with one `vmaf_vulkan_kernel_pipeline_destroy()` call.
- **Per-file LOC deltas**:
  - `moment_vulkan.c`: −60 LOC (450 → 390).
  - `ciede_vulkan.c`: −59 LOC (536 → 477).
  - Net: **−119 LOC**.
- **Bit-exactness preserved**: spec-constants (width/height/bpc/
  subgroup_size identical across both), push-constant structs
  (`MomentPushConsts`, `CiedePushConsts`), shader bytecodes
  (`moment_spv`, `ciede_spv`), dispatch grid math, and host-side
  reductions are byte-identical to the prior implementation.
  Cross-backend parity gates (places=4 for moment integer reduce;
  places=2 for ciede transcendentals per ADR-0187) re-run unchanged.
- **`motion_vulkan.c` deferred**: motion uses two pipelines (first
  frame vs subsequent) sharing one DSL + layout + shader + pool.
  The template's current shape produces one pipeline per descriptor;
  splitting motion across two `VmafVulkanKernelPipeline` instances
  would duplicate the shared objects. Tracked as a follow-up
  template extension (multi-pipeline support).
- **On upstream sync**: zero interaction. Both files are
  fork-introduced (T7-23 / ADR-0182 / ADR-0187).
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false \
      -Denable_vulkan=enabled
  ninja -C build
  meson test -C build  # 50/50 pass on lavapipe (under ASan/UBSan)
  python scripts/ci/cross_backend_parity_gate.py --feature float_moment_ref1st --places 4
  python scripts/ci/cross_backend_parity_gate.py --feature ciede2000 --places 2
### 0101 — GPU backend pattern doc (ADR-0240)

- **PR**: docs/gpu-backend-template.
- **What rebases need to know**: doc-only PR. Adds
  `docs/development/gpu-backend-template.md` (recipe new GPU
  backends follow) and `libvmaf/include/libvmaf/AGENTS.md`
  (public-headers-tree invariant note). No source code, no meson
  changes, no ABI impact.
- **On upstream sync**: zero interaction. Both files are
  fork-introduced.
- **Re-test on rebase**:

  ```bash
  # Doc-only — verify links resolve:
  test -f docs/development/gpu-backend-template.md
  test -f libvmaf/include/libvmaf/AGENTS.md
  grep -c 'gpu-backend-template' libvmaf/include/libvmaf/AGENTS.md
### 0102 — Tiny-AI test registration macro (`tiny_ai_test_template.h`)

- **PR**: refactor/test-registration-macro.
- **What rebases need to know**: new
  `libvmaf/test/tiny_ai_test_template.h` emits the four standard
  registration tests (`<name>_is_registered`,
  `<name>_provides_primary_feature`,
  `<name>_options_table_well_formed`,
  `<name>_init_rejects_missing_model`) via the
  `VMAF_TINY_AI_DEFINE_REGISTRATION_TESTS(ext, feat, env, prefix)`
  macro. The four per-extractor test files
  (`test_lpips.c`, `test_mobilesal.c`, `test_transnet_v2.c`,
  `test_fastdvdnet_pre.c`) shrank from ~140 LOC each to ~20-50 LOC.
  Net −286 LOC. Behavior bit-exact preserved (same assertions,
  same env-var save/restore dance, same setenv shim for MSVCRT).
  TransNet V2 keeps two extractor-specific extra tests
  (binary-flag round-trip + provided_features list-termination)
  that the macro doesn't cover.
- **On upstream sync**: zero interaction. The four test files are
  fork-introduced (per ADR-0042 / ADR-0168 / ADR-0220 / ADR-0223 /
  ADR-0215).
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build test_lpips test_mobilesal test_transnet_v2 test_fastdvdnet_pre
  # 4/4 binaries pass; 18 individual tests total (4x4 standard + 2
  # TransNet V2 extras).
### 0103 — `integer_psnr_cuda.c` migrated to `cuda/kernel_template.h`

- **PR**: refactor/migrate-psnr-cuda-to-template.
- **What rebases need to know**: `cuda/kernel_template.h` shipped
  with no consumers in PR #251 (ADR-0246). This PR migrates the
  first consumer (`integer_psnr_cuda.c`) — the file the template's
  own docstring explicitly designated as the reference. The
  `CUstream + CUevent + CUevent` triple and the
  `(VmafCudaBuffer device, void *host_pinned, size_t bytes)`
  readback pair are now dispensed by the template helpers
  (`vmaf_cuda_kernel_lifecycle_init/_close`,
  `vmaf_cuda_kernel_readback_alloc/_free`,
  `vmaf_cuda_kernel_submit_pre_launch`,
  `vmaf_cuda_kernel_collect_wait`) instead of being open-coded.
  `PsnrStateCuda` shrinks: replaces three fields
  (`event` + `finished` + `str`) with one `VmafCudaKernelLifecycle`
  + replaces (`sse` + `sse_host`) with one `VmafCudaKernelReadback`.
- **Net LOC delta**: +8 LOC on `integer_psnr_cuda.c` alone — the
  helpers add per-call boilerplate. The dedup win materialises as
  more CUDA feature kernels (motion / moment / ssim / vif / adm)
  migrate one-at-a-time in follow-up PRs. Each subsequent migration
  saves ~15 LOC.
- **Bit-exactness gates**: kernel launch + reduction logic
  unchanged. The migration only touches state-management
  boilerplate around the kernel; the SSE accumulator math, the
  per-bpc kernel function lookup, the host-side `log10` score
  formula, and the dispatch grid-dim calculation are byte-identical
  to the prior implementation. Netflix golden gate + CPU/CUDA
  cross-backend parity gate (places=4) re-run unchanged.
- **On upstream sync**: zero interaction. `integer_psnr_cuda.c` is
  fork-introduced (T7-23 / ADR-0182).
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_cuda=true
  ninja -C build
  meson test -C build  # CUDA test suite must pass
  # Cross-backend parity gate:
  python scripts/ci/cross_backend_parity_gate.py --feature psnr_y --places 4
  ```

### 0125 — Vulkan submit-side template + fence pool + descriptor pre-alloc bundle (ADR-0256)

- **Touches**:
  - `libvmaf/src/vulkan/kernel_template.h` — fork-local. Output landing in `runs/phase_a/` is gitignored — rerun the script to reproduce.
    `VmafVulkanKernelSubmitPool` struct + `_create` / `_destroy` /
    `_acquire` helpers + `vmaf_vulkan_kernel_descriptor_sets_alloc`
    helper. Upstream has no Vulkan backend — no merge surface.
  - `libvmaf/src/feature/vulkan/{psnr_hvs,vif,float_vif,float_adm}_vulkan.c`
    — fork-local kernel TUs, also no upstream peer.
- **Invariant**: the four migrated kernels keep all per-frame
  `VkFence` + `VkCommandBuffer` + `VkDescriptorSet` resources
  alive across frames in the pool. Pre-bound descriptor sets rely
  on the kernel's `VmafVulkanBuffer *` handles being init-time
  stable (allocated in `init()`, freed only in `close_fex`).
  `vmaf_vulkan_kernel_pipeline_destroy` destroys the descriptor
  pool — pre-allocated sets are released implicitly via the pool;
  callers must NOT call `vkFreeDescriptorSets` on them.
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_vulkan=enabled
  ninja -C build
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
      meson test -C build test_vulkan_smoke \
                          test_vulkan_async_pending_fence \
                          test_vulkan_pic_preallocation
  python scripts/ci/cross_backend_vif_diff.py \
      --vmaf-binary build/tools/vmaf \
      --reference testdata/ref_576x324_48f.yuv \
      --distorted testdata/dis_576x324_48f.yuv \
      --width 576 --height 324 --pixel-format 420 --bitdepth 8 \
      --feature vif --backend vulkan --places 4
  python scripts/ci/cross_backend_vif_diff.py \
      --vmaf-binary build/tools/vmaf \
      --reference testdata/ref_576x324_48f.yuv \
      --distorted testdata/dis_576x324_48f.yuv \
      --width 576 --height 324 --pixel-format 420 --bitdepth 8 \
      --feature adm --backend vulkan --places 4
  ```

### 0107 — `psnr_hvs_cuda` async upload + persistent pinned staging (T-GPU-OPT-2/3)

- **Touches**:
  - `libvmaf/src/feature/cuda/integer_psnr_hvs_cuda.c` — only
    consumer; fork-local from inception (T7-23 / ADR-0188 /
    ADR-0191). State adds `upload_str` (dedicated H2D stream),
    `upload_done` (cross-stream completion event), and per-plane
    persistent pinned `h_uint_ref[3]` / `h_uint_dist[3]` staging
    buffers allocated once in `init_fex_cuda`. The per-call
    helper `upload_plane_cuda` is split into `issue_d2h_plane`
    (pic-stream D2H), `convert_plane` (CPU normalise), and
    `issue_h2d_plane` (upload-stream H2D). `submit_fex_cuda`
    runs the three phases explicitly and records `upload_done`
    after the last H2D, then `cuStreamWaitEvent`s on `lc.str`
    before kernel launches.
  - `libvmaf/src/cuda/AGENTS.md` — adds a rebase-sensitive
    invariant entry under §Rebase-sensitive invariants
    documenting the three-phase flow + persistent staging
    contract.
- **Invariant**: the pinned `h_uint_*` and `h_ref` / `h_dist`
  buffers are never freed and re-allocated mid-stream; the H2Ds
  must run on `upload_str` (not on `lc.str`) so the
  `cuStreamWaitEvent` cross-stream link is meaningful; the
  `upload_done` event is recorded *after the last H2D for the
  current frame* and waited on once before *the first* kernel
  launch of that frame. CUDA graph capture (future T-GPU-OPT-N)
  depends on the no-per-frame-alloc invariant; collapsing the
  three-phase split or re-introducing per-frame
  `vmaf_cuda_buffer_host_alloc` calls breaks that follow-up.
  Bit-exactness gate is `places=3` for `psnr_hvs_y / cb / cr`
  and the combined `psnr_hvs` (matches the existing matrix; not
  `places=4`).
- **On upstream sync**: zero interaction. `integer_psnr_hvs_cuda.c`
  is fork-introduced (T7-23 / ADR-0188 / ADR-0191).
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_cuda=true -Denable_sycl=false
  ninja -C build
  meson test -C build
  python3 scripts/ci/cross_backend_vif_diff.py \
    --vmaf-binary libvmaf/build/tools/vmaf \
    --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
    --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
    --width 576 --height 324 --pixel-format 420 --bitdepth 8 \
    --feature psnr_hvs --backend cuda --places 3
  ```

### 0227 — `output.c` writer-format unit tests (R3 of coverage-gap-2026-05-02)

- **Touches**:
  - `libvmaf/test/test_output.c` (new) — exercises the four writers
    in `libvmaf/src/output.c` (XML / JSON / CSV / SUB) end-to-end via
    `tmpfile()`-backed sinks and a synthetic `VmafFeatureCollector`.
    Pure test-only; no production code change.
  - `libvmaf/test/meson.build` — registers `test_output` next to
    `test_feature_collector` (mirrors that test's wiring:
    `link_with: libvmaf` + libsvm objects + log/predict/metadata
    helpers).
- **Invariant**: the test pulls `libvmaf.c` and `output.c` in via
  `#include "*.c"` (mirroring the precedent in
  `test_feature_collector.c`) so the per-translation-unit `.gcno`
  lands in the test build dir and gcovr aggregates output.c's
  coverage. The mu-test framework macro (`mu_assert`) deliberately
  early-returns from each `static char *test_*()` body — that's why
  every test body trips `clang-analyzer-unix.Malloc` "potential leak"
  notes (cleanup runs only on the success-tail path). This pattern
  is shared across every `libvmaf/test/test_*.c` file and is *load-
  bearing* (per ADR-0141 NOLINT carve-out): replacing it with goto-
  cleanup would obscure the per-assertion failure message.
- **On upstream sync**: zero interaction. `output.c` is upstream-
  mirrored, but this PR doesn't touch it. The test only depends on
  the four public function signatures (`vmaf_write_output_{xml,
  json,csv,sub}`); if Netflix renames or reorders those, the test
  fails to compile and the rebase author updates it then.
- **Re-test on rebase**:

  ```bash
  cd libvmaf && meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build && ./build/test/test_output
  ```

### 0126 — OSSF Scorecard policy (ADR-0263)

- **Touches**: `.github/workflows/scorecard.yml` (line 45 — the
  `github/codeql-action/upload-sarif@<sha>` pin). The rest of the
  policy is doc-only (`docs/adr/0263-*.md`,
  `docs/research/0053-*.md`, `changelog.d/security/`). Upstream
  Netflix/vmaf does not ship a Scorecard workflow, so the path
  itself is fork-introduced and won't conflict.
- **Invariant**: the `upload-sarif` SHA must point to a commit that
  currently exists in `github/codeql-action`'s git tree. A SHA that
  was once `v4` head but no longer exists in the action repository
  triggers Scorecard's "imposter commit" defence and breaks the
  workflow with a 400 error against `api.scorecard.dev`. Verify on
  every Dependabot bump by spot-checking
  `gh api /repos/github/codeql-action/commits/<sha>` returns 200.
- **On upstream sync**: zero interaction.
- **Re-test on rebase**:

  ```bash
  # Confirm the pin still resolves to a real commit:
  pin=$(grep -oE 'codeql-action/upload-sarif@[a-f0-9]{40}' \
        .github/workflows/scorecard.yml | head -1 | cut -d@ -f2)
  gh api "/repos/github/codeql-action/commits/$pin" --jq '.sha'
  # Then watch the next master push for a green Scorecard run:
  gh run list --workflow scorecard --repo lusoris/vmaf --limit 1
### 0228 — U-2-Net `u2netp` saliency replacement deferred (ADR-0265)

- **Touches**: docs-only.
  - `docs/adr/0265-u2netp-saliency-replacement-blocked.md` — new
    ADR continuing the deferral chain started by ADR-0257.
  - `docs/research/0055-u2netp-saliency-replacement-survey.md` —
    new research digest (upstream survey + license + distribution
    + op-allowlist audit + alternatives walk).
  - `docs/ai/models/mobilesal.md` — pointer block updated to
    reference both ADR-0257 (first blocker) and ADR-0265 (second
    blocker).
  - `model/tiny/registry.json` — `mobilesal_placeholder_v0` `notes`
    field updated to reference ADR-0265 alongside ADR-0257 (no
    schema / sha256 / file changes).
  - `model/tiny/mobilesal.json` — sidecar `notes` field updated in
    lockstep.
  - `scripts/gen_mobilesal_placeholder_onnx.py` — generator notes
    string updated so re-running is idempotent against the new
    sidecar / registry text.
  - `CHANGELOG.md` — Changed entry via
    `changelog.d/changed/T6-2a-followup-u2netp-replacement-deferred.md`.
  - `docs/adr/README.md` — index row via
    `docs/adr/_index_fragments/0265-u2netp-saliency-replacement-blocked.md`.
- **Invariant**: zero C-side surface change. `feature_mobilesal.c`
  tensor-name contract (input `input` → output `saliency_map`,
  NCHW float32 `[1, 3, H, W]` → `[1, 1, H, W]`) is unchanged; the
  on-disk `model/tiny/mobilesal.onnx` (sha256
  `f1226310…`) is unchanged; `mobilesal_placeholder_v0`'s
  `smoke: true` flag is unchanged. Any future drop-in (U-2-Net via
  `T6-2a-mirror-u2netp-via-release` + `T6-2a-widen-allowlist-resize`,
  distilled student, or BASNet / PoolNet survey result) replaces the
  `.onnx` and bumps the registry sha256 without touching the C side.
- **On upstream sync**: zero interaction. `feature_mobilesal.c`,
  the registry, the ADR, and the research digest are all fork-local
  (T6-2a; ADR-0218 / ADR-0257 / ADR-0265; not present in Netflix
  upstream).
- **Re-test on rebase**:

  ```bash
  meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build test_mobilesal
  python3 ai/scripts/validate_model_registry.py
  bash scripts/docs/concat-adr-index.sh --check
  bash scripts/release/concat-changelog-fragments.sh --check
  ```


### 0108 — `ssim_accumulate_avx512` per-lane double reduction vectorised

- **ADR**: [ADR-0139](adr/0139-ssim-simd-bitexact-double.md)
  (existing; no new ADR — the per-lane reduction order is unchanged).
- **Touches**:
  - [`libvmaf/src/feature/x86/ssim_avx512.c`](../libvmaf/src/feature/x86/ssim_avx512.c)
    — the `ssim_accumulate_block_avx512` body. The per-lane scalar
    `ssim_accumulate_lane` calls (16 of them) are replaced by two
    8-wide `__m512d` passes that compute `lv`, `cv`, `sv`, and
    `lv*cv*sv` lane-wise in vector double. Aligned `double[16]`
    spill buffers replace the previous `_Alignas(64) float[16]×6`
    spill, and the scalar accumulation loop now does 4×16
    `vaddsd` instead of 16 invocations of the per-lane helper.
  - `CHANGELOG.md` — Changed entry.
  - This file — this entry.
- **Invariant** (load-bearing for ADR-0139 bit-exactness):
  1. **Per-lane double computation order is byte-identical**:
     `((2.0 * rm) * cm + C1) / l_den`, then
     `(2.0 * srsc + C2) / c_den`, then `(lv * cv) * sv`. No FMA
     contraction (separate `_mm512_mul_pd` + `_mm512_add_pd` —
     `_mm512_fmadd_pd` is forbidden because it changes the
     rounding count and would diverge from scalar's two-step
     `mul`+`add`).
  2. **Float→double widening uses `_mm512_cvtps_pd`** which is
     IEEE-754-exact for finite floats (52-bit mantissa fits 23-bit
     float losslessly).
  3. **Lane-by-lane left-to-right reduction order preserved**:
     `local_ssim += t_ssim[k]` for `k = 0..15`. Tree reductions
     (pairwise add, dual-accumulator unroll) are forbidden — they
     break running-sum associativity against scalar.
  4. **AVX2 / NEON twins kept on the per-lane scalar path**.
     Verified bit-identical against the new AVX-512 at
     `--precision max` on the Netflix `src01_hrc00/01_576x324`
     and the `checkerboard_1920_1080_10_3_*_0` pairs. The
     bit-exactness contract (ADR-0139) is per-lane, not per-ISA
     algorithm — so AVX2 / NEON stay scalar-per-lane until a
     dedicated PR vectorises them with the same care.
- **Rebase impact**: zero conflict with Netflix upstream — the
  whole SSIM SIMD surface is fork-local (no upstream SSIM SIMD
  exists). Conflicts only arise if upstream changes
  `ssim_accumulate_default_scalar` in `iqa/ssim_tools.c`; in that
  case both the AVX2 / NEON per-lane helper **and** the AVX-512
  vector-double block need a coordinated update preserving the
  three invariants above.
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  meson test -C build
  # Bit-exact at --precision max, scalar vs AVX2 vs AVX-512:
  for MASK in 0 16 255; do
    libvmaf/build/tools/vmaf -r python/test/resource/yuv/src01_hrc00_576x324.yuv \
      -d python/test/resource/yuv/src01_hrc01_576x324.yuv \
      -w 576 -h 324 -p 420 -b 8 \
      --feature float_ms_ssim --feature float_ssim \
      --xml -o /tmp/m${MASK}.xml --precision max --cpumask $MASK
  done
  diff <(grep -v 'fyi fps' /tmp/m0.xml) <(grep -v 'fyi fps' /tmp/m16.xml)   # empty
  diff <(grep -v 'fyi fps' /tmp/m0.xml) <(grep -v 'fyi fps' /tmp/m255.xml)  # empty
  ```
- **Why this matters on rebase**: an upstream commit that touches
  `libvmaf/src/feature/ssimulacra2.c` could prompt a "let's also
  port the GPU XYB while we're here" follow-up. The ledger entry
  is the standing answer: don't, the measurement was redone on
  NVIDIA in May 2026 and the result still failed `places=4` by
  five decades. See
  [Research-0047](research/0051-ssimulacra2-gpu-xyb-precision.md).

### 0126 — FastDVDnet real upstream weights drop (ADR-0253)

- **What changed**: replaces `model/tiny/fastdvdnet_pre.onnx` with
  the wrapped real upstream FastDVDnet checkpoint (sha256
  `eb9444cf6f07eefdc7f4f68d09131074dbd1dcee6f88a331ba684dd2fb5937d4`,
  ~9.5 MiB), refreshes the sidecar `model/tiny/fastdvdnet_pre.json`,
  flips the registry row's `smoke: true → false` and adds
  `license: "MIT"` + the upstream commit pin `c8fdf61`. New exporter
  `ai/scripts/export_fastdvdnet_pre.py` (the older
  `_placeholder.py` exporter is retained for reference). New ADR
  `docs/adr/0255-fastdvdnet-pre-real-weights.md`; user-facing doc
  `docs/ai/models/fastdvdnet_pre.md` rewritten with provenance,
  license attribution, and reproduce-the-export instructions.
- **Upstream source**: fork-local. Netflix/vmaf does not ship a
  FastDVDnet temporal pre-filter; the C extractor and ONNX surface
  are entirely fork-introduced (ADR-0215). The wrapped weights are
  attribution-only (upstream `m-tassano/fastdvdnet` MIT).
- **On upstream sync**: zero interaction. Every file touched
  (`ai/scripts/export_fastdvdnet_pre*.py`, `model/tiny/fastdvdnet_pre.*`,
  `docs/ai/models/fastdvdnet_pre.md`, `docs/adr/0253-*.md`,
  CHANGELOG fragment, ADR index fragment) lives in fork-introduced
  trees.
- **Re-test on rebase**:

  ```bash
  # Re-derive the ONNX from the pinned upstream checkpoint.
  mkdir -p /tmp/fastdvdnet_upstream && cd /tmp/fastdvdnet_upstream
  curl -L -O https://raw.githubusercontent.com/m-tassano/fastdvdnet/c8fdf61/model.pth
  curl -L -O https://raw.githubusercontent.com/m-tassano/fastdvdnet/c8fdf61/models.py
  cd /path/to/vmaf
  python3 ai/scripts/export_fastdvdnet_pre.py \
      --upstream-dir /tmp/fastdvdnet_upstream
  python3 ai/scripts/validate_model_registry.py
  meson test -C build --suite=fast --print-errorlogs test_fastdvdnet_pre
  ```

### 0127 — ONNX op-allowlist gains `Resize` (ADR-0258)

- **Touches**:
  - `libvmaf/src/dnn/op_allowlist.c` — fork-local file (no upstream
    counterpart). One new entry `"Resize"` under the
    `/* convolutional */` block.
  - `libvmaf/test/dnn/test_op_allowlist.c`,
    `libvmaf/test/dnn/test_onnx_scan.c` — fork-local DNN tests.
  - `ai/tests/test_op_allowlist.py` — fork-local Python parity test.
- **Invariant**: the C allowlist is the single source of truth; the
  Python regex parser in `ai/src/vmaf_train/op_allowlist.py` walks
  the same `op_allowlist.c` file. Any future entry only needs the
  C edit — Python symmetry is automatic.
- **Upstream source**: fork-local. Netflix/vmaf has no ONNX op-
  allowlist surface; the entire `libvmaf/src/dnn/` tree is fork-
  introduced.
- **On upstream sync**: zero interaction. Every file touched lives
  in fork-introduced trees.
- **Re-test on rebase**:

  ```bash
  meson test -C build test_op_allowlist test_onnx_scan
  PYTHONPATH=ai/src python -m pytest ai/tests/test_op_allowlist.py
  ```
### 0231 — `vif.comp` + `ciede.comp` `precise` decorations (ADR-0269 / Step A of Vulkan 1.4 bump)

- **Touches**:
  [`libvmaf/src/feature/vulkan/shaders/vif.comp`](../libvmaf/src/feature/vulkan/shaders/vif.comp)
  (3 local-variable type qualifiers: `g`, `sv_sq`, `gg_sigma_f` →
  `precise float`),
  [`libvmaf/src/feature/vulkan/shaders/ciede.comp`](../libvmaf/src/feature/vulkan/shaders/ciede.comp)
  (`yuv_to_rgb` outputs, `rgb_to_xyz` matmul accumulators, `ciede2000`
  chroma magnitudes + half-axes + s_l/c/h + lightness/chroma/hue +
  final ΔE).
- **Invariant**: Both shaders are fork-local (Vulkan backend is
  fork-added; upstream Netflix/vmaf has no Vulkan compute kernels).
  The `precise` keyword is GLSL 4.50 standard syntax; glslc 2026.1
  lowers it to per-result `OpDecorate NoContraction`. The decorations
  are *load-bearing for the cross-backend gate* on NVIDIA driver
  595.71+ — removing them would re-introduce the 42/48 ciede
  regression at API 1.3 documented in research-0054.
- **On upstream sync**: zero interaction. Both shader files are
  entirely fork-introduced; upstream has no Vulkan compute path.
- **Re-test on rebase**:

  ```bash
  # Re-confirm the cross-backend gate on a Vulkan-capable host.
  meson setup libvmaf/build -Denable_vulkan=enabled
  ninja -C libvmaf/build
  python3 scripts/ci/cross_backend_vif_diff.py \
      --vmaf-binary libvmaf/build/tools/vmaf \
      --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
      --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
      --width 576 --height 324 --pixel-format 420 --bitdepth 8 \
      --feature vif --backend vulkan --places 4
  python3 scripts/ci/cross_backend_vif_diff.py \
      --vmaf-binary libvmaf/build/tools/vmaf \
      --reference python/test/resource/yuv/src01_hrc00_576x324.yuv \
      --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv \
      --width 576 --height 324 --pixel-format 420 --bitdepth 8 \
      --feature ciede --backend vulkan --places 4
  # Confirm SPIR-V still emits NoContraction post-rebase.
  glslc --target-env=vulkan1.3 -O \
      libvmaf/src/feature/vulkan/shaders/vif.comp -o /tmp/vif.spv
  spirv-dis /tmp/vif.spv | grep -c NoContraction   # expect ≥ 60
  ```

  Expected on NVIDIA 595.71+: vif 0/48 OK, ciede 5/48 FAIL (max abs
  8.9e-05 — pre-existing fork debt at API 1.3, see ADR-0269). On
  RADV / lavapipe: bit-exact (`precise` is a no-op there).
### 0229 — `fr_regressor_v2` codec-aware scaffold (ADR-0272)

- **ADR**: [ADR-0272](adr/0272-fr-regressor-v2-codec-aware-scaffold.md)
- **Touches**:
  - `ai/scripts/train_fr_regressor_v2.py` (new) — Phase A JSONL
    consumer; trains the codec-aware FRRegressor.
  - `model/tiny/fr_regressor_v2.onnx` (new, smoke) — placeholder
    ONNX from `--smoke` mode; re-baked on production training.
  - `model/tiny/fr_regressor_v2.json` (new) — sidecar.
  - `model/tiny/registry.json` — new entry with `smoke: true`.
  - `docs/adr/0272-fr-regressor-v2-codec-aware-scaffold.md` (new).
  - `docs/adr/README.md` — index row.
  - `docs/research/0058-fr-regressor-v2-feasibility.md` (new).
  - `docs/ai/models/fr_regressor_v2.md` (new) — model card.
  - `ai/AGENTS.md` — invariant note (codec block layout +
    ENCODER_VOCAB ordering).
  - `CHANGELOG.md` — Added entry.
- **Invariant**: the 8-D codec block layout is
  `[encoder_onehot(6), preset_norm, crf_norm]` with
  `ENCODER_VOCAB = (libx264, libx265, libsvtav1, libvvenc, libvpx-vp9, unknown)`
  in load-bearing order. CRF normaliser is `/63` (union upper bound).
  Preset normaliser is `/9`. Bumping the vocabulary requires a
  re-train; existing checkpoints pin the order they were trained
  against via `encoder_vocab_version` in the sidecar. The two-input
  ONNX (`features`, `codec`) follows the LPIPS-Sq precedent
  (ADR-0040 / ADR-0041).
- **Rebase impact**: entirely fork-local; pure additive; no
  upstream-mirror file is touched. Phase A schema (consumed by this
  trainer) is itself fork-local (`tools/vmaf-tune/`). No conflict
  expected on `/sync-upstream`.
- **Re-test on rebase**:

  ```bash
  python ai/scripts/train_fr_regressor_v2.py --smoke
  python ai/scripts/validate_model_registry.py
  ```

### 0229 — libFuzzer scaffold for the YUV4MPEG2 parser (ADR-0270)

- **ADR**: [ADR-0270](adr/0270-fuzzing-scaffold.md)
- **Touches**:
  - `libvmaf/test/fuzz/fuzz_y4m_input.c` (new)
  - `libvmaf/test/fuzz/meson.build` (new)
  - `libvmaf/test/fuzz/README.md` (new)
  - `libvmaf/test/fuzz/y4m_input_corpus/*` (new — six seeds)
  - `libvmaf/test/fuzz/y4m_input_known_crashes/*` (new — one
    411-chroma OOB reproducer; **excluded** from CI corpus)
  - `libvmaf/test/meson.build` — `subdir('fuzz')` line.
  - `libvmaf/meson_options.txt` — new `option('fuzz', ...)`.
  - `.github/workflows/fuzz.yml` (new — nightly 5-minute job).
  - `docs/development/fuzzing.md` (new — operator runbook).
  - `docs/adr/0270-fuzzing-scaffold.md` (new)
  - `docs/research/0059-libfuzzer-scaffold-y4m.md` (new)
  - `docs/state.md` — new Open-bug row for the 411-chroma
    OOB write.
  - `CHANGELOG.md` — Added entry.
- **Invariant**: the fuzz scaffold is *opt-in* — every default
  `meson setup` invocation must continue to skip it. The harness
  links statically against `libvmaf/tools/{y4m_input,yuv_input,vidinput}.c`
  rather than `libvmaf.so` so the public C-API surface stays
  unchanged.
- **Rebase impact**: the harness re-includes `libvmaf/tools/y4m_input.c`
  as a build input. Any upstream Netflix/vmaf change that splits
  or renames the tool sources (e.g. moves the parser into
  `libvmaf/src/`) needs the corresponding `meson.build` source
  list update *and* the harness re-test below. The
  `y4m_input_known_crashes/y4m_411_w2_h4_oob_dst.y4m` reproducer
  is the regression gate for the parser fix; do not delete it
  on upstream sync — if upstream lands the same fix, port the
  reproducer back into `y4m_input_corpus/` as a permanent seed.
- **Re-test on rebase**:

  ```bash
  CC=clang CXX=clang++ \
    meson setup build-fuzz libvmaf \
      --buildtype=debug \
      -Db_sanitize=address \
      -Db_lundef=false \
      -Dfuzz=true \
      -Denable_cuda=false -Denable_sycl=false -Denable_vulkan=disabled
  ninja -C build-fuzz test/fuzz/fuzz_y4m_input
  ./build-fuzz/test/fuzz/fuzz_y4m_input \
      -max_total_time=60 \
      libvmaf/test/fuzz/y4m_input_corpus/
  # Verify the known-crash reproducer still triggers (until the fix lands):
  ./build-fuzz/test/fuzz/fuzz_y4m_input \
      libvmaf/test/fuzz/y4m_input_known_crashes/y4m_411_w2_h4_oob_dst.y4m
  ```

### 0231 — HIP seventh-consumer kernel `float_motion_hip` (ADR-0273)

- **ADR**: [ADR-0273](adr/0273-hip-seventh-consumer-float-motion.md)
- **Touches**:
  - `libvmaf/src/feature/hip/float_motion_hip.c` (new) — seventh
    consumer of `libvmaf/src/hip/kernel_template.h`. Mirrors
    `libvmaf/src/feature/cuda/float_motion_cuda.c`
    call-graph-for-call-graph; `init/submit/collect/close` invoke
    the kernel-template helpers in the same order; `flush()`
    callback for tail-frame motion2 emission; `motion_force_zero`
    short-circuit posture (`fex->extract` swap with
    `submit / collect / flush / close` nulled). Submit path
    intentionally bypasses `vmaf_hip_kernel_submit_pre_launch`
    (kernel writes per-WG SAD float partials directly, no atomic,
    no memset).
  - `libvmaf/src/feature/hip/float_motion_hip.h` (new)
  - `libvmaf/src/hip/meson.build` — new entry in `hip_sources`.
  - `libvmaf/src/feature/feature_extractor.c` — extern declaration
    plus `feature_extractor_list[]` entry under `#if HAVE_HIP`.
  - `libvmaf/test/test_hip_smoke.c` — new sub-test
    `test_float_motion_hip_extractor_registered` (also asserts the
    `VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag bit) and a row in
    `test_table[]`.
  - `docs/adr/0273-hip-seventh-consumer-float-motion.md` (new)
  - `docs/adr/README.md` — index row.
  - `docs/backends/hip/overview.md` — seventh / eighth consumer
    note.
  - `libvmaf/src/hip/AGENTS.md` — invariant note.
  - `CHANGELOG.md` — Added entry (joint with ADR-0274).
- **Invariant — three-buffer ping-pong + `motion_force_zero`
  short-circuit are load-bearing**. The state struct carries three
  `uintptr_t` buffer slots (`ref_in`, `blur[2]`) that the runtime
  PR (T7-10b) will swap for real device-buffer handles matching
  the CUDA twin's `VmafCudaBuffer *ref_in` + `VmafCudaBuffer *blur[2]`
  field shape. The `motion_force_zero` short-circuit (`fex->extract`
  swap, kernel-template helpers nulled) must stay aligned with the
  CUDA twin on every refactor — otherwise the runtime PR's
  helper-body flip diverges between the two backends. The
  `submit_pre_launch` bypass mirrors the CUDA twin; if a future PR
  adds a `submit_pre_launch` call to `float_motion_cuda.c`'s submit
  path, the HIP twin must follow in the same PR.
- **Rebase impact**: entirely fork-local. New files are
  HIP-specific. The only upstream-touching edit is
  `feature_extractor.c`, but the change sits inside an existing
  `#if HAVE_HIP` block (ADR-0241); upstream has no `HAVE_HIP` so
  no conflict is expected.
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_hip=true \
    -Denable_cuda=false -Denable_sycl=false -Denable_vulkan=disabled
  ninja -C build
  meson test -C build test_hip_smoke
  ```

### 0232 — HIP eighth-consumer kernel `float_ssim_hip` (ADR-0274)

- **ADR**: [ADR-0274](adr/0274-hip-eighth-consumer-float-ssim.md)
- **Touches**:
  - `libvmaf/src/feature/hip/float_ssim_hip.c` (new) — eighth
    consumer of `libvmaf/src/hip/kernel_template.h`. Mirrors
    `libvmaf/src/feature/cuda/integer_ssim_cuda.c`
    call-graph-for-call-graph (the CUDA file registers
    `vmaf_fex_float_ssim_cuda` despite its `integer_` filename).
    First multi-dispatch HIP consumer
    (`chars.n_dispatches_per_frame == 2`). Submit path
    intentionally bypasses `vmaf_hip_kernel_submit_pre_launch`
    (kernel writes per-block float partials directly). State
    struct carries five `uintptr_t` intermediate float buffer slots
    (`h_ref_mu`, `h_cmp_mu`, `h_ref_sq`, `h_cmp_sq`, `h_refcmp`)
    tracked outside the kernel-template's readback bundle.
    `validate_dims_hip` and `init_dims_hip` helpers extracted from
    `init()` to fit the `readability-function-size` budget.
  - `libvmaf/src/feature/hip/float_ssim_hip.h` (new)
  - `libvmaf/src/hip/meson.build` — new entry in `hip_sources`.
  - `libvmaf/src/feature/feature_extractor.c` — extern declaration
    plus `feature_extractor_list[]` entry under `#if HAVE_HIP`.
  - `libvmaf/test/test_hip_smoke.c` — new sub-test
    `test_float_ssim_hip_extractor_registered` (also asserts
    `chars.n_dispatches_per_frame == 2`) and a row in `test_table[]`.
  - `docs/adr/0274-hip-eighth-consumer-float-ssim.md` (new)
  - `docs/adr/README.md` — index row.
  - `docs/backends/hip/overview.md` — seventh / eighth consumer
    note (joint).
  - `libvmaf/src/hip/AGENTS.md` — invariant note.
  - `CHANGELOG.md` — Added entry (joint with ADR-0273).
- **Invariant — multi-dispatch + five-slot buffer pyramid + v1
  `scale=1` validation are load-bearing**. The state struct carries
  five `uintptr_t` intermediate float buffer slots that the runtime
  PR (T7-10b) will swap for real device-buffer handles matching
  the CUDA twin's `VmafCudaBuffer *h_*` field shape — any drift in
  the CUDA twin's slot count requires a paired update here. The
  `chars.n_dispatches_per_frame == 2` characteristic is asserted
  in the smoke test; do not silently lower it. The v1 `scale=1`
  `-EINVAL` validation surface (in `validate_dims_hip`) must stay
  aligned with the CUDA twin's `compute_scale` / `vmaf_log` chain.
  The HIP twin's `validate_dims_hip` / `init_dims_hip` extraction
  is intentional for the function-size budget; do not re-inline
  without verifying the budget still passes.
- **Rebase impact**: entirely fork-local; same posture as ADR-0273.
- **Re-test on rebase**:

  ```bash
  meson setup build libvmaf -Denable_hip=true \
    -Denable_cuda=false -Denable_sycl=false -Denable_vulkan=disabled
  ninja -C build
  meson test -C build test_hip_smoke
  ```

### 0229 — `vmaf_tiny_v3` + `vmaf_tiny_v4` dynamic-PTQ int8 sidecars (ADR-0275)

- **ADR**: [ADR-0275](adr/0275-vmaf-tiny-v3-v4-ptq.md)
- **Touches**:
  - `model/tiny/vmaf_tiny_v3.int8.onnx` (new, 4 267 B)
  - `model/tiny/vmaf_tiny_v4.int8.onnx` (new, 7 769 B)
  - `model/tiny/registry.json` — new `vmaf_tiny_v3` and `vmaf_tiny_v4`
    rows with `quant_mode`, `int8_sha256`,
    `quant_accuracy_budget_plcc` fields.
  - `model/tiny/vmaf_tiny_v3.json`,
    `model/tiny/vmaf_tiny_v4.json` — same fields mirrored into the
    per-model sidecars.
  - `docs/ai/models/vmaf_tiny_v3.md`,
    `docs/ai/models/vmaf_tiny_v4.md` — new "Quantisation" sections.
  - `docs/adr/0275-vmaf-tiny-v3-v4-ptq.md` (new) and ADR index row.
  - `CHANGELOG.md` — Added entry.
- **Invariant**: `python ai/scripts/measure_quant_drop.py --all`
  reports `[PASS]` for both `vmaf_tiny_v3` (drop ≤ 0.001 on Netflix
  features) and `vmaf_tiny_v4` (drop ≤ 0.001), inside the 0.01
  per-model budget. The runtime redirect from ADR-0174 picks the
  `.int8.onnx` sibling when an operator's registry overlay declares
  `quant_mode: dynamic`.
- **Rebase impact**: entirely fork-local — neither v3 nor v4 nor
  the dynamic-PTQ harness exists upstream. The new int8 ONNX bytes
  ship as committed binaries (mirroring `learned_filter_v1` and
  `nr_metric_v1`); they are well below the few-MB external-data
  threshold and don't require the sigstore + `.onnx.data` pattern.
- **Re-test on rebase**:

  ```bash
  python ai/scripts/validate_model_registry.py
  python ai/scripts/measure_quant_drop.py --all
### 0229 — NVIDIA-Vulkan ciede2000 places=4 fork debt root-cause (ADR-0273)

- **Touched files**: docs-only.
  - `docs/adr/0273-...precision-gap.md` (new) +
    `_index_fragments/` row + `_order.txt` append.
  - `docs/research/0055-ciede-vulkan-nvidia-f32-f64-root-cause.md`
    (new) + `docs/research/README.md` index row.
  - `docs/state.md` — Open-bugs row `T-VK-CIEDE-F32-F64`.
  - `docs/backends/vulkan/overview.md` — NVIDIA-hardware caveat.
  - `changelog.d/changed/ciede-vulkan-nvidia-f32-f64-precision-gap.md` (new).
  - `libvmaf/src/vulkan/AGENTS.md` — invariant cross-link.
- **Invariant**: the ciede.comp shader's f32 precision contract is
  load-bearing — promoting to f64 would silently change scores on
  every Vulkan device that supports `shaderFloat64` and create a
  per-device-feature-bit divergence (RTX 4090 has it; many consumer
  GPUs don't). The CPU `ciede.c::get_lab_color` doing its
  colour-space chain in `double` is upstream Netflix behaviour and
  must not be narrowed to f32 to "fix" the GPU gap (would change
  Netflix golden ground truth). The 5/48 NVIDIA places=4 mismatch
  on the highest-ΔE frames is *expected* and documented; do not
  attempt to "fix" it without re-reading ADR-0273 first.
- **Rebase impact**: zero — docs-only. The CPU and shader sources
  this ADR analyses are unchanged by this PR. If a future upstream
  rebase touches `ciede.c::get_lab_color` (the `double` chain) the
  ADR's reasoning still holds; if upstream changes the CPU
  reference's precision posture, ADR-0273 needs a `Status:
  Superseded` entry.
- **Re-test on rebase**: a manual NVIDIA-hardware run if available:

  ```bash
  cd libvmaf && meson setup build \
    -Denable_vulkan=enabled -Denable_cuda=false && ninja -C build
  cd ..
  python3 scripts/ci/cross_backend_vif_diff.py \
    --vmaf-binary $PWD/libvmaf/build/tools/vmaf \
    --reference testdata/ref_576x324_48f.yuv \
    --distorted testdata/dis_576x324_48f.yuv \
    --width 576 --height 324 \
    --feature ciede --backend vulkan --device 0 --places 4
  # Expected post-PR-346 (when merged): 5/48 mismatches at 1.78× threshold.
  # Expected pre-PR-346 (current master): 42/48 mismatches at higher ratio.
  # If the count drops below 5/48 on NVIDIA, ADR-0273 should record the
  # delta and consider closing T-VK-CIEDE-F32-F64.
### 0229 — `tools/vmaf-tune fast` Phase A.5 scaffold (ADR-0276)

- **Touches**: `tools/vmaf-tune/src/vmaftune/fast.py` (new),
  `tools/vmaf-tune/src/vmaftune/cli.py` (new `fast` subcommand
  branch), `tools/vmaf-tune/pyproject.toml` (new `[fast]` extra),
  `tools/vmaf-tune/tests/test_fast.py` (new),
  `tools/vmaf-tune/AGENTS.md` (new invariants),
  `docs/usage/vmaf-tune.md` (new "Phase A.5" section),
  `docs/adr/0276-vmaf-tune-fast-path.md` (new ADR),
  `docs/research/0060-vmaf-tune-fast-path.md` (new digest).
- **Invariant**: the `fast` subcommand is opt-in and never
  automatically replaces the Phase A grid path. The slow grid is
  the ground-truth corpus generator (ADR-0237 contract); fast-path
  is for the recommendation use case only. Optuna is a lazy-imported
  optional dep gated behind the `[fast]` extra — importing it at
  module scope outside `fast.py` (or its tests) breaks the
  zero-dep core install.
- **Rebase impact**: entirely fork-local; the tool sits under
  `tools/vmaf-tune/` which is fork-added, and no upstream files are
  touched. Upstream Netflix/vmaf has no analogous surface.
- **Re-test on rebase**:

  ```bash
  pip install -e 'tools/vmaf-tune[fast]'
  pytest tools/vmaf-tune/tests/test_fast.py -v
  vmaf-tune fast --smoke --target-vmaf 92
  ```


### 0229 — `vmaf-tune recommend` subcommand (ADR-0237 Phase B-lite)

- **Touches**:
  - `tools/vmaf-tune/src/vmaftune/recommend.py` (new). Wholly
    fork-local — no upstream Netflix/vmaf path overlap.
  - `tools/vmaf-tune/src/vmaftune/cli.py` — adds `recommend`
    subparser; `corpus` subcommand untouched.
  - `tools/vmaf-tune/tests/test_recommend.py` (new). 13-case smoke
    suite, mocks all binaries; runs in <100 ms.
  - `docs/usage/vmaf-tune.md` — adds `## recommend` section.
- **Invariant**: `recommend` consumes the existing
  `CORPUS_ROW_KEYS` schema unchanged — `vmaf_score`,
  `bitrate_kbps`, `crf`, `preset`, `encoder`, `exit_status`. No
  schema bump. If a future PR bumps `SCHEMA_VERSION`, both the
  `corpus` writer and the `recommend` reader must be updated in
  lockstep; tests assert this via `test_corpus_row_keys_match_init_contract`.
- **Rebase impact**: zero — `tools/vmaf-tune/` is wholly
  fork-local; no upstream surface touches it.
- **Re-test on rebase**:

  ```bash
  pytest tools/vmaf-tune/tests/
  ```

### 0228 — `integer_ms_ssim_cuda.c` joins drain_batch (T-GPU-OPT-2 / ADR-0271)

- **Touches**: `libvmaf/src/feature/cuda/integer_ms_ssim_cuda.c`. No
  upstream Netflix/vmaf changes expected here — the file is
  fork-added (CUDA twin of the upstream-port `ms_ssim_score.cu`) and
  the surface this PR redrew (per-scale `l_partials[i]` /
  `c_partials[i]` / `s_partials[i]` arrays + the per-scale
  `h_l_partials[i]` / `h_c_partials[i]` / `h_s_partials[i]` pinned
  host shadows + the `submit()` <→ `collect()` work redistribution +
  the `cuEventRecord(s->lc.finished, s->lc.str)` +
  `vmaf_cuda_drain_batch_register(&s->lc)` tail) is also entirely
  fork-local.
- **Invariant**: the engine-scope drain-batch contract from ADR-0271
  / drain_batch.h. The kernel-launch order on `s->lc.str` must stay
  stable: `decimate (× 4)` then **for each scale `i ∈ 0..4`**
  `horiz` ⇒ `vert_lcs` ⇒ DtoH(l_partials[i]) ⇒ DtoH(c_partials[i])
  ⇒ DtoH(s_partials[i])` then `cuEventRecord(s->lc.finished, s->lc.str)`
  then `vmaf_cuda_drain_batch_register(&s->lc)`. Same-stream
  ordering is what makes the shared SSIM intermediates
  (`h_ref_mu`, `h_cmp_mu`, `h_ref_sq`, `h_cmp_sq`, `h_refcmp`)
  safe across scales without explicit sync — any change that
  parallelises the per-scale work onto multiple streams **breaks
  bit-exactness** unless per-scale intermediates are also added.
- **On upstream sync**: zero interaction (the file is fork-added).
  If a future upstream PR adds an `integer_ms_ssim_cuda.c` of its
  own, the merger must reconcile the per-scale partials topology +
  the drain_batch tail with whatever the new upstream shape brings.
- **Re-test on rebase**:

  ```bash
  cd libvmaf && meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build  # confirms the CPU build still links cleanly
  # If the dev host has a working nvcc / host-compiler pair:
  meson setup build_cuda -Denable_cuda=true -Denable_sycl=false
  ninja -C build_cuda src/liblibvmaf_feature.a.p/feature_cuda_integer_ms_ssim_cuda.c.o
  # Netflix CPU golden gate (CPU is the bit-exactness ground truth):
  make test-netflix-golden
  # Cross-backend parity (places=4 gate, ADR-0214):
  /cross-backend-diff
  ```
### 0277 — ffmpeg-patches refresh against `n8.1` — 2026-05-04 (ADR-0277)

- **Touches**: `ffmpeg-patches/` is unchanged (no content drift).
  Doc-only entries land in:
  - [`docs/adr/0277-ffmpeg-patches-refresh-2026-05-04.md`](adr/0277-ffmpeg-patches-refresh-2026-05-04.md) — new ADR.
  - [`docs/adr/_index_fragments/0277-ffmpeg-patches-refresh-2026-05-04.md`](adr/_index_fragments/0277-ffmpeg-patches-refresh-2026-05-04.md) — index row.
  - `docs/adr/_index_fragments/_order.txt` — manifest append.
  - `changelog.d/changed/ffmpeg-patches-refresh-2026-05-04.md` — Changed entry.
  - This file — this entry.
- **Invariant**: `ffmpeg-patches/series.txt` order is load-bearing —
  patches `0002…0006` build on each other and only apply cleanly
  cumulatively. The verification gate is a **series replay**, not
  a per-patch `git apply --check` (per ADR-0118 + CLAUDE.md §12 r14).
- **On upstream sync**: zero interaction. Netflix/vmaf has no
  `ffmpeg-patches/` tree; this is a fork-local integration surface.
- **Re-test on rebase** (also: re-replay procedure for the next
  refresh):

  ```bash
  # Clone pristine n8.1
  git -C /tmp clone --depth 1 --branch n8.1 \
    https://github.com/FFmpeg/FFmpeg.git ff-replay-$(date +%F)
  cd /tmp/ff-replay-$(date +%F)
  git switch -c refresh-$(date +%F)
  git config user.email refresh@local && git config user.name "Refresh Bot"

  # Replay the series cumulatively
  for p in /path/to/vmaf/ffmpeg-patches/000*-*.patch; do
    git am --3way "$p" || break
  done

  # Regenerate and compare to in-tree
  mkdir -p /tmp/ff-regen-$(date +%F)
  git format-patch n8.1.. -o /tmp/ff-regen-$(date +%F)/

  # Diff old vs new excluding pure format-patch noise
  for i in 1 2 3 4 5 6; do
    orig=$(ls /path/to/vmaf/ffmpeg-patches/000${i}-*.patch)
    regen=$(ls /tmp/ff-regen-$(date +%F)/000${i}-*.patch)
    diff -u \
      <(grep -v "^From [0-9a-f]\|^Date:\|^index " "$orig") \
      <(grep -v "^From [0-9a-f]\|^Date:\|^index " "$regen") \
      | head -40
  done
  ```

  If only stylistic diffs surface (`PATCH N/M` numbering, MIME
  headers, hunk-context counts, hunk offset shifts against
  cumulative state), keep originals — record a no-drift refresh
  ADR. If real content drift surfaces, regenerate and ship the
  refresh PR with the regenerated patches plus a content-summary
  ADR.

  End-to-end `vf_libvmaf` smoke is best run from CI
  ([`ffmpeg-integration.yml`](../.github/workflows/ffmpeg-integration.yml))
  against an installed libvmaf prefix — the meson-uninstalled
  `.pc` does not satisfy FFmpeg's `#include <libvmaf.h>` probe
  (the headers live under `libvmaf/libvmaf.h` only; the
  system-installed `.pc` carries an extra
  `-I${includedir}/libvmaf` shortcut that the uninstalled `.pc`
  omits).
### 0229 — T7-5 NOLINT-sweep closeout (ADR-0278)

- **Touched files**:
  - `libvmaf/src/feature/integer_adm.c` (1 NOLINT cite, line ~988
    `adm_decouple_s123` — upstream-mirror Netflix `966be8d5`).
  - `libvmaf/src/feature/cuda/ssimulacra2_cuda.c` (3 NOLINT cites:
    `ss2c_picture_to_linear_rgb`, `ss2c_host_combine`,
    `ss2c_run_scale_gpu` / `extract_fex_cuda`).
  - `libvmaf/src/feature/vulkan/ssimulacra2_vulkan.c` (3 NOLINT
    cites: `ss2v_setup_gaussian`, `ss2v_picture_to_linear_rgb`,
    `ss2v_run_scale`).
  - `libvmaf/src/feature/vulkan/cambi_vulkan.c` (1 NOLINT cite:
    `cambi_vk_extract`).
  - `libvmaf/src/feature/sycl/integer_adm_sycl.cpp` (6 cites,
    SYCL kernel-launch entries).
  - `libvmaf/src/feature/sycl/integer_motion_sycl.cpp` (2 cites).
  - `libvmaf/src/feature/sycl/integer_vif_sycl.cpp` (4 cites).
  - `libvmaf/tools/vmaf.c` (3 cites: `copy_picture_data`,
    `init_gpu_backends`, `main`).
- **Invariant**: zero behavioural change. Edits are inside
  comment blocks — appended `(ADR-0141 §2 ... load-bearing
  invariant; T7-5 sweep closeout — ADR-0278)` to existing
  prose justifications. No function bodies split. The 12 SYCL
  sites share an identical justification string verbatim;
  preserving the byte-for-byte duplicate is the load-bearing
  documentation pattern (grep-able across the SYCL TUs).
- **On upstream sync**: minimal interaction. The cite-only
  edits live inside comment blocks above the function
  signatures; rebases will surface them as touched lines but
  the function bodies are unchanged. For `integer_adm.c`'s
  upstream-mirror block (Netflix `966be8d5`), the comment
  edit at line 984–991 is cosmetic — keep the fork's version
  on conflict (it merely names the ADR; the underlying prose
  is unchanged).
- **Re-test on rebase**:

  ```bash
  # 1. Programmatic audit must report 0 missing citations
  python3 - <<'PY'
  import re, os
  paths = [os.path.join(r, f) for r, _, fs in os.walk('libvmaf/src')
           for f in fs if f.endswith(('.c','.cpp','.h'))]
  paths.append('libvmaf/tools/vmaf.c')
  miss = total = 0
  for p in paths:
      with open(p) as fh: ls = fh.readlines()
      for i, line in enumerate(ls):
          if 'NOLINT' in line and 'readability-function-size' in line and 'NOLINTEND' not in line:
              total += 1
              ctx = [line]; j = i - 1
              while j >= 0 and j > i - 14:
                  s = ls[j].strip()
                  if not s: break
                  if s.startswith(('//','/*','*')):
                      ctx.insert(0, ls[j]); j -= 1
                  else: break
              buf = ''.join(ctx)
              if 'ADR-' not in buf and not re.search(r'[Rr]esearch-?\d', buf):
                  miss += 1
  print(f"sites={total} missing={miss}")
  PY

  # 2. Build + Netflix golden gate
  meson setup build -Denable_cuda=false -Denable_sycl=false
  ninja -C build
  make test-netflix-golden
### 0231 — `vmaf-tune` score path decodes mp4 -> raw YUV

- **Touches**: `tools/vmaf-tune/src/vmaftune/score.py` (new
  `_decode_to_raw_yuv` + `_needs_decode` helpers, `run_score`
  shells out to ffmpeg when `req.distorted.suffix not in
  {.yuv, .y4m}`); `tools/vmaf-tune/tests/test_corpus.py` (3 new
  regression tests + the smoke-end-to-end mock now also stubs the
  ffmpeg decode call).
- **Invariant**: the decode-back is the contract the libvmaf CLI
  imposes — mp4/webm/etc. `--distorted` is silently rejected as
  raw-yuv with the wrong byte count, surfacing as `exit_status=234`.
  Future encoder adapters that emit non-raw containers inherit this
  decode automatically. **Do not** "optimise" the temp YUV away
  without first migrating the corpus pipeline to the
  `ffmpeg+libvmaf` filter (which can pipe an mp4 stream in directly).
- **On upstream sync**: zero interaction. `vmaf-tune` is fork-only
  tooling; upstream Netflix/vmaf has no analogue.
- **Re-test on rebase**:

  ```bash
  cd tools/vmaf-tune && python3 -m pytest tests/
  # plus an end-to-end smoke (needs a real raw YUV + ffmpeg + vmaf):
  ./vmaf-tune corpus --source /path/to/ref.yuv --width 1920 \
      --height 1080 --pix-fmt yuv420p --framerate 25 --duration 6 \
      --encoder libx264 --preset medium --crf 23 \
      --output /tmp/smoke.jsonl --no-source-hash
  # expect: vmaf_score is a real number, not NaN.
### 0232 — CUDA build pins nvcc `--std c++20`

- **Touches**: `libvmaf/src/meson.build` line 686 (`cuda_flags = [...]`).
- **Invariant**: nvcc 12.x clamps host C++ at C++17 by default; 13.x
  accepts up to C++20. Bumping the host stdlib past nvcc's default (any
  gcc >= 16, libstdc++ ships C++23 features) breaks the host-side parse
  in `<type_traits>` / `<bits/utility.h>`. Forcing `--std c++20` on
  CUDA 13+ keeps the host headers parseable. **Do not** drop this flag
  without first checking the host gcc version against nvcc's default.
- **On upstream sync**: zero interaction. Netflix/vmaf doesn't ship the
  `cuda_flags` list shape we use (their CUDA build is the original
  pre-fork pattern); a sync that touches `libvmaf/src/meson.build`
  around the `is_cuda_enabled` branch should keep the `--std c++20`
  injection.
- **Re-test on rebase**:

  ```bash
  meson setup libvmaf/build-cuda -Denable_cuda=true \
      -Denable_sycl=false -Denable_vulkan=disabled
  ninja -C libvmaf/build-cuda
  # smoke
  ./libvmaf/build-cuda/tools/vmaf --gpumask=0 --no_sycl --no_vulkan \
      -r .workingdir2/netflix/ref/BigBuckBunny_25fps.yuv \
      -d .workingdir2/netflix/ref/BigBuckBunny_25fps.yuv \
      -w 1920 -h 1080 -p 420 -b 8
  ```

### 0233 — CUDA motion `flush_fex_cuda` idempotency guard

- **Touches**: `libvmaf/src/feature/cuda/integer_motion_cuda.c` —
  factored an `append_if_unwritten` helper and routed the two
  motion2 / motion3 final-frame writes through it.
- **Invariant**: under T-GPU-OPT-1 (PR #312 / ADR-0242), the
  pending-collect inside `flush_context_cuda` may already have
  written `motion2_score[s->index]` / `motion3_score[s->index]`
  before `flush_fex_cuda` runs. Any future motion-cuda flush logic
  that emits the same (feature, index) pair must keep this
  idempotency contract or `flush_context_cuda` will mis-surface as
  "context could not be synchronized".
- **On upstream sync**: the bug only exists because the fork's
  `flush_context_cuda` runs the pending-collect *before* the
  per-extractor flush. Netflix/vmaf upstream doesn't have the
  T-GPU-OPT-1 drain pattern, so the pre-#312 code path didn't
  duplicate-write. If Netflix lands a similar pattern, the fix
  shape mirrors what's done here.
- **Re-test on rebase**:

  ```bash
  ninja -C libvmaf/build-cuda
  ./libvmaf/build-cuda/tools/vmaf --gpumask=0 --no_sycl --no_vulkan \
    -r python/test/resource/yuv/src01_hrc00_576x324.yuv \
    -d python/test/resource/yuv/src01_hrc01_576x324.yuv \
    -w 576 -h 324 -p 420 -b 8 \
    --model path=model/vmaf_v0.6.1.json --threads 1 -q \
    --output /tmp/cuda.json --json
  # Expect: clean run, no "cannot be overwritten" warning,
  # no "problem flushing context" error.
  ```

### 0234 — `hw_encoder_corpus.py` Phase A real-corpus runner

- **Touches**: new `scripts/dev/hw_encoder_corpus.py` (no existing
  caller; opt-in tooling). Output landing in `runs/phase_a/` is gitignored — rerun the script to reproduce.
  `docs/development/intel-arc-vaapi-driver-priority.md`. Output landing in `runs/phase_a/` is gitignored — rerun the script to reproduce.
  stratified sample, 58 KiB).
- **Invariant**: the script's QSV path forces
  `env['LIBVA_DRIVER_NAME']='iHD'` (set by the calling shell, not
  inside the script) when targeting `/dev/dri/renderD129` on a
  multi-card host that has NVIDIA's libva-driver-nvidia shim
  installed. Without that, libva picks up NVIDIA's NVDEC-VAAPI
  translation and the MFX session handshake fails with -9. See the
  companion doc for the failure mode + fix.
- **On upstream sync**: zero interaction. The script lives under
  `scripts/dev/` (fork-only); upstream Netflix/vmaf has no comparable
  Phase A corpus tooling.
- **Re-test on rebase**:

  ```bash
  python3 scripts/dev/hw_encoder_corpus.py \
    --vmaf-bin libvmaf/build-cuda/tools/vmaf \
    --source .workingdir2/netflix/ref/BigBuckBunny_25fps.yuv \
    --width 1920 --height 1080 --pix-fmt yuv420p --framerate 25 \
    --encoder h264_nvenc --cq 25 \
    --out /tmp/smoke.jsonl
  # Expect: 1 cell × ~150 frames, per-frame canonical-6 + vmaf,
  # encoder=h264_nvenc, cq=25.
  ```
