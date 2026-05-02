# Change Log

> The "Unreleased / lusoris fork" section below tracks fork-specific changes
> on top of upstream Netflix/vmaf. From here on, release-please generates
> entries automatically from Conventional Commits.

## [4.0.0-lusoris.0](https://github.com/lusoris/vmaf/compare/v3.0.0-lusoris.0...v4.0.0-lusoris.0) (2026-05-02)


### ⚠ BREAKING CHANGES

* **cli:** %.6f default + unref skipped frames so Netflix golden gate passes ([#55](https://github.com/lusoris/vmaf/issues/55))
* **cli:** default CLI score precision changed from %.6f to %.17g. Parsers that string-equal-compare VMAF outputs will see additional digits. Pass --precision=legacy to restore old behavior.

### Features

* **ai:** BVI-DVC feature-extraction pipeline (corpus-3 for tiny-AI v2) ([#214](https://github.com/lusoris/vmaf/issues/214)) ([362d9e7](https://github.com/lusoris/vmaf/commit/362d9e7f256288fd469215aec45fdd91966c1eb2))
* **ai:** Combined Netflix + KoNViD-1k tiny-AI trainer driver ([#180](https://github.com/lusoris/vmaf/issues/180)) ([a143e25](https://github.com/lusoris/vmaf/commit/a143e255c5d8130872be25bdc1fdb757fcc878cf))
* **ai:** Full vmaf-train package — fr/nr/filter families + CLI + registry ([91b5558](https://github.com/lusoris/vmaf/commit/91b5558e133bca0b47d1199be8ecff85bb3ac288))
* **ai:** KoNViD-1k → VMAF-pair acquisition + loader bridge ([#178](https://github.com/lusoris/vmaf/issues/178)) ([219f653](https://github.com/lusoris/vmaf/commit/219f65315d6470b20bc60e7af1905c683524404d))
* **ai:** Model validators + MCP eval + variance head + INT8 PTQ (4/5) ([#8](https://github.com/lusoris/vmaf/issues/8)) ([81f63a4](https://github.com/lusoris/vmaf/commit/81f63a4e4695e11d33276fe50b44644fb64e1fa7))
* **ai:** Research-0028 + Phase-3 subset-sweep driver (negative result) ([#188](https://github.com/lusoris/vmaf/issues/188)) ([7e4e884](https://github.com/lusoris/vmaf/commit/7e4e88423702b962512f630e642c937097fe7dfe))
* **ai:** Research-0029 Phase-3b — StandardScaler validates broader-feature hypothesis ([#192](https://github.com/lusoris/vmaf/issues/192)) ([a1453bf](https://github.com/lusoris/vmaf/commit/a1453bff4caad21e873a303a9666eb4fc586b103))
* **ai:** Research-0030 Phase-3b multi-seed validation (Gate 1 passed) ([#190](https://github.com/lusoris/vmaf/issues/190)) ([1f128b0](https://github.com/lusoris/vmaf/commit/1f128b01f75678c95f63dd8f848aaf2b4fc2e7b2))
* **ai:** Retrain vmaf_tiny_v2 on 4-corpus (NF+KV+BVI-A+B+C+D) ([#255](https://github.com/lusoris/vmaf/issues/255)) ([53837dc](https://github.com/lusoris/vmaf/commit/53837dcbba23c1a72653a238dfe6cd0a56b10411))
* **ai:** Ship vmaf_tiny_v2 (canonical-6 + StandardScaler + lr=1e-3 + 90ep, validated +0.018 PLCC over v1) ([#250](https://github.com/lusoris/vmaf/issues/250)) ([3999cda](https://github.com/lusoris/vmaf/commit/3999cdab6163cb4512bb5fb7b682b9e78c5e99cd))
* **ai:** T5-3d — nr_metric_v1 dynamic-batch fix + PTQ pipeline ([#247](https://github.com/lusoris/vmaf/issues/247)) ([9ff5d1d](https://github.com/lusoris/vmaf/commit/9ff5d1d48428d4f1ef01236a8927b2da26c6b3f5))
* **ai:** T5-3e empirical PTQ accuracy across CPU/CUDA/OpenVINO EPs ([#174](https://github.com/lusoris/vmaf/issues/174)) ([4130149](https://github.com/lusoris/vmaf/commit/413014963259ce54d3f06ac2f0d1659f6a2d2477))
* **ai:** T6-1a — fr_regressor_v1 C1 baseline (Netflix Public, unblocked) ([#249](https://github.com/lusoris/vmaf/issues/249)) ([f809ce0](https://github.com/lusoris/vmaf/commit/f809ce09c214156259790c91109cea02ce2640f6))
* **ai:** T6-2a — MobileSal saliency feature extractor ([#208](https://github.com/lusoris/vmaf/issues/208)) ([fa7d4f5](https://github.com/lusoris/vmaf/commit/fa7d4f52833d087a8d47236e5bdc85be5dd37205))
* **ai:** T6-3a — TransNet V2 shot-boundary feature extractor ([#210](https://github.com/lusoris/vmaf/issues/210)) ([08b7644](https://github.com/lusoris/vmaf/commit/08b7644f8a31c70623f52e5a1dd5c47ca5f08493))
* **ai:** T6-7 — FastDVDnet temporal pre-filter (5-frame window) ([#203](https://github.com/lusoris/vmaf/issues/203)) ([cf1d670](https://github.com/lusoris/vmaf/commit/cf1d670d74ea98ed66baf2e3e57acf686189a456))
* **ai:** T6-9 — model registry schema + Sigstore --tiny-model-verify ([#199](https://github.com/lusoris/vmaf/issues/199)) ([9293d69](https://github.com/lusoris/vmaf/commit/9293d6983d5b87525163c873f54a1b0b84845c74))
* **ai:** T7-CODEC-AWARE — codec-conditioned FR regressor surface (training BLOCKED) ([#237](https://github.com/lusoris/vmaf/issues/237)) ([876382d](https://github.com/lusoris/vmaf/commit/876382dc805bc53fe7f98d647ad5d0b98c72da60))
* **ai:** T7-DISTS — scaffold DISTS extractor proposal (ADR-0236) ([#259](https://github.com/lusoris/vmaf/issues/259)) ([da6d7b0](https://github.com/lusoris/vmaf/commit/da6d7b0e2fa6d0a692e2d9afd94c8af4743bda5e))
* **ai:** T7-GPU-ULP-CAL — scaffold GPU-gen ULP calibration head (proposal) ([#238](https://github.com/lusoris/vmaf/issues/238)) ([fdb8a56](https://github.com/lusoris/vmaf/commit/fdb8a5652d2f42bcf7d243af19804f1d4eb81b12))
* **ai:** Tiny-AI 3-arch LOSO evaluation harness + Research-0023 ([#176](https://github.com/lusoris/vmaf/issues/176)) ([0e483a3](https://github.com/lusoris/vmaf/commit/0e483a3e16461727762c01b9f2a7b1bdf2db1d72))
* **ai:** Tiny-AI feature-set registry (Research-0026 Phase 1) ([#185](https://github.com/lusoris/vmaf/issues/185)) ([4336736](https://github.com/lusoris/vmaf/commit/4336736ae2e52ab6a12728f9708777886fe6a5dd))
* **ai:** Tiny-AI LOSO evaluation harness for mlp_small ([#165](https://github.com/lusoris/vmaf/issues/165)) ([427272b](https://github.com/lusoris/vmaf/commit/427272b48b18e0feb3d239112c24318e0187f34c))
* **ai:** Tiny-AI Phase-2 analysis scaffolding (Research-0026) ([#191](https://github.com/lusoris/vmaf/issues/191)) ([0df6543](https://github.com/lusoris/vmaf/commit/0df6543a8803017a67ae5c0454d6ea0c53093152))
* **ai:** Tiny-AI Quantization-Aware Training (QAT) implementation (T5-4) ([#179](https://github.com/lusoris/vmaf/issues/179)) ([d71ac52](https://github.com/lusoris/vmaf/commit/d71ac52cf5b995e21262a89f87d468fc7c28eade))
* **ai:** Tiny-AI training prep (loader + eval + Lightning harness for Netflix corpus) ([#158](https://github.com/lusoris/vmaf/issues/158)) ([aa74eaa](https://github.com/lusoris/vmaf/commit/aa74eaa3b5ac74263caceb053a218ba113a32345))
* **arch:** T7-26 — feature-characteristics registry + per-backend dispatch_strategy ([#124](https://github.com/lusoris/vmaf/issues/124)) ([817b4dd](https://github.com/lusoris/vmaf/commit/817b4dd9279d35d75fad49ac5003c5206fb615b5))
* **batch-a:** Port four OPEN Netflix upstream PRs (ADR-0131/0132/0134/0135) ([#72](https://github.com/lusoris/vmaf/issues/72)) ([df622ea](https://github.com/lusoris/vmaf/commit/df622ea782987e3ab37d0f71aaf00903bf964c72))
* **ci:** Add FFmpeg + Vulkan integration lane (lavapipe) ([#235](https://github.com/lusoris/vmaf/issues/235)) ([5d6c1af](https://github.com/lusoris/vmaf/commit/5d6c1afe0cad9c2e0f49c5c94f48d71d9fd4e189))
* **ci:** Add MCP smoke CI lane (T7-MCP-SMOKE-CI) ([#256](https://github.com/lusoris/vmaf/issues/256)) ([41f3cee](https://github.com/lusoris/vmaf/commit/41f3ceee7b7de08d20c273eeea9945b146cac489))
* **ci:** Nightly bisect-model-quality + sticky tracker (closes [#4](https://github.com/lusoris/vmaf/issues/4)) ([#41](https://github.com/lusoris/vmaf/issues/41)) ([6cd4fb0](https://github.com/lusoris/vmaf/commit/6cd4fb01a81cff13996ad94cf552f7b61104cf60))
* **ci:** T6-8 — GPU-parity cross-device variance gate ([#202](https://github.com/lusoris/vmaf/issues/202)) ([e0b381b](https://github.com/lusoris/vmaf/commit/e0b381bfb8132c94f684d59d1dd7547131db3677))
* **cli:** --precision flag for IEEE-754 round-trip lossless scores ([c989fbd](https://github.com/lusoris/vmaf/commit/c989fbd913c7228b40e86af5176d265d3353c284))
* **cli:** Wire --tiny-model / --tiny-device / --no-reference flags ([6381819](https://github.com/lusoris/vmaf/commit/63818196a6d12a7cd37175867ae1910beb085946))
* **cuda,sycl:** GPU long-tail batch 1c parts 2 + 3 — ciede_{cuda,sycl} ([#137](https://github.com/lusoris/vmaf/issues/137)) ([c2fb1de](https://github.com/lusoris/vmaf/commit/c2fb1de667126f9233a22510addbdb5b672d2f7e))
* **cuda,sycl:** GPU long-tail batch 1d parts 2 + 3 — float_moment_{cuda,sycl} ([#135](https://github.com/lusoris/vmaf/issues/135)) ([0c9c117](https://github.com/lusoris/vmaf/commit/0c9c117be653014c9f8d4d39cd13598c30c9b521))
* **cuda,sycl:** GPU long-tail batch 2 parts 1b + 1c — float_ssim_{cuda,sycl} ([#140](https://github.com/lusoris/vmaf/issues/140)) ([c0f6979](https://github.com/lusoris/vmaf/commit/c0f6979493edefebd35473b824a7d86508301d3f))
* **cuda,sycl:** GPU long-tail batch 2 parts 2b + 2c — float_ms_ssim_{cuda,sycl} ([#142](https://github.com/lusoris/vmaf/issues/142)) ([8db2715](https://github.com/lusoris/vmaf/commit/8db2715ac20f19902ad63b09afbe0d5b24a52b8b))
* **cuda,sycl:** GPU long-tail batch 2 parts 3b + 3c — psnr_hvs_{cuda,sycl} ([#144](https://github.com/lusoris/vmaf/issues/144)) ([4fa2504](https://github.com/lusoris/vmaf/commit/4fa2504dbcd9bae7a84de913f29e5b1f1697d292))
* **cuda,sycl:** GPU long-tail batch 3 part 6b + 6c — float_adm_{cuda,sycl} (redo of [#157](https://github.com/lusoris/vmaf/issues/157)) ([#163](https://github.com/lusoris/vmaf/issues/163)) ([5af6e0b](https://github.com/lusoris/vmaf/commit/5af6e0b6f7c3a9ccf405f6316a7a433108f6472c))
* **cuda,sycl:** GPU long-tail batch 3 part 7b + 7c — ssimulacra2_{cuda,sycl} ([#162](https://github.com/lusoris/vmaf/issues/162)) ([f560bb2](https://github.com/lusoris/vmaf/commit/f560bb2ca197364770d6ed95e72e97c9b3080077))
* **cuda,sycl:** GPU long-tail batch 3 parts 1b + 1c — motion_v2_{cuda,sycl} ([#147](https://github.com/lusoris/vmaf/issues/147)) ([eea948f](https://github.com/lusoris/vmaf/commit/eea948f4c00b73f4ce2fd2d0e5aceff59eb0e880))
* **cuda:** GPU long-tail batch 1b part 1 — psnr_cuda kernel + host ([#129](https://github.com/lusoris/vmaf/issues/129)) ([27cca81](https://github.com/lusoris/vmaf/commit/27cca81642c5cc6c01c257309d5338bc237134fa))
* **dev-llm:** Ollama-backed review / commitmsg / docgen helpers ([80ead16](https://github.com/lusoris/vmaf/commit/80ead1668e8b3ffba6bee7c7bde2492b649da68c))
* DNN session runtime + SYCL list-devices + windows CI (1/5) ([#5](https://github.com/lusoris/vmaf/issues/5)) ([3cfa85b](https://github.com/lusoris/vmaf/commit/3cfa85bb4cad0b054df76c1e75c2a9761a4bdc56))
* **dnn:** Admit Loop + If on ONNX op-allowlist with recursive subgraph scan (T6-5, ADR-0169) ([#105](https://github.com/lusoris/vmaf/issues/105)) ([c4bd6ff](https://github.com/lusoris/vmaf/commit/c4bd6ff04fd03616cd24c7291d4be213a3638575))
* **dnn:** Bounded Loop.M trip-count guard (T6-5b, ADR-0171) ([#107](https://github.com/lusoris/vmaf/issues/107)) ([c264eec](https://github.com/lusoris/vmaf/commit/c264eec78d51e5a19cf35a59c76f3a7393c1e071))
* **dnn:** Enforce VMAF_TINY_MODEL_DIR path jail on model load ([#31](https://github.com/lusoris/vmaf/issues/31)) ([22ba0e5](https://github.com/lusoris/vmaf/commit/22ba0e5789853edbd22d6dd5a42a7fa43e64fb99))
* **dnn:** LPIPS-SqueezeNet FR extractor + ONNX + registry entry ([#23](https://github.com/lusoris/vmaf/issues/23)) ([0267dfb](https://github.com/lusoris/vmaf/commit/0267dfbe83b28366fcad56268d74d351f3b42db7))
* **dnn:** Multi-input session API + ImageNet RGB tensor helper ([#22](https://github.com/lusoris/vmaf/issues/22)) ([d47481b](https://github.com/lusoris/vmaf/commit/d47481b9c84b9cdb14eb5e1de2e0825c5cede2a4))
* **dnn:** Ordered EP selection + fp16_io host-side cast ([#34](https://github.com/lusoris/vmaf/issues/34)) ([fad2b13](https://github.com/lusoris/vmaf/commit/fad2b13ff2eec27682377fd2def359f81d9bfcea))
* **dnn:** Runtime op-allowlist walk + tiny-model registry v0 ([#21](https://github.com/lusoris/vmaf/issues/21)) ([e98a220](https://github.com/lusoris/vmaf/commit/e98a220b2d8af2a3df8505bc5ea27365df589d07))
* **dnn:** Scaffold tiny-AI training, ONNX export, and DNN C seam ([d122b72](https://github.com/lusoris/vmaf/commit/d122b72122e35a5e40399a0625ee07c227baf765))
* **dnn:** Tests + ffmpeg patches + ci dnn job ([1e5336d](https://github.com/lusoris/vmaf/commit/1e5336d343022f5e292095c593441c906bd49ef7))
* **dnn:** Vmaf_pre 10/12-bit + optional chroma (T6-4, ADR-0170) ([#106](https://github.com/lusoris/vmaf/issues/106)) ([8b4a64c](https://github.com/lusoris/vmaf/commit/8b4a64c567a9620c019dd8e15b7a5cf012c18d44))
* **ffmpeg:** T7-28 — add libvmaf_sycl filter (zero-copy QSV/VAAPI) ([#127](https://github.com/lusoris/vmaf/issues/127)) ([a2aa094](https://github.com/lusoris/vmaf/commit/a2aa09489006d87e631eed92f40b753302d3d0f2))
* **gpu:** GPU long-tail batch 1a — psnr_vulkan kernel + chars seeds ([#125](https://github.com/lusoris/vmaf/issues/125)) ([058c970](https://github.com/lusoris/vmaf/commit/058c970a4d1c3b89f4cac404559b231912f9e697))
* **gpu:** T3-15(c) — motion3 GPU coverage on Vulkan + CUDA + SYCL ([#248](https://github.com/lusoris/vmaf/issues/248)) ([f4ea952](https://github.com/lusoris/vmaf/commit/f4ea9523e437db9ca5b41abaf8c6fd4e866f37fc))
* **gpu:** T7-35 — enable_lcs MS-SSIM on CUDA + Vulkan + SYCL ([#207](https://github.com/lusoris/vmaf/issues/207)) ([dbccc1b](https://github.com/lusoris/vmaf/commit/dbccc1b5290afa2f0fe69c98613c2a8925af53df))
* **hip:** T7-10 — HIP (AMD) backend scaffold (audit-first) ([#200](https://github.com/lusoris/vmaf/issues/200)) ([2d1ccb1](https://github.com/lusoris/vmaf/commit/2d1ccb1a721dec6c13d78bacb71d7c3acdb7f471))
* **iqa,ssim:** Bit-exact AVX2 + AVX-512 convolve + SSIM accumulate ([#76](https://github.com/lusoris/vmaf/issues/76)) ([cf063b8](https://github.com/lusoris/vmaf/commit/cf063b8bfba6525a027c00fbea575833cf68e056))
* **libvmaf/feature:** Port upstream ADM updates (Netflix 966be8d5) ([#44](https://github.com/lusoris/vmaf/issues/44)) ([d06dd6c](https://github.com/lusoris/vmaf/commit/d06dd6cfc82b11f958fcf92c7d90219ca8ed2c2d))
* **libvmaf/feature:** Port upstream motion updates (Netflix PR [#1486](https://github.com/lusoris/vmaf/issues/1486)) ([#45](https://github.com/lusoris/vmaf/issues/45)) ([9371a0a](https://github.com/lusoris/vmaf/commit/9371a0aa176342f7e0e1de6e3da3585174c1c49f))
* **libvmaf:** Compile picture_pool unconditionally, drop VMAF_PICTURE_POOL gate ([#32](https://github.com/lusoris/vmaf/issues/32)) ([65460e3](https://github.com/lusoris/vmaf/commit/65460e3ad8d5c0b6399981d156e75e52b8f742bc))
* **libvmaf:** DNN runtime — public dnn.h + ORT backend + sidecar loader ([9b98594](https://github.com/lusoris/vmaf/commit/9b9859467aa3cf0cd99c49732b1b4ee9eeeea9b8))
* **mcp:** Add vmaf-mcp server exposing VMAF to LLM tooling ([0f233b8](https://github.com/lusoris/vmaf/commit/0f233b8075529b1a42fbe77becacf86b5075d1a3))
* **mcp:** Describe_worst_frames tool with VLM fallback (T6-6, ADR-0172) ([#108](https://github.com/lusoris/vmaf/issues/108)) ([5de8a79](https://github.com/lusoris/vmaf/commit/5de8a793eb2c13cfca10079fa3d18c0620cea130))
* **mcp:** T5-2 — embedded MCP scaffold (audit-first) ([#195](https://github.com/lusoris/vmaf/issues/195)) ([8f46b22](https://github.com/lusoris/vmaf/commit/8f46b22cdad272d8f45853ef57e186ec4df1b0e2))
* **motion:** Motion_v2 NEON SIMD port (ADR-0145) ([#81](https://github.com/lusoris/vmaf/issues/81)) ([2dee153](https://github.com/lusoris/vmaf/commit/2dee15322806abacf7c9ebf064e028cde7da2e2b))
* **motion:** Port Netflix b949cebf — feature/motion: port several feature extractor options ([#197](https://github.com/lusoris/vmaf/issues/197)) ([3806beb](https://github.com/lusoris/vmaf/commit/3806beb99c6c288ba2ab700885542893bfeeba29))
* **ms-ssim:** NEON bit-identical decimate (ADR-0125) ([4531913](https://github.com/lusoris/vmaf/commit/45319133e2e8e3c4f11806e924e7cbf49e4f1900))
* **ms-ssim:** Separable decimate + AVX2/AVX-512 SIMD (ADR-0125) ([3283a61](https://github.com/lusoris/vmaf/commit/3283a61238cc6296c6b825d35b1d6f39bbcadc9f))
* **process:** T7 bundle — docs/state.md, MCP release channel, GPU runner guide, doc-drift enforcement ([#103](https://github.com/lusoris/vmaf/issues/103)) ([8e6b99a](https://github.com/lusoris/vmaf/commit/8e6b99acfc8eaf31c825c3358a9463720e9dfa3d))
* **psnr_hvs:** AVX2 bit-exact port — 8×8 integer DCT vectorized (T3-5, ADR-0159) ([#96](https://github.com/lusoris/vmaf/issues/96)) ([ad57d33](https://github.com/lusoris/vmaf/commit/ad57d331b17e5a1f4d74b02a6f85c0fbf1ec2095))
* **psnr_hvs:** NEON aarch64 sister port — 8×8 integer DCT vectorized (T3-5-neon, ADR-0160) ([#97](https://github.com/lusoris/vmaf/issues/97)) ([98d359f](https://github.com/lusoris/vmaf/commit/98d359f1c3c22a93c02cee4b8806fa5d1a933a6e))
* **simd:** Float_moment AVX2 + NEON parity (T7-19, ADR-0179) ([#122](https://github.com/lusoris/vmaf/issues/122)) ([095e68a](https://github.com/lusoris/vmaf/commit/095e68af611e4b339fbc2cc263d9a170a25dd74b))
* **simd:** SIMD DX framework + NEON bit-exactness port (ADR-0140) ([#77](https://github.com/lusoris/vmaf/issues/77)) ([0d23d8c](https://github.com/lusoris/vmaf/commit/0d23d8c1e99730c58cb51406e13694a4ab18874a))
* **simd:** T7-38 — SSIMULACRA 2 PTLR + IIR-blur SVE2 ports ([#201](https://github.com/lusoris/vmaf/issues/201)) ([0471d75](https://github.com/lusoris/vmaf/commit/0471d75b3aa57706576f6806f9b7ac05a71a92db))
* **ssimulacra2:** IIR blur SIMD port — AVX2 + AVX-512 + NEON (T3-1 phase 2, ADR-0162) ([#99](https://github.com/lusoris/vmaf/issues/99)) ([34bef60](https://github.com/lusoris/vmaf/commit/34bef60a8db3b9d4c009f21414672f37132892c4))
* **ssimulacra2:** Picture_to_linear_rgb SIMD — AVX2 + AVX-512 + NEON (T3-1 phase 3, ADR-0163) ([#100](https://github.com/lusoris/vmaf/issues/100)) ([3b9d351](https://github.com/lusoris/vmaf/commit/3b9d351dcd25f40d86b66bbb886479f492187c79))
* **ssimulacra2:** Scalar port with libjxl FastGaussian IIR blur ([#68](https://github.com/lusoris/vmaf/issues/68)) ([410cf7b](https://github.com/lusoris/vmaf/commit/410cf7bc2180f3ebdc8793de0e92c696cac9d037))
* **sycl:** GPU long-tail batch 1b part 2 — psnr_sycl kernel ([#130](https://github.com/lusoris/vmaf/issues/130)) ([d10343f](https://github.com/lusoris/vmaf/commit/d10343fcc312960fed3b7d25239759fa7f43dee6))
* **sycl:** Implement USM-backed picture pre-allocation pool ([#33](https://github.com/lusoris/vmaf/issues/33)) ([3f9a134](https://github.com/lusoris/vmaf/commit/3f9a1345b6d1480f07ac10b383e334f433ac8514))
* **sycl:** Implement vmaf_sycl_import_d3d11_surface (closes [#27](https://github.com/lusoris/vmaf/issues/27)) ([#35](https://github.com/lusoris/vmaf/issues/35)) ([8ea83f7](https://github.com/lusoris/vmaf/commit/8ea83f791412887c30873edb55c8d19b131488f2))
* **tiny-ai:** C2 nr_metric_v1 + C3 learned_filter_v1 baselines on KoNViD-1k (T6-1, ADR-0168) ([#104](https://github.com/lusoris/vmaf/issues/104)) ([1a48eab](https://github.com/lusoris/vmaf/commit/1a48eab976f39a926dec5f01d94bc3c363755522))
* **tiny-ai:** First per-model PTQ — learned_filter_v1 dynamic int8 (T5-3b, ADR-0174) ([#110](https://github.com/lusoris/vmaf/issues/110)) ([07a3053](https://github.com/lusoris/vmaf/commit/07a30535c97314851849237a235981549d5af4cf))
* **tiny-ai:** PTQ int8 audit harness — registry + scripts + sidecar parser (T5-3, ADR-0173) ([#109](https://github.com/lusoris/vmaf/issues/109)) ([a7f2664](https://github.com/lusoris/vmaf/commit/a7f2664cb4948a985c5ec52e7b1cc508596f5804))
* **tools:** T6-2b — vmaf-roi sidecar for per-CTU QP offsets (x265 / SVT-AV1) ([#246](https://github.com/lusoris/vmaf/issues/246)) ([a0a20e2](https://github.com/lusoris/vmaf/commit/a0a20e2fcb855b8ad4271078ba9b2b9e80950b15))
* **tools:** T6-3b — vmaf-perShot per-shot CRF predictor ([#244](https://github.com/lusoris/vmaf/issues/244)) ([f398668](https://github.com/lusoris/vmaf/commit/f398668f1f2437a7cc234aedd7b0d36de50c255b))
* **upstream:** Port 8a289703 + 1b6c3886 — 32-bit ADM/cpu fallbacks ([#212](https://github.com/lusoris/vmaf/issues/212)) ([968cb3f](https://github.com/lusoris/vmaf/commit/968cb3fce396471f2727cff8c0d3192769e53b4b))
* **upstream:** Port c70debb1 — adm+vif test deltas ([#211](https://github.com/lusoris/vmaf/issues/211)) ([e66b510](https://github.com/lusoris/vmaf/commit/e66b5100195eab29e07108a7aa20a5f77a12344c))
* **upstream:** Port d3647c73 — feature/speed extractors (speed_chroma + speed_temporal) ([#213](https://github.com/lusoris/vmaf/issues/213)) ([32f2757](https://github.com/lusoris/vmaf/commit/32f2757882a2e469d0263c8fb9cd8d2dd04a451a))
* **vulkan,cuda,sycl:** GPU long-tail batch 3 part 2 — float_ansnr on all three backends ([#148](https://github.com/lusoris/vmaf/issues/148)) ([3017969](https://github.com/lusoris/vmaf/commit/30179695ae6255b63256eb7222bb4121cc9df1a9))
* **vulkan,cuda,sycl:** GPU long-tail batch 3 part 3 — float_psnr on all three backends ([#149](https://github.com/lusoris/vmaf/issues/149)) ([c9f0b99](https://github.com/lusoris/vmaf/commit/c9f0b99dc5a5cce1469b5e1f511749a77407e459))
* **vulkan,cuda,sycl:** GPU long-tail batch 3 part 4 — float_motion on all three backends ([#150](https://github.com/lusoris/vmaf/issues/150)) ([ac91174](https://github.com/lusoris/vmaf/commit/ac91174a71949a7c183749784227f564890f7eaf))
* **vulkan,cuda,sycl:** GPU long-tail batch 3 part 5 — float_vif on all three backends ([#151](https://github.com/lusoris/vmaf/issues/151)) ([a34e08d](https://github.com/lusoris/vmaf/commit/a34e08d96212a50b367a5191e955e625329cfe49))
* **vulkan:** GPU long-tail batch 1c part 1 — ciede_vulkan kernel ([#136](https://github.com/lusoris/vmaf/issues/136)) ([9286ace](https://github.com/lusoris/vmaf/commit/9286ace919ef7666727367f916eab999266a7824))
* **vulkan:** GPU long-tail batch 1d part 1 — float_moment_vulkan kernel ([#133](https://github.com/lusoris/vmaf/issues/133)) ([a64cde6](https://github.com/lusoris/vmaf/commit/a64cde633aedb60f9a6f408bae4b0bc4f982502c))
* **vulkan:** GPU long-tail batch 2 part 1a — float_ssim_vulkan kernel ([#139](https://github.com/lusoris/vmaf/issues/139)) ([0d3767e](https://github.com/lusoris/vmaf/commit/0d3767e4eea4176fe95ec3c72c24d180cd124166))
* **vulkan:** GPU long-tail batch 2 part 2a — float_ms_ssim_vulkan kernel ([#141](https://github.com/lusoris/vmaf/issues/141)) ([ad185a1](https://github.com/lusoris/vmaf/commit/ad185a1129346330633a38a1e1450e97600f54cb))
* **vulkan:** GPU long-tail batch 2 part 3a — psnr_hvs_vulkan kernel ([#143](https://github.com/lusoris/vmaf/issues/143)) ([94fb39f](https://github.com/lusoris/vmaf/commit/94fb39f9e5dc7e451e0a65fcb5a9205a5c906bb2))
* **vulkan:** GPU long-tail batch 3 part 1a — motion_v2_vulkan kernel ([#146](https://github.com/lusoris/vmaf/issues/146)) ([41f3ae4](https://github.com/lusoris/vmaf/commit/41f3ae4c7307e7a0f41a44a9195c77d530282ad3))
* **vulkan:** GPU long-tail batch 3 part 6 — float_adm_vulkan kernel ([#154](https://github.com/lusoris/vmaf/issues/154)) ([ef73440](https://github.com/lusoris/vmaf/commit/ef73440c4edab3b12ed36e47ea633da82ddcb060))
* **vulkan:** GPU long-tail batch 3 part 7 — ssimulacra2_vulkan kernel ([#156](https://github.com/lusoris/vmaf/issues/156)) ([2ebf376](https://github.com/lusoris/vmaf/commit/2ebf3764124a2279eb294380610ac3fbce35bfc2))
* **vulkan:** Scaffold-only audit-first PR for the Vulkan compute backend (T5-1, ADR-0175) ([#111](https://github.com/lusoris/vmaf/issues/111)) ([0b787cf](https://github.com/lusoris/vmaf/commit/0b787cf7f12f867f80c990774c3e02f95b353143))
* **vulkan:** T-VULKAN-PREALLOC — picture preallocation surface (ADR-0238) ([#264](https://github.com/lusoris/vmaf/issues/264)) ([2493d90](https://github.com/lusoris/vmaf/commit/2493d90bffee7c8256d0af4e8717e086d3e8ee84))
* **vulkan:** T3-15(b) — psnr chroma (psnr_cb / psnr_cr) on Vulkan ([#204](https://github.com/lusoris/vmaf/issues/204)) ([a78d6e9](https://github.com/lusoris/vmaf/commit/a78d6e95ecc218457a0853089139cecc8ecb9371))
* **vulkan:** T5-1b runtime + VIF dispatch pathfinder (Arc A380 verified) ([#116](https://github.com/lusoris/vmaf/issues/116)) ([bf5f861](https://github.com/lusoris/vmaf/commit/bf5f861b5815f7e87ea002c2ad83e53df4d56f11))
* **vulkan:** T5-1b-iv VIF math port (4-scale GLSL kernel + vif_vulkan extractor) ([#117](https://github.com/lusoris/vmaf/issues/117)) ([acf9f5b](https://github.com/lusoris/vmaf/commit/acf9f5b8e3a9bcc3d13e33bd2cd6708f2b1cd4f3))
* **vulkan:** T5-1b-v cross-backend gate + CLI (--vulkan_device) ([#118](https://github.com/lusoris/vmaf/issues/118)) ([50758ea](https://github.com/lusoris/vmaf/commit/50758ea8f24b0cc44e1b85059d92f6c1207d1028))
* **vulkan:** T5-1c motion kernel + cross-backend gate extension ([#119](https://github.com/lusoris/vmaf/issues/119)) ([32e31e4](https://github.com/lusoris/vmaf/commit/32e31e4568a9b71f223e43a500c2af8fecf4401d))
* **vulkan:** T5-1c-adm ADM kernel + cross-backend bug-fix marathon ([#120](https://github.com/lusoris/vmaf/issues/120)) ([7c5b63a](https://github.com/lusoris/vmaf/commit/7c5b63a282afb4e804815d0600a322485f4e8d11))
* **vulkan:** T7-29 follow-up [#3](https://github.com/lusoris/vmaf/issues/3) — public max_outstanding_frames knob (ADR-0235) ([#260](https://github.com/lusoris/vmaf/issues/260)) ([1891746](https://github.com/lusoris/vmaf/commit/18917462a1afa2d626019d379bde7a677da46c4e))
* **vulkan:** T7-29 part 1 — VkImage import C-API scaffold ([#128](https://github.com/lusoris/vmaf/issues/128)) ([6bea86d](https://github.com/lusoris/vmaf/commit/6bea86d858a11af923f79cee88128a9d71532c39))
* **vulkan:** T7-29 part 4 — v2 async pending-fence ring (ADR-0235) ([#241](https://github.com/lusoris/vmaf/issues/241)) ([e266bf8](https://github.com/lusoris/vmaf/commit/e266bf8ee034f007490fa9653e9a74038fd0f0c2))
* **vulkan:** T7-29 parts 2 + 3 — VkImage import impl + libvmaf_vulkan filter ([#134](https://github.com/lusoris/vmaf/issues/134)) ([fe31f80](https://github.com/lusoris/vmaf/commit/fe31f803a43c5ea532fabe2c55a9dea871a3f6cd))
* **vulkan:** T7-36 — cambi Vulkan integration (Strategy II) ([#196](https://github.com/lusoris/vmaf/issues/196)) ([9f88b91](https://github.com/lusoris/vmaf/commit/9f88b91b7b7589686eee265391d7954cc877de56))


### Bug Fixes

* **adm:** Close T7-16 — Vulkan/SYCL adm_scale2 drift verified at places=4 ([#173](https://github.com/lusoris/vmaf/issues/173)) ([0fb482f](https://github.com/lusoris/vmaf/commit/0fb482fba944cf9b21dd671914deb732df0413de))
* **ai:** Bump torch &gt;=2.8 + lightning &gt;=2.5 for CVE fixes ([#11](https://github.com/lusoris/vmaf/issues/11)) ([98da94d](https://github.com/lusoris/vmaf/commit/98da94d616433879c5d5d9326941fbcec3143a33))
* **ai:** Regenerate bisect cache with pinned pandas 2.3.3 ([#42](https://github.com/lusoris/vmaf/issues/42)) ([cdb9b9e](https://github.com/lusoris/vmaf/commit/cdb9b9ea01dd70682f7e9b6a7cdf16bfa2bbca9a))
* **ai:** Switch lightning → pytorch-lightning (PyPI 404) ([#232](https://github.com/lusoris/vmaf/issues/232)) ([c182dfe](https://github.com/lusoris/vmaf/commit/c182dfeb5672159685f03b4d80128daac859cd6a))
* **bench:** Testdata/bench_all.sh engages the backends it benches ([#171](https://github.com/lusoris/vmaf/issues/171)) ([1a1e5eb](https://github.com/lusoris/vmaf/commit/1a1e5eb0af82fedc0bf6c2396e346e612f9e6944))
* **build:** Install libvmaf_vulkan.h under prefix when enable_vulkan ([#175](https://github.com/lusoris/vmaf/issues/175)) ([4b43ad2](https://github.com/lusoris/vmaf/commit/4b43ad2f60c812fe199d7fc64caa74f93cf24b14))
* **ci:** Coverage gate lcov→gcovr + ORT + lint upstream tests in-tree ([#46](https://github.com/lusoris/vmaf/issues/46)) ([652aa70](https://github.com/lusoris/vmaf/commit/652aa70b3457213db881240c07b821abee374af9))
* **ci:** Drop dead sycl trigger + consolidate windows.yml into libvmaf.yml ([#50](https://github.com/lusoris/vmaf/issues/50)) ([e01314e](https://github.com/lusoris/vmaf/commit/e01314e109247ee7b86b66fd3c1d38c301a60889))
* **ci:** Exclude binary model payloads from mixed-line-ending hook ([#36](https://github.com/lusoris/vmaf/issues/36)) ([0f61607](https://github.com/lusoris/vmaf/commit/0f61607d617afcb6390b0aa29bb8a21bd334233e))
* **ci:** Pin pip &lt;26.1 in Tiny AI workflow (lightning regression) ([#231](https://github.com/lusoris/vmaf/issues/231)) ([dd0a4fb](https://github.com/lusoris/vmaf/commit/dd0a4fb410e530ccc6f95643e6ddfba5dd155eb7))
* **ci:** Post bisect sticky comment via stdin instead of -f [@file](https://github.com/file) ([#43](https://github.com/lusoris/vmaf/issues/43)) ([f453d1b](https://github.com/lusoris/vmaf/commit/f453d1b7ae6b39764844c7c9e5b8b76327f3afb7))
* **ci:** Silence Coverage Gate annotations (upload-artifact v7 + gcovr filter) ([#54](https://github.com/lusoris/vmaf/issues/54)) ([fef9a86](https://github.com/lusoris/vmaf/commit/fef9a86d7a8a45eab924ae1e8650339558a09d34))
* **ci:** Skip pytest doctest collection of vmaf/resource/ data files ([#51](https://github.com/lusoris/vmaf/issues/51)) ([d23005c](https://github.com/lusoris/vmaf/commit/d23005c704a9191feaa4c6b9216074a672736f57))
* **cli:** --backend cuda actually engages CUDA (was silently CPU) ([#170](https://github.com/lusoris/vmaf/issues/170)) ([334c518](https://github.com/lusoris/vmaf/commit/334c518400382bbd85827724cb86cb04db786000))
* **cli:** %.6f default + unref skipped frames so Netflix golden gate passes ([#55](https://github.com/lusoris/vmaf/issues/55)) ([aa08d84](https://github.com/lusoris/vmaf/commit/aa08d84f8a7ff49191de1ea0b4fb1cb030447060))
* **cuda:** Graceful error propagation instead of assert(0) (Netflix[#1420](https://github.com/lusoris/vmaf/issues/1420), ADR-0156) ([#93](https://github.com/lusoris/vmaf/issues/93)) ([49a6408](https://github.com/lusoris/vmaf/commit/49a640887f81819219d577bb611f86acd6cf2de4))
* **cuda:** Guard vmaf_picture_ref against NULL src-&gt;ref on device-only path ([#62](https://github.com/lusoris/vmaf/issues/62)) ([661a8ac](https://github.com/lusoris/vmaf/commit/661a8ac9a934d33aef7ec3e32bc05444187addea))
* **cuda:** Preallocation memory leak + new vmaf_cuda_state_free API (Netflix[#1300](https://github.com/lusoris/vmaf/issues/1300), ADR-0157) ([#94](https://github.com/lusoris/vmaf/issues/94)) ([fd1b22c](https://github.com/lusoris/vmaf/commit/fd1b22c267c1656af09502f6d8e465a83071f56b))
* **cuda:** Unconditional sm_86/sm_89 cubin coverage + actionable init-failure logging ([#60](https://github.com/lusoris/vmaf/issues/60)) ([d3b6fad](https://github.com/lusoris/vmaf/commit/d3b6fad62756fd36d7870f449f7df8d871e0a82e))
* **ffmpeg-patches:** Dynamically load vkGetDeviceQueue (VK_NO_PROTOTYPES) ([#234](https://github.com/lusoris/vmaf/issues/234)) ([3130ca4](https://github.com/lusoris/vmaf/commit/3130ca4152292315629d1c6d96b082fa339a8216))
* **float_ms_ssim:** Reject &lt;176x176 at init with -EINVAL (Netflix[#1414](https://github.com/lusoris/vmaf/issues/1414), ADR-0153) ([#90](https://github.com/lusoris/vmaf/issues/90)) ([7905ac7](https://github.com/lusoris/vmaf/commit/7905ac78448095109e6b3421ebb5a04c0dc64c37))
* **gitleaks:** Allowlist manifests/README.md false positive ([#59](https://github.com/lusoris/vmaf/issues/59)) ([5786a70](https://github.com/lusoris/vmaf/commit/5786a707b8c06a843aec2944e34298617ff73f30))
* **hooks:** Parse Claude Code hook input from stdin JSON ([#89](https://github.com/lusoris/vmaf/issues/89)) ([0e7327d](https://github.com/lusoris/vmaf/commit/0e7327d34b3640f470e3c614c10f22f4cd6ff6ee))
* **libvmaf/feature:** Free VIF init base pointer on fail path ([#47](https://github.com/lusoris/vmaf/issues/47)) ([d8ab927](https://github.com/lusoris/vmaf/commit/d8ab927fb52c39470398edf08bf68e26c8cc9e0c))
* **libvmaf:** Gate -fsycl link arg on icpx CXX, allow gcc/clang host linker ([#52](https://github.com/lusoris/vmaf/issues/52)) ([4a8322d](https://github.com/lusoris/vmaf/commit/4a8322d27f285fe94dd969b36282b249942bbf47))
* **libvmaf:** Score_pooled returns -EAGAIN for pending features (Netflix[#755](https://github.com/lusoris/vmaf/issues/755), ADR-0154) ([#91](https://github.com/lusoris/vmaf/issues/91)) ([9b983e0](https://github.com/lusoris/vmaf/commit/9b983e0a96f23892960c380295f6b7e92bf2d7b9))
* **libvmaf:** Vmaf_read_pictures rejects non-monotonic indices (Netflix[#910](https://github.com/lusoris/vmaf/issues/910), ADR-0152) ([#88](https://github.com/lusoris/vmaf/issues/88)) ([f478c65](https://github.com/lusoris/vmaf/commit/f478c65de675543e46d5cf43f441645d8faeeadd))
* **motion:** Close T7-15 — CUDA/SYCL motion drift verified bit-exact on master ([#172](https://github.com/lusoris/vmaf/issues/172)) ([00cbc92](https://github.com/lusoris/vmaf/commit/00cbc921a5709f42d517d3d0281fcc119bda80af))
* SIMD bit-identical reductions + CI fixes ([#18](https://github.com/lusoris/vmaf/issues/18)) ([f082cfd](https://github.com/lusoris/vmaf/commit/f082cfd3a5eb471ca5b32e8f7ea32854c95ed152))
* **sycl:** Require icpx as project C++ compiler when enable_sycl=true ([#115](https://github.com/lusoris/vmaf/issues/115)) ([b8b8f50](https://github.com/lusoris/vmaf/commit/b8b8f50b59ea14089591aed833600fd501fd5819))
* **test:** Gate test_speed on enable_float to match speed.c compile guard ([#263](https://github.com/lusoris/vmaf/issues/263)) ([cb1d49c](https://github.com/lusoris/vmaf/commit/cb1d49c63ac15e1ac283739bbab59ab4c13bee15))
* **vulkan:** Move volk -include flag off volk_dep.compile_args (ADR-0200) ([#155](https://github.com/lusoris/vmaf/issues/155)) ([8bc4f65](https://github.com/lusoris/vmaf/commit/8bc4f65be8ec2ee1149da72ac3b016f66f7768ad))
* **vulkan:** Rename volk vk* symbols to vmaf_priv_vk* for static archives (ADR-0198) ([#152](https://github.com/lusoris/vmaf/issues/152)) ([73620ff](https://github.com/lusoris/vmaf/commit/73620ff504f999ad8b5b972f200c82b5fc4475ba))
* **vulkan:** T7-31 — hide volk / vk* symbols from libvmaf.so public ABI ([#132](https://github.com/lusoris/vmaf/issues/132)) ([a1c1a20](https://github.com/lusoris/vmaf/commit/a1c1a2075379107d9e152c8738a4b5add286aafa))


### Performance

* **sycl:** T7-17 — fp64-less device fallback for VIF gain-limiting ([#209](https://github.com/lusoris/vmaf/issues/209)) ([606a3fc](https://github.com/lusoris/vmaf/commit/606a3fc0d8348b2e9f2bd50fae407f05b936d1cb))
* **thread_pool:** Recycle job slots + inline data buffer (ADR-0147) ([#83](https://github.com/lusoris/vmaf/issues/83)) ([8fb2fe1](https://github.com/lusoris/vmaf/commit/8fb2fe17425ae1593d0006ef5b1c186d9dd9047d))


### Refactors

* **ai:** Extract tiny-AI extractor template (cuts 150→30 LOC per new extractor) ([#251](https://github.com/lusoris/vmaf/issues/251)) ([1444bb7](https://github.com/lusoris/vmaf/commit/1444bb7ab50c38a09b7e1c46b75526b946f4f5d5))
* **ai:** Migrate feature_mobilesal + transnet_v2 to tiny_extractor_template.h ([#265](https://github.com/lusoris/vmaf/issues/265)) ([4c59ece](https://github.com/lusoris/vmaf/commit/4c59ece2abdfe2269a7b788db26abed05623af56))
* **cuda:** T-GPU-DEDUP-4 — first consumer of cuda/kernel_template.h (psnr_cuda) ([#269](https://github.com/lusoris/vmaf/issues/269)) ([133dd5f](https://github.com/lusoris/vmaf/commit/133dd5f30f8bda678e5d9fb40497dcbf1bf82994))
* **gpu:** Land per-backend kernel scaffolding templates (CUDA + Vulkan, no migrations) ([#254](https://github.com/lusoris/vmaf/issues/254)) ([4aa75b1](https://github.com/lusoris/vmaf/commit/4aa75b1e486c13dda146398a4273e8e7d6f849a9))
* **gpu:** T-GPU-DEDUP-1 — promote ring_buffer to gpu_picture_pool (ADR-0239) ([#266](https://github.com/lusoris/vmaf/issues/266)) ([19d7eda](https://github.com/lusoris/vmaf/commit/19d7eda2036162e7b595c817135f805981211a88))
* **iqa:** Rename reserved-identifier surface + lint cascade sweep (ADR-0148) ([#84](https://github.com/lusoris/vmaf/issues/84)) ([985be1b](https://github.com/lusoris/vmaf/commit/985be1b9af6a13d32b1159570b816be01824b337))
* **libvmaf:** Sweep readability-function-size NOLINTs (ADR-0146) ([#82](https://github.com/lusoris/vmaf/issues/82)) ([07615a2](https://github.com/lusoris/vmaf/commit/07615a26b86dd6f0168c5d149c78a3341c9acab7))
* **lint:** T7-7 SYCL lint cleanup (162 → 4 findings) ([#114](https://github.com/lusoris/vmaf/issues/114)) ([1f1c742](https://github.com/lusoris/vmaf/commit/1f1c742e191cd9be1f2aa437040de23d023e8687))
* **lint:** Whole-codebase clang-tidy auto-fix subset (52% cleared) ([#113](https://github.com/lusoris/vmaf/issues/113)) ([a87a8f5](https://github.com/lusoris/vmaf/commit/a87a8f51eb1f3c3d82f05843a54a4100052d515e))
* **test:** Extract SIMD bit-exact test harness (cuts 50→20 LOC per new test) ([#252](https://github.com/lusoris/vmaf/issues/252)) ([f970975](https://github.com/lusoris/vmaf/commit/f97097560036be65b4eace28f4163c2cb3808e2c))
* **test:** Tiny-AI registration macro — 4 test files dedup (-286 LOC) ([#268](https://github.com/lusoris/vmaf/issues/268)) ([6e64899](https://github.com/lusoris/vmaf/commit/6e64899e72550d0e100b3e7f3f893109f4f6838d))
* **vulkan:** T-GPU-DEDUP-5 — first consumer of vulkan/kernel_template.h ([#270](https://github.com/lusoris/vmaf/issues/270)) ([14b5ed1](https://github.com/lusoris/vmaf/commit/14b5ed14c86d584a0b013d0d3ddab08e63a8e4c7))
* **vulkan:** T-GPU-DEDUP-6 — moment + ciede consumers of vulkan/kernel_template.h ([#271](https://github.com/lusoris/vmaf/issues/271)) ([eae9e85](https://github.com/lusoris/vmaf/commit/eae9e8587aa97792a8561f1eb337dc0b13a9b0d3))
* **vulkan:** T-GPU-DEDUP-7 — migrate motion + ssim to kernel_template ([#272](https://github.com/lusoris/vmaf/issues/272)) ([69cc940](https://github.com/lusoris/vmaf/commit/69cc940ac6bcc17386509cd9a1eb931c4d2520a9))


### Documentation

* **.claude:** Refresh skill/agent/hook descriptions to current repo state ([#242](https://github.com/lusoris/vmaf/issues/242)) ([9852fcc](https://github.com/lusoris/vmaf/commit/9852fcc37d086a7d88d5d058063f46c3822e0a18))
* Add community hygiene files (SECURITY, CoC, CODEOWNERS, templates) ([c28dd78](https://github.com/lusoris/vmaf/commit/c28dd78eb3e2a68681beca6382d32c70b62e8e42))
* Add per-distro install guides + BENCHMARKS + hygiene files ([fe8c744](https://github.com/lusoris/vmaf/commit/fe8c744d4da174cb453b98c5be0afa4b539e383d))
* Add SYCL backend, SIMD, and GPU documentation ([3a0ee4f](https://github.com/lusoris/vmaf/commit/3a0ee4f915ce89ec1f14c0e9ae00f75f4149683f))
* Add SYCL bundling guide for self-contained deployment ([5fe843f](https://github.com/lusoris/vmaf/commit/5fe843fd03a90114de40f2bc6e5fc4a1a189fa5a))
* **adr:** ADR-0180 — CPU coverage audit closes 5 stale gaps ([#123](https://github.com/lusoris/vmaf/issues/123)) ([fb47f44](https://github.com/lusoris/vmaf/commit/fb47f44ee0b2ff5e7f4bfe66ce7122a8319af6c6))
* **adr:** ADR-0188 — GPU long-tail batch 2 scope (ssim / ms_ssim / psnr_hvs) ([#138](https://github.com/lusoris/vmaf/issues/138)) ([96fbe59](https://github.com/lusoris/vmaf/commit/96fbe5997c29801217e702d0c03dd1c2805126af))
* **adr:** ADR-0192 — GPU long-tail batch 3 scope (every remaining gap) ([#145](https://github.com/lusoris/vmaf/issues/145)) ([57a03db](https://github.com/lusoris/vmaf/commit/57a03db4cd48270c7d64dc34e5d1ad93d0eb445e))
* **adr:** ADR-0205 — cambi GPU feasibility spike (defer integration) ([#159](https://github.com/lusoris/vmaf/issues/159)) ([f48685f](https://github.com/lusoris/vmaf/commit/f48685f3161384ee239e03dfe7aa6a2ca9fa6ec5))
* **adr:** ADR-0207 — tiny-AI Quantization-Aware Training design ([#168](https://github.com/lusoris/vmaf/issues/168)) ([ca56951](https://github.com/lusoris/vmaf/commit/ca569512dd75b1fb2d74074d5179592690c80dcf))
* **adr:** Q2 modernization governance — SSIMULACRA 2, Vulkan, MCP-in-libvmaf, PTQ int8 ([#67](https://github.com/lusoris/vmaf/issues/67)) ([0c9e331](https://github.com/lusoris/vmaf/commit/0c9e3313601645934ccb5b79d50871b48e93f7c5))
* **adr:** Retroactive errata — ULP=0 claims in ADR-0176/0177 were bogus ([#121](https://github.com/lusoris/vmaf/issues/121)) ([4ddb80a](https://github.com/lusoris/vmaf/commit/4ddb80a475f0d859a24338dfa07698fd26020f02))
* **adr:** Supersede 0025/0028/0036 with paraphrased bodies ([#37](https://github.com/lusoris/vmaf/issues/37)) ([4955dde](https://github.com/lusoris/vmaf/commit/4955dde1b99c26339465e01c4d8211fe176007a4))
* **adr:** T-VMAF-TUNE — quality-aware encode automation umbrella spec (ADR-0237) ([#261](https://github.com/lusoris/vmaf/issues/261)) ([f642602](https://github.com/lusoris/vmaf/commit/f642602c65b9b43398ea84278d809336296837e2))
* **agents:** Document libvmaf backend-engagement foot-guns ([#169](https://github.com/lusoris/vmaf/issues/169)) ([96f1ef1](https://github.com/lusoris/vmaf/commit/96f1ef18ea902c219214d4b0b956d1b921f5450d))
* **audit:** T7-4 — quarterly upstream-backlog re-audit (2026-04-29) ([#205](https://github.com/lusoris/vmaf/issues/205)) ([10a71ac](https://github.com/lusoris/vmaf/commit/10a71acc8de6b7ec5e69d6185361c1f30676ff17))
* **backlog:** Land Section-A audit decisions as T-NN cross-links ([#167](https://github.com/lusoris/vmaf/issues/167)) ([58ab35a](https://github.com/lusoris/vmaf/commit/58ab35ad3cbc66e0c6598d0bc27a3f24115f532c))
* **benchmarks:** T7-37 fill TBD cells with measured numbers ([#177](https://github.com/lusoris/vmaf/issues/177)) ([851375a](https://github.com/lusoris/vmaf/commit/851375a172057b3b9cc78fcbb46d44f582ca59c1))
* Consolidate + reorganise documentation tree ([#12](https://github.com/lusoris/vmaf/issues/12)) ([4bbd573](https://github.com/lusoris/vmaf/commit/4bbd573f5d45a9cf6ebfe31757c19c4e0caff703))
* **gpu:** T-GPU-DEDUP-2 — GPU backend public-API pattern doc (ADR-0240) ([#267](https://github.com/lusoris/vmaf/issues/267)) ([394d028](https://github.com/lusoris/vmaf/commit/394d028a172644473d24595d0a16ab4263606ddc))
* **integer_adm:** Verify + defer Netflix[#955](https://github.com/lusoris/vmaf/issues/955) i4_adm_cm rounding overflow (ADR-0155) ([#92](https://github.com/lusoris/vmaf/issues/92)) ([f7e5ecf](https://github.com/lusoris/vmaf/commit/f7e5ecf26238a74ed3148e5021a33b0fbdb9b59d))
* Migrate from Sphinx+Doxygen to MkDocs Material ([#17](https://github.com/lusoris/vmaf/issues/17)) ([40fe6f1](https://github.com/lusoris/vmaf/commit/40fe6f18dad514bb95552d0934b28fb7d0ccfcba))
* **planning:** Refresh AGENTS.md invariants + ADR index sync ([#243](https://github.com/lusoris/vmaf/issues/243)) ([2d891a2](https://github.com/lusoris/vmaf/commit/2d891a2e0ff04ab2d7f99375e83fc8a0b7dee4e2))
* Project-wide doc-substance sweep (ADR-0100 batches 1-4) ([#25](https://github.com/lusoris/vmaf/issues/25)) ([4f3f992](https://github.com/lusoris/vmaf/commit/4f3f992d76483a1c769e57d4e11eca58eaa8aee5))
* **readme:** Rewrite for the Lusoris fork, preserve upstream credit ([e68befa](https://github.com/lusoris/vmaf/commit/e68befa6ffda099026ef9c4ef651f44824a22e33))
* **rebase-notes:** Netflix[#1486](https://github.com/lusoris/vmaf/issues/1486) motion updates verified present (ADR-0158) ([#95](https://github.com/lusoris/vmaf/issues/95)) ([383190a](https://github.com/lusoris/vmaf/commit/383190a481d2170e37a5a4f7619b704e722cba0b))
* **research:** Land Bristol NVC review + 2026-05 CI audit digests ([#240](https://github.com/lusoris/vmaf/issues/240)) ([267b67f](https://github.com/lusoris/vmaf/commit/267b67f92446181e1f2de3e6ce8918b0c284f4d2))
* **research:** Research-0024 — vif/adm upstream-divergence digest ([#182](https://github.com/lusoris/vmaf/issues/182)) ([d8cf89d](https://github.com/lusoris/vmaf/commit/d8cf89d3cf7e4d43558a6bc24d631d7f301f34f7))
* **research:** Research-0025 — FoxBird outlier resolved via KoNViD combined training ([#183](https://github.com/lusoris/vmaf/issues/183)) ([da2de07](https://github.com/lusoris/vmaf/commit/da2de07c93a331b913072ebea7bf6e4250bad87c))
* **research:** Research-0026 — cross-metric feature fusion plan ([#184](https://github.com/lusoris/vmaf/issues/184)) ([d58271c](https://github.com/lusoris/vmaf/commit/d58271c98d9af496b3b6983fd0b21cf496543d40))
* **research:** Research-0027 — Phase-2 feature importance results ([#187](https://github.com/lusoris/vmaf/issues/187)) ([a79f9bf](https://github.com/lusoris/vmaf/commit/a79f9bf20655b761af3283d3e0c5d56f8385a286))
* **research:** T7-9 — Intel AI-PC NPU/EP applicability digest ([#194](https://github.com/lusoris/vmaf/issues/194)) ([e1244aa](https://github.com/lusoris/vmaf/commit/e1244aab686f55e6688c9d8558e6704650a1dc8a))
* **research:** Tiny-AI corpus + architecture survey for next iteration ([#166](https://github.com/lusoris/vmaf/issues/166)) ([91ade5b](https://github.com/lusoris/vmaf/commit/91ade5bc8b02fcd86aec9d68aeb692acb289f545))
* **state:** Refresh post-session-2026-04-29 (verify rows, unblock T6-1a Netflix Public dataset row) ([#245](https://github.com/lusoris/vmaf/issues/245)) ([42b6d35](https://github.com/lusoris/vmaf/commit/42b6d354dc7d59e3cb6d500b12fe371b4df41bd6))
* **tiny-ai:** User docs + README row + LFS + release-please subpackages ([66bd6bd](https://github.com/lusoris/vmaf/commit/66bd6bd1f5cd5e120513b85fa224de6a11cf97d8))
* **top:** Refresh README + supporting top-level docs to current codebase state ([#215](https://github.com/lusoris/vmaf/issues/215)) ([9ab3fdb](https://github.com/lusoris/vmaf/commit/9ab3fdb3cc083c001da0d8ef5f7ae32be0663e1d))
* **usage:** Add NVC-style BD-rate recipe with VMAF (Research-0033 [#4](https://github.com/lusoris/vmaf/issues/4)) ([#258](https://github.com/lusoris/vmaf/issues/258)) ([5b32dad](https://github.com/lusoris/vmaf/commit/5b32dade87c619730a965fac10309e97bb7a6d76))
* **usage:** T7-27 — ffmpeg per-backend copy-paste examples ([#126](https://github.com/lusoris/vmaf/issues/126)) ([f7098a7](https://github.com/lusoris/vmaf/commit/f7098a791b548a0cb465aea65ad5ce9f42db2cd3))
* **user:** Refresh user-facing docs to current codebase state ([#216](https://github.com/lusoris/vmaf/issues/216)) ([a3f3e4f](https://github.com/lusoris/vmaf/commit/a3f3e4f919ad0141910283395428422c0a1b48d7))
* Whole-codebase sweep filling post-T5-1 audit gaps ([#112](https://github.com/lusoris/vmaf/issues/112)) ([0423ebd](https://github.com/lusoris/vmaf/commit/0423ebdc2c45d83a1202a22f68490b07c40b261a))


### Tests

* Add SYCL unit tests, GPU validation scores, and benchmark scripts ([e704022](https://github.com/lusoris/vmaf/commit/e704022fb655bf73d0142b8bf74b9276334d48c6))
* **ssimulacra2:** Snapshot-JSON regression gate (T3-3, ADR-0164) ([#102](https://github.com/lusoris/vmaf/issues/102)) ([eaad393](https://github.com/lusoris/vmaf/commit/eaad393c7b8d0f3520603346321b7a57f312b530))


### Build System

* Add SYCL backend to meson build and fix pkgconfig for static linking ([c66e478](https://github.com/lusoris/vmaf/commit/c66e4780f9adbefee90087f871df656bc6895c4f))
* CUDA 13 + oneAPI 2025.3 + clang-format 22 + black 26 (3/5) ([#7](https://github.com/lusoris/vmaf/issues/7)) ([a7be84c](https://github.com/lusoris/vmaf/commit/a7be84cb5cc6b80659bf2c799aaf62221b335dab))
* Dev infra + scaffolding templates + MCP Docker ([e1482c8](https://github.com/lusoris/vmaf/commit/e1482c84d0a4350f2ede3ffd76a28fc905ad6c11))
* **make:** Add lint/format/sec/sbom/netflix-golden targets ([a077407](https://github.com/lusoris/vmaf/commit/a077407d12b92c9411748012f7703eea41f50366))
* Multi-distro dev setup scripts + extra Dockerfiles ([d1cfc97](https://github.com/lusoris/vmaf/commit/d1cfc972ae56779cedffe1a7de572d14899b39c1))


### CI / Infrastructure

* Add explicit permissions to workflow files ([e6af077](https://github.com/lusoris/vmaf/commit/e6af0772d47837f1477fea51be9714bddedf5fbe))
* Add lint configs, pre-commit, and GitHub Actions gates ([9d72f75](https://github.com/lusoris/vmaf/commit/9d72f750c46fe925c968621fd9a028941a9a8211))
* **ai:** Add DNN-enabled matrix legs (gcc + clang + macOS) ([#56](https://github.com/lusoris/vmaf/issues/56)) ([75ad729](https://github.com/lusoris/vmaf/commit/75ad7298007ca02cac4a726ac04de4b9c332b32c))
* **build:** Add i686 (32-bit x86) build-only matrix row (ADR-0151) ([#87](https://github.com/lusoris/vmaf/issues/87)) ([978f958](https://github.com/lusoris/vmaf/commit/978f95838c5c7e16c617b76b7745027eb182aca2))
* Coverage + assertion-density gates + VPL host upload (replaces [#6](https://github.com/lusoris/vmaf/issues/6)) ([#13](https://github.com/lusoris/vmaf/issues/13)) ([ea7b524](https://github.com/lusoris/vmaf/commit/ea7b52427ecbc8cd401d7214d6ba3d178ba4c354))
* Extract deliverables-check.sh + add make pr-check (ADR-0108 local gate) ([#262](https://github.com/lusoris/vmaf/issues/262)) ([3ffb688](https://github.com/lusoris/vmaf/commit/3ffb68859ea874678ec8a033b7446d183bd9588e))
* Fill lint/CI gaps (nightly, docs, editorconfig, iwyu, gitleaks, codeql) ([75a581b](https://github.com/lusoris/vmaf/commit/75a581b1c5d436411b40eda0448b9f4874a8e940))
* **lint:** Scan push-event delta, not full tree (ADR-0133) ([#71](https://github.com/lusoris/vmaf/issues/71)) ([af65ada](https://github.com/lusoris/vmaf/commit/af65ada191962a0819eb1874919bdd414c62fd1c))
* Release-please automation + dependabot groupings ([8573524](https://github.com/lusoris/vmaf/commit/8573524963b181cb7c5e6bdabf2dd8edcc995468))
* **rule-enforcement:** Automate ADR-0100/0105/0106/0108 checks ([#63](https://github.com/lusoris/vmaf/issues/63)) ([1aa45ec](https://github.com/lusoris/vmaf/commit/1aa45ecd0d009f3585d5954d1891b7d26f58fdc7))
* **rule-enforcement:** Strip markdown emphasis before deliverables grep (ADR-0136) ([#73](https://github.com/lusoris/vmaf/issues/73)) ([95bea61](https://github.com/lusoris/vmaf/commit/95bea6148e165803d3b3c91efa93fc5d126b5c4f))
* Update workflows for SYCL, CUDA, and static builds ([eaad704](https://github.com/lusoris/vmaf/commit/eaad7046281829017b171a0996b9d430c14ab90b))


### Miscellaneous

* **adr:** Adopt ADR-0108 deep-dive deliverables rule + backfill rebase notes ([#39](https://github.com/lusoris/vmaf/issues/39)) ([d60e63a](https://github.com/lusoris/vmaf/commit/d60e63aebb4bf9c1bb399dab74a5e94c0ece6674))
* **adr:** Migrate to Nygard one-file-per-decision + golusoris-alignment sweep ([#24](https://github.com/lusoris/vmaf/issues/24)) ([8e3cd22](https://github.com/lusoris/vmaf/commit/8e3cd22c1240ce9f11bb01f8bfd95a2230a598b1))
* **backlog:** T7-32 — 3 micro-investigations bundled (motion_v2 srlv64 + tiny-vmaf-v2 identity + routine.py FIXME) ([#198](https://github.com/lusoris/vmaf/issues/198)) ([8e0eb8f](https://github.com/lusoris/vmaf/commit/8e0eb8f7cf9b1f88b4a73b089f113bb3d1ef24ad))
* **ci:** Cache Netflix vmaf_resource fixtures (actions/cache@v5) ([#131](https://github.com/lusoris/vmaf/issues/131)) ([f1399c7](https://github.com/lusoris/vmaf/commit/f1399c7bcc888703d8f5587eba48c00840da8181))
* **ci:** Finish Node 24 bump — scorecard artifact SHA + nightly-bisect setup-python ([#49](https://github.com/lusoris/vmaf/issues/49)) ([3eb9af7](https://github.com/lusoris/vmaf/commit/3eb9af7ea2a568aab88ae619db49a2e2a1b2a3cf))
* **ci:** Rename workflows + Title Case display names (ADR-0116) ([#53](https://github.com/lusoris/vmaf/issues/53)) ([f4379c8](https://github.com/lusoris/vmaf/commit/f4379c870750400b643c38644261fd361e2f59fd))
* **ci:** Revert pip&lt;26.1 pin in Tiny AI workflow (PR [#231](https://github.com/lusoris/vmaf/issues/231)) ([#233](https://github.com/lusoris/vmaf/issues/233)) ([ca8c964](https://github.com/lusoris/vmaf/commit/ca8c96483706944f41fac89f80d88dba87b46853))
* **ci:** T7-CI-DEDUP — drop redundant python-lint + shellcheck, demote docker-image to advisory ([#257](https://github.com/lusoris/vmaf/issues/257)) ([5bf941d](https://github.com/lusoris/vmaf/commit/5bf941d34264d9041716b2a8bec342b177308de8))
* **dnn:** T7-12 — remove VMAF_MAX_MODEL_BYTES env override ([#193](https://github.com/lusoris/vmaf/issues/193)) ([f87384f](https://github.com/lusoris/vmaf/commit/f87384fc0257dacdc8bfd832bd00c280bd27cc2e))
* **docs:** Audit of untracked follow-up items (2026-04-28) ([#161](https://github.com/lusoris/vmaf/issues/161)) ([3a6e598](https://github.com/lusoris/vmaf/commit/3a6e5982be35d9dec94a862c71c6f89130f0a084))
* **docs:** Clear Section C stale comments (audit-2026-04-28) ([#164](https://github.com/lusoris/vmaf/issues/164)) ([91b4f75](https://github.com/lusoris/vmaf/commit/91b4f75bad706ca63d4d370d8012819480d86546))
* **ffmpeg-patches:** Refresh against FFmpeg n8.1 + clarify SSIMULACRA 2 is patchless ([#101](https://github.com/lusoris/vmaf/issues/101)) ([439282e](https://github.com/lusoris/vmaf/commit/439282ea5691433bd3a9142392981fbde079ee64))
* **license:** Bump Netflix copyright from 2016-2020 to 2016-2026 ([c159761](https://github.com/lusoris/vmaf/commit/c159761dbe8bc9ee6d459e149dc7000ea50760ef))
* **license:** Correct fork copyright year to 2026 ([0e98c94](https://github.com/lusoris/vmaf/commit/0e98c949e2598d8d05c40a75b88f42d8b6d5c063))
* **license:** Re-attribute SYCL files to Lusoris and Claude ([a185f8e](https://github.com/lusoris/vmaf/commit/a185f8ef52d0e166dabab0a03588e111c126f2a2))
* **lint:** Clang-tidy upstream cleanup rounds 2-4 ([#2](https://github.com/lusoris/vmaf/issues/2)) ([722d21f](https://github.com/lusoris/vmaf/commit/722d21fd4e3c106abc411d584b4a84ce306d758e))
* Post-merge cleanup — CI fix + lint + supply-chain + scorecard + dependabot ([#14](https://github.com/lusoris/vmaf/issues/14)) ([798db39](https://github.com/lusoris/vmaf/commit/798db3941dea7757b764287f5fda784064430a96))
* **release:** Introduce CHANGELOG + ADR-index fragment files (drop merge-conflict pain) ([#253](https://github.com/lusoris/vmaf/issues/253)) ([1254a5a](https://github.com/lusoris/vmaf/commit/1254a5ac7e08751acc31dd1f8bca0b551fe48df9))
* **repo:** Add AI-agent scaffolding and engineering-principles docs ([b799db5](https://github.com/lusoris/vmaf/commit/b799db5be6380cbde3c2e2d215610f9d0f2024a1))
* **skills:** Sync-upstream detects port-only topology ([#75](https://github.com/lusoris/vmaf/issues/75)) ([40b97cd](https://github.com/lusoris/vmaf/commit/40b97cdf98264a798d4bc1320b08bb9034ad5f76))
* **sycl:** T7-13 — toolchain cleanup (oneAPI multi-version recipe + icpx clang-tidy wrapper) ([#206](https://github.com/lusoris/vmaf/issues/206)) ([cd806e3](https://github.com/lusoris/vmaf/commit/cd806e3d9899e6b30940c0e21bed15d2690ac3d5))
* **upstream:** Port 798409e3 + 314db130 — CUDA null-deref + remove all.c ([#181](https://github.com/lusoris/vmaf/issues/181)) ([6eab09c](https://github.com/lusoris/vmaf/commit/6eab09c05c454db4c0a0f585379d1d4fb21f124e))
* **upstream:** Record ours-merge of Netflix 966be8d5 (already ported in d06dd6cf) ([fddc5ca](https://github.com/lusoris/vmaf/commit/fddc5ca7cbc0f406d0269c7c5ff98e0487d819b8))
* **upstream:** Record ours-merge of Netflix 966be8d5 (bookkeeping) ([27ce439](https://github.com/lusoris/vmaf/commit/27ce43910d5f6c14ec04342966bbc0204e5b2958))
* **vscode:** Clangd-first workspace settings + debug launch configs ([04d18f1](https://github.com/lusoris/vmaf/commit/04d18f1a5234c994d4ed76f0e521e916098e1994))


### Ports from upstream Netflix/vmaf

* Cambi effective_eotf (Netflix [#2](https://github.com/lusoris/vmaf/issues/2)c9bb74e) ([#160](https://github.com/lusoris/vmaf/issues/160)) ([79288e8](https://github.com/lusoris/vmaf/commit/79288e8d76b457814982135511be0682d7c1d05f))
* **common:** Generalized AVX convolve from Netflix upstream f3a628b4 (ADR-0143) ([#79](https://github.com/lusoris/vmaf/issues/79)) ([025a754](https://github.com/lusoris/vmaf/commit/025a754668e7ef2fde42e227b9afcf87a8c33925))
* **cuda:** Enable CUDA feature extraction on Windows MSYS2/MinGW (ADR-0150) ([#86](https://github.com/lusoris/vmaf/issues/86)) ([f9d1cae](https://github.com/lusoris/vmaf/commit/f9d1cae22a30860eebd1ebbf3df676dce9af257d))
* **libvmaf:** Thread-local locale handling from Netflix/vmaf[#1430](https://github.com/lusoris/vmaf/issues/1430) (ADR-0137) ([#74](https://github.com/lusoris/vmaf/issues/74)) ([e0e78db](https://github.com/lusoris/vmaf/commit/e0e78db3e39ab3d10d923ad89d5a1d94f37b8acb))
* **python:** Netflix[#1376](https://github.com/lusoris/vmaf/issues/1376) FIFO-hang fix via multiprocessing.Semaphore (ADR-0149) ([#85](https://github.com/lusoris/vmaf/issues/85)) ([e5a52e7](https://github.com/lusoris/vmaf/commit/e5a52e74f9768ff08df638b13ef424244c68d6fe))
* **vif:** Vif_sigma_nsq parameter from Netflix upstream 18e8f1c5 (ADR-0142) ([#78](https://github.com/lusoris/vmaf/issues/78)) ([d241758](https://github.com/lusoris/vmaf/commit/d2417584f482d056d8d0fc5ace540c897b36057e))


### SYCL Backend

* Implement oneAPI/SYCL compute backend for ADM, VIF, and Motion ([e899082](https://github.com/lusoris/vmaf/commit/e8990823421d617fa73fbe1e583156f0e078c6f8))


### CUDA Backend

* Double-buffer GPU frames and optimize kernels ([ea9a9d4](https://github.com/lusoris/vmaf/commit/ea9a9d420beef8ae72488f6a76d8de61e9767c7a))
* Eliminate ADM decouple intermediate buffers  Inline the decouple computation into the CSF and CM kernels, eliminating 6 intermediate buffers (decouple_r, decouple_a, csf_a for both int16 scale-0 and int32 scales 1-3 paths).  CSF kernels now read ref/dis DWT2 directly from buffer, compute decouple inline, and write only csf_f. CM kernels inline both decouple_r and csf_a computation per-pixel, reading only csf_f from buffer (needed for 3x3 stencil).  Shared device functions extracted to adm_decouple_inline.cuh. The adm_decouple.cu source is no longer compiled.  Reduces GPU memory allocation from 30 to ~16.5 buf_sz_one units (~107 MB savings at 4K resolution). ([787e338](https://github.com/lusoris/vmaf/commit/787e33822d94429e3fc36d4c2a32a304185fe9d6))
* VIF rd_stride optimization - reduce read buffer allocation  Add rd_stride field to VifBufferCuda for texture-aligned read buffer access. Compute rd_stride with tex_alignment padding, reducing allocation from full frame_size to rd_size (rd_stride * ceil(h/2)). Update filter1d.cu kernels to use rd_stride for rd buffer indexing. ([0d2a196](https://github.com/lusoris/vmaf/commit/0d2a196715787b2f1e8a482bfc141d5951b61493))


### SIMD

* Add ARM NEON implementations for feature extractors ([e020421](https://github.com/lusoris/vmaf/commit/e020421442bd1292c0f66a9dd14e7e67ecc343d9))
* Add x86 AVX2/AVX-512 implementations for feature extractors ([81fcd42](https://github.com/lusoris/vmaf/commit/81fcd42e0ff5bc9267cc2db673cea4b0952fe9ae))
* Use double-precision accumulation in float ADM reductions  Float ADM sum_cube and csf_den_scale functions accumulated cubed values in float32, causing ~8e-5 drift between scalar and SIMD paths due to different accumulation order. Fix by computing val^3 in float SIMD, converting to double via _mm256_cvtps_pd / _mm512_cvtps_pd before accumulating, and using double for outer accumulators in all paths (scalar, AVX2, AVX512). Update test expectations accordingly. ([24c88a3](https://github.com/lusoris/vmaf/commit/24c88a32b9b85071defedbf6140bcd14e8c532cd))

## [Unreleased] — lusoris fork (3.0.0-lusoris.0)

### Added

- **Vulkan VmafPicture preallocation surface (T-VULKAN-PREALLOC /
  ADR-0238).** Closes the API parity gap with CUDA / SYCL. New
  public entry points `vmaf_vulkan_preallocate_pictures` +
  `vmaf_vulkan_picture_fetch`; new enum
  `VmafVulkanPicturePreallocationMethod` (`NONE` / `HOST` /
  `DEVICE`); new struct `VmafVulkanPictureConfiguration`. Mirrors
  the SYCL surface — `HOST` uses `vmaf_picture_alloc`; `DEVICE`
  backs each picture's luma plane with a host-visible Vulkan
  buffer (VMA `AUTO_PREFER_HOST`) so callers write directly into
  the memory the kernel descriptor sets read. Pool depth is fixed
  at the canonical `frames-in-flight = 2` (matches SYCL); fetch
  falls back to a host-backed picture if the caller skipped
  preallocation. New `VMAF_PICTURE_BUFFER_TYPE_VULKAN_DEVICE`
  picture buffer-type tag. Six smoke tests in
  `libvmaf/test/test_vulkan_pic_preallocation.c` pin the contract
  under ASan/UBSan. FFmpeg-side adoption (per CLAUDE.md §12 r14)
  is a deliberate follow-up — no in-tree patch consumes the
  preallocation surface today, matching the SYCL precedent. See
  [`docs/api/gpu.md`](docs/api/gpu.md) and
  [ADR-0238](docs/adr/0238-vulkan-picture-preallocation.md).
- **`VmafVulkanConfiguration.max_outstanding_frames` knob (ADR-0235
  follow-up #3, T7-29 part 4).** The v2 async pending-fence ring
  depth is now caller-tunable via the public Vulkan configuration
  struct. `0` selects the canonical default (4); values are clamped
  to `[1, VMAF_VULKAN_RING_MAX]` (currently 8) internally. The
  clamped result is observable via the new
  `vmaf_vulkan_state_max_outstanding_frames()` accessor, so callers
  that pass a value can log what they actually got. External-handles
  callers (`vmaf_vulkan_state_init_external`) still receive the
  default — extending `VmafVulkanExternalHandles` is a separate ABI
  bump. Contract pinned in
  `libvmaf/test/test_vulkan_async_pending_fence.c`
  (`test_ring_size_*` group, 4 cases). See
  [`docs/api/gpu.md`](docs/api/gpu.md) and
  [ADR-0235](docs/adr/0235-vulkan-async-pending-fence.md).
- **`tools/vmaf-tune/` quality-aware encode automation spec
  (T-VMAF-TUNE / ADR-0237 — Proposed).** Umbrella spec for a new
  fork-local automation surface that closes the loop between the
  fork's quality stack and FFmpeg's encoders. The tool drives FFmpeg
  with parameter grids, captures bitrate + per-metric quality, and
  recommends encoder parameters per source × codec × target. Six-phase
  roadmap (A: harness MVP — `libx264` only, ~1 week; B: target-VMAF
  bisect; C: per-title CRF predictor; D: per-shot dynamic CRF gated
  on T6-3b; E: Pareto ABR ladder; F: MCP tools), each phase
  standalone-shippable. Multi-codec from day one via a codec-adapter
  interface — `libx264` / `libx265` / `libsvtav1` / `libvpx-vp9` /
  `libvvenc` / LCEVC / EVC / AVS3 / neural-codec adapters in an
  opt-in extra. Closes the loop on the codec-aware FR regressor
  (ADR-0235 codec collision, currently BLOCKED on corpus) and the
  per-shot CRF predictor (T6-3b). **Spec only — no code in this PR.**
  Companion research digest:
  [`docs/research/0044-quality-aware-encode-automation.md`](docs/research/0044-quality-aware-encode-automation.md).
- **Codec-aware FR regressor surface (T7-CODEC-AWARE / ADR-0235).**
  New `ai/src/vmaf_train/codec.py` ships a closed, order-stable
  6-bucket codec vocabulary (`x264`, `x265`, `libsvtav1`,
  `libvvenc`, `libvpx-vp9`, `unknown`) with `codec_index` + alias
  table (`h264` → `x264`, `hevc` → `x265`, `av1` → `libsvtav1`,
  `vp9` → `libvpx-vp9`, `vvc` / `h266` → `libvvenc`) and one-hot
  helpers. `FRRegressor` gains an optional `num_codecs` constructor
  arg that concatenates the one-hot codec id to the
  `FULL_FEATURES` input vector before the first MLP layer; the
  default `num_codecs=0` keeps the v1 single-input contract so
  existing checkpoints load unchanged. Feature-dump scripts
  (`ai/scripts/bvi_dvc_to_full_features.py`,
  `ai/scripts/extract_full_features.py`) now emit a `codec` column
  in their per-clip parquet — BVI-DVC defaults to `"x264"` (the
  internal libx264 encode), Netflix Public defaults to `"unknown"`
  (pre-encoded distortions, no in-band metadata). New
  `docs/adr/0235-codec-aware-fr-regressor.md`,
  `docs/research/0040-codec-aware-fr-conditioning.md`, and
  `docs/ai/models/fr_regressor_v2_codec_aware.md`. **Training run
  + PLCC delta measurement is BLOCKED** in this PR (the cached
  Netflix Public training corpus is not reachable from the
  authoring sandbox); the follow-up PR re-runs the trainer +
  ships `model/tiny/fr_regressor_v2_codec_aware.onnx` only if the
  empirical PLCC lift exceeds the 0.005 bar. Cites the 2026
  Bristol VI-Lab review §5.3 + Bampis 2018 (ST-VMAF) + Zhang 2021
  (Bull lab "Enhancing VMAF").

### Changed

- **`psnr_vulkan.c` migrated to `vulkan/kernel_template.h` (T-GPU-DEDUP-5,
  first consumer).** The dormant `vulkan/kernel_template.h` (410 LOC,
  ADR-0221) shipped with zero consumers; its docstring designated
  `psnr_vulkan.c` as the reference implementation. This commit lands
  that migration. The 5 long-lived pipeline objects (DSL, pipeline
  layout, shader module, compute pipeline, descriptor pool) collapse
  to one `VmafVulkanKernelPipeline pl` bundle. `create_pipeline()`
  shrinks ~104 LOC → ~30 LOC via
  `vmaf_vulkan_kernel_pipeline_create()`. `close_fex()`'s
  `vkDeviceWaitIdle` + 5×`vkDestroy*` sweep collapses to one
  `vmaf_vulkan_kernel_pipeline_destroy()`. Bit-exactness preserved —
  spec-constants, push-constant struct, shader bytecode, dispatch
  grid math, and host-side reduction are byte-identical to the prior
  implementation. **Net −55 LOC.** Pairs with PR #269 (first CUDA
  template consumer) — both dormant kernel templates now have real
  consumers.

- **`moment_vulkan.c` + `ciede_vulkan.c` migrated to
  `vulkan/kernel_template.h` (T-GPU-DEDUP-6).** Second + third
  consumers of the dormant Vulkan kernel template after PR #270
  (psnr_vulkan, the first consumer). Both files follow the
  identical migration pattern: 5 individual pipeline-object fields
  (`dsl`, `pipeline_layout`, `shader`, `pipeline`, `desc_pool`)
  collapse to one `VmafVulkanKernelPipeline pl`; ~100 LOC of
  `create_pipeline()` body collapses to a single
  `vmaf_vulkan_kernel_pipeline_create()` call;
  `close_fex()`'s `vkDeviceWaitIdle` + 5×`vkDestroy*` sweep
  collapses to one `vmaf_vulkan_kernel_pipeline_destroy()`. Net
  **−119 LOC** (moment −60, ciede −59). Bit-exactness preserved —
  spec-constants, push-constant structs, shader bytecodes,
  dispatch grid math, and host-side reductions are byte-identical.
  `motion_vulkan.c` deferred (uses two pipelines sharing a layout;
  needs a multi-pipeline template extension).
- **Tiny-AI test registration macro (`tiny_ai_test_template.h`).**
  The four tiny-AI extractor test files (`test_lpips.c`,
  `test_mobilesal.c`, `test_transnet_v2.c`, `test_fastdvdnet_pre.c`)
  shipped near-identical ~140 LOC of registration smoke tests each.
  New `libvmaf/test/tiny_ai_test_template.h` emits the four standard
  tests via `VMAF_TINY_AI_DEFINE_REGISTRATION_TESTS(ext, feat, env,
  prefix)`; each per-extractor test file now lands in 20-50 LOC.
  Behavior bit-exact preserved (same assertions, same env-var
  save/restore dance, same `_putenv_s` shim for MSVCRT). TransNet V2
  keeps its two extractor-specific extra tests (binary-flag
  round-trip + `provided_features` NULL-termination check). Net
  −286 LOC. Top-leverage non-GPU dedup finding from the
  2026-05-02 whole-codebase audit.
- **Vulkan kernel template — multi-pipeline support + ssim/motion
  migration (T-GPU-DEDUP-7).** Extended
  [`libvmaf/src/vulkan/kernel_template.h`](libvmaf/src/vulkan/kernel_template.h)
  with `vmaf_vulkan_kernel_pipeline_add_variant()` so kernels that
  need multiple `VkPipeline`s sharing one layout / shader / DSL /
  descriptor pool (different spec-constant) can express that
  without re-implementing the boilerplate. Migrated
  [`libvmaf/src/feature/vulkan/motion_vulkan.c`](libvmaf/src/feature/vulkan/motion_vulkan.c)
  and
  [`libvmaf/src/feature/vulkan/ssim_vulkan.c`](libvmaf/src/feature/vulkan/ssim_vulkan.c)
  to the template — net duplicated DSL / pipeline layout / shader /
  pipeline / descriptor-pool create-and-destroy code removed. As
  part of motion's migration, the dormant `pipelines[2]` array
  (kept "for SYCL parity" but functionally identical because
  COMPUTE_SAD is passed via push constants, not spec-constants) was
  collapsed to a single `VkPipeline`. SSIM still owns two pipelines
  (horizontal pass=0 / vertical pass=1 differ in spec constant) —
  pass=0 is the template's base pipeline; pass=1 is attached via
  `_add_variant()`. Validated against the Netflix-pair smoke
  (`float_ssim` mean 0.863, 48 frames) + `meson test
  test_vulkan_smoke test_vulkan_async_pending_fence
  test_vulkan_pic_preallocation` (all green).

- **`integer_psnr_cuda.c` migrated to `cuda/kernel_template.h`
  (T-GPU-DEDUP-4, first consumer).** The dormant
  `cuda/kernel_template.h` (296 LOC, ADR-0221) shipped with zero
  consumers; its docstring designated `integer_psnr_cuda.c` as the
  reference implementation. This commit lands the migration: the
  per-frame async lifecycle (private stream + submit/finished
  event pair) and the device + pinned-host readback pair now come
  from the template helpers
  (`vmaf_cuda_kernel_lifecycle_init/_close`,
  `vmaf_cuda_kernel_readback_alloc/_free`,
  `vmaf_cuda_kernel_submit_pre_launch`,
  `vmaf_cuda_kernel_collect_wait`). `PsnrStateCuda` shrinks
  (3 fields → 1 `VmafCudaKernelLifecycle`; 2 fields → 1
  `VmafCudaKernelReadback`). Bit-exactness preserved — the
  kernel launch, the per-bpc function lookup, the SSE accumulator
  math, and the host-side `log10` score formula are byte-identical
  to the prior implementation. Single-consumer net LOC delta is
  +8 (helpers add per-call boilerplate); the dedup win materialises
  as more CUDA feature kernels migrate one-at-a-time in follow-ups.
- **`feature_mobilesal.c` + `transnet_v2.c` migrated to `tiny_extractor_template.h`.**
  PR #251 shipped the shared template (`vmaf_tiny_ai_resolve_model_path`,
  `vmaf_tiny_ai_open_session`, `vmaf_tiny_ai_yuv8_to_rgb8_planes`,
  `VMAF_TINY_AI_MODEL_PATH_OPTION`) but left the existing
  `feature_mobilesal.c` + `transnet_v2.c` open-coding the same
  boilerplate. Both files now consume the template, eliminating
  ~98 LOC of duplicated model-path lookup / session-open log /
  YUV→RGB kernel / VmafOption row. Behavior bit-exact preserved
  (the template hoists the literal copies the migrated files
  carried). `feature_lpips.c` and `fastdvdnet_pre.c` were already
  migrated. Closes the AI-template adoption gap noted in the
  2026-05-02 dedup audit. See `dnn/tiny_extractor_template.h`.
- **GPU picture pool dedup — `cuda/ring_buffer.{c,h}` →
  `gpu_picture_pool.{c,h}` (ADR-0239).** The CUDA picture-pool
  primitive is promoted out of `libvmaf/src/cuda/` into the
  backend-agnostic `libvmaf/src/gpu_picture_pool.{c,h}` (Netflix's
  `VmafRingBuffer` was always callback-based and shape-agnostic;
  only the directory and symbol names implied otherwise). Symbols
  rename: `VmafRingBuffer` → `VmafGpuPicturePool`,
  `vmaf_ring_buffer_*` → `vmaf_gpu_picture_pool_*`. SYCL's
  `vmaf_sycl_picture_pool_*` keeps its public-internal API but
  now delegates to the generic pool — the SYCL wrapper just owns
  the `VmafSyclCookie` storage; `std::mutex` drops out. Vulkan's
  `picture_vulkan_pool.c` (added in PR #264) rewrites as a thin
  wrapper around the generic pool with the same pattern. Net
  structural win: ONE round-robin / mutex / unwind implementation
  across all three GPU backends. The `Netflix#1300`
  mutex-destroy-order fix (ADR-0157) travels with the file. Test
  renamed `test_ring_buffer.c` → `test_gpu_picture_pool.c`.

- **GPU backend public-API pattern doc (ADR-0240).** New
  [`docs/development/gpu-backend-template.md`](docs/development/gpu-backend-template.md)
  ships the recipe new GPU backends follow — shared lifecycle
  (`vmaf_<backend>_state_init` / `_import_state` / `_state_free`),
  optional sections (`_available` / `_list_devices` / picture
  preallocation / hwaccel zero-copy import), Doxygen + ABI
  conventions, and the SYCL/Vulkan `NONE / HOST / DEVICE`
  three-method picture-preallocation convention as the
  new-backend default. New
  [`libvmaf/include/libvmaf/AGENTS.md`](libvmaf/include/libvmaf/AGENTS.md)
  pins the public-headers-tree invariant (rebase ordering of the
  four backend headers, ABI-additive rule). PR3 of the GPU dedup
  sequence — doc-only, not codegen, after a 2026-05-02 audit
  measured the four headers at ~20 of ~200 lines truly shared
  (state lifecycle); the rest is genuinely backend-specific
  feature surface. Mirrors the tiny-AI ADR-0221 "recipe doc +
  shared helpers, not codegen" precedent.
- **ADR-0108 deliverables gate now runnable locally (`make pr-check`).**
  The Deep-Dive Deliverables Checklist gate
  (`.github/workflows/rule-enforcement.yml`) previously inlined ~80
  lines of bash that parsed PR bodies + verified ticked file
  references against the PR diff. The check is fundamentally a
  PR-body-vs-diff coherence test, which pre-commit hooks cannot run
  (neither artefact exists at commit time). The bash now lives in
  [`scripts/ci/deliverables-check.sh`](scripts/ci/deliverables-check.sh)
  as the single source of truth, and the workflow calls the script
  in one step. New `make pr-check PR=<num>` (or `make pr-check
  BODY=<file>`) target runs the same gate locally before
  `gh pr create` — saves the typical 60-second CI round-trip when
  a checkbox is ticked instead of opted out (the bug that hit
  PR #260's first run).
- **Top-level docs refresh (post-session-2026-04-29).** `README.md`,
  `CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`, and `docs/principles.md`
  refreshed to current codebase reality: Vulkan backend (T5-1 / T7-36),
  embedded MCP scaffold (T5-2), GPU-parity CI gate (T6-8 / ADR-0214),
  KoNViD-1k tiny-AI corpus, fp64-less SYCL fallback (T7-17), `--tiny-*`
  CLI flags, `enable_mcp` Meson option, hooks-install quickstart step,
  and CLAUDE §12 r12-r14 quality-gate rules (touched-file lint-clean,
  state.md updates, ffmpeg-patches sync). No code or schema changes.
- **User-facing docs refresh (post-session 2026-04-29).** Refreshed
  [`docs/metrics/features.md`](docs/metrics/features.md) (de-duplicated
  the extractor overview table; CAMBI Vulkan, integer-SSIM Vulkan,
  SSIMULACRA 2 CUDA / SYCL twins, PSNR Vulkan chroma WIP all now
  reflected with footnotes) and [`docs/api/gpu.md`](docs/api/gpu.md)
  (Vulkan section flipped from "scaffold only" to "T5-1c full
  default-model coverage"; added pointers to T7-29 image-import API
  and T7-10 HIP scaffold). No code changes; doc-only.
- **AI-tooling docs refresh (`.claude/`).** Skill descriptions and
  one agent description in `.claude/skills/*/SKILL.md` and
  `.claude/agents/vulkan-reviewer.md` have been refreshed to
  match the current state of the fork: `add-gpu-backend` now
  cites the Vulkan T5-1 scaffold ([ADR-0175](docs/adr/0175-vulkan-backend-scaffold.md))
  and image-import ([ADR-0186](docs/adr/0186-vulkan-image-import-impl.md))
  as the canonical recent precedent; `cross-backend-diff` and
  `validate-scores` mention the T6-8 GPU-parity CI gate
  ([ADR-0214](docs/adr/0214-gpu-parity-ci-gate.md)) as the
  contract their local-dev runs mirror, and add `vulkan` to the
  default backend list; `port-upstream-commit` extends the
  SIMD/GPU twin list with the Vulkan kernel + GLSL shader
  surfaces that any upstream port must propagate to. The
  `vulkan-reviewer` agent description and `## Status` block drop
  the stale "forward-looking — backend does not yet exist"
  framing now that `libvmaf/src/vulkan/` and
  `libvmaf/src/feature/vulkan/` are live across all features.
  Root `AGENTS.md` §7 skill table extends to cover the
  AI-tooling skills that existed but were not listed
  (`build-ffmpeg-with-vmaf`, `refresh-ffmpeg-patches`,
  `validate-scores`, `run-netflix-bench`, `bisect-model-quality`,
  `regen-docs`, `format-all`, `lint-all`, the four `dev-llm-*`).
  Hook header comments in `.claude/hooks/*.sh` were audited
  against current behavior and required no edits. No functional
  change to any hook script body. No user-discoverable surface
  changed.
- **Planning + AGENTS.md refresh** (fork-local doc): root
  `AGENTS.md` gains a new §13 "Rebase-sensitive invariants
  (project-wide)" indexing the shipped surfaces from the
  2026-04-29 session — GPU long-tail terminus
  ([ADR-0210](docs/adr/0210-cambi-vulkan-integration.md) closes
  the matrix), Vulkan backend + image import, ssim/ms_ssim
  Vulkan, motion_v2 GPU port, MCP scaffold (ADR-0209), HIP
  scaffold (ADR-0212 placeholder), SVE2 SIMD (ADR-0213
  placeholder), GPU-parity CI gate
  ([ADR-0214](docs/adr/0214-gpu-parity-ci-gate.md)), MobileSal
  (ADR-0218 placeholder), TransNet V2, FastDVDnet (ADR-0215
  placeholder), psnr chroma Vulkan (ADR-0216 placeholder), SYCL
  fp64-less device contract
  ([ADR-0220](docs/adr/0220-sycl-fp64-fallback.md)), model
  registry + Sigstore (ADR-0211 placeholder), and the upstream
  `feature/motion` port from `b949cebf`
  ([PR #197](https://github.com/lusoris/vmaf/pull/197)) +
  `feature/speed` port from `d3647c73`
  ([PR #213](https://github.com/lusoris/vmaf/pull/213)).
  [`libvmaf/AGENTS.md`](libvmaf/AGENTS.md),
  [`libvmaf/src/feature/AGENTS.md`](libvmaf/src/feature/AGENTS.md),
  [`libvmaf/src/sycl/AGENTS.md`](libvmaf/src/sycl/AGENTS.md),
  [`libvmaf/src/dnn/AGENTS.md`](libvmaf/src/dnn/AGENTS.md), and
  the new
  [`libvmaf/src/vulkan/AGENTS.md`](libvmaf/src/vulkan/AGENTS.md)
  pick up the per-subdirectory invariants. No code change.

- **`docs/state.md` post-2026-04-29-session refresh.** Updated
  `_Updated:` stamp to 2026-04-29; rewrote the "Tiny-AI C1 baseline
  `fr_regressor_v1.onnx`" deferral row in the "Deferred (waiting on
  external dataset access)" section to mark its reopen-trigger as
  TRIGGERED — the Netflix Public training corpus that gated C1 is
  now locally available at `.workingdir2/netflix/` (9 ref + 70 dis
  YUVs, ~37 GB, gitignored; provided by lawrence 2026-04-27),
  unblocking BACKLOG T6-1a. Other rows verified accurate vs the
  2026-04-29-session merged PR set (#193–#205, #209) — every PR was
  feature / chore / docs / perf with no bug-status delta to record
  per CLAUDE §12 rule 13. No code changes.
- **SIMD bit-exact test harness (ADR-0221).** New
  [`libvmaf/test/simd_bitexact_test.h`](libvmaf/test/simd_bitexact_test.h)
  centralises the per-test SIMD-parity scaffolding: `xorshift32` PRNG,
  portable POSIX/MinGW/MSVC aligned allocator, x86 AVX2 CPUID gate,
  and `SIMD_BITEXACT_ASSERT_MEMCMP` / `SIMD_BITEXACT_ASSERT_RELATIVE`
  assertion macros. `test_psnr_hvs_avx2.c`, `test_psnr_hvs_neon.c`,
  `test_moment_simd.c`, and `test_motion_v2_simd.c` migrate to the
  harness as proof — net `-106` LOC across the four files. New SIMD
  parity tests now cost ~20 LOC of test-body code instead of ~50–100
  LOC of scaffolding + body. `test_ssimulacra2_simd.c` is intentionally
  unchanged (its `fill_random` FP rounding order is load-bearing for
  input bit patterns; a separate dedup PR with snapshot rerun under
  `/cross-backend-diff` can migrate it). All 41 `meson test` cases
  pass post-refactor; clang-format clean. New
  [`libvmaf/test/AGENTS.md`](libvmaf/test/AGENTS.md) "New SIMD parity
  test" rebase-sensitive invariant row pins the include-order rule
  (`#include "test.h"` MUST precede `#include "simd_bitexact_test.h"`
  because `test.h` lacks a header guard). See
  [ADR-0221](docs/adr/0221-simd-bitexact-test-harness.md).
- **`nr_metric_v1` flips to dynamic-PTQ int8 (T5-3d / ADR-0221).**
  The C2 NR-MOS tiny model joins the dynamic-PTQ family alongside
  `learned_filter_v1` (ADR-0174). Root cause of the previous
  `quantize_dynamic` failure (`Inferred shape and existing shape
  differ in dimension 0: (128) vs (1)`) traced to
  `torch.onnx.export` duplicating every initialiser into
  `graph.value_info` with static-shape annotations that do not
  survive the dynamic-batch axis substitution — same class of bug
  fixed for `vmaf_tiny_v1*.onnx` in PR #174 (T5-3e). Ported the
  `_save_inlined` strip pattern into both
  [`ai/src/vmaf_train/models/exports.py`](ai/src/vmaf_train/models/exports.py)
  (so future fork-trained tiny models are PTQ-clean by
  construction) and
  [`ai/scripts/ptq_dynamic.py`](ai/scripts/ptq_dynamic.py) (so
  pre-existing on-disk ONNX files quantise without re-export).
  Re-saved `model/tiny/nr_metric_v1.onnx` with `value_info`
  duplicates stripped — ORT-CPU produces bit-identical inference
  before vs after on a deterministic 16-sample input set; the
  registry sha256 rolls forward `60c2bd59…` → `75eff676…`. Registry
  entry now carries `quant_mode: "dynamic"`, `int8_sha256:
  "e5ba2086…"`, and `quant_accuracy_budget_plcc: 0.01`.
  `ai/scripts/measure_quant_drop.py` reports `[PASS]
  PLCC=0.992326 drop=0.007674 budget=0.0100`. 2.0× size shrink
  (119 KB → 58 KB). See
  [ADR-0221](docs/adr/0221-nr-metric-v1-ptq.md).
- **Tiny-AI extractor template — shared scaffolding header (ADR-0221).**
  New
  [`libvmaf/src/dnn/tiny_extractor_template.h`](libvmaf/src/dnn/tiny_extractor_template.h)
  ships three `static inline` helpers
  (`vmaf_tiny_ai_resolve_model_path`, `vmaf_tiny_ai_open_session`,
  `vmaf_tiny_ai_yuv8_to_rgb8_planes`) plus one struct-literal-emitting
  macro (`VMAF_TINY_AI_MODEL_PATH_OPTION`) that deduplicate the
  model-path-option-then-env-var resolution + session-open
  log-line pattern + BT.709 limited-range YUV→RGB chroma upsample +
  `model_path` `VmafOption[]` row across the four tiny-AI extractors
  (`feature_lpips.c`, `fastdvdnet_pre.c`, in-flight
  `feature_mobilesal.c` from PR #208, planned `feature_transnet_v2.c`).
  `feature_lpips.c` shrinks 305 → 205 LOC and `fastdvdnet_pre.c`
  341 → 317 LOC (net −100 lines across the two master extractors);
  new tiny-AI extractors target ~30 LOC of extractor-specific tensor
  wiring instead of ~150 LOC where 70 % is plumbing. Bit-exact
  behaviour preserved (YUV→RGB body and option-table layout are
  literal moves; all 40 libvmaf CPU-build tests + the 10 dnn-suite
  smoke tests pass). Power-of-10 friendly — no setjmp/longjmp, no
  recursion, bounded loops, single struct-literal macro (rule 9
  compliant). Recipe doc
  [`docs/ai/extractor-template.md`](docs/ai/extractor-template.md);
  `libvmaf/src/dnn/AGENTS.md` invariant row pins the contract for
  rebase. See
  [ADR-0221](docs/adr/0221-tiny-ai-extractor-template.md).

- **SYCL fp64-less device init log (T7-17 / ADR-0220).** The init
  message emitted on devices that lack `sycl::aspect::fp64` (Intel
  Arc A-series, most Intel iGPUs, many mobile / embedded GPUs) is
  reworded from the misleading WARNING-level "device lacks fp64
  support — using int64 emulation for gain limiting" to an
  INFO-level "device lacks native fp64 — kernels already use fp32
  + int64 paths, no emulation overhead". An audit confirmed every
  SYCL feature kernel is already fp64-free in its device code:

- **DISTS extractor proposal (T7-DISTS / ADR-0236).** Proposal-stage
  scaffolding for a DISTS perceptual-similarity tiny-AI FR
  extractor ([Ding 2020 PAMI](https://doi.org/10.1109/TPAMI.2020.3045810))
  mirroring `lpips_sq`'s public surface (`relu1_2`–`relu5_3` VGG-16
  features, channel-wise SSIM-style texture + structure terms,
  scalar `dists_sq` per frame). Closes Research-0033 actionable
  item #5 from the Bristol VI-Lab 2026 NVC review. Status:
  **Proposed only** — no extractor C code, no ONNX, no registry
  entry; tracked separately as T7-DISTS for the implementation PR.
  Companion research digest:
  [`docs/research/0043-dists-extractor-design.md`](docs/research/0043-dists-extractor-design.md).
  ADM gain limiting uses an int64 Q31 split-multiply
  (`gain_limit_to_q31` + `launch_decouple_csf<false>` in
  `libvmaf/src/feature/sycl/integer_adm_sycl.cpp`), VIF gain
  limiting uses fp32 `sycl::fmin`, and accumulators use
  `sycl::plus<int64_t>`. There is no fp64-emulation fallback — the
  previous wording suggested one. New
  [`docs/backends/sycl/overview.md`](docs/backends/sycl/overview.md)
  § "fp64-less device contract (T7-17)" documents the
  no-`double`-in-kernel-lambdas rule + the SPIR-V module-taint
  rationale; new `libvmaf/src/sycl/AGENTS.md` invariant row pins
  the contract on rebase. The originally reported 5–10× Arc A380
  vs Vulkan perf gap has a different root cause (kernel geometry,
  sub-group size, memory pattern) — out of T7-17's scope. See
  [ADR-0220](docs/adr/0220-sycl-fp64-fallback.md).

- **Tiny-AI: vmaf_tiny_v2 retrained on full 4-corpus.** The shipped
  `model/tiny/vmaf_tiny_v2.onnx` is now trained on
  `runs/full_features_4corpus.parquet` (Netflix + KoNViD + BVI-DVC
  A+B+C+D, 330 499 rows) instead of the previous 3-corpus parquet
  (305 795 rows; ADR-0216 originally referenced this set). The
  hyperparameters, architecture (`mlp_small`, canonical-6,
  `lr=1e-3`, 90 epochs), bundled-scaler ONNX layout (ADR-0049),
  opset (17), and registry entry shape are unchanged — only the
  weights and the embedded `(mean, std)` constants change. Train
  PLCC `0.9999` / RMSE `0.153`; ship-gate validation on the
  Netflix slice (first 100 rows) PLCC `0.9999` / RMSE `0.191`.
  Sha256 updated in `model/tiny/registry.json`. See
  [`docs/ai/models/vmaf_tiny_v2.md`](docs/ai/models/vmaf_tiny_v2.md).

### Added

- **GPU-gen ULP calibration head (proposal-stage, T7-GPU-ULP-CAL / ADR-0234).**
  Scaffolds a tiny per-arch `(arch_id, raw_gpu_score) → cpu_score`
  calibration head to close the ~1e-4 cross-backend ULP divergence
  currently within `places=4` tolerance but observable on float
  metrics. Status: **Proposed only** — no model trained, no
  `--gpu-calibrated` CLI flag enabled, no runtime path lit. Future
  PR (gated on per-arch held-out PLCC ≥ 0.9999 and worst-case
  residual ≤ 1e-6) flips the flag default once the empirical case
  is made. Companion research digest:
  [`docs/research/0041-gpu-gen-ulp-calibration.md`](docs/research/0041-gpu-gen-ulp-calibration.md).
  Data-collection scaffold lands at
  [`ai/scripts/collect_gpu_calibration_data.py`](ai/scripts/collect_gpu_calibration_data.py).
- **Vulkan VkImage import — v2 async pending-fence ring (T7-29
  part 4 / ADR-0235).** The synchronous in-call fence wait that
  shipped in [ADR-0186](docs/adr/0186-vulkan-image-import-impl.md)
  is replaced with a per-frame fence ring keyed by
  `frame_index % ring_size`. `vmaf_vulkan_import_image` now
  records, submits, and returns immediately; the FFmpeg
  `libvmaf_vulkan` filter's decoder thread can run ahead while
  libvmaf drains in the background. `vmaf_vulkan_wait_compute`
  blocks on every outstanding fence in submission order and is
  the natural drain point before
  `vmaf_vulkan_state_build_pictures` reads the host mappings.
  Default ring depth is 4 ("frames-in-flight" canonical value);
  staging arena scales `2 × ring_size` per state (~16 MiB
  host-visible at 1080p 8-bit Y, default depth). Public ABI
  preserved — the ring is fully internal to `VmafVulkanState`,
  so [`ffmpeg-patches/0006-libvmaf-add-libvmaf-vulkan-filter.patch`](ffmpeg-patches/0006-libvmaf-add-libvmaf-vulkan-filter.patch)
  needs no changes. New contract smoke
  [`libvmaf/test/test_vulkan_async_pending_fence.c`](libvmaf/test/test_vulkan_async_pending_fence.c)
  pins the v1 → v2 swap (`-EINVAL` on `vk_image == 0` regardless
  of slot index, idle `wait_compute` is a no-op, NULL-ctx checks
  in `read_imported_pictures` for slot 0, ring_size-1, and
  post-wrap indices). New
  [`libvmaf/src/vulkan/AGENTS.md`](libvmaf/src/vulkan/AGENTS.md)
  carries the rebase-sensitive ring invariants (fixed depth,
  reset-after-wait, drain-before-destroy). Cross-backend gate
  contract unchanged — async submission only changes when the
  host can read, not which bytes the staging buffer receives;
  `places=4` holds. Status flips from Proposed to Accepted
  after the lavapipe / hardware wall-clock measurement gate
  (`v2 ≤ 0.7 × v1` on Netflix normal pair) lands.
- **`enable_lcs` MS-SSIM extras on CUDA + Vulkan (T7-35 / ADR-0215).**
  The Vulkan `float_ms_ssim_vulkan` and CUDA `float_ms_ssim_cuda`
  extractors now honour the `enable_lcs` option, emitting the same
  15 per-scale metrics the CPU reference does:
  `float_ms_ssim_{l,c,s}_scale{0..4}`. The GPU kernels already
  computed the per-scale L/C/S means (the CUDA vert kernel is even
  named `ms_ssim_vert_lcs`); only the host-side
  `vmaf_feature_collector_append` calls were missing. No kernel,
  shader or device-buffer changes — default-path
  (`enable_lcs=false`) output is bit-identical to the pre-T7-35
  binary. The Vulkan option help text is rewritten to drop the
  "(reserved; not yet implemented in the GPU path)" caveat; the
  CUDA file-header drops the corresponding "v1 does NOT
  implement enable_lcs" deferral. New `float_ms_ssim_lcs`
  pseudo-feature in
  [`scripts/ci/cross_backend_vif_diff.py`](scripts/ci/cross_backend_vif_diff.py)
  and
  [`scripts/ci/cross_backend_parity_gate.py`](scripts/ci/cross_backend_parity_gate.py)
  pins all 16 metrics at `places=4`. The SYCL `float_ms_ssim_sycl`
  twin does not currently expose `enable_lcs` (empty `options[]`
  table) and stays out-of-scope for this PR; tracked as a
  follow-up under T7-35.
- **MobileSal saliency feature extractor (T6-2a / ADR-0218).** New
  no-reference scoring-side extractor `mobilesal` under
  [`libvmaf/src/feature/feature_mobilesal.c`](libvmaf/src/feature/feature_mobilesal.c).
  Emits scalar `saliency_mean` per frame via
  `vmaf_feature_collector_append`. The session binds tensors by name —
  input `input` (NCHW float32 RGB, ImageNet-normalised) and output
  `saliency_map` (NCHW float32 [1,1,H,W]) — so a future drop-in replaces
  the placeholder without C changes. Shipped checkpoint
  [`model/tiny/mobilesal.onnx`](model/tiny/mobilesal.onnx) is a
  smoke-only synthetic placeholder (330 bytes; 3→1 1×1 Conv + Sigmoid)
  generated by
  [`scripts/gen_mobilesal_placeholder_onnx.py`](scripts/gen_mobilesal_placeholder_onnx.py)
  and labelled `smoke: true` in
  [`model/tiny/registry.json`](model/tiny/registry.json); real upstream
  [`yun-liu/MobileSal`](https://github.com/yun-liu/MobileSal) (MIT)
  weights are tracked as T6-2a-followup. Per-model docs at
  [`docs/ai/models/mobilesal.md`](docs/ai/models/mobilesal.md). Smoke
  test [`libvmaf/test/test_mobilesal.c`](libvmaf/test/test_mobilesal.c)
  mirrors `test_lpips.c`. The encoder-side per-CTU QP-offset sidecar
  (`tools/vmaf-roi`, saliency-weighted L1/L2 FR feature) is the
  T6-2b follow-up. See
  [ADR-0218](docs/adr/0218-mobilesal-saliency-extractor.md).
- **TransNet V2 shot-boundary detector (T6-3a)** — new feature
  extractor `transnet_v2` under
  [`libvmaf/src/feature/transnet_v2.c`](libvmaf/src/feature/transnet_v2.c)
  registers a 100-frame-sliding-window shot-change detector backed
  by the public `vmaf_dnn_session_*` API. ONNX I/O contract:
  `frames` float32 `[1, 100, 3, 27, 48]` (100-frame stack of 27x48
  RGB thumbnails) → `boundary_logits` float32 `[1, 100]`
  (per-frame logits before sigmoid). Internal 100-slot ring buffer
  with head-clamp at clip start; per-frame **two** scalars
  appended through the existing feature-collector plumbing —
  `shot_boundary_probability` (sigmoid of the centre-slot logit)
  and `shot_boundary` (binary 0/1 thresholded at 0.5). Picks up
  `model_path` from the feature option or
  `VMAF_TRANSNET_V2_MODEL_PATH` env var (mirrors LPIPS /
  FastDVDnet); declines cleanly with `-EINVAL` when neither is
  set. **Placeholder weights only** —
  `model/tiny/transnet_v2.onnx` is a smoke-only ~125 KB
  randomly-initialised tiny MLP with the right I/O shape; real
  upstream weights (Soucek & Lokoc 2020 MIT) tracked as
  **T6-3a-followup**, the per-shot CRF predictor +
  `tools/vmaf-perShot` CLI as **T6-3b**. New ADR
  [ADR-0220](docs/adr/0223-transnet-v2-shot-detector.md), user-facing
  doc [`docs/ai/models/transnet_v2.md`](docs/ai/models/transnet_v2.md),
  placeholder-export driver
  [`ai/scripts/export_transnet_v2_placeholder.py`](ai/scripts/export_transnet_v2_placeholder.py),
  registry row in [`model/tiny/registry.json`](model/tiny/registry.json),
  6-subtest smoke at
  [`libvmaf/test/test_transnet_v2.c`](libvmaf/test/test_transnet_v2.c).
- **Port upstream `8a289703` + `1b6c3886` — 32-bit ADM/cpu fallbacks
  (T-NEW-3, audit row from PR #205).** Cherry-picks two upstream
  Netflix/vmaf commits by Christopher Degawa that make the AVX2 /
  AVX-512 ADM SIMD paths build on 32-bit x86: a portable
  `extract_epi64` fallback for `_mm256_extract_epi64` (unavailable
  on i686) in `libvmaf/src/feature/x86/adm_avx2.c` and
  `libvmaf/src/feature/x86/adm_avx512.c`, and removal of the
  `#if ARCH_X86_64` guard around the AVX/AVX2/AVX-512 detection
  block in `libvmaf/src/x86/cpu.c`. Pairs with the existing
  i686 build-only CI matrix lane (PR #87 / ADR-0151), which
  currently runs with `-Denable_asm=false`; together with this
  port the lane can move toward exercising the integer ADM SIMD
  paths once additional 32-bit intrinsic fallbacks (motion / psnr,
  not in this PR) land. Conflict resolution preserved the fork's
  clang-format-100col layout and the `_Alignas(64)` LTO-correctness
  slot in adm_avx512.c (see
  [`docs/development/known-upstream-bugs.md`](docs/development/known-upstream-bugs.md)).
  Verbatim 2-commit upstream port; no fork-local algorithmic
  changes. Original-commits:
  [`8a289703`](https://github.com/Netflix/vmaf/commit/8a289703),
  [`1b6c3886`](https://github.com/Netflix/vmaf/commit/1b6c3886).
- **`speed_chroma` + `speed_temporal` feature extractors (T-NEW-1,
  upstream port).** Verbatim cherry-pick of Netflix/vmaf
  [`d3647c73`](https://github.com/Netflix/vmaf/commit/d3647c73)
  ("feature/speed: port speed_chroma and speed_temporal extractors")
  with its dependency
  [`4ad6e0ea`](https://github.com/Netflix/vmaf/commit/4ad6e0ea)
  ("feature/vif: port helper functions"). Adds two research-stage
  float-only extractors: `speed_chroma` (per-channel chroma fidelity
  on Y / U / V / UV) and `speed_temporal` (temporal Speed score over
  a small cyclic frame buffer). Registers behind
  `VMAF_FLOAT_FEATURES` so only `-Denable_float=true` builds ship
  them. The cherry-pick widens the in-tree `picture_copy()` signature
  with a new `int channel` parameter; every fork-local caller
  (CUDA / Vulkan MS-SSIM, SSIM) is updated to pass `channel=0`. New
  registration smoke test at `libvmaf/test/test_speed.c` (5 cases).
  Per-feature doc rows + section under
  [`docs/metrics/features.md`](docs/metrics/features.md). Closes
  T-NEW-1 from the T7-4 quarterly upstream audit (PR #205).
- **BVI-DVC feature-extraction pipeline (corpus-3 for tiny-AI v2).**
  New
  [`ai/scripts/bvi_dvc_to_full_features.py`](ai/scripts/bvi_dvc_to_full_features.py)
  mirrors the canonical-21
  [`ai/scripts/konvid_to_full_features.py`](ai/scripts/konvid_to_full_features.py)
  pipeline (PR #178) but sources clips from BVI-DVC Part 1 (Ma,
  Zhang, Bull 2021) — a 4-tier 4:2:0 10-bit YCbCr reference
  corpus (A=3840x2176, B=1920x1088, C=960x544, D=480x272, 193
  clips per tier, 64 frames each). Streams individual `.mp4`
  entries from the 84 GB zip without unpacking the archive,
  decodes 10-bit YUV reference, encodes a CRF-35 distorted side,
  runs libvmaf with the verbatim FULL_FEATURES tuple +
  vmaf_v0.6.1 model attached for the per-frame teacher score,
  caches per-clip JSON under
  `~/.cache/vmaf-tiny-ai-bvi-dvc-full/<key>.json`, and
  aggregates to `runs/full_features_bvi_dvc_<tier>.parquet`
  (gitignored). The 10-bit input path is the only structural
  delta vs. the 8-bit konvid invocation (`-pix_fmt yuv420p10le`
  and `--bitdepth 10`). `--tier {A,B,C,D,all}` selects the
  resolution; D-tier wall-clock is ~3 s/clip on CPU, ~10 min
  for the full 193-clip tier. Reference-only dataset — VMAF
  score is the teacher signal since BVI-DVC ships no DMOS.
  Smoke command: `python3 ai/scripts/bvi_dvc_to_full_features.py
  --tier D --bvi-zip "<path>/BVI-DVC Part 1.zip" --vmaf-bin
  build-cpu/tools/vmaf --max-clips 5`. Feeds the upcoming
  3-corpus (Netflix + KoNViD + BVI-DVC) Phase-3b sweep that
  validates Subset B's win across resolution scales the first
  two corpora don't cover.
- **Per-backend GPU kernel scaffolding templates (T7-XX / ADR-0221).**
  New header-only inline-helper templates
  [`libvmaf/src/cuda/kernel_template.h`](libvmaf/src/cuda/kernel_template.h)
  (296 LOC) and
  [`libvmaf/src/vulkan/kernel_template.h`](libvmaf/src/vulkan/kernel_template.h)
  (410 LOC) absorb the lifecycle boilerplate every fork-added GPU
  feature kernel currently hand-rolls. CUDA template captures the
  private non-blocking stream + 2 events + device accumulator +
  pinned host readback slot (`VmafCudaKernelLifecycle` +
  `VmafCudaKernelReadback` + 6 inlines wrapping `cuCtxPushCurrent` /
  `cuStreamCreateWithPriority` / `cuEventCreate` / `cuMemsetD8Async` /
  `cuStreamWaitEvent` / destroy ladders). Vulkan template captures
  the descriptor-set layout + pipeline layout + shader module +
  compute pipeline + descriptor pool plus per-frame command-buffer
  + fence pair (`VmafVulkanKernelPipeline` + `VmafVulkanKernelSubmit`
  + 5 inlines wrapping `vkCreateDescriptorSetLayout` /
  `vkCreatePipelineLayout` / `vkCreateShaderModule` /
  `vkCreateComputePipelines` / `vkCreateDescriptorPool` / per-frame
  `vkAllocateCommandBuffers` / `vkBeginCommandBuffer` /
  `vkCreateFence` / `vkQueueSubmit` / `vkWaitForFences` and
  reverse-order destroys). Templates are **template-only** — no
  existing kernel includes them in this PR. Each future kernel
  migration ships in its own PR gated by the `places=4`
  cross-backend-diff lane (per
  [ADR-0214](docs/adr/0214-gpu-parity-ci-gate.md)) plus the Netflix
  CPU golden gate. Per-backend (not cross-backend) because CUDA's
  async-stream + event model and Vulkan's command-buffer + fence +
  descriptor-pool model share no concrete shape. Helper functions
  (not macros) for cuda-gdb / Nsight / RenderDoc step-debugging.
  New user doc
  [`docs/backends/kernel-scaffolding.md`](docs/backends/kernel-scaffolding.md)
  covers the helper signatures + a per-kernel migration sketch;
  AGENTS.md invariant rows added under
  `libvmaf/src/cuda/AGENTS.md` and a new
  `libvmaf/src/vulkan/AGENTS.md`. Deferred migrations:
  - **T7-XX-followup-a**: SYCL kernel-template refactor (deferred
    — needs icpx host).
  - **T7-XX-followup-b**: migrate fork-added CUDA kernels to
    template (`integer_psnr_cuda` first, then `ssimulacra2_cuda`).
  - **T7-XX-followup-c**: migrate fork-added Vulkan kernels to
    template (`psnr_vulkan` first, then `motion_vulkan` /
    `ssim_vulkan` / `cambi_vulkan`).

  See [ADR-0221](docs/adr/0221-gpu-kernel-template.md).
- **`vmaf-perShot` per-shot CRF predictor sidecar (T6-3b / ADR-0222).**
  New standalone CLI under `libvmaf/tools/vmaf_per_shot.c` that
  consumes a YUV reference, segments it into shots via a
  frame-difference heuristic, computes per-shot complexity +
  motion-energy + length signals, and emits a per-shot CRF plan as
  CSV (default) or JSON. v1 uses a transparent linear-blend
  predictor (no training corpus needed); v2 will swap in a small
  trained MLP under the same schema. Pairs with the in-flight
  TransNet V2 extractor (T6-3a / ADR-0220) — once that lands, the
  tool will accept a pre-computed shot map via `--shots`. Smoke
  test under `libvmaf/tools/test/test_vmaf_per_shot.sh` runs
  against the `testdata/ref_576x324_48f.yuv` fixture (≥1 shot
  detected, all CRFs in `[crf-min, crf-max]`). New user doc
  [`docs/usage/vmaf-perShot.md`](docs/usage/vmaf-perShot.md);
  cross-linked from [`docs/usage/cli.md`](docs/usage/cli.md);
  `libvmaf/tools/AGENTS.md` carries the standalone-sidecar +
  stable-schema invariants. See
  [ADR-0222](docs/adr/0222-vmaf-per-shot-tool.md).
- **`vmaf-roi` sidecar binary for per-CTU QP offsets (T6-2b /
  ADR-0221).** New CLI tool at `libvmaf/tools/vmaf_roi.c` that
  consumes raw planar YUV + a 0-based frame index, optionally runs
  the `mobilesal` saliency model through the public
  `vmaf_dnn_session_run_luma8()` API (T6-2a / ADR-0218 / PR #208 is
  the scoring-side counterpart), reduces the per-pixel saliency
  map to a per-CTU mean, and emits an encoder-native QP-offset
  sidecar — ASCII per-row grid for x265 (`--qpfile-style`) or raw
  `int8_t` binary for SVT-AV1 (`--roi-map-file`). Mapping is
  `qp = clamp(-strength * (2 * saliency - 1), -12, +12)`: high
  saliency drives the offset negative (boost quality), low saliency
  positive (save bits). 8-bit YUV only in T6-2b; 10/12-bit follows
  the `mobilesal` bit-depth upgrade. When `--saliency-model` is
  absent the tool falls back to a deterministic center-weighted
  radial placeholder, documented as smoke-test-only — not for real
  encodes. New `docs/usage/vmaf-roi.md`. New header-only helper TU
  `libvmaf/tools/vmaf_roi_core.h` lets the smoke test
  (`libvmaf/test/test_vmaf_roi.c`) compile the per-CTU mean
  reducer and QP-offset mapper without dragging libvmaf's link
  surface in. See [ADR-0221](docs/adr/0221-vmaf-roi-tool.md).

- **`motion3_score` GPU coverage on Vulkan + CUDA + SYCL
  (T3-15(c) / ADR-0219).** The `motion` extractor's third output
  (`integer_motion3`) is now emitted by the three GPU twins —
  `motion_vulkan`, `motion_cuda`, `motion_sycl` — in 3-frame window
  mode (the default; the 5-frame `motion_five_frame_window=true`
  mode is now explicitly rejected with `-ENOTSUP` at `init()`).
  motion3 is a deterministic scalar post-process of `motion2`
  (`motion_blend(motion2 * motion_fps_weight, motion_blend_factor,
  motion_blend_offset)` clipped to `motion_max_val`, with optional
  moving-average), so the GPU port is purely a host-side add-on
  after the existing SAD reduction — no kernel changes. The full
  CPU options surface (`motion_blend_factor`, `motion_blend_offset`,
  `motion_fps_weight`, `motion_max_val`, `motion_moving_average`)
  is added to each GPU extractor with defaults matching CPU. The
  cross-backend parity gates
  ([`scripts/ci/cross_backend_parity_gate.py`](scripts/ci/cross_backend_parity_gate.py)
  + [`scripts/ci/cross_backend_vif_diff.py`](scripts/ci/cross_backend_vif_diff.py))
  extend `FEATURE_METRICS["motion"]` to compare `integer_motion3`
  at `places=4`. Closes the GPU coverage gap that was tracked as
  T3-17 in the backlog audit (now T3-15(c)) and the "Vulkan motion3
  GPU gap" in [`docs/backlog-audit-2026-04-28.md`](docs/backlog-audit-2026-04-28.md)
  row A.1.4. See [ADR-0219](docs/adr/0219-motion3-gpu-coverage.md).
- **Tiny-AI Wave-1 C1 baseline `fr_regressor_v1` (T6-1a / ADR-0221).**
  Ships `model/tiny/fr_regressor_v1.onnx` — a 6-feature canonical
  MLP (FRRegressor: 2-layer GELU, hidden=64) that maps libvmaf's
  classical feature vector (`adm2`, `vif_scale0..3`, `motion2`) to a
  per-frame VMAF score. Trained on the locally-available Netflix
  Public Dataset (9 ref + 70 dis YUVs at `.workingdir2/netflix/`,
  unblocked from the access-gated deferral in ADR-0168) with
  `vmaf_v0.6.1` as the per-frame DMOS-aligned teacher. New trainer
  [`ai/scripts/train_fr_regressor.py`](ai/scripts/train_fr_regressor.py)
  runs 9-fold leave-one-source-out (LOSO), refuses to overwrite the
  registry below mean PLCC 0.95, then re-trains on all sources for
  the shipping checkpoint. Sidecar
  [`model/tiny/fr_regressor_v1.json`](model/tiny/fr_regressor_v1.json)
  pins the per-feature standardisation (`feature_mean` /
  `feature_std`) so callers can reproduce the input pipeline; the
  ONNX itself stays standardisation-agnostic. New user doc
  [`docs/ai/models/fr_regressor_v1.md`](docs/ai/models/fr_regressor_v1.md);
  Wave-1 roadmap §2.1 row flips from **Deferred** to **Shipped
  2026-04-29**; `ai/AGENTS.md` carries the `fr_regressor_v1`
  contract row + rebase-sensitive invariants. See
  [ADR-0221](docs/adr/0221-fr-regressor-v1.md).
- **Tiny-AI: vmaf_tiny_v2 (ADR-0216).** Ship `model/tiny/vmaf_tiny_v2.onnx`
  — the production-grade tiny VMAF fusion model on the Phase-3 validated
  configuration: `mlp_small` (6 → 16 → 8 → 1, ~257 params) over the
  canonical-6 features (`adm2`, `vif_scale0..3`, `motion2`) with the
  StandardScaler `(mean, std)` baked into the ONNX graph as Constant
  `Sub` + `Div` nodes (single-file deploy, trust-root sha256 covers
  calibration values). 90 epochs Adam @ lr=1e-3, MSE, batch 256.
  Validated PLCC `0.9978 ± 0.0021` on Netflix LOSO; `0.9998` on
  KoNViD 5-fold; +0.005–0.018 over the prior Subset-B baseline
  (Phase-3a/b/c chain). Three new scripts: `train_vmaf_tiny_v2.py`,
  `export_vmaf_tiny_v2.py`, `validate_vmaf_tiny_v2.py`. Registered as
  `vmaf_tiny_v2` (kind `fr`, `quant_mode: fp32`, `smoke: false`) in
  `model/tiny/registry.json`. New user-facing doc
  [`docs/ai/models/vmaf_tiny_v2.md`](docs/ai/models/vmaf_tiny_v2.md);
  [`docs/ai/inference.md`](docs/ai/inference.md) default flips v1 →
  v2; [`docs/ai/roadmap.md`](docs/ai/roadmap.md) row marked shipped.

- **GPU-parity matrix CI gate (T6-8 / ADR-0214).** New
  [`scripts/ci/cross_backend_parity_gate.py`](scripts/ci/cross_backend_parity_gate.py)
  iterates every `(feature, backend-pair)` cell, diffs per-frame
  metrics with a feature-specific absolute tolerance declared in
  one place (`FEATURE_TOLERANCE`), and emits one JSON record + one
  Markdown row per cell. New CI lane `vulkan-parity-matrix-gate`
  in
  [`.github/workflows/tests-and-quality-gates.yml`](.github/workflows/tests-and-quality-gates.yml)
  runs the gate over CPU ↔ Vulkan/lavapipe on every PR (no GPU
  hardware needed); CUDA / SYCL / hardware-Vulkan are advisory
  until a self-hosted runner is registered. New user-facing doc
  at [`docs/development/cross-backend-gate.md`](docs/development/cross-backend-gate.md);
  `docs/backends/index.md` cross-references it. Generalises and
  is the long-term replacement for the per-feature
  `cross_backend_vif_diff.py` lane (kept for one release cycle).
  See [ADR-0214](docs/adr/0214-gpu-parity-ci-gate.md).
- **FastDVDnet temporal pre-filter (T6-7)** — new feature
  extractor `fastdvdnet_pre` under
  [`libvmaf/src/feature/fastdvdnet_pre.c`](libvmaf/src/feature/fastdvdnet_pre.c)
  registers a 5-frame-sliding-window temporal denoiser backed by
  the public `vmaf_dnn_session_*` API. ONNX I/O contract:
  `frames` float32 NCHW `[1, 5, H, W]` (channel axis stacks
  `[t-2, t-1, t, t+1, t+2]`) → `denoised` float32 NCHW
  `[1, 1, H, W]`. Internal 5-slot ring buffer with replicate-edge
  clamp at clip start/end; per-frame scalar
  `fastdvdnet_pre_l1_residual` appended through the existing
  feature-collector plumbing. Picks up `model_path` from the
  feature option or `VMAF_FASTDVDNET_PRE_MODEL_PATH` env var
  (mirrors LPIPS); declines cleanly with `-EINVAL` when neither
  is set. **Placeholder weights only** —
  `model/tiny/fastdvdnet_pre.onnx` is a smoke-only ~6 KB
  randomly-initialised 3-layer CNN with the right I/O shape;
  real upstream-derived FastDVDnet weights + the FFmpeg
  `vmaf_pre_temporal` filter that consumes the denoised frame
  buffer are tracked as **T6-7b**. New ADR
  [ADR-0215](docs/adr/0215-fastdvdnet-pre-filter.md), user-facing
  doc [`docs/ai/models/fastdvdnet_pre.md`](docs/ai/models/fastdvdnet_pre.md),
  registration smoke test
  [`libvmaf/test/test_fastdvdnet_pre.c`](libvmaf/test/test_fastdvdnet_pre.c)
  mirroring `test_lpips.c`. Closes backlog item T6-7.
- **Vulkan PSNR — chroma extension (T3-15(b))** — `psnr_vulkan`
  now emits `psnr_cb` and `psnr_cr` alongside `psnr_y`. Three
  back-to-back dispatches of the existing plane-agnostic
  `psnr.comp` shader against per-plane SSBOs and per-plane
  `(width, height, num_workgroups_x)` push constants; YUV400
  clamps to luma-only at runtime. Cross-backend gate
  (`scripts/ci/cross_backend_vif_diff.py --feature psnr`)
  extended to assert all three plane scores at `places=4`;
  measured `max_abs_diff = 0.0` across 48 frames at 576×324 on
  lavapipe (deterministic int64 SSE accumulators on both sides).
  See [ADR-0216](docs/adr/0216-vulkan-chroma-psnr.md). Doc at
  [`docs/backends/vulkan/overview.md`](docs/backends/vulkan/overview.md).

- **Embedded MCP server scaffold (T5-2, audit-first)** — new
  public header
  [`libvmaf/include/libvmaf/libvmaf_mcp.h`](libvmaf/include/libvmaf/libvmaf_mcp.h)
  declaring the in-process MCP API (`vmaf_mcp_init` /
  `_start_sse` / `_start_uds` / `_start_stdio` / `_stop` /
  `_close` / `_available` / `_transport_available`); stub TU at
  `libvmaf/src/mcp/mcp.c` returning `-ENOSYS` (or `-EINVAL` on
  bad arguments); new umbrella `enable_mcp` boolean (default
  `false`) plus per-transport sub-flags `enable_mcp_sse`,
  `enable_mcp_uds`, `enable_mcp_stdio`; 12-sub-test smoke at
  `libvmaf/test/test_mcp_smoke.c` pinning the `-ENOSYS` +
  NULL-guard contract; user-facing doc at
  [`docs/mcp/embedded.md`](docs/mcp/embedded.md). **Scaffold
  only** — the T5-2b follow-up PR vendors cJSON + mongoose,
  spawns the dedicated MCP pthread + SPSC ring buffer, and
  fills in the SSE / UDS / stdio transport bodies. Same
  audit-first shape as ADR-0175 (Vulkan T5-1) and ADR-0184
  (T7-29 part 1). See
  [ADR-0209](docs/adr/0209-mcp-embedded-scaffold.md) +
  [ADR-0128](docs/adr/0128-embedded-mcp-in-libvmaf.md) +
  [Research-0005](docs/research/0005-embedded-mcp-transport.md).
- **`cambi_vulkan` feature extractor (T7-36 / ADR-0210)** — closes
  the GPU long-tail matrix terminus. Strategy II hybrid: GPU
  shaders run preprocess, per-pixel derivative, 7×7 spatial-mask
  SAT, 2× decimate, 3-tap separable mode filter; the
  precision-sensitive sliding-histogram `calculate_c_values` + top-K
  pool stay on the host. Bit-exact w.r.t. CPU by construction;
  cross-backend gate runs at `places=4`. New ADR
  [`docs/adr/0210-cambi-vulkan-integration.md`](docs/adr/0210-cambi-vulkan-integration.md)
  + research digest
  [`docs/research/0032-cambi-vulkan-integration.md`](docs/research/0032-cambi-vulkan-integration.md).
- **T6-9: Tiny-model registry schema + Sigstore `--tiny-model-verify`**
  ([ADR-0211](docs/adr/0211-model-registry-sigstore.md)). Formal
  JSON Schema (Draft 2020-12) at
  [`model/tiny/registry.schema.json`](model/tiny/registry.schema.json)
  extended with `license`, `license_url`, and `sigstore_bundle`
  fields per entry; `schema_version` bumped to `1`. New CLI flag
  `--tiny-model-verify` wires `cosign verify-blob` via
  `posix_spawnp(3p)` against the registry's `sigstore_bundle` path,
  failing closed on missing bundle / missing cosign / non-zero exit.
  Public C entry point: `vmaf_dnn_verify_signature()` in
  [`libvmaf/include/libvmaf/dnn.h`](libvmaf/include/libvmaf/dnn.h).
  Python validator at
  [`ai/scripts/validate_model_registry.py`](ai/scripts/validate_model_registry.py)
  (Draft 2020-12 with a structural fallback when `jsonschema` is
  absent) covers schema + cross-file consistency and is a pre-push
  gate. Documentation: new
  [`docs/ai/model-registry.md`](docs/ai/model-registry.md), updated
  [`docs/ai/inference.md`](docs/ai/inference.md) and
  [`docs/ai/security.md`](docs/ai/security.md). Tests:
  `python/test/model_registry_schema_test.py` (10 cases) and
  `libvmaf/test/dnn/test_tiny_model_verify.c` (18 failure-mode
  cases on Unix + 1 NULL-arg case covering malformed JSON,
  default-registry
  derivation, fake-cosign success / non-zero exit, and
  empty / missing PATH branches — drives `model_loader.c`
  coverage to ≥85% per the Coverage Gate critical-file rule;
  ENOSYS smoke on Windows). All five shipped models gain
  license metadata (BSD-3-Clause-Plus-Patent for fork-trained;
  BSD-2-Clause for the upstream LPIPS-Sq export).

### Removed

- **`VMAF_MAX_MODEL_BYTES` env override retired (T7-12)**: the
  historical environment-variable knob that let callers raise (or
  lower, for tests) the tiny-AI ONNX file-size cap has been removed
  from `vmaf_dnn_session_open()` and `vmaf_use_tiny_model()`. Two
  release cycles passed without a shipped model approaching the cap,
  so the testing-hatch is retired in favour of the compile-time
  constant `VMAF_DNN_DEFAULT_MAX_BYTES` (50 MB) as the single source
  of truth. Callers that genuinely need a larger envelope must bump
  the constant in
  [`libvmaf/src/dnn/model_loader.h`](libvmaf/src/dnn/model_loader.h)
  and rebuild. The two env-driven unit tests
  (`test_session_open_respects_max_bytes_env`,
  `test_session_open_ignores_invalid_max_bytes_env`) are removed; all
  other size-cap coverage (oversize fixture rejection, `S_ISREG`
  check, allowlist) stays intact.

### Changed

- **Quarterly upstream-backlog re-audit (T7-4)** (fork-local doc):
  new
  [`docs/upstream-backlog-audit-2026-04-29.md`](docs/upstream-backlog-audit-2026-04-29.md)
  walks the 12 upstream Netflix/vmaf commits landed since the
  fork's last `chore(upstream): port` boundary
  (`798409e3` / `314db130`, PR #181). 8 are already on fork
  (cherry-picked, ported, or covered by an Accepted ADR /
  Research digest); 4 are flagged for fork action and surface
  as 4 recommended new T-rows: port `feature/speed`
  (`d3647c73`), port adm + vif test deltas from `c70debb1`,
  port 32-bit ADM/cpu fallbacks (`8a289703` + `1b6c3886`), and
  schedule the next re-audit for 2026-07-29. No code changes,
  no `docs/state.md` changes (no upstream commit ruled in/out
  a fork bug). Doubles as the ADR-0108 research digest for the
  audit PR.

- **T7-32: backlog-hygiene S-task bundle (3 micro-investigations).**
  One PR closes three S-effort follow-ups identified by the
  2026-04-28 BACKLOG audit. (a) `motion_v2` AVX2 `srlv_epi64`
  audit: new fork-local libvmaf C unit test
  [`libvmaf/test/test_motion_v2_simd.c`](libvmaf/test/test_motion_v2_simd.c)
  exercises adversarial negative-`accum` 16-bit fixtures (10-bit
  and 12-bit, uniform-negative and alternating-mixed-sign) against
  the AVX2 path in
  [`libvmaf/src/feature/x86/motion_v2_avx2.c`](libvmaf/src/feature/x86/motion_v2_avx2.c);
  on the bench host the post-`abs()` aggregation absorbs the
  per-lane logical-vs-arithmetic shift difference and SAD totals
  match scalar — the test stays as a permanent regression guard
  and the
  [`docs/rebase-notes.md`](docs/rebase-notes.md) §0038 placeholder
  follow-up is closed.  (b)
  [`docs/research/0006-tinyai-ptq-accuracy-targets.md`](docs/research/0006-tinyai-ptq-accuracy-targets.md)
  §4 now references the actual shipped `vmaf_tiny_v1_medium.onnx`
  checkpoint (landed by [PR
  #158](https://github.com/lusoris/vmaf/pull/158)) instead of the
  fictional `tiny-vmaf-v2` prototype name; the digest's QAT
  cost/budget framing is unchanged.  (c)
  [`python/vmaf/routine.py`](python/vmaf/routine.py) — both
  `cv_on_dataset` (line ~937) and `explain_model_on_dataset` (line
  ~1109) now mirror `VmafQualityRunner`'s contract: cv reads
  `feature_param.feature_optional_dict` when the param exposes it;
  explain reads `model.model_dict["feature_opts_dicts"]` from the
  serialised model. The two `# FIXME: as set to None, potential bug
  with inconsistent behavior with VmafQualityRunner` comments are
  removed. New regression test
  [`python/test/routine_feature_option_dict_test.py`](python/test/routine_feature_option_dict_test.py)
  covers both `None` and populated dict paths via
  `FeatureAssembler` mock. No behaviour change for callers that did
  not declare per-extractor options.

- **SYCL toolchain cleanup (T7-13 / ADR-0217).** Two thin shims land
  to unblock SYCL files for the changed-file CI lint gate and to make
  multi-version oneAPI installs usable for `vmaf_bench`:
  [`scripts/ci/sycl-bench-env.sh <version>`](scripts/ci/sycl-bench-env.sh)
  resolves an oneAPI install (side-by-side `/opt/intel/oneapi-<version>/`
  or default `/opt/intel/oneapi/`), sources its `setvars.sh`, and emits
  an `eval`-able env block (`CMPLR_ROOT`, `LD_LIBRARY_PATH`,
  `LIBRARY_PATH`, `PATH`);
  [`scripts/ci/clang-tidy-sycl.sh`](scripts/ci/clang-tidy-sycl.sh) is
  an icpx-aware `clang-tidy` wrapper that injects the oneAPI SYCL
  include path + `-D__SYCL_DEVICE_ONLY__=0` + `-Wno-unknown-warning-option`
  + `-Wno-unknown-pragmas` so stock LLVM clang-tidy resolves
  `<sycl/sycl.hpp>` and stops emitting the 4 residual
  `'sycl/sycl.hpp' file not found` errors that left
  `libvmaf/src/sycl/**` excluded from the lint gate after T7-7. New CI
  lane `Clang-Tidy SYCL (Changed Files, Advisory)` in
  [`.github/workflows/lint-and-format.yml`](.github/workflows/lint-and-format.yml)
  installs the DPC++/C++ component of oneAPI, configures a
  `build-sycl/` tree with `CC=icx CXX=icpx -Denable_sycl=true`, and
  runs the wrapper over changed SYCL TUs. Lane is advisory
  (`continue-on-error: true`) on this PR; tightens to required after
  one green run on master. New "Multi-version coexistence" section in
  [`docs/development/oneapi-install.md`](docs/development/oneapi-install.md);
  the existing "Verify SYCL clang-tidy still works" recipe rewritten
  to use the wrapper. See [ADR-0217](docs/adr/0217-sycl-toolchain-cleanup.md).

- **Upstream `c70debb1` partial port — `adm_csf` + `barten_csf`
  unit tests** (T-NEW-2). Cherry-picks the additive halves of
  Netflix upstream commit `c70debb1` ("libvmaf/test: port new
  adm/vif/speed tests", Kyle Swanson, 2026-04-28): the new
  [`libvmaf/src/feature/adm_csf_tools.h`](libvmaf/src/feature/adm_csf_tools.h)
  header (declaring the inline `adm_native_csf` DLM-CSF helper)
  and the two new unit-test files
  [`libvmaf/test/test_adm_csf.c`](libvmaf/test/test_adm_csf.c)
  (2 cases) +
  [`libvmaf/test/test_barten_csf.c`](libvmaf/test/test_barten_csf.c)
  (23 cases). The other two halves of `c70debb1` — `test_vif_tools.c`
  and `test_speed_chroma.c` — are deliberately deferred:
  `test_vif_tools` depends on the upstream `vif` runtime-helper
  chain (`NUM_KERNELSCALES`, `vif_validate_kernelscale`, etc.) that
  the fork skips per
  [Research-0024](docs/research/0024-vif-upstream-divergence.md)
  Strategy E to protect the ADR-0138 / 0139 / 0142 / 0143 SIMD
  bit-exactness contract; `test_speed_chroma` `#include`s
  `feature/speed.c`, which the fork does not yet ship (audit row
  T-NEW-1). Tracked alongside the
  [2026-04-29 quarterly upstream-backlog re-audit (PR #205)](docs/upstream-backlog-audit-2026-04-29.md).

- **Research-0031: Intel AI-PC NPU/EP applicability digest (T7-9)**
  (fork-local doc): new
  [`docs/research/0031-intel-ai-pc-applicability.md`](docs/research/0031-intel-ai-pc-applicability.md)
  evaluates whether the tiny-AI surface should add first-class
  support for the NPU on Intel Meteor / Lunar / Arrow Lake AI-PC
  platforms. Verdict: **defer the NPU path** until a maintainer
  has hardware to validate int8 + fp16 accuracy gates against
  Research-0006's PTQ pipeline. The integrated Xe / Xe2 GPU
  portion of an AI-PC platform is already reachable today through
  the existing `--tiny-device openvino` path (same code path the
  Arc A380 uses), so the iGPU surface costs the fork zero
  additional code; only the NPU device type is genuinely new
  surface and is the part that's deferred. One forward-pointer
  added to [`docs/ai/inference.md`](docs/ai/inference.md) so
  readers of the EP matrix find the digest. No code change.

- **Research-0024 + AGENTS.md: deliberately diverge from Netflix
  upstream `vif` + `float_adm` option-port chains** (fork-local
  doc): new
  [`docs/research/0024-vif-upstream-divergence.md`](docs/research/0024-vif-upstream-divergence.md)
  is a 5-strategy decision matrix on whether to port the Netflix
  upstream vif chain (`4ad6e0ea` runtime helpers / `8c645ce3`
  prescale options / `41d42c9e` edge-mirror bugfix) and the
  float_adm chain (`4dcc2f7c` 12-parameter `compute_adm`
  signature change + new `score_aim` output). Verdict:
  **Strategy E (skip + document)** for both vif and float_adm
  because (a) the fork's `vif_filter1d_table_s` precomputed
  Gaussian table preserves the ADR-0138/0139/0142/0143 SIMD
  bit-exactness contract that runtime-computed Gaussians would
  break, and (b) threading 12 new ADM parameters through the
  SIMD paths + 3 GPU backends is multi-day work without a
  concrete user demand for the new `aim` feature. **Strategy A
  (verbatim)** stays approved for the motion chain
  (`b949cebf` float_motion-only side) because float_motion has
  no precomputed-table investment to protect. Two new invariants
  added to
  [`libvmaf/src/feature/AGENTS.md`](libvmaf/src/feature/AGENTS.md)
  documenting the vif and adm divergences so future sessions
  don't accidentally re-port the chains. No code change.

- **`docs/benchmarks.md` `TBD` cells filled with measured numbers
  (T7-37)**: first end-to-end fork-bench rerun after the bench-script
  fixes (PR #169 / #170 / #171) and the Vulkan header install (PR
  #175). New per-backend tables for the Netflix 576×324 normal pair
  (CPU 598 fps, CUDA 278 fps, SYCL Arc-A380 315 fps, Vulkan 171 fps),
  the 1920×1080 5-frame pair, and the BBB 4K 200-frame pair (CUDA
  227 fps = 16.4× CPU). CPU SIMD-ISA breakdown shows AVX-512 buys
  6.62× over scalar on Zen 5; AVX2 alone gets 2.96×. `--precision`
  overhead measurement confirms `=max` (`%.17g`) is wall-time-free
  (<1 % delta) but +25.8 % JSON byte-count vs the `%.6f` default.
  Hardware-profile table updated to match the actual bench host
  (`ryzen-4090-arc`: Ryzen 9 9950X3D + RTX 4090 + Arc A380, Linux
  7.0.x CachyOS). Each backend's `frames[0].metrics` key count was
  verified per-row (CPU=15, CUDA=12, SYCL/Vulkan=34) to confirm no
  silent CPU fallback.
- **Tiny-AI PTQ accuracy across Execution Providers measured (T5-3e,
  retires the deferred GPU-EP open question in
  `docs/research/0006-tinyai-ptq-accuracy-targets.md`)**: empirical
  PLCC-drop sweep across `CPUExecutionProvider`,
  `CUDAExecutionProvider` (RTX 4090), and the OpenVINO runtime on
  Intel Arc A380 plus the OpenVINO CPU plugin. CPU EP and CUDA EP
  agree to 6 decimal places on every shipped tiny model
  (`learned_filter_v1`, `vmaf_tiny_v1`, `vmaf_tiny_v1_medium` —
  PLCC drop ≤ 1.2×10⁻⁴, well under the 1×10⁻² registry budget).
  OpenVINO CPU plugin agrees to ~10⁻⁴. Intel Arc through OpenVINO
  2026.1 is currently int8-broken: `Conv`-based int8 graphs fail to
  compile (`No layout format available for convolution: byxf /
  i32`); MLP int8 graphs (`MatMulInteger` + `DynamicQuantizeLinear`)
  compile but emit `inf`/`NaN`. Arc fp32 path is healthy. New
  harness `ai/scripts/measure_quant_drop_per_ep.py` + user doc
  `docs/ai/quant-eps.md` document the reproduction recipe and the
  CUDA-12-ABI `LD_LIBRARY_PATH` shim required on CUDA-13 hosts.
- **Backlog: 9 promote-to-T-NN rows landed from the 2026-04-28
  Section-A audit** (fork-local docs): converts the 12 untracked
  follow-up items captured in
  [`docs/backlog-audit-2026-04-28.md`](docs/backlog-audit-2026-04-28.md)
  §A and the user direction frozen in
  [`.workingdir2/decisions/section-a-decisions-2026-04-28.md`](.workingdir2/decisions/section-a-decisions-2026-04-28.md)
  into actual `.workingdir2/BACKLOG.md` rows. New rows: **T3-17**
  (motion3 GPU coverage on Vulkan + CUDA + SYCL), **T3-18** (GPU
  chroma upload + chroma metrics on Vulkan + CUDA), **T5-3e** (PTQ
  accuracy investigation on CUDA + Intel Arc — the deferral framing
  in `docs/research/0006-tinyai-ptq-accuracy-targets.md` is now
  superseded), **T5-4** (Quantization-Aware Training: implement, do
  not close — `ai/scripts/qat_train.py` remains scaffold until this
  ships), **T7-35** (`enable_lcs` MS-SSIM extra metrics on CUDA +
  Vulkan), **T7-36** (cambi GPU integration PR — replaces ADR-0205
  spike scaffolds with a real lifecycle), **T7-37** (run Netflix
  bench + replace `TBD` cells in `docs/benchmarks.md`), **T7-38**
  (SVE2 SIMD parity for SSIMULACRA 2 PTLR + IIR-blur via
  `qemu-aarch64-static` — no CI hardware required). T6-1a row
  extended to fold in §A.2.2 (DMOS-aligned bisect-cache fixture
  rides along once Netflix Public Dataset access lands). §A.1.2
  (cambi v2 c-values strategy-III) intentionally not opened — gated
  on T7-36 landing first, per the audit decisions doc. §A.4.1
  upstream `libvmaf.c` FIXMEs intentionally not opened — rebase-
  fidelity carve-out applies; first PR that touches the file sweeps
  them per CLAUDE.md §12 r12. ADR-0205 + Research-0020 +
  Research-0006 cross-link the new T-numbers in their respective
  decision / follow-up sections.

### Added

- **HIP (AMD ROCm) compute backend — scaffold-only audit-first PR
  (T7-10, ADR-0212)**: new public header
  [`libvmaf/include/libvmaf/libvmaf_hip.h`](libvmaf/include/libvmaf/libvmaf_hip.h)
  declaring `VmafHipState`, `VmafHipConfiguration`,
  `vmaf_hip_state_init` / `_import_state` / `_state_free`,
  `vmaf_hip_list_devices`, `vmaf_hip_available`. New
  `libvmaf/src/hip/` (`common.{c,h}`, `picture_hip.{c,h}`,
  `dispatch_strategy.{c,h}`) + `libvmaf/src/feature/hip/` (3 kernel
  stubs: `adm_hip.c`, `vif_hip.c`, `motion_hip.c`). All entry points
  return `-ENOSYS` until the runtime PR (T7-10b) lands. New
  `enable_hip` boolean option (default **false**) in
  [`libvmaf/meson_options.txt`](libvmaf/meson_options.txt) with
  conditional `subdir('hip')` in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build). New 9-sub-test
  smoke at
  [`libvmaf/test/test_hip_smoke.c`](libvmaf/test/test_hip_smoke.c)
  pinning the `-ENOSYS` / `-EINVAL` contract for every public
  C-API entry point. New CI matrix row `Build — Ubuntu HIP (T7-10
  scaffold)` in
  [`.github/workflows/libvmaf-build-matrix.yml`](.github/workflows/libvmaf-build-matrix.yml)
  compiling with `-Denable_hip=true` (no ROCm SDK on the runner —
  the scaffold has no SDK requirement). New
  [`docs/backends/hip/overview.md`](docs/backends/hip/overview.md);
  [`docs/backends/index.md`](docs/backends/index.md) flipped from
  "planned" to "scaffold only". New
  [`docs/research/0033-hip-applicability.md`](docs/research/0033-hip-applicability.md)
  digest covering AMD market share + ROCm 6.x Linux maturity.
  Mirrors the Vulkan T5-1 scaffold (ADR-0175); validates the
  abstraction-layer-clean-enough-to-reproduce gating condition for
  T7-10. **Zero hard runtime dependencies** —
  `dependency('hip-lang', required: false)` is silently absent on
  stock Ubuntu runners.

- **SSIMULACRA 2 SVE2 SIMD parity (T7-38, ADR-0213)** (fork-local):
  new aarch64 SVE2 sister TU
  ([`libvmaf/src/feature/arm64/ssimulacra2_sve2.c`](libvmaf/src/feature/arm64/ssimulacra2_sve2.c))
  ports the seven SSIMULACRA 2 SIMD entry points (`multiply_3plane`,
  `linear_rgb_to_xyb`, `downsample_2x2`, `ssim_map`, `edge_diff_map`,
  `blur_plane`, `picture_to_linear_rgb`) under a fixed 4-lane
  `svwhilelt_b32(0, 4)` predicate — bit-identical to the NEON sibling
  irrespective of the runtime vector length, satisfying the
  [ADR-0138](docs/adr/0138-simd-bit-exactness-policy.md) /
  [ADR-0139](docs/adr/0139-ssim-simd-bitexact-double.md) /
  [ADR-0140](docs/adr/0140-ssimulacra2-simd-bitexact.md) byte-exact
  contract. New runtime probe
  [`libvmaf/src/arm/cpu.c`](libvmaf/src/arm/cpu.c) reads
  `getauxval(AT_HWCAP2) & HWCAP2_SVE2`; new build probe in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build) runs
  `cc.compiles(... -march=armv9-a+sve2)` so toolchains without SVE2
  intrinsics gracefully fall back to NEON. The dispatch table in
  [`libvmaf/src/feature/ssimulacra2.c`](libvmaf/src/feature/ssimulacra2.c)
  is purely additive: NEON stays the fallback; SVE2 overrides only
  when the bit is set. Validated under `qemu-aarch64-static -cpu max`:
  dispatch reports `NEON=1 SVE2=1`, all 11 `test_ssimulacra2_simd`
  bit-exactness subtests pass byte-for-byte against the scalar
  reference (37/37 host x86 + 36/36 cross-aarch64 SVE2 suites green).
  Closes the "SVE2 deferred pending CI hardware" footnote in
  [Research-0016](docs/research/0016-ssimulacra2-iir-blur-simd.md) /
  [Research-0017](docs/research/0017-ssimulacra2-ptlr-simd.md) and
  backlog row T7-38.
- **Research-0030 — Phase-3b multi-seed validation (Gate 1 PASSED)**
  (fork-local doc): 5-seed retry of Phase-3b confirms the Subset B
  win is robust and *widens* with more seeds. Aggregate over
  5 seeds × 9 LOSO folds: canonical-6 mean PLCC 0.9633 (seed-mean-std
  0.0150) vs **Subset B mean PLCC 0.9807 (seed-mean-std 0.0019)** —
  **Δ = +0.0175**, 3.5× the Research-0027 stopping-rule threshold of
  +0.005. Subset B is also **8× more stable across seeds** than
  canonical-6 — likely because the consensus-7 feature set carries
  overlapping-but-not-identical signal, acting as an in-network
  regularizer. canonical-6's seed=4 ran to PLCC 0.9381 (3.6 pp below
  best seed); Subset B never strays from `[0.9783, 0.9833]`. **Gate 1
  cleanly passed**; Subset B advances to Gate 2 (KoNViD cross-corpus,
  ~3h extraction) and Gate 3 (Phase-3c lr-sweep on canonical-6).
  9.2σ-equivalent margin on the seed-only std means the headline win
  isn't seed-luck. New `--seeds 0,1,2,3,4` flag on
  [`ai/scripts/phase3_subset_sweep.py`](ai/scripts/phase3_subset_sweep.py)
  with seed-mean-std reporting in the summary.

- **Research-0029 — Phase-3b: StandardScaler retry validates
  broader-feature hypothesis** (fork-local doc): empirical retry
  of the Research-0028 negative result with per-fold StandardScaler.
  **Subset B (consensus-7 with redundancy pruning) clears the
  Research-0027 +0.005 PLCC stopping rule by 2× (+0.0106).** Mean
  LOSO PLCC over 9 folds: canonical-6 = 0.9677, Subset A
  (canonical+ssimulacra2) = 0.9669, **Subset B = 0.9783**, Subset
  C (full-21) = 0.9597. The Phase-3a failure was a preprocessing
  artefact, not a feature-signal artefact — `psnr_*`/`cambi`/
  `ciede2000` (range 0–100) had been dominating gradient updates
  over normalised features (range 0–1). With per-fold
  `(mean, std)` standardisation (statistics fit on train, applied
  to both train and val so no fold-leakage), the Research-0026
  hypothesis is confirmed. Two findings: (1) Subset B's feature
  composition (`adm2`, `adm_scale3`, `vif_scale2`, `motion2`,
  `ssimulacra2`, `psnr_hvs`, `float_ssim`) validates all four
  Research-0027 consensus features and the redundancy pruning
  recommendations; (2) Subset C (full-21) loses even with
  StandardScaler — including all features without pruning hurts
  the tiny `mlp_small` architecture. Three gates before
  `vmaf_tiny_v2.onnx` ships: multi-seed validation
  (`seed ∈ {0..4}`), KoNViD cross-corpus check, and Phase-3c
  `lr`-sweep on canonical-6 to verify the +0.0106 holds under
  matched preprocessing. New `--standardize` flag on
  [`ai/scripts/phase3_subset_sweep.py`](ai/scripts/phase3_subset_sweep.py).
  Driver shared with PR #188 (Research-0028); this PR adds the
  flag + the retry results.

- **Research-0028 — Phase-3 MLP subset sweep (negative-result
  digest)** (fork-local doc): empirical close of Research-0026
  Phase 3. The pre-registered Research-0027 stopping rule fires —
  Subset A (canonical-6 + ssimulacra2) lands LOSO mean PLCC 0.9655
  vs canonical-6 0.9845, a 0.019 *deficit* against the required
  +0.005 to advance. Subsets B (consensus-7) and C (full-21) also
  fail PLCC. **canonical-6 stays the default; no v2 model ships
  from this Phase.** Counterintuitive secondary finding: every
  subset cuts mean RMSE by ~40 % (canonical-6 RMSE 15.20 → A 9.13
  / B 8.91 / C 8.50), strongly suggesting the PLCC drop is a
  feature-scale-variance artefact (raw features fed to mlp_small
  without StandardScaler; psnr / cambi / ciede2000 dominate
  gradient updates by 2 orders of magnitude). Three follow-up
  experiments scoped: Phase-3b (StandardScaler retry), Phase-3c
  (`mlp_medium` / wider epoch sweep), Phase-3d (per-feature
  ablation in Subset C). New driver
  [`ai/scripts/phase3_subset_sweep.py`](ai/scripts/phase3_subset_sweep.py)
  ships with the Phase-3b/c/d follow-ups in mind. No code change
  to the trainer or sidecar; pure results document.

- **Research-0027 — Phase-2 feature correlation, MI, and importance
  results** (fork-local doc): empirical close of Research-0026
  Phase 2 on the full Netflix corpus (11 040 frame rows × 21 features
  extracted via PR #186 over ~118 min wall-clock). **Phase-3 GO
  signal is clear**: consensus top-10 across MI + LASSO + random-forest
  importance methods narrows to **4 features** (`adm2`, `adm_scale3`,
  `ssimulacra2`, `vif_scale2`) — two of which (`adm_scale3`,
  `ssimulacra2`) are NOT in the canonical `vmaf_v0.6.1` 6-tuple.
  11 redundant pairs at `|r| ≥ 0.95` reveal that the motion family
  is internally redundant (motion2 ↔ motion3 r=0.9926), VIF scales
  1/2/3 are pairwise redundant, and `vif_scale1 ↔ ssimulacra2`
  cross-family redundancy at r=0.9807 is the most surprising
  finding. Three Phase-3 candidate subsets recommended:
  **Subset A** (canonical-6 + ssimulacra2, conservative single-feature
  add); **Subset B** (consensus-7 = canonical core + adm_scale3 +
  ssimulacra2 + psnr_hvs + float_ssim, redundant scales dropped);
  **Subset C** (full-21, sanity ceiling). Stopping rules + per-subset
  Pareto criteria documented. Aligns with Research-0023 §5 (data
  axis) + Research-0025 (data resolved) + Research-0026 (feature axis
  framework) — this digest empirically validates the framework.
  No code change; pure results document.

- **Research-0025 — FoxBird outlier resolved via Netflix + KoNViD-1k
  combined training** (fork-local doc): empirical close of
  Research-0023 §5's open question. The canonical combined-trainer
  run (`mlp_small`, 30 epochs, val=Tennis + 10 % KoNViD-holdout) on
  the union of the Netflix Public 9-source corpus (9 690 frames) and
  the KoNViD-1k 1 200-clip parquet (270 051 frames) produces an
  ONNX whose FoxBird PLCC is **0.9936** (vs Netflix-only mlp_small
  baseline `vmaf_tiny_v1.onnx` at 0.9632) — a +3.04 percentage-point
  absolute gain on the canonical outlier. RMSE on FoxBird drops
  17.296 → 3.216 (5.4× lower); SROCC +0.0233. No regression on the
  Netflix-native sources (PLCC ≥ 0.998 on 7/9 clips). Validates
  PR #178 (KoNViD acquisition + loader) + PR #180 (combined trainer
  driver) infrastructure end-to-end. Closes Research-0023 §5
  unblocker question: KoNViD-1k is sufficient — no need to acquire
  BVI-DVC or AOM-CTC for this specific failure mode. Full numbers
  + caveats + next-experiment list in
  [`docs/research/0025-foxbird-resolved-via-konvid.md`](docs/research/0025-foxbird-resolved-via-konvid.md).
  No code change in this PR; docs-only.

- **Tiny-AI combined Netflix + KoNViD-1k trainer driver** (fork-local):
  new [`ai/train/train_combined.py`](ai/train/train_combined.py) feeds
  the union of `NetflixFrameDataset` (Netflix Public 9-source corpus)
  and `KoNViDPairDataset` (KoNViD-1k synthetic-distortion FR pairs)
  into the same `_build_model` + `_train_loop` + `export_onnx`
  pipeline that `ai/train/train.py` uses, so model factory + ONNX
  layout stay identical to the canonical baselines. Five validation
  modes: `netflix-source` (default; mirrors ADR-0203), `konvid-holdout`
  (deterministic 10 % of KoNViD clip keys, whole-clip granularity so
  no frame leakage), `netflix-source-and-konvid-holdout` (union of
  both), and the single-corpus `netflix-only` / `konvid-only`
  fallbacks. Addresses Research-0023 §5 (FoxBird-class outlier needs a
  broader content distribution; KoNViD-1k adds 1 200 UGC clips on top
  of the existing 70 Netflix dis-pairs). Documented in
  [`docs/ai/training.md`](docs/ai/training.md) "Combining KoNViD with
  the Netflix corpus" subsection. New 5-test smoke under
  [`ai/tests/test_train_combined_smoke.py`](ai/tests/test_train_combined_smoke.py)
  verifies the `--epochs 0` initial-ONNX path, deterministic
  KoNViD key splitter, and missing-data fallbacks without touching
  libvmaf or the real corpus.

- **Tiny-AI KoNViD-1k → VMAF-pair acquisition + loader bridge**
  (fork-local): direct follow-up to Research-0023 §5 (FoxBird-class
  variance needs a larger / more diverse corpus). New
  [`ai/scripts/konvid_to_vmaf_pairs.py`](ai/scripts/konvid_to_vmaf_pairs.py)
  takes raw KoNViD-1k `.mp4` sources from
  `$VMAF_DATA_ROOT/konvid-1k/`, generates a synthetic distorted
  variant per clip via libx264 CRF=35 round-trip (same recipe as
  the Netflix dis-pairs), runs libvmaf on each (ref, dis) pair to
  extract the 6 `vmaf_v0.6.1` model features + per-frame VMAF
  teacher score, and dumps to
  `ai/data/konvid_vmaf_pairs.parquet` (gitignored). Per-clip JSON
  caches under `$VMAF_TINY_AI_CACHE/konvid-1k/<key>.json` make
  re-runs idempotent. Smoke (5 clips) takes ~7 s wall; full
  1 200-clip run ~30 min on the `ryzen-4090` profile. New
  [`ai/train/konvid_pair_dataset.py::KoNViDPairDataset`](ai/train/konvid_pair_dataset.py)
  loader bridge mirrors `NetflixFrameDataset`'s interface
  (`feature_dim=6`, `numpy_arrays() → (X, y)`) so the existing
  LOSO trainer can ingest KoNViD pairs unchanged. `keep_keys`
  filter supports LOSO-style holdouts. 5 pytest cases under
  [`ai/tests/test_konvid_pair_dataset.py`](ai/tests/test_konvid_pair_dataset.py)
  cover shape, holdout filter, missing-column error, empty-after-
  filter, and torch tensor item shape — all green. Documented in
  [`docs/ai/training.md`](docs/ai/training.md) §"C1 (KoNViD-1k
  corpus)". Future work: a driver that concatenates Netflix +
  KoNViD `(X, y)` arrays and runs the LOSO sweep on the union.

- **Tiny-AI LOSO evaluation harness for `mlp_small`** (fork-local):
  new `ai/scripts/eval_loso_mlp_small.py` scores each of the 9
  leave-one-source-out fold checkpoints (`mlp_small_final.onnx`)
  on its own held-out clip, plus the two shipped baselines
  (`vmaf_tiny_v1.onnx`, `vmaf_tiny_v1_medium.onnx`) per-clip and
  on the all-clips concatenation. Reports per-fold +
  mean ± std PLCC / SROCC / RMSE to JSON and Markdown. Documented
  in [`docs/ai/loso-eval.md`](docs/ai/loso-eval.md). Numbers from
  the 2026-04-28 sweep on the Netflix corpus (LOSO mean PLCC
  0.9808 ± 0.0214, SROCC 0.9848 ± 0.0176) are captured in
  [Research Digest 0022](docs/research/0022-loso-mlp-small-results.md).
  Mirrors the per-fold accounting that MCP `compare_models` does
  for a single split, but respects the LOSO split structure
  without requiring 9 separate comparison calls.

- **`ssimulacra2_cuda` + `ssimulacra2_sycl` GPU twins
  (ADR-0206)** (fork-local): closes batch 3 part 7 across all
  three GPU backends. CUDA + SYCL extractors are direct ports of
  the [ADR-0201](docs/adr/0201-ssimulacra2-vulkan-kernel.md)
  Vulkan hybrid host/GPU pipeline — host runs YUV→linear-RGB,
  2×2 pyramid downsample, linear-RGB→XYB, and the per-pixel SSIM
  + EdgeDiff combine in double precision (verbatim ports of
  `ssimulacra2.c`); GPU runs the 3-plane elementwise multiply
  (`ssimulacra2_mul3`) and the separable 3-pole IIR Gaussian
  blur (`ssimulacra2_blur_h` / `ssimulacra2_blur_v`). The CUDA
  IIR fatbin is pinned with `-Xcompiler=-ffp-contract=off
  --fmad=false` via a per-kernel `cuda_cu_extra_flags` map in
  `libvmaf/src/meson.build`; SYCL relies on the existing
  `-fp-model=precise` for the same effect. Empirical: Netflix
  normal pair `max_abs_diff = 1.0e-6` on CUDA, both checkerboard
  pairs **bit-exact** (0.0). New extractor names:
  `ssimulacra2_cuda`, `ssimulacra2_sycl` (pair with
  `--backend cuda` / `--backend sycl` for exclusive GPU
  dispatch). New sources:
  `libvmaf/src/feature/cuda/ssimulacra2_cuda.{c,h}`,
  `libvmaf/src/feature/cuda/ssimulacra2/ssimulacra2_blur.cu`,
  `libvmaf/src/feature/cuda/ssimulacra2/ssimulacra2_mul.cu`,
  `libvmaf/src/feature/sycl/ssimulacra2_sycl.cpp`. With Vulkan
  ([ADR-0201](docs/adr/0201-ssimulacra2-vulkan-kernel.md))
  already in master and float_adm twins
  ([ADR-0202](docs/adr/0202-float-adm-cuda-sycl.md)) merging in
  parallel, batch 3 is now feature-complete on every GPU
  backend.

- **Backlog audit — untracked follow-up items (2026-04-28)**
  (fork-local, doc-only): one-shot audit of in-tree
  TODO / FIXME / "deferred" / "scaffold only" / "v2" mentions
  cross-referenced against the canonical backlog tracking
  surfaces (`.workingdir2/OPEN.md`, `.workingdir2/BACKLOG.md`,
  `docs/state.md`, `docs/rebase-notes.md`, ADR Decision /
  Consequences blocks, open GitHub issues / PRs). Output lands
  at [`docs/backlog-audit-2026-04-28.md`](docs/backlog-audit-2026-04-28.md);
  35 distinct clusters across ~1 270 raw hits. Section A lists
  14 untracked items needing decision (cambi v2 GPU c-values
  phase, cambi integration PR, Vulkan motion3 GPU gap,
  `picture_vulkan` luma-only chroma gap, `enable_lcs` GPU stub,
  QAT trainer hook, `docs/benchmarks.md` `TBD` cells, SVE2
  SIMD parity, etc.); Section B lists 5 partially-tracked items
  in ADRs / digests with no T-number (notably
  `iqa_convolve` AVX-512 ADR-0138 follow-up,
  `motion_v2` AVX2 srlv_epi64 audit). Section C lists 4
  resolved-but-stale comments (`libvmaf_vulkan.h` "scaffold
  only", `ssimulacra2.c` "SIMD variants are follow-up PRs",
  `meson.build` Vulkan blurbs) — comment-only fixes for the
  next session that touches each file. No source files
  modified.

- **cambi GPU feasibility spike — hybrid host/GPU verdict + Vulkan
  scaffold (ADR-0205)** (fork-local): closes the spike mandated by
  [ADR-0192](docs/adr/0192-gpu-long-tail-batch-3.md) §Consequences.
  Verdict: cambi is feasible on GPU as a **hybrid host/GPU pipeline**,
  mirroring [ADR-0201](docs/adr/0201-ssimulacra2-vulkan-kernel.md)'s
  precedent. GPU dispatch chain covers preprocessing + per-pixel
  derivative + 7×7 spatial-mask summed-area table + 2× decimate +
  3-tap separable mode filter (all integer + bit-exact w.r.t. CPU);
  the precision-sensitive `calculate_c_values` sliding-histogram
  pass + top-K spatial pooling stay on the host. Because the GPU
  buffers are bit-identical to the CPU's and the c-values phase
  runs the exact CPU code path, the v1 contract tightens to
  **`places=4`** (ADR-0192 carried `places=2` as a planning
  placeholder; ADR-0205 ratchets to the fork's canonical `places=4`
  baseline since the architecture forces ULP=0). Three classical re-formulations
  evaluated in [research digest 0020](docs/research/0020-cambi-gpu-strategies.md):
  (I) single-WG direct port — rejected, ~1/64 GPU utilisation;
  (II) parallel-scan reformulation — rejected for v1, materialises
  17 GiB intermediate at 4K; (III) direct per-pixel histogram —
  deferred to v2 as profile-driven perf polish, ~9× CPU bandwidth.
  Literature surveyed: Blelloch 1990, Sengupta 2007, Merrill &
  Grimshaw 2016. v1 LOC estimate: ~1230 (host glue ~700 + 6 shaders
  ~400 + wiring ~130). This PR ships the architecture sketch
  ([ADR-0205](docs/adr/0205-cambi-gpu-feasibility.md)) + research
  digest + reference shader scaffolds (`cambi_derivative.comp`,
  `cambi_decimate.comp`, `cambi_filter_mode.comp` under
  [`libvmaf/src/feature/vulkan/shaders/`](libvmaf/src/feature/vulkan/shaders/))
  + dormant `cambi_vulkan.c` host skeleton (not yet build-wired,
  matching ssimulacra2 precedent). After the integration follow-up
  PR lands, every registered feature extractor in the fork has at
  least one GPU twin (lpips remains ORT-delegated per
  [ADR-0022](docs/adr/0022-inference-runtime-onnx.md)) and the GPU
  long-tail terminus declared in
  [ADR-0192](docs/adr/0192-gpu-long-tail-batch-3.md) is closed.
- **Tiny-AI Netflix-corpus training prep (ADR-0203)** (fork-local):
  runnable loader + feature extractor + `vmaf_v0.6.1` distillation +
  PyTorch dataset + PLCC/SROCC/KROCC/RMSE eval harness + Lightning-
  style training entry point under `ai/data/` and `ai/train/`.
  Three architectures registered with the entry point: `linear`
  (7 params), `mlp_small` (257 params, default), `mlp_medium`
  (2 561 params). Default validation split holds out the
  `Tennis_24fps` source (1-source-out, content-disjoint). Per-clip
  JSON cache at `$VMAF_TINY_AI_CACHE` (default
  `~/.cache/vmaf-tiny-ai/<source>/<dis-stem>.json`) with atomic
  write-rename. Smoke command `python ai/train/train.py --epochs 0
  --assume-dims 16x16` works without the real corpus or a built
  `vmaf` binary so CI can verify the harness end-to-end. The first canonical training run on the full Netflix corpus
  (mlp_small, 30 epochs, val=Tennis) is documented in ADR-0203
  §"Training results"; final ONNX shipped at
  `model/tiny/vmaf_tiny_v1.onnx` (PLCC 0.9750 / SROCC 0.9792 vs
  vmaf_v0.6.1 distillation target). New
  [`docs/ai/training.md`](docs/ai/training.md) "C1 (Netflix corpus)"
  section + 25 unit tests under [`ai/tests/`](ai/tests/).
  Files: new
  [`ai/data/netflix_loader.py`](ai/data/netflix_loader.py),
  [`ai/data/feature_extractor.py`](ai/data/feature_extractor.py),
  [`ai/data/scores.py`](ai/data/scores.py),
  [`ai/train/dataset.py`](ai/train/dataset.py),
  [`ai/train/eval.py`](ai/train/eval.py),
  [`ai/train/train.py`](ai/train/train.py),
  [`ai/scripts/run_training.sh`](ai/scripts/run_training.sh).

- **GPU long-tail batch 3 part 2 — `float_ansnr_{vulkan,cuda,sycl}`
  extractors (T7-23 / ADR-0192 / ADR-0194)** (fork-local): closes
  the ANSNR matrix gap (was CPU-only float, no GPU twin). Single-
  dispatch GPU kernels apply the CPU's 3x3 ref filter
  ([`ansnr_tools.c::ansnr_filter2d_ref_s`](libvmaf/src/feature/ansnr_tools.c))
  and 5x5 dis filter (Netflix-tuned weights summing to 1.0,
  `/571`) inline from a 20×20 shared / SLM tile, then accumulate
  per-pixel `sig = ref_filtr²` and `noise = (ref_filtr - filtd)²`
  into per-WG float partials. Host reduces in `double` and applies
  the CPU formulas for `float_ansnr` and `float_anpsnr`. Edge-
  replicating mirror (`2*size - idx - 1`) matches CPU
  `ansnr_filter2d_s` — same divergence-from-motion footgun as
  motion_v2 (ADR-0193). Empirical floor on cross-backend gate
  fixture: `max_abs_diff = 6e-6` (8-bit, 48 frames) / `2e-6`
  (10-bit, 3 frames) on **all three backends with identical
  numbers** (Vulkan = CUDA = SYCL — strong evidence the kernel
  logic is correct). Files: new
  [`shaders/float_ansnr.comp`](libvmaf/src/feature/vulkan/shaders/float_ansnr.comp),
  [`float_ansnr_vulkan.c`](libvmaf/src/feature/vulkan/float_ansnr_vulkan.c),
  [`float_ansnr/float_ansnr_score.cu`](libvmaf/src/feature/cuda/float_ansnr/float_ansnr_score.cu),
  [`float_ansnr_cuda.c`](libvmaf/src/feature/cuda/float_ansnr_cuda.c),
  [`float_ansnr_sycl.cpp`](libvmaf/src/feature/sycl/float_ansnr_sycl.cpp).
  New `float_ansnr` lavapipe gate step in
  [`tests-and-quality-gates.yml`](.github/workflows/tests-and-quality-gates.yml)
  + `FEATURE_METRICS` entry in the cross-backend gate.

### Changed

- **Port Netflix upstream `314db130` — remove empty translation unit
  `libvmaf/src/feature/all.c`** (upstream port): the file had been
  reduced to includes + forward declarations + a `MIN` macro with no
  active call sites in the fork (`compute_*` entry points are reached
  via per-extractor TUs, not via `all.c`). Upstream removed it as
  dead code. Drops the file, the `meson.build` line that compiled it,
  and updates the trailing `// NOLINTNEXTLINE` comment in
  `offset.c:22` that listed `all.c` among the per-feature consumers.
  Build + 37 unit tests green after removal.

- **Audit Section C cleanup — refresh stale "scaffold only" / "follow-up
  PR" comments** (fork-local): four code surfaces still advertised work
  that has long since landed. Updated `libvmaf_vulkan.h` (top-level
  header doc-comment + the T7-29 zero-copy import block), the
  `ssimulacra2.c` SIMD blurb (ADR-0161 / 0162 / 0164 + GPU twins
  ADR-0201 / 0206), and the Vulkan blurbs in `libvmaf/src/meson.build`
  + `libvmaf/meson_options.txt`. Comment-only; no behavioural change.
  Closes Section C of `docs/backlog-audit-2026-04-28.md`.

- **Whole-codebase lint sweep — auto-fix subset (52% findings cleared)**
  (fork-local): post-T5-1 + docs-sweep follow-up. Baseline clang-tidy
  whole-codebase scan flagged 1533 unique findings across 84 files.
  This PR clears the auto-fixable categories —
  `readability-isolate-declaration`,
  `readability-braces-around-statements`, `modernize-use-nullptr`,
  `misc-const-correctness`, and `cert-err33-c` — leaving 736 manual
  / NOLINT-with-justification findings (widening-mul on stride math,
  function-size on SIMD reductions per ADR-0138, use-internal-linkage
  on cross-TU dispatch helpers, anonymous-namespace on C++ helpers,
  mt-unsafe `getenv`/`strerror`, etc.) for follow-up sweeps. Build
  green; all 37 meson unit tests pass after each of the four commits;
  clang-format clean on every touched file. Touched 68 files,
  ~1500-line diff.

- **CPU coverage matrix audit — closes 5 stale gaps in one pass
  (no code changes)** (fork-local): post-T7-19 verification
  exposed five matrix entries and backlog rows that were either
  already-shipped work or phantom rows from earlier audit
  snapshots. **T7-22** (`ms_ssim` per-scale SIMD) was already
  shipped via ADR-0138/0139/0140 — verified 3.2× wall-clock
  speedup vs `--cpumask 0xfffffffe`. **CAMBI scalar fallback**
  already exists at
  [`cambi.c:446-460`](libvmaf/src/feature/cambi.c). **motion_v2
  NEON** already exists at
  [`arm64/motion_v2_neon.c`](libvmaf/src/feature/arm64/motion_v2_neon.c).
  **integer `ansnr`** is a phantom row — no extractor is
  registered. **T7-21** (`psnr_hvs` AVX-512) closes as **AVX2
  ceiling** with empirical evidence (1.17× speedup of AVX2 vs
  scalar; AVX-512 widening would force a 2-block host batch
  without measurable payoff). Same verdict for deferred
  float_moment AVX-512. The CPU SIMD column is now closed. See
  [ADR-0180](docs/adr/0180-cpu-coverage-audit.md). Next gap
  surface: GPU long-tail (psnr / ssim / ssimulacra2 / cambi /
  psnr_hvs on CUDA / SYCL / Vulkan).

### Fixed

- **CI: Clang-Tidy job no longer fails on PRs that delete C/C++ files**
  (fork-local CI fix): `.github/workflows/lint-and-format.yml`'s
  `Clang-Tidy (Changed C/C++ Files)` step used `git diff --name-only`
  without `--diff-filter=d`, so a deleted file (e.g.
  `libvmaf/src/feature/all.c` in this PR's upstream port of
  `314db130`) was passed to `clang-tidy`, which then failed with
  `clang-diagnostic-error: no such file or directory`. Added
  `--diff-filter=d` to all three `git diff` invocations in the
  Clang-Tidy step (PR / push / push-with-fallback). No effect on
  Add/Modify paths.

- **Port Netflix upstream `798409e3` — null-deref on `prev_ref` update
  in pure CUDA pipelines** (upstream port, completes fork's earlier
  partial fix): the fork's `read_pictures_update_prev_ref` helper
  (`libvmaf.c:1593`) already carries the `if (ref && ref->ref)` guard
  for the main `vmaf_read_pictures` path, but the same shape was
  missing from `threaded_enqueue_one` (line 1057) and
  `threaded_read_pictures_batch` (line 1105) — both could deref a
  zero-initialised `ref_host` when every registered extractor carries
  `VMAF_FEATURE_EXTRACTOR_CUDA` and `translate_picture_device`
  early-returns without downloading. Patch mirrors lawrence's
  upstream fix (Netflix/vmaf `798409e3`, 2026-04-20). No behavioural
  change on non-CUDA pipelines; preserves the existing ADR-0123
  null-guard rationale across all three call sites.

- **T7-16: NVIDIA-Vulkan + SYCL `adm_scale2` 2.4e-4 boundary drift
  is gone — verified at `places=4` on master** (fork-local doc
  close, sister of T7-15): the cross-backend gate at PR #120
  surfaced a 2.4e-4 score offset on 1/48 frames for `adm_scale2`
  on Vulkan-on-NVIDIA-RTX-4090 (proprietary driver) and SYCL-on-
  Arc. Re-running on master with the same reproducer
  (`python3 scripts/ci/cross_backend_vif_diff.py --feature adm
  --backend vulkan --device 0`) reports `adm_scale2` max_abs_diff
  = 1e-6 (JSON `%f` print floor; ULP=0) on Vulkan device 0
  (RTX 4090, NVIDIA proprietary 595.58.3.0), Vulkan device 1
  (Arc Mesa anv 26.0.5), AND SYCL device 0 (Arc A380). All three
  pass `places=4` at 0/48 mismatches across all 5 ADM metrics.
  Same NVCC / driver / SYCL-runtime upgrade hypothesis as T7-15
  — no `adm_vulkan.c` / `adm_sycl.cpp` commits since PR #120
  (`7c5b63a2`). Verification-only close; the cross-backend gate
  locks the contract going forward.

- **T7-15: `motion_cuda` + `motion_sycl` 2.6e-3 drift vs CPU
  `integer_motion` is gone — verified bit-exact on master**
  (fork-local doc close, PR #172): the cross-backend gate at PR #120
  surfaced a 2.6e-3 score offset on 47/48 frames for both
  `motion_cuda` and `motion_sycl` on the Netflix golden 576×324
  pair. Re-running on master with the same reproducer
  (`python3 scripts/ci/cross_backend_vif_diff.py --feature motion
  --backend cuda`) reports `max_abs_diff=0.0` over all 48 frames
  at `places=8`; SYCL on Arc and Vulkan on Arc Mesa anv both show
  1e-6 (the JSON `%f` print-rounding floor; ULP=0). All three
  pass the existing `places=4` contract and the cross-backend
  gate now locks it going forward. No motion-kernel commits
  landed between PR #120 (`7c5b63a2`) and master, so the
  resolution is most likely the NVCC 13.x / NVIDIA-driver upgrade
  since PR #120 — the kernel source is unchanged but the emitted
  SASS now matches CPU rounding bit-exactly.

- **`libvmaf_vulkan.h` now installs under the prefix when
  `-Denable_vulkan=enabled`** (fork-local): `libvmaf/include/libvmaf/meson.build`
  had install gates for `is_cuda_enabled` and `is_sycl_enabled` but
  none for Vulkan, so `meson install` dropped `libvmaf_cuda.h` and
  `libvmaf_sycl.h` under `<prefix>/include/libvmaf/` but never
  `libvmaf_vulkan.h`. Symptom (lawrence, 2026-04-28): FFmpeg
  `configure` accepts `--enable-libvmaf-vulkan` and reports it as
  enabled, but only `vmaf_pre` and the regular `libvmaf` filter
  end up built — the `libvmaf_vulkan` filter is silently dropped
  because `check_pkg_config libvmaf_vulkan "libvmaf >= 3.0.0"
  libvmaf/libvmaf_vulkan.h vmaf_vulkan_state_init_external` (in
  ffmpeg-patches/0006) can't find the header. Fix: add an
  `is_vulkan_enabled` gate (handles the `feature` option's
  `enabled` and `auto` states), append `libvmaf_vulkan.h` to
  `platform_specific_headers` when active. Verified: a fresh
  `meson install --destdir /tmp/x` now drops the header alongside
  `libvmaf_cuda.h` and `libvmaf_sycl.h`. No CHANGELOG breakage for
  pre-existing CUDA/SYCL consumers — the install set is purely
  additive.

- **`--backend cuda` actually engages CUDA now (was silently CPU)**
  (fork-local): the CLI's `--backend cuda` selector previously set
  `gpumask = 1` intending it as a device pin, but
  `VmafConfiguration::gpumask` is a CUDA-*disable* bitmask per the
  public-header contract — `compute_fex_flags` enables the CUDA
  dispatch slot only when `gpumask == 0`. Net effect: every
  `--backend cuda` run from CLI initialised CUDA but then routed
  the actual feature extractors through the CPU path, producing
  bit-exact CPU-equivalent pools and no GPU speedup. Symptom on
  bench fixtures: identical fps + identical `vmaf` pool across
  `cpu` / `cuda` / `sycl` rows. Fix in
  [`libvmaf/tools/cli_parse.c`](libvmaf/tools/cli_parse.c) — set
  `gpumask = 0` (default) so the runtime engages CUDA after
  `vmaf_cuda_state_init` succeeds. `--gpumask=N --backend cuda`
  combinations preserve the user-supplied N. 5 new regression tests
  in [`libvmaf/test/test_cli_parse.c`](libvmaf/test/test_cli_parse.c)
  cover the four backends + the explicit-gpumask case. End-to-end
  smoke: `--backend cuda` on the Netflix golden 576×324 pair now
  emits 12 feature keys (CUDA extractor set) instead of 15 (CPU
  extractor set). The legacy
  `--gpumask=0 --no_sycl --no_vulkan` invocation continues to work
  as before. Documented in
  [`libvmaf/AGENTS.md`](libvmaf/AGENTS.md) §"Backend-engagement
  foot-guns" — same surface as PR #169.

- **`libvmaf.pc` Cflags leak in static-archive builds (ADR-0200)**
  (fork-local): bug-fix follow-up to ADR-0198. The
  `-include volk_priv_remap.h` flag was attached to
  `volk_dep.compile_args`; on `default_library=static` builds meson
  copies dependency `compile_args` into the generated `libvmaf.pc`
  `Cflags:` so downstream consumers can re-link against transitive
  static deps. Lawrence's BtbN-style fully-static FFmpeg build
  (cross-toolchain glibc-2.28, 2026-04-27) hit:

  ```text
  <command-line>: fatal error: /<libvmaf-build-dir>/subprojects/
    volk-vulkan-sdk-1.4.341.0/volk_priv_remap.h: No such file or directory
  compilation terminated.
  ```

  on FFmpeg's `check_func_headers aom/aom_codec.h` probe — the
  libvmaf-build-dir absolute path no longer existed after libvmaf
  was installed to `/opt/ffbuild/`. Fix: move the `-include` off
  `volk_dep.compile_args` and onto libvmaf's private `c_args`
  via `vmaf_cflags_common += ['-include', volk_priv_remap_h_path]`
  in `libvmaf/src/vulkan/meson.build`, where the path is pulled
  from `subproject('volk').get_variable('volk_priv_remap_h_path')`.
  `c_args:` on a `library()` are private to the target and do
  NOT leak into the generated pkg-config Cflags; the
  symbol-rename behaviour from ADR-0198 stays byte-for-byte
  identical. Post-fix `pkg-config --cflags libvmaf` returns
  `-I${includedir} -I${includedir}/libvmaf -DVK_NO_PROTOTYPES -pthread`
  on both shared and static builds. `nm libvmaf.a` still reports
  0 GLOBAL `vk*` and 719 `vmaf_priv_vk*`. See
  [ADR-0200](docs/adr/0200-volk-priv-remap-pkgconfig-leak-fix.md).

- **Volk / `vk*` symbol clash in fully-static link environments
  (ADR-0198)** (fork-local): follow-up to ADR-0185.
  `-Wl,--exclude-libs,ALL` only takes effect at the
  `gcc -shared` step that produces `libvmaf.so` — when libvmaf
  is built with `default_library=static -Denable_vulkan=enabled`,
  no link step happens and volk's full `vk*` PFN dispatcher
  table stays as STB_GLOBAL inside `libvmaf.a`. BtbN-style
  fully-static FFmpeg builds (lawrence's repro 2026-04-27 on a
  cross-toolchain glibc-2.28 build) that stitched
  `libvmaf.a + libvulkan.a` into one binary hit ~700 GNU-ld
  multi-definition errors:

  ```text
  ld: volk.c.o (symbol from plugin):
      multiple definition of `vkGetInstanceProcAddr';
      libvulkan.a(loader.c.o): first defined here
  ```

  Fixed by renaming volk's `vk*` symbols to `vmaf_priv_vk*`
  at the C preprocessor level via a force-included header
  generated from `volk.h`. The packagefile parses every
  `extern PFN_vkXxx vkXxx;` declaration, emits
  `#define vkXxx vmaf_priv_vkXxx` (784 entries for volk-1.4.341),
  and `-include`s the result on `volk.c` and every libvmaf TU
  pulling in `volk_dep`. Identical behaviour for shared and
  static — no per-build-mode meson branches. Verified: shared
  `nm -D libvmaf.so` reports 0 leaked `vk*` (unchanged from
  ADR-0185); static `nm libvmaf.a` reports 0 GLOBAL `vk*` (was
  ~700) and 719 `vmaf_priv_vk*`; BtbN-style
  `gcc -static main.c libvmaf.a libvulkan-stub.a` link succeeds;
  `test_vulkan_smoke` 10/10 pass on the renamed build (volk
  runtime `dlsym` dispatch still functional). See
  [ADR-0198](docs/adr/0198-volk-priv-remap-static-archive.md).

- **Hide volk / vk* symbols from libvmaf.so's public ABI
  (T7-31, ADR-0185)** (fork-local): when libvmaf is built with
  `-Denable_vulkan=enabled`, the bundled volk Vulkan-loader
  leaked ~30 `volk*` + the full `vk*` API into the .so's
  exported symbols. Static FFmpeg builds (BtbN-style
  cross-toolchain releases, glibc-2.28 environments, etc.)
  that link **both** libvmaf and libvulkan.a got GNU-ld
  multiple-definition errors at the final link:

  ```text
  /opt/ffbuild/lib/libvulkan.a(loader.c.o):
    multiple definition of `vkGetInstanceProcAddr';
  volk.c.o (symbol from plugin): first defined here
  ```

  Fixed by passing `-Wl,--exclude-libs,ALL` on the libvmaf.so
  link command in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build); gated
  off Darwin / Windows where the flag isn't supported (those
  linkers don't auto-export static-archive symbols anyway).
  Verified via `nm -D libvmaf.so` (zero `vk*` / `volk*` post-
  fix); smoke + end-to-end `psnr_vulkan` on Arc A380 unchanged
  (`psnr_y = 34.760779` matches PR #125's bit-exact reference).
  See [ADR-0185](docs/adr/0185-vulkan-hide-volk-symbols.md).

### Added

- **GPU long-tail batch 3 parts 1b + 1c — `motion_v2_cuda` +
  `motion_v2_sycl` extractors (T7-23 / ADR-0192 / ADR-0193)**
  (fork-local): closes batch 3 part 1 (and the integer_motion family
  GPU coverage). CUDA + SYCL twins of the Vulkan motion_v2 kernel
  shipped in PR #146. Both inherit the per-WG int64 SAD partial
  pattern, the raw-pixel ping-pong, and the edge-replicating mirror
  that diverges by one pixel from the motion kernel's mirror.
  - **CUDA**: nested 5x5 filter on the (prev - cur) diff loaded into
    a 20x20 shared int32 tile, warp-reduced via `__shfl_down_sync`,
    `atomicAdd` to a single int64 device buffer. New
    [`integer_motion_v2/motion_v2_score.cu`](libvmaf/src/feature/cuda/integer_motion_v2/motion_v2_score.cu)
    (~180 LOC PTX) +
    [`integer_motion_v2_cuda.c`](libvmaf/src/feature/cuda/integer_motion_v2_cuda.c)
    (~290 LOC host glue with submit/collect async stream pattern).
    Bit-exact vs CPU on 8-bit (48 frames) and 10-bit (3 frames) —
    `max_abs_diff = 0.0` on RTX 4090.
  - **SYCL**: separable V→H 5-tap filter on a 12x36 SLM tile, sub-
    group `reduce_over_group` then SLM cross-subgroup reduction →
    `atomic_ref::fetch_add` to int64. Self-contained (does NOT
    register with `vmaf_sycl_graph_register` because motion_v2
    needs the previous frame's raw ref pixels which the
    `shared_frame` luma buffer doesn't preserve across calls — same
    pattern as ciede_sycl). New
    [`integer_motion_v2_sycl.cpp`](libvmaf/src/feature/sycl/integer_motion_v2_sycl.cpp)
    (~330 LOC). Bit-exact vs CPU on Intel Arc A380 + oneAPI 2025.3.
- **GPU long-tail batch 3 part 1a — `motion_v2_vulkan` extractor
  (T7-23 / ADR-0192 / ADR-0193)** (fork-local): first kernel of
  batch 3, the smallest fork-local Vulkan kernel by far (~280 LOC
  GLSL + ~360 LOC host glue). Single-dispatch design exploits
  convolution linearity:
  `SAD(blur(prev), blur(cur)) == sum(|blur(prev - cur)|)` so the
  kernel reads both `prev_ref` and `cur_ref` planes, computes the
  full V→H separable 5-tap Gaussian over the signed diff in one
  dispatch, and accumulates `|h|` directly into per-WG `int64`
  partials. No blurred-state buffer (vs `motion_vulkan`'s
  ping-pong) — replaced by a smaller raw-pixel ping-pong of
  `ref_buf[2]`. Bit-exact vs CPU on 8-bit and 10-bit (max_abs_diff
  = 0.0 across 48 frames at 576×324, Intel Arc A380 + Mesa anv);
  cross-backend gate runs at `places=4`. Mirror padding
  **diverges** from `motion.comp` — CPU `integer_motion_v2.c`
  uses edge-replicating reflective mirror (`2*size - idx - 1`)
  while `integer_motion.c::edge_8`/`edge_16` use the non-
  replicating variant (`2*(size-1) - idx`); difference is one
  pixel at the boundary and the GLSL must follow the CPU it's
  porting. CUDA + SYCL twins follow as ADR-0192 batch 3 parts
  1b + 1c. New `motion_v2` lavapipe lane step + `FEATURE_METRICS`
  entry in
  [`scripts/ci/cross_backend_vif_diff.py`](scripts/ci/cross_backend_vif_diff.py).
- **GPU long-tail batch 3 scope (T7-23 / ADR-0192)** (fork-local,
  doc-only PR #145): scoping ADR for batch 3 — closes every
  remaining metric gap on the matrix. Group A (no GPU twin yet):
  `integer_motion_v2`, `float_ansnr`, `ssimulacra2`, `cambi`.
  Group B (float twins of int kernels already on GPU):
  `float_psnr` / `float_motion` / `float_vif` / `float_adm`,
  kept native (not aliased to the int kernels — different input
  domains). 21+ PRs to close (7 metrics × 3 backends). After
  batch 3, every registered feature extractor in the fork has at
  least one GPU twin (`lpips` remains ORT-delegated per ADR-0022).
- **GPU long-tail batch 3 part 3 — `float_psnr_{vulkan,cuda,sycl}`
  extractors (T7-23 / ADR-0192 / ADR-0195)** (fork-local): first
  Group B float twin from ADR-0192. Smallest GPU twin in the
  long-tail (~120 LOC GLSL + ~110 LOC PTX + ~150 LOC SYCL). Single-
  dispatch kernels — no halo, no shared tile — compute per-pixel
  `(ref - dis)²` in float, reduce per-WG via sub-group + SLM, host
  accumulates in `double` and applies CPU formula
  `MIN(10·log10(peak² / max(noise / (w·h), 1e-10)), psnr_max)`.
  **Empirically bit-exact** vs CPU on all three backends, both
  8-bit (48 frames) and 10-bit (3 frames) — `max_abs_diff = 0.0e+00`
  everywhere on Intel Arc A380 (Vulkan + SYCL) and NVIDIA RTX 4090
  (CUDA). Float-domain kernel too simple to drift; host-side
  `double` reduction absorbs any per-WG ULP noise. Drive-by docs
  fix: features.md row claimed `float_psnr_y / _cb / _cr` plane
  outputs which were wrong — the CPU extractor only emits a single
  luma `float_psnr` score; corrected in this PR. New
  [`shaders/float_psnr.comp`](libvmaf/src/feature/vulkan/shaders/float_psnr.comp),
  [`float_psnr_vulkan.c`](libvmaf/src/feature/vulkan/float_psnr_vulkan.c),
  [`float_psnr/float_psnr_score.cu`](libvmaf/src/feature/cuda/float_psnr/float_psnr_score.cu),
  [`float_psnr_cuda.c`](libvmaf/src/feature/cuda/float_psnr_cuda.c),
  [`float_psnr_sycl.cpp`](libvmaf/src/feature/sycl/float_psnr_sycl.cpp).
  New `float_psnr` lavapipe gate step + `FEATURE_METRICS` entry.
- **GPU long-tail batch 3 part 4 — `float_motion_{vulkan,cuda,sycl}`
  extractors (T7-23 / ADR-0192 / ADR-0196)** (fork-local): second
  Group B float twin from ADR-0192. Float-domain twin of
  `integer_motion`'s GPU kernels: same V→H 5-tap separable Gaussian
  blur (FILTER_5_s float weights summing to ~1.0), same 2-buffer
  ping-pong of blurred refs, same per-WG float SAD partials + host
  `double` reduction. `motion = sad / (w·h)`,
  `motion2 = min(prev, cur)` emitted at `index - 1` (delayed-by-one
  pattern, matches CPU `float_motion.c::extract`). Mirror padding:
  skip-boundary `2*(sup-1) - idx` matches CPU
  `convolution_internal.h::convolution_edge_s` (NOT motion_v2's
  edge-replicating). **Identical** `max_abs_diff = 3e-6` (8-bit, 48
  frames) / `1e-6` (10-bit, 3 frames) across all three backends —
  strong correctness signal (any algebraic bug would produce
  backend-specific drift). New
  [`shaders/float_motion.comp`](libvmaf/src/feature/vulkan/shaders/float_motion.comp),
  [`float_motion_vulkan.c`](libvmaf/src/feature/vulkan/float_motion_vulkan.c),
  [`float_motion/float_motion_score.cu`](libvmaf/src/feature/cuda/float_motion/float_motion_score.cu),
  [`float_motion_cuda.{c,h}`](libvmaf/src/feature/cuda/float_motion_cuda.c),
  [`float_motion_sycl.cpp`](libvmaf/src/feature/sycl/float_motion_sycl.cpp).
  New `float_motion` lavapipe gate step + `FEATURE_METRICS` entry.
- **GPU long-tail batch 3 part 5 — `float_vif_{vulkan,cuda,sycl}`
  extractors (T7-23 / ADR-0192 / ADR-0197)** (fork-local): third
  Group B float twin from ADR-0192. 4-scale Gaussian pyramid with
  separable `{17, 9, 5, 3}`-tap filters at the default
  `vif_kernelscale = 1.0` (other kernelscale values rejected at
  init for v1 — production uses 1.0 exclusively). 7 dispatches per
  frame (4 compute + 3 decimate). CPU's `VIF_OPT_HANDLE_BORDERS`
  branch: per-scale dims = prev/2 (no border crop), decimation
  samples at `(2*gx, 2*gy)` with mirror padding handling taps near
  the edge. **Mirror-asymmetry fix:** CPU has two H-mirror formulas
  that differ by 1 —
  [`vif_mirror_tap_h`](libvmaf/src/feature/vif_tools.c) returns
  `2 * extent - idx - 1` (scalar fallback only), while
  [`convolution_edge_s`](libvmaf/src/feature/common/convolution_internal.h)
  returns `2 * width - idx - 2` (AVX2 production border path). The
  GPU follows the AVX2 form because that's what production runs;
  using scalar's form drifted `5.46e-4` at scale 1, the AVX2 form
  closes that to `1.4e-5`. **places=4 across all 4 scales,
  identical numbers across all three backends** (`1e-6 / 1.4e-5
  / 1.8e-5 / 3.7e-5` at 8-bit, tighter at 10-bit on Intel Arc A380,
  Mesa anv, NVIDIA RTX 4090, oneAPI 2025.3). New
  [`shaders/float_vif.comp`](libvmaf/src/feature/vulkan/shaders/float_vif.comp),
  [`float_vif_vulkan.c`](libvmaf/src/feature/vulkan/float_vif_vulkan.c),
  [`float_vif/float_vif_score.cu`](libvmaf/src/feature/cuda/float_vif/float_vif_score.cu),
  [`float_vif_cuda.{c,h}`](libvmaf/src/feature/cuda/float_vif_cuda.c),
  [`float_vif_sycl.cpp`](libvmaf/src/feature/sycl/float_vif_sycl.cpp).
  New `float_vif` lavapipe gate step + `FEATURE_METRICS` entry at
  places=4.
- **Tiny-AI training scaffold for the Netflix VMAF corpus (ADR-0199)**
  (fork-local): scaffold-only PR preparing the tiny-AI training pipeline for
  the local Netflix VMAF corpus (9 ref / 70 distorted YUVs at
  `.workingdir2/netflix/`). Ships `docs/ai/training-data.md` with the corpus
  path convention and `--data-root` loader API; `docs/adr/0199-*.md` with
  the architecture-choice space and distillation-vs-from-scratch alternatives
  table; `docs/research/0019-tiny-ai-netflix-training.md` surveying VMAF
  training methodology and distillation literature; and an MCP end-to-end
  smoke test (`mcp-server/vmaf-mcp/tests/test_smoke_e2e.py`) that exercises
  the `vmaf_score` JSON-RPC tool against the Netflix golden fixture. No
  training runs; no Netflix golden assertions modified. Follow-up PR will
  select architecture and run training.
- **GPU long-tail batch 3 parts 6b + 6c — `float_adm_cuda` +
  `float_adm_sycl` extractors (T7-23 / ADR-0192 / ADR-0202)**
  (fork-local): CUDA + SYCL twins of the Vulkan kernel shipped in
  PR #154 / [ADR-0199](docs/adr/0199-float-adm-vulkan.md). Direct
  ports of the four-stage / four-scale Vulkan pipeline: same
  `-1` mirror form, same fused stage 3 with cross-band CM
  threshold, same per-scale 6-slot WG partials reduced on the
  host in `double`. New files:
  [`libvmaf/src/feature/cuda/float_adm/float_adm_score.cu`](libvmaf/src/feature/cuda/float_adm/float_adm_score.cu),
  [`libvmaf/src/feature/cuda/float_adm_cuda.{c,h}`](libvmaf/src/feature/cuda/float_adm_cuda.c),
  [`libvmaf/src/feature/sycl/float_adm_sycl.cpp`](libvmaf/src/feature/sycl/float_adm_sycl.cpp).
  Two precision-critical fixes from bring-up: (1) `--fmad=false`
  on the `float_adm_score` fatbin via a new per-kernel
  `cuda_cu_extra_flags` dict in `meson.build` — NVCC's default
  FMA contraction in the angle-flag dot product cascaded
  through the cube reductions and pushed scale-3 / adm2 past
  `places=4` (3.6e-4 max_abs vs CPU before fix). Scoped to this
  one kernel; integer ADM keeps its existing FMA-on path.
  (2) Parent-LL dimension trap — stage 0 at `scale > 0` clamps
  against the **parent's LL output dims** (`scale_w/h[scale]`),
  NOT the parent's full-resolution image dims
  (`scale_w/h[scale - 1]`). Verified `max_abs_diff ≤ 6e-6`
  across all five outputs (adm2, adm_scale0..3) on the Netflix
  normal pair via `cross_backend_vif_diff.py --backend cuda
  --feature float_adm --places 4` (0/48 mismatches);
  checkerboard 1px is bit-exact. SYCL gates on lavapipe-equivalent
  CI lanes already cover the `float_adm` feature surface from
  PR #154 ([ADR-0199](docs/adr/0199-float-adm-vulkan.md)).
- **GPU long-tail batch 3 part 6 — `float_adm_vulkan` extractor
  (T7-23 / ADR-0192 / ADR-0199)** (fork-local): sixth and final
  Group B float twin. Vulkan compute kernel for the float ADM
  feature extractor. Float twin of `integer_adm_vulkan`
  ([ADR-0178](docs/adr/0178-integer-adm-vulkan.md)) — same 4-stage
  / 4-scale wave-of-stages design (16 pipelines) but with float
  buffers and host-side `double` accumulation. New files:
  [`libvmaf/src/feature/vulkan/float_adm_vulkan.c`](libvmaf/src/feature/vulkan/float_adm_vulkan.c),
  [`libvmaf/src/feature/vulkan/shaders/float_adm.comp`](libvmaf/src/feature/vulkan/shaders/float_adm.comp).
  Mirror-asymmetry status: float_adm has NO trap analogous to
  [ADR-0197](docs/adr/0197-float-vif-gpu.md) — both the scalar
  `adm_dwt2_s` and the AVX2 `float_adm_dwt2_avx2` consume the same
  `dwt2_src_indices_filt_s` index buffer (`2 * sup - idx - 1` for
  both axes); the GPU follows that. `places=4` cross-backend
  contract on the lavapipe lane (new step in
  [`tests-and-quality-gates.yml`](.github/workflows/tests-and-quality-gates.yml)).
  `scripts/ci/cross_backend_vif_diff.py` gains a `float_adm` entry
  in `FEATURE_METRICS`. CUDA + SYCL twins land in a focused
  follow-up PR.
- **GPU long-tail batch 3 part 7 — `ssimulacra2_vulkan` extractor
  (T7-23 / ADR-0192 / ADR-0201)** (fork-local): Vulkan twin of the
  CPU `ssimulacra2` extractor with a hybrid host/GPU pipeline. The
  GPU runs the IIR blur (separable Charalampidis 2016 3-pole, one
  workgroup per row / per column) and the 3-plane elementwise
  product. Host runs YUV → linear-RGB, 2×2 pyramid downsample,
  linear-RGB → XYB (bit-exact port of `linear_rgb_to_xyb`), and
  the per-pixel SSIM + EdgeDiff combine in double precision over
  the GPU-blurred mu/sigma buffers. Empirical CPU-vs-Vulkan on
  Netflix normal pair (576×324, 48 frames): pooled `ssimulacra2`
  `max_abs_diff = 1.81e-7` (mean 3.65e-8, P95 1.56e-7). The
  cross-backend gate runs at `places=4` — matching the rest of
  the Vulkan VIF/MS-SSIM family. ADR-0201 §Precision investigation
  documents the five-tactic measurement chain that drove the
  contract from a `places=1` first-iteration shipping condition
  to `places=4`. CUDA + SYCL twins follow in a separate PR. See
  [ADR-0201](docs/adr/0201-ssimulacra2-vulkan-kernel.md).
- **Port Netflix upstream b949cebf — feature/motion: port several feature
  extractor options** (upstream port, Research-0024 Strategy A):
  Verbatim port of Netflix/vmaf commit b949cebf (Kyle Swanson, 2026-04-27).
  Adds 8 new options to `float_motion` and 3 missing options to
  `integer_motion`: `motion_blend_factor` (default 1.0, no-op),
  `motion_blend_offset` (default 40.0), `motion_fps_weight` (default 1.0,
  no-op), `motion_add_scale1` (default false, no-op), `motion_filter_size`
  (default 5, no-op — preserves FILTER_5_s), `motion_add_uv` (default
  false, no-op), `motion_max_val` (default 10000.0, no-op). Adds
  `VMAF_feature_motion3_score` and `VMAF_integer_feature_motion3_score`
  outputs. Adds `FILTER_3_s`, `FILTER_5_NO_OP_s`, and
  `DEFAULT_MOTION_FILTER_SIZE` to `motion_tools.h`. Adds `motion_decimate`
  parameter to `compute_motion()` and `motion_add_scale1` to
  `vmaf_image_sad_c()`. Also ports `picture_copy()` channel-parameter
  change (from d3647c73) required as prerequisite. All defaults are no-ops:
  integer_motion2 and float_motion2 scores are bit-identical to pre-port
  baseline. Netflix golden assertions unaffected.

- **GPU long-tail batch 2 parts 3b + 3c — `psnr_hvs_cuda` +
  `psnr_hvs_sycl` extractors (T7-23 / ADR-0188 / ADR-0191)**
  (fork-local): closes batch 2 part 3 (and batch 2 entirely).
  CUDA + SYCL twins of the Vulkan psnr_hvs kernel shipped in
  PR #143. Both inherit the per-plane single-dispatch design
  — one work-group per output 8×8 block (step=7), 64 threads
  per WG, cooperative load + thread-0-serial reductions
  matching CPU's exact i,j summation order (locks float
  bit-order to `calc_psnrhvs`). Three dispatches per frame
  (Y / Cb / Cr); host accumulates per-plane partials in float
  matching CPU's `ret` register pattern, then
  `10·log10(1/score)` per plane and combined
  `psnr_hvs = 0.8·Y + 0.1·(Cb + Cr)`.
  - **CUDA** (~270 LOC PTX in
    [`integer_psnr_hvs/psnr_hvs_score.cu`](libvmaf/src/feature/cuda/integer_psnr_hvs/psnr_hvs_score.cu)
    + ~330 LOC host in
    [`integer_psnr_hvs_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_psnr_hvs_cuda.c)):
    picture_copy host-side via `cuMemcpy2DAsync` D2H per
    plane (honours pitched `cuMemAllocPitch` device buffer —
    same fix as ms_ssim_cuda PR #142). Per-plane state arrays
    (`d_ref[3] / d_dist[3] / d_partials[3]`) + pinned host
    staging.
  - **SYCL** (~420 LOC, single TU
    [`integer_psnr_hvs_sycl.cpp`](libvmaf/src/feature/sycl/integer_psnr_hvs_sycl.cpp)):
    self-contained submit/collect (mirrors `ms_ssim_sycl`).
    Host-pinned USM staging carries the picture_copy-
    normalised float planes per plane. Inline picture_copy
    clone (libvmaf's `picture_copy` hardcodes plane 0).
    fp64-free.
  - **Verification**: 48 frames at 576×324 vs CPU scalar.
    **CUDA on NVIDIA RTX 4090** → `max_abs = 8.3e-5` (Y plane,
    same floor as Vulkan), `0/48 places=3 mismatches`.
    **SYCL on Intel Arc A380** → `max_abs = 1.0e-6` across all
    four metrics, `0/48 places=4 mismatches`. SYCL is the
    only backend that hits `places=4` on psnr_hvs — icpx's
    `-fp-model=precise` flag (project-wide SYCL strict-FP
    setting) produces tighter CPU-matching precision than
    nvcc default or glslc default. Investigation in
    [ADR-0191 §"Why not places=4"](docs/adr/0191-psnr-hvs-vulkan.md)
    documents what was tried for the CUDA + Vulkan paths;
    `--fmad=false` was tested for CUDA and didn't help,
    ruling out FMA fusion as the dominant drift source.
  - **v1 limitation**: rejects YUV400P (no chroma) and
    `bpc > 12` (matches CPU). 3 dispatches/frame at 1080p.

- **GPU long-tail batch 2 part 3a — `psnr_hvs_vulkan` extractor
  (T7-23 / ADR-0188 / ADR-0191)** (fork-local): first DCT-based
  GPU kernel in the fork. Vulkan twin of the active CPU
  `psnr_hvs` extractor. Per-plane single-dispatch design — one
  workgroup per output 8×8 block (step=7 sliding window),
  64 threads per workgroup. Cooperative load + per-quadrant
  reductions + scalar Xiph integer DCT (`od_bin_fdct8x8`,
  lifting + RSHIFT, `int32` arithmetic byte-for-byte against
  `third_party/xiph/psnr_hvs.c`) + per-coefficient masking +
  `subgroupAdd` per-block float partial. Host accumulates
  partials in `double` per plane, applies
  `score / pixels / samplemax²` then `10·log10(1/score)` per
  plane. Combined `psnr_hvs = 0.8·Y + 0.1·(Cb + Cr)`. New
  [`libvmaf/src/feature/vulkan/shaders/psnr_hvs.comp`](libvmaf/src/feature/vulkan/shaders/psnr_hvs.comp)
  + [`libvmaf/src/feature/vulkan/psnr_hvs_vulkan.c`](libvmaf/src/feature/vulkan/psnr_hvs_vulkan.c)
  (~540 LOC host). Three pipelines (one per plane, baked-in
  PLANE + BPC specialisation constants); CSF tables per plane
  baked into shader as `const float[64]` arrays. Rejects
  YUV400P (no chroma) and `bpc > 12` (matches CPU). Empirical:
  48 frames at 576×324 on **Intel Arc A380** vs CPU scalar —
  `max_abs = 8.2e-5` across all four metrics
  (`psnr_hvs_y / _cb / _cr / psnr_hvs`), `0/48 places=3
  mismatches`. Gate runs at `places=3` (better than ADR-0188's
  `places=2` floor). New CI step `psnr_hvs cross-backend diff
  (CPU vs Vulkan/lavapipe)` on the lavapipe lane. CUDA + SYCL
  twins follow as batch 2 parts 3b + 3c.

- **GPU long-tail batch 2 parts 2b + 2c — `float_ms_ssim_cuda` +
  `float_ms_ssim_sycl` extractors (T7-23 / ADR-0188 / ADR-0190)**
  (fork-local): closes batch 2 part 2. CUDA + SYCL twins of the
  Vulkan ms_ssim kernel shipped in PR #141. Both inherit the
  three-kernel design — decimate (9-tap 9/7 biorthogonal LPF +
  2× downsample), horiz (11-tap separable Gaussian over five
  SSIM stats), vert+lcs (vertical 11-tap + per-pixel l/c/s +
  per-WG / per-block float partials × 3). Host accumulates
  partials in `double` per scale and applies the Wang weights
  for the `MS-SSIM = ∏_i l[i]^α[i]·c[i]^β[i]·s[i]^γ[i]`
  combine.
  - **CUDA** (~210 LOC PTX in
    [`integer_ms_ssim/ms_ssim_score.cu`](libvmaf/src/feature/cuda/integer_ms_ssim/ms_ssim_score.cu)
    + ~470 LOC host in
    [`integer_ms_ssim_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_ms_ssim_cuda.c)):
    picture_copy normalisation runs on the host (uint → float in
    `[0, 255]`) via `cuMemcpy2DAsync` D2H of the pitched device
    plane into a contiguous pinned host buffer, then H2D upload
    to pyramid level 0. Surfaced one bring-up bug: the device
    plane is allocated by `cuMemAllocPitch` with `stride[0] ≥
    width·bpc` — naïve `cuMemcpyDtoHAsync` of `width·height·bpc`
    bytes mis-copies row N≥1 because it ignores the device
    pitch. Fix: 2D copy honouring `srcPitch = ref_pic->stride[0]`
    + `dstPitch = width·bpc` produces the contiguous host buffer
    `picture_copy` expects.
  - **SYCL** (~510 LOC, single TU
    [`integer_ms_ssim_sycl.cpp`](libvmaf/src/feature/sycl/integer_ms_ssim_sycl.cpp)):
    self-contained submit/collect (does NOT register with
    `vmaf_sycl_graph_register` — same rationale as ssim_sycl).
    Host-pinned USM staging carries the picture_copy-normalised
    float planes; `nd_range<2>` vert+lcs kernel uses
    `sycl::reduce_over_group` × 3 for per-WG partials.
    fp64-free (Intel Arc A380).
  - **Verification**: 48 frames at 576×324 vs CPU scalar —
    `max_abs = 1.0e-6`, `0/48 places=4 mismatches` on **NVIDIA
    RTX 4090** (CUDA) and **Intel Arc A380** (SYCL). Same
    precision floor as `ms_ssim_vulkan` (PR #141).
  - **v1 limitation** (same as ms_ssim_vulkan): no `enable_lcs`
    — 15 extra per-scale metrics deferred to a focused
    follow-up.

- **GPU long-tail batch 2 part 2a — `float_ms_ssim_vulkan`
  extractor (T7-23 / ADR-0188 / ADR-0190)** (fork-local):
  Wang multi-scale SSIM on Vulkan. 5-level pyramid built via
  9-tap 9/7 biorthogonal LPF + 2× downsample
  ([`ms_ssim_decimate.comp`](libvmaf/src/feature/vulkan/shaders/ms_ssim_decimate.comp),
  matches `ms_ssim_decimate_scalar` byte-for-byte). Per-scale
  SSIM compute via a variant of `ssim.comp` that emits **three**
  per-WG partials (`l, c, s`) instead of a single combined
  SSIM
  ([`ms_ssim.comp`](libvmaf/src/feature/vulkan/shaders/ms_ssim.comp)).
  Host accumulates partials in `double` per scale, applies the
  Wang weights `α/β/γ` (matches `ms_ssim.c::g_alphas/g_betas/
  g_gammas` byte-for-byte) for the
  `MS-SSIM = ∏_i l[i]^α[i]·c[i]^β[i]·s[i]^γ[i]` combine on host.
  New
  [`libvmaf/src/feature/vulkan/ms_ssim_vulkan.c`](libvmaf/src/feature/vulkan/ms_ssim_vulkan.c)
  (~700 LOC). Min-dim guard mirrors
  [ADR-0153](docs/adr/0153-float-ms-ssim-min-dim-netflix-1414.md)
  (176×176 minimum). v1 does **not** implement `enable_lcs` (15
  extra per-scale metrics) — deferred to a focused follow-up.
  Surfaced one bring-up bug: `ssim_variance_scalar` clamps
  `ref_sigma_sqd / cmp_sigma_sqd` to `MAX(0, ...)` before the
  sqrt at line 165 of `iqa/ssim_tools.c`; missing this clamp
  produces NaN at scale 0 when float ULP errors push variances
  slightly negative on flat regions. Empirical: 48 frames at
  576×324 on **Intel Arc A380** vs CPU scalar — `max_abs =
  2.0e-6`, `0/48 places=4 mismatches`. New CI step
  `float_ms_ssim cross-backend diff (CPU vs Vulkan/lavapipe)`
  on the lavapipe lane. CUDA + SYCL twins follow as batch 2
  parts 2b + 2c.

- **GPU long-tail batch 2 parts 1b + 1c — `float_ssim_cuda` +
  `float_ssim_sycl` extractors (T7-23 / ADR-0188 / ADR-0189)**
  (fork-local): closes batch 2 part 1. CUDA + SYCL twins of
  the Vulkan ssim kernel shipped in PR #139. Both inherit the
  two-pass design (horizontal 11-tap separable Gaussian → 5
  intermediate float buffers, then vertical 11-tap + per-pixel
  SSIM combine + per-WG / per-block float partial sums; host
  accumulates in `double`).
  - **CUDA** (~210 LOC PTX in
    [`integer_ssim/ssim_score.cu`](libvmaf/src/feature/cuda/integer_ssim/ssim_score.cu)
    + ~340 LOC host in
    [`integer_ssim_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_ssim_cuda.c)):
    `picture_copy` normalisation (uint → float / scaler) inlined
    in the horizontal kernel — no extra host-side conversion
    since picture_cuda already uploaded the raw uint plane.
    Per-block float partials reduced on host in `double`.
  - **SYCL** (~370 LOC, single TU
    [`integer_ssim_sycl.cpp`](libvmaf/src/feature/sycl/integer_ssim_sycl.cpp)):
    self-contained submit/collect (does NOT register with
    `vmaf_sycl_graph_register` — `shared_frame` is luma-only
    packed at uint width and SSIM needs `picture_copy`-normalised
    float planes). Host-pinned USM staging carries the
    normalised ref/cmp; `nd_range<2>` vertical kernel with
    `sycl::reduce_over_group` builds per-WG partials. fp64-free
    (Intel Arc A380 lacks native fp64 — same constraint as
    ciede_sycl).
  - **Verification**: 48 frames at 576×324 vs CPU scalar —
    `max_abs = 1.0e-6`, `0/48 places=4 mismatches` on **NVIDIA
    RTX 4090** (CUDA) and **Intel Arc A380** (SYCL). Same
    precision floor as `ssim_vulkan` (PR #139). Comfortably
    under `places=4` threshold (5e-5).
  - **v1 limitation** (same as ssim_vulkan): GPU paths support
    `scale=1` only — auto-detect rejects `scale > 1` with
    `-EINVAL`. Production 1080p needs
    `--feature float_ssim_{cuda,sycl}:scale=1` pinned (or
    smaller input). GPU-side decimation is a v2 follow-up.
  ms_ssim (batch 2 part 2) follows next.

- **GPU long-tail batch 2 part 1a — `float_ssim_vulkan`
  extractor (T7-23 / ADR-0188 / ADR-0189)** (fork-local):
  Vulkan twin of the active CPU `float_ssim`. **Two-dispatch
  design** — horizontal 11-tap separable Gaussian over
  ref / cmp / ref² / cmp² / ref·cmp into five intermediate
  float buffers, then vertical 11-tap + per-pixel SSIM
  combine + per-WG float partial sums. Host accumulates
  partials in `double` and divides by `(W-10)·(H-10)`
  (matches CPU's `iqa_ssim` valid-region averaging). 11-tap
  Gaussian weights baked into GLSL byte-for-byte from
  `g_gaussian_window_h` in `iqa/ssim_tools.h`. picture_copy
  host-side normalises uint sample → float `[0, 255]` before
  upload (matches `float_ssim.c::extract`). New
  [`libvmaf/src/feature/vulkan/shaders/ssim.comp`](libvmaf/src/feature/vulkan/shaders/ssim.comp)
  + [`libvmaf/src/feature/vulkan/ssim_vulkan.c`](libvmaf/src/feature/vulkan/ssim_vulkan.c)
  (~510 LOC host). **v1 limitation**: GPU path supports
  `scale=1` only — auto-detect rejects `scale > 1` with
  `-EINVAL`; production 1080p needs
  `--feature float_ssim_vulkan:scale=1` pinned (or smaller
  input). Cross-backend gate fixture (576×324) auto-resolves
  to `scale=1`. GPU-side decimation is a v2 follow-up.
  Empirical: 48 frames at 576×324 on **Intel Arc A380** vs
  CPU scalar — `max_abs = 1.0e-6`, `0/48 places=4
  mismatches`. Comfortably under the `places=4` threshold
  (5e-5). New CI step `float_ssim cross-backend diff (CPU vs
  Vulkan/lavapipe)` on the lavapipe lane. CUDA + SYCL twins
  follow as batch 2 parts 1b + 1c.

- **GPU long-tail batch 1c parts 2 + 3 — `ciede_cuda` +
  `ciede_sycl` extractors (T7-23 / ADR-0182 / ADR-0187)**
  (fork-local): closes batch 1c. CUDA + SYCL twins of the
  Vulkan ciede kernel shipped in PR #136. Both emit the
  `ciede2000` metric (logarithmic transform `45 - 20·log10(mean_ΔE)`).
  - **CUDA** (~270 LOC PTX +
    [`integer_ciede_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_ciede_cuda.c)
    ~245 LOC host): per-pixel float ciede2000 (chroma read
    inline at the subsampled position — avoids the host-side
    upscale step), per-block partials reduced on the host in
    `double`. Surfaced a latent `vmaf_cuda_picture_upload_async`
    bug: the bitmask was hardcoded to `0x1` (luma only) in
    `libvmaf.c::translate_picture_host`, leaving chroma device
    buffers uninitialised — fine for every prior CUDA extractor
    (psnr / motion / adm / vif / moment all luma-only) but
    wrong for ciede. Bitmask now picks `0x7` for any pix_fmt
    other than YUV400P; CUDA chroma-aware kernels are unblocked.
  - **SYCL** (~470 LOC, single TU
    [`integer_ciede_sycl.cpp`](libvmaf/src/feature/sycl/integer_ciede_sycl.cpp)):
    self-contained submit/collect (does **not** register with
    `vmaf_sycl_graph_register` — `shared_frame` buffers are
    luma-only). Host-pinned USM staging upscales chroma to
    luma resolution (mirrors `ciede.c::scale_chroma_planes`);
    `nd_range<2>` kernel with `sycl::reduce_over_group` builds
    per-WG float partials; host accumulates in `double`.
    fp64-free (Intel Arc A380 lacks native fp64 — earlier
    `sycl::reduction<double>` attempt threw at runtime).
  - **Verification**: 48 frames at 576×324 vs CPU scalar —
    `max_abs = 1.2e-5`, `0/48 places=4 mismatches` on
    **NVIDIA RTX 4090** (CUDA) and **Intel Arc A380** (SYCL).
    Same precision floor as `ciede_vulkan` (PR #136).
  Closes batch 1c (Vulkan + CUDA + SYCL all done) — every GPU
  long-tail metric in [ADR-0182](docs/adr/0182-gpu-long-tail-batch-1.md)
  now has a working twin on at least one GPU backend.

- **GPU long-tail batch 1c part 1 — `ciede_vulkan` extractor
  (T7-23 / ADR-0187)** (fork-local): Vulkan twin of the CPU
  `ciede` extractor — the first non-bit-exact GPU kernel in
  the fork. Per-pixel ciede2000 ΔE uses ~40 transcendental ops
  (`pow` / `sqrt` / `sin` / `atan2`), so bit-exactness against
  the libm-based CPU is not on the table. New GLSL shader
  ([`libvmaf/src/feature/vulkan/shaders/ciede.comp`](libvmaf/src/feature/vulkan/shaders/ciede.comp))
  emits per-WG `float` partial sums; host accumulates in
  `double`, divides by W·H, and applies the CPU's logarithmic
  transform `45 - 20·log10(mean_ΔE)` for the final `ciede2000`
  metric. 6 storage-buffer bindings (ref + dis Y/U/V at full
  luma resolution); chroma upscaled host-side via the same
  pattern as `ciede.c::scale_chroma_planes`. New
  [`libvmaf/src/feature/vulkan/ciede_vulkan.c`](libvmaf/src/feature/vulkan/ciede_vulkan.c)
  (~480 LOC). Empirical: 48 frames at 576×324 on **Intel Arc
  A380** vs CPU scalar — `max_abs = 1.0e-5`, `0/48 places=4
  mismatches`. Empirical floor lands well under `places=4`
  threshold (≤5e-5), so the cross-backend gate runs at
  `places=4` for parity with the existing kernels. New CI
  step `ciede cross-backend diff (CPU vs Vulkan/lavapipe)` on
  the lavapipe lane. CUDA + SYCL twins follow as batch 1c
  parts 2 + 3 (last GPU long-tail rows).

- **GPU long-tail batch 1d parts 2 + 3 — `float_moment_cuda`
  and `float_moment_sycl` extractors (T7-23 / ADR-0182)**
  (fork-local): closes batch 1d. CUDA + SYCL twins of the
  Vulkan kernel shipped in PR #133 (ADR-0182). Both emit all
  four metrics — `float_moment_ref1st`, `float_moment_dis1st`,
  `float_moment_ref2nd`, `float_moment_dis2nd` — in **one
  kernel pass** via four atomic int64 counters.
  - **CUDA** (~120 LOC PTX +
    [`integer_moment_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_moment_cuda.c)
    ~225 LOC host): warp-shuffle int64 reduction (uint64 via
    two uint32 shuffles, same trick as `psnr_score.cu`) + four
    `atomicAdd(unsigned long long *)`. Same async submit /
    collect model as `psnr_cuda` (PR #129).
  - **SYCL** (~270 LOC, single TU
    [`integer_moment_sycl.cpp`](libvmaf/src/feature/sycl/integer_moment_sycl.cpp)):
    `sycl::atomic_ref<int64_t, ...>` × 4 in a single kernel.
    Rides the existing combined-graph submit / wait machinery
    via `vmaf_sycl_graph_register` (mirrors `psnr_sycl`,
    PR #130).
  - **Verification**: 48 frames at 576×324 vs CPU scalar —
    `max_abs = 0.0`, `0/48 places=4 mismatches` × 4 metrics
    on **NVIDIA RTX 4090** (CUDA) and **Intel Arc A380** (SYCL).
    Byte-exact at JSON precision; `int64` sum is exact on
    integer YUV inputs. `scripts/ci/cross_backend_vif_diff.py
    --feature float_moment --backend {cuda,sycl}`.
  Closes batch 1d (Vulkan + CUDA + SYCL all done); next is
  batch 1c (ciede across 3 backends).

- **Vulkan VkImage zero-copy import — implementation + FFmpeg
  filter (T7-29 parts 2 + 3, ADR-0186)** (fork-local): drops
  the `-ENOSYS` stubs from PR #128 and ships the matching
  FFmpeg-side filter in the same PR. libvmaf side: per-state
  ref/dis staging `VkBuffer` pair (HOST_VISIBLE,
  `DATA_ALIGN`-strided), `vkCmdCopyImageToBuffer` + timeline-
  semaphore wait per frame, no-op-release `VmafPicture` builder
  so `read_imported_pictures` routes through standard
  `vmaf_read_pictures`. New
  [`vmaf_vulkan_state_init_external`](libvmaf/include/libvmaf/libvmaf_vulkan.h)
  adopts the caller's VkInstance/VkDevice (required because
  source VkImage handles are device-bound). FFmpeg side: new
  [`ffmpeg-patches/0006-libvmaf-add-libvmaf-vulkan-filter.patch`](ffmpeg-patches/0006-libvmaf-add-libvmaf-vulkan-filter.patch)
  packages the `libvmaf_vulkan` filter consuming
  `AV_PIX_FMT_VULKAN` frames, pulling `AVVkFrame *` from
  `data[0]`, calling `vmaf_vulkan_state_init_external` with
  the device's compute queue, then `import_image` +
  `read_imported_pictures` per frame. Synchronous v1 design
  (fence-wait inside `import_image`); async pending-fence v2
  deferred. Smoke 10/10 (extends `test_vulkan_smoke.c` with
  five contract tests for the new surface). float_moment
  cross-backend gate clean — confirms the state-struct
  refactor doesn't regress existing kernel paths. Closes T7-29.

- **Fork rule §12 r14 — FFmpeg-patch updates ship in the same
  PR (ADR-0186)** (fork-local, process): every PR that touches
  a libvmaf public surface used by `ffmpeg-patches/` (C-API
  entry points, public headers, CLI flags,
  `meson_options.txt`, symbols probed by the
  `enabled libvmaf*` `check_pkg_config` lines) updates the
  relevant patch in the **same PR**. Pure libvmaf-internal
  refactors, doc-only, and test-only PRs are exempt. Reviewers
  verify with
  `for p in ffmpeg-patches/000*-*.patch; do git -C ffmpeg-8
  apply --check "$p"; done`. Closes a recurring failure mode
  where C-API drift broke the patch stack silently for the
  next rebase.

- **GPU long-tail batch 1d part 1 — `float_moment_vulkan`
  extractor (T7-23 / ADR-0182)** (fork-local): Vulkan twin of
  the CPU `float_moment` extractor. Single GLSL compute kernel
  ([`libvmaf/src/feature/vulkan/shaders/moment.comp`](libvmaf/src/feature/vulkan/shaders/moment.comp))
  emits all four metrics — `float_moment_ref1st`,
  `float_moment_dis1st`, `float_moment_ref2nd`,
  `float_moment_dis2nd` — in one dispatch via four atomic
  `int64` counters, using subgroup int64 reduction
  (`GL_EXT_shader_atomic_int64` +
  `GL_EXT_shader_explicit_arithmetic_types_int64`) into a
  shared array, then a single cross-subgroup
  `atomicAdd` per accumulator. Host divides the four sums by
  `width × height` to recover the raw moments. New
  [`libvmaf/src/feature/vulkan/moment_vulkan.c`](libvmaf/src/feature/vulkan/moment_vulkan.c)
  (~370 LOC) mirrors the `psnr_vulkan` scaffolding (3-binding
  descriptor set, single dispatch per frame, 8/10/12/16 bpc via
  spec constants). Empirical: 48 frames at 576×324 on Intel Arc
  A380 (lavapipe-equivalent) vs CPU scalar — `max_abs = 0.0`,
  `0/48 places=4 mismatches` × 4 metrics via
  `scripts/ci/cross_backend_vif_diff.py --feature float_moment
  --backend vulkan`. CUDA + SYCL twins follow as batch 1d parts
  2 and 3.

- **GPU long-tail batch 1b part 2 — `psnr_sycl` extractor
  (T7-23 / ADR-0182)** (fork-local): SYCL twin of `psnr_cuda`
  (PR #129) and `psnr_vulkan` (PR #125). Per-pixel int64
  squared-error reduction with `sycl::atomic_ref` accumulation
  into a shared device counter. Single kernel per frame, rides
  the existing combined-graph submit/wait machinery via
  `vmaf_sycl_graph_register`. New
  [`libvmaf/src/feature/sycl/integer_psnr_sycl.cpp`](libvmaf/src/feature/sycl/integer_psnr_sycl.cpp)
  (~280 LOC). Empirical: 48 frames at 576×324 on Intel Arc
  A380 vs CPU scalar — `max_abs_diff = 0.0`, `0/48 places=4
  mismatches` via `scripts/ci/cross_backend_vif_diff.py
  --backend sycl`. Closes "psnr on all 3 GPU backends" goal
  from ADR-0182.

- **GPU long-tail batch 1b part 1 — `psnr_cuda` extractor
  (T7-23 / ADR-0182)** (fork-local): CUDA twin of the
  `psnr_vulkan` kernel shipped in PR #125. Per-pixel int64
  squared-error reduction with warp-shuffle + atomicAdd
  (same pattern as `motion_score.cu`'s SAD reduction).
  Single dispatch per frame; emits luma-only `psnr_y` v1.
  New
  [`libvmaf/src/feature/cuda/integer_psnr/psnr_score.cu`](libvmaf/src/feature/cuda/integer_psnr/psnr_score.cu)
  (~120 LOC PTX) +
  [`libvmaf/src/feature/cuda/integer_psnr_cuda.{c,h}`](libvmaf/src/feature/cuda/integer_psnr_cuda.c)
  (~210 LOC host using CUDA's async submit/collect model).
  Empirical: 48 frames at 576×324 on NVIDIA RTX 4090 vs CPU
  scalar — `max_abs_diff = 0.0`, `0/48 places=4 mismatches`
  via `scripts/ci/cross_backend_vif_diff.py --backend cuda`.
  `psnr_sycl` follows in batch 1b part 2.

- **Vulkan VkImage zero-copy import C-API scaffold — T7-29
  part 1 (ADR-0184)** (fork-local): adds three new entry
  points in
  [`libvmaf_vulkan.h`](libvmaf/include/libvmaf/libvmaf_vulkan.h)
  — `vmaf_vulkan_import_image`, `vmaf_vulkan_wait_compute`,
  `vmaf_vulkan_read_imported_pictures` — mirroring the SYCL
  backend's existing import surface. Lets future FFmpeg-side
  filters consume `AVFrame->format == AV_PIX_FMT_VULKAN`
  frames without a `hwdownload,format=yuv420p` round-trip.
  Header purity: Vulkan handles cross the ABI as `uintptr_t`
  to keep the surface usable from translation units that
  don't have `<vulkan/vulkan.h>` in scope (matches the
  libvmaf_cuda.h precedent). **Scaffold only**: every
  function returns `-ENOSYS`, mirroring how the original
  Vulkan backend shipped via ADR-0175. T7-29 part 2 (real
  `vkCmdCopyImageToBuffer` + timeline-semaphore wait) and
  part 3 (FFmpeg-side `libvmaf_vulkan` filter as
  `ffmpeg-patches/0006-*`) follow in subsequent PRs. See
  [ADR-0184](docs/adr/0184-vulkan-image-import-scaffold.md).

- **`libvmaf_sycl` FFmpeg filter — zero-copy QSV/VAAPI import
  (T7-28, ADR-0183)** (fork-local): closes the hwdec ergonomic
  gap exposed by PR #126. New
  [`ffmpeg-patches/0005-libvmaf-add-libvmaf-sycl-filter.patch`](ffmpeg-patches/0005-libvmaf-add-libvmaf-sycl-filter.patch)
  adds a dedicated `libvmaf_sycl` filter that consumes oneVPL
  `mfxFrameSurface1` frames (`AVFrame->data[3]`), extracts the
  underlying VA surface ID, and routes through
  `vmaf_sycl_import_va_surface` for zero-copy DMA-BUF import on
  the Level Zero / SYCL compute queue. Build FFmpeg with
  `--enable-libvmaf-sycl` (in addition to `--enable-libvmaf`).
  Removes the `hwdownload,format=yuv420p` round-trip for the
  common Intel QSV hwdec path. Pairs with the existing
  `0003-libvmaf-wire-sycl-backend-selector.patch` so users have
  two paths: `libvmaf=sycl_device=N` for software frames + SYCL
  compute, `libvmaf_sycl=…` for QSV hwdec + zero-copy SYCL.
  Validated on Intel Arc A380. **T7-29** (Vulkan VkImage import)
  remains open — needs new C-API surface in
  [`libvmaf_vulkan.h`](libvmaf/include/libvmaf/libvmaf_vulkan.h)
  before the FFmpeg-side filter can land. See
  [ADR-0183](docs/adr/0183-ffmpeg-libvmaf-sycl-filter.md).

- **GPU long-tail batch 1a — `psnr_vulkan` extractor (T7-23 /
  ADR-0182)** (fork-local): first kernel of the GPU long-tail
  batch. Per-pixel squared-error reduction on the Vulkan compute
  backend; emits `psnr_y` (luma-only v1; chroma is a focused
  follow-up since `picture_vulkan` upload is luma-only today).
  New
  [`libvmaf/src/feature/vulkan/shaders/psnr.comp`](libvmaf/src/feature/vulkan/shaders/psnr.comp)
  (89 LOC GLSL, 16×8 WG, subgroup-int64 reduction) +
  [`libvmaf/src/feature/vulkan/psnr_vulkan.c`](libvmaf/src/feature/vulkan/psnr_vulkan.c)
  (391 LOC host C, single dispatch/frame, no temporal state).
  Cross-backend gate gains a 4th step ("PSNR cross-backend diff")
  on the lavapipe lane. Empirical: 48 frames at 576×324 on
  Intel Arc A380 / Mesa anv vs CPU scalar — `max_abs_diff = 0.0`,
  `0/48 places=4 mismatches`. Foundation also adds chars
  descriptors to the existing scalar `psnr` / `ciede` /
  `float_moment` registrations ahead of CUDA / SYCL twins
  landing in batches 1b–1d. See
  [ADR-0182](docs/adr/0182-gpu-long-tail-batch-1.md).

- **T7-26 — Global feature-characteristics registry + per-backend
  dispatch-strategy modules** (fork-local): consolidates the
  per-context SYCL graph-replay heuristic into a per-feature
  decision driven by a registry on `VmafFeatureExtractor`. New
  [`libvmaf/src/feature/feature_characteristics.h`](libvmaf/src/feature/feature_characteristics.h)
  exposes the descriptor struct (`n_dispatches_per_frame`,
  `is_reduction_only`, `min_useful_frame_area`,
  `dispatch_hint`). Per-backend glue under
  [`libvmaf/src/{cuda,sycl,vulkan}/dispatch_strategy.{c,h}`](libvmaf/src/sycl/dispatch_strategy.cpp)
  translates the descriptor to backend primitives (SYCL graph
  replay today; CUDA graph capture and Vulkan secondary-cmdbuf
  reuse are stubs that ship the env-override surface for a
  follow-up PR to enable). New env knobs:
  `VMAF_SYCL_DISPATCH=feature:graph,feature:direct,...`,
  `VMAF_CUDA_DISPATCH=...`,
  `VMAF_VULKAN_DISPATCH=feature:reuse,feature:primary,...`.
  Legacy `VMAF_SYCL_USE_GRAPH` / `VMAF_SYCL_NO_GRAPH` kept as
  global aliases. Descriptors seeded for vif (4 dispatches),
  motion (2 dispatches, 1080p area), adm (16 dispatches, 720p
  area). Empirical: ADM at 576×324 within 0.5% of pre-T7-26
  behaviour (registry preserves byte-for-byte AUTO + 720p
  semantics). Foundation for adding the GPU long-tail (14
  metrics × 3 backends = up to 42 future kernels) without
  duplicate dispatch logic. Side-fix: pre-existing GCC LTO
  type-mismatch surfaced by the new `chars` field —
  `null.c` / `feature_lpips.c` / `ssimulacra2.c` were missing
  `#include "config.h"` and saw a smaller `VmafFeatureExtractor`
  struct than `feature_extractor.c`. See
  [ADR-0181](docs/adr/0181-feature-characteristics-registry.md).

- **`float_moment` SIMD parity (AVX2 + NEON) — T7-19, closes
  the only fully-scalar row in the SIMD-coverage matrix**
  (fork-local): new
  [`libvmaf/src/feature/x86/moment_avx2.{c,h}`](libvmaf/src/feature/x86/moment_avx2.c)
  and [`libvmaf/src/feature/arm64/moment_neon.{c,h}`](libvmaf/src/feature/arm64/moment_neon.c)
  implement `compute_1st_moment` / `compute_2nd_moment` 8-wide
  (AVX2) and 4-wide (NEON) following the `ansnr_avx2.c` pattern:
  square in float, accumulate into `double` via scattered-tmp
  (AVX2) or lane-pair widening via `vcvt_f64_f32` (NEON).
  Dispatched from
  [`float_moment.c::init`](libvmaf/src/feature/float_moment.c)
  via function pointers selected from `vmaf_get_cpu_flags()`.
  Tolerance-bounded contract (1e-7 relative — ~500× tighter than
  the production snapshot gate's `places=4`), matching the
  established kernel header documentation. New
  [`test_moment_simd`](libvmaf/test/test_moment_simd.c) runs
  four cases per arch (two random seeds, an aligned width, and a
  tiny edge case to exercise the per-row tail). End-to-end CLI
  output unchanged at JSON `%g` precision. See
  [ADR-0179](docs/adr/0179-float-moment-simd.md).

- **Vulkan ADM kernel + cross-backend gate fixes — T5-1c (closes T5-1c)**
  (fork-local): replaces the 37-line adm_vulkan.c stub with a real
  `VmafFeatureExtractor` (~700 LOC) backed by a new GLSL compute shader
  [`shaders/adm.comp`](libvmaf/src/feature/vulkan/shaders/adm.comp)
  (~660 LOC). Implements 4-scale CDF 9/7 DWT, decouple+CSF fused
  pass, and per-band CSF-denominator + contrast-measure reductions.
  16 pipelines per extractor (one per `(scale, stage)`). Provides the
  standard `integer_adm2`, `integer_adm_scale0..3` outputs.

  This PR ALSO uncovered and fixed three latent bugs that made the
  cross-backend gate land bogus ULP=0 results since PR #118:
  (a) `tools/meson.build` never set `-DHAVE_VULKAN`, so every
  `--vulkan_device` call silently no-op'd; (b)
  `vmaf_use_feature()` skipped `set_fex_vulkan_state()` so the
  imported state never reached the extractor; (c) the script's
  `--feature X` invocation collided with the default model's CPU
  extractors, dropping the GPU writer's scores. Plus a header
  shadowing fix (Vulkan `common.h` → `vulkan_common.h` so
  CUDA+Vulkan can build together) and a new `--backend
  {auto,cpu,cuda,sycl,vulkan}` CLI flag that closes the
  multi-backend dispatcher conflict (first-match-wins favored CUDA
  over Vulkan).

  Real cross-backend numbers (576x324, 48 frames, post-fix gate):

  | backend (device) | vif | motion | adm |
  | --- | --- | --- | --- |
  | CUDA (RTX 4090) | ULP=0 ✓ | **2.6e-3 ❌ 47/48** | ≤1e-6 ✓ |
  | SYCL (Arc, oneAPI) | ≤1e-6 ✓ | **2.6e-3 ❌ 47/48** | scale2 2.4e-4 ❌ 1/48 |
  | Vulkan (RTX) | ≤1e-6 ✓ | ≈1e-6 ✓ | scale2 2.4e-4 ❌ 1/48 |
  | Vulkan (Arc, Mesa anv) | ≤1e-6 ✓ | ≈1e-6 ✓ | ≤3.1e-5 ✓ |

  Three pre-existing kernel-side bugs surfaced by the working gate:
  (1) CUDA motion AND SYCL motion both drift by 2.6e-3 (47/48
  frames) vs CPU — same magnitude, likely shared algorithmic
  inheritance; (2) NVIDIA Vulkan + SYCL `adm_scale2` both drift
  by 2.4e-4 (1/48 frames) — likely shared host-side reduction
  order divergence; (3) SYCL on fp64-less GPUs (e.g. Arc A380)
  uses int64 emulation for gain limiting, causing a 5-10×
  slowdown vs Vulkan on the same hardware. Each tracked as a
  follow-up.

  Vulkan-on-Arc (the path under the lavapipe blocking gate via
  Mesa anv) is the only fully-clean GPU backend in the current
  matrix. Closes T5-1c. See
  [ADR-0178](docs/adr/0178-vulkan-adm-kernel.md).
- **Vulkan motion kernel — T5-1c (motion + motion2)** (fork-local):
  replaces the 37-line motion_vulkan.c stub with a real
  `VmafFeatureExtractor` backed by a new GLSL compute shader
  [`shaders/motion.comp`](libvmaf/src/feature/vulkan/shaders/motion.comp).
  Separable 5-tap Gaussian blur (`{3571, 16004, 26386, 16004, 3571}`,
  sum=65536) + per-WG `int64` SAD reduction; ping-pong blurred-frame
  storage; `integer_motion2` emitted with the standard 1-frame lag.
  `motion3` (5-frame window mode) deliberately deferred. Cross-backend
  diff script generalized: `scripts/ci/cross_backend_vif_diff.py`
  gains `--feature {vif,motion}`. Empirical: ≤1e-6 vs CPU on Arc
  via Vulkan (Mesa anv); 2.6e-3 drift on CUDA/SYCL motion is a
  pre-existing kernel bug surfaced by PR #120's gate fix. See
  [ADR-0177](docs/adr/0177-vulkan-motion-kernel.md). **NOTE**: the
  original "ULP=0" claim in this entry was bogus — the gate was
  comparing CPU-vs-CPU due to the build-system bug PR #120 fixes.
- **Vulkan VIF cross-backend gate + CLI (`--vulkan_device`) — T5-1b-v**
  (fork-local): wires Vulkan into the libvmaf dispatcher and `vmaf`
  CLI. New `--vulkan_device <N>` (auto-pick `-1`, default disabled)
  and `--no_vulkan` flags. Adds `VMAF_FEATURE_EXTRACTOR_VULKAN = 1 << 5`
  and the public state-level API (`vmaf_vulkan_state_init` / `_free` /
  `_available` / `_list_devices`). New
  [`scripts/ci/cross_backend_vif_diff.py`](scripts/ci/cross_backend_vif_diff.py)
  with two CI lanes: `Vulkan VIF Cross-Backend (lavapipe, places=4)` runs
  on every PR via Mesa lavapipe (no GPU runner needed), Arc-A380
  nightly advisory parked behind `if: false` until a self-hosted runner
  with label `vmaf-arc` is registered. **NOTE**: the original
  "ULP=0 vs CPU" claim in this entry was bogus — the meson glue
  for `-DHAVE_VULKAN` in `tools/meson.build` was missing, so
  `--vulkan_device` silently no-op'd. PR #120 fixes the build
  system, the framework state propagation in `vmaf_use_feature()`,
  and the script's invocation pattern. See
  [ADR-0176](docs/adr/0176-vulkan-vif-cross-backend-gate.md).
- **Vulkan VIF math port — T5-1b-iv (4-scale GLSL kernel)** (fork-local):
  full numerical port of the SYCL VIF kernel to a GLSL compute shader.
  `shaders/vif.comp` runs four pipelines (one per `SCALE` specialization
  constant) compiled to SPIR-V via `glslc`, embedded as a byte array,
  dispatched in a single command buffer with pipeline barriers between
  scales. Uses native `int64` accumulators
  (`GL_EXT_shader_explicit_arithmetic_types_int64`) for deterministic
  reductions matching the CPU integer reference. First feature kernel
  to actually run end-to-end on Intel Arc A380.
- **Vulkan runtime bring-up — T5-1b** (fork-local): replaces the T5-1
  scaffold's `-ENOSYS` stubs with a real volk + Vulkan 1.3 + VMA bring-up.
  `vmaf_vulkan_context_new` picks a compute-capable physical device
  (auto: discrete > integrated > virtual > cpu; override via
  `device_index`), creates a dedicated compute queue family, attaches
  a VMA allocator, and exposes a command pool that per-feature dispatch
  wrappers under `libvmaf/src/feature/vulkan/` reuse. New `vma_impl.cpp`
  (C++17 TU isolating the VMA implementation), new `picture_vulkan.{c,h}`
  (VkBuffer alloc / flush / mapped-host pointer accessors). `volk` and
  `VulkanMemoryAllocator` pulled via Meson wrap files (no system install
  required); `glslc` becomes a build-time requirement when
  `-Denable_vulkan=enabled`.
- **Whole-codebase docs sweep — close audit-identified gaps**
  (fork-local): post-T5-1 docs audit identified four undocumented
  user-discoverable surfaces and one stale CLI flag entry. Adds
  [`docs/backends/arm/overview.md`](docs/backends/arm/overview.md)
  for the ARM NEON backend (build, runtime control via `--cpumask`,
  per-feature coverage table, bit-exactness contracts, CI matrix
  pointer). Documents the `motion_v2`, `lpips`, and `float_moment`
  feature extractors in
  [`docs/metrics/features.md`](docs/metrics/features.md) (table
  rows + per-feature sections covering invocation, output metrics,
  output range, input formats, options, backends, limitations).
  Expands the `--no-reference` flag entry in
  [`docs/usage/cli.md`](docs/usage/cli.md) with preconditions
  (`reference_required: false` registry field), failure modes, and
  report-format implications. Updates
  [`docs/api/gpu.md`](docs/api/gpu.md) title to include
  `libvmaf_vulkan.h` and links the new ARM overview from
  [`docs/backends/index.md`](docs/backends/index.md). No code
  changes; touched-file lint cleanup converts standalone
  `**Heading**` lines to `####` H4 (MD036), aligns options
  tables exactly (MD060), and tags bare fenced code blocks with
  `text` (MD040).
- **Vulkan compute backend — scaffold-only audit-first PR**
  (fork-local): closes BACKLOG T5-1 audit half. New public header
  [`libvmaf_vulkan.h`](libvmaf/include/libvmaf/libvmaf_vulkan.h)
  declaring the `VmafVulkanState` API surface (state_init,
  import_state, state_free, list_devices, available). New
  `libvmaf/src/vulkan/` + `libvmaf/src/feature/vulkan/` trees with
  every entry point returning `-ENOSYS`. New `enable_vulkan`
  feature option (default **disabled**) and conditional
  `subdir('vulkan')` in libvmaf's meson. New 4-sub-test smoke
  pinning the stub contract. New CI matrix row compiles with
  `-Denable_vulkan=enabled`. New ffmpeg patch
  [`0004-libvmaf-wire-vulkan-backend-selector.patch`](ffmpeg-patches/0004-libvmaf-wire-vulkan-backend-selector.patch)
  mirroring the SYCL selector — adds a `vulkan_device` libvmaf
  filter option. **Zero runtime dependencies** for the scaffold;
  `dependency('vulkan')` + volk + glslc + VMA land with the
  runtime PR per ADR-0127's "VIF as pathfinder" sequence. See
  [ADR-0175](docs/adr/0175-vulkan-backend-scaffold.md).
- **First per-model PTQ — `learned_filter_v1` flips to dynamic int8**
  (fork-local): closes T5-3 fully (audit half via ADR-0173;
  first-model half + CI gate via this PR). 80 KB → 33 KB (2.4×
  shrink). Drop measurement: PLCC 0.999883 vs fp32 on a 16-sample
  synthetic input set, drop 0.000117 vs the per-model budget 0.01
  (100× margin). Runtime `.int8.onnx` redirect wired in
  `vmaf_dnn_session_open` — when the sidecar declares
  `quant_mode != FP32`, the loader strips trailing `.onnx`,
  appends `.int8.onnx`, re-validates, and passes that path to ORT.
  Fp32 file stays on disk as the regression baseline. New
  `int8_sha256` registry/sidecar field (required when
  `quant_mode != fp32`). New `ai/scripts/measure_quant_drop.py`
  walks the registry and gates each non-fp32 model. New
  `ai-quant-accuracy` step in the `Tiny AI` CI job runs the gate
  on every PR. C2 `nr_metric_v1` stays fp32 — its dynamic-batch
  ONNX export trips ORT's internal shape inference (tracked as
  T5-3c follow-up). See
  [ADR-0174](docs/adr/0174-first-model-quantisation.md). Closes
  BACKLOG T5-3b.
- **PTQ int8 audit harness (audit-first)** (fork-local): scaffolds
  the per-model quantisation pipeline from ADR-0129. Three new
  optional fields in `model/tiny/registry.schema.json`
  (`quant_mode` enum `fp32` / `dynamic` / `static` / `qat`;
  `quant_calibration_set`; `quant_accuracy_budget_plcc` default
  0.01). Three new scripts under `ai/scripts/` (`ptq_dynamic.py`,
  `ptq_static.py`, `qat_train.py` — the last is a CLI scaffold that
  raises `NotImplementedError` until a per-model QAT PR lands the
  trainer hook). New `VmafModelQuantMode` enum + sidecar parser
  branch in `libvmaf/src/dnn/model_loader.{h,c}`; default FP32
  fail-safe on unknown sidecar values. 4 Python smoke tests + 3 C
  sidecar tests. **No shipped model flips its `quant_mode`** in
  this PR — runtime `.int8.onnx` redirect + the `ai-quant-accuracy`
  CI gate land with the first per-model quantisation PR (T5-3b).
  New
  [`docs/ai/quantization.md`](docs/ai/quantization.md) user
  reference. See
  [ADR-0173](docs/adr/0173-ptq-int8-audit-impl.md). Closes
  BACKLOG T5-3 audit half (T5-3b queued for the gate).
- **MCP `describe_worst_frames` tool with VLM fallback**
  (fork-local): new MCP tool that scores a `(ref, dis)` pair, picks
  the N worst-VMAF frames (default 5, capped at 32), extracts each
  as PNG via `ffmpeg`, and runs a vision-language model
  (SmolVLM → Moondream2 cascade) to describe the visible
  artefacts. Returns `{model_id, frames: [{frame_index, vmaf, png,
  description}]}`. Falls back to metadata-only output with a clear
  hint when the new `vlm` optional dependency group isn't
  installed. New `[vlm]` extras (`transformers + torch + Pillow +
  accelerate`); base MCP install stays light. First concrete
  consumer of ADR-0171's bounded-Loop guard — VLM autoregressive
  token generation needs `Loop` nodes. 5 new tests; all 17 MCP
  tests pass. See
  [ADR-0172](docs/adr/0172-mcp-describe-worst-frames.md). Closes
  BACKLOG T6-6.
- **Bounded `Loop.M` trip-count guard** (fork-local): closes the
  follow-up deferred in ADR-0169. Two layers, mirroring the
  ADR-0167 doc-drift enforcement pattern. (1) Python export-time
  `vmaf_train.op_allowlist` traces every `Loop`'s first input back
  to a `Constant` int64 scalar (recurses into subgraphs); rejects
  graph-input M, non-Constant producers, and values outside
  `[0, MAX_LOOP_TRIP_COUNT]` (default 1024, per-call overridable).
  `AllowlistReport.loop_violations` carries actionable diagnostics.
  (2) C wire-format scanner caps total `Loop` nodes per model at
  `VMAF_DNN_MAX_LOOP_NODES = 16` via a counter threaded through
  `scan_graph` / `scan_node` / `scan_attribute`; rejects with
  `-EPERM` and `first_bad="Loop"` on exceedance. C cap is
  intentionally coarser than the Python data-flow check —
  reproducing producer-map lookup would violate the ADR D39
  "no libprotobuf-c" scanner-scope constraint. 5 new Python tests
  plus 1 new C test. See
  [ADR-0171](docs/adr/0171-bounded-loop-trip-count.md). Closes
  BACKLOG T6-5b.
- **`vmaf_pre` ffmpeg filter handles 10/12-bit + optional chroma**
  (fork-local): new public libvmaf API `vmaf_dnn_session_run_plane16`
  accepts packed `uint16` LE single-plane buffers with a `bpc`
  argument (range 9..16). The ffmpeg filter now admits
  `yuv{420,422,444}p1{0,2}le` + `gray{10,12}le` pixel formats and
  dispatches the matching entrypoint by bit-depth. New
  `chroma=0|1` option (default 0 preserves luma-only back-compat)
  re-runs the same session on U/V planes at chroma-subsampled
  dimensions when set. Two new tensor helpers
  (`vmaf_tensor_from_plane16` / `vmaf_tensor_to_plane16`) with 3
  round-trip tests pinning 10-bit identity, bpc bounds, and 12-bit
  clamp behaviour. See
  [ADR-0170](docs/adr/0170-vmaf-pre-10bit-chroma.md). Closes
  BACKLOG T6-4.
- **ONNX op-allowlist admits `Loop` + `If`** (fork-local): unblocks
  MUSIQ / RAFT / small-VLM-class tiny-AI baselines that need
  control-flow ops. The wire-format scanner in
  [`onnx_scan.c`](libvmaf/src/dnn/onnx_scan.c) gains mutually-recursive
  `scan_attribute` / `scan_node` / `scan_graph` helpers that descend
  into `NodeProto.attribute` → `AttributeProto.g` / `.graphs` so a
  forbidden op cannot hide inside a `Loop.body` /
  `If.then_branch` / `If.else_branch` subgraph. Recursion depth-capped
  at `VMAF_DNN_MAX_SUBGRAPH_DEPTH = 8` as a defence-in-depth bound.
  Python `vmaf_train.op_allowlist` mirrors the recursion via a new
  `_collect_op_types` helper so the export-time check and the runtime
  load-time check stay in lockstep. `Scan` stays off the allowlist
  (variant-typed input/output binding makes static bound-checking
  impractical). The bounded-iteration guard for `Loop.M → Constant ≤
  MAX_LOOP_ITERATIONS` is **explicitly deferred** to a follow-up ADR
  (T6-5b). 4 existing tests flipped + 4 new subgraph-recursion tests
  added (2 C, 2 Python). See
  [ADR-0169](docs/adr/0169-onnx-allowlist-loop-if.md). Closes
  BACKLOG T6-5.
- **Tiny-AI Wave 1 baselines C2 + C3** (fork-local): trained ONNX
  checkpoints `nr_metric_v1.onnx` (NR MobileNet, ~19K params,
  224×224 grayscale → MOS) and `learned_filter_v1.onnx` (residual
  CNN for ffmpeg `vmaf_pre`, ~19K params, denoise-style residual
  filter) shipped in `model/tiny/`. Both trained on KoNViD-1k:
  C2 supervised on its MOS labels, C3 self-supervised on
  synthetic gaussian + JPEG degradation pairs derived from the same
  middle-frames. Op-allowlist + ORT roundtrip atol 1e-4 both pass.
  Four new scripts under `ai/scripts/` (`fetch_konvid_1k.py` /
  `extract_konvid_frames.py` / `train_konvid.py` /
  `export_tiny_models.py`); two new datamodule classes
  (`FrameMOSDataset`, `PairedFrameDataset`) in
  `vmaf_train.data.frame_dataset`. Registry schema +
  `VmafModelKind` enum extended with `kind: "filter"` to accommodate
  C3 (registry trust-root for filter models — NOT loaded by
  libvmaf's scoring path). KoNViD-1k MOS values are not
  redistributed; populated manifest stays gitignored. **C1
  (`fr_regressor_v1.onnx`) is deferred** — Netflix Public Dataset
  is access-gated (Google Drive, manual approval) and cannot be
  downloaded programmatically; tracked in
  [`docs/state.md`](docs/state.md). See
  [ADR-0168](docs/adr/0168-tinyai-konvid-baselines.md). Closes
  BACKLOG T6-1 partially (2 of 3).
- **Path-mapped doc-drift enforcement** (fork-local): closes the
  gap surfaced by the 2026-04-25 docs audit. New project hook
  [`.claude/hooks/docs-drift-warn.sh`](.claude/hooks/docs-drift-warn.sh)
  emits an informational `NOTICE` when an Edit/Write touches a
  user-discoverable surface (libvmaf headers / feature extractors /
  SIMD twins / CLI / MCP / tiny-AI CLI / ffmpeg patches) but no
  matching `docs/<topic>/` file is touched. CI counterpart in
  [`rule-enforcement.yml`](.github/workflows/rule-enforcement.yml)
  promoted from advisory to blocking + rewritten to use a
  path-mapped surface→docs check; ADR additions no longer satisfy
  it (ADRs are decisions, not user docs). Per-PR opt-out
  `no docs needed: REASON` for genuine internal-refactor PRs.
  See [ADR-0167](docs/adr/0167-doc-drift-enforcement.md).
- **Documentation refresh covering 16 recent PRs** (fork-local): in
  the same PR as ADR-0167, the audit-flagged gaps are filled —
  `vmaf_cuda_state_free()` API documented in
  [`docs/api/gpu.md`](docs/api/gpu.md); `-EAGAIN` semantics +
  `vmaf_read_pictures` monotonic-index requirement in
  [`docs/api/index.md`](docs/api/index.md); SSIMULACRA 2 + PSNR-HVS
  SIMD coverage matrix and `float_ms_ssim` <176×176 minimum
  documented in [`docs/metrics/features.md`](docs/metrics/features.md).
- **Tracked `docs/state.md` + bug-status hygiene rule** (fork-local):
  closes [Issue #20](https://github.com/lusoris/vmaf/issues/20) and
  backlog item T7-1. New tracked file [`docs/state.md`](docs/state.md)
  is the canonical in-tree register of bug status (Open / Recently
  closed / Confirmed not-affected / Deferred). New CLAUDE.md §12
  rule 13 mandates a same-PR update on every bug close / open /
  rule-out; the PR template carries a checkbox. ADRs cover decisions,
  this file covers bug status — distinct artifacts. See
  [ADR-0165](docs/adr/0165-state-md-bug-tracking.md).
- **MCP server release artifact channel — PyPI + GitHub release
  attachment + Sigstore** (fork-local): closes backlog item T7-2.
  [`supply-chain.yml`](.github/workflows/supply-chain.yml) extended
  with new `mcp-build` / `mcp-sign` / `mcp-publish-pypi` jobs.
  After this lands, `pip install vmaf-mcp` works (PyPI Trusted
  Publishing via OIDC, no token); the same wheel + sdist also
  attach to the GitHub release with a Sigstore keyless `.bundle` +
  PEP 740 attestation + SLSA L3 provenance. One-time PyPI
  Trusted-Publisher binding required (operational note in the ADR).
  See [ADR-0166](docs/adr/0166-mcp-server-release-channel.md).
- **Self-hosted GPU runner enrollment guide** (fork-local): closes
  backlog item T7-3. New
  [`docs/development/self-hosted-runner.md`](docs/development/self-hosted-runner.md)
  pins the registration steps so the next operator (or the user's
  local dev box, per popup 2026-04-25) can stand a runner up in
  ~10 minutes. The fine-grained label scheme (`gpu-cuda`,
  `gpu-intel`, `avx512`) is documented for future job targeting.
- **`motion_v2` NEON SIMD** (fork-local): aarch64 users now get a
  NEON fast path for the `motion_v2` feature. Scalar + AVX2 + AVX-512
  variants already existed; this closes the ISA-parity gap (backlog
  T3-4). The NEON impl uses arithmetic right-shift throughout
  (`vshrq_n_s64`, `vshlq_s64(v, -bpc)`) to match the scalar C `>>`
  semantics byte-for-byte — deliberately diverging from the fork's
  AVX2 variant, which uses logical `_mm256_srlv_epi64` and can
  diverge on negative-diff pixels; an AVX2 re-audit is queued as
  follow-up. Five small `static inline` helpers keep every function
  under ADR-0141's 60-line budget; zero clang-tidy warnings, no
  NOLINT. Verified bit-exact under QEMU user-mode on the Netflix
  `src01_hrc00/01_576x324` pair. See
  [ADR-0145](docs/adr/0145-motion-v2-neon-bitexact.md).

### Fixed

- **`float_ms_ssim` rejects input below 176×176 at init**
  (Netflix upstream issue
  [#1414](https://github.com/Netflix/vmaf/issues/1414)). The
  5-level 11-tap MS-SSIM pyramid walks off the kernel footprint
  at a mid-level scale for inputs below 176×176 (QCIF and
  smaller), previously producing a confusing mid-run `error:
  scale below 1x1!` + cascading `problem reading pictures` /
  `problem flushing context`. The fix checks `w < GAUSSIAN_LEN
  << (SCALES - 1)` at init and returns `-EINVAL` with a helpful
  error that names the input resolution, the required minimum
  (176×176), and the upstream issue. Minimum is derived from
  the existing filter constants so it stays in sync if those
  ever change. Visible behaviour: init now fails immediately
  instead of mid-stream; zero impact on inputs ≥176×176. New
  3-subtest reducer in `test_float_ms_ssim_min_dim.c` verified
  to fail pre-fix and pass post-fix. Closes backlog item T1-4.
  See [ADR-0153](docs/adr/0153-float-ms-ssim-min-dim-netflix-1414.md).

### Added

- **SSIMULACRA 2 regression gate** (fork-local, backlog T3-3). New
  `python/test/ssimulacra2_test.py` invokes `vmaf --feature ssimulacra2`
  on the canonical `src01_hrc00/01_576x324` pair + the small 160×90
  derived fixture and pins the per-frame + pooled output scores
  against reference values with 4-place tolerance. Catches unintended
  drift in the extractor output — complements the kernel-level SIMD
  bit-exact unit tests with an end-to-end integration gate. Closes
  ADR-0130's T3-3 deferral; backlog T3-1 + T3-3 both close now.
  See [ADR-0164](docs/adr/0164-ssimulacra2-snapshot-gate.md).

- **SSIMULACRA 2 `picture_to_linear_rgb` SIMD (AVX2 + AVX-512 + NEON)**
  (fork-local, backlog T3-1 phase 3 — closes T3-1 in full). The last
  scalar hot path in the SSIMULACRA 2 extractor is now vectorised on
  all 3 ISAs. YUV → linear RGB with BT.709/BT.601 matmul + sRGB
  EOTF, handling all pixel formats: BT.709/BT.601 × limited/full,
  any chroma subsampling ratio (420/422/444/irregular), 8-16 bpc.
  Strategy: per-lane scalar pixel reads fill an aligned scratch
  (handles all chroma ratios + bit depths uniformly); SIMD matmul +
  normalise + clamp; per-lane scalar `powf` for the sRGB EOTF branch
  (mirrors the phase-1 `cbrtf` pattern). Byte-for-byte bit-exact to
  scalar under `FLT_EVAL_METHOD == 0`. New shared header
  `ssimulacra2_simd_common.h` with `simd_plane_t` decouples SIMD
  TUs from `VmafPicture`. Five new test subtests
  (420-8bit/420-10bit/444-8bit/444-10bit/422-8bit) — 11/11 pass on
  AVX-512 host + 11/11 under `qemu-aarch64-static` (NEON).
  SSIMULACRA 2 now has **zero scalar hot paths**. See
  [ADR-0163](docs/adr/0163-ssimulacra2-ptlr-simd.md).

- **SSIMULACRA 2 FastGaussian IIR blur SIMD (AVX2 + AVX-512 + NEON)**
  (fork-local, backlog T3-1 phase 2). `blur_plane` — the single
  largest wall-clock cost in the SSIMULACRA 2 extractor (30 calls
  per frame across 5 blur-combinations × 6 scales) — now runs on
  SIMD. Horizontal pass batches N rows with `_mm256_i32gather_ps` /
  `_mm512_i32gather_ps` (AVX2 N=8, AVX-512 N=16) or 4 explicit
  `vsetq_lane_f32` calls (NEON N=4, no native gather on aarch64).
  Vertical pass uses column-SIMD loads/stores over the per-column
  IIR state arrays. Byte-for-byte bit-exact to scalar under
  `FLT_EVAL_METHOD == 0` — verified via new `test_blur` subtest
  (6/6 on AVX-512 host, 6/6 under `qemu-aarch64-static` for NEON).
  Dispatched via a new `blur_fn` function pointer in `Ssimu2State`
  assigned in `init_simd_dispatch()`. Only `picture_to_linear_rgb`
  remains scalar — deferred to follow-up. Closes backlog T3-1
  phase 2. See
  [ADR-0162](docs/adr/0162-ssimulacra2-iir-blur-simd.md).

- **SSIMULACRA 2 SIMD fast paths (AVX2 + AVX-512 + NEON)** (fork-local,
  backlog T3-1 + T3-2). Five of the eight hot kernels in the
  SSIMULACRA 2 pipeline now run on SIMD: `multiply_3plane`,
  `linear_rgb_to_xyb` (per-lane scalar `cbrtf` preserves bit-exactness),
  `downsample_2x2`, `ssim_map`, `edge_diff_map`. All 15 kernels
  (5 × 3 ISAs) produce **byte-for-byte identical output to scalar**
  under `FLT_EVAL_METHOD == 0` — verified via new unit test
  `test_ssimulacra2_simd.c` (5/5 pass on AVX-512 host; 5/5 under
  `qemu-aarch64-static` on NEON). Scalar summation order preserved
  left-to-right throughout to avoid IEEE-754 non-associativity drift
  (caught pre-merge by the bit-exact test). Reductions on
  `ssim_map` / `edge_diff_map` use the ADR-0139 per-lane `double`
  scalar tail. Runtime dispatch via function pointers in
  `Ssimu2State` with AVX-512 > AVX2 > NEON > scalar precedence.
  **Deferred to follow-up PRs**: IIR blur (`fast_gaussian_1d` / `blur_plane`,
  serial recurrence + per-column state) and `picture_to_linear_rgb`
  (`powf` EOTF) — see ADR-0161 §Alternatives. Closes backlog T3-1 + T3-2
  partially. See
  [ADR-0161](docs/adr/0161-ssimulacra2-simd-bitexact.md).

- **`psnr_hvs` NEON SIMD path** (fork-local, backlog T3-5-neon).
  Sister port to the AVX2 variant; aarch64 users now get the same
  byte-identical vectorized Xiph/Daala 8×8 integer DCT. NEON's
  4-wide `int32x4_t` means each 8-column row splits into
  `lo` (cols 0-3) + `hi` (cols 4-7); the 30-butterfly runs twice
  per DCT pass and the 8×8 transpose decomposes into four 4×4
  `vtrn1q_s32` / `vtrn2q_s32` / `vtrn1q_s64` / `vtrn2q_s64`
  stages plus a top-right ↔ bottom-left block swap. Float
  accumulators stay scalar per ADR-0139/0159 bit-exactness rule;
  `accumulate_error()` threads the outer `ret` by pointer
  (ADR-0159 summation-order lesson inherited). New unit test
  `test_psnr_hvs_neon.c`: 5/5 DCT subtests pass under
  `qemu-aarch64-static`. 576×324 8-bit Netflix golden pair
  scalar-vs-NEON diff: byte-identical `psnr_hvs_{y,cb,cr}` scores.
  1080p 10-bit pairs deferred to native-aarch64 CI (QEMU segfaults
  on heavy 10-bit threadpool allocations — known emulator limit,
  not a defect in the port). Runtime dispatch gated by
  `VMAF_ARM_CPU_FLAG_NEON`. ISA-parity matrix for psnr_hvs now
  closes: scalar + AVX2 + NEON. See
  [ADR-0160](docs/adr/0160-psnr-hvs-neon-bitexact.md).

- **`psnr_hvs` AVX2 SIMD path** (fork-local, backlog T3-5). x86_64
  users with AVX2 now get a vectorized Xiph/Daala 8×8 integer DCT
  (the hot inner kernel of psnr_hvs). Scalar + AVX2 paths are
  **byte-identical** on every Netflix golden pair — verified per-
  frame via `VMAF_CPU_MASK=0` vs default. **3.58× DCT speedup** on
  a microbenchmark (11.0 → 39.3 Mblocks/s at `-O3 -mavx2 -mfma`);
  real-world speedup scales with resolution (at 1080p × 3 planes
  the DCT is the dominant cost). Butterfly network vectorized 8
  rows in parallel via `__m256i` registers + matrix transpose
  between row and column passes. Float accumulators (means /
  variances / mask / error) kept scalar by construction for
  bit-exactness (ADR-0139 precedent). Includes new unit test
  `test_psnr_hvs_avx2.c` pinning the bit-exactness contract on 5
  reproducible inputs. NEON sister port landed as
  [ADR-0160](docs/adr/0160-psnr-hvs-neon-bitexact.md). See
  [ADR-0159](docs/adr/0159-psnr-hvs-avx2-bitexact.md).

- **`vmaf_cuda_state_free()` public API** (Netflix upstream issue
  [#1300](https://github.com/Netflix/vmaf/issues/1300)). New
  symbol in [`libvmaf/include/libvmaf/libvmaf_cuda.h`](libvmaf/include/libvmaf/libvmaf_cuda.h)
  that frees a `VmafCudaState` allocated by `vmaf_cuda_state_init()`.
  Must be called AFTER `vmaf_close()` on any VmafContext that
  imported the state. Mirrors the SYCL backend's
  `vmaf_sycl_state_free()` ownership pattern — caller allocates,
  framework imports by-value, caller frees after close. Safe to
  pass NULL. Closes the per-cycle host-memory leak where users
  had no public way to free the struct. See
  [ADR-0157](docs/adr/0157-cuda-preallocation-leak-netflix-1300.md).

### Fixed

- **CUDA preallocation memory leak** (Netflix upstream issue
  [#1300](https://github.com/Netflix/vmaf/issues/1300)). Users
  running CUDA-accelerated VMAF in init/preallocate/fetch/close
  loops saw GPU memory rise monotonically across cycles. Four
  framework-side leaks confirmed by ASan and fixed in this PR:
  (1) `VmafCudaState` heap allocation had no public free (fixed by
  the new `vmaf_cuda_state_free()` API above); (2)
  `vmaf_cuda_release()` destroyed the CUDA stream + context but
  never called `cuda_free_functions()` to release the dlopen'd
  driver function-pointer table — fixed by adding the free call
  after the existing `memset`, via a saved pointer; (3)
  `vmaf_ring_buffer_close()` locked `pthread_mutex` + freed the
  buffer but never unlocked or destroyed the mutex (POSIX UB) —
  fixed by adding `pthread_mutex_unlock` + `pthread_mutex_destroy`
  before the `free` calls; (4) adjacent cold-start leak in
  `init_with_primary_context()` where a retained CUDA primary
  context wasn't released if `cuStreamCreateWithPriority()` failed
  — fixed in the same commit. New GPU-gated reducer
  `test_cuda_preallocation_leak.c` does 10x init/preallocate/fetch
  /close cycles and reports zero framework-side leaked bytes under
  ASan (183 bytes remain in `libcuda.so.1`'s internal
  process-lifetime driver cache, matching SYCL behaviour).
  **Visible behaviour change**: every CUDA caller must now call
  `vmaf_cuda_state_free(cu_state)` AFTER `vmaf_close(vmaf)` —
  callers relying on informal `free(cu_state)` will double-free.
  Preserves ADR-0122 / ADR-0123 null-guards and ADR-0156
  `CHECK_CUDA_GOTO` cleanup paths verbatim. Closes backlog item
  T1-7. See [ADR-0157](docs/adr/0157-cuda-preallocation-leak-netflix-1300.md).

- **CUDA backend: graceful error propagation on `cuMemAlloc`
  OOM and all other CUDA failures** (Netflix upstream issue
  [#1420](https://github.com/Netflix/vmaf/issues/1420)). The
  `CHECK_CUDA` macro previously fired `assert(0)` on every
  CUDA error, which aborted the process — two concurrent
  VMAF-CUDA analyses crashed the second one immediately when
  it OOMed on `cuMemAlloc`. Wholesale refactor: replaced all
  178 `CHECK_CUDA(...)` call sites across 7 CUDA TUs
  (`common.c`, `picture_cuda.c`, `libvmaf.c`,
  `integer_motion_cuda.c`, `integer_vif_cuda.c`,
  `integer_adm_cuda.c`, `cuda_helper.cuh`) with two new
  macros — `CHECK_CUDA_GOTO(label)` (cleanup-aware) and
  `CHECK_CUDA_RETURN` (immediate-return) — that map `CUresult`
  to `-errno` via `vmaf_cuda_result_to_errno` and propagate
  the error through cleanup labels. `cuMemAlloc` OOM now
  returns `-ENOMEM`; resource exhaustion on
  `cuStreamCreate` / `cuEventCreate` returns `-EIO`;
  context / device-loss errors return `-ENODEV`; invalid
  handle / value / context errors return `-EINVAL`. Twelve
  `static` helper functions promoted from `void → int` to
  carry errors upward. New GPU-gated reducer in
  `test_cuda_buffer_alloc_oom.c` verifies `cuMemAlloc(1 TiB)`
  now returns `-ENOMEM` (was: `assert(0)`). Fixes the NDEBUG
  footgun (`assert(0)` was a no-op in release builds →
  silent continue into segfault). Preserves ADR-0122 /
  ADR-0123 null-guards on public entry points verbatim.
  Closes backlog item T1-6. See
  [ADR-0156](docs/adr/0156-cuda-graceful-error-propagation-netflix-1420.md).

### Documented (not fixed)

- **ADM `i4_adm_cm` int32 rounding overflow** (Netflix upstream
  issue [#955](https://github.com/Netflix/vmaf/issues/955)) is
  deliberately preserved. `add_bef_shift_flt[idx] = (1u <<
  (shift_flt[idx] - 1))` in `libvmaf/src/feature/integer_adm.c`
  scales 1–3 overflows `int32_t` (`1u << 31 = 0x80000000` wraps
  to `-2147483648`), so every `(prod + add_bef_shift) >> 32`
  subtracts 2^31 instead of adding it — ADM scales 1–3 biased
  low by ≈1 LSB per summed term. The buggy arithmetic is encoded
  in the Netflix golden assertions (project hard rule #1 /
  [ADR-0024](docs/adr/0024-netflix-golden-preserved.md)); fixing
  it unilaterally would diverge from every published VMAF number
  calibrated on these outputs. In-file warning comments, a
  rebase-notes invariant, and `AGENTS.md` pin the decision.
  Closes backlog item T1-8 as "verified present, deliberately
  preserved". See
  [ADR-0155](docs/adr/0155-adm-i4-rounding-deferred-netflix-955.md).

### Changed

- **`vmaf_score_pooled` returns `-EAGAIN` for pending features**
  (Netflix upstream issue
  [#755](https://github.com/Netflix/vmaf/issues/755)). Several
  extractors (integer_motion's motion2/motion3) write frame N's
  score retroactively when frame N+1 is extracted — or on flush
  for the tail. Previously `vmaf_score_pooled(vmaf, ..., i, i)`
  called immediately after `vmaf_read_pictures(vmaf, ref, dist,
  i)` returned `-EINVAL`, indistinguishable from programmer
  error. Now: `-EAGAIN` for the transient "valid but not yet
  written" case; `-EINVAL` stays reserved for genuine misuse
  (bad pointer, out-of-range, feature-name typo). Inline
  `vmaf_feature_vector_get_score` previously returned a literal
  `-1` for both cases; now splits the same way. **Visible
  behaviour change** for callers that want to distinguish
  transient from fatal — they can now branch on `-EAGAIN` and
  retry after one more read or after flush. Callers that treat
  any non-zero as fatal are unchanged. Drive-by: reserved
  `__VMAF_FEATURE_COLLECTOR_H__` header guard renamed to
  `VMAF_FEATURE_COLLECTOR_INCLUDED` (ADR-0141). 4-subtest
  reducer in `test_score_pooled_eagain.c` verified to fail
  pre-fix and pass post-fix. Closes backlog item T1-1. See
  [ADR-0154](docs/adr/0154-score-pooled-eagain-netflix-755.md).

- **`vmaf_read_pictures` now rejects non-monotonic indices with
  `-EINVAL`** (Netflix upstream issue
  [#910](https://github.com/Netflix/vmaf/issues/910)). The
  `integer_motion` / motion2 / motion3 extractors keep sliding-
  window state keyed by `index % N`, so submitting frames out of
  order or with duplicate indices silently corrupts their
  ring-buffers. The reported symptom was a missing
  `integer_motion2_score` on the last frame whenever submission
  order didn't match frame order. The fix is a monotonic-index
  guard at the API boundary (new `last_index` + `have_last_index`
  fields on `VmafContext`, checked inside the existing
  `read_pictures_validate_and_prep` helper from ADR-0146): strictly
  increasing indices are accepted (gaps fine); duplicates and
  regressions return `-EINVAL`. **Visible behaviour change**:
  duplicates / out-of-order submissions that previously produced
  silent-wrong-answer now fail with `-EINVAL` — well-defined
  rejection replaces ill-defined corruption. Zero impact on
  in-tree callers (vmaf CLI + test suite already iterate strictly
  increasing); downstream integrations that deliberately submit
  non-monotonic indices need to either track the next-index
  themselves or reset the context. 3-subtest reducer in
  `test_read_pictures_monotonic.c` verified to fail pre-fix and
  pass post-fix. Closes backlog item T1-2. See
  [ADR-0152](docs/adr/0152-vmaf-read-pictures-monotonic-index.md).

### Added

- **i686 (32-bit x86) build-only CI job** (reproduces Netflix
  upstream issue [#1481](https://github.com/Netflix/vmaf/issues/1481)).
  New matrix row in `.github/workflows/libvmaf-build-matrix.yml`
  (`Build — Ubuntu i686 gcc (CPU, no-asm)`) invokes
  `meson setup libvmaf libvmaf/build --cross-file=build-aux/i686-linux-gnu.ini -Denable_asm=false`,
  pinning the workaround documented in upstream's bug report.
  New cross-file `build-aux/i686-linux-gnu.ini` (gcc + `-m32`,
  `cpu_family = 'x86'`, `cpu = 'i686'`) + new install-deps step
  installing `gcc-multilib` + `g++-multilib`. Test + tox steps
  skipped for the i686 leg because meson marks cross-built tests
  as `SKIP 77` (the host can run i686 binaries natively but meson
  doesn't know that). Fixing the underlying AVX2
  `_mm256_extract_epi64` compile failure (24 call sites in
  `adm_avx2.c`) is **explicitly out of scope** — this entry adds
  the CI gate only. Closes backlog item T4-8. See
  [ADR-0151](docs/adr/0151-i686-ci-netflix-1481.md).

- **Windows MSYS2/MinGW CUDA build support** (port of Netflix
  upstream PR [#1472](https://github.com/Netflix/vmaf/pull/1472),
  birkdev, 2026-03-16, OPEN). Enables
  `-Denable_cuda=true -Denable_nvcc=true` on Windows with MSYS2 +
  MinGW-GCC host compiler + MSVC Build Tools + CUDA toolkit.
  Source-portability guards in CUDA headers + `.cu` files: drop
  `<pthread.h>` from `cuda/common.h`; DEVICE_CODE guards on
  `<ffnvcodec/*>` vs `<cuda.h>` in `cuda_helper.cuh` +
  `picture.h`; `#ifndef DEVICE_CODE` around `feature_collector.h`
  in 5 ADM `.cu` files. Meson build plumbing: `vswhere`-based
  `cl.exe` discovery (without adding it to PATH, which would
  break MinGW-GCC CPU build), Windows SDK + MSVC include path
  injection via `-I` flags to nvcc, CUDA version detection via
  `nvcc --version` (replaces `meson.get_compiler('cuda')` which
  needs MSVC as default C compiler). Fork carve-outs: keep
  positional (not `#ifndef __CUDACC__`) initializers in
  `integer_adm.h`; keep `pthread_dependency` on `cuda_static_lib`
  because `ring_buffer.c` still uses pthread directly; merge
  fork's ADR-0122 gencode coverage block with upstream's new
  nvcc-detect block. Drive-by: rename reserved `__VMAF_SRC_*_H__`
  header guards to `VMAF_SRC_*_INCLUDED` per ADR-0141. Linux
  CPU build 32/32 + Linux CUDA build 35/35 pass; Windows CUDA
  build not yet CI-validated (tracked as T7-3 — self-hosted
  Windows+CUDA runner enrollment). Closes backlog item T4-2.
  See [ADR-0150](docs/adr/0150-port-netflix-1472-cuda-windows.md).

### Fixed

- **FIFO-mode workfile/procfile opens no longer race-hang on slow
  systems** (port of Netflix upstream PR
  [#1376](https://github.com/Netflix/vmaf/pull/1376)). The Python
  harness under `python/vmaf/core/executor.py` +
  `python/vmaf/core/raw_extractor.py` previously waited for child
  processes to create named pipes via a 1-second `os.path.exists()`
  polling loop, which could time out on loaded CI / virtualised
  hosts. Replaced with `multiprocessing.Semaphore(0)` signalled
  by the child processes after `os.mkfifo(...)`; parent acquires
  with 5-second soft-timeout warning then blocks indefinitely.
  Applied to both the base `Executor` class and the
  `ExternalVmafExecutor`-style subclass. Fork carve-outs:
  upstream's `__version__ = "3.0.0" → "4.0.0"` bump is **not**
  applied (fork tracks its own versioning per ADR-0025); unused
  `from time import sleep` imports removed per ADR-0141.
  Closes backlog item T4-7. See
  [ADR-0149](docs/adr/0149-port-netflix-1376-fifo-semaphore.md).

### Changed

- **IQA reserved-identifier rename + touched-file lint cascade
  cleanup** (refactor, fork-local). Rename every `_iqa_*` /
  `struct _kernel` / `_ssim_int` / `_map_reduce` / `_map` /
  `_reduce` / `_context` / `_ms_ssim_map` / `_ssim_map` /
  `_ms_ssim_reduce` / `_ssim_reduce` / `_alloc_buffers` /
  `_free_buffers` symbol and the underscore-prefixed header
  guards (`_CONVOLVE_H_`, `_DECIMATE_H_`, `_SSIM_TOOLS_H_`,
  `__VMAF_MS_SSIM_DECIMATE_H__`) to their non-reserved
  spellings across the IQA tree (21 files). Sweeps the
  ADR-0141 touched-file lint cascade that surfaced
  (~40 pre-existing warnings across `ssim.c`, `ms_ssim.c`,
  `integer_ssim.c`, `iqa/*.{c,h}`, `convolve_*.{c,h}`,
  `test_iqa_convolve.c`): `static` / cross-TU NOLINT for
  `misc-use-internal-linkage`, `size_t` casts for
  `bugprone-implicit-widening-of-multiplication-result`,
  multi-decl splits, function-size refactors of `calc_ssim` /
  `compute_ssim` / `compute_ms_ssim` / `run_gauss_tests` via
  small named `static` helpers, `(void)` casts for unused
  feature-extractor lifecycle parameters, and scoped
  NOLINTBEGIN/END for `clang-analyzer-security.ArrayBound`
  false positives on the kernel-offset clamps and for
  `clang-analyzer-unix.Malloc` on test-helper allocations
  that leak by design at process exit. Bit-identical VMAF
  score on Netflix golden pair `src01_hrc00/01_576x324`
  (scalar vs SIMD, with `--feature float_ssim --feature
  float_ms_ssim` and the full `vmaf_v0.6.1` model). Closes
  backlog item T7-6. See
  [ADR-0148](docs/adr/0148-iqa-rename-and-cleanup.md).

- **Thread-pool job-object recycling** (perf, fork-local port of
  Netflix upstream PR [#1464](https://github.com/Netflix/vmaf/pull/1464),
  thread-pool portion only). `libvmaf/src/thread_pool.c` now recycles
  `VmafThreadPoolJob` slots via a mutex-protected free list rather
  than `malloc`/`free` on every enqueue, and stores payloads ≤ 64
  bytes inline in the job struct (`char inline_data[64]`) so the
  common-case enqueue path avoids a second allocation entirely.
  Adapted to the fork's `void (*func)(void *data, void **thread_data)`
  signature and per-worker `VmafThreadPoolWorker` data path (which
  upstream lacks). **~1.8–2.6× enqueue throughput** on a 500 000-job
  4-thread micro-benchmark; bit-identical VMAF scores between
  `--threads 4` and serial, and between `VMAF_CPU_MASK=0` and `=255`
  under `--threads 4`. Closes the thread-pool half of backlog T3-6
  (the AVX2 PSNR half was already covered by fork commit `81fcd42e`).
  See [ADR-0147](docs/adr/0147-thread-pool-job-pool.md).

- **Function-size NOLINT sweep** — refactored every
  `readability-function-size` NOLINT suppression in `libvmaf/src/` (20
  sites across 12 files: `dict.c`, `picture.c`, `picture_pool.c`,
  `predict.c`, `libvmaf.c`, `output.c`, `read_json_model.c`,
  `feature/feature_extractor.c`, `feature/feature_collector.c`,
  `feature/iqa/convolve.c`, `feature/iqa/ssim_tools.c`,
  `feature/x86/vif_statistic_avx2.c`) into small named `static`
  helpers. IQA / SIMD files use `static inline` helpers threaded
  through an explicit `struct vif_simd8_lane` to preserve the
  ADR-0138 / ADR-0139 bit-exactness invariants (per-lane scalar-float
  reduction, single-rounded float-mul → widen → double-add).
  Netflix-golden-pair VMAF score remains bit-identical between
  `VMAF_CPU_MASK=0` and `VMAF_CPU_MASK=255`. Zero new NOLINTs
  introduced. Drive-by fixes: TU-static `_calc_scale` →
  `iqa_calc_scale` for `bugprone-reserved-identifier`; tightened
  `calloc(w * h, ...)` widening; separated multi-declaration forms;
  `model_collection_parse_loop` now writes directly through
  `cfg_name` instead of the aliased `c->name` (drops the last
  `readability-non-const-parameter` NOLINT). See
  [ADR-0146](docs/adr/0146-nolint-sweep-function-size.md).

- **VIF AVX2 convolve: generalised for arbitrary filter widths** (port of
  Netflix upstream [`f3a628b4`](https://github.com/Netflix/vmaf/commit/f3a628b4),
  Kyle Swanson, 2026-04-21). `libvmaf/src/feature/common/convolution_avx.c`
  drops from 2,747 LoC of branch-unrolled kernels specialised to
  `fwidth ∈ {3, 5, 9, 17}` down to 247 LoC of a single parametric 1-D
  scanline pair. New `MAX_FWIDTH_AVX_CONV` ceiling in `convolution.h`
  lets the VIF AVX2 dispatch in `vif_tools.c` drop its hard-coded
  fwidth whitelist. Fork cleanup per ADR-0141: four helpers now
  `static`, strides widened to `ptrdiff_t` to eliminate
  `bugprone-implicit-widening-of-multiplication-result` on every
  pointer-offset site. Paired with a 10× loosening of the Netflix
  golden tolerance on two full-VMAF assertions
  (`VMAF_score`, `VMAFEXEC_score`: `places=2` → `places=1`),
  matching Netflix's own upstream test update. The generalised
  kernel's accumulation order differs at ULP scale vs the
  specialised ones; drift is orders of magnitude below perceptual
  discriminability. See
  [ADR-0143](docs/adr/0143-port-netflix-f3a628b4-generalized-avx-convolve.md).

### Added

- **VIF: configurable `vif_sigma_nsq` feature parameter**: port of
  Netflix upstream [`18e8f1c5`](https://github.com/Netflix/vmaf/commit/18e8f1c5)
  (Kyle Swanson, 2026-04-20) promoting VIF's hard-coded neural-noise
  variance `static const float sigma_nsq = 2` into a runtime-configurable
  double parameter `vif_sigma_nsq` (range `[0.0, 5.0]`, alias `snsq`,
  default `2.0`). Threaded through `compute_vif` → `vif_statistic_s` and
  the fork-local `vif_statistic_s_avx2` AVX2 variant (which upstream does
  not ship; its signature was extended in lockstep so both paths agree on
  the new 14-argument contract). Default-path scores bit-identical to
  pre-port master. Use via CLI:
  `vmaf --feature float_vif:snsq=2.5 ...` or per-model. See
  [ADR-0142](docs/adr/0142-port-netflix-18e8f1c5-vif-sigma-nsq.md).
- **Governance — Q2 2026 modernization ADRs**: four Proposed ADRs +
  four research digests scoping the next modernization workstreams
  (no implementation yet):
  - [ADR-0126](docs/adr/0126-ssimulacra2-extractor.md) /
    [Research-0003](docs/research/0003-ssimulacra2-port-sourcing.md):
    SSIMULACRA 2 feature extractor (port libjxl C++ reference).
  - [ADR-0127](docs/adr/0127-vulkan-compute-backend.md) /
    [Research-0004](docs/research/0004-vulkan-backend-design.md):
    Vulkan compute backend (volk + GLSL→SPIR-V + VMA, VIF
    pathfinder).
  - [ADR-0128](docs/adr/0128-embedded-mcp-in-libvmaf.md) /
    [Research-0005](docs/research/0005-embedded-mcp-transport.md):
    Embedded MCP server in libvmaf (SSE + UDS + stdio, flag-gated).
  - [ADR-0129](docs/adr/0129-tinyai-ptq-quantization.md) /
    [Research-0006](docs/research/0006-tinyai-ptq-accuracy-targets.md):
    Tiny-AI PTQ int8 (static + dynamic + QAT per-model via
    `model/registry.json`).
- **SIMD DX framework — `simd_dx.h` + upgraded `/add-simd-path` skill**:
  fork-internal header
  ([`libvmaf/src/feature/simd_dx.h`](libvmaf/src/feature/simd_dx.h))
  that codifies the ADR-0138 (widen-then-add) and ADR-0139 (per-lane
  scalar-double reduce) patterns as ISA-suffixed macros
  (`SIMD_WIDEN_ADD_F32_F64_AVX2_4L` / `_AVX512_8L` / `_NEON_4L`,
  `SIMD_ALIGNED_F32_BUF_*`, `SIMD_LANES_*`). Zero runtime overhead —
  each macro documents its scalar C equivalent and is guarded by the
  matching `__AVX2__` / `__AVX512F__` / `__ARM_NEON` ifdef. The
  `/add-simd-path` skill
  ([`.claude/skills/add-simd-path/SKILL.md`](.claude/skills/add-simd-path/SKILL.md))
  gained `--kernel-spec=widen-add-f32-f64|per-lane-scalar-double|none`,
  `--lanes=N`, and `--tail=scalar|masked` flags so new SIMD TUs
  scaffold from a short declaration instead of a cold copy-paste.
  Demonstrated on two real kernels in the same PR: a new bit-exact
  `iqa_convolve_neon`
  ([`libvmaf/src/feature/arm64/convolve_neon.c`](libvmaf/src/feature/arm64/convolve_neon.c))
  and a bit-exactness fix for `ssim_accumulate_neon` that mirrors the
  ADR-0139 x86 fix. Together they complete the SSIM / MS-SSIM SIMD
  coverage on aarch64. See
  [ADR-0140](docs/adr/0140-simd-dx-framework.md) +
  [research digest 0013](docs/research/0013-simd-dx-framework.md).
- **aarch64 cross-compile lane**:
  [`build-aux/aarch64-linux-gnu.ini`](build-aux/aarch64-linux-gnu.ini)
  meson cross-file for `aarch64-linux-gnu-gcc` +
  `qemu-aarch64-static`. The `test_iqa_convolve` meson target now
  covers `arm64` / `aarch64` alongside `x86_64` / `x86` so future NEON
  convolve changes gate on the same bit-exactness contract as the x86
  variants.
- **I18N / thread-safety**: `thread_locale.h/.c` cross-platform thread-local
  locale abstraction ported from upstream PR
  [Netflix/vmaf#1430](https://github.com/Netflix/vmaf/pull/1430) (Diego Nieto,
  Fluendo). `vmaf_write_output_{xml,json,csv,sub}`, `svm_save_model`,
  `vmaf_read_json_model`, and both SVM model parsers now switch the calling
  thread's locale to `"C"` for numeric I/O instead of using the
  process-global `setlocale` bracket. POSIX.1-2008 `uselocale` +
  `newlocale(LC_ALL_MASK)` on Linux/macOS/BSD; `_configthreadlocale` +
  per-thread `setlocale` on Windows; graceful no-op fallback elsewhere.
  Fixes a latent data-race under multi-threaded hosts (ffmpeg filter graphs
  with multiple VMAF instances, MCP server worker pools) where one thread's
  `setlocale(LC_ALL, "C")` bracket would clobber another thread's active
  locale mid-call. See
  [ADR-0137](docs/adr/0137-thread-local-locale-for-numeric-io.md).
- **Public API**: `vmaf_model_version_next(prev, &version)` iterator for
  enumerating the built-in VMAF model versions compiled into the
  library. Opaque-handle cursor — NULL to start, NULL-return to stop.
  Ports [Netflix#1424](https://github.com/Netflix/vmaf/pull/1424) with
  three correctness corrections (NULL-pointer arithmetic UB,
  off-by-one returning the sentinel, const-qualifier mismatches in the
  test); see [ADR-0135](docs/adr/0135-port-netflix-1424-expose-builtin-model-versions.md).
- **Build**: libvmaf now exports `libvmaf_dep` via `declare_dependency`
  and registers an `override_dependency('libvmaf', ...)` in
  `libvmaf/src/meson.build`, so the fork is consumable as a meson
  subproject with the standard `dependency('libvmaf')` idiom. Ports
  [Netflix#1451](https://github.com/Netflix/vmaf/pull/1451); see
  [ADR-0134](docs/adr/0134-port-netflix-1451-meson-declare-dependency.md).
- **Metric**: SSIMULACRA 2 scalar feature extractor
  ([`libvmaf/src/feature/ssimulacra2.c`](libvmaf/src/feature/ssimulacra2.c))
  — port of libjxl's perceptual similarity metric on top of the fork's
  YUV pipeline. Ingests YUV 4:2:0/4:2:2/4:4:4 at 8/10/12 bpc with a
  configurable YUV→RGB matrix (`yuv_matrix` option, BT.709 limited
  default), converts through linear RGB → XYB → 6-scale pyramid with
  SSIMMap + EdgeDiffMap + canonical 108-weight polynomial pool.
  Pyramid blur is a bit-close C port of libjxl's `FastGaussian`
  3-pole recursive IIR (`lib/jxl/gauss_blur.cc`,
  Charalampidis 2016 truncated-cosine approximation, k={1,3,5},
  zero-pad boundaries). Registered as feature `ssimulacra2` — one
  scalar per frame in `[0, 100]`, identity inputs return exactly
  `100.000000`. Scalar only; AVX2/AVX-512/NEON follow-ups are
  separate PRs. See
  [ADR-0130](docs/adr/0130-ssimulacra2-scalar-implementation.md) +
  [Research-0007](docs/research/0007-ssimulacra2-scalar-port.md).
- **CLI**: `--precision $spec` flag for score output formatting.
  - `N` (1..17) → `printf "%.<N>g"`
  - `max` / `full` → `"%.17g"` (round-trip lossless, opt-in)
  - `legacy` → `"%.6f"` (synonym for the default)
  - default (no flag) → `"%.6f"` (Netflix-compatible per ADR-0119;
    supersedes ADR-0006's original `%.17g` default)
- **Public API**: `vmaf_write_output_with_format()` accepts a `score_format`
  string; old `vmaf_write_output()` routes through the new function with
  `"%.6f"` default.
- **GPU backends**: SYCL/oneAPI backend (Lusoris + Claude); CUDA backend
  optimizations (decoupled buffer elimination, VIF rd_stride, ADM inline
  decouple).
- **Numerical correctness**: float ADM `sum_cube` and `csf_den_scale` use
  double-precision accumulation in scalar/AVX2/AVX512 paths to eliminate
  ~8e-5 drift between scalar and SIMD reductions.
- **MS-SSIM SIMD**: separable scalar-FMA decimate with AVX2 (8-wide),
  AVX-512 (16-wide), and NEON (4-wide) variants for the 9-tap 9/7
  biorthogonal wavelet LPF used by the MS-SSIM scale pyramid. Per-lane
  `_mm{256,512}_fmadd_ps` (x86) / `vfmaq_n_f32` (aarch64) with
  broadcast coefficients produces output byte-identical to the scalar
  reference; stride-2 horizontal deinterleave via
  `_mm256_shuffle_ps`+`_mm256_permute4x64_pd` (AVX2),
  `_mm512_permutex2var_ps` (AVX-512), and `vld2q_f32` (NEON). Runtime
  dispatch prefers AVX-512 > AVX2 > scalar on x86 and NEON > scalar on
  arm64. Netflix MS-SSIM golden passes at places=4 through every
  dispatched path; 10 synthetic `memcmp` cases (1x1 border, odd
  dimensions, 1920x1080) verify strict byte-equality in
  [`libvmaf/test/test_ms_ssim_decimate.c`](libvmaf/test/test_ms_ssim_decimate.c).
  See [ADR-0125](docs/adr/0125-ms-ssim-decimate-simd.md).
- **AI-agent scaffolding**: `.claude/` directory with 7 specialized review
  agents (c-, cuda-, sycl-, vulkan-, simd-, meson-reviewer, perf-profiler),
  18 task skills, hooks for unsafe-bash blocking and auto-format,
  `CLAUDE.md` + `AGENTS.md` onboarding, `docs/principles.md` (Power-of-10 +
  JPL-C-STD + CERT + MISRA).
- **Quality gates**: GitHub Actions workflows for CI (Netflix golden gate
  D24, sanitizers, cross-backend ULP), lint (clang-tidy, cppcheck,
  pre-commit), security (semgrep, CodeQL, gitleaks, dependency-review),
  supply-chain (SBOM, Sigstore keyless signing, SLSA L3 provenance).
- **Tiny AI**: nightly `bisect-model-quality` workflow
  ([`.github/workflows/nightly-bisect.yml`](.github/workflows/nightly-bisect.yml))
  runs `vmaf-train bisect-model-quality` against a deterministic
  synthetic placeholder cache
  ([`ai/testdata/bisect/`](ai/testdata/bisect/),
  reproducible from
  [`ai/scripts/build_bisect_cache.py`](ai/scripts/build_bisect_cache.py))
  and posts the verdict + per-step PLCC/SROCC/RMSE table to sticky
  tracker issue #40. Real DMOS-aligned cache swaps in via a follow-up;
  see [ADR-0109](docs/adr/0109-nightly-bisect-model-quality.md) +
  [Research-0001](docs/research/0001-bisect-model-quality-cache.md).
  Closes #4.
- **CI**: three DNN-enabled matrix legs in
  [`.github/workflows/libvmaf-build-matrix.yml`](.github/workflows/libvmaf-build-matrix.yml)
  — `Build — Ubuntu gcc (CPU) + DNN`, `Build — Ubuntu clang (CPU) + DNN`,
  `Build — macOS clang (CPU) + DNN`. Each leg installs ONNX Runtime
  (Linux: MS tarball pinned to 1.22.0; macOS: Homebrew) and runs the
  meson `dnn` test suite plus full `ninja test`. The two Linux legs
  are pinned to required status checks on `master`; the macOS leg
  stays `experimental: true` because Homebrew ORT floats. See
  [ADR-0120](docs/adr/0120-ai-enabled-ci-matrix-legs.md) +
  [`docs/rebase-notes.md` entry 0021](docs/rebase-notes.md).
- **CI**: two Windows GPU build-only matrix legs in
  [`.github/workflows/libvmaf-build-matrix.yml`](.github/workflows/libvmaf-build-matrix.yml)
  — `Build — Windows MSVC + CUDA (build only)` and
  `Build — Windows MSVC + oneAPI SYCL (build only)`. Both gate the
  MSVC build-portability of the CUDA host code and SYCL `vmaf_sycl_*`
  C-API entry points, respectively. No test step (windows-latest has
  no GPU). Both legs are pinned to required status checks on `master`.
  See [ADR-0121](docs/adr/0121-windows-gpu-build-only-legs.md) +
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: Win32 `pthread.h` compat shim at
  [`libvmaf/src/compat/win32/pthread.h`](libvmaf/src/compat/win32/pthread.h)
  — header-only, maps the in-use pthread subset (mutex / cond / thread
  create+join+detach + `PTHREAD_MUTEX_INITIALIZER` /
  `PTHREAD_COND_INITIALIZER`) onto Win32 SRWLOCK + CONDITION_VARIABLE +
  `_beginthreadex`. Wired in via a new `pthread_dependency` in
  `libvmaf/meson.build`, gated on `cc.check_header('pthread.h')`
  failing — POSIX and MinGW (winpthreads) builds are untouched. Lets
  the Windows MSVC GPU legs from ADR-0121 actually compile the libvmaf
  core (~14 TUs `#include <pthread.h>` unconditionally). Pattern
  mirrors the long-standing `compat/gcc/stdatomic.h` shim. nvcc fatbin
  and icpx SYCL `custom_target`s additionally thread the shim include
  path through `cuda_extra_includes` / `sycl_inc_flags` on Windows
  (custom targets bypass meson's `dependencies:` plumbing).
- **Build**: SYCL Windows host-arg handling in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build) — `icpx-cl`
  on Windows targets `x86_64-pc-windows-msvc` and rejects `-fPIC`.
  `sycl_common_args` / `sycl_feature_args` now route the flag through
  `sycl_pic_arg = host_machine.system() != 'windows' ? ['-fPIC'] : []`
  instead of hard-coding it. PIC is the default for Windows DLLs, so
  dropping the flag is the correct build-system fix, not a workaround.
- **Build**: SYCL Windows source portability — four MSVC C++
  blockers fixed so `icpx-cl` compiles the SYCL TUs.
  (1) [`libvmaf/src/ref.h`](libvmaf/src/ref.h) +
  [`libvmaf/src/feature/feature_extractor.h`](libvmaf/src/feature/feature_extractor.h)
  (UPSTREAM) gained an `#if defined(__cplusplus) && defined(_MSC_VER)`
  branch that pulls `atomic_int` via `using std::atomic_int;` —
  MSVC's `<stdatomic.h>` only surfaces the C11 typedefs in
  `namespace std::` under C++, while gcc/clang expose them globally
  via a GNU extension. POSIX paths fall through to the original
  `<stdatomic.h>` line; ABI unchanged. (2)
  [`libvmaf/src/sycl/d3d11_import.cpp`](libvmaf/src/sycl/d3d11_import.cpp)
  switched `<libvmaf/log.h>` (non-existent) to `"log.h"` (the actual
  internal header). (3)
  [`libvmaf/src/sycl/dmabuf_import.cpp`](libvmaf/src/sycl/dmabuf_import.cpp)
  moved `<unistd.h>` inside `#if HAVE_SYCL_DMABUF` — POSIX `close()`
  is only used in the VA-API path, so non-DMA-BUF hosts (Windows
  MSVC, macOS) no longer fail with `'unistd.h' file not found`. (4)
  [`libvmaf/src/sycl/common.cpp`](libvmaf/src/sycl/common.cpp)
  replaced POSIX `clock_gettime(CLOCK_MONOTONIC)` with
  `std::chrono::steady_clock` — guaranteed monotonic by the C++
  standard and portable on every supported host. All four preserve
  POSIX/Linux behaviour bit-identically. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: CUDA Windows source portability — fifth MSVC blocker
  fixed on the CUDA leg's CPU SIMD compile path.
  [`libvmaf/src/feature/x86/motion_avx2.c`](libvmaf/src/feature/x86/motion_avx2.c)
  (UPSTREAM) line 529 indexed an `__m256i` directly
  (`final_accum[0] + ... + final_accum[3]`) — gcc/clang allow this
  via the GNU vector extension, MSVC rejects it with `C2088:
  built-in operator '[' cannot be applied to an operand of type
  '__m256i'`. Replaced with four `_mm256_extract_epi64` calls,
  summed — bit-exact lane sum on every compiler. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: x86 SIMD Windows source portability — sweep that
  finishes the MSVC compile of the libvmaf CPU SIMD layer.
  Round-19 surfaced the same vector-extension pattern at 19 more
  call sites plus 6 GCC-style `(__m256i)x` casts.
  [`libvmaf/src/feature/x86/adm_avx2.c`](libvmaf/src/feature/x86/adm_avx2.c)
  (UPSTREAM) had 6 lines using
  `(__m256i)(_mm256_cmp_ps(...))` casts (replaced with
  `_mm256_castps_si256(...)`) and 12 sites of `__m128i[N]`
  lane-extract reductions (replaced with `_mm_extract_epi64`).
  [`libvmaf/src/feature/x86/adm_avx512.c`](libvmaf/src/feature/x86/adm_avx512.c)
  (UPSTREAM) had 6 sister lane-extract reductions on the
  AVX-512 paths.
  [`libvmaf/src/feature/x86/motion_avx512.c`](libvmaf/src/feature/x86/motion_avx512.c)
  (UPSTREAM, ported from PR #1486) had one final lane-extract
  reduction. All 19 + 6 fixes are bit-exact rewrites — gcc/clang
  emit identical vextract+padd sequences either way.
  Additionally
  [`libvmaf/src/sycl/d3d11_import.cpp`](libvmaf/src/sycl/d3d11_import.cpp)
  switched from C-style COBJMACROS helpers
  (`ID3D11Device_CreateTexture2D`, etc.) to C++ method-call syntax
  (`device->CreateTexture2D`) because d3d11.h gates COBJMACROS
  behind `!defined(__cplusplus)` and the TU compiles as C++
  under icpx-cl. ABI-equivalent. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: x86 SIMD alignment specifier — round-20 swap from
  GCC trailing `__attribute__((aligned(N)))` to C11-standard
  leading `_Alignas(N)` across 17 scratch-buffer sites in
  `vif_statistic_avx2.c` (UPSTREAM), `ansnr_avx{2,512}.c`
  (UPSTREAM), `float_adm_avx{2,512}.c` (UPSTREAM),
  `float_psnr_avx{2,512}.c` (UPSTREAM) and `ssim_avx{2,512}.c`
  (UPSTREAM). Same alignment guarantee, MSVC-portable
  (`/std:c11`). The pre-existing portable `ALIGNED(x)` macro in
  `vif_avx{2,512}.c` was already MSVC-clean and remains untouched.
- **Build**: `mkdirp` Windows portability —
  [`libvmaf/src/feature/mkdirp.c`](libvmaf/src/feature/mkdirp.c)
  and
  [`libvmaf/src/feature/mkdirp.h`](libvmaf/src/feature/mkdirp.h)
  (third-party MIT-licensed micro-library) gate `<unistd.h>` to
  non-Windows, add `<direct.h>` + `_mkdir` on MSVC, and provide a
  local `mode_t` typedef (MSVC's `<sys/types.h>` doesn't declare
  it). The `mode` argument is silently ignored on the Windows
  path — same behaviour as before for POSIX callers. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: round-21 MSVC mop-up —
  [`libvmaf/src/feature/x86/adm_avx512.c`](libvmaf/src/feature/x86/adm_avx512.c)
  (UPSTREAM) adds six more `_mm_extract_epi64` rewrites at lines
  2128 / 2135 / 2142 / 2589 / 2595 / 2601 that the round-19 sweep
  missed (bit-exact).
  [`libvmaf/src/log.c`](libvmaf/src/log.c) (UPSTREAM) gates
  `<unistd.h>` to non-Windows and pulls `_isatty` / `_fileno` from
  `<io.h>` on MSVC via macro redirection; the single `isatty(fileno
  (stderr))` call site compiles unchanged on every platform.
  See [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **CI**: `.github/workflows/lint-and-format.yml` pre-commit job
  now checks out with `lfs: true`. Without it `model/tiny/*.onnx`
  lands as LFS pointer stubs and pre-commit's "changes made by
  hooks" reporter flags the stubs as pre-commit-induced
  modifications against HEAD's resolved blobs, failing the job
  even though no hook touched them. See
  [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md).
- **Build**: round-21e MSVC mop-up — the Windows MSVC legs now
  build the full tree (CLI tools, unit tests, `libvmaf.dll`)
  instead of the earlier short cut of skipping tools / tests.
  Source changes:
  (i) eight C99 variable-length arrays converted to compile-time
  constants or heap allocations —
  [`libvmaf/src/predict.c:385,453`](libvmaf/src/predict.c),
  [`libvmaf/src/libvmaf.c:1741`](libvmaf/src/libvmaf.c),
  [`libvmaf/src/read_json_model.c:517,520`](libvmaf/src/read_json_model.c),
  [`libvmaf/test/test_feature_extractor.c:56`](libvmaf/test/test_feature_extractor.c),
  [`libvmaf/test/test_cambi.c:254`](libvmaf/test/test_cambi.c),
  [`libvmaf/test/test_pic_preallocation.c:382,506`](libvmaf/test/test_pic_preallocation.c);
  (ii) fork-added POSIX/GNU `getopt_long` shim at
  [`libvmaf/tools/compat/win32/`](libvmaf/tools/compat/win32/)
  (header + ~260-line companion source) declared via a single
  `getopt_dependency` in
  [`libvmaf/meson.build`](libvmaf/meson.build) that
  auto-propagates the .c into the `vmaf` CLI and
  `test_cli_parse`;
  (iii) `pthread_dependency` threaded through the eleven test
  targets in
  [`libvmaf/test/meson.build`](libvmaf/test/meson.build)
  that transitively include `<pthread.h>` via
  `feature_collector.h`;
  (iv) `<unistd.h>` → `<io.h>` redirection
  (`isatty`/`fileno` → `_isatty`/`_fileno`) added to
  [`libvmaf/tools/vmaf.c`](libvmaf/tools/vmaf.c);
  (v) `<unistd.h>` → `<windows.h>` + `Sleep` macros
  added to
  [`libvmaf/test/test_ring_buffer.c`](libvmaf/test/test_ring_buffer.c)
  and
  [`libvmaf/test/test_pic_preallocation.c`](libvmaf/test/test_pic_preallocation.c)
  for `usleep` / `sleep`;
  (vi) `__builtin_clz` / `__builtin_clzll` MSVC fallback via
  `__lzcnt` / `__lzcnt64` extracted into
  [`libvmaf/src/feature/compat_builtin.h`](libvmaf/src/feature/compat_builtin.h)
  and included from the three TUs that use the builtin
  (`integer_adm.c`, `x86/adm_avx2.c`, `x86/adm_avx512.c`);
  (vii) `extern "C"` wrap added around
  `#include "log.h"` in
  [`libvmaf/src/sycl/d3d11_import.cpp`](libvmaf/src/sycl/d3d11_import.cpp)
  so `vmaf_log` resolves against the C-linkage symbol
  produced by `log.c` when this .cpp TU gets pulled into
  a SYCL-enabled test executable by icpx-cl. Upstream
  `log.h` has no `__cplusplus` guard; the wrap keeps the
  fork-local fix inside the fork-added .cpp instead of
  touching the shared header.
  Workflow change: both Windows MSVC matrix legs now pass
  `--default-library=static` in `meson_extra` because libvmaf's
  public API carries no `__declspec(dllexport)` — a vanilla
  MSVC shared build produces an empty import lib and tools
  fail with `LNK1181`. Mirrors the MinGW leg's static-link
  choice. Both MSVC CUDA and MSVC SYCL legs validated
  locally end-to-end on a Windows Server 2022 VM with
  CUDA 13.0, oneAPI 2025.3, and Level Zero loader v1.18.5
  prior to push.
  See [`docs/rebase-notes.md` entry 0022](docs/rebase-notes.md)
  paragraphs (h)–(p).
- **CUDA**: out-of-the-box GPU coverage for Ampere `sm_86` (RTX 30xx)
  and Ada `sm_89` (RTX 40xx). The gencode array in
  [`libvmaf/src/meson.build`](libvmaf/src/meson.build) now
  unconditionally emits cubins for `sm_75` / `sm_80` / `sm_86` /
  `sm_89` plus a `compute_80` PTX backward-JIT fallback, independent
  of host `nvcc` version. Upstream Netflix only shipped cubins at Txx
  major boundaries, so Ampere-`sm_86` / Ada-`sm_89` ran on a
  `compute_90` PTX that cannot JIT backward and fell over at
  kernel-load time on every consumer 3080/3090/4070/4090. See
  [ADR-0122](docs/adr/0122-cuda-gencode-coverage-and-init-hardening.md)
  and [`docs/rebase-notes.md` entry 0023](docs/rebase-notes.md).
- **CUDA**: actionable init-failure logging in
  [`libvmaf/src/cuda/common.c`](libvmaf/src/cuda/common.c). When
  `cuda_load_functions()` (the `nv-codec-headers` dlopen wrapper
  around `libcuda.so.1`) fails, `vmaf_cuda_state_init()` now emits a
  multi-line message naming the missing library, the loader-path
  check command (`ldconfig -p | grep libcuda`), and the docs section
  at
  [`docs/backends/cuda/overview.md#runtime-requirements`](docs/backends/cuda/overview.md#runtime-requirements).
  A parallel message on `cuInit(0)` failure distinguishes
  driver-load failure from userspace/kernel version skew. Also fixes
  a pre-existing leak on both error paths (`cuda_free_functions()` +
  `free(c)` + `*cu_state = NULL`). See
  [ADR-0122](docs/adr/0122-cuda-gencode-coverage-and-init-hardening.md).
- **Automated rule-enforcement for four process ADRs**: new workflow
  [`.github/workflows/rule-enforcement.yml`](.github/workflows/rule-enforcement.yml)
  plus a pre-commit `check-copyright` hook
  ([`scripts/ci/check-copyright.sh`](scripts/ci/check-copyright.sh)) close
  the "rule-without-a-check" gap on
  [ADR-0100](docs/adr/0100-project-wide-doc-substance-rule.md),
  [ADR-0105](docs/adr/0105-copyright-handling-dual-notice.md),
  [ADR-0106](docs/adr/0106-adr-maintenance-rule.md), and
  [ADR-0108](docs/adr/0108-deep-dive-deliverables-rule.md). The
  ADR-0108 six-deliverable checklist is **blocking**; the other
  three are advisory comments because their predicates require
  human judgement (pure-refactor exemption, ADR-triviality call,
  copyright-template choice). Upstream-port PRs (`port:` title /
  `port/` branch) are exempt. Reviewer documentation at
  [`docs/development/automated-rule-enforcement.md`](docs/development/automated-rule-enforcement.md).
  First `--all-files` pass also backfilled 18 pre-existing missing
  headers (13 upstream C files Netflix 2016–2026, 4 fork-authored
  NEON sources + `python/compat/config.h` Lusoris+Claude 2026);
  `libvmaf/src/pdjson.{c,h}` (vendored JSON parser) and
  `python/vmaf/matlab/` (upstream MATLAB MEX) are excluded from
  the hook rather than receiving synthetic headers. See
  [ADR-0124](docs/adr/0124-automated-rule-enforcement.md) and
  [Research-0002](docs/research/0002-automated-rule-enforcement.md).

### Changed

- **Upstream port — ADM** (Netflix `966be8d5`, fork PR #44, merged
  `d06dd6cf`): integer ADM kernels + AVX2/AVX-512 SIMD paths +
  `barten_csf_tools.h` ported wholesale; `i4_adm_cm` signature extended
  from 8 to 13 args. Netflix golden VMAF mean unchanged at
  `76.66890` (places=4 OK). See
  [`docs/rebase-notes.md` entry 0012](docs/rebase-notes.md).
- **Upstream port — motion** (Netflix PR #1486 head `2aab9ef1`, sister
  to ADM port): integer motion + AVX2/AVX-512 paths +
  `motion_blend_tools.h` ported wholesale; new `integer_motion3`
  sub-feature appears in the default VMAF model output. Golden mean
  shifts `76.66890` → `76.66783` (within `places=2` tolerance the
  upstream PR loosened to). See
  [`docs/rebase-notes.md` entry 0013](docs/rebase-notes.md).
- Python diagnostic output (`Result._get_perframe_score_str`) now emits
  scores at `%.17g` instead of `%.6f` for round-trip reproducibility.
- Copyright headers across Netflix-authored sources updated `2016-2020` →
  `2016-2026`.
- **CI hygiene — Node 24 stragglers**: finish the `@v7` bump left over
  after the rename sweep (rebase-notes 0019/0020). `scorecard.yml`
  SHA-pinned `actions/upload-artifact@330a01c4... # v5` →
  `@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1`, and
  `nightly-bisect.yml` `actions/setup-python@v5` → `@v6`. Resolves the
  last `Node.js 20 actions are deprecated` runner warnings ahead of
  the **2026-06-02** forced-Node-24 cutover (full Node-20 removal
  2026-09-16). Every other workflow was already bumped on master.
- **Engineering process**: every fork-local PR now ships the six
  deep-dive deliverables (research digest, decision matrix in the
  ADR, `AGENTS.md` invariant note, reproducer command, fork-changelog
  entry, rebase note) in the same PR. New scaffolding:
  [`docs/research/`](docs/research/),
  [`docs/rebase-notes.md`](docs/rebase-notes.md), updated
  [`PR template`](.github/PULL_REQUEST_TEMPLATE.md). See
  [ADR-0108](docs/adr/0108-deep-dive-deliverables-rule.md). Existing
  fork-local PRs have a one-shot rebase-notes backfill (10 grouped
  workstream entries) so the next upstream sync starts from a
  populated ledger. Closes #38.
- **Coverage gate**: end-to-end overhaul. (1) Build with
  `-fprofile-update=atomic` (CPU + GPU jobs) so parallel meson tests
  stop racing the `.gcda` counters on instrumented SIMD inner loops —
  eliminates the "Unexpected negative count for vif_avx2.c:673"
  geninfo hard-fail. (2) Run `meson test --num-processes 1` in the
  coverage steps so multiple test binaries don't concurrently merge
  their counters into the same `.gcda` files for the shared
  `libvmaf.so` (the on-exit merge is a multi-PROCESS race the atomic
  flag does not cover). (3) Switch `lcov` → `gcovr`: `gcovr`
  deduplicates `.gcno` files belonging to the same source compiled
  into multiple targets, fixing the `dnn_api.c — 1176%` over-count
  that surfaced after (1)+(2) on the first attempt. (4) Install
  ONNX Runtime in the coverage job and build with
  `-Denable_dnn=enabled` so `libvmaf/src/dnn/*.c` contribute real
  coverage instead of stubs (the 85% per-critical-file gate was
  previously unmeasurable). Coverage artifact is now
  `coverage.{xml,json,txt}` (Cobertura + gcovr JSON summary +
  human-readable text). (5) Carve `vmaf_use_tiny_model` out of
  `libvmaf/src/dnn/dnn_api.c` into a new
  `libvmaf/src/dnn/dnn_attach_api.c` so the unit-test binaries —
  which pull in `dnn_sources` for `feature_lpips.c` but never link
  `libvmaf.c` — don't end up with an undefined reference to
  `vmaf_ctx_dnn_attach` once `enable_dnn=enabled` activates the real
  bodies. The new TU is wired into `libvmaf.so` only via a separate
  `dnn_libvmaf_only_sources` list.
  See [ADR-0110](docs/adr/0110-coverage-gate-fprofile-update-atomic.md)
  (race fixes) and
  [ADR-0111](docs/adr/0111-coverage-gate-gcovr-with-ort.md)
  (gcovr + ORT) and
  [`docs/rebase-notes.md` entry 0014](docs/rebase-notes.md).
- **Lint scope**: upstream-mirror Python tests under `python/test/*.py`
  are now linted at the same standard as fork-added code. Mechanical
  Black + isort reformat applied to the four Netflix golden test
  files (`feature_extractor_test.py`, `quality_runner_test.py`,
  `vmafexec_test.py`, `vmafexec_feature_extractor_test.py`) — no
  assertion values changed; imports regrouped, line wrapping
  normalised. `python/test/resource/` (binary fixtures) remains
  excluded. Per user direction "don't skip linting on upstream
  things": `/sync-upstream` and `/port-upstream-commit` will
  re-trigger lint failures whenever upstream rewrites these files,
  and the fix is another in-tree reformat pass — never an exclusion.
  See [`docs/rebase-notes.md` entry 0014](docs/rebase-notes.md).
- **Coverage Gate annotations cleanup**: `actions/upload-artifact@v5|@v6
  → @v7` (and `actions/download-artifact@v5 → @v7` on supply-chain.yml)
  across every workflow under `.github/workflows/`, ahead of GitHub's
  2026-06-02 forced-Node-24 cutoff that turns the current Node 20
  deprecation banner into a hard error. Coverage Gate gcovr
  invocations also pipe stderr through `grep -vE 'Ignoring
  (suspicious|negative) hits' ... || true` so the chatty annotation
  for legitimately-large hit counts on tight inner loops (e.g.
  `ansnr_tools.c:207` at ~4.93 G hits across an HD multi-frame
  coverage suite) is dropped without losing the underlying data —
  `--gcov-ignore-parse-errors=suspicious_hits.warn` still tells
  gcovr to accept the count, only the annotation is filtered. The
  filter regex is anchored to gcov's exact warning prefix, so any
  *other* gcovr warning still surfaces. See
  [ADR-0117](docs/adr/0117-coverage-gate-warning-noise-suppression.md)
  and [`docs/rebase-notes.md` entry 0015](docs/rebase-notes.md).

### Fixed

- **SSIM / MS-SSIM NEON bit-exactness to scalar**: fork-local
  `ssim_accumulate_neon`
  ([`libvmaf/src/feature/arm64/ssim_neon.c`](libvmaf/src/feature/arm64/ssim_neon.c))
  previously carried the same ~0.13 float-ULP drift on
  `float_ms_ssim` / ~6 × 10⁻⁸ drift on `float_ssim` that ADR-0139
  fixed for AVX2 / AVX-512 — it was never surfaced because CI has no
  aarch64 runner. The NEON accumulator now computes the float-valued
  intermediates in vector float (`float32x4_t`) and spills to
  `SIMD_ALIGNED_F32_BUF_NEON(4)` buffers so the
  `2.0 * mu1 * mu2 + C1` numerator + division + `l*c*s` triple
  product run per-lane in scalar double, matching the x86 fix. Also
  plugged the aarch64 `iqa_convolve` gap — there was no NEON convolve
  at all before this PR; the VIF / ADM features used the scalar path
  on aarch64 while x86 ran AVX2 / AVX-512. Verified bit-identical
  under `qemu-aarch64-static` on both Netflix `src01_hrc00/01_576x324`
  and `checkerboard_1920_1080_10_3` pairs at `--precision max`. See
  [ADR-0140](docs/adr/0140-simd-dx-framework.md) +
  [research digest 0013](docs/research/0013-simd-dx-framework.md).
- **SSIM / MS-SSIM AVX2 + AVX-512 bit-exactness to scalar**: fork-local
  `ssim_accumulate_avx2` / `ssim_accumulate_avx512`
  ([`libvmaf/src/feature/x86/ssim_avx2.c`](libvmaf/src/feature/x86/ssim_avx2.c),
  [`libvmaf/src/feature/x86/ssim_avx512.c`](libvmaf/src/feature/x86/ssim_avx512.c))
  previously computed the `l`, `c`, `s` factors as vector float and
  produced the `l * c * s` triple product in float before accumulating
  to double — that diverged from the scalar reference by ~0.13 float
  ULPs (8th decimal) on `float_ms_ssim`, because scalar evaluates
  `2.0 * mu1 * mu2 + C1` and `2.0 * srsc + C2` in double (the literal
  `2.0` is a C `double`) and runs `lv * cv * sv` as three double
  multiplies. The SIMD accumulators now compute the float-valued
  intermediates (`srsc`, denominators, `sv`) in vector float and do
  the double-precision numerator + division + triple product per-lane
  in scalar double inside an 8/16-wide inner loop, matching scalar
  byte-for-byte. Verified: scalar = AVX2 = AVX-512 bit-identical at
  `--precision max` on both Netflix `src01_hrc00/01_576x324` and
  `checkerboard_1920_1080_10_3` pairs. `ssim_precompute_*` and
  `ssim_variance_*` were already bit-exact (pure elementwise float
  ops). Companion fix to the new bit-exact `_iqa_convolve_avx2/512`
  dispatch. See
  [ADR-0139](docs/adr/0139-ssim-simd-bitexact-double.md) +
  [ADR-0138](docs/adr/0138-iqa-convolve-avx2-bitexact-double.md).
- **CUDA multi-session `vmaf_cuda_picture_free` assertion-0 crash**:
  two or more concurrent CUDA sessions freeing pictures tripped
  `Assertion 0 failed` inside the driver because
  `cuMemFreeAsync(ptr, stream)` enqueued the free on a stream that
  was destroyed two statements later. The fork swaps the async call
  for synchronous `cuMemFree` at
  [`libvmaf/src/cuda/picture_cuda.c:247`](libvmaf/src/cuda/picture_cuda.c#L247);
  the preceding `cuStreamSynchronize` already removed any async
  overlap so perf is unchanged. Ports
  [Netflix#1382](https://github.com/Netflix/vmaf/pull/1382)
  (tracking [Netflix#1381](https://github.com/Netflix/vmaf/issues/1381));
  see [ADR-0131](docs/adr/0131-port-netflix-1382-cumemfree.md).
- **`vmaf_feature_collector_mount_model` list-corruption on ≥3
  models**: the upstream body advanced the `*head` pointer-to-pointer
  instead of walking a local cursor, overwriting the head element
  with its own successor and losing every entry past the second.
  Fork rewrites mount/unmount in
  [`libvmaf/src/feature/feature_collector.c`](libvmaf/src/feature/feature_collector.c)
  with a correct traversal, and `unmount_model` now returns
  `-ENOENT` (not `-EINVAL`) when the requested model isn't mounted
  so callers can distinguish misuse from not-found. Test coverage
  extended to a 3-element mount / unmount sequence. Ports
  [Netflix#1406](https://github.com/Netflix/vmaf/pull/1406);
  see [ADR-0132](docs/adr/0132-port-netflix-1406-feature-collector-model-list.md).
- **`KBND_SYMMETRIC` sub-kernel-radius out-of-bounds reflection**:
  upstream's 2-D symmetric boundary extension reflected the index a
  single time, which leaves out-of-bounds values whenever the input
  dimension is smaller than the kernel half-width (for the 9-tap
  MS-SSIM LPF, `n ≤ 3`). The fork rewrites `KBND_SYMMETRIC` in
  [`libvmaf/src/feature/iqa/convolve.c`](libvmaf/src/feature/iqa/convolve.c)
  and the scalar / AVX2 / AVX-512 / NEON `ms_ssim_decimate_mirror`
  helpers into the period-based form (`period = 2*n`) that bounces
  correctly for any offset. Netflix golden outputs are unchanged
  because 576×324 and 1920×1080 inputs never exercise the
  sub-kernel-radius regime. See
  [`docs/development/known-upstream-bugs.md`](docs/development/known-upstream-bugs.md)
  and [ADR-0125](docs/adr/0125-ms-ssim-decimate-simd.md).
- **`adm_decouple_s123_avx512` LTO+release SEGV**: the stack array
  `int64_t angle_flag[16]` is read via two `_mm512_loadu_si512`
  calls. Under `--buildtype=release -Db_lto=true`, link-time
  alignment inference promotes them to `vmovdqa64`, which faults
  because the C default stack alignment for `int64_t[16]` is 8
  bytes. Annotating the array with `_Alignas(64)` at
  [`libvmaf/src/feature/x86/adm_avx512.c:1317`](libvmaf/src/feature/x86/adm_avx512.c#L1317)
  keeps both the unaligned source form and the LTO-promoted aligned
  form correct. Debug / no-LTO builds, and every CI sanitizer job,
  are unaffected.
- **`test_pic_preallocation` VmafModel leaks**:
  `test_picture_pool_basic` / `_small` / `_yuv444` loaded a
  `VmafModel` via `vmaf_model_load` and never freed it, so
  LeakSanitizer reported 208 B direct + 23 KiB indirect per test.
  Paired each load with `vmaf_model_destroy(model)` in
  [`libvmaf/test/test_pic_preallocation.c`](libvmaf/test/test_pic_preallocation.c).
- **`libvmaf_cuda` ffmpeg filter segfault on first frame**: external
  reporter (2026-04-19) hit a SIGSEGV in `vmaf_ref_fetch_increment` on
  every invocation of ffmpeg's `libvmaf_cuda` filter against the fork's
  master build. Root cause is a three-commit composition: upstream
  `32b115df` (2026-04-07) added the experimental `VMAF_PICTURE_POOL`
  with an always-live `vmaf->prev_ref` slot; upstream `f740276a`
  (2026-04-09) moved the `vmaf_picture_ref(&vmaf->prev_ref, ref)` tail
  onto the non-threaded path without guarding against `ref->ref ==
  NULL`; fork commit `65460e3a` ([ADR-0104](docs/adr/0104-picture-pool-always-on.md))
  dropped the `VMAF_PICTURE_POOL` meson gate for ABI stability
  (+10 fps CPU gain), exposing the unguarded deref to every default
  build. On the CUDA-device-only extractor set that the ffmpeg filter
  registers, `rfe_hw_flags` returns `HW_FLAG_DEVICE` only,
  `translate_picture_device` early-returns without downloading, and
  `ref_host` stays zero-initialised — the subsequent
  `vmaf_picture_ref(&prev_ref, &ref_host)` deref'd `NULL`. Fix is a
  narrow null-guard at `libvmaf/src/libvmaf.c:1428`
  (`if (ref && ref->ref) vmaf_picture_ref(...)`). Semantically correct,
  not merely defensive: the only `VMAF_FEATURE_EXTRACTOR_PREV_REF`
  consumer is CPU `integer_motion_v2`, which is never registered
  alongside a pure-CUDA set. SYCL is unaffected (`vmaf_read_pictures_sycl`
  does not touch `prev_ref`). Always-on picture pool stays. See
  [ADR-0123](docs/adr/0123-cuda-post-cubin-load-regression-32b115df.md);
  follow-up item to port the null-guard upstream to Netflix/vmaf.
- **VIF `init()` fail-path leak**: `libvmaf/src/feature/integer_vif.c`'s
  `init()` carves one `aligned_malloc` into the VifBuffer sub-pointers by
  walking a `uint8_t *data` cursor forward through the allocation. When
  `vmaf_feature_name_dict_from_provided_features` returned NULL, the
  fail-path called `aligned_free(data)` on the *advanced* cursor — not a
  valid `aligned_malloc` return — leaking the whole block and passing a
  garbage pointer to `free`. Fail path now frees `s->public.buf.data`,
  the saved base pointer. Ported from Netflix upstream PR
  [#1476](https://github.com/Netflix/vmaf/pull/1476); the companion
  void*→uint8_t* UB portability fix from that PR is already on master
  (commit `b0a4ac3a`, rebase-notes 0022 §e).
- **CLI precision default reverted to `%.6f` (Netflix-compat)**: ADR-0006
  shipped `%.17g` as the default for round-trip-lossless output, but
  several Netflix golden tests in `python/test/command_line_test.py`,
  `vmafexec_test.py` etc. do *exact-string* matches against XML output
  (not `assertAlmostEqual`), so the wider default broke the gate. Default
  now matches upstream Netflix byte-for-byte; `--precision=max` (alias
  `full`) is the explicit opt-in for `%.17g`. `--precision=legacy` is
  preserved as a synonym for the (new) default. Library
  `vmaf_write_output_with_format(..., NULL)` and `python/vmaf/core/result.py`
  formatters revert in lockstep. See
  [ADR-0119](docs/adr/0119-cli-precision-default-revert.md) (supersedes
  [ADR-0006](docs/adr/0006-cli-precision-17g-default.md)). Latent on
  master 2026-04-15 → 2026-04-19; surfaced by ADR-0115's CI consolidation
  routing tox through master-targeting PRs.
- **`--frame_skip_ref` / `--frame_skip_dist` hang**: the skip loops in
  `libvmaf/tools/vmaf.c` fetched pictures from the preallocated picture
  pool (now always-on per ADR-0104) but never `vmaf_picture_unref`'d
  them, exhausting the pool after N skips and blocking the next fetch
  indefinitely. Each skipped picture is now unref'd immediately after
  fetch. Surfaced by `test_run_vmafexec_with_frame_skipping{,_unequal}`
  hanging locally (timeout 60 s, no output written) once tox started
  exercising both flags on master-targeting PRs.
- **CI tox doctest collection**: `pytest --doctest-modules` errored on five
  upstream files under `python/vmaf/resource/` (parameter / dataset / example
  config files; `vmaf_v7.2_bootstrap.py` and friends — dots in the stem make
  them unimportable as Python modules). Tox commands now pass
  `--ignore=vmaf/resource` so doctest collection skips that subtree. The
  files carry no doctests to begin with, so this is correctness, not a
  workaround. Surfaced by ADR-0115's CI trigger consolidation, which finally
  ran tox on PRs to master.

- **SYCL build with non-icpx host CXX**: `libvmaf/src/meson.build`
  unconditionally added `-fsycl` to the libvmaf shared-library link args
  whenever SYCL was enabled, even when the project's C++ compiler was
  gcc / clang / msvc. The host link driver does not understand `-fsycl`
  and failed with `g++: error: unrecognized command-line option '-fsycl'`
  at the `libvmaf.so` link step. The arg is now gated on
  `meson.get_compiler('cpp').get_id() == 'intel-llvm'`. The runtime
  libraries (libsycl + libsvml + libirc + libze_loader) declared as link
  dependencies already cover the gcc/clang link path, matching the
  documented "host C++ + sidecar icpx" project mode. Surfaced by
  ADR-0115's CI consolidation, which added an Ubuntu SYCL job that
  exercises this configuration on PRs to master.

- **FFmpeg patch series application**: `Dockerfile` and
  `.github/workflows/ffmpeg.yml` now walk `ffmpeg-patches/series.txt`
  and apply each patch in order via `git apply` with a `patch -p1`
  fallback. The Dockerfile previously `COPY`'d only patch 0003 (which
  fails to apply standalone because it references `LIBVMAFContext`
  fields added by patch 0001), and `ffmpeg.yml` referenced a stale
  `../patches/ffmpeg-libvmaf-sycl.patch` that no longer existed.
  Patches `0001-libvmaf-add-tiny-model-option.patch`,
  `0002-add-vmaf_pre-filter.patch`, and
  `0003-libvmaf-wire-sycl-backend-selector.patch` were also
  regenerated via real `git format-patch -3` so they carry valid
  `index <sha>..<sha> <mode>` header lines (the originals were
  hand-stubbed with placeholder SHAs and `git apply` choked on them).
  Docker images and CI FFmpeg-SYCL builds now exercise the full
  fork-added FFmpeg surface (tiny-AI + `vmaf_pre` + SYCL selector),
  not just SYCL. Also drops the bogus `--enable-libvmaf-sycl`
  configure flag (patch 0003 wires SYCL via `check_pkg_config`
  auto-detection — there is no such configure switch) and splits
  the Dockerfile's nvcc flags into a libvmaf set
  (`NVCC_FLAGS`, retains the four `-gencode` lines plus
  `--extended-lambda` and the `--expt-*` flags for Thrust/CUB) and
  an FFmpeg set (`FFMPEG_NVCC_FLAGS`, single-arch
  `compute_75,sm_75` matching FFmpeg's own modern-nvcc default —
  PTX is forward-compatible via driver JIT) so FFmpeg's
  `check_nvcc -ptx` probe stops failing with `nvcc fatal: Option
  '--ptx (-ptx)' is not allowed when compiling for multiple GPU
  architectures`. Also drops `--enable-libnpp` from FFmpeg
  configure — FFmpeg n8.1 explicitly `die`s if libnpp >= 13.0
  (configure:7335-7336 `"libnpp support is deprecated, version
  13.0 and up are not supported"`), and we don't actually use
  scale_npp / transpose_npp filters in VMAF workflows; cuvid +
  nvdec + nvenc + libvmaf-cuda are what we exercise. Patch 0002
  also gained a missing `#include "libavutil/imgutils.h"` for
  `av_image_copy_plane` (caught by the local docker build —
  upstream FFmpeg builds with `-Werror=implicit-function-declaration`).
  See ADR-0118 and entry 0018.

- **CI workflow naming**: renamed all six core `.github/workflows/*.yml`
  files to purpose-descriptive kebab-case (e.g. `ci.yml` →
  `tests-and-quality-gates.yml`, `libvmaf.yml` →
  `libvmaf-build-matrix.yml`) and normalised every workflow `name:` and
  job `name:` to Title Case. Required-status-check contexts in
  `master` branch protection re-pinned in the same merge window. See
  [ADR-0116](docs/adr/0116-ci-workflow-naming-convention.md) +
  [`docs/rebase-notes.md` entry 0020](docs/rebase-notes.md).

### Re-attributed

- 11 SYCL files in `libvmaf/{include,src,test}/.../sycl/` from
  `Netflix, Inc.` to `Lusoris and Claude (Anthropic)` — these files were
  authored entirely by the fork.
### Changed

- **CHANGELOG + ADR-index fragment files (T7-39 / ADR-0221)** — every PR
  in flight before this change fought merge conflicts in
  [`CHANGELOG.md`](CHANGELOG.md) and
  [`docs/adr/README.md`](docs/adr/README.md) (each PR adds a row, every
  other PR's row collides). PRs now drop a single fragment file under
  `changelog.d/<section>/<topic>.md` (Keep-a-Changelog sections: `added`,
  `changed`, `deprecated`, `removed`, `fixed`, `security`) and one row
  fragment under `docs/adr/_index_fragments/NNNN-slug.md`. Two new in-tree
  shell scripts —
  [`scripts/release/concat-changelog-fragments.sh`](scripts/release/concat-changelog-fragments.sh)
  and [`scripts/docs/concat-adr-index.sh`](scripts/docs/concat-adr-index.sh)
  — render `CHANGELOG.md`'s Unreleased block and `docs/adr/README.md`
  from the fragment trees; both ship `--check` (CI) and `--write`
  (release-please / local) modes. Migration is content-preserving: the
  existing 3119-line Unreleased body is archived verbatim under
  `changelog.d/_pre_fragment_legacy.md`, and 159 ADR rows are split into
  per-slug fragment files driven by a frozen `_order.txt` manifest that
  preserves the existing commit-merge order. New PRs append one fragment
  file (and one line to `_order.txt`) instead of editing the consolidated
  files. Doc-Substance Gate (ADR-0167) recognises a new
  `changelog.d/<section>/<row>.md` as a CHANGELOG entry. See
  [ADR-0221](docs/adr/0221-changelog-adr-fragment-pattern.md) +
  [`docs/research/0034-changelog-fragment-pattern.md`](docs/research/0034-changelog-fragment-pattern.md).

## (2022-04-11) [v2.3.1]

This is a minor release with some CAMBI extensions and speed-ups and adding it
to AOM CTC v3, as well as a few minor fixes/cleanups.

- CAMBI extensions: full reference, PQ eotf, up to 16 bit-depth support,
  max_log_contrast parameter.
- CAMBI: option to output heatmaps.

## (2021-10-16) [v2.3.0]

New release to add CAMBI (Contrast Aware Multiscale Banding Index).

- Python library: add encode width and height to Asset.
- libvmaf: add pixel format VMAF_PIX_FMT_YUV400P.
- Add cambi; add tests.
- Improve documentation. (#912)

## (2021-09-20) [v2.2.1]

This is another minor release to address a few last minute items for the AOM CTC
v2, as well as a few minor fixes/cleanups.

- Fix a race condition in vmaf_thread_pool_wait(). (#894)
- Avoid chroma resampling for 420mpeg2 y4m input (#906)

## (2021-07-02) [v2.2.0]

This is a minor release to address a few items for the AOM CTC v2, as well as a
few minor fixes/cleanups.

- Fixes a CIEDE-2000 precision issue, where cross-platform mismatches were seen.
  (#878)
- Adds libvmaf API function vmaf_feature_dictionary_free(). (#879)

## (2021-01-13) [v2.1.1]

This is a minor release to address a few last minute items for the initial AOM CTC.

**New features:**

- Fixes a SSIM/MS-SSIM precision bug where a lossless comparison did not always
  result in a perfect 1.0 score. (#796).
- Adds feature extractor options to clip the dB scores for both PSNR/SSIM.
  --aom_ctc v1.0 has been updated to use these clipping options according to the
  AOM CTC. (#802).

## (2020-12-30) [v2.1.0]

This is a minor release for the initial AOM CTC. Support has been added for
templated feature names. While this is a general purpose software feature,
templated feature names are immediately useful for simultaneous computation of
VMAF and VMAF NEG since the two metrics rely on slightly different VIF/ADM
variations. Global feature overrides via the `--feature` flag are no longer
supported, instead individual models can have their features overloaded
individually, the syntax for which is as follows:

 ```sh
--model version=vmaf_v0.6.1:vif.vif_enhn_gain_limit=1.0:adm.adm_enhn_gain_limit=1.0
```

**New features:**

- Per-model feature overloading via new API `vmaf_model_feature_overload()`.
- Multiple unique configurations of the same feature extractor may be registered
  run at the same time.
- `--aom_ctc v1.0` preset, encompassing all metrics specified by the AOM CTC.

## (2020-12-4) [2.0.0]

**New features:**

- Add PSNR-HVS and CIEDE2000 metrics.
- ci/actions: upload linux/macos artifacts (#738)
- libvmaf/feature: deprecate daala_ssim (#735)
- libvmaf: remove support for pkl models
- libvmaf/psnr: rewrite using integer types, 2x speedup
- vmaf: if no model is specified, enable v0.6.1 by default (#730)
- libvmaf/x86: add AVX2/AVX-512 optimizations for adm, vif and motion
- ci/actions: add xxd to build dependencies for Windows
- libvmaf: add support for built-in models
- libvmaf/integer_vif: use symmetrical mirroring on edges
- Fix log2 by replacing log2f_approx with log2f
- libvmaf_rc: provide a backwards compatible compute_vmaf(), link vmafossexec with
  libvmaf
- libvmaf: add framework support for json models
- libvmaf/libsvm: update libsvm to version 324
- libvmaf/motion: add motion_force_zero to motion fex
- return sha1 if Asset string is longer than 255
- Add CID/iCID Matlab source code
- build: unbreak x86 builds (Fixes: #374)
- Add 12bit and 16bit support for python YUV reader; add tests.
- Add PypsnrFeatureExtractor
- Add processes to FeatureAssembler. (#662)

**Fixed bugs:**

- fix motion flush for single frame input
- Fixing the perf_metric for a single entry list input

## (2020-8-24) [1.5.3]

(Updates since 1.5.1)

**Fixed bugs:**

- Fix inverted height and width in integer_motion in vmaf_rc (#650).

**New features:**

- libvmaf: add support for CSV and JSON logging
- Python: Add an (optional) step in Executor class to do python-based processing
  to ref/dis files (#523).
- Restructure python project and documentation (#544).
- Move test resource to Netflix/vmaf_resource repo (#552).
- Add Github CI (#558).
- Add vmaf_float_v0.6.1neg model; add vif_enhn_gain_limit and adm_enhn_gain_limit
  options to vmaf_rc.
- Update documentation for FFmpeg+libvmaf.
- Improvements to AucPerfMetric (#643).
- Add motion_force_zero option to vmaf_rc.

## (2020-6-30) [1.5.2]

**Fixed bugs:**

- Fix pkgconfig version sync issue (#572)

**New features:**

- libvmaf_rc general improvements

## (2020-2-27) [1.5.1]

**New features:**

- `libvmaf` has been relocated, and now has its own self-enclosed source tree
  (`./libvmaf/`) and build system (`meson`).
- Update license to BSD+Patent.
- Migrate the build system from makefile to meson.
- Introduce a new release candidate API with the associated library `libvmaf_rc`
  and executable `vmaf_rc` under `./libvmaf/build`.
- Add SI and TI feature extractor python classes.
- Add fixed-point SSIM implementation.
- Migrate to python3.

## (2019-9-8) [1.3.15]

**Fixed bugs:**

- Fix a case when CPU cores > 128(MAX_NUM_THREADS) / 3 (#319).
- Avoid dis-filtering ref when not needed, fix return type (#325).
- Update name of file for failed dis_path fopen (#334).
- A few compilation fixes (warnings and errors) (#326).
- Bump up g++ version to 9 for travis (#352).
- Use stat struct instead of ftell to retrieve the file size (#350).

**New features:**

- Write aggregate scores, exec FPS to json output.
- Add support for python3 (#332).
- Print progress in vmafossexec (#337).
- Add VMAF logo.
- Add link to report VMAF bad cases.

## (2019-3-1) [1.3.14]

**Fixed bugs:**

- Fix VMAF value mismatch on 160x90 videos after optimization (#315).
- Fix w10 error with using uninitialized offset_flag variable (#302).

**New features:**

- Add automated Windows builds with AddVeyor (#313).
- Report aggregate CI scores and fix empty model name in log (#304).

## (2019-1-31) [1.3.13]

**New features:**

- Optimized C code for speed. Running in multithreading mode, `vmafossexec`
  achieves ~40% run time reduction compared to the previous version.
- Printed out individual vmaf bootstrap scores in text file from `vmafossexec`.
- refactored windows solution (#283) (#284) (#285) (#291) (#298).

## (2018-12-17) [1.3.11]

**New features:**

- Revise number of bootstrap models definition:
  model/vmaf_rb_v0.6.3/vmaf_rb_v0.6.3.pkl has 21 models (20 bootstrap models and
  one using the full data). From these 21 models, the 20 of them are same as
  v0.6.2, only added an additional bootstrap model.
- Output the per bootstrap model predictions from wrapper/vmafossexec.
- Print bootstrap individual scores in xml and json.
- Add BD-rate calculator and update documentation.
- Report aggregate PSNR, SSIM, and MS-SSIM scores.
- Add sklearn linear regression class to TrainTestModel.
- Enable BRISQUE feature in VMAF training with bootstrapping.
- Add --save-plot option to command line tools.
- Add ST-RREDOpt (time optimized), ST-MAD feature extractors, quality runners and
  unittestts. Refactor ST-RRED feature extractor. (#216)

**Fixed bugs:**

- Bug fixed. When start vmaf in multi-thread at the same time. (#239)
- Fix name of min function in vmaf.h and vmaf.cpp. (#227)
- Fix implicit declaration of functions (#225)

## (2018-9-13) [1.3.10]

**New features:**

- Remove sureal as a submodule to vmaf. sureal is now available through pip install.

## (2018-8-7) [1.3.9]

**Fixed bugs:**

- libvmaf: fix case where user defined read_frame() callback was being ignored.

## (2018-6-21) [1.3.8]

**Fixed bugs:**

- Fix compute_vmaf boolean type issue (#178).

## (2018-6-12) [1.3.7]

**New features:**

- Add the --ci option to calculate confidence intervals to predicted VMAF scores
  (run_vmaf, run_vmaf_in_batch, ffmpeg2vmaf, vmafossexec).
- Update libvmaf version to 1.3.7 after compute_vmaf() interface change (added
  enable_conf_interval option).
- Add new models: 1) model/vmaf_4k_v0.6.1.pkl for 4KTV viewing at distance 1.5H,
  2) model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl for VMAF prediction with a confidence
  interval, 3) model/vmaf_4k_rb_v0.6.2/vmaf_4k_rb_v0.6.2.pkl for 4KTV viewing at
  distance 1.5H, with a confidence interval.

## (2018-6-4) [1.3.6]

**New features:**

- Update libvmaf version to 1.3.6 (to make consistent with VDK version from now
  on) after compute_vmaf() interface change (added thread and subsample options).
- Add the option to set the number of threads to use in vmafossexec.
- Add the option to subsample frames to save computation in vmafossexec.

## (2018-5-23) [1.3.5]

**New features:**

- Add multi-threading to vmafossexec.

## (2018-5-8) [1.3.4]

**Refactoring:**

- Refactor mos out of vmaf repo; rename to sureal as submodule.
- Refactor TrainTestModel to make predict() to output dictionary.
- Refactor TrainTestModel.
- Rename KFLK metric to AUC (Area Under the Curve) for better interpretability.

**New features:**

- Add bootstrapping to VMAF. Add two new classes BootstrapVmafQualityRunner and
  BaggingVmafQualityRunner
- Add Resolving Power Performance Metric.
- Add BRISQUE and NIQE feature extractors. Added two new classes
  BrisqueNorefFeatureExtractor and NiqeNorefFeatureExtractor. Add
  NiqeQualityRunner.

**Fixed bugs:**

- Add .gitattributes (#127). Force .pkl and .model files to retain LF line-ending.
  Required for use on Windows where model files would otherwise be checked out as
  CRLF which VMAF's parser doesn't handle.
- Allow MinGW compilation of ptools (#133). ptools doesn't build on MinGW as *nix
  socket headers are included. This patch selects Windows headers for MinGW
  builds.
- Update compute vmaf interface (#138). Update VMAF version in libvmaf.pc and etc.
  Catch logic error (resulted from wrong model file format) in compute_vmaf(). Use
  custom error code.

## (2017-12-3) [1.3.3]

**Fixed bugs:**

- Update VMAF version to 0.6.2 after compute_vmaf() interface change (#124).

## (2017-12-3) [1.3.2]

**Refactoring:**

- Lift check for exec existence during program load.
- Refactor psnr, ssim, ms_ssim and vmaf_feature to call ExternalProgramCaller.
- Refactor feature/Makefile to make executables depend on libvmaf.a.
- Refactor wrapper/Makefile to include additional objs in libvmaf.a but exclude
  main.o.
- Remove ar -d command after removing main.o from libvmaf.a.

**New features:**

- Generalize read_dataset.
- Update default Asset resampling method to bicubic (#116).
- Extend ffmpeg2vmaf script to allow ref/dis input to be YUV (#118).
- Improve README.md (#121).

**Fixed bugs:**

- Temporary fix Visual Studio builds (#112).
- Avoid unnecessary dependency on matplotlib in run_vmaf (#114).
- Remove unneeded dependencies in Dockerfile, fixes #115 (#117).
- MinGW support (#123).
- Change compute_vmaf() interface to return an error code instead of throw an
  error #124 (#126).

## (2017-8-12) [1.3.1]

**Refactoring:**

- Refactor NorefExecutorMixin to eliminate repeated codes.
- Refactor C code: get rid of unused double functions; uniformly use read_frame
  callback function to void repeated code;
- Add strip option to Makefile.

**New features:**

- Update Asset class: add copy functions to Asset; add ref/dis_yuv_type; deprecate
  yuv_type; add ref/dis_start_sec;
- Update subjective models: add confidence interval to subjective model
  parameters; refactor MLE model and make subclasses; add run_subj command line.
- Recommend pip, add ffmpeg2vmaf info and reorganize prerequisite installation (#88).
- Reduce sleep time in parallel_map.
- Add library interface for VMAF (#90).
- Add VisualStudio2015 support (#92).
- Add example of image dataset notyuv.
- Add pkgconfig file and changed Makefile.
- Add VmafPhoneQualityRunner class.
- Add DMOS_MLE_CO subjective model.

**Fixed bugs:**

- Update RegressionMixin to handle AUC exception for dicitonary-style dataset.
- Fix Makefile fedora libptools issue. (#98)

## (2017-4-13) [1.2.4]

**Refactoring:**

- Deprecate run_executors_in_parallel.
- Refactor NorefFeatureExtractor into NorefExecutorMixin so that it can be used
  for all executors.
- Add abstract methods to some base classes.

**New features:**

- Add ST-RRED runner (StrredQualityRunner), based on "Video Quality Assessment by
  Reduced Reference Spatio-Temporal Entropic Differencing", by R. Soundararaajan,
  A. Bovik.
- Add start/end frame support for Executor.

## (2017-3-8) [1.2.3]

**New features:**

- Refactor to replace config.ROOT with config.VmafConfig.

## (2017-3-1) [1.2.2]

**New features:**

- Generalize Result and FileSystemResultStore to allow None values.

## (2017-2-27) [1.2.1]

**Tasks:**

- Refactor to prepare for pypi packaging.

## (2017-2-20) [1.2.0]

**New features:**

- Updated VMAF model to version v0.6.1. Changes include: 1) added a custom model
  for cellular phone screen viewing; 2) trained using new dataset, covering more
  difficult content; 3) elementary metric fixes: ADM behavior at near-black
  frames, motion behavior at scene boundaries; 4) compressed quality score range
  by 20% to accommodate higher dynamic range; 5) Use MLE instead of DMOS as
  subjective model.

## (2017-1-24) [1.1.23]

**Fixed bugs:**

- Replace subprocess.call with run_process (checking return value).

## (2017-1-22) [1.1.22]

**New features:**

- Add command line ffmpeg2vmaf, which takes encoded videos as input.

## (2017-1-18) [1.1.21]

**New features:**

- Allow processing non-YUV input videos.

## (2016-12-20) [1.1.20]

**New features:**

- Add STRRED runner.

## (2016-12-19) [1.1.19]

**New features:**

- Allow specifying crop and pad parameter in dataset files.

## (2016-12-8) [1.1.18]

**Fixed bugs:**

- Replace pathos with custom function for parallel executor running.

## (2016-12-8) [1.1.17]

**Fixed bugs:**

- Fix command line run_testing issue. Add command line test cases.

## (2016-12-5) [1.1.16]

**New features:**

- Speed up VMAF convolution operation by AVX.

## (2016-11-30) [1.1.15]

**Fixed bugs:**

- Fix vmafossexec memory leakage.

## (2016-11-28) [1.1.14]

**New features:**

- Add enable_transform_score option to VmafQualityRunner, VmafossExecQualityRunner.

## (2016-11-18) [1.1.13]

**Fixed bugs:**

- Fix a bug in DatasetReader.to_aggregated_dataset_file.

## (2016-11-15) [1.1.12]

**New features:**

- Add Travis continuous integration.

## (2016-11-11) [1.1.11]

**New features:**

- Add implementation of AUC (Area Under the Curve) - quality metric evaluation
  method based on AUC. Refer to: L. Krasula, K. Fliegel, P. Le Callet, M.Klima,
  "On the accuracy of objective image and video quality models: New methodology
  for performance evaluation", QoMEX 2016.

## (2016-11-07) [1.1.10]

**New features:**

- Add options to use custom subjective models in run_vmaf_training and run_testing
  commands.

## (2016-11-02) [1.1.9]

**New features:**

- Add DatasetReader and subclasses; add SubjectiveModel and subclasses.

## (2016-10-19) [1.1.8]

**New features:**

- Add quality runners for each individual VMAF elementary metrics.

## (2016-10-14) [1.1.7]

**Fixed bugs:**

- Issue #36: SSIM and MS-SSIM sometimes get negative values.

## (2016-10-10) [1.1.6]

**New features:**

- Add Xcode project support.
- Add more pooling options (median, percx) to CLIs.

## (2016-10-8) [1.1.5]

**New features:**

- Add support for docker usage (#30).

## (2016-10-7) [1.1.4]

**Fixed bugs:**

- Issue #29: Make ptools build under Fedora.

## (2016-10-6) [1.1.3]

**New features:**

- Generalize dataset format to allow per-content YUV format.

## (2016-10-5) [1.1.2]

**Fixed bugs:**

- Make ptools work under Mac OS.
- Update SklearnRandomForestTrainTestModel test with sklearn 0.18.

## (2016-09-29) [1.1.1]

**New features:**

- Update command lines run_vmaf, run_psnr, run_vmaf_in_batch, run_cleaning_cache,
  run_vmaf_training and run_testing.

## (2016-09-28) [1.1.0]

**New features:**

- Update wrapper/vmafossexec: 1) it now takes pkl model file as input, so that
  slopes/intercepts are no longer hard-coded; 2) it now takes multiple YUV input
  formats; 3) add flag to enable/disable VMAF score clipping at 0/100; 4) allow
  customly running PSNR/SSIM/MS-SSIM; 5) allow customly outputing XML/JSON
- Add SSIM/MS-SSIM option in run_testing.

## (2016-09-09) [1.0.9]

**Fixed bugs:**

- Move VmafQualityRunnerWithLocalExplainer to quality_runner_adhoc to resolve
  multiple instances of VMAF found when calling QualityRunner.find_subclass.

**New features:**

- Add custom_clip_0to1 to TrainTestModel.

## (2016-09-07) [1.0.8]

**New features:**

- Generalize read_dataset to allow specifying width, height and resampling method
  on which to calculate quality.
- Add bicubic to SUPPORTED_RESAMPLING_TYPES for Asset.
- Update Asset rule with resampling_type in **str** to avoid duplicates in data
  store.

## (2016-08-20) [1.0.7]

**New features:**

- Update VmafFeatureExtractor to 0.2.2b with scaled ADM features exposed (adm_scale0-3).

## (2016-08-20) [1.0.6]

**New features:**

- Add DisYUVRawVideoExtractor and related classes.
- Add NeuralNetworkTrainTestModel base class that integrates TensorFlow.
- Add example class ToddNoiseClassifierTrainTestModel.

## (2016-08-20) [1.0.5]

**New features:**

- Add LocalExplainer class.
- Add show_local_explanation option to run_vmaf script.

## (2016-07-21) [1.0.4]

**Fixed bugs:**

- Fix a series of numerical issues in VMAF features, increment
  VmafFeatureExtractor version number.
- Retrain VmafQualityRunner after feature update, increment version number.

## (2016-07-20) [1.0.3]

**New features:**

- Add base class NorefFeatureExtractor for any feature extractor that do not
  use a reference video.
- Add MomentNorefFeatureExtractor subclassing NorefFeatureExtractor as an example
  implementation.

## (2016-06-16) [1.0.2]

**New features:**

- Refactor feature code to expose ssim/ms-ssim, speed up ssim/ms-ssim.

## (2016-06-10) [1.0.1]

**Fixed bugs:**

- Fix feature while looping by moving feof to after read_image.
- Fix issue #2 use hashed string for log filename and result filename to avoid
  file names getting too long.

**New features:**

- Add SsimFeatureExtractor and MsSsimFeatureExtractor with intermediate features
  (luminence, contrast, structure).
