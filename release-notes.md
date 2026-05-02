:robot: I have created a release *beep* *boop*
---


<details><summary>4.0.0-lusoris.0</summary>

## [4.0.0-lusoris.0](https://github.com/lusoris/vmaf/compare/v3.0.0-lusoris.0...v4.0.0-lusoris.0) (2026-05-02)


###   BREAKING CHANGES

* **cli:** %.6f default + unref skipped frames so Netflix golden gate passes ([#55](https://github.com/lusoris/vmaf/issues/55))
* **cli:** default CLI score precision changed from %.6f to %.17g. Parsers that string-equal-compare VMAF outputs will see additional digits. Pass --precision=legacy to restore old behavior.

### Features

* **ai:** BVI-DVC feature-extraction pipeline (corpus-3 for tiny-AI v2) ([#214](https://github.com/lusoris/vmaf/issues/214)) ([362d9e7](https://github.com/lusoris/vmaf/commit/362d9e7f256288fd469215aec45fdd91966c1eb2))
* **ai:** Combined Netflix + KoNViD-1k tiny-AI trainer driver ([#180](https://github.com/lusoris/vmaf/issues/180)) ([a143e25](https://github.com/lusoris/vmaf/commit/a143e255c5d8130872be25bdc1fdb757fcc878cf))
* **ai:** Full vmaf-train package  fr/nr/filter families + CLI + registry ([91b5558](https://github.com/lusoris/vmaf/commit/91b5558e133bca0b47d1199be8ecff85bb3ac288))
* **ai:** KoNViD-1k ’ VMAF-pair acquisition + loader bridge ([#178](https://github.com/lusoris/vmaf/issues/178)) ([219f653](https://github.com/lusoris/vmaf/commit/219f65315d6470b20bc60e7af1905c683524404d))
* **ai:** Model validators + MCP eval + variance head + INT8 PTQ (4/5) ([#8](https://github.com/lusoris/vmaf/issues/8)) ([81f63a4](https://github.com/lusoris/vmaf/commit/81f63a4e4695e11d33276fe50b44644fb64e1fa7))
* **ai:** Research-0028 + Phase-3 subset-sweep driver (negative result) ([#188](https://github.com/lusoris/vmaf/issues/188)) ([7e4e884](https://github.com/lusoris/vmaf/commit/7e4e88423702b962512f630e642c937097fe7dfe))
* **ai:** Research-0029 Phase-3b  StandardScaler validates broader-feature hypothesis ([#192](https://github.com/lusoris/vmaf/issues/192)) ([a1453bf](https://github.com/lusoris/vmaf/commit/a1453bff4caad21e873a303a9666eb4fc586b103))
* **ai:** Research-0030 Phase-3b multi-seed validation (Gate 1 passed) ([#190](https://github.com/lusoris/vmaf/issues/190)) ([1f128b0](https://github.com/lusoris/vmaf/commit/1f128b01f75678c95f63dd8f848aaf2b4fc2e7b2))
* **ai:** Retrain vmaf_tiny_v2 on 4-corpus (NF+KV+BVI-A+B+C+D) ([#255](https://github.com/lusoris/vmaf/issues/255)) ([53837dc](https://github.com/lusoris/vmaf/commit/53837dcbba23c1a72653a238dfe6cd0a56b10411))
* **ai:** Ship vmaf_tiny_v2 (canonical-6 + StandardScaler + lr=1e-3 + 90ep, validated +0.018 PLCC over v1) ([#250](https://github.com/lusoris/vmaf/issues/250)) ([3999cda](https://github.com/lusoris/vmaf/commit/3999cdab6163cb4512bb5fb7b682b9e78c5e99cd))
* **ai:** T5-3d  nr_metric_v1 dynamic-batch fix + PTQ pipeline ([#247](https://github.com/lusoris/vmaf/issues/247)) ([9ff5d1d](https://github.com/lusoris/vmaf/commit/9ff5d1d48428d4f1ef01236a8927b2da26c6b3f5))
* **ai:** T5-3e empirical PTQ accuracy across CPU/CUDA/OpenVINO EPs ([#174](https://github.com/lusoris/vmaf/issues/174)) ([4130149](https://github.com/lusoris/vmaf/commit/413014963259ce54d3f06ac2f0d1659f6a2d2477))
* **ai:** T6-1a  fr_regressor_v1 C1 baseline (Netflix Public, unblocked) ([#249](https://github.com/lusoris/vmaf/issues/249)) ([f809ce0](https://github.com/lusoris/vmaf/commit/f809ce09c214156259790c91109cea02ce2640f6))
* **ai:** T6-2a  MobileSal saliency feature extractor ([#208](https://github.com/lusoris/vmaf/issues/208)) ([fa7d4f5](https://github.com/lusoris/vmaf/commit/fa7d4f52833d087a8d47236e5bdc85be5dd37205))
* **ai:** T6-3a  TransNet V2 shot-boundary feature extractor ([#210](https://github.com/lusoris/vmaf/issues/210)) ([08b7644](https://github.com/lusoris/vmaf/commit/08b7644f8a31c70623f52e5a1dd5c47ca5f08493))
* **ai:** T6-7  FastDVDnet temporal pre-filter (5-frame window) ([#203](https://github.com/lusoris/vmaf/issues/203)) ([cf1d670](https://github.com/lusoris/vmaf/commit/cf1d670d74ea98ed66baf2e3e57acf686189a456))
* **ai:** T6-9  model registry schema + Sigstore --tiny-model-verify ([#199](https://github.com/lusoris/vmaf/issues/199)) ([9293d69](https://github.com/lusoris/vmaf/commit/9293d6983d5b87525163c873f54a1b0b84845c74))
* **ai:** T7-CODEC-AWARE  codec-conditioned FR regressor surface (training BLOCKED) ([#237](https://github.com/lusoris/vmaf/issues/237)) ([876382d](https://github.com/lusoris/vmaf/commit/876382dc805bc53fe7f98d647ad5d0b98c72da60))
* **ai:** T7-DISTS  scaffold DISTS extractor proposal (ADR-0236) ([#259](https://github.com/lusoris/vmaf/issues/259)) ([da6d7b0](https://github.com/lusoris/vmaf/commit/da6d7b0e2fa6d0a692e2d9afd94c8af4743bda5e))
* **ai:** T7-GPU-ULP-CAL  scaffold GPU-gen ULP calibration head (proposal) ([#238](https://github.com/lusoris/vmaf/issues/238)) ([fdb8a56](https://github.com/lusoris/vmaf/commit/fdb8a5652d2f42bcf7d243af19804f1d4eb81b12))
* **ai:** Tiny-AI 3-arch LOSO evaluation harness + Research-0023 ([#176](https://github.com/lusoris/vmaf/issues/176)) ([0e483a3](https://github.com/lusoris/vmaf/commit/0e483a3e16461727762c01b9f2a7b1bdf2db1d72))
* **ai:** Tiny-AI feature-set registry (Research-0026 Phase 1) ([#185](https://github.com/lusoris/vmaf/issues/185)) ([4336736](https://github.com/lusoris/vmaf/commit/4336736ae2e52ab6a12728f9708777886fe6a5dd))
* **ai:** Tiny-AI LOSO evaluation harness for mlp_small ([#165](https://github.com/lusoris/vmaf/issues/165)) ([427272b](https://github.com/lusoris/vmaf/commit/427272b48b18e0feb3d239112c24318e0187f34c))
* **ai:** Tiny-AI Phase-2 analysis scaffolding (Research-0026) ([#191](https://github.com/lusoris/vmaf/issues/191)) ([0df6543](https://github.com/lusoris/vmaf/commit/0df6543a8803017a67ae5c0454d6ea0c53093152))
* **ai:** Tiny-AI Quantization-Aware Training (QAT) implementation (T5-4) ([#179](https://github.com/lusoris/vmaf/issues/179)) ([d71ac52](https://github.com/lusoris/vmaf/commit/d71ac52cf5b995e21262a89f87d468fc7c28eade))
* **ai:** Tiny-AI training prep (loader + eval + Lightning harness for Netflix corpus) ([#158](https://github.com/lusoris/vmaf/issues/158)) ([aa74eaa](https://github.com/lusoris/vmaf/commit/aa74eaa3b5ac74263caceb053a218ba113a32345))
* **arch:** T7-26  feature-characteristics registry + per-backend dispatch_strategy ([#124](https://github.com/lusoris/vmaf/issues/124)) ([817b4dd](https://github.com/lusoris/vmaf/commit/817b4dd9279d35d75fad49ac5003c5206fb615b5))
* **batch-a:** Port four OPEN Netflix upstream PRs (ADR-0131/0132/0134/0135) ([#72](https://github.com/lusoris/vmaf/issues/72)) ([df622ea](https://github.com/lusoris/vmaf/commit/df622ea782987e3ab37d0f71aaf00903bf964c72))
* **ci:** Add FFmpeg + Vulkan integration lane (lavapipe) ([#235](https://github.com/lusoris/vmaf/issues/235)) ([5d6c1af](https://github.com/lusoris/vmaf/commit/5d6c1afe0cad9c2e0f49c5c94f48d71d9fd4e189))
* **ci:** Add MCP smoke CI lane (T7-MCP-SMOKE-CI) ([#256](https://github.com/lusoris/vmaf/issues/256)) ([41f3cee](https://github.com/lusoris/vmaf/commit/41f3ceee7b7de08d20c273eeea9945b146cac489))
* **ci:** Nightly bisect-model-quality + sticky tracker (closes [#4](https://github.com/lusoris/vmaf/issues/4)) ([#41](https://github.com/lusoris/vmaf/issues/41)) ([6cd4fb0](https://github.com/lusoris/vmaf/commit/6cd4fb01a81cff13996ad94cf552f7b61104cf60))
* **ci:** T6-8  GPU-parity cross-device variance gate ([#202](https://github.com/lusoris/vmaf/issues/202)) ([e0b381b](https://github.com/lusoris/vmaf/commit/e0b381bfb8132c94f684d59d1dd7547131db3677))
* **cli:** --precision flag for IEEE-754 round-trip lossless scores ([c989fbd](https://github.com/lusoris/vmaf/commit/c989fbd913c7228b40e86af5176d265d3353c284))
* **cli:** Wire --tiny-model / --tiny-device / --no-reference flags ([6381819](https://github.com/lusoris/vmaf/commit/63818196a6d12a7cd37175867ae1910beb085946))
* **cuda,sycl:** GPU long-tail batch 1c parts 2 + 3  ciede_{cuda,sycl} ([#137](https://github.com/lusoris/vmaf/issues/137)) ([c2fb1de](https://github.com/lusoris/vmaf/commit/c2fb1de667126f9233a22510addbdb5b672d2f7e))
* **cuda,sycl:** GPU long-tail batch 1d parts 2 + 3  float_moment_{cuda,sycl} ([#135](https://github.com/lusoris/vmaf/issues/135)) ([0c9c117](https://github.com/lusoris/vmaf/commit/0c9c117be653014c9f8d4d39cd13598c30c9b521))
* **cuda,sycl:** GPU long-tail batch 2 parts 1b + 1c  float_ssim_{cuda,sycl} ([#140](https://github.com/lusoris/vmaf/issues/140)) ([c0f6979](https://github.com/lusoris/vmaf/commit/c0f6979493edefebd35473b824a7d86508301d3f))
* **cuda,sycl:** GPU long-tail batch 2 parts 2b + 2c  float_ms_ssim_{cuda,sycl} ([#142](https://github.com/lusoris/vmaf/issues/142)) ([8db2715](https://github.com/lusoris/vmaf/commit/8db2715ac20f19902ad63b09afbe0d5b24a52b8b))
* **cuda,sycl:** GPU long-tail batch 2 parts 3b + 3c  psnr_hvs_{cuda,sycl} ([#144](https://github.com/lusoris/vmaf/issues/144)) ([4fa2504](https://github.com/lusoris/vmaf/commit/4fa2504dbcd9bae7a84de913f29e5b1f1697d292))
* **cuda,sycl:** GPU long-tail batch 3 part 6b + 6c  float_adm_{cuda,sycl} (redo of [#157](https://github.com/lusoris/vmaf/issues/157)) ([#163](https://github.com/lusoris/vmaf/issues/163)) ([5af6e0b](https://github.com/lusoris/vmaf/commit/5af6e0b6f7c3a9ccf405f6316a7a433108f6472c))
* **cuda,sycl:** GPU long-tail batch 3 part 7b + 7c  ssimulacra2_{cuda,sycl} ([#162](https://github.com/lusoris/vmaf/issues/162)) ([f560bb2](https://github.com/lusoris/vmaf/commit/f560bb2ca197364770d6ed95e72e97c9b3080077))
* **cuda,sycl:** GPU long-tail batch 3 parts 1b + 1c  motion_v2_{cuda,sycl} ([#147](https://github.com/lusoris/vmaf/issues/147)) ([eea948f](https://github.com/lusoris/vmaf/commit/eea948f4c00b73f4ce2fd2d0e5aceff59eb0e880))
* **cuda:** GPU long-tail batch 1b part 1  psnr_cuda kernel + host ([#129](https://github.com/lusoris/vmaf/issues/129)) ([27cca81](https://github.com/lusoris/vmaf/commit/27cca81642c5cc6c01c257309d5338bc237134fa))
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
* **ffmpeg:** T7-28  add libvmaf_sycl filter (zero-copy QSV/VAAPI) ([#127](https://github.com/lusoris/vmaf/issues/127)) ([a2aa094](https://github.com/lusoris/vmaf/commit/a2aa09489006d87e631eed92f40b753302d3d0f2))
* **gpu:** GPU long-tail batch 1a  psnr_vulkan kernel + chars seeds ([#125](https://github.com/lusoris/vmaf/issues/125)) ([058c970](https://github.com/lusoris/vmaf/commit/058c970a4d1c3b89f4cac404559b231912f9e697))
* **gpu:** T3-15(c)  motion3 GPU coverage on Vulkan + CUDA + SYCL ([#248](https://github.com/lusoris/vmaf/issues/248)) ([f4ea952](https://github.com/lusoris/vmaf/commit/f4ea9523e437db9ca5b41abaf8c6fd4e866f37fc))
* **gpu:** T7-35  enable_lcs MS-SSIM on CUDA + Vulkan + SYCL ([#207](https://github.com/lusoris/vmaf/issues/207)) ([dbccc1b](https://github.com/lusoris/vmaf/commit/dbccc1b5290afa2f0fe69c98613c2a8925af53df))
* **hip:** T7-10  HIP (AMD) backend scaffold (audit-first) ([#200](https://github.com/lusoris/vmaf/issues/200)) ([2d1ccb1](https://github.com/lusoris/vmaf/commit/2d1ccb1a721dec6c13d78bacb71d7c3acdb7f471))
* **iqa,ssim:** Bit-exact AVX2 + AVX-512 convolve + SSIM accumulate ([#76](https://github.com/lusoris/vmaf/issues/76)) ([cf063b8](https://github.com/lusoris/vmaf/commit/cf063b8bfba6525a027c00fbea575833cf68e056))
* **libvmaf/feature:** Port upstream ADM updates (Netflix 966be8d5) ([#44](https://github.com/lusoris/vmaf/issues/44)) ([d06dd6c](https://github.com/lusoris/vmaf/commit/d06dd6cfc82b11f958fcf92c7d90219ca8ed2c2d))
* **libvmaf/feature:** Port upstream motion updates (Netflix PR [#1486](https://github.com/lusoris/vmaf/issues/1486)) ([#45](https://github.com/lusoris/vmaf/issues/45)) ([9371a0a](https://github.com/lusoris/vmaf/commit/9371a0aa176342f7e0e1de6e3da3585174c1c49f))
* **libvmaf:** Compile picture_pool unconditionally, drop VMAF_PICTURE_POOL gate ([#32](https://github.com/lusoris/vmaf/issues/32)) ([65460e3](https://github.com/lusoris/vmaf/commit/65460e3ad8d5c0b6399981d156e75e52b8f742bc))
* **libvmaf:** DNN runtime  public dnn.h + ORT backend + sidecar loader ([9b98594](https://github.com/lusoris/vmaf/commit/9b9859467aa3cf0cd99c49732b1b4ee9eeeea9b8))
* **mcp:** Add vmaf-mcp server exposing VMAF to LLM tooling ([0f233b8](https://github.com/lusoris/vmaf/commit/0f233b8075529b1a42fbe77becacf86b5075d1a3))
* **mcp:** Describe_worst_frames tool with VLM fallback (T6-6, ADR-0172) ([#108](https://github.com/lusoris/vmaf/issues/108)) ([5de8a79](https://github.com/lusoris/vmaf/commit/5de8a793eb2c13cfca10079fa3d18c0620cea130))
* **mcp:** T5-2  embedded MCP scaffold (audit-first) ([#195](https://github.com/lusoris/vmaf/issues/195)) ([8f46b22](https://github.com/lusoris/vmaf/commit/8f46b22cdad272d8f45853ef57e186ec4df1b0e2))
* **motion:** Motion_v2 NEON SIMD port (ADR-0145) ([#81](https://github.com/lusoris/vmaf/issues/81)) ([2dee153](https://github.com/lusoris/vmaf/commit/2dee15322806abacf7c9ebf064e028cde7da2e2b))
* **motion:** Port Netflix b949cebf  feature/motion: port several feature extractor options ([#197](https://github.com/lusoris/vmaf/issues/197)) ([3806beb](https://github.com/lusoris/vmaf/commit/3806beb99c6c288ba2ab700885542893bfeeba29))
* **ms-ssim:** NEON bit-identical decimate (ADR-0125) ([4531913](https://github.com/lusoris/vmaf/commit/45319133e2e8e3c4f11806e924e7cbf49e4f1900))
* **ms-ssim:** Separable decimate + AVX2/AVX-512 SIMD (ADR-0125) ([3283a61](https://github.com/lusoris/vmaf/commit/3283a61238cc6296c6b825d35b1d6f39bbcadc9f))
* **process:** T7 bundle  docs/state.md, MCP release channel, GPU runner guide, doc-drift enforcement ([#103](https://github.com/lusoris/vmaf/issues/103)) ([8e6b99a](https://github.com/lusoris/vmaf/commit/8e6b99acfc8eaf31c825c3358a9463720e9dfa3d))
* **psnr_hvs:** AVX2 bit-exact port  8×8 integer DCT vectorized (T3-5, ADR-0159) ([#96](https://github.com/lusoris/vmaf/issues/96)) ([ad57d33](https://github.com/lusoris/vmaf/commit/ad57d331b17e5a1f4d74b02a6f85c0fbf1ec2095))
* **psnr_hvs:** NEON aarch64 sister port  8×8 integer DCT vectorized (T3-5-neon, ADR-0160) ([#97](https://github.com/lusoris/vmaf/issues/97)) ([98d359f](https://github.com/lusoris/vmaf/commit/98d359f1c3c22a93c02cee4b8806fa5d1a933a6e))
* **simd:** Float_moment AVX2 + NEON parity (T7-19, ADR-0179) ([#122](https://github.com/lusoris/vmaf/issues/122)) ([095e68a](https://github.com/lusoris/vmaf/commit/095e68af611e4b339fbc2cc263d9a170a25dd74b))
* **simd:** SIMD DX framework + NEON bit-exactness port (ADR-0140) ([#77](https://github.com/lusoris/vmaf/issues/77)) ([0d23d8c](https://github.com/lusoris/vmaf/commit/0d23d8c1e99730c58cb51406e13694a4ab18874a))
* **simd:** T7-38  SSIMULACRA 2 PTLR + IIR-blur SVE2 ports ([#201](https://github.com/lusoris/vmaf/issues/201)) ([0471d75](https://github.com/lusoris/vmaf/commit/0471d75b3aa57706576f6806f9b7ac05a71a92db))
* **ssimulacra2:** IIR blur SIMD port  AVX2 + AVX-512 + NEON (T3-1 phase 2, ADR-0162) ([#99](https://github.com/lusoris/vmaf/issues/99)) ([34bef60](https://github.com/lusoris/vmaf/commit/34bef60a8db3b9d4c009f21414672f37132892c4))
* **ssimulacra2:** Picture_to_linear_rgb SIMD  AVX2 + AVX-512 + NEON (T3-1 phase 3, ADR-0163) ([#100](https://github.com/lusoris/vmaf/issues/100)) ([3b9d351](https://github.com/lusoris/vmaf/commit/3b9d351dcd25f40d86b66bbb886479f492187c79))
* **ssimulacra2:** Scalar port with libjxl FastGaussian IIR blur ([#68](https://github.com/lusoris/vmaf/issues/68)) ([410cf7b](https://github.com/lusoris/vmaf/commit/410cf7bc2180f3ebdc8793de0e92c696cac9d037))
* **sycl:** GPU long-tail batch 1b part 2  psnr_sycl kernel ([#130](https://github.com/lusoris/vmaf/issues/130)) ([d10343f](https://github.com/lusoris/vmaf/commit/d10343fcc312960fed3b7d25239759fa7f43dee6))
* **sycl:** Implement USM-backed picture pre-allocation pool ([#33](https://github.com/lusoris/vmaf/issues/33)) ([3f9a134](https://github.com/lusoris/vmaf/commit/3f9a1345b6d1480f07ac10b383e334f433ac8514))
* **sycl:** Implement vmaf_sycl_import_d3d11_surface (closes [#27](https://github.com/lusoris/vmaf/issues/27)) ([#35](https://github.com/lusoris/vmaf/issues/35)) ([8ea83f7](https://github.com/lusoris/vmaf/commit/8ea83f791412887c30873edb55c8d19b131488f2))
* **tiny-ai:** C2 nr_metric_v1 + C3 learned_filter_v1 baselines on KoNViD-1k (T6-1, ADR-0168) ([#104](https://github.com/lusoris/vmaf/issues/104)) ([1a48eab](https://github.com/lusoris/vmaf/commit/1a48eab976f39a926dec5f01d94bc3c363755522))
* **tiny-ai:** First per-model PTQ  learned_filter_v1 dynamic int8 (T5-3b, ADR-0174) ([#110](https://github.com/lusoris/vmaf/issues/110)) ([07a3053](https://github.com/lusoris/vmaf/commit/07a30535c97314851849237a235981549d5af4cf))
* **tiny-ai:** PTQ int8 audit harness  registry + scripts + sidecar parser (T5-3, ADR-0173) ([#109](https://github.com/lusoris/vmaf/issues/109)) ([a7f2664](https://github.com/lusoris/vmaf/commit/a7f2664cb4948a985c5ec52e7b1cc508596f5804))
* **tools:** T6-2b  vmaf-roi sidecar for per-CTU QP offsets (x265 / SVT-AV1) ([#246](https://github.com/lusoris/vmaf/issues/246)) ([a0a20e2](https://github.com/lusoris/vmaf/commit/a0a20e2fcb855b8ad4271078ba9b2b9e80950b15))
* **tools:** T6-3b  vmaf-perShot per-shot CRF predictor ([#244](https://github.com/lusoris/vmaf/issues/244)) ([f398668](https://github.com/lusoris/vmaf/commit/f398668f1f2437a7cc234aedd7b0d36de50c255b))
* **upstream:** Port 8a289703 + 1b6c3886  32-bit ADM/cpu fallbacks ([#212](https://github.com/lusoris/vmaf/issues/212)) ([968cb3f](https://github.com/lusoris/vmaf/commit/968cb3fce396471f2727cff8c0d3192769e53b4b))
* **upstream:** Port c70debb1  adm+vif test deltas ([#211](https://github.com/lusoris/vmaf/issues/211)) ([e66b510](https://github.com/lusoris/vmaf/commit/e66b5100195eab29e07108a7aa20a5f77a12344c))
* **upstream:** Port d3647c73  feature/speed extractors (speed_chroma + speed_temporal) ([#213](https://github.com/lusoris/vmaf/issues/213)) ([32f2757](https://github.com/lusoris/vmaf/commit/32f2757882a2e469d0263c8fb9cd8d2dd04a451a))
* **vulkan,cuda,sycl:** GPU long-tail batch 3 part 2  float_ansnr on all three backends ([#148](https://github.com/lusoris/vmaf/issues/148)) ([3017969](https://github.com/lusoris/vmaf/commit/30179695ae6255b63256eb7222bb4121cc9df1a9))
* **vulkan,cuda,sycl:** GPU long-tail batch 3 part 3  float_psnr on all three backends ([#149](https://github.com/lusoris/vmaf/issues/149)) ([c9f0b99](https://github.com/lusoris/vmaf/commit/c9f0b99dc5a5cce1469b5e1f511749a77407e459))
* **vulkan,cuda,sycl:** GPU long-tail batch 3 part 4  float_motion on all three backends ([#150](https://github.com/lusoris/vmaf/issues/150)) ([ac91174](https://github.com/lusoris/vmaf/commit/ac91174a71949a7c183749784227f564890f7eaf))
* **vulkan,cuda,sycl:** GPU long-tail batch 3 part 5  float_vif on all three backends ([#151](https://github.com/lusoris/vmaf/issues/151)) ([a34e08d](https://github.com/lusoris/vmaf/commit/a34e08d96212a50b367a5191e955e625329cfe49))
* **vulkan:** GPU long-tail batch 1c part 1  ciede_vulkan kernel ([#136](https://github.com/lusoris/vmaf/issues/136)) ([9286ace](https://github.com/lusoris/vmaf/commit/9286ace919ef7666727367f916eab999266a7824))
* **vulkan:** GPU long-tail batch 1d part 1  float_moment_vulkan kernel ([#133](https://github.com/lusoris/vmaf/issues/133)) ([a64cde6](https://github.com/lusoris/vmaf/commit/a64cde633aedb60f9a6f408bae4b0bc4f982502c))
* **vulkan:** GPU long-tail batch 2 part 1a  float_ssim_vulkan kernel ([#139](https://github.com/lusoris/vmaf/issues/139)) ([0d3767e](https://github.com/lusoris/vmaf/commit/0d3767e4eea4176fe95ec3c72c24d180cd124166))
* **vulkan:** GPU long-tail batch 2 part 2a  float_ms_ssim_vulkan kernel ([#141](https://github.com/lusoris/vmaf/issues/141)) ([ad185a1](https://github.com/lusoris/vmaf/commit/ad185a1129346330633a38a1e1450e97600f54cb))
* **vulkan:** GPU long-tail batch 2 part 3a  psnr_hvs_vulkan kernel ([#143](https://github.com/lusoris/vmaf/issues/143)) ([94fb39f](https://github.com/lusoris/vmaf/commit/94fb39f9e5dc7e451e0a65fcb5a9205a5c906bb2))
* **vulkan:** GPU long-tail batch 3 part 1a  motion_v2_vulkan kernel ([#146](https://github.com/lusoris/vmaf/issues/146)) ([41f3ae4](https://github.com/lusoris/vmaf/commit/41f3ae4c7307e7a0f41a44a9195c77d530282ad3))
* **vulkan:** GPU long-tail batch 3 part 6  float_adm_vulkan kernel ([#154](https://github.com/lusoris/vmaf/issues/154)) ([ef73440](https://github.com/lusoris/vmaf/commit/ef73440c4edab3b12ed36e47ea633da82ddcb060))
* **vulkan:** GPU long-tail batch 3 part 7  ssimulacra2_vulkan kernel ([#156](https://github.com/lusoris/vmaf/issues/156)) ([2ebf376](https://github.com/lusoris/vmaf/commit/2ebf3764124a2279eb294380610ac3fbce35bfc2))
* **vulkan:** Scaffold-only audit-first PR for the Vulkan compute backend (T5-1, ADR-0175) ([#111](https://github.com/lusoris/vmaf/issues/111)) ([0b787cf](https://github.com/lusoris/vmaf/commit/0b787cf7f12f867f80c990774c3e02f95b353143))
* **vulkan:** T-VULKAN-PREALLOC  picture preallocation surface (ADR-0238) ([#264](https://github.com/lusoris/vmaf/issues/264)) ([2493d90](https://github.com/lusoris/vmaf/commit/2493d90bffee7c8256d0af4e8717e086d3e8ee84))
* **vulkan:** T3-15(b)  psnr chroma (psnr_cb / psnr_cr) on Vulkan ([#204](https://github.com/lusoris/vmaf/issues/204)) ([a78d6e9](https://github.com/lusoris/vmaf/commit/a78d6e95ecc218457a0853089139cecc8ecb9371))
* **vulkan:** T5-1b runtime + VIF dispatch pathfinder (Arc A380 verified) ([#116](https://github.com/lusoris/vmaf/issues/116)) ([bf5f861](https://github.com/lusoris/vmaf/commit/bf5f861b5815f7e87ea002c2ad83e53df4d56f11))
* **vulkan:** T5-1b-iv VIF math port (4-scale GLSL kernel + vif_vulkan extractor) ([#117](https://github.com/lusoris/vmaf/issues/117)) ([acf9f5b](https://github.com/lusoris/vmaf/commit/acf9f5b8e3a9bcc3d13e33bd2cd6708f2b1cd4f3))
* **vulkan:** T5-1b-v cross-backend gate + CLI (--vulkan_device) ([#118](https://github.com/lusoris/vmaf/issues/118)) ([50758ea](https://github.com/lusoris/vmaf/commit/50758ea8f24b0cc44e1b85059d92f6c1207d1028))
* **vulkan:** T5-1c motion kernel + cross-backend gate extension ([#119](https://github.com/lusoris/vmaf/issues/119)) ([32e31e4](https://github.com/lusoris/vmaf/commit/32e31e4568a9b71f223e43a500c2af8fecf4401d))
* **vulkan:** T5-1c-adm ADM kernel + cross-backend bug-fix marathon ([#120](https://github.com/lusoris/vmaf/issues/120)) ([7c5b63a](https://github.com/lusoris/vmaf/commit/7c5b63a282afb4e804815d0600a322485f4e8d11))
* **vulkan:** T7-29 follow-up [#3](https://github.com/lusoris/vmaf/issues/3)  public max_outstanding_frames knob (ADR-0235) ([#260](https://github.com/lusoris/vmaf/issues/260)) ([1891746](https://github.com/lusoris/vmaf/commit/18917462a1afa2d626019d379bde7a677da46c4e))
* **vulkan:** T7-29 part 1  VkImage import C-API scaffold ([#128](https://github.com/lusoris/vmaf/issues/128)) ([6bea86d](https://github.com/lusoris/vmaf/commit/6bea86d858a11af923f79cee88128a9d71532c39))
* **vulkan:** T7-29 part 4  v2 async pending-fence ring (ADR-0235) ([#241](https://github.com/lusoris/vmaf/issues/241)) ([e266bf8](https://github.com/lusoris/vmaf/commit/e266bf8ee034f007490fa9653e9a74038fd0f0c2))
* **vulkan:** T7-29 parts 2 + 3  VkImage import impl + libvmaf_vulkan filter ([#134](https://github.com/lusoris/vmaf/issues/134)) ([fe31f80](https://github.com/lusoris/vmaf/commit/fe31f803a43c5ea532fabe2c55a9dea871a3f6cd))
* **vulkan:** T7-36  cambi Vulkan integration (Strategy II) ([#196](https://github.com/lusoris/vmaf/issues/196)) ([9f88b91](https://github.com/lusoris/vmaf/commit/9f88b91b7b7589686eee265391d7954cc877de56))


### Bug Fixes

* **adm:** Close T7-16  Vulkan/SYCL adm_scale2 drift verified at places=4 ([#173](https://github.com/lusoris/vmaf/issues/173)) ([0fb482f](https://github.com/lusoris/vmaf/commit/0fb482fba944cf9b21dd671914deb732df0413de))
* **ai:** Bump torch &gt;=2.8 + lightning &gt;=2.5 for CVE fixes ([#11](https://github.com/lusoris/vmaf/issues/11)) ([98da94d](https://github.com/lusoris/vmaf/commit/98da94d616433879c5d5d9326941fbcec3143a33))
* **ai:** Regenerate bisect cache with pinned pandas 2.3.3 ([#42](https://github.com/lusoris/vmaf/issues/42)) ([cdb9b9e](https://github.com/lusoris/vmaf/commit/cdb9b9ea01dd70682f7e9b6a7cdf16bfa2bbca9a))
* **ai:** Switch lightning ’ pytorch-lightning (PyPI 404) ([#232](https://github.com/lusoris/vmaf/issues/232)) ([c182dfe](https://github.com/lusoris/vmaf/commit/c182dfeb5672159685f03b4d80128daac859cd6a))
* **bench:** Testdata/bench_all.sh engages the backends it benches ([#171](https://github.com/lusoris/vmaf/issues/171)) ([1a1e5eb](https://github.com/lusoris/vmaf/commit/1a1e5eb0af82fedc0bf6c2396e346e612f9e6944))
* **build:** Install libvmaf_vulkan.h under prefix when enable_vulkan ([#175](https://github.com/lusoris/vmaf/issues/175)) ([4b43ad2](https://github.com/lusoris/vmaf/commit/4b43ad2f60c812fe199d7fc64caa74f93cf24b14))
* **ci:** Coverage gate lcov’gcovr + ORT + lint upstream tests in-tree ([#46](https://github.com/lusoris/vmaf/issues/46)) ([652aa70](https://github.com/lusoris/vmaf/commit/652aa70b3457213db881240c07b821abee374af9))
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
* **motion:** Close T7-15  CUDA/SYCL motion drift verified bit-exact on master ([#172](https://github.com/lusoris/vmaf/issues/172)) ([00cbc92](https://github.com/lusoris/vmaf/commit/00cbc921a5709f42d517d3d0281fcc119bda80af))
* SIMD bit-identical reductions + CI fixes ([#18](https://github.com/lusoris/vmaf/issues/18)) ([f082cfd](https://github.com/lusoris/vmaf/commit/f082cfd3a5eb471ca5b32e8f7ea32854c95ed152))
* **sycl:** Require icpx as project C++ compiler when enable_sycl=true ([#115](https://github.com/lusoris/vmaf/issues/115)) ([b8b8f50](https://github.com/lusoris/vmaf/commit/b8b8f50b59ea14089591aed833600fd501fd5819))
* **test:** Gate test_speed on enable_float to match speed.c compile guard ([#263](https://github.com/lusoris/vmaf/issues/263)) ([cb1d49c](https://github.com/lusoris/vmaf/commit/cb1d49c63ac15e1ac283739bbab59ab4c13bee15))
* **vulkan:** Move volk -include flag off volk_dep.compile_args (ADR-0200) ([#155](https://github.com/lusoris/vmaf/issues/155)) ([8bc4f65](https://github.com/lusoris/vmaf/commit/8bc4f65be8ec2ee1149da72ac3b016f66f7768ad))
* **vulkan:** Rename volk vk* symbols to vmaf_priv_vk* for static archives (ADR-0198) ([#152](https://github.com/lusoris/vmaf/issues/152)) ([73620ff](https://github.com/lusoris/vmaf/commit/73620ff504f999ad8b5b972f200c82b5fc4475ba))
* **vulkan:** T7-31  hide volk / vk* symbols from libvmaf.so public ABI ([#132](https://github.com/lusoris/vmaf/issues/132)) ([a1c1a20](https://github.com/lusoris/vmaf/commit/a1c1a2075379107d9e152c8738a4b5add286aafa))


### Performance

* **sycl:** T7-17  fp64-less device fallback for VIF gain-limiting ([#209](https://github.com/lusoris/vmaf/issues/209)) ([606a3fc](https://github.com/lusoris/vmaf/commit/606a3fc0d8348b2e9f2bd50fae407f05b936d1cb))
* **thread_pool:** Recycle job slots + inline data buffer (ADR-0147) ([#83](https://github.com/lusoris/vmaf/issues/83)) ([8fb2fe1](https://github.com/lusoris/vmaf/commit/8fb2fe17425ae1593d0006ef5b1c186d9dd9047d))


### Refactors

* **ai:** Extract tiny-AI extractor template (cuts 150’30 LOC per new extractor) ([#251](https://github.com/lusoris/vmaf/issues/251)) ([1444bb7](https://github.com/lusoris/vmaf/commit/1444bb7ab50c38a09b7e1c46b75526b946f4f5d5))
* **ai:** Migrate feature_mobilesal + transnet_v2 to tiny_extractor_template.h ([#265](https://github.com/lusoris/vmaf/issues/265)) ([4c59ece](https://github.com/lusoris/vmaf/commit/4c59ece2abdfe2269a7b788db26abed05623af56))
* **cuda:** T-GPU-DEDUP-4  first consumer of cuda/kernel_template.h (psnr_cuda) ([#269](https://github.com/lusoris/vmaf/issues/269)) ([133dd5f](https://github.com/lusoris/vmaf/commit/133dd5f30f8bda678e5d9fb40497dcbf1bf82994))
* **gpu:** Land per-backend kernel scaffolding templates (CUDA + Vulkan, no migrations) ([#254](https://github.com/lusoris/vmaf/issues/254)) ([4aa75b1](https://github.com/lusoris/vmaf/commit/4aa75b1e486c13dda146398a4273e8e7d6f849a9))
* **gpu:** T-GPU-DEDUP-1  promote ring_buffer to gpu_picture_pool (ADR-0239) ([#266](https://github.com/lusoris/vmaf/issues/266)) ([19d7eda](https://github.com/lusoris/vmaf/commit/19d7eda2036162e7b595c817135f805981211a88))
* **iqa:** Rename reserved-identifier surface + lint cascade sweep (ADR-0148) ([#84](https://github.com/lusoris/vmaf/issues/84)) ([985be1b](https://github.com/lusoris/vmaf/commit/985be1b9af6a13d32b1159570b816be01824b337))
* **libvmaf:** Sweep readability-function-size NOLINTs (ADR-0146) ([#82](https://github.com/lusoris/vmaf/issues/82)) ([07615a2](https://github.com/lusoris/vmaf/commit/07615a26b86dd6f0168c5d149c78a3341c9acab7))
* **lint:** T7-7 SYCL lint cleanup (162 ’ 4 findings) ([#114](https://github.com/lusoris/vmaf/issues/114)) ([1f1c742](https://github.com/lusoris/vmaf/commit/1f1c742e191cd9be1f2aa437040de23d023e8687))
* **lint:** Whole-codebase clang-tidy auto-fix subset (52% cleared) ([#113](https://github.com/lusoris/vmaf/issues/113)) ([a87a8f5](https://github.com/lusoris/vmaf/commit/a87a8f51eb1f3c3d82f05843a54a4100052d515e))
* **test:** Extract SIMD bit-exact test harness (cuts 50’20 LOC per new test) ([#252](https://github.com/lusoris/vmaf/issues/252)) ([f970975](https://github.com/lusoris/vmaf/commit/f97097560036be65b4eace28f4163c2cb3808e2c))
* **test:** Tiny-AI registration macro  4 test files dedup (-286 LOC) ([#268](https://github.com/lusoris/vmaf/issues/268)) ([6e64899](https://github.com/lusoris/vmaf/commit/6e64899e72550d0e100b3e7f3f893109f4f6838d))
* **vulkan:** T-GPU-DEDUP-5  first consumer of vulkan/kernel_template.h ([#270](https://github.com/lusoris/vmaf/issues/270)) ([14b5ed1](https://github.com/lusoris/vmaf/commit/14b5ed14c86d584a0b013d0d3ddab08e63a8e4c7))
* **vulkan:** T-GPU-DEDUP-6  moment + ciede consumers of vulkan/kernel_template.h ([#271](https://github.com/lusoris/vmaf/issues/271)) ([eae9e85](https://github.com/lusoris/vmaf/commit/eae9e8587aa97792a8561f1eb337dc0b13a9b0d3))
* **vulkan:** T-GPU-DEDUP-7  migrate motion + ssim to kernel_template ([#272](https://github.com/lusoris/vmaf/issues/272)) ([69cc940](https://github.com/lusoris/vmaf/commit/69cc940ac6bcc17386509cd9a1eb931c4d2520a9))


### Documentation

* **.claude:** Refresh skill/agent/hook descriptions to current repo state ([#242](https://github.com/lusoris/vmaf/issues/242)) ([9852fcc](https://github.com/lusoris/vmaf/commit/9852fcc37d086a7d88d5d058063f46c3822e0a18))
* Add community hygiene files (SECURITY, CoC, CODEOWNERS, templates) ([c28dd78](https://github.com/lusoris/vmaf/commit/c28dd78eb3e2a68681beca6382d32c70b62e8e42))
* Add per-distro install guides + BENCHMARKS + hygiene files ([fe8c744](https://github.com/lusoris/vmaf/commit/fe8c744d4da174cb453b98c5be0afa4b539e383d))
* Add SYCL backend, SIMD, and GPU documentation ([3a0ee4f](https://github.com/lusoris/vmaf/commit/3a0ee4f915ce89ec1f14c0e9ae00f75f4149683f))
* Add SYCL bundling guide for self-contained deployment ([5fe843f](https://github.com/lusoris/vmaf/commit/5fe843fd03a90114de40f2bc6e5fc4a1a189fa5a))
* **adr:** ADR-0180  CPU coverage audit closes 5 stale gaps ([#123](https://github.com/lusoris/vmaf/issues/123)) ([fb47f44](https://github.com/lusoris/vmaf/commit/fb47f44ee0b2ff5e7f4bfe66ce7122a8319af6c6))
* **adr:** ADR-0188  GPU long-tail batch 2 scope (ssim / ms_ssim / psnr_hvs) ([#138](https://github.com/lusoris/vmaf/issues/138)) ([96fbe59](https://github.com/lusoris/vmaf/commit/96fbe5997c29801217e702d0c03dd1c2805126af))
* **adr:** ADR-0192  GPU long-tail batch 3 scope (every remaining gap) ([#145](https://github.com/lusoris/vmaf/issues/145)) ([57a03db](https://github.com/lusoris/vmaf/commit/57a03db4cd48270c7d64dc34e5d1ad93d0eb445e))
* **adr:** ADR-0205  cambi GPU feasibility spike (defer integration) ([#159](https://github.com/lusoris/vmaf/issues/159)) ([f48685f](https://github.com/lusoris/vmaf/commit/f48685f3161384ee239e03dfe7aa6a2ca9fa6ec5))
* **adr:** ADR-0207  tiny-AI Quantization-Aware Training design ([#168](https://github.com/lusoris/vmaf/issues/168)) ([ca56951](https://github.com/lusoris/vmaf/commit/ca569512dd75b1fb2d74074d5179592690c80dcf))
* **adr:** Q2 modernization governance  SSIMULACRA 2, Vulkan, MCP-in-libvmaf, PTQ int8 ([#67](https://github.com/lusoris/vmaf/issues/67)) ([0c9e331](https://github.com/lusoris/vmaf/commit/0c9e3313601645934ccb5b79d50871b48e93f7c5))
* **adr:** Retroactive errata  ULP=0 claims in ADR-0176/0177 were bogus ([#121](https://github.com/lusoris/vmaf/issues/121)) ([4ddb80a](https://github.com/lusoris/vmaf/commit/4ddb80a475f0d859a24338dfa07698fd26020f02))
* **adr:** Supersede 0025/0028/0036 with paraphrased bodies ([#37](https://github.com/lusoris/vmaf/issues/37)) ([4955dde](https://github.com/lusoris/vmaf/commit/4955dde1b99c26339465e01c4d8211fe176007a4))
* **adr:** T-VMAF-TUNE  quality-aware encode automation umbrella spec (ADR-0237) ([#261](https://github.com/lusoris/vmaf/issues/261)) ([f642602](https://github.com/lusoris/vmaf/commit/f642602c65b9b43398ea84278d809336296837e2))
* **agents:** Document libvmaf backend-engagement foot-guns ([#169](https://github.com/lusoris/vmaf/issues/169)) ([96f1ef1](https://github.com/lusoris/vmaf/commit/96f1ef18ea902c219214d4b0b956d1b921f5450d))
* **audit:** T7-4  quarterly upstream-backlog re-audit (2026-04-29) ([#205](https://github.com/lusoris/vmaf/issues/205)) ([10a71ac](https://github.com/lusoris/vmaf/commit/10a71acc8de6b7ec5e69d6185361c1f30676ff17))
* **backlog:** Land Section-A audit decisions as T-NN cross-links ([#167](https://github.com/lusoris/vmaf/issues/167)) ([58ab35a](https://github.com/lusoris/vmaf/commit/58ab35ad3cbc66e0c6598d0bc27a3f24115f532c))
* **benchmarks:** T7-37 fill TBD cells with measured numbers ([#177](https://github.com/lusoris/vmaf/issues/177)) ([851375a](https://github.com/lusoris/vmaf/commit/851375a172057b3b9cc78fcbb46d44f582ca59c1))
* Consolidate + reorganise documentation tree ([#12](https://github.com/lusoris/vmaf/issues/12)) ([4bbd573](https://github.com/lusoris/vmaf/commit/4bbd573f5d45a9cf6ebfe31757c19c4e0caff703))
* **gpu:** T-GPU-DEDUP-2  GPU backend public-API pattern doc (ADR-0240) ([#267](https://github.com/lusoris/vmaf/issues/267)) ([394d028](https://github.com/lusoris/vmaf/commit/394d028a172644473d24595d0a16ab4263606ddc))
* **integer_adm:** Verify + defer Netflix[#955](https://github.com/lusoris/vmaf/issues/955) i4_adm_cm rounding overflow (ADR-0155) ([#92](https://github.com/lusoris/vmaf/issues/92)) ([f7e5ecf](https://github.com/lusoris/vmaf/commit/f7e5ecf26238a74ed3148e5021a33b0fbdb9b59d))
* Migrate from Sphinx+Doxygen to MkDocs Material ([#17](https://github.com/lusoris/vmaf/issues/17)) ([40fe6f1](https://github.com/lusoris/vmaf/commit/40fe6f18dad514bb95552d0934b28fb7d0ccfcba))
* **planning:** Refresh AGENTS.md invariants + ADR index sync ([#243](https://github.com/lusoris/vmaf/issues/243)) ([2d891a2](https://github.com/lusoris/vmaf/commit/2d891a2e0ff04ab2d7f99375e83fc8a0b7dee4e2))
* Project-wide doc-substance sweep (ADR-0100 batches 1-4) ([#25](https://github.com/lusoris/vmaf/issues/25)) ([4f3f992](https://github.com/lusoris/vmaf/commit/4f3f992d76483a1c769e57d4e11eca58eaa8aee5))
* **readme:** Rewrite for the Lusoris fork, preserve upstream credit ([e68befa](https://github.com/lusoris/vmaf/commit/e68befa6ffda099026ef9c4ef651f44824a22e33))
* **rebase-notes:** Netflix[#1486](https://github.com/lusoris/vmaf/issues/1486) motion updates verified present (ADR-0158) ([#95](https://github.com/lusoris/vmaf/issues/95)) ([383190a](https://github.com/lusoris/vmaf/commit/383190a481d2170e37a5a4f7619b704e722cba0b))
* **research:** Land Bristol NVC review + 2026-05 CI audit digests ([#240](https://github.com/lusoris/vmaf/issues/240)) ([267b67f](https://github.com/lusoris/vmaf/commit/267b67f92446181e1f2de3e6ce8918b0c284f4d2))
* **research:** Research-0024  vif/adm upstream-divergence digest ([#182](https://github.com/lusoris/vmaf/issues/182)) ([d8cf89d](https://github.com/lusoris/vmaf/commit/d8cf89d3cf7e4d43558a6bc24d631d7f301f34f7))
* **research:** Research-0025  FoxBird outlier resolved via KoNViD combined training ([#183](https://github.com/lusoris/vmaf/issues/183)) ([da2de07](https://github.com/lusoris/vmaf/commit/da2de07c93a331b913072ebea7bf6e4250bad87c))
* **research:** Research-0026  cross-metric feature fusion plan ([#184](https://github.com/lusoris/vmaf/issues/184)) ([d58271c](https://github.com/lusoris/vmaf/commit/d58271c98d9af496b3b6983fd0b21cf496543d40))
* **research:** Research-0027  Phase-2 feature importance results ([#187](https://github.com/lusoris/vmaf/issues/187)) ([a79f9bf](https://github.com/lusoris/vmaf/commit/a79f9bf20655b761af3283d3e0c5d56f8385a286))
* **research:** T7-9  Intel AI-PC NPU/EP applicability digest ([#194](https://github.com/lusoris/vmaf/issues/194)) ([e1244aa](https://github.com/lusoris/vmaf/commit/e1244aab686f55e6688c9d8558e6704650a1dc8a))
* **research:** Tiny-AI corpus + architecture survey for next iteration ([#166](https://github.com/lusoris/vmaf/issues/166)) ([91ade5b](https://github.com/lusoris/vmaf/commit/91ade5bc8b02fcd86aec9d68aeb692acb289f545))
* **state:** Refresh post-session-2026-04-29 (verify rows, unblock T6-1a Netflix Public dataset row) ([#245](https://github.com/lusoris/vmaf/issues/245)) ([42b6d35](https://github.com/lusoris/vmaf/commit/42b6d354dc7d59e3cb6d500b12fe371b4df41bd6))
* **tiny-ai:** User docs + README row + LFS + release-please subpackages ([66bd6bd](https://github.com/lusoris/vmaf/commit/66bd6bd1f5cd5e120513b85fa224de6a11cf97d8))
* **top:** Refresh README + supporting top-level docs to current codebase state ([#215](https://github.com/lusoris/vmaf/issues/215)) ([9ab3fdb](https://github.com/lusoris/vmaf/commit/9ab3fdb3cc083c001da0d8ef5f7ae32be0663e1d))
* **usage:** Add NVC-style BD-rate recipe with VMAF (Research-0033 [#4](https://github.com/lusoris/vmaf/issues/4)) ([#258](https://github.com/lusoris/vmaf/issues/258)) ([5b32dad](https://github.com/lusoris/vmaf/commit/5b32dade87c619730a965fac10309e97bb7a6d76))
* **usage:** T7-27  ffmpeg per-backend copy-paste examples ([#126](https://github.com/lusoris/vmaf/issues/126)) ([f7098a7](https://github.com/lusoris/vmaf/commit/f7098a791b548a0cb465aea65ad5ce9f42db2cd3))
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
* **backlog:** T7-32  3 micro-investigations bundled (motion_v2 srlv64 + tiny-vmaf-v2 identity + routine.py FIXME) ([#198](https://github.com/lusoris/vmaf/issues/198)) ([8e0eb8f](https://github.com/lusoris/vmaf/commit/8e0eb8f7cf9b1f88b4a73b089f113bb3d1ef24ad))
* **ci:** Cache Netflix vmaf_resource fixtures (actions/cache@v5) ([#131](https://github.com/lusoris/vmaf/issues/131)) ([f1399c7](https://github.com/lusoris/vmaf/commit/f1399c7bcc888703d8f5587eba48c00840da8181))
* **ci:** Finish Node 24 bump  scorecard artifact SHA + nightly-bisect setup-python ([#49](https://github.com/lusoris/vmaf/issues/49)) ([3eb9af7](https://github.com/lusoris/vmaf/commit/3eb9af7ea2a568aab88ae619db49a2e2a1b2a3cf))
* **ci:** Rename workflows + Title Case display names (ADR-0116) ([#53](https://github.com/lusoris/vmaf/issues/53)) ([f4379c8](https://github.com/lusoris/vmaf/commit/f4379c870750400b643c38644261fd361e2f59fd))
* **ci:** Revert pip&lt;26.1 pin in Tiny AI workflow (PR [#231](https://github.com/lusoris/vmaf/issues/231)) ([#233](https://github.com/lusoris/vmaf/issues/233)) ([ca8c964](https://github.com/lusoris/vmaf/commit/ca8c96483706944f41fac89f80d88dba87b46853))
* **ci:** T7-CI-DEDUP  drop redundant python-lint + shellcheck, demote docker-image to advisory ([#257](https://github.com/lusoris/vmaf/issues/257)) ([5bf941d](https://github.com/lusoris/vmaf/commit/5bf941d34264d9041716b2a8bec342b177308de8))
* **dnn:** T7-12  remove VMAF_MAX_MODEL_BYTES env override ([#193](https://github.com/lusoris/vmaf/issues/193)) ([f87384f](https://github.com/lusoris/vmaf/commit/f87384fc0257dacdc8bfd832bd00c280bd27cc2e))
* **docs:** Audit of untracked follow-up items (2026-04-28) ([#161](https://github.com/lusoris/vmaf/issues/161)) ([3a6e598](https://github.com/lusoris/vmaf/commit/3a6e5982be35d9dec94a862c71c6f89130f0a084))
* **docs:** Clear Section C stale comments (audit-2026-04-28) ([#164](https://github.com/lusoris/vmaf/issues/164)) ([91b4f75](https://github.com/lusoris/vmaf/commit/91b4f75bad706ca63d4d370d8012819480d86546))
* **ffmpeg-patches:** Refresh against FFmpeg n8.1 + clarify SSIMULACRA 2 is patchless ([#101](https://github.com/lusoris/vmaf/issues/101)) ([439282e](https://github.com/lusoris/vmaf/commit/439282ea5691433bd3a9142392981fbde079ee64))
* **license:** Bump Netflix copyright from 2016-2020 to 2016-2026 ([c159761](https://github.com/lusoris/vmaf/commit/c159761dbe8bc9ee6d459e149dc7000ea50760ef))
* **license:** Correct fork copyright year to 2026 ([0e98c94](https://github.com/lusoris/vmaf/commit/0e98c949e2598d8d05c40a75b88f42d8b6d5c063))
* **license:** Re-attribute SYCL files to Lusoris and Claude ([a185f8e](https://github.com/lusoris/vmaf/commit/a185f8ef52d0e166dabab0a03588e111c126f2a2))
* **lint:** Clang-tidy upstream cleanup rounds 2-4 ([#2](https://github.com/lusoris/vmaf/issues/2)) ([722d21f](https://github.com/lusoris/vmaf/commit/722d21fd4e3c106abc411d584b4a84ce306d758e))
* Post-merge cleanup  CI fix + lint + supply-chain + scorecard + dependabot ([#14](https://github.com/lusoris/vmaf/issues/14)) ([798db39](https://github.com/lusoris/vmaf/commit/798db3941dea7757b764287f5fda784064430a96))
* **release:** Introduce CHANGELOG + ADR-index fragment files (drop merge-conflict pain) ([#253](https://github.com/lusoris/vmaf/issues/253)) ([1254a5a](https://github.com/lusoris/vmaf/commit/1254a5ac7e08751acc31dd1f8bca0b551fe48df9))
* **repo:** Add AI-agent scaffolding and engineering-principles docs ([b799db5](https://github.com/lusoris/vmaf/commit/b799db5be6380cbde3c2e2d215610f9d0f2024a1))
* **skills:** Sync-upstream detects port-only topology ([#75](https://github.com/lusoris/vmaf/issues/75)) ([40b97cd](https://github.com/lusoris/vmaf/commit/40b97cdf98264a798d4bc1320b08bb9034ad5f76))
* **sycl:** T7-13  toolchain cleanup (oneAPI multi-version recipe + icpx clang-tidy wrapper) ([#206](https://github.com/lusoris/vmaf/issues/206)) ([cd806e3](https://github.com/lusoris/vmaf/commit/cd806e3d9899e6b30940c0e21bed15d2690ac3d5))
* **upstream:** Port 798409e3 + 314db130  CUDA null-deref + remove all.c ([#181](https://github.com/lusoris/vmaf/issues/181)) ([6eab09c](https://github.com/lusoris/vmaf/commit/6eab09c05c454db4c0a0f585379d1d4fb21f124e))
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
</details>

<details><summary>0.2.0</summary>

## [0.2.0](https://github.com/lusoris/vmaf/compare/v0.1.0...v0.2.0) (2026-05-02)


### Features

* **ai:** BVI-DVC feature-extraction pipeline (corpus-3 for tiny-AI v2) ([#214](https://github.com/lusoris/vmaf/issues/214)) ([362d9e7](https://github.com/lusoris/vmaf/commit/362d9e7f256288fd469215aec45fdd91966c1eb2))
* **ai:** Combined Netflix + KoNViD-1k tiny-AI trainer driver ([#180](https://github.com/lusoris/vmaf/issues/180)) ([a143e25](https://github.com/lusoris/vmaf/commit/a143e255c5d8130872be25bdc1fdb757fcc878cf))
* **ai:** Full vmaf-train package  fr/nr/filter families + CLI + registry ([91b5558](https://github.com/lusoris/vmaf/commit/91b5558e133bca0b47d1199be8ecff85bb3ac288))
* **ai:** KoNViD-1k ’ VMAF-pair acquisition + loader bridge ([#178](https://github.com/lusoris/vmaf/issues/178)) ([219f653](https://github.com/lusoris/vmaf/commit/219f65315d6470b20bc60e7af1905c683524404d))
* **ai:** Model validators + MCP eval + variance head + INT8 PTQ (4/5) ([#8](https://github.com/lusoris/vmaf/issues/8)) ([81f63a4](https://github.com/lusoris/vmaf/commit/81f63a4e4695e11d33276fe50b44644fb64e1fa7))
* **ai:** Research-0028 + Phase-3 subset-sweep driver (negative result) ([#188](https://github.com/lusoris/vmaf/issues/188)) ([7e4e884](https://github.com/lusoris/vmaf/commit/7e4e88423702b962512f630e642c937097fe7dfe))
* **ai:** Research-0029 Phase-3b  StandardScaler validates broader-feature hypothesis ([#192](https://github.com/lusoris/vmaf/issues/192)) ([a1453bf](https://github.com/lusoris/vmaf/commit/a1453bff4caad21e873a303a9666eb4fc586b103))
* **ai:** Research-0030 Phase-3b multi-seed validation (Gate 1 passed) ([#190](https://github.com/lusoris/vmaf/issues/190)) ([1f128b0](https://github.com/lusoris/vmaf/commit/1f128b01f75678c95f63dd8f848aaf2b4fc2e7b2))
* **ai:** Retrain vmaf_tiny_v2 on 4-corpus (NF+KV+BVI-A+B+C+D) ([#255](https://github.com/lusoris/vmaf/issues/255)) ([53837dc](https://github.com/lusoris/vmaf/commit/53837dcbba23c1a72653a238dfe6cd0a56b10411))
* **ai:** Ship vmaf_tiny_v2 (canonical-6 + StandardScaler + lr=1e-3 + 90ep, validated +0.018 PLCC over v1) ([#250](https://github.com/lusoris/vmaf/issues/250)) ([3999cda](https://github.com/lusoris/vmaf/commit/3999cdab6163cb4512bb5fb7b682b9e78c5e99cd))
* **ai:** T5-3d  nr_metric_v1 dynamic-batch fix + PTQ pipeline ([#247](https://github.com/lusoris/vmaf/issues/247)) ([9ff5d1d](https://github.com/lusoris/vmaf/commit/9ff5d1d48428d4f1ef01236a8927b2da26c6b3f5))
* **ai:** T5-3e empirical PTQ accuracy across CPU/CUDA/OpenVINO EPs ([#174](https://github.com/lusoris/vmaf/issues/174)) ([4130149](https://github.com/lusoris/vmaf/commit/413014963259ce54d3f06ac2f0d1659f6a2d2477))
* **ai:** T6-1a  fr_regressor_v1 C1 baseline (Netflix Public, unblocked) ([#249](https://github.com/lusoris/vmaf/issues/249)) ([f809ce0](https://github.com/lusoris/vmaf/commit/f809ce09c214156259790c91109cea02ce2640f6))
* **ai:** T6-2a  MobileSal saliency feature extractor ([#208](https://github.com/lusoris/vmaf/issues/208)) ([fa7d4f5](https://github.com/lusoris/vmaf/commit/fa7d4f52833d087a8d47236e5bdc85be5dd37205))
* **ai:** T6-3a  TransNet V2 shot-boundary feature extractor ([#210](https://github.com/lusoris/vmaf/issues/210)) ([08b7644](https://github.com/lusoris/vmaf/commit/08b7644f8a31c70623f52e5a1dd5c47ca5f08493))
* **ai:** T6-7  FastDVDnet temporal pre-filter (5-frame window) ([#203](https://github.com/lusoris/vmaf/issues/203)) ([cf1d670](https://github.com/lusoris/vmaf/commit/cf1d670d74ea98ed66baf2e3e57acf686189a456))
* **ai:** T6-9  model registry schema + Sigstore --tiny-model-verify ([#199](https://github.com/lusoris/vmaf/issues/199)) ([9293d69](https://github.com/lusoris/vmaf/commit/9293d6983d5b87525163c873f54a1b0b84845c74))
* **ai:** T7-CODEC-AWARE  codec-conditioned FR regressor surface (training BLOCKED) ([#237](https://github.com/lusoris/vmaf/issues/237)) ([876382d](https://github.com/lusoris/vmaf/commit/876382dc805bc53fe7f98d647ad5d0b98c72da60))
* **ai:** T7-GPU-ULP-CAL  scaffold GPU-gen ULP calibration head (proposal) ([#238](https://github.com/lusoris/vmaf/issues/238)) ([fdb8a56](https://github.com/lusoris/vmaf/commit/fdb8a5652d2f42bcf7d243af19804f1d4eb81b12))
* **ai:** Tiny-AI 3-arch LOSO evaluation harness + Research-0023 ([#176](https://github.com/lusoris/vmaf/issues/176)) ([0e483a3](https://github.com/lusoris/vmaf/commit/0e483a3e16461727762c01b9f2a7b1bdf2db1d72))
* **ai:** Tiny-AI feature-set registry (Research-0026 Phase 1) ([#185](https://github.com/lusoris/vmaf/issues/185)) ([4336736](https://github.com/lusoris/vmaf/commit/4336736ae2e52ab6a12728f9708777886fe6a5dd))
* **ai:** Tiny-AI LOSO evaluation harness for mlp_small ([#165](https://github.com/lusoris/vmaf/issues/165)) ([427272b](https://github.com/lusoris/vmaf/commit/427272b48b18e0feb3d239112c24318e0187f34c))
* **ai:** Tiny-AI Phase-2 analysis scaffolding (Research-0026) ([#191](https://github.com/lusoris/vmaf/issues/191)) ([0df6543](https://github.com/lusoris/vmaf/commit/0df6543a8803017a67ae5c0454d6ea0c53093152))
* **ai:** Tiny-AI Quantization-Aware Training (QAT) implementation (T5-4) ([#179](https://github.com/lusoris/vmaf/issues/179)) ([d71ac52](https://github.com/lusoris/vmaf/commit/d71ac52cf5b995e21262a89f87d468fc7c28eade))
* **ai:** Tiny-AI training prep (loader + eval + Lightning harness for Netflix corpus) ([#158](https://github.com/lusoris/vmaf/issues/158)) ([aa74eaa](https://github.com/lusoris/vmaf/commit/aa74eaa3b5ac74263caceb053a218ba113a32345))
* **ci:** Nightly bisect-model-quality + sticky tracker (closes [#4](https://github.com/lusoris/vmaf/issues/4)) ([#41](https://github.com/lusoris/vmaf/issues/41)) ([6cd4fb0](https://github.com/lusoris/vmaf/commit/6cd4fb01a81cff13996ad94cf552f7b61104cf60))
* DNN session runtime + SYCL list-devices + windows CI (1/5) ([#5](https://github.com/lusoris/vmaf/issues/5)) ([3cfa85b](https://github.com/lusoris/vmaf/commit/3cfa85bb4cad0b054df76c1e75c2a9761a4bdc56))
* **dnn:** Admit Loop + If on ONNX op-allowlist with recursive subgraph scan (T6-5, ADR-0169) ([#105](https://github.com/lusoris/vmaf/issues/105)) ([c4bd6ff](https://github.com/lusoris/vmaf/commit/c4bd6ff04fd03616cd24c7291d4be213a3638575))
* **dnn:** Bounded Loop.M trip-count guard (T6-5b, ADR-0171) ([#107](https://github.com/lusoris/vmaf/issues/107)) ([c264eec](https://github.com/lusoris/vmaf/commit/c264eec78d51e5a19cf35a59c76f3a7393c1e071))
* **dnn:** LPIPS-SqueezeNet FR extractor + ONNX + registry entry ([#23](https://github.com/lusoris/vmaf/issues/23)) ([0267dfb](https://github.com/lusoris/vmaf/commit/0267dfbe83b28366fcad56268d74d351f3b42db7))
* **dnn:** Scaffold tiny-AI training, ONNX export, and DNN C seam ([d122b72](https://github.com/lusoris/vmaf/commit/d122b72122e35a5e40399a0625ee07c227baf765))
* **tiny-ai:** C2 nr_metric_v1 + C3 learned_filter_v1 baselines on KoNViD-1k (T6-1, ADR-0168) ([#104](https://github.com/lusoris/vmaf/issues/104)) ([1a48eab](https://github.com/lusoris/vmaf/commit/1a48eab976f39a926dec5f01d94bc3c363755522))
* **tiny-ai:** First per-model PTQ  learned_filter_v1 dynamic int8 (T5-3b, ADR-0174) ([#110](https://github.com/lusoris/vmaf/issues/110)) ([07a3053](https://github.com/lusoris/vmaf/commit/07a30535c97314851849237a235981549d5af4cf))
* **tiny-ai:** PTQ int8 audit harness  registry + scripts + sidecar parser (T5-3, ADR-0173) ([#109](https://github.com/lusoris/vmaf/issues/109)) ([a7f2664](https://github.com/lusoris/vmaf/commit/a7f2664cb4948a985c5ec52e7b1cc508596f5804))


### Bug Fixes

* **ai:** Bump torch &gt;=2.8 + lightning &gt;=2.5 for CVE fixes ([#11](https://github.com/lusoris/vmaf/issues/11)) ([98da94d](https://github.com/lusoris/vmaf/commit/98da94d616433879c5d5d9326941fbcec3143a33))
* **ai:** Regenerate bisect cache with pinned pandas 2.3.3 ([#42](https://github.com/lusoris/vmaf/issues/42)) ([cdb9b9e](https://github.com/lusoris/vmaf/commit/cdb9b9ea01dd70682f7e9b6a7cdf16bfa2bbca9a))
* **ai:** Switch lightning ’ pytorch-lightning (PyPI 404) ([#232](https://github.com/lusoris/vmaf/issues/232)) ([c182dfe](https://github.com/lusoris/vmaf/commit/c182dfeb5672159685f03b4d80128daac859cd6a))
* SIMD bit-identical reductions + CI fixes ([#18](https://github.com/lusoris/vmaf/issues/18)) ([f082cfd](https://github.com/lusoris/vmaf/commit/f082cfd3a5eb471ca5b32e8f7ea32854c95ed152))


### Build System

* CUDA 13 + oneAPI 2025.3 + clang-format 22 + black 26 (3/5) ([#7](https://github.com/lusoris/vmaf/issues/7)) ([a7be84c](https://github.com/lusoris/vmaf/commit/a7be84cb5cc6b80659bf2c799aaf62221b335dab))


### Miscellaneous

* **adr:** Migrate to Nygard one-file-per-decision + golusoris-alignment sweep ([#24](https://github.com/lusoris/vmaf/issues/24)) ([8e3cd22](https://github.com/lusoris/vmaf/commit/8e3cd22c1240ce9f11bb01f8bfd95a2230a598b1))
* **lint:** Clang-tidy upstream cleanup rounds 2-4 ([#2](https://github.com/lusoris/vmaf/issues/2)) ([722d21f](https://github.com/lusoris/vmaf/commit/722d21fd4e3c106abc411d584b4a84ce306d758e))
* Post-merge cleanup  CI fix + lint + supply-chain + scorecard + dependabot ([#14](https://github.com/lusoris/vmaf/issues/14)) ([798db39](https://github.com/lusoris/vmaf/commit/798db3941dea7757b764287f5fda784064430a96))
* **upstream:** Record ours-merge of Netflix 966be8d5 (already ported in d06dd6cf) ([fddc5ca](https://github.com/lusoris/vmaf/commit/fddc5ca7cbc0f406d0269c7c5ff98e0487d819b8))
* **upstream:** Record ours-merge of Netflix 966be8d5 (bookkeeping) ([27ce439](https://github.com/lusoris/vmaf/commit/27ce43910d5f6c14ec04342966bbc0204e5b2958))
</details>

<details><summary>0.2.0</summary>

## [0.2.0](https://github.com/lusoris/vmaf/compare/v0.1.0...v0.2.0) (2026-05-02)


### Features

* **ai:** Model validators + MCP eval + variance head + INT8 PTQ (4/5) ([#8](https://github.com/lusoris/vmaf/issues/8)) ([81f63a4](https://github.com/lusoris/vmaf/commit/81f63a4e4695e11d33276fe50b44644fb64e1fa7))
* **dev-llm:** Ollama-backed review / commitmsg / docgen helpers ([80ead16](https://github.com/lusoris/vmaf/commit/80ead1668e8b3ffba6bee7c7bde2492b649da68c))


### Miscellaneous

* Post-merge cleanup  CI fix + lint + supply-chain + scorecard + dependabot ([#14](https://github.com/lusoris/vmaf/issues/14)) ([798db39](https://github.com/lusoris/vmaf/commit/798db3941dea7757b764287f5fda784064430a96))
* **upstream:** Record ours-merge of Netflix 966be8d5 (already ported in d06dd6cf) ([fddc5ca](https://github.com/lusoris/vmaf/commit/fddc5ca7cbc0f406d0269c7c5ff98e0487d819b8))
* **upstream:** Record ours-merge of Netflix 966be8d5 (bookkeeping) ([27ce439](https://github.com/lusoris/vmaf/commit/27ce43910d5f6c14ec04342966bbc0204e5b2958))
</details>

<details><summary>0.2.0</summary>

## [0.2.0](https://github.com/lusoris/vmaf/compare/v0.1.0...v0.2.0) (2026-05-02)


### Features

* **ai:** Model validators + MCP eval + variance head + INT8 PTQ (4/5) ([#8](https://github.com/lusoris/vmaf/issues/8)) ([81f63a4](https://github.com/lusoris/vmaf/commit/81f63a4e4695e11d33276fe50b44644fb64e1fa7))
* **mcp:** Add vmaf-mcp server exposing VMAF to LLM tooling ([0f233b8](https://github.com/lusoris/vmaf/commit/0f233b8075529b1a42fbe77becacf86b5075d1a3))
* **mcp:** Describe_worst_frames tool with VLM fallback (T6-6, ADR-0172) ([#108](https://github.com/lusoris/vmaf/issues/108)) ([5de8a79](https://github.com/lusoris/vmaf/commit/5de8a793eb2c13cfca10079fa3d18c0620cea130))


### Miscellaneous

* Post-merge cleanup  CI fix + lint + supply-chain + scorecard + dependabot ([#14](https://github.com/lusoris/vmaf/issues/14)) ([798db39](https://github.com/lusoris/vmaf/commit/798db3941dea7757b764287f5fda784064430a96))
* **upstream:** Record ours-merge of Netflix 966be8d5 (already ported in d06dd6cf) ([fddc5ca](https://github.com/lusoris/vmaf/commit/fddc5ca7cbc0f406d0269c7c5ff98e0487d819b8))
* **upstream:** Record ours-merge of Netflix 966be8d5 (bookkeeping) ([27ce439](https://github.com/lusoris/vmaf/commit/27ce43910d5f6c14ec04342966bbc0204e5b2958))
</details>

---
This PR was generated with [Release Please](https://github.com/googleapis/release-please). See [documentation](https://github.com/googleapis/release-please#release-please).