# Changelog

## [0.2.0](https://github.com/lusoris/vmaf/compare/v0.1.0...v0.2.0) (2026-05-09)


### Features

* **ai:** BVI-DVC corpus ingestion → bigger fr_regressor training corpus (ADR-0310) ([#407](https://github.com/lusoris/vmaf/issues/407)) ([a159ba9](https://github.com/lusoris/vmaf/commit/a159ba9da304563cb424abba946be6d9effe5edf))
* **ai:** BVI-DVC feature-extraction pipeline (corpus-3 for tiny-AI v2) ([#214](https://github.com/lusoris/vmaf/issues/214)) ([362d9e7](https://github.com/lusoris/vmaf/commit/362d9e7f256288fd469215aec45fdd91966c1eb2))
* **ai:** Combined Netflix + KoNViD-1k tiny-AI trainer driver ([#180](https://github.com/lusoris/vmaf/issues/180)) ([a143e25](https://github.com/lusoris/vmaf/commit/a143e255c5d8130872be25bdc1fdb757fcc878cf))
* **ai:** ENCODER_VOCAB v3 (16-slot) schema expansion + retrain plan (ADR-0302) ([#401](https://github.com/lusoris/vmaf/issues/401)) ([6871625](https://github.com/lusoris/vmaf/commit/687162530b2d290b374d75eded65ba9c8eef1408))
* **ai:** Fr_regressor_v2 codec-aware scaffold (Phase B prereq) ([#347](https://github.com/lusoris/vmaf/issues/347)) ([b40d63a](https://github.com/lusoris/vmaf/commit/b40d63aedb8cca4816a92cb9d37e183654ce84ee))
* **ai:** Fr_regressor_v2 ENCODER_VOCAB v2 (hw codec extension) ([#394](https://github.com/lusoris/vmaf/issues/394)) ([a5ca35d](https://github.com/lusoris/vmaf/commit/a5ca35d0178ae1e093e9ff408684795bc8cca57d))
* **ai:** Fr_regressor_v2 ensemble — full production flip (real ONNX + sidecars, ADR-0321) ([#424](https://github.com/lusoris/vmaf/issues/424)) ([13174fc](https://github.com/lusoris/vmaf/commit/13174fcd4652fad04502d633e0bdfa89e091f901))
* **ai:** Fr_regressor_v2 ensemble — production flip trainer + CI gate (ADR-0303) ([#399](https://github.com/lusoris/vmaf/issues/399)) ([453f136](https://github.com/lusoris/vmaf/commit/453f1364d0511b7b69387f0a523e3f341d2254d9))
* **ai:** Fr_regressor_v2 ensemble — real-corpus retrain harness + flip workflow (ADR-0309) ([#405](https://github.com/lusoris/vmaf/issues/405)) ([e45299e](https://github.com/lusoris/vmaf/commit/e45299e1073dc9a380a97614ef78fe6edc93d239))
* **ai:** Fr_regressor_v2 ensemble LOSO trainer — real corpus loader + per-fold training (ADR-0319) ([#422](https://github.com/lusoris/vmaf/issues/422)) ([882ecbf](https://github.com/lusoris/vmaf/commit/882ecbff21a84978e31dc6fefbb6493e96c6c5a3))
* **ai:** Fr_regressor_v2 probabilistic head (deep-ensemble + conformal scaffold) ([#372](https://github.com/lusoris/vmaf/issues/372)) ([de6c0a0](https://github.com/lusoris/vmaf/commit/de6c0a05f51edeb92fb665ef79d6c8c233b607fc))
* **ai:** Fr_regressor_v3 — train + register on ENCODER_VOCAB v3 (16-slot) ([#428](https://github.com/lusoris/vmaf/issues/428)) ([1447e92](https://github.com/lusoris/vmaf/commit/1447e92a271569195543ea1291c096299dad994f))
* **ai:** Full vmaf-train package — fr/nr/filter families + CLI + registry ([91b5558](https://github.com/lusoris/vmaf/commit/91b5558e133bca0b47d1199be8ecff85bb3ac288))
* **ai:** KoNViD-1k → VMAF-pair acquisition + loader bridge ([#178](https://github.com/lusoris/vmaf/issues/178)) ([219f653](https://github.com/lusoris/vmaf/commit/219f65315d6470b20bc60e7af1905c683524404d))
* **ai:** Model validators + MCP eval + variance head + INT8 PTQ (4/5) ([#8](https://github.com/lusoris/vmaf/issues/8)) ([81f63a4](https://github.com/lusoris/vmaf/commit/81f63a4e4695e11d33276fe50b44644fb64e1fa7))
* **ai:** Research-0028 + Phase-3 subset-sweep driver (negative result) ([#188](https://github.com/lusoris/vmaf/issues/188)) ([7e4e884](https://github.com/lusoris/vmaf/commit/7e4e88423702b962512f630e642c937097fe7dfe))
* **ai:** Research-0029 Phase-3b — StandardScaler validates broader-feature hypothesis ([#192](https://github.com/lusoris/vmaf/issues/192)) ([a1453bf](https://github.com/lusoris/vmaf/commit/a1453bff4caad21e873a303a9666eb4fc586b103))
* **ai:** Research-0030 Phase-3b multi-seed validation (Gate 1 passed) ([#190](https://github.com/lusoris/vmaf/issues/190)) ([1f128b0](https://github.com/lusoris/vmaf/commit/1f128b01f75678c95f63dd8f848aaf2b4fc2e7b2))
* **ai:** Retrain vmaf_tiny_v2 on 4-corpus (NF+KV+BVI-A+B+C+D) ([#255](https://github.com/lusoris/vmaf/issues/255)) ([53837dc](https://github.com/lusoris/vmaf/commit/53837dcbba23c1a72653a238dfe6cd0a56b10411))
* **ai:** Saliency_student_v1 — fork-trained on DUTS (replaces mobilesal placeholder) ([#359](https://github.com/lusoris/vmaf/issues/359)) ([5ee4cc1](https://github.com/lusoris/vmaf/commit/5ee4cc13a857ff3c1e3b01c2fb3196609df60d0f))
* **ai:** Ship vmaf_tiny_v2 (canonical-6 + StandardScaler + lr=1e-3 + 90ep, validated +0.018 PLCC over v1) ([#250](https://github.com/lusoris/vmaf/issues/250)) ([3999cda](https://github.com/lusoris/vmaf/commit/3999cdab6163cb4512bb5fb7b682b9e78c5e99cd))
* **ai:** T5-3d — nr_metric_v1 dynamic-batch fix + PTQ pipeline ([#247](https://github.com/lusoris/vmaf/issues/247)) ([9ff5d1d](https://github.com/lusoris/vmaf/commit/9ff5d1d48428d4f1ef01236a8927b2da26c6b3f5))
* **ai:** T5-3d-followup — PTQ int8 sidecars for vmaf_tiny v3 + v4 ([#351](https://github.com/lusoris/vmaf/issues/351)) ([8a3bf4c](https://github.com/lusoris/vmaf/commit/8a3bf4c66e448a74788ce860bf5a8cd578509429))
* **ai:** T5-3e empirical PTQ accuracy across CPU/CUDA/OpenVINO EPs ([#174](https://github.com/lusoris/vmaf/issues/174)) ([4130149](https://github.com/lusoris/vmaf/commit/413014963259ce54d3f06ac2f0d1659f6a2d2477))
* **ai:** T6-1a — fr_regressor_v1 C1 baseline (Netflix Public, unblocked) ([#249](https://github.com/lusoris/vmaf/issues/249)) ([f809ce0](https://github.com/lusoris/vmaf/commit/f809ce09c214156259790c91109cea02ce2640f6))
* **ai:** T6-2a — MobileSal saliency feature extractor ([#208](https://github.com/lusoris/vmaf/issues/208)) ([fa7d4f5](https://github.com/lusoris/vmaf/commit/fa7d4f52833d087a8d47236e5bdc85be5dd37205))
* **ai:** T6-3a — TransNet V2 shot-boundary feature extractor ([#210](https://github.com/lusoris/vmaf/issues/210)) ([08b7644](https://github.com/lusoris/vmaf/commit/08b7644f8a31c70623f52e5a1dd5c47ca5f08493))
* **ai:** T6-3a-followup — transnet_v2 real upstream weights ([#334](https://github.com/lusoris/vmaf/issues/334)) ([4e253fc](https://github.com/lusoris/vmaf/commit/4e253fcd59b6fa8ed8f7f80c2fc6e9cb23f6253d))
* **ai:** T6-7 — FastDVDnet temporal pre-filter (5-frame window) ([#203](https://github.com/lusoris/vmaf/issues/203)) ([cf1d670](https://github.com/lusoris/vmaf/commit/cf1d670d74ea98ed66baf2e3e57acf686189a456))
* **ai:** T6-7b — FastDVDnet real upstream weights via luma adapter ([#326](https://github.com/lusoris/vmaf/issues/326)) ([bacecba](https://github.com/lusoris/vmaf/commit/bacecba82888ebbb1d9536787eba7575ef61d950))
* **ai:** T6-9 — model registry schema + Sigstore --tiny-model-verify ([#199](https://github.com/lusoris/vmaf/issues/199)) ([9293d69](https://github.com/lusoris/vmaf/commit/9293d6983d5b87525163c873f54a1b0b84845c74))
* **ai:** T7-CODEC-AWARE — codec-conditioned FR regressor surface (training BLOCKED) ([#237](https://github.com/lusoris/vmaf/issues/237)) ([876382d](https://github.com/lusoris/vmaf/commit/876382dc805bc53fe7f98d647ad5d0b98c72da60))
* **ai:** T7-GPU-ULP-CAL — scaffold GPU-gen ULP calibration head (proposal) ([#238](https://github.com/lusoris/vmaf/issues/238)) ([fdb8a56](https://github.com/lusoris/vmaf/commit/fdb8a5652d2f42bcf7d243af19804f1d4eb81b12))
* **ai:** Tiny-AI 3-arch LOSO evaluation harness + Research-0023 ([#176](https://github.com/lusoris/vmaf/issues/176)) ([0e483a3](https://github.com/lusoris/vmaf/commit/0e483a3e16461727762c01b9f2a7b1bdf2db1d72))
* **ai:** Tiny-AI feature-set registry (Research-0026 Phase 1) ([#185](https://github.com/lusoris/vmaf/issues/185)) ([4336736](https://github.com/lusoris/vmaf/commit/4336736ae2e52ab6a12728f9708777886fe6a5dd))
* **ai:** Tiny-AI LOSO evaluation harness for mlp_small ([#165](https://github.com/lusoris/vmaf/issues/165)) ([427272b](https://github.com/lusoris/vmaf/commit/427272b48b18e0feb3d239112c24318e0187f34c))
* **ai:** Tiny-AI Phase-2 analysis scaffolding (Research-0026) ([#191](https://github.com/lusoris/vmaf/issues/191)) ([0df6543](https://github.com/lusoris/vmaf/commit/0df6543a8803017a67ae5c0454d6ea0c53093152))
* **ai:** Tiny-AI Quantization-Aware Training (QAT) implementation (T5-4) ([#179](https://github.com/lusoris/vmaf/issues/179)) ([d71ac52](https://github.com/lusoris/vmaf/commit/d71ac52cf5b995e21262a89f87d468fc7c28eade))
* **ai:** Tiny-AI training prep (loader + eval + Lightning harness for Netflix corpus) ([#158](https://github.com/lusoris/vmaf/issues/158)) ([aa74eaa](https://github.com/lusoris/vmaf/commit/aa74eaa3b5ac74263caceb053a218ba113a32345))
* **ai:** Vmaf_tiny v3 + v4 multi-seed Netflix LOSO + KoNViD 5-fold eval ([#311](https://github.com/lusoris/vmaf/issues/311)) ([be897e4](https://github.com/lusoris/vmaf/commit/be897e442f8225035d8b01e84220b030b45664f3))
* **ai:** Vmaf_tiny_v3 — mlp_medium arch trained on 4-corpus (PLCC=0.9986) ([#294](https://github.com/lusoris/vmaf/issues/294)) ([bc482ad](https://github.com/lusoris/vmaf/commit/bc482ad801bb078127ce3d047dc70dd582470ad7))
* **ai:** Vmaf_tiny_v4 — mlp_large arch trained on 4-corpus (PLCC=0.9987 vs v3's 0.9986, arch ladder saturates) ([#299](https://github.com/lusoris/vmaf/issues/299)) ([a1fd676](https://github.com/lusoris/vmaf/commit/a1fd676d235f73917fefefa9b940fb95785f9193))
* **ci:** Nightly bisect-model-quality + sticky tracker (closes [#4](https://github.com/lusoris/vmaf/issues/4)) ([#41](https://github.com/lusoris/vmaf/issues/41)) ([6cd4fb0](https://github.com/lusoris/vmaf/commit/6cd4fb01a81cff13996ad94cf552f7b61104cf60))
* DNN session runtime + SYCL list-devices + windows CI (1/5) ([#5](https://github.com/lusoris/vmaf/issues/5)) ([3cfa85b](https://github.com/lusoris/vmaf/commit/3cfa85bb4cad0b054df76c1e75c2a9761a4bdc56))
* **dnn:** Admit Loop + If on ONNX op-allowlist with recursive subgraph scan (T6-5, ADR-0169) ([#105](https://github.com/lusoris/vmaf/issues/105)) ([c4bd6ff](https://github.com/lusoris/vmaf/commit/c4bd6ff04fd03616cd24c7291d4be213a3638575))
* **dnn:** Allowlist Resize op (unblocks U-2-Net + saliency) ([#345](https://github.com/lusoris/vmaf/issues/345)) ([077d5c9](https://github.com/lusoris/vmaf/commit/077d5c980084d9b82b15bd5270544c3e5f4a934c))
* **dnn:** Bounded Loop.M trip-count guard (T6-5b, ADR-0171) ([#107](https://github.com/lusoris/vmaf/issues/107)) ([c264eec](https://github.com/lusoris/vmaf/commit/c264eec78d51e5a19cf35a59c76f3a7393c1e071))
* **dnn:** LPIPS-SqueezeNet FR extractor + ONNX + registry entry ([#23](https://github.com/lusoris/vmaf/issues/23)) ([0267dfb](https://github.com/lusoris/vmaf/commit/0267dfbe83b28366fcad56268d74d351f3b42db7))
* **dnn:** Scaffold tiny-AI training, ONNX export, and DNN C seam ([d122b72](https://github.com/lusoris/vmaf/commit/d122b72122e35a5e40399a0625ee07c227baf765))
* **tiny-ai:** C2 nr_metric_v1 + C3 learned_filter_v1 baselines on KoNViD-1k (T6-1, ADR-0168) ([#104](https://github.com/lusoris/vmaf/issues/104)) ([1a48eab](https://github.com/lusoris/vmaf/commit/1a48eab976f39a926dec5f01d94bc3c363755522))
* **tiny-ai:** First per-model PTQ — learned_filter_v1 dynamic int8 (T5-3b, ADR-0174) ([#110](https://github.com/lusoris/vmaf/issues/110)) ([07a3053](https://github.com/lusoris/vmaf/commit/07a30535c97314851849237a235981549d5af4cf))
* **tiny-ai:** PTQ int8 audit harness — registry + scripts + sidecar parser (T5-3, ADR-0173) ([#109](https://github.com/lusoris/vmaf/issues/109)) ([a7f2664](https://github.com/lusoris/vmaf/commit/a7f2664cb4948a985c5ec52e7b1cc508596f5804))
* **vmaf-tune:** FR-from-NR corpus adapter (ADR-0346) ([#536](https://github.com/lusoris/vmaf/issues/536)) ([f705cd1](https://github.com/lusoris/vmaf/commit/f705cd1b0ca23a50f7fbd121d902fa2559ed9ba8))


### Bug Fixes

* **ai:** Bump torch &gt;=2.8 + lightning &gt;=2.5 for CVE fixes ([#11](https://github.com/lusoris/vmaf/issues/11)) ([98da94d](https://github.com/lusoris/vmaf/commit/98da94d616433879c5d5d9326941fbcec3143a33))
* **ai:** Regenerate bisect cache with pinned pandas 2.3.3 ([#42](https://github.com/lusoris/vmaf/issues/42)) ([cdb9b9e](https://github.com/lusoris/vmaf/commit/cdb9b9ea01dd70682f7e9b6a7cdf16bfa2bbca9a))
* **ai:** Switch lightning → pytorch-lightning (PyPI 404) ([#232](https://github.com/lusoris/vmaf/issues/232)) ([c182dfe](https://github.com/lusoris/vmaf/commit/c182dfeb5672159685f03b4d80128daac859cd6a))
* **ai:** Unstick nightly bisect tracker on issue [#40](https://github.com/lusoris/vmaf/issues/40) (ADR-0253) ([#335](https://github.com/lusoris/vmaf/issues/335)) ([b78f58b](https://github.com/lusoris/vmaf/commit/b78f58b1676b88b8ed6b580420bb5beffb1f9393))
* SIMD bit-identical reductions + CI fixes ([#18](https://github.com/lusoris/vmaf/issues/18)) ([f082cfd](https://github.com/lusoris/vmaf/commit/f082cfd3a5eb471ca5b32e8f7ea32854c95ed152))


### Documentation

* **adr:** ADR-0349 — fr_regressor_v3 namespace + reserve _v3plus_features ([#550](https://github.com/lusoris/vmaf/issues/550)) ([650509a](https://github.com/lusoris/vmaf/commit/650509afbb26bd456cab23585e94f5909676dafa))
* **adr:** Dedup duplicate-NNNN ADRs (10 renumbered, keeps earliest at original) ([#310](https://github.com/lusoris/vmaf/issues/310)) ([af227b0](https://github.com/lusoris/vmaf/commit/af227b026d05d2b78cbf412595ef2eb9c64a493b))
* **ai:** Vmaf_tiny_v5 — YouTube UGC corpus-expansion probe (defer) ([#361](https://github.com/lusoris/vmaf/issues/361)) ([0adb595](https://github.com/lusoris/vmaf/commit/0adb5952cc169b5e236ddb0ab4fb711d75bacfa5))
* **research:** 0062 content-aware fr_regressor_v2 feasibility ([#395](https://github.com/lusoris/vmaf/issues/395)) ([c98dbac](https://github.com/lusoris/vmaf/commit/c98dbacb15565e29261e8f64edd056ffd97190fb))


### Build System

* CUDA 13 + oneAPI 2025.3 + clang-format 22 + black 26 (3/5) ([#7](https://github.com/lusoris/vmaf/issues/7)) ([a7be84c](https://github.com/lusoris/vmaf/commit/a7be84cb5cc6b80659bf2c799aaf62221b335dab))


### Miscellaneous

* **adr:** Migrate to Nygard one-file-per-decision + golusoris-alignment sweep ([#24](https://github.com/lusoris/vmaf/issues/24)) ([8e3cd22](https://github.com/lusoris/vmaf/commit/8e3cd22c1240ce9f11bb01f8bfd95a2230a598b1))
* **lint:** Clang-tidy upstream cleanup rounds 2-4 ([#2](https://github.com/lusoris/vmaf/issues/2)) ([722d21f](https://github.com/lusoris/vmaf/commit/722d21fd4e3c106abc411d584b4a84ce306d758e))
* Post-merge cleanup — CI fix + lint + supply-chain + scorecard + dependabot ([#14](https://github.com/lusoris/vmaf/issues/14)) ([798db39](https://github.com/lusoris/vmaf/commit/798db3941dea7757b764287f5fda784064430a96))
* **upstream:** Record ours-merge of Netflix 966be8d5 (already ported in d06dd6cf) ([fddc5ca](https://github.com/lusoris/vmaf/commit/fddc5ca7cbc0f406d0269c7c5ff98e0487d819b8))
* **upstream:** Record ours-merge of Netflix 966be8d5 (bookkeeping) ([27ce439](https://github.com/lusoris/vmaf/commit/27ce43910d5f6c14ec04342966bbc0204e5b2958))
