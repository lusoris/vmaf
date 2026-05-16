# Changelog

## [0.2.0](https://github.com/lusoris/vmaf/compare/v0.1.0...v0.2.0) (2026-05-16)


### Features

* **ai:** Add chug hdr audit splits ([#832](https://github.com/lusoris/vmaf/issues/832)) ([07f1f59](https://github.com/lusoris/vmaf/commit/07f1f59ef3eb68f6a51c8076bf4e175136cc327e))
* **ai:** Add chug hdr corpus ingestion ([#802](https://github.com/lusoris/vmaf/issues/802)) ([0e5de9c](https://github.com/lusoris/vmaf/commit/0e5de9ca419705ec4dc1f089604332ca523d39b7))
* **ai:** Add CHUG parquet metadata enrichment ([#834](https://github.com/lusoris/vmaf/issues/834)) ([925ce1a](https://github.com/lusoris/vmaf/commit/925ce1a19b812ed89ac0e23392fa69cf3249bf8b))
* **ai:** Add saliency block evaluation ([#836](https://github.com/lusoris/vmaf/issues/836)) ([404a574](https://github.com/lusoris/vmaf/commit/404a57407f2391c98e6fd6d3a8469fba9300a112))
* **ai:** Add speed_temporal + speed_chroma to K150K/CHUG extraction (lawrence's recipe) ([#855](https://github.com/lusoris/vmaf/issues/855)) ([d306b71](https://github.com/lusoris/vmaf/commit/d306b71fd32503161760d5dd87fc17439e02f9fa))
* **ai:** Add vmaf train tune cli ([#810](https://github.com/lusoris/vmaf/issues/810)) ([ed19d8d](https://github.com/lusoris/vmaf/commit/ed19d8d883d6b5d8cbefe2d41bd8315e7d3fb34a))
* **ai:** Compute vmaf column via vmaf_v0.6.1 SDR baseline in CHUG extraction ([#909](https://github.com/lusoris/vmaf/issues/909)) ([eb9b3c9](https://github.com/lusoris/vmaf/commit/eb9b3c9d6a619488ccb3d155b0f8641bd620dd05))
* **ai:** Fr_regressor_v2 ensemble — full production flip (real ONNX + sidecars, ADR-0321) ([#424](https://github.com/lusoris/vmaf/issues/424)) ([13174fc](https://github.com/lusoris/vmaf/commit/13174fcd4652fad04502d633e0bdfa89e091f901))
* **ai:** Fr_regressor_v2 ensemble LOSO trainer — real corpus loader + per-fold training (ADR-0319) ([#422](https://github.com/lusoris/vmaf/issues/422)) ([882ecbf](https://github.com/lusoris/vmaf/commit/882ecbff21a84978e31dc6fefbb6493e96c6c5a3))
* **ai:** Fr_regressor_v2 ensemble seeds — flip smoke→prod (ADR-0320) ([#674](https://github.com/lusoris/vmaf/issues/674)) ([a00606c](https://github.com/lusoris/vmaf/commit/a00606cf0c760897fcd2f05a80a1522217eacd79))
* **ai:** Fr_regressor_v3 — train + register on ENCODER_VOCAB v3 (16-slot) ([#428](https://github.com/lusoris/vmaf/issues/428)) ([1447e92](https://github.com/lusoris/vmaf/commit/1447e92a271569195543ea1291c096299dad994f))
* **ai:** Hardware-capability priors for FR-regressor corpus (ADR-0335) ([#482](https://github.com/lusoris/vmaf/issues/482)) ([da6df89](https://github.com/lusoris/vmaf/commit/da6df893e1c97d2a6fb77e3510bb2159c4038974))
* **ai:** Ingest konvid 150k split score layout ([#784](https://github.com/lusoris/vmaf/issues/784)) ([5964909](https://github.com/lusoris/vmaf/commit/5964909cd23a511528808a4871c4ad355ed19a7f))
* **ai:** K150K-A FULL_FEATURES extraction pipeline (ADR-0362) ([#572](https://github.com/lusoris/vmaf/issues/572)) ([75e379e](https://github.com/lusoris/vmaf/commit/75e379eb75b43b852e363985031a2903e0dc8952))
* **ai:** KonViD MOS head v1 (ADR-0336, ADR-0325 Phase 3) ([#491](https://github.com/lusoris/vmaf/issues/491)) ([553806e](https://github.com/lusoris/vmaf/commit/553806e6af91ad4af8ea65897038eda8aecab9fa))
* **ai:** KonViD-150k MOS-corpus JSONL ingestion (ADR-0325 Phase 2) ([#447](https://github.com/lusoris/vmaf/issues/447)) ([d88a16d](https://github.com/lusoris/vmaf/commit/d88a16ddf302c10a44d913af30b561f29f51a424))
* **ai:** KonViD-1k MOS-corpus JSONL ingestion (ADR-0325 Phase 1) ([#440](https://github.com/lusoris/vmaf/issues/440)) ([a83046e](https://github.com/lusoris/vmaf/commit/a83046e09a2ff660140beb734ae3635fabf0cf7b))
* **ai:** LIVE-VQC MOS-corpus JSONL ingestion (ADR-0370) ([#586](https://github.com/lusoris/vmaf/issues/586)) ([7c3e4dd](https://github.com/lusoris/vmaf/commit/7c3e4ddc700200de39e341f3b705976240a59024))
* **ai:** LSVQ MOS-corpus JSONL ingestion (ADR-0333) ([#471](https://github.com/lusoris/vmaf/issues/471)) ([c4cc498](https://github.com/lusoris/vmaf/commit/c4cc49824a645d3eae32c139539a327e11c60d0b))
* **ai:** Materialise bisect cache from real features ([#798](https://github.com/lusoris/vmaf/issues/798)) ([f3a924c](https://github.com/lusoris/vmaf/commit/f3a924c6b3618d2e6dd8e0b6eb37fec0b7721f35))
* **ai:** Multi-corpus MOS aggregation for v2 trainer (ADR-0340) ([#518](https://github.com/lusoris/vmaf/issues/518)) ([041eb47](https://github.com/lusoris/vmaf/commit/041eb47dc7998381fcbada809c8bcae51429e8df))
* **ai:** Predictor v2 real-corpus LOSO trainer + ADR-0303 gate (Phase 2) ([#487](https://github.com/lusoris/vmaf/issues/487)) ([2caec7d](https://github.com/lusoris/vmaf/commit/2caec7d769940f3929406bdf88909a82d94774bd))
* **ai:** Saliency_student_v2 — Resize-decoder ablation on v1 recipe ([#515](https://github.com/lusoris/vmaf/issues/515)) ([9bda0b7](https://github.com/lusoris/vmaf/commit/9bda0b758a021f1427843fd1bb9e279ac2b6a8ad))
* **ai:** U2netp fork-local release-artefact mirror scaffold (ADR-0325) ([#469](https://github.com/lusoris/vmaf/issues/469)) ([b87ce48](https://github.com/lusoris/vmaf/commit/b87ce487a81887bc1c283b4b46c60e0fc8f5b32f))
* **ai:** Waterloo IVC 4K-VQA MOS-corpus JSONL ingestion (ADR-0334) ([#485](https://github.com/lusoris/vmaf/issues/485)) ([96d0007](https://github.com/lusoris/vmaf/commit/96d00075b11b07ff3553d4ca60f1f7976d3b0d4f))
* **ai:** YouTube UGC MOS-corpus JSONL ingestion (ADR-0334) ([#481](https://github.com/lusoris/vmaf/issues/481)) ([e3bec36](https://github.com/lusoris/vmaf/commit/e3bec36ff8ba93b3097dda054fe42e6916e3636e))
* **vmaf-tune:** Auto Phase F.5 calibrated recipe overrides (ADR-0325) ([#514](https://github.com/lusoris/vmaf/issues/514)) ([d4bc607](https://github.com/lusoris/vmaf/commit/d4bc607257879febd7f7193094a91cd206861bcb))
* **vmaf-tune:** Corpus schema v3 — canonical-6 per-feature aggregates ([#462](https://github.com/lusoris/vmaf/issues/462)) ([1358cc1](https://github.com/lusoris/vmaf/commit/1358cc128f2f75ba6f8d315449bb36bb83982274))
* **vmaf-tune:** FR-from-NR corpus adapter (ADR-0346) ([#536](https://github.com/lusoris/vmaf/issues/536)) ([f705cd1](https://github.com/lusoris/vmaf/commit/f705cd1b0ca23a50f7fbd121d902fa2559ed9ba8))


### Bug Fixes

* **ai/tests:** Update feature-count assertions after PR [#855](https://github.com/lusoris/vmaf/issues/855) added 4 speed features ([#1047](https://github.com/lusoris/vmaf/issues/1047)) ([eb86986](https://github.com/lusoris/vmaf/commit/eb869864242c6ee18ae1f8a0584d1e24cb3974c8))
* **ai:** Correct isort import order in train_konvid.py ([#713](https://github.com/lusoris/vmaf/issues/713)) ([a7c4161](https://github.com/lusoris/vmaf/commit/a7c4161f0f9b8fed5dffa786032f75bcfd045f49))
* **ai:** Drop stale integer_psnr_y/cb/cr second-candidate aliases ([#1061](https://github.com/lusoris/vmaf/issues/1061)) ([11c4c67](https://github.com/lusoris/vmaf/commit/11c4c670ba40abab1208025093f89fbc343beb30))
* **ai:** K150K/CHUG extractor passes HDR + HFR per-feature options (closes [#837](https://github.com/lusoris/vmaf/issues/837)) ([#851](https://github.com/lusoris/vmaf/issues/851)) ([f8e7dce](https://github.com/lusoris/vmaf/commit/f8e7dce1b4ac6e57db42af4ee651423fd533272e))
* **ai:** Keep "does not ship" on one line in corpus adapter docstrings ([#753](https://github.com/lusoris/vmaf/issues/753)) ([7993013](https://github.com/lusoris/vmaf/commit/7993013ef372ca8663522a5222e76a5cd08dd60a))
* **ai:** Load packed color frames ([#797](https://github.com/lusoris/vmaf/issues/797)) ([dc33912](https://github.com/lusoris/vmaf/commit/dc339127fcab004153e7fdf7929d7b08289fc121))
* **ai:** Promote cambi_cuda + float_ssim_cuda from CPU residual to CUDA primary pass ([#854](https://github.com/lusoris/vmaf/issues/854)) ([cf53499](https://github.com/lusoris/vmaf/commit/cf53499106f5c6de2be264692beeafc60f058b20))
* **ai:** Re-export save_progress + dataset hints in 5 corpus adapters ([#717](https://github.com/lusoris/vmaf/issues/717)) ([be67d19](https://github.com/lusoris/vmaf/commit/be67d19bf7b72bd7e6918c3c9a63222b6d623f4c))
* **ai:** Resolve 633/634 mypy real-bug errors in ai/ and scripts/ ([#697](https://github.com/lusoris/vmaf/issues/697)) ([0905e42](https://github.com/lusoris/vmaf/commit/0905e421c344f35afece5fa34b46334f263453d1))
* **ai:** Run_ensemble_v2_real_corpus_loso.sh — wrapper-trainer interface mismatch + Phase-A pre-step doc (ADR-0318) ([#673](https://github.com/lusoris/vmaf/issues/673)) ([628821b](https://github.com/lusoris/vmaf/commit/628821b6eec5cbbecd3c5f8181a9e771d347c105))
* **ai:** Split fr-from-nr cuda feature extraction ([#815](https://github.com/lusoris/vmaf/issues/815)) ([9aae789](https://github.com/lusoris/vmaf/commit/9aae789ad7c5245fbb2511c74b46e88f7f1e4e12))
* **ci:** Cambi stddef + aiutils _mean fallback — master CI unblock bundle ([#914](https://github.com/lusoris/vmaf/issues/914)) ([b8ed583](https://github.com/lusoris/vmaf/commit/b8ed5834978503e208831f8a521302b1c5bc1cae))
* **ci:** Master CI triple unblock bundle (SYCL + tests + HIP) ([#1052](https://github.com/lusoris/vmaf/issues/1052)) ([8fd940a](https://github.com/lusoris/vmaf/commit/8fd940a6f8807b9a3990da5aa810cf8eb6e6e92c))
* **libvmaf:** Replace assert(0) + log silent EINVAL + HIP init parity + copyright headers ([#843](https://github.com/lusoris/vmaf/issues/843)) ([d85f913](https://github.com/lusoris/vmaf/commit/d85f91366d2b68af0df353c25943b8ea34eab4a4))
* **test:** _probe_geometry mock returns 5-tuple (color_meta added) ([#1053](https://github.com/lusoris/vmaf/issues/1053)) ([c626848](https://github.com/lusoris/vmaf/commit/c626848bbab8decc232fa2f1d0b4e88bfc85c9f7))
* **tune:** Retire auto HDR and AI scaffold leftovers ([#774](https://github.com/lusoris/vmaf/issues/774)) ([d924ba8](https://github.com/lusoris/vmaf/commit/d924ba8dd060d0cdd5cc095cb4066553648b51fa))
* **vif:** Full upstream Netflix sync — on-the-fly filter + test recalibrations (ADR-0416) ([#758](https://github.com/lusoris/vmaf/issues/758)) ([97ff046](https://github.com/lusoris/vmaf/commit/97ff04627cefbfd443de0fe7a5996a748e9c1409))


### Performance

* **ai:** Drop ssimulacra2 from K150K/CHUG self-vs-self CUDA extraction ([#897](https://github.com/lusoris/vmaf/issues/897)) ([9587f85](https://github.com/lusoris/vmaf/commit/9587f8590739f0cb8b92b52ee3e089e1717450c3))
* **ai:** Eliminate quadratic parquet I/O and ffprobe per-clip in CHUG extractor ([#900](https://github.com/lusoris/vmaf/issues/900)) ([8d57335](https://github.com/lusoris/vmaf/commit/8d573354efd5a918d78b6e1a73e31547a054f162))
* **ai:** K150K driver — parallel CPU workers, 4× speedup, CUDA dedup fix (ADR-0382) ([#739](https://github.com/lusoris/vmaf/issues/739)) ([59b40ed](https://github.com/lusoris/vmaf/commit/59b40ede4095d41f6184bee1ebf5a9a68baee7e3))


### Refactors

* **ai:** Extract aiutils/ shared helpers (sha256, time, jsonl, parquet) ([#908](https://github.com/lusoris/vmaf/issues/908)) ([a3123a8](https://github.com/lusoris/vmaf/commit/a3123a8dc2b47957b2662d5819e2a181c29ac80d))
* **ai:** Shared CorpusIngestBase for 7 ingestion scripts ([#664](https://github.com/lusoris/vmaf/issues/664)) ([f67d197](https://github.com/lusoris/vmaf/commit/f67d1979f736ce47018b1336f36bcd7cee94d856))


### Documentation

* **adr:** ADR-0349 — fr_regressor_v3 namespace + reserve _v3plus_features ([#550](https://github.com/lusoris/vmaf/issues/550)) ([650509a](https://github.com/lusoris/vmaf/commit/650509afbb26bd456cab23585e94f5909676dafa))


### Build System

* 644/644 ninja targets, 56/56 meson tests pass. ([d85f913](https://github.com/lusoris/vmaf/commit/d85f91366d2b68af0df353c25943b8ea34eab4a4))


### Miscellaneous

* **ai:** Update corpus path references from .workingdir2/ to .corpus/ in scripts ([#1009](https://github.com/lusoris/vmaf/issues/1009)) ([f12d741](https://github.com/lusoris/vmaf/commit/f12d741b08c3c249be7d63d9243392c65706d7e0))
* **deps:** Update dependency matplotlib to &gt;=3.10.9 ([#628](https://github.com/lusoris/vmaf/issues/628)) ([c75dd21](https://github.com/lusoris/vmaf/commit/c75dd219e83e48f157191c02f47146c61969da97))
* **deps:** Update dependency mypy to &gt;=1.20.2 ([#629](https://github.com/lusoris/vmaf/issues/629)) ([8a44127](https://github.com/lusoris/vmaf/commit/8a441278ef3344ab1cf19b19d582ff8d781d06a1))
* **deps:** Update dependency mypy to v2 ([#651](https://github.com/lusoris/vmaf/issues/651)) ([2246dcc](https://github.com/lusoris/vmaf/commit/2246dcc1588156143a984aef702cf54598052107))
* **deps:** Update dependency numpy to v1.26.4 ([#617](https://github.com/lusoris/vmaf/issues/617)) ([e60f63c](https://github.com/lusoris/vmaf/commit/e60f63c9b401ea6197dc0f4f9420249f404dd133))
* **deps:** Update dependency numpy to v2 ([#652](https://github.com/lusoris/vmaf/issues/652)) ([8eefdf2](https://github.com/lusoris/vmaf/commit/8eefdf27b2b338256ba6cda2050b0df6106041af))
* **deps:** Update dependency onnx to &gt;=1.21.0,&lt;2.0 [SECURITY] ([#592](https://github.com/lusoris/vmaf/issues/592)) ([fb14b45](https://github.com/lusoris/vmaf/commit/fb14b457dd3edbe8b85da71a163da3037cf5dab4))
* **deps:** Update dependency onnxruntime to v1.25.1 ([#630](https://github.com/lusoris/vmaf/issues/630)) ([551fa11](https://github.com/lusoris/vmaf/commit/551fa11cb2cd00bd15b7275d362b2663d4744254))
* **deps:** Update dependency onnxscript to &gt;=0.7.0 ([#631](https://github.com/lusoris/vmaf/issues/631)) ([043ce96](https://github.com/lusoris/vmaf/commit/043ce969133ec4595e2e14d4f5fde6e43f4fe837))
* **deps:** Update dependency optuna to v4.8.0 ([#654](https://github.com/lusoris/vmaf/issues/654)) ([db97237](https://github.com/lusoris/vmaf/commit/db972379d7d4ab85246685a8158687d30998caa7))
* **deps:** Update dependency pandas to v2.3.3 ([#633](https://github.com/lusoris/vmaf/issues/633)) ([afe2c8f](https://github.com/lusoris/vmaf/commit/afe2c8f5d3659f1d401a4da1727863913bd01939))
* **deps:** Update dependency pandas to v3 ([#655](https://github.com/lusoris/vmaf/issues/655)) ([39da276](https://github.com/lusoris/vmaf/commit/39da27619399ee167f1528c92442922953708b62))
* **deps:** Update dependency pillow to v12 [SECURITY] ([#601](https://github.com/lusoris/vmaf/issues/601)) ([46308ab](https://github.com/lusoris/vmaf/commit/46308abcda6462385b3c6c7bb743ec36c4d0c616))
* **deps:** Update dependency pyarrow to v24 ([#656](https://github.com/lusoris/vmaf/issues/656)) ([1ed36fa](https://github.com/lusoris/vmaf/commit/1ed36fa30fa99dab8d9ecf0c972cd55d54dbbe8a))
* **deps:** Update dependency pytest to &gt;=8.4.2 [SECURITY] ([#594](https://github.com/lusoris/vmaf/issues/594)) ([7e261b4](https://github.com/lusoris/vmaf/commit/7e261b4e4224c8d12aac8a4bfcaa55f5bbcbe1fd))
* **deps:** Update dependency pytest to v9 [SECURITY] ([#605](https://github.com/lusoris/vmaf/issues/605)) ([f842c62](https://github.com/lusoris/vmaf/commit/f842c62914422b251a15e8d2794fe953ed5d60a3))
* **deps:** Update dependency pytorch-lightning to &gt;=2.6.1,&lt;3.0 ([#636](https://github.com/lusoris/vmaf/issues/636)) ([18b0535](https://github.com/lusoris/vmaf/commit/18b05355458f1f390b1456e6945af4452df938f5))
* **deps:** Update dependency pyyaml to &gt;=6.0.3 ([#619](https://github.com/lusoris/vmaf/issues/619)) ([8f05872](https://github.com/lusoris/vmaf/commit/8f05872e48824468075e5bae4fd7e2ea7ab30347))
* **deps:** Update dependency ray to &gt;=2.55.1 [SECURITY] ([#595](https://github.com/lusoris/vmaf/issues/595)) ([1f14522](https://github.com/lusoris/vmaf/commit/1f14522e597bf4095ea721bbc1c948fdfa589d9a))
* **deps:** Update dependency rich to &gt;=13.9.4 ([#637](https://github.com/lusoris/vmaf/issues/637)) ([8044c01](https://github.com/lusoris/vmaf/commit/8044c01602711454115ec1ce4964891295eb3b99))
* **deps:** Update dependency rich to v15 ([#660](https://github.com/lusoris/vmaf/issues/660)) ([dfe02c5](https://github.com/lusoris/vmaf/commit/dfe02c5e95ad87b39da185002fb0f49bf5d67f2c))
* **deps:** Update dependency ruff to &gt;=0.15.12 ([#638](https://github.com/lusoris/vmaf/issues/638)) ([76ed58e](https://github.com/lusoris/vmaf/commit/76ed58e6075fe93fd17ac0bf5dabd401a88e3191))
* **deps:** Update dependency scikit-learn to &gt;=1.8.0 ([#640](https://github.com/lusoris/vmaf/issues/640)) ([7b1aef9](https://github.com/lusoris/vmaf/commit/7b1aef9e2505976be341ffae3e4bbffc8ee97724))
* **deps:** Update dependency scipy to &gt;=1.17.1 ([#641](https://github.com/lusoris/vmaf/issues/641)) ([9a63119](https://github.com/lusoris/vmaf/commit/9a6311937d1e624c7aaabde62bd7efdfdf32f307))
* **deps:** Update dependency seaborn to &gt;=0.13.2 ([#620](https://github.com/lusoris/vmaf/issues/620)) ([bb53fe4](https://github.com/lusoris/vmaf/commit/bb53fe490c4c8d74d05e490629199173593183d6))
* **deps:** Update dependency torch to &gt;=2.11.0,&lt;3.0 ([#643](https://github.com/lusoris/vmaf/issues/643)) ([ff132cf](https://github.com/lusoris/vmaf/commit/ff132cff889a0202880afe39309f397599d2d1bd))
* **deps:** Update dependency tqdm to &gt;=4.67.3 [SECURITY] ([#599](https://github.com/lusoris/vmaf/issues/599)) ([e1c29af](https://github.com/lusoris/vmaf/commit/e1c29aff1d2d34dbf5785b8784b11ac122824683))
* **deps:** Update dependency typer to &gt;=0.25.1 ([#644](https://github.com/lusoris/vmaf/issues/644)) ([5dd60b2](https://github.com/lusoris/vmaf/commit/5dd60b2930dba8768719e5d2b613d6549f5b436c))
