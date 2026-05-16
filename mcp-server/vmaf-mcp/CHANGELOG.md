# Changelog

## [0.1.1](https://github.com/lusoris/vmaf/compare/v0.1.0...v0.1.1) (2026-05-16)


### Bug Fixes

* **mcp:** Add vulkan/hip/metal to backend enum + dispatch parity (ADR-0436) ([#848](https://github.com/lusoris/vmaf/issues/848)) ([d344994](https://github.com/lusoris/vmaf/commit/d34499458e3df4881b0bb116ad6e8de3ddc00202))
* **mcp:** Correct CLI flags, backend dispatch, and entry-point alias ([#991](https://github.com/lusoris/vmaf/issues/991)) ([48dcc7d](https://github.com/lusoris/vmaf/commit/48dcc7ddb951fd432b552f16440f9cec82d2984d))
* **mcp:** Purge stale PNGs at start of each describe_worst_frames call ([ab27ea0](https://github.com/lusoris/vmaf/commit/ab27ea043287d15b13cffcea0642906dd23933ad))
* **mcp:** Refresh stale docs and ssimulacra2 snapshot ([#786](https://github.com/lusoris/vmaf/issues/786)) ([dd67de9](https://github.com/lusoris/vmaf/commit/dd67de9bf4dc13c4e02d42eb51320453b323d090))


### Miscellaneous

* **deps:** Pin dependencies ([#614](https://github.com/lusoris/vmaf/issues/614)) ([02b021f](https://github.com/lusoris/vmaf/commit/02b021f2b471d749d1754be21d2a0e1cccc68c69))
* **deps:** Update dependency accelerate to &gt;=0.34.2 ([#615](https://github.com/lusoris/vmaf/issues/615)) ([585b937](https://github.com/lusoris/vmaf/commit/585b9375844a542780a0c3e9949442bf79efcc76))
* **deps:** Update dependency accelerate to v1 ([#649](https://github.com/lusoris/vmaf/issues/649)) ([694c20c](https://github.com/lusoris/vmaf/commit/694c20c435ebf64df1e96e42acb5a72b2e3fb56c))
* **deps:** Update dependency anyio to &gt;=4.13.0 ([#625](https://github.com/lusoris/vmaf/issues/625)) ([a244436](https://github.com/lusoris/vmaf/commit/a244436eaf4d694bc51e3bcfff9515fa9dd69e7a))
* **deps:** Update dependency mcp to &gt;=1.27.0 [SECURITY] ([#589](https://github.com/lusoris/vmaf/issues/589)) ([79ec3a8](https://github.com/lusoris/vmaf/commit/79ec3a8383a03f7ed7d928db966c2b9fd54f0609))
* **deps:** Update dependency mcp to &gt;=1.27.1 ([#616](https://github.com/lusoris/vmaf/issues/616)) ([79fbef0](https://github.com/lusoris/vmaf/commit/79fbef0df8e60063421eb91faaf53d7c6889da8d))
* **deps:** Update dependency mypy to &gt;=1.20.2 ([#629](https://github.com/lusoris/vmaf/issues/629)) ([8a44127](https://github.com/lusoris/vmaf/commit/8a441278ef3344ab1cf19b19d582ff8d781d06a1))
* **deps:** Update dependency mypy to v2 ([#651](https://github.com/lusoris/vmaf/issues/651)) ([2246dcc](https://github.com/lusoris/vmaf/commit/2246dcc1588156143a984aef702cf54598052107))
* **deps:** Update dependency numpy to v1.26.4 ([#617](https://github.com/lusoris/vmaf/issues/617)) ([e60f63c](https://github.com/lusoris/vmaf/commit/e60f63c9b401ea6197dc0f4f9420249f404dd133))
* **deps:** Update dependency numpy to v2 ([#652](https://github.com/lusoris/vmaf/issues/652)) ([8eefdf2](https://github.com/lusoris/vmaf/commit/8eefdf27b2b338256ba6cda2050b0df6106041af))
* **deps:** Update dependency onnxruntime to v1.25.1 ([#630](https://github.com/lusoris/vmaf/issues/630)) ([551fa11](https://github.com/lusoris/vmaf/commit/551fa11cb2cd00bd15b7275d362b2663d4744254))
* **deps:** Update dependency pandas to v2.3.3 ([#633](https://github.com/lusoris/vmaf/issues/633)) ([afe2c8f](https://github.com/lusoris/vmaf/commit/afe2c8f5d3659f1d401a4da1727863913bd01939))
* **deps:** Update dependency pandas to v3 ([#655](https://github.com/lusoris/vmaf/issues/655)) ([39da276](https://github.com/lusoris/vmaf/commit/39da27619399ee167f1528c92442922953708b62))
* **deps:** Update dependency pillow to v12 [SECURITY] ([#601](https://github.com/lusoris/vmaf/issues/601)) ([46308ab](https://github.com/lusoris/vmaf/commit/46308abcda6462385b3c6c7bb743ec36c4d0c616))
* **deps:** Update dependency pyarrow to &gt;=15.0.2 [SECURITY] ([#588](https://github.com/lusoris/vmaf/issues/588)) ([7c79173](https://github.com/lusoris/vmaf/commit/7c7917347799c371a55cfef50920ac846e01cb0e))
* **deps:** Update dependency pyarrow to v24 [SECURITY] ([#604](https://github.com/lusoris/vmaf/issues/604)) ([4acae78](https://github.com/lusoris/vmaf/commit/4acae78a86f3f67a19d1e8f2b330d9a2eda6d83e))
* **deps:** Update dependency pydantic to &gt;=2.13.4 ([#634](https://github.com/lusoris/vmaf/issues/634)) ([c2dd644](https://github.com/lusoris/vmaf/commit/c2dd6449c821d571352d55f21522baaf776b5265))
* **deps:** Update dependency pytest to &gt;=8.4.2 [SECURITY] ([#594](https://github.com/lusoris/vmaf/issues/594)) ([7e261b4](https://github.com/lusoris/vmaf/commit/7e261b4e4224c8d12aac8a4bfcaa55f5bbcbe1fd))
* **deps:** Update dependency pytest to v9 [SECURITY] ([#605](https://github.com/lusoris/vmaf/issues/605)) ([f842c62](https://github.com/lusoris/vmaf/commit/f842c62914422b251a15e8d2794fe953ed5d60a3))
* **deps:** Update dependency pytest-asyncio to &gt;=0.26.0 ([#635](https://github.com/lusoris/vmaf/issues/635)) ([a6c3836](https://github.com/lusoris/vmaf/commit/a6c383643cbcdb887528e856d7b25550d51f04c3))
* **deps:** Update dependency pytest-asyncio to v1 ([#658](https://github.com/lusoris/vmaf/issues/658)) ([00ac0b0](https://github.com/lusoris/vmaf/commit/00ac0b07e356e14f75c7c08778d3d630a174aefd))
* **deps:** Update dependency ruff to &gt;=0.15.12 ([#638](https://github.com/lusoris/vmaf/issues/638)) ([76ed58e](https://github.com/lusoris/vmaf/commit/76ed58e6075fe93fd17ac0bf5dabd401a88e3191))
* **deps:** Update dependency scipy to &gt;=1.17.1 ([#641](https://github.com/lusoris/vmaf/issues/641)) ([9a63119](https://github.com/lusoris/vmaf/commit/9a6311937d1e624c7aaabde62bd7efdfdf32f307))
* **deps:** Update dependency torch to &gt;=2.11.0 [SECURITY] ([#598](https://github.com/lusoris/vmaf/issues/598)) ([467a1b1](https://github.com/lusoris/vmaf/commit/467a1b1e256b2c9893f2c1da5ea3b034ff417e6a))
* **deps:** Update dependency transformers to &gt;=4.57.6 [SECURITY] ([#600](https://github.com/lusoris/vmaf/issues/600)) ([d574916](https://github.com/lusoris/vmaf/commit/d57491639516620391b21111b740f0c379b37bff))
* **deps:** Update dependency transformers to v5 [SECURITY] ([#610](https://github.com/lusoris/vmaf/issues/610)) ([bed072f](https://github.com/lusoris/vmaf/commit/bed072f01d359411c939a0fdb01007101d70f5b2))
* **deps:** Update docker/dockerfile Docker tag to v1.23 ([#645](https://github.com/lusoris/vmaf/issues/645)) ([5e37a71](https://github.com/lusoris/vmaf/commit/5e37a715976706996fdc95b988fffcd1dd4bf1ea))
* **deps:** Update ubuntu Docker tag to v26 ([#667](https://github.com/lusoris/vmaf/issues/667)) ([42e429c](https://github.com/lusoris/vmaf/commit/42e429c563689cf5a376957ce0a1e1a9cecd013f))
