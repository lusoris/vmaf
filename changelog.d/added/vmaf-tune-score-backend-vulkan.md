- **vmaf-tune `--score-backend=vulkan`** — vendor-neutral GPU
  scoring path on top of libvmaf's existing Vulkan backend
  (ADR-0127 / ADR-0175 / ADR-0186). Restores the `--score-backend`
  argparse wiring that was lost between PR #378 and HEAD (post-#378
  rebases dropped the `cli.py` hunks) and admits `vulkan` as a
  fourth strict-mode value alongside `cpu` / `cuda` / `sycl`. Runs
  on Mesa anv/RADV/lavapipe (Linux), the proprietary NVIDIA driver,
  and macOS via MoltenVK — the obvious answer for AMD, Intel Arc,
  and any contributor box without `nvidia-smi`. `auto` mode still
  walks `cuda → vulkan → sycl → cpu`, so existing NVIDIA
  configurations see no behaviour change. Strict-mode failures
  (e.g. `--score-backend vulkan` on a host without a Vulkan loader)
  raise `BackendUnavailableError` and exit 2 — no silent CPU
  downgrade. ADR-0314.
