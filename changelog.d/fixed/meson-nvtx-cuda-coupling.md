## Fixed

- **`enable_nvtx=true` + `enable_cuda=false` now fails with a clear error**
  instead of a cryptic `Include dir /usr/local/cuda/include does not exist`
  message. NVTX range markers require the CUDA toolkit headers; the combination
  is now rejected at `meson setup` time with an actionable message.
  ([libvmaf/src/meson.build](../../libvmaf/src/meson.build))

- **Hardcoded `/usr/local/cuda/include` replaced with a derived path.**
  The CUDA toolkit include directory is now derived from the `nvcc` binary
  location (`<nvcc-dir>/../../include`), with an env-var fallback
  (`$CUDA_HOME` / `$CUDA_PATH`). This fixes `meson setup` for hosts with
  CUDA installed at `/opt/cuda` (the path documented in `CLAUDE.md §2`) or
  any other non-default prefix.
  ([libvmaf/src/meson.build](../../libvmaf/src/meson.build))
