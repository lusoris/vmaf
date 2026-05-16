# AGENTS.md — dev/ (container infra)

Invariants for the `dev/` tree that agents must preserve across rebases and
follow-up PRs. See [Research-0135](../docs/research/0135-dev-mcp-container-stage-3-fix-2026-05-16.md)
for the diagnosis that established these.

## Containerfile invariants

### USER ordering and directory ownership (stage 3)

`WORKDIR` always creates directories as **root**, regardless of any previous
`USER` directive. After `COPY --chown=vmaf:vmaf . /dest/`, only the
*contents* are owned by `vmaf`; the destination directory itself is still
owned by root.

**Rule**: Any `RUN` step that executes as a non-root user and needs to
create a subdirectory inside a `WORKDIR`-created path must be preceded by:

```dockerfile
RUN chown <user>:<group> /parent /parent/dest
USER <user>
```

Do NOT rely on `COPY --chown` alone to make a directory writable — it does
not change the directory entry's owner, only file/subdirectory contents.

Violating this causes `meson setup build` (and any other tool that calls
`os.makedirs`) to fail with `PermissionError: [Errno 13] Permission denied`
at exactly the build-dir creation step, exit code 13.

### CUDA package names

- Use `cuda-toolkit` (the current unversioned meta-package).
- Do NOT install `libcuda1` (the runtime driver) — it must come from
  `nvidia-container-runtime` at run-time; baking it in shadows the host driver.
- Do NOT install `cuda-compiler` — this is a legacy alias that no longer
  exists in the NVIDIA CUDA channels; `cuda-toolkit` already provides `nvcc`.

### Intel oneAPI package name

- Use `intel-basekit` (unversioned meta-package).
- Do NOT use `intel-basekit-<year>.<quarter>` (e.g., `intel-basekit-2025.3`):
  Intel does not publish year-quarter-versioned meta-package names in
  `apt.repos.intel.com/oneapi`. Using the versioned name causes
  `E: Unable to locate package`.

### ROCm / HIP package names

- Use `rocm-hip-runtime-dev` (not `rocm-hip-sdk`).
- `rocm-hip-sdk` transitively installs `rccl` (multi-GPU collectives) which
  depends on `libdrm-amdgpu-amdgpu1` + `libdrm2-amdgpu` — packages absent
  from the ROCm 6.4 noble apt repo. libvmaf HIP kernels use one GPU per
  worker; rccl is not needed.

### SHELL / hadolint DL4006

- `SHELL ["/bin/bash", "-o", "pipefail", "-c"]` is set in the `gpu-sdks`
  stage and **inherited** by `libvmaf-build` and `dev-mcp`.
- hadolint does not track cross-stage SHELL inheritance. Any `RUN` step in
  a derived stage that contains a pipe will trigger DL4006 as a false positive.
  Suppress with `# hadolint ignore=DL4006` and note that SHELL is inherited.
