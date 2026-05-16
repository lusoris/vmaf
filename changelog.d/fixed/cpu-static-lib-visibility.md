Apply `-fvisibility=hidden` to `libvmaf_cpu_static_lib` in `libvmaf/src/meson.build`
so the internal CPU-dispatch helpers (`vmaf_get_cpu_flags`, `vmaf_init_cpu`,
`vmaf_set_cpu_flags_mask`, `vmaf_get_cpu_flags_x86`) are no longer exported as T
symbols in the final `libvmaf.so`.  Previously these functions leaked into the
dynamic-symbol table despite having no `VMAF_EXPORT` annotation and no declaration
under `include/libvmaf/`, making them de-facto ABI.
(Build-matrix audit §2b, 2026-05-16.)
