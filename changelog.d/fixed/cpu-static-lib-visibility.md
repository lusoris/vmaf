- **`libvmaf_cpu_static_lib` cpu-probe helpers no longer leak as exported
  symbols in `libvmaf.so`.** Four internal symbols —
  `vmaf_get_cpu_flags`, `vmaf_get_cpu_flags_x86`, `vmaf_init_cpu`, and
  `vmaf_set_cpu_flags_mask` — were compiled without `-fvisibility=hidden`
  because `libvmaf_cpu_static_lib` in `libvmaf/src/meson.build` was missing
  `c_args : vmaf_cflags_common`. All four appeared as default-visibility `T`
  entries in the DSO dynamic symbol table (audit finding 2b,
  `audit-build-matrix-symbols-2026-05-16`).  This constituted a de-facto ABI
  despite the functions being undocumented internal helpers.  Fixed by:
  (1) adding `c_args : vmaf_cflags_common` to the `static_library()` call so
  the TUs are compiled with `-fvisibility=hidden`; (2) annotating the four
  function declarations in `libvmaf/src/cpu.h`, `libvmaf/src/x86/cpu.h`, and
  `libvmaf/src/arm/cpu.h` with the new `VMAF_HIDDEN` macro (defined in
  `libvmaf/include/libvmaf/macros.h`) as a belt-and-suspenders guard against
  future meson.build regressions.  Symbol count in the DSO dropped from 44 to
  40 exported `T` symbols.  See ADR-0379.
