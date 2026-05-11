- `libvmaf_metal` ffmpeg filter — dedicated VideoToolbox hwdec zero-copy
  import path mirroring `libvmaf_vulkan`. Consumes
  `AV_PIX_FMT_VIDEOTOOLBOX` frames, pulls the IOSurface backing each
  `CVPixelBufferRef` via `CVPixelBufferGetIOSurface`, and routes it
  through new `vmaf_metal_picture_import` /
  `vmaf_metal_read_imported_pictures` C-API entry points
  (`libvmaf/include/libvmaf/libvmaf_metal.h`). Audit-first scaffold:
  the libvmaf-side runtime returns -ENOSYS and the filter fails fast
  with a clear pointer at
  [ADR-0423](docs/adr/0423-metal-iosurface-import-scaffold.md) until
  T8-IOS-b lands the
  `[id<MTLDevice> newTextureWithDescriptor:iosurface:plane:]` wiring.
  Companion to `--enable-libvmaf-metal` from
  `ffmpeg-patches/0012-libvmaf-wire-metal-backend-selector.patch`
  (software AVFrame input + Metal compute on the regular `libvmaf`
  filter); together the two ship the full Metal integration surface
  on FFmpeg. New ffmpeg patch:
  `ffmpeg-patches/0013-libvmaf-add-libvmaf-metal-filter.patch`.
