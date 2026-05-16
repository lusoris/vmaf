## OrtMemoryInfo caching

Cache the CPU memory descriptor (`OrtMemoryInfo *`) at DNN session-open time instead of allocating and releasing it on every inference call. ONNX Runtime recommends caching this handle to avoid repeated allocator round-trips in per-frame inference loops. Eliminates one ORT allocation per frame when using tiny-AI models with `--tiny-model`.

Perf impact: ~0.5–2 ms/frame recovery on 1080p streams with tiny-model inference (per perf-audit win #2). Bit-identical output — lifecycle-only change with no math changes.
