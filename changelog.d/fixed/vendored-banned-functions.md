## Fixed: Banned functions in vendored cJSON and libsvm (ADR-0455)

Replace `sprintf` / `strcpy` calls in `libvmaf/src/mcp/3rdparty/cJSON/cJSON.c`
with `snprintf` / `memmove` / `memcpy` equivalents. Replace `rand()` calls in
`libvmaf/src/svm.cpp` with a thread-local `rand_r`-based PRNG. A new public API
`svm_set_rand_seed(unsigned seed)` enables deterministic cross-validation for
reproducible test execution.

The project's "vendored is in scope" policy now explicitly requires the same
banned-function standards in vendored third-party code as in fork-original code.
