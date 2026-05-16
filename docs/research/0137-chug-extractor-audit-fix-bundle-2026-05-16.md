# Research-0137 — CHUG Extractor Audit Fix Bundle (2026-05-16)

Three independent bugs found in `ai/scripts/extract_k150k_features.py` during
a post-merge audit of the K150K-A extraction pipeline. All three are safe to
bundle: they touch non-overlapping functions, none alters the output schema,
and the fixes are each one-to-five lines.

---

## Fix 1 — Data corruption on HDR clips (severity: High / scope: Small)

**Symptom.** Every CHUG 10-bit HDR clip was silently extracted as 8-bit.
`_load_jsonl_metadata` filtered sidecar rows through a `keep` tuple that
omitted `"chug_bit_depth"`. The downstream `_geometry_from_sidecar` received
`None` for that field and fell back to `pix_fmt="yuv420p"` (8-bit), so the
libvmaf invocation decoded the 10-bit stream with the wrong pixel format.
The resulting feature vectors are numerically wrong for all HDR rows.

**Root cause.** The `keep` tuple was the authoritative allow-list for sidecar
fields passed through the metadata dict. `"chug_bit_depth"` was added to the
sidecar JSONL schema after the tuple was first written and the omission was
not caught.

**Fix.** Add `"chug_bit_depth"` to the `keep` tuple. One-line change.

**Alternatives ruled out.** Removing the allow-list filter entirely (pass the
whole row dict) — rejected: the filter is load-bearing for downstream column
consistency; passing unknown sidecar fields into the parquet schema would
widen it silently and break schema-version assertions. Making
`_geometry_from_sidecar` tolerate `None` with a logging message — rejected:
a silent 8-bit fallback is the bug; the correct fix is to ensure the field
is present, not to make the fallback less silent.

---

## Fix 2 — Dead disk write per CUDA clip (severity: Medium / scope: Small)

**Symptom.** For every CUDA-mode clip, `_run_feature_passes` wrote the merged
frame list back to `out_json` immediately after calling `_merge_frame_metrics`.
That file was never read by any caller (the caller receives the returned Python
list). In `_process_clip`, the `finally` block calls `out_json.unlink()`,
discarding the file unconditionally.

**Root cause.** The `write_text` call was likely left over from an earlier
design where `out_json` was read back rather than passed as a list. The
refactor that switched to returning the list in-memory did not remove the now-
dead write.

**Fix.** Remove the `out_json.write_text(...)` call. Three-line removal.

**Alternatives ruled out.** Keeping the write but also reading back from it
to unify code paths — rejected: the per-clip write makes total disk I/O
O(N) for an output that is thrown away; the Research-0135 Win-1 principle
(parquet writes at-end-only) generalises here.

---

## Fix 3 — Silent frame-count truncation (severity: Low / scope: Small)

**Symptom.** When a CUDA pass and a CPU residual pass returned different frame
counts, `_merge_frame_metrics` silently truncated to `min(len(primary),
len(residual))`. The discrepancy was undetectable in the output parquet and
provided no audit signal for diagnosing CUDA/CPU pass mismatches.

**Root cause.** The original implementation treated count equality as an
implicit contract. CUDA context synchronisation failures or mid-file decode
errors can produce shorter-than-expected frame lists, making the silent
truncation a subtle data-quality gap.

**Fix.** Emit a `warnings.warn(...)` before the `min()` when the counts
differ. The soft-fail (warn, not raise) is intentional: the extraction is
still usable data for analysis; the audit signal is what matters. Uses stdlib
`warnings` consistent with the existing `warnings.catch_warnings()` pattern
in `_aggregate_frames`.

**Alternatives ruled out.** Raising an exception — rejected: a hard error
would halt the entire corpus run for a clip that produced partially valid
data. Using `logging.warning` — rejected: existing style in the same module
uses `warnings` for non-fatal anomalies visible to the caller; logging is
reserved for structured run-level output.
