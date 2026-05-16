# Refreshing the in-tree FFmpeg patch series (stub)

> **Stub** — placeholder per
> [Research-0086](../research/0086-usage-doc-coverage-audit-2026-05-08.md).
> Cite the ADR for the authoritative shape; full prose follows in a
> later PR.

The fork's FFmpeg integration is a stack of patches under
`ffmpeg-patches/series.txt`, applied on top of upstream FFmpeg `n8.1`
(per [ADR-0118](../adr/0118-ffmpeg-patch-series-application.md)).
The stack drifts as upstream FFmpeg moves and as fork-side
libvmaf surfaces change ([CLAUDE.md §12 r14](../../CLAUDE.md)).

**Refreshing** the stack means:

1. Replay the patches against a clean `n8.1` checkout to verify the
   stack still applies cleanly.
2. Resolve any conflicts that fall out of upstream churn.
3. Re-export the patch files from the rebased branch.
4. Land the refresh in tree alongside an entry in
   [`docs/rebase-notes.md`](../rebase-notes.md).

The most recent refresh is documented in
[ADR-0277](../adr/0277-ffmpeg-patches-refresh-2026-05-04.md). The
`/refresh-ffmpeg-patches` skill (declared in
[`CLAUDE.md` §7](../../CLAUDE.md)) automates the replay step.

## Verification

The correct gate is a *cumulative-state* replay, not per-patch
`git apply --check` — patches `0002` … `0006` build on each other
and standalone-apply cleanly only against the cumulative state from
earlier patches:

```bash
git -C /path/to/ffmpeg-8 reset --hard n8.1
for p in ffmpeg-patches/000*-*.patch; do
    git -C /path/to/ffmpeg-8 am --3way "$p" || break
done
```

## See also

- [`docs/usage/ffmpeg.md`](../usage/ffmpeg.md) — the user-facing
  FFmpeg-with-libvmaf doc.
- [ADR-0118](../adr/0118-ffmpeg-patch-series-application.md) — the
  series.txt model.
- [ADR-0277](../adr/0277-ffmpeg-patches-refresh-2026-05-04.md) —
  most recent refresh log.
- [`docs/rebase-notes.md`](../rebase-notes.md) — fork-wide rebase
  log.
