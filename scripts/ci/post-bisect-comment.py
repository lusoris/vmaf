#!/usr/bin/env python3
# Copyright 2026 Lusoris and Claude (Anthropic)
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
"""Post or update the sticky bisect-tracker comment on a tracking issue.

Used by `.github/workflows/nightly-bisect.yml`. The first time the
workflow runs against a new tracking issue, this writes a fresh comment.
On every subsequent run, it edits the *same* comment in place so the
issue history stays a flat audit log instead of a wall of duplicates.

Identification: the most recent comment authored by the workflow's
GITHUB_ACTOR whose body starts with the magic header `<!-- bisect-tracker -->`.

Usage:

    GH_TOKEN=...  python scripts/ci/post-bisect-comment.py \\
        --issue 42 --report bisect-result.json [--repo owner/name]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

STICKY_HEADER = "<!-- bisect-tracker -->"


def _gh(*args: str) -> str:
    """Run `gh` and return stdout; raise on non-zero exit."""
    # S603/S607: trusted args (caller-controlled, not user input);
    # `gh` resolved from $PATH is correct here — GitHub Actions
    # provides it pre-installed and at varying absolute locations.
    cmd = ["gh", *args]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    return result.stdout


def _format_body(report: dict, run_url: str) -> str:
    threshold = f"{report['threshold_kind']} = {report['threshold_value']:g}"
    n_models = report["n_models"]
    n_visited = len(report["steps"])
    first_bad = report.get("first_bad_index")
    last_good = report.get("last_good_index")
    if first_bad is None:
        verdict_line = f"**Verdict**: {report['verdict']}"
    else:
        verdict_line = (
            f"**REGRESSION** — first bad at index {first_bad} "
            f"(`{Path(report['first_bad_model']).name}`); "
            f"last good at index {last_good}"
            + (f" (`{Path(report['last_good_model']).name}`)" if last_good is not None else "")
        )

    rows = [
        "| idx | status | PLCC | SROCC | RMSE | model |",
        "| --: | :-: | --: | --: | --: | :-- |",
    ]
    for s in sorted(report["steps"], key=lambda r: r["index"]):
        status = "GOOD" if s["passed"] else "BAD"
        r = s["report"]
        rows.append(
            f"| {s['index']} | {status} | {r['plcc']:+.4f} | "
            f"{r['srocc']:+.4f} | {r['rmse']:.4f} | `{Path(s['model']).name}` |"
        )

    return "\n".join(
        [
            STICKY_HEADER,
            "## Latest nightly bisect-model-quality run",
            "",
            verdict_line,
            "",
            f"- threshold: `{threshold}`",
            f"- models in timeline: `{n_models}`",
            f"- models visited: `{n_visited}`",
            f"- workflow run: {run_url}",
            "",
            "<details><summary>per-step report</summary>",
            "",
            *rows,
            "",
            "</details>",
        ]
    )


def _find_existing_comment(repo: str, issue: int) -> int | None:
    """Return the comment ID of the existing sticky, or None."""
    raw = _gh(
        "api",
        f"repos/{repo}/issues/{issue}/comments",
        "--paginate",
    )
    comments = json.loads(raw)
    for c in reversed(comments):
        if c.get("body", "").startswith(STICKY_HEADER):
            return int(c["id"])
    return None


def _write_comment(repo: str, issue: int, body: str) -> None:
    comment_id = _find_existing_comment(repo, issue)
    body_path = Path(".bisect-comment-body.md")
    body_path.write_text(body, encoding="utf-8")
    try:
        if comment_id is None:
            _gh(
                "api",
                f"repos/{repo}/issues/{issue}/comments",
                "-f",
                f"body=@{body_path}",
            )
            print(f"posted new sticky comment on issue #{issue}")
        else:
            _gh(
                "api",
                "--method",
                "PATCH",
                f"repos/{repo}/issues/comments/{comment_id}",
                "-f",
                f"body=@{body_path}",
            )
            print(f"edited sticky comment {comment_id} on issue #{issue}")
    finally:
        body_path.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--issue", type=int, required=True, help="Tracking issue number")
    p.add_argument("--report", type=Path, required=True, help="Path to bisect JSON report")
    p.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "lusoris/vmaf"),
        help="owner/name (defaults to $GITHUB_REPOSITORY or lusoris/vmaf)",
    )
    args = p.parse_args()

    if not args.report.is_file():
        print(f"report not found: {args.report}", file=sys.stderr)
        return 1
    report = json.loads(args.report.read_text(encoding="utf-8"))

    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID", "?")
    run_url = f"{server_url}/{args.repo}/actions/runs/{run_id}"

    body = _format_body(report, run_url)
    _write_comment(args.repo, args.issue, body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
