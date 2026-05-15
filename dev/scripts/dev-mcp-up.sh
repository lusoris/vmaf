#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
# Copyright 2026 Lusoris and Claude (Anthropic)
#
# dev/scripts/dev-mcp-up.sh — start the dev-MCP stack
#
# Usage:
#   ./dev/scripts/dev-mcp-up.sh          # CPU + lavapipe Vulkan only
#   NVIDIA_VISIBLE_DEVICES=all CONTAINER_RUNTIME=nvidia \
#       ./dev/scripts/dev-mcp-up.sh      # with NVIDIA GPU passthrough

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROBE_DIR="${REPO_ROOT}/.workingdir/dev-mcp-probes"

# Ensure probe output directory exists
mkdir -p "${PROBE_DIR}"

echo "[dev-mcp-up] Building container image (if needed)…"
docker compose \
  --project-directory "${REPO_ROOT}" \
  -f "${REPO_ROOT}/dev/docker-compose.yml" \
  build

echo "[dev-mcp-up] Starting dev-MCP stack…"
docker compose \
  --project-directory "${REPO_ROOT}" \
  -f "${REPO_ROOT}/dev/docker-compose.yml" \
  up -d

echo "[dev-mcp-up] Stack is up. Services:"
docker compose \
  --project-directory "${REPO_ROOT}" \
  -f "${REPO_ROOT}/dev/docker-compose.yml" \
  ps

echo ""
echo "[dev-mcp-up] MCP socket: .workingdir/dev-mcp-probes/ (probes written here)"
echo "[dev-mcp-up] Attach:     ./dev/scripts/dev-mcp-shell.sh"
echo "[dev-mcp-up] Probe now:  ./dev/scripts/dev-mcp-probe.sh"
echo "[dev-mcp-up] Stop:       ./dev/scripts/dev-mcp-down.sh"
