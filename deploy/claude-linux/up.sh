#!/usr/bin/env bash
# Start or redeploy DeepTutor on the Linux host. The Claude token arrives on
# stdin (README.md: piped from the Mac through `op run ... | ssh`) and reaches
# docker compose only through this process's environment: never echoed, never
# written to a file.
set -euo pipefail
cd "$(dirname "$0")"

if [ -t 0 ]; then
  echo "up.sh: pipe CLAUDE_CODE_OAUTH_TOKEN in on stdin (see README.md); refusing to read it from a terminal" >&2
  exit 1
fi
IFS= read -r CLAUDE_CODE_OAUTH_TOKEN || true
if [ -z "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
  echo "up.sh: no CLAUDE_CODE_OAUTH_TOKEN on stdin (see README.md)" >&2
  exit 1
fi
export CLAUDE_CODE_OAUTH_TOKEN

# Bind sources Docker would otherwise create root-owned (as scripts/docker_compose.py does).
mkdir -p ../../data/user/workspace/outputs ../../data/cli-apps

files=(-f compose.yaml)
if [[ $(docker info --format '{{json .Runtimes}}') == *'"nvidia"'* ]]; then
  echo "up.sh: nvidia runtime found, adding compose.gpu.yaml"
  files+=(-f compose.gpu.yaml)
fi
docker compose "${files[@]}" up -d --remove-orphans
