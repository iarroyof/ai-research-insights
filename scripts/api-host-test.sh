#!/usr/bin/env bash
set -euo pipefail

# Run API tests/tools inside the API image without Docker bridge networking.
#
# This exists for no-GPU unit/provider tests. Blue-demon's Docker daemon uses
# the NVIDIA runtime as the host default, so force runc for hosted/no-GPU tests.
# Do not use this as production runtime.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec docker run --rm \
  --runtime "${DOCKER_CPU_RUNTIME:-runc}" \
  --network host \
  --env-file .env \
  -e APP_CONFIG=/app/config/default.yaml \
  -e PYTHONPATH=/app \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD/services/api:/app" \
  -v "$PWD/config:/app/config:ro" \
  --entrypoint python \
  "${API_IMAGE:-ai-research-insights-api:latest}" \
  "$@"
