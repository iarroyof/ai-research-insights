#!/usr/bin/env bash
set -euo pipefail

# Hosted-provider mode: API/chat/context-manager use NVIDIA, BioNLI uses HF.
# GPU/local model-serving services stay disabled because no Compose profile is enabled.
export COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
export COMPOSE_PROFILES="${COMPOSE_PROFILES:-}"
export LLM_CHAT_PROVIDER="${LLM_CHAT_PROVIDER:-nvidia}"
export CONTEXT_MANAGER_PROVIDER="${CONTEXT_MANAGER_PROVIDER:-nvidia}"
export NLI_PROVIDER="${NLI_PROVIDER:-hf_api}"
export DOCKER_CPU_RUNTIME="${DOCKER_CPU_RUNTIME:-runc}"
export API_DOCKERFILE="${API_DOCKERFILE:-services/api/Dockerfile.hosted}"
export WORKER_REQUIREMENTS_FILE="${WORKER_REQUIREMENTS_FILE:-services/api/requirements.hosted.txt}"

exec docker compose "$@"
