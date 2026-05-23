#!/usr/bin/env bash
set -euo pipefail

# Local GPU mode: explicitly opt into services that load local models.
export COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
export COMPOSE_PROFILES="${COMPOSE_PROFILES:-gpu}"
export LLM_CHAT_PROVIDER="${LLM_CHAT_PROVIDER:-local}"
export CONTEXT_MANAGER_PROVIDER="${CONTEXT_MANAGER_PROVIDER:-local}"
export DOCKER_CPU_RUNTIME="${DOCKER_CPU_RUNTIME:-runc}"
export DOCKER_GPU_RUNTIME="${DOCKER_GPU_RUNTIME:-nvidia}"
export API_DOCKERFILE="${API_DOCKERFILE:-services/api/Dockerfile}"
export WORKER_REQUIREMENTS_FILE="${WORKER_REQUIREMENTS_FILE:-services/api/requirements.hosted.txt}"
export WORKER_GPU_REQUIREMENTS_FILE="${WORKER_GPU_REQUIREMENTS_FILE:-services/api/requirements.txt}"

exec docker compose --profile gpu "$@"
