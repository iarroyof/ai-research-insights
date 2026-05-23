#!/usr/bin/env bash
set -euo pipefail

# Non-invasive preflight. It does not start, stop, remove, or restart anything.
# It only checks that this project's default published ports are not already
# owned by containers outside the AI Research Insights Compose project.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROJECT_PREFIX="${COMPOSE_PROJECT_NAME:-ai-research-insights}"
NETWORK_NAME="${AI_RESEARCH_NETWORK_NAME:-ai_research_insights_net}"

ports=(
  "${CADDY_HTTP_PORT:-8080}"
  "${CADDY_HTTPS_PORT:-8443}"
  "${API_HOST_PORT:-18081}"
  "${OPENSEARCH_HOST_PORT:-19200}"
  "${NEO4J_HTTP_PORT:-7474}"
  "${NEO4J_BOLT_PORT:-7687}"
)

conflicts=0
for port in "${ports[@]}"; do
  [[ -n "$port" ]] || continue
  owners="$(docker ps --format '{{.Names}}\t{{.Ports}}' | awk -v port=":$port->" 'index($0, port) {print $1}')"
  [[ -n "$owners" ]] || continue
  while IFS= read -r owner; do
    [[ -n "$owner" ]] || continue
    if [[ "$owner" != "$PROJECT_PREFIX"* && "$owner" != ai-research-insights* ]]; then
      echo "port-conflict port=$port owner=$owner"
      conflicts=1
    fi
  done <<< "$owners"
done

if docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
  foreign_network_users="$(
    docker network inspect "$NETWORK_NAME" \
      --format '{{range $id, $c := .Containers}}{{println $c.Name}}{{end}}' |
      awk -v prefix="$PROJECT_PREFIX" '$0 !~ "^" prefix && $0 !~ "^ai-research-insights" {print}'
  )"
  if [[ -n "$foreign_network_users" ]]; then
    echo "network-shared network=$NETWORK_NAME"
    echo "$foreign_network_users"
    conflicts=1
  fi
fi

if [[ "$conflicts" -ne 0 ]]; then
  echo "isolation-check-failed"
  exit 1
fi

echo "isolation-check-ok network=$NETWORK_NAME ports=${ports[*]}"
