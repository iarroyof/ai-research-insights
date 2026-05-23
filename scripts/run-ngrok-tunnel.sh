#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/sabia/ai-research-insights}"
NGROK_URL="${NGROK_URL:-https://bayleigh-juxtapositional-shirleen.ngrok-free.dev}"
NGROK_TARGET="${NGROK_TARGET:-8080}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/ai-research-insights-ngrok}"
LOG_FILE="${LOG_FILE:-$STATE_DIR/ngrok-supervisor.log}"

mkdir -p "$STATE_DIR"
touch "$LOG_FILE"

exec 9>"$STATE_DIR/supervisor.lock"
if ! flock -n 9; then
  echo "$(date -Is) supervisor already running" >> "$LOG_FILE"
  exit 0
fi

cd "$PROJECT_ROOT"
echo "$(date -Is) supervisor started url=$NGROK_URL target=$NGROK_TARGET" >> "$LOG_FILE"

while true; do
  echo "$(date -Is) starting ngrok" >> "$LOG_FILE"
  set +e
  ngrok http --url="$NGROK_URL" "$NGROK_TARGET" --log=stdout >> "$LOG_FILE" 2>&1
  code=$?
  set -e
  echo "$(date -Is) ngrok exited code=$code; restarting in 5s" >> "$LOG_FILE"
  sleep 5
done
