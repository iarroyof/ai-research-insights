#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        TRIPLETS INGESTION WITH MONITORING                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Start monitoring in background
echo "🚀 Starting real-time monitor in background..."
./monitor_ingestion.sh > ingestion_monitor.log 2>&1 &
MONITOR_PID=$!
echo "   Monitor PID: $MONITOR_PID"
echo "   Monitor log: tail -f ingestion_monitor.log"
echo ""

# Start ingestion
echo "📤 Starting ingestion (this will take 30-60 minutes)..."
echo "   Started at: $(date)"
echo ""

curl -X POST "http://localhost:18081/triplets/ingest_csv" \
  -H "X-Tenant-Id: default" \
  -H "X-API-Key: dev" \
  -F "file=@/mnt/data/Datos_cancer_pulmon/triplets_CC_BY_probabilidad.csv" \
  2>&1 | tee /tmp/ingest_result.json

echo ""
echo "   Finished at: $(date)"
echo ""

# Kill monitor
kill $MONITOR_PID 2>/dev/null || true

# Show results
echo "═══════════════════════════════════════════════════════════"
echo "📊 INGESTION RESULTS:"
cat /tmp/ingest_result.json | jq .
echo ""

# Run final verification
echo "🔍 Running final verification..."
./test_partial_data.sh

echo ""
echo "✅ All done! Check Streamlit UI at http://localhost:8501"
