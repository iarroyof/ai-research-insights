#!/bin/bash

echo "=== Real-Time Ingestion Monitor ==="
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "  INGESTION MONITORING - $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    # Index stats
    echo "📊 INDEX STATISTICS:"
    COUNT=$(curl -s "http://localhost:19200/triplets_default/_count" 2>/dev/null | jq -r '.count // "N/A"')
    SIZE=$(curl -s "http://localhost:19200/triplets_default/_stats" 2>/dev/null | jq -r '.indices.triplets_default.total.store.size // "N/A"')
    echo "   Documents: $COUNT"
    echo "   Index Size: $SIZE"
    echo ""
    
    # Latest API logs
    echo "📝 LATEST API LOGS (last 10 lines):"
    docker compose logs --tail=10 api 2>/dev/null | grep -E "Progress:|INFO|docs indexed|complete" | tail -10
    echo ""
    
    echo "════════════════════════════════════════════════════════════"
    echo "Refreshing in 10 seconds... (Ctrl+C to stop)"
    sleep 10
done
