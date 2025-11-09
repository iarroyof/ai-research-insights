#!/bin/bash

echo "=== Testing Search with Partial Data ==="
echo ""

# Check how many docs we have
DOC_COUNT=$(curl -s "http://localhost:19200/triplets_default/_count" | jq -r '.count')
echo "📊 Current document count: $DOC_COUNT"
echo ""

if [ "$DOC_COUNT" -lt 1000 ]; then
    echo "⏳ Waiting for at least 1000 documents to be indexed..."
    echo "   Current: $DOC_COUNT"
    exit 1
fi

echo "🔍 Testing search queries..."
echo ""

# Test 1: Search for "cancer"
echo "Test 1: Searching for 'cancer'"
docker compose exec -T streamlit python3 - <<'PY'
import requests, json
url = "http://api:8080/search/"
headers = {"X-Tenant-Id":"default","X-API-Key":"dev","Content-Type":"application/json"}
payload = {"query":"cancer","target":"all","filters":{},"k":3}
r = requests.post(url, headers=headers, json=payload)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    items = data.get("items", [])
    print(f"Results: {len(items)} hits")
    if items:
        print("\nFirst result:")
        print(f"  Subject: {items[0].get('subject')}")
        print(f"  Relation: {items[0].get('relation')}")
        print(f"  Object: {items[0].get('object')}")
        print(f"  Score: {items[0].get('score')}")
        print(f"  Text: {items[0].get('text', '')[:100]}...")
    else:
        print("⚠️  No results found")
else:
    print(f"❌ Error: {r.text}")
PY

echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""

# Test 2: Search for "lung"
echo "Test 2: Searching for 'lung'"
docker compose exec -T streamlit python3 - <<'PY'
import requests, json
url = "http://api:8080/search/"
headers = {"X-Tenant-Id":"default","X-API-Key":"dev","Content-Type":"application/json"}
payload = {"query":"lung","target":"all","filters":{},"k":3}
r = requests.post(url, headers=headers, json=payload)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    items = data.get("items", [])
    print(f"Results: {len(items)} hits")
    if items:
        for i, item in enumerate(items[:2], 1):
            print(f"\n  Result {i}:")
            print(f"    {item.get('subject')} → {item.get('relation')} → {item.get('object')}")
else:
    print(f"❌ Error: {r.text}")
PY

echo ""
echo "✅ Early verification complete!"

