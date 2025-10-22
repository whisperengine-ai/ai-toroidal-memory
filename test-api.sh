#!/bin/bash
# Test script for Toroidal Memory API

set -e

API_URL="${API_URL:-http://localhost:3000}"

echo "🧪 Testing Toroidal Memory API at $API_URL"
echo ""

# Test 1: Health check
echo "1️⃣  Testing health endpoint..."
HEALTH=$(curl -s "$API_URL/health")
if [ "$HEALTH" = "OK" ]; then
    echo "   ✅ Health check passed"
else
    echo "   ❌ Health check failed: $HEALTH"
    exit 1
fi

# Test 2: Create memory
echo ""
echo "2️⃣  Creating memory instance..."
CREATE_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/memories" \
    -H 'Content-Type: application/json' \
    -d '{"width": 50, "height": 50}')

MEMORY_ID=$(echo "$CREATE_RESPONSE" | grep -o '"memory_id":"[^"]*"' | cut -d'"' -f4)

if [ -n "$MEMORY_ID" ]; then
    echo "   ✅ Created memory: $MEMORY_ID"
else
    echo "   ❌ Failed to create memory"
    echo "   Response: $CREATE_RESPONSE"
    exit 1
fi

# Test 3: Set cell value
echo ""
echo "3️⃣  Setting cell value..."
SET_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/memories/$MEMORY_ID/cell" \
    -H 'Content-Type: application/json' \
    -d '{"x": 25, "y": 25, "value": 1.0}')

if echo "$SET_RESPONSE" | grep -q '"success":true'; then
    echo "   ✅ Cell value set successfully"
else
    echo "   ❌ Failed to set cell value"
    exit 1
fi

# Test 4: Run diffusion
echo ""
echo "4️⃣  Running diffusion..."
DIFF_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/memories/$MEMORY_ID/diffusion" \
    -H 'Content-Type: application/json' \
    -d '{"steps": 5}')

if echo "$DIFF_RESPONSE" | grep -q '"steps_executed":5'; then
    echo "   ✅ Diffusion completed"
else
    echo "   ❌ Diffusion failed"
    exit 1
fi

# Test 5: Get statistics
echo ""
echo "5️⃣  Getting memory statistics..."
STATS_RESPONSE=$(curl -s "$API_URL/api/v1/memories/$MEMORY_ID/stats")

if echo "$STATS_RESPONSE" | grep -q '"total_cells":2500'; then
    echo "   ✅ Statistics retrieved"
    ACTIVE_CELLS=$(echo "$STATS_RESPONSE" | grep -o '"active_cells":[0-9]*' | cut -d':' -f2)
    echo "   📊 Active cells: $ACTIVE_CELLS"
else
    echo "   ❌ Failed to get statistics"
    exit 1
fi

# Test 6: Save to file
echo ""
echo "6️⃣  Saving to file..."
SAVE_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/memories/$MEMORY_ID/save" \
    -H 'Content-Type: application/json' \
    -d '{"filename": "test_api.json"}')

if echo "$SAVE_RESPONSE" | grep -q '"success":true'; then
    echo "   ✅ Memory saved to file"
else
    echo "   ❌ Failed to save memory"
    exit 1
fi

# Test 7: Load from file
echo ""
echo "7️⃣  Loading from file..."
LOAD_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/memories/$MEMORY_ID/load" \
    -H 'Content-Type: application/json' \
    -d '{"filename": "test_api.json"}')

if echo "$LOAD_RESPONSE" | grep -q '"success":true'; then
    echo "   ✅ Memory loaded from file"
else
    echo "   ❌ Failed to load memory"
    exit 1
fi

# Test 8: Delete memory
echo ""
echo "8️⃣  Deleting memory..."
DELETE_RESPONSE=$(curl -s -X DELETE "$API_URL/api/v1/memories/$MEMORY_ID")

if echo "$DELETE_RESPONSE" | grep -q '"success":true'; then
    echo "   ✅ Memory deleted"
else
    echo "   ❌ Failed to delete memory"
    exit 1
fi

echo ""
echo "🎉 All tests passed!"
echo ""
