#!/usr/bin/env bash
# Self-improvement loop: replay (generate trades with the current strategy)
# → reflect (change one variable), one round per run.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

echo "=== $(date) ==="

# Read current version
VERSION=$(grep '^version:' state/strategy.yaml | tr -d '"' | awk '{print $2}')
echo "Current strategy version: $VERSION"

# Step 1: Replay — generate fresh trades with the current strategy
echo "[1/2] Running replay (GC=F, 730d, 1h)..."
uv run python -m trading_lab.replay --asset GC=F --period 730d --interval 1h 2>&1 | tail -15

# Step 2: Reflect — score trades, change one variable
echo "[2/2] Running reflection..."
uv run python -m trading_lab.reflect --fallback 2>&1

NEW_VERSION=$(grep '^version:' state/strategy.yaml | tr -d '"' | awk '{print $2}')
echo ""
echo "=== RESULT: v$VERSION → v$NEW_VERSION ==="

echo "--- strategy.yaml ---"
cat state/strategy.yaml

echo ""
echo "--- latest hypothesis ---"
tail -1 state/hypotheses.jsonl | python3 -m json.tool 2>/dev/null || tail -1 state/hypotheses.jsonl

echo ""
echo "=== DONE ==="
