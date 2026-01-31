#!/bin/bash
# EdgeRunner long-running background test script
# Usage: ./scripts/start_test.sh
#
# All processes run in background, output redirected to log files
# Prevents terminal buffer overflow from blocking the program

set -e

# Configuration
TEST_DURATION=${1:-25200}  # Default: 7 hours = 25200 seconds
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${TIMESTAMP}"

echo "==========================================="
echo "EdgeRunner Long-Running Background Test"
echo "==========================================="
echo "Test duration: $TEST_DURATION seconds"
echo "Timestamp: $TIMESTAMP"
echo ""

# Phase 1: System preparation
echo "Step 1: Running jetson_clocks..."
sudo jetson_clocks 2>/dev/null || true
echo "   ✅ jetson_clocks enabled"

echo "Step 2: Suppressing kernel messages..."
sudo dmesg -n 1 2>/dev/null || true
echo "   ✅ dmesg silenced"

# Phase 2: Create log directory
echo "Step 3: Creating log directory..."
mkdir -p "$LOG_DIR"
echo "   ✅ Log directory: $LOG_DIR"

# Phase 3: Clean up old processes
echo "Step 4: Cleaning up old processes..."
# Kill old main.py processes
pkill -f "src.main" 2>/dev/null || true
pkill -f "stability_test" 2>/dev/null || true
sleep 2
echo "   ✅ Old processes cleaned"

# Phase 4: Wait for llama-server
echo "Step 5: Waiting for llama-server..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "   ❌ llama-server not ready, exiting"
        exit 1
    fi
    sleep 2
done
echo "   ✅ llama-server ready"

# Phase 5: Start main.py (background, log redirect)
MAIN_LOG="$LOG_DIR/main.log"
echo "Step 6: Starting main.py (background)..."

# Run with nohup, redirect all output to log
nohup python -m src.main --quiet > "$MAIN_LOG" 2>&1 &
MAIN_PID=$!

echo "   ✅ main.py started (PID: $MAIN_PID)"
echo "   📄 Log: $MAIN_LOG"

sleep 5

# Verify main.py is running
if ! kill -0 $MAIN_PID 2>/dev/null; then
    echo "   ❌ main.py failed to start, check log:"
    tail -20 "$MAIN_LOG"
    exit 1
fi

# Check API availability
if ! curl -sf http://localhost:8090/api/status > /dev/null 2>&1; then
    echo "   ⚠️  API not ready, waiting..."
    sleep 10
fi
echo "   ✅ main.py running normally"

# Phase 6: Start stability test (background, log redirect)
TEST_LOG="$LOG_DIR/stability.log"
echo "Step 7: Starting stability test (background)..."

nohup python -m tools.stability_test --duration $TEST_DURATION > "$TEST_LOG" 2>&1 &
TEST_PID=$!

echo "   ✅ stability_test started (PID: $TEST_PID)"
echo "   📄 Log: $TEST_LOG"

# Phase 7: Save PID file
PID_FILE="$LOG_DIR/pids.txt"
echo "Step 8: Saving PID info..."
cat > "$PID_FILE" << EOF
# EdgeRunner long-running test PID file
# Started: $(date)
MAIN_PID=$MAIN_PID
TEST_PID=$TEST_PID
TEST_DURATION=$TEST_DURATION
LOG_DIR=$LOG_DIR
EOF
echo "   ✅ PID file: $PID_FILE"

# Summary
echo ""
echo "==========================================="
echo "✅ Background test started!"
echo "==========================================="
echo ""
echo "Process info:"
echo "   • main.py PID:       $MAIN_PID"
echo "   • stability PID:     $TEST_PID"
echo ""
echo "Log files:"
echo "   • main.py log:       $MAIN_LOG"
echo "   • stability log:     $TEST_LOG"
echo ""
echo "Useful commands:"
echo "   • Watch main log:       tail -f $MAIN_LOG"
echo "   • Watch stability log:  tail -f $TEST_LOG"
echo "   • Check processes:      ps aux | grep -E 'src.main|stability'"
echo "   • Stop test:            kill $MAIN_PID $TEST_PID"
echo ""
echo "Estimated completion: $(date -d "+$TEST_DURATION seconds" 2>/dev/null || echo 'N/A')"
