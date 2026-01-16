#!/bin/bash
# Quick script to check test progress

LOG_FILE="/tmp/volatility_clustering_test.log"

if [ -f "$LOG_FILE" ]; then
    echo "=== Test Progress ==="
    echo ""
    echo "Last 20 lines:"
    tail -20 "$LOG_FILE"
    echo ""
    echo "=== Summary Status ==="
    if grep -q "SUMMARY:" "$LOG_FILE"; then
        echo "Test completed!"
        tail -100 "$LOG_FILE" | grep -A 100 "SUMMARY:"
    else
        echo "Test still running..."
        echo ""
        echo "Latest activity:"
        tail -5 "$LOG_FILE"
    fi
else
    echo "Log file not found. Test may not have started yet."
fi
