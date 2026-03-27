#!/bin/bash
#
# run_monthly.sh - Run CN scoring for the 1st of each month from Nov 2023 to Feb 2026
#
# Usage:
#   ./run_monthly.sh
#
# This script iterates through each month and calls run_at_date.py for each.
# If a date fails, it logs the error and continues to the next date.
#

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATES=(
    2023-11-01
    2023-12-01
    2024-01-01
    2024-02-01
    2024-03-01
    2024-04-01
    2024-05-01
    2024-06-01
    2024-07-01
    2024-08-01
    2024-09-01
    2024-10-01
    2024-11-01
    2024-12-01
    2025-01-01
    2025-02-01
    2025-03-01
    2025-04-01
    2025-05-01
    2025-06-01
    2025-07-01
    2025-08-01
    2025-09-01
    2025-10-01
    2025-11-01
    2025-12-01
    2026-01-01
    2026-02-01
)

TOTAL=${#DATES[@]}
SUCCEEDED=0
FAILED=0
FAILED_DATES=()

echo "========================================"
echo "  CN Monthly Scoring Pipeline"
echo "  Running $TOTAL dates"
echo "========================================"
echo ""

for i in "${!DATES[@]}"; do
    DATE="${DATES[$i]}"
    NUM=$((i + 1))

    echo ""
    echo "============================================"
    echo "  [$NUM/$TOTAL] Running CN scoring for: $DATE"
    echo "============================================"

    if python "$SCRIPT_DIR/run_at_date.py" --date "$DATE"; then
        SUCCEEDED=$((SUCCEEDED + 1))
        echo "[$NUM/$TOTAL] SUCCESS: $DATE"
    else
        FAILED=$((FAILED + 1))
        FAILED_DATES+=("$DATE")
        echo "[$NUM/$TOTAL] FAILED: $DATE (continuing to next date)"
    fi
done

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo "  Total:     $TOTAL"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "  Failed dates:"
    for d in "${FAILED_DATES[@]}"; do
        echo "    - $d"
    done
fi

echo ""
echo "Done."
