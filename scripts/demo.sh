#!/usr/bin/env bash
# Crater-hunt demo: teach the board (if needed), calibrate (if needed), scan.
#
#   ./scripts/demo.sh                    scan using the saved corners
#   ./scripts/demo.sh --teach            re-teach the board corners first
#   ./scripts/demo.sh --threshold 0.7    anything else is passed to board-scan
#
# Open the "live view:" address it prints in a browser to watch the camera and
# the crater confidence while it runs. Ctrl+C stops it at any point.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

teach=0
args=()
for a in "$@"; do
  if [ "$a" = "--teach" ]; then teach=1; else args+=("$a"); fi
done

if [ "$teach" = 1 ] || [ ! -f data/board_corners.json ]; then
  echo "== Teach the board corners: crosshair on each corner in order, m to mark, s to save"
  python3 -m pantilt teach
fi

if [ ! -f data/board_step.json ] || ! grep -q pan_vec data/board_step.json; then
  echo "== Calibrate the servo step (about 30 s, keep the board still)"
  python3 -m pantilt calibrate-step
fi

echo "== Plan"
python3 -m pantilt board-scan --dry-run "${args[@]}"

echo
echo "== Scan the board, then go back to points of interest"
python3 -m pantilt board-scan "${args[@]}"

latest=$(ls -td data/runs/board-* 2>/dev/null | head -1 || true)
if [ -n "$latest" ]; then
  echo
  echo "== Results in $latest"
  echo "   overview:   $latest/overview_p1_0.jpg"
  echo "   photos:     $(ls "$latest/photos" 2>/dev/null | grep -vc _crop || true)"
fi
