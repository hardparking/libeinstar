#!/usr/bin/env bash
# First-contact test with a real Einstar. Collects everything into one folder
# (and a .tar.gz) that can be attached to an issue.
#
#   scripts/einstar-bringup.sh [build-dir]
#
# Steps 1-3 only read from the scanner. Step 4 starts and stops the scan
# trigger (volatile settings only) and is skipped unless you confirm.
# Close EXStar first: only one program can claim the scanner.
set -u

BUILD="${1:-build}"
PROBE="$BUILD/einstar-probe"
CALIB="$BUILD/einstar-calib"
RTABMAP="$BUILD/integrations/rtabmap/einstar-rtabmap"
CCF_DIR="/Applications/EXStar.app/Contents/Resources/res/Einscan-E10/200x150"
OUT="einstar-bringup-$(date +%Y%m%d-%H%M%S)"

if [ ! -x "$PROBE" ]; then
  echo "einstar-probe not found in $BUILD; build first (see README)." >&2
  exit 1
fi
if pgrep -x EXStar >/dev/null 2>&1 || pgrep -f SnSyncService >/dev/null 2>&1; then
  echo "EXStar or its SnSyncService is running; quit it so libeinstar can claim the scanner." >&2
  exit 1
fi
mkdir -p "$OUT/frames"

run() {  # run <label> <command...>: log the command, its output and exit code
  local label="$1"; shift
  echo "== $label: $*" | tee -a "$OUT/log.txt"
  "$@" 2>&1 | tee -a "$OUT/log.txt"
  local rc=${PIPESTATUS[0]}
  echo "== exit $rc" | tee -a "$OUT/log.txt"
  return "$rc"
}

{
  echo "date: $(date)"
  echo "host: $(uname -a)"
  echo "libeinstar: $(git -C "$(dirname "$0")/.." describe --always --dirty 2>/dev/null)"
} > "$OUT/env.txt"
system_profiler SPUSBDataType > "$OUT/usb-tree.txt" 2>/dev/null || lsusb -v > "$OUT/usb-tree.txt" 2>/dev/null

run "1. enumerate (read-only)" "$PROBE"
run "2. identity and settings (read-only)" "$PROBE" --info || {
  echo "Retrying with the XOR command mask enabled" | tee -a "$OUT/log.txt"
  run "2b. identity with --mask" "$PROBE" --info --mask
}
run "3. calibration flash blob (read-only)" "$PROBE" --dump-flash "$OUT/custom-info.bin"
if [ -x "$CALIB" ]; then
  [ -s "$OUT/custom-info.bin" ] && "$CALIB" "$OUT/custom-info.bin" > "$OUT/calibration-device.yaml" 2>>"$OUT/log.txt"
  [ -d "$CCF_DIR" ] && "$CALIB" "$CCF_DIR" > "$OUT/calibration-exstar.yaml" 2>>"$OUT/log.txt"
  if [ -s "$OUT/calibration-device.yaml" ] && [ -s "$OUT/calibration-exstar.yaml" ]; then
    if cmp -s "$OUT/calibration-device.yaml" "$OUT/calibration-exstar.yaml"; then
      echo "== device calibration matches EXStar's cache" | tee -a "$OUT/log.txt"
    else
      echo "== device calibration differs from EXStar's cache (different scanner or decoder issue)" | tee -a "$OUT/log.txt"
    fi
  fi
fi

echo
read -r -p "Step 4 starts the scanner's trigger and captures 30 frames. Continue? [y/N] " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
  run "4. scan and capture" "$PROBE" --scan --frames 30 --out "$OUT/frames"
  if [ -x "$RTABMAP" ] && ls "$OUT"/frames/*.pgm >/dev/null 2>&1; then
    # Prefer the calibration just read from the scanner; fall back to EXStar's cache.
    CALIB_SRC="$OUT/custom-info.bin"
    [ -s "$CALIB_SRC" ] || CALIB_SRC="$CCF_DIR"
    run "5. RTAB-Map replay of the capture" "$RTABMAP" --calib "$CALIB_SRC" --replay "$OUT/frames" \
        --db "$OUT/replay.db" --poses "$OUT/odom.txt"
  fi
fi

tar -czf "$OUT.tar.gz" "$OUT"
echo
echo "Results: $OUT/ and $OUT.tar.gz"
echo "custom-info.bin and calibration-*.yaml are your scanner's calibration; share them only if you choose to."
