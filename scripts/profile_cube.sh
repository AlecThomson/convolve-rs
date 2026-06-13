#!/usr/bin/env bash
#
# End-to-end profiling of `convolvers 3d` on a synthetic cube. Captures
# wall-clock, peak RSS, and (if available) CPU counters, so the I/O-vs-compute
# balance and per-tier speedups can be measured and re-measured.
#
# Usage:
#   scripts/profile_cube.sh [NX] [NY] [NCHAN] [DTYPE] [MODE]
# Defaults: 2048 2048 64 float32 total
#
# Requires: a Python with astropy (for scripts/make_test_cube.py) and a release
# build of the binary (built automatically below).
set -euo pipefail

NX="${1:-2048}"
NY="${2:-2048}"
NCHAN="${3:-64}"
DTYPE="${4:-float32}"
MODE="${5:-total}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
CUBE="$WORK/profile_cube.fits"

PYTHON="${PYTHON:-python3}"

echo "==> Building release binary"
cargo build --release --quiet --manifest-path "$ROOT/Cargo.toml"
BIN="$ROOT/target/release/convolvers"

echo "==> Generating ${NCHAN}×${NY}×${NX} ${DTYPE} cube"
"$PYTHON" "$ROOT/scripts/make_test_cube.py" \
    --nx "$NX" --ny "$NY" --nchan "$NCHAN" --dtype "$DTYPE" -o "$CUBE"

echo "==> Smoothing (mode=$MODE)"
# Build a best-effort wrapper: `perf stat` for CPU counters if present, else
# GNU `/usr/bin/time -v` for peak RSS if present, else run the binary directly
# (we still get wall-clock from `date` below). The shell `time` keyword is not
# used here because it cannot be invoked as a command array.
WRAP=()
if command -v perf >/dev/null 2>&1; then
    WRAP=(perf stat --)
elif [ -x /usr/bin/time ]; then
    WRAP=(/usr/bin/time -v)
fi

START=$(date +%s.%N)
"${WRAP[@]}" "$BIN" 3d "$CUBE" --mode "$MODE" --outdir "$WORK"
END=$(date +%s.%N)

awk -v s="$START" -v e="$END" -v n="$NCHAN" 'BEGIN {
    wall = e - s;
    printf "==> Wall: %.2f s   Throughput: %.1f channels/s\n", wall, n / wall;
}'
