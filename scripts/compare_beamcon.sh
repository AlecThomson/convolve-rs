#!/bin/zsh
# Quick CLI benchmark of convolvers vs RACS-tools beamcon_2D on a batch of images.
# A lightweight companion to scripts/benchmark.ipynb for a fast terminal check.
#
# Times each tool with the median of N fresh runs (median is robust to jitter;
# a fresh output dir per run keeps prior *.sm.fits outputs from being re-read).
#
# Binaries: found on PATH, or override with env vars:
#   CONVOLVERS_BIN=/path/to/convolvers  BEAMCON_BIN=/path/to/beamcon_2D
#
# Usage:
#   scripts/compare_beamcon.sh [N] [SIZE] [REPS]
# Defaults: N=8 images, SIZE=4096, REPS=3. Generates dummy data under bench_out/.
set -e
cd "$(dirname "$0")/.."

N="${1:-8}"
SIZE="${2:-4096}"
REPS="${3:-3}"
TARGET=(--bmaj 20 --bmin 20 --bpa 0)

CONVOLVERS="${CONVOLVERS_BIN:-$(command -v convolvers || echo target/release/convolvers)}"
BEAMCON="${BEAMCON_BIN:-$(command -v beamcon_2D || echo beamcon_2D)}"
DATA="bench_out/compare_${N}x${SIZE}/data"

echo "convolvers: $CONVOLVERS"
echo "beamcon_2D: $BEAMCON"

echo "generating $N x ${SIZE}^2 images..."
uv run --extra benchmark python scripts/make_dummy_fits.py \
  --n "$N" --size "$SIZE" --outdir "$DATA" >/dev/null

run_once() {  # echo wall seconds for one run of $1=convolvers|beamcon
  local d; d=$(mktemp -d)
  if [[ "$1" == convolvers ]]; then
    { /usr/bin/time -p "$CONVOLVERS" 2d -o "$d" "${TARGET[@]}" "$DATA"/*.fits >/dev/null; } 2>/tmp/_cmptime
  else
    { /usr/bin/time -p "$BEAMCON" -o "$d" --executor process "${TARGET[@]}" "$DATA"/*.fits >/dev/null; } 2>/tmp/_cmptime
  fi
  rm -rf "$d"
  grep '^real' /tmp/_cmptime | awk '{print $2}'
}

median() {  # median wall (s) of REPS runs; $1=convolvers|beamcon
  local vals=()
  for r in $(seq 1 "$REPS"); do vals+=$(run_once "$1"); done
  printf '%s\n' "${vals[@]}" | sort -n | sed -n "$(( (REPS + 1) / 2 ))p"
}

echo "\n=== median wall over $REPS runs ($N x ${SIZE}^2) ==="
rs=$(median convolvers)
py=$(median beamcon)
printf "  convolvers      : %ss\n" "$rs"
printf "  beamcon-process : %ss\n" "$py"
awk -v p="$py" -v r="$rs" 'BEGIN { printf "  ratio (beamcon/convolvers): %.2fx\n", p/r }'
