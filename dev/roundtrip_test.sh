#!/usr/bin/env bash
# roundtrip_test.sh - end-to-end fast round-trip gate for the accel backend.
# Exercises the real fast-iteration path on a large deck: read .inp -> GMG solve -> fast .dat output ->
# read the displacement field back -> verify it. This is the regression guard for the desploppify work:
# any edit that breaks correctness, regresses convergence, or breaks the output is caught here.
#
# GATES (a regression in any FAILS):
#   * correctness : maxU(read back from the .dat) within REL (default 1e-2, the 1% screening bar) of GOLDEN.
#   * no-slowdown : GMG PCG iters <= ITER_CAP (default 40). Iters are load-INDEPENDENT, so this is the robust
#                   "did we slow the solve down" signal -- an algorithmic regression raises iters.
#   * round-trip  : the .dat is produced and parses (the field actually comes back).
#   * wall        : end-to-end wall reported; PASS if < WALL_MAX (default 25 s). Wall is load-dependent, so
#                   when the 1-min load average is high it is reported as INDETERMINATE (not a hard fail) --
#                   re-run on a quiet box for the authoritative wall number.
#
# Usage: roundtrip_test.sh [deck.inp] [golden_maxU]
#   deck   default: a bundled large reproducer (~120 MB; not bundled -> SKIP if absent).
#   golden default: 107.0569734213358 (the default deck's direct-solve golden). Pass both for another deck.
# Env: OMP_NUM_THREADS (default 16), GMG_TOL (default 3e-3), ITER_CAP, WALL_MAX, REL.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SRC=../src
BIN="$(cd "$SRC" && pwd)/CalculiX_MT"
DECK_IN="${1:-../../repro_cases/row236_path02_fulltrunk_hubfan_20260615/input/r236_full_trunk_hub_fan_static_y.inp}"
GOLDEN="${2:-107.0569734213358}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
GMG_TOL="${GMG_TOL:-3e-3}"; export GMG_TOL
ITER_CAP="${ITER_CAP:-40}"; WALL_MAX="${WALL_MAX:-25}"; REL="${REL:-1e-2}"

[ -x "$BIN" ] || { echo "build first: (cd $SRC && make -f Makefile_MT SPOOLES_DIR=../SPOOLES.2.2 -j8 CalculiX_MT)"; exit 2; }
if [ ! -f "$DECK_IN" ]; then echo "ROUND-TRIP: SKIP (deck not found: $DECK_IN -- pass a deck + golden maxU)"; exit 0; fi
DECK="$(cd "$(dirname "$DECK_IN")" && pwd)/$(basename "$DECK_IN")"

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
cd "$WORK"; ln -sf "$DECK" rt.inp
load1=$(uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/')

t0=$(python3 -c 'import time;print(time.time())')
env CCX_ACCEL_SOLVE=gmg CCX_ACCEL_VERBOSE=1 CCX_ACCEL_OUT_DAT_FAST="$WORK/rt.dat" "$BIN" rt >o.log 2>e.log
rc=$?
t1=$(python3 -c 'import time;print(time.time())')
wall=$(python3 -c "print(f'{$t1-$t0:.1f}')")

iters=$(grep -oE 'PCG iters=[0-9]+' e.log | tail -1 | grep -oE '[0-9]+')
solve=$(grep -oE 'solve=[0-9.]+s' e.log | tail -1 | grep -oE '[0-9.]+')
# read the displacement field back out of the .dat (the round-trip) and take maxU
maxu=$(awk '/displacements/{f=1;next} f&&/stresses/{exit} f&&NF==4{m=sqrt($2*$2+$3*$3+$4*$4);if(m>x)x=m} END{printf "%.10f",x+0}' rt.dat 2>/dev/null)

echo "deck=$(basename "$DECK")  threads=$OMP_NUM_THREADS  GMG_TOL=$GMG_TOL  load1m=$load1"
echo "rc=$rc  wall=${wall}s  GMG solve=${solve:-?}s  PCG iters=${iters:-?}  maxU=${maxu:-none}  golden=$GOLDEN"

fail=0
# round-trip + correctness
if [ -z "$maxu" ] || [ "$maxu" = "0.0000000000" ]; then echo "  [FAIL] round-trip: no displacement read back from .dat"; fail=1
else
  ok=$(awk -v a="$maxu" -v g="$GOLDEN" -v r="$REL" 'BEGIN{print (g>0 && (a>g?a-g:g-a)/g<=r)?1:0}')
  [ "$ok" = 1 ] && echo "  [PASS] correctness: maxU within $REL of golden" || { echo "  [FAIL] correctness: maxU $maxu vs golden $GOLDEN"; fail=1; }
fi
# no-slowdown (load-independent)
if [ -n "$iters" ]; then
  [ "$iters" -le "$ITER_CAP" ] && echo "  [PASS] no-slowdown: PCG iters $iters <= $ITER_CAP" || { echo "  [FAIL] no-slowdown: PCG iters $iters > $ITER_CAP (convergence regressed)"; fail=1; }
else echo "  [FAIL] no GMG solve ran (no PCG iters line)"; fail=1; fi
# wall (load-aware)
wok=$(python3 -c "print(1 if $wall < $WALL_MAX else 0)")
if [ "$wok" = 1 ]; then echo "  [PASS] wall ${wall}s < ${WALL_MAX}s"
elif python3 -c "import sys;sys.exit(0 if $load1 > 6 else 1)"; then echo "  [INDETERMINATE] wall ${wall}s >= ${WALL_MAX}s but load1m=$load1 high -> re-run on a quiet box"
else echo "  [FAIL] wall ${wall}s >= ${WALL_MAX}s on a quiet box (regression)"; fail=1; fi

[ "$fail" = 0 ] && { echo "ROUND-TRIP: GREEN"; exit 0; } || { echo "ROUND-TRIP: RED"; exit 1; }
