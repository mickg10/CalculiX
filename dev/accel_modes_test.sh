#!/usr/bin/env bash
# accel_modes_test.sh - exercise the accel solve modes and check they agree on the displacement field.
# Runs one deck through CCX_ACCEL_SOLVE = direct | float-pcg | defl-pcg (and gmg if the deck is a regular
# voxel grid), parses maxU from each .dat, and compares to the `direct` result (the exact Accelerate Cholesky
# golden). GREEN iff every mode matches direct within REL (default 1e-4). Tooling / test-only.
#
# Usage: accel_modes_test.sh [deck.inp] [rel]
#   The deck must write displacement to the .dat (i.e. have *NODE PRINT,U). Default = a bundled near-singular source coupon
#   (near-singular -> the case defl-pcg's RBM deflation targets); pass any such deck explicitly otherwise.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SRC=../src
BIN="$(cd "$SRC" && pwd)/CalculiX_MT"                                  # absolute (we cd into temp dirs below)
DECK_IN="${1:-../../../reproducers/ccx_thread_divergence_source_coupon/input/source_coupon_static_y.inp}"
DECK="$(cd "$(dirname "$DECK_IN")" && pwd)/$(basename "$DECK_IN")"     # absolute
REL="${2:-1e-4}"
[ -x "$BIN" ] || { echo "build first: (cd $SRC && make -f Makefile_MT SPOOLES_DIR=../SPOOLES.2.2 -j8 CalculiX_MT)"; exit 2; }
[ -f "$DECK" ] || { echo "deck not found: $DECK"; exit 2; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
stem=$(basename "$DECK" .inp)

maxu() {  # maxU from a .dat displacement block
  awk '/displacements/{f=1;next} f&&/stresses/{exit} f&&NF==4{m=sqrt($2*$2+$3*$3+$4*$4);if(m>x)x=m} END{printf "%.10f",x+0}' "$1"
}

declare -A MU
for mode in direct float-pcg defl-pcg; do
  d="$WORK/$mode"; mkdir -p "$d"; cp "$DECK" "$d/$stem.inp"
  ( cd "$d" && env CCX_ACCEL_SOLVE="$mode" CCX_ACCEL_VERBOSE=1 "$BIN" "$stem" >o.log 2>e.log )
  MU[$mode]=$(maxu "$d/$stem.dat")
  it=$(grep -oE "iters=[0-9]+" "$d/e.log" | tail -1)
  printf "  %-9s maxU=%s  %s\n" "$mode" "${MU[$mode]:-<none>}" "$it"
done

gold=${MU[direct]}
[ -n "$gold" ] && [ "$gold" != "0.0000000000" ] || { echo "ACCEL MODES: INVALID (no direct golden)"; exit 2; }
fail=0
for mode in float-pcg defl-pcg; do
  v=${MU[$mode]}
  ok=$(awk -v a="$v" -v g="$gold" -v r="$REL" 'BEGIN{print (g>0 && (a>g?a-g:g-a)/g<=r)?"1":"0"}')
  [ "$ok" = 1 ] || { echo "  RED $mode: $v vs direct $gold"; fail=1; }
done
if [ "$fail" = 0 ]; then echo "ACCEL MODES: GREEN (direct/float-pcg/defl-pcg agree within $REL on $stem)"; else echo "ACCEL MODES: RED"; exit 1; fi
