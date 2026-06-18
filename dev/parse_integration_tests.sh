#!/usr/bin/env bash
# parse_integration_tests.sh — END-TO-END parser equivalence across the bundled CalculiX test decks.
# Builds two binaries that differ ONLY in the parse path (CCX_FAST_PARSE on vs off), runs each deck through
# both with NO accel env (default stock SPOOLES solver, so only the PARSE differs), and byte-compares outputs:
#   .dat (clean numbers, no timestamp) full compare; .frd header-stripped (drops version/date line) compare.
# identical => parse equivalent. asymmetric output => RED. neither produced output (both fail same) => skip.
# Tooling / test-only.  Usage: parse_integration_tests.sh [N_decks=0(all)] [per_deck_timeout_s=25]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SRC=../src
N="${1:-0}"; TO="${2:-25}"
SRCABS="$(cd "$SRC" && pwd)"
FAST="$SRCABS/CalculiX_MT_fastparse"; STOCK="$SRCABS/CalculiX_MT_noparse"
# Deterministic solve so any fast-vs-stock output difference is the PARSE, not MT-SPOOLES nondeterminism
# (the goal documents row164/row217 multithreaded irreproducibility). Single thread = reproducible.
export OMP_NUM_THREADS=1

echo "== build fast-parse + stock-parse binaries (differ only in CCX_FAST_PARSE) =="
( cd "$SRC" && make -f Makefile_MT SPOOLES_DIR=../SPOOLES.2.2 -j8 CalculiX_MT >/dev/null 2>&1 \
  && git checkout -- CalculiX.c frd.c CalculiXstep.c 2>/dev/null && cp -f CalculiX_MT CalculiX_MT_fastparse )
( cd "$SRC" && make -f Makefile_MT SPOOLES_DIR=../SPOOLES.2.2 FAST_PARSE_DEF= -j8 CalculiX_MT >/dev/null 2>&1 \
  && git checkout -- CalculiX.c frd.c CalculiXstep.c 2>/dev/null && cp -f CalculiX_MT CalculiX_MT_noparse )
[ -x "$FAST" ] && [ -x "$STOCK" ] || { echo "build failed"; exit 1; }

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
run_to(){ local to=$1; shift; "$@" >/dev/null 2>&1 & local pid=$!; ( sleep "$to" && kill -9 "$pid" 2>/dev/null ) & local k=$!; wait "$pid" 2>/dev/null; kill -9 "$k" 2>/dev/null; wait "$k" 2>/dev/null; return 0; }

pass=0; fail=0; skip=0; i=0; failed_list=""
for inp in ../test/*.inp; do
  [ "$N" != 0 ] && [ "$i" -ge "$N" ] && break
  i=$((i+1)); stem=$(basename "$inp" .inp)
  df="$WORK/f"; ds="$WORK/s"; rm -rf "$df" "$ds"; mkdir -p "$df" "$ds"
  cp "$inp" "$df/$stem.inp"; ( cd "$df" && run_to "$TO" "$FAST" "$stem" )
  cp "$inp" "$ds/$stem.inp"; ( cd "$ds" && run_to "$TO" "$STOCK" "$stem" )
  bad=0; any=0
  for e in dat frd; do
    if [ -s "$df/$stem.$e" ] || [ -s "$ds/$stem.$e" ]; then any=1; fi
    if [ -s "$df/$stem.$e" ] && [ -s "$ds/$stem.$e" ]; then
      if [ "$e" = frd ]; then  # drop the run-time DATE/TIME/HOST header lines before comparing
        diff -q <(grep -av -E '1U(DATE|TIME|HOST)' "$df/$stem.$e") <(grep -av -E '1U(DATE|TIME|HOST)' "$ds/$stem.$e") >/dev/null 2>&1 || bad=1
      else cmp -s "$df/$stem.$e" "$ds/$stem.$e" || bad=1; fi
    elif { [ -s "$df/$stem.$e" ] && [ ! -s "$ds/$stem.$e" ]; } || { [ ! -s "$df/$stem.$e" ] && [ -s "$ds/$stem.$e" ]; }; then
      bad=1   # asymmetric output -> parse divergence
    fi
  done
  if [ "$any" = 0 ]; then skip=$((skip+1));
  elif [ "$bad" = 0 ]; then pass=$((pass+1));
  else fail=$((fail+1)); failed_list="$failed_list $stem"; echo "  RED $stem"; fi
  [ $((i % 50)) -eq 0 ] && echo "  ... $i decks: $pass pass / $fail fail / $skip skip"
done
echo "=== parse integration: $i decks -> $pass pass, $fail fail, $skip skip (no output) ==="
[ -n "$failed_list" ] && echo "FAILED:$failed_list"
if [ "$pass" = 0 ]; then echo "PARSE INTEGRATION: INVALID (no deck produced comparable output -- check binaries)"; exit 2; fi
if [ "$skip" -gt $(( i / 2 )) ]; then echo "PARSE INTEGRATION: INVALID (>50% of decks skipped -- under-coverage, check build/env)"; exit 2; fi
if [ "$fail" = 0 ]; then echo "PARSE INTEGRATION: GREEN ($pass decks compared identical)"; else echo "PARSE INTEGRATION: RED"; exit 1; fi
