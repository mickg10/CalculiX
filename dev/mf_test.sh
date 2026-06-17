#!/usr/bin/env bash
# mf_test.sh — build the matrix-free C3D8 element operator BOTH ways and compare. Tooling/experimental.
#   REFERENCE : -DCCX_GMG_NOACCEL  -> portable OpenMP colored apply, no Apple libs (works on Linux: -fopenmp + libgomp)
#   ACCEL     : default on macOS    -> Apple Accelerate AMX cblas_dgemm (batched per color)
# Both must reproduce the assembled A0 to roundoff and agree with each other; speeds are reported.
#
# Usage:  ./mf_test.sh <dir-with-m2.mat-and-coordmap.bin>     (dump via CCX_ACCEL_DUMP / CCX_ACCEL_DUMP2)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
W="${1:?usage: $0 <dir-with-m2.mat-and-coordmap.bin>}"
LIBOMP="$(brew --prefix libomp 2>/dev/null || echo /usr)"
OMPF="-Xpreprocessor -fopenmp -I$LIBOMP/include -L$LIBOMP/lib -lomp"
COMMON="-std=c++17 -O3 -DNDEBUG"

echo "== build REFERENCE (portable, -DCCX_GMG_NOACCEL, no Apple libs) =="
clang++ $COMMON -DCCX_GMG_NOACCEL $OMPF mf_verify.cpp -o mf_verify_ref
echo "== build ACCEL (macOS default, Apple AMX dgemm) =="
clang++ $COMMON -DACCELERATE_NEW_LAPACK $OMPF mf_verify.cpp -o mf_verify_accel -framework Accelerate

R=$(OMP_NUM_THREADS=16 ./mf_verify_ref   "$W/m2.mat" "$W/coordmap.bin" 8 0.49 0.1 | grep SUMMARY); echo "REF:   $R"
A=$(OMP_NUM_THREADS=16 ./mf_verify_accel "$W/m2.mat" "$W/coordmap.bin" 8 0.49 0.1 | grep SUMMARY); echo "ACCEL: $A"
get(){ echo "$1"|tr ' ' '\n'|grep "^$2="|cut -d= -f2; }
yr=$(get "$R" ynorm); ya=$(get "$A" ynorm); kr=$(get "$R" ke_ok); ka=$(get "$A" ke_ok)
ydiff=$(python3 -c "print(abs($yr-$ya)/$ya)")
echo "ref-vs-accel answer agreement |dy|/y = $ydiff"
[ "$kr" = 1 ] && [ "$ka" = 1 ] && python3 -c "import sys;sys.exit(0 if $ydiff<1e-10 else 1)" \
  && echo "PASS: both build, both exact, identical answer" || { echo "TEST FAILED"; exit 1; }
