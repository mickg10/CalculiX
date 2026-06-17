#!/usr/bin/env bash
# gmg_bench.sh — build the PRODUCTION gmg_solve.cpp BOTH ways and benchmark + compare. Tooling/experimental.
#   ACCEL     : Apple Accelerate LAPACK (default on macOS; -DACCELERATE_NEW_LAPACK + -framework Accelerate)
#   REFERENCE : portable -DCCX_GMG_NOACCEL -> standard LAPACK Fortran ABI linked against OpenBLAS (Apple-free)
# Both must produce the SAME maxU (correctness); the only difference is the tiny coarse dpotrf.
#
# Usage:  ./gmg_bench.sh <dir-with-m2.mat-and-coordmap.bin>
#   Dump those from any deck:  CCX_ACCEL_SOLVE=gmg CCX_ACCEL_DUMP=<dir>/m2.mat CCX_ACCEL_DUMP2=<dir>/coordmap.bin ./ccx -i job
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
W="${1:?usage: $0 <dir-with-m2.mat-and-coordmap.bin>}"
SRC="../src/gmg_solve.cpp"; DRV="gmg_driver.cpp"
LIBOMP="$(brew --prefix libomp 2>/dev/null || echo /usr)"
OPENBLAS="$(brew --prefix openblas 2>/dev/null || echo /usr)"
OMPF="-Xpreprocessor -fopenmp -I$LIBOMP/include -L$LIBOMP/lib -lomp"
COMMON="-std=c++17 -O3 -DNDEBUG"

echo "== build ACCEL (Apple Accelerate) =="
clang++ $COMMON -DACCELERATE_NEW_LAPACK $OMPF "$SRC" "$DRV" -framework Accelerate -o gmg_accel
echo "== build REFERENCE (portable, OpenBLAS, -DCCX_GMG_NOACCEL) =="
clang++ $COMMON -DCCX_GMG_NOACCEL $OMPF "$SRC" "$DRV" -L"$OPENBLAS/lib" -lopenblas -o gmg_ref

echo "== run ACCEL =="    ; A=$(OMP_NUM_THREADS=16 ./gmg_accel "$W/m2.mat" "$W/coordmap.bin" 1e-4 | grep SUMMARY); echo "$A"
echo "== run REFERENCE ==" ; R=$(OMP_NUM_THREADS=16 ./gmg_ref   "$W/m2.mat" "$W/coordmap.bin" 1e-4 | grep SUMMARY); echo "$R"

get(){ echo "$1"|tr ' ' '\n'|grep "^$2="|cut -d= -f2; }
ma=$(get "$A" maxU); mr=$(get "$R" maxU); sa=$(get "$A" solve_s); sr=$(get "$R" solve_s)
rca=$(get "$A" rc); rcr=$(get "$R" rc)
echo "------------------------------------------------------------"
printf "ACCEL     rc=%s maxU=%s solve=%ss\n" "$rca" "$ma" "$sa"
printf "REFERENCE rc=%s maxU=%s solve=%ss\n" "$rcr" "$mr" "$sr"
d=$(python3 -c "print(abs($ma-$mr)/($mr if $mr else 1))")
printf "maxU agreement |dU|/U = %s ; ref/accel solve ratio = %s\n" "$d" "$(python3 -c "print(f'{$sr/$sa:.3f}')")"
ok=1
[ "$rca" = 0 ] && [ "$rcr" = 0 ] || { echo "FAIL: a build did not converge (rc!=0)"; ok=0; }
python3 -c "import sys; sys.exit(0 if $d<1e-9 else 1)" || { echo "FAIL: maxU disagree"; ok=0; }
[ $ok = 1 ] && echo "PASS: production gmg_solve builds both ways, both converge, identical maxU" || { echo "TEST FAILED"; exit 1; }
