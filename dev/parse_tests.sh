#!/usr/bin/env bash
# parse_tests.sh — build + run the differential red/green tests for the fast C deck parsers.
# Reference = the Fortran reads (parse_ref.f) + stock splitline.f (compiled WITHOUT -DCCX_FAST_PARSE).
# Under test = ccx_fastnum.c (ccxftoi/ccxftof/ccxsplit). Tooling / test-only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SRC=../src
CC="${CC:-clang}"
FC="${FC:-gfortran-14}"
command -v "$FC" >/dev/null 2>&1 || FC=gfortran

echo "== build parser differential tests =="
$FC -O2 -c parse_ref.f -o parse_ref.o
$FC -O2 -cpp -c "$SRC/splitline.f" -o splitline_ref.o        # stock Fortran reference (no -DCCX_FAST_PARSE)
$CC -O2 -c "$SRC/ccx_fastnum.c" -o ccx_fastnum.o             # C parsers under test
$CC -O2 -c parse_test.c -o parse_test.o
$FC -O2 -o parse_test parse_test.o parse_ref.o splitline_ref.o ccx_fastnum.o

echo "== run =="
if ./parse_test; then
  echo "PARSE TESTS: GREEN"
else
  echo "PARSE TESTS: RED"; exit 1
fi
