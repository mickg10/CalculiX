# Accel backend — developer tests

Tests for the optional Apple Accelerate / geometric-multigrid backend (`-DCCX_ACCEL`, runtime opt-in via
`CCX_ACCEL_SOLVE`). Build the binary first:

```
(cd ../src && make -f Makefile_MT SPOOLES_DIR=../SPOOLES.2.2 ARROW=1 -j8 CalculiX_MT)
```

## Tests

- **`roundtrip_test.sh [deck.inp] [golden_maxU]`** — end-to-end fast-iteration gate on a large deck: read
  `.inp` → GMG solve → fast `.dat` output → read the displacement field back → verify it. Gates: correctness
  (maxU within 1% of golden), no-slowdown (GMG PCG iters under a cap; load-independent), round-trip (the `.dat`
  parses), and wall time (PASS under 25 s; reported INDETERMINATE when the machine is loaded). Defaults to a
  bundled large reproducer if present, else SKIPs; pass a deck + golden maxU for any other case.

- **`accel_modes_test.sh [deck.inp] [rel]`** — runs one deck through `CCX_ACCEL_SOLVE` = `direct` / `float-pcg`
  / `defl-pcg` and checks every mode agrees with the exact `direct` Cholesky golden (default 1e-4). The deck
  must request displacement output (`*NODE PRINT,U`).

- **`parse_tests.sh`** — differential red/green suite for the fast C deck parser (`ccxftoi`/`ccxftof`/
  `ccxsplit`, `-DCCX_FAST_PARSE`): a corpus plus a large reproducible fuzz run, each compared bit-for-bit
  against the Fortran reads they replace (`parse_ref.f`). Builds `parse_test.c`.

- **`parse_integration_tests.sh`** — runs whole decks through CalculiX with the fast parser on vs off and
  diffs the results, so the parser is exercised end-to-end, not just in isolation.

- **`arrow_read.py`** — reads a CCX Arrow IPC field file (written when `CCX_ACCEL_OUT_ARROW=<path>`) with
  pyarrow and prints schema / row count / field ranges; a reader/validation example for the Arrow output.
