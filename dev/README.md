# GMG solver: portable reference vs Apple-accelerated (build-both tests)

The geometric-multigrid solver (`src/gmg_solve.cpp`) is **portable by default** (OpenMP + standard LAPACK
Fortran ABI) with Apple Accelerate layered on macOS behind a compile-time switch. Everything here builds and
compares **both** variants so the algorithm is never silently locked to one platform.

## Compile-time switch (`src/gmg_solve.cpp`)
- default on macOS → `CCX_GMG_ACCEL=1` (Apple Accelerate LAPACK; this is also where AMX lives)
- `-DCCX_GMG_NOACCEL` → portable reference: standard `dpotrf_/dpotrs_` linked against **OpenBLAS / MKL / Netlib**
- `-DCCX_GMG_ACCEL=1` or `=0` → force either path explicitly
- non-Apple platforms → automatically the portable reference

Only the tiny coarse dense solve (`dpotrf`) is platform-specific; the SpMV / Chebyshev smoother / PCG / Galerkin
RAP are plain OpenMP C++.

## Tests
Both need a dumped matrix + coord/DOF map (any CalculiX run with the accel backend:
`CCX_ACCEL_SOLVE=gmg CCX_ACCEL_DUMP=m.bin CCX_ACCEL_DUMP2=cm.bin ./ccx -i job` → builds the CSC matrix + map
and exits). Pass the directory holding `m2.mat` + `coordmap.bin` as the first arg.

- **`gmg_bench.sh <dir>`** — builds the *production* `gmg_solve.cpp` BOTH ways (Apple Accelerate vs OpenBLAS),
  runs the full solve in each, and checks they converge to the **same maxU** (correctness) while reporting each
  solve time. Result on row236: identical maxU to 6e-10, ~equal speed (the GMG is memory-bound, so Accelerate
  buys ~nothing — the reference is Apple-free with no penalty).
- **`mf_test.sh <dir>`** — builds the matrix-free C3D8 element operator (`mf_verify.cpp`) BOTH ways
  (portable OpenMP vs Apple-AMX `cblas_dgemm`), verifies each reproduces the assembled `A0` to roundoff, and
  reports the apply speed. Finding: the apply is memory-bound, so the portable reference is competitive/faster
  and AMX does not help.

Tooling / experimental.
